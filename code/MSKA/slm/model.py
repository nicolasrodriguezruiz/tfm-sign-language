import torch
from recognition import Recognition
from translation import TranslationNetwork
from vl_mapper import VLMapper


class SignLanguageModel(torch.nn.Module):
    """
    Modelo completo de reconocimiento/traducción de lenguaje de señas.

    Soporta dos tareas:
      - S2G: solo reconocimiento → glosas
      - S2T: reconocimiento + traducción → texto alemán

    Arquitectura S2T con Qwen:
        keypoints → Recognition → VLMapper → ┐
                                             ├→ Qwen2.5-1.5B (LoRA) → texto
        glosas (opcionales) ─────────────────┘

    Flags de ablation (se configuran en TranslationNetwork via YAML):
        use_visual_features: incluir features visuales (default: True)
        use_gloss_tokens:    incluir glosas como contexto (default: False)
        gloss_source:        'ground_truth' o 'predicted' (default: 'ground_truth')
    """

    def __init__(self, cfg, args):
        super().__init__()
        self.args   = args
        self.task   = cfg['task']
        self.device = cfg['device']
        model_cfg   = cfg['model']

        # --- Módulo de reconocimiento (compartido por S2G y S2T) ---
        self.recognition_network = Recognition(
            cfg=model_cfg['RecognitionNetwork'],
            args=self.args,
        )
        self.gloss_tokenizer = self.recognition_network.gloss_tokenizer

        if self.task == 'S2G':
            self.text_tokenizer = None

        elif self.task == 'S2T':
            # El recognition ya tiene buenos pesos preentrenados.
            # Se congela para ahorrar VRAM y evitar que el entrenamiento
            # de la traducción distorsione las features visuales aprendidas.
            for param in self.recognition_network.parameters():
                param.requires_grad = False
            self.recognition_weight = model_cfg.get('recognition_weight', 1)
            self.translation_weight = model_cfg.get('translation_weight', 1)

            # --- Módulo de traducción (Qwen2.5-1.5B + LoRA) ---
            self.translation_network = TranslationNetwork(
                cfg=model_cfg['TranslationNetwork'],
                task=self.task,
            )
            self.text_tokenizer = self.translation_network.tokenizer

            # --- VLMapper: proyecta features visuales al espacio de Qwen ---
            mapper_type = model_cfg['VLMapper'].get('type', 'projection')
            if mapper_type == 'projection':
                if 'in_features' in model_cfg['VLMapper']:
                    in_features = model_cfg['VLMapper'].pop('in_features')
                else:
                    in_features = model_cfg['RecognitionNetwork']['fuse_visual_head']['hidden_size']
            else:
                in_features = len(self.gloss_tokenizer)

            self.vl_mapper = VLMapper(
                cfg=model_cfg['VLMapper'],
                in_features=in_features,
                out_features=self.translation_network.input_dim,
                gloss_id2str=self.gloss_tokenizer.id2gloss,
                gls2embed=getattr(self.translation_network, 'gls2embed', None),
            )

            # Cargar pesos preentrenados del VLMapper si existen (Fase 1 → Fase 2)
            pretrained_mapper = model_cfg['VLMapper'].get('pretrained_mapper', None)
            if pretrained_mapper:
                self.vl_mapper.load_state_dict(torch.load(pretrained_mapper, map_location='cpu'))
                print(f"VLMapper cargado desde preentrenamiento: {pretrained_mapper}")

    def forward(self, src_input, **kwargs):
        """
        Forward pass del modelo.

        En S2T:
          1. Recognition extrae features y predice glosas.
          2. VLMapper proyecta las features al espacio de Qwen.
          3. TranslationNetwork construye el prefijo [visual, glosas] y genera texto.

        Las glosas que se pasan al translator dependen de gloss_source:
          - 'ground_truth': glosas reales del batch (solo disponibles en entrenamiento)
          - 'predicted':    top-1 argmax sobre los logits de fuse_head (siempre disponible)
        """
        if self.task == 'S2G':
            recognition_outputs = self.recognition_network(src_input)
            model_outputs = {**recognition_outputs}
            model_outputs['total_loss'] = recognition_outputs['recognition_loss']

        else:  # S2T
            # 1. Reconocimiento
            recognition_outputs = self.recognition_network(src_input)

            # 2. Proyección visual
            mapped_feature = self.vl_mapper(visual_outputs=recognition_outputs)

            # 3. Seleccionar qué glosas pasar al translator
            use_gloss = self.translation_network.use_gloss_tokens
            gloss_source = self.translation_network.gloss_source

            if use_gloss:
                if gloss_source == 'ground_truth' and self.training:
                    # Entrenamiento: usar glosas reales del batch
                    gloss_ids     = src_input['gloss_input']['gloss_labels']
                    gloss_lengths = src_input['gloss_input']['gls_lengths']
                else:
                    # Evaluación o gloss_source='predicted': usar top-1 de fuse_head
                    # argmax sobre el vocabulario de glosas para cada frame
                    gloss_ids = recognition_outputs['fuse_gloss_logits'].argmax(-1)  # (B, T)
                    # La longitud válida es new_src_lengths (frames reales tras downsampling)
                    gloss_lengths = src_input['new_src_lengths']
            else:
                gloss_ids     = None
                gloss_lengths = None

            # 4. Traducción
            translation_inputs = {
                **src_input['translation_inputs'],  # labels, decoder_input_ids
                'input_feature':  mapped_feature,
                'input_lengths':  recognition_outputs['input_lengths'],
                'gloss_ids':      gloss_ids,
                'gloss_lengths':  gloss_lengths,
            }
            translation_outputs = self.translation_network(**translation_inputs)

            # 5. Combinar outputs
            model_outputs = {**translation_outputs, **recognition_outputs}

            # Loss total ponderada
            model_outputs['total_loss'] = (
                self.recognition_weight * model_outputs['recognition_loss'] +
                self.translation_weight * model_outputs['translation_loss']
            )

        return model_outputs

    def generate_txt(self, transformer_inputs=None, generate_cfg={}, **kwargs):
        """
        Genera texto en evaluación usando beam search de Qwen.
        Reutiliza el prefijo visual/glosa guardado en el forward.
        """
        return self.translation_network.generate(
            **transformer_inputs,
            **generate_cfg,
        )

    def predict_gloss_from_logits(self, gloss_logits, beam_size, input_lengths):
        """Decodifica logits a glosas mediante CTC beam search."""
        return self.recognition_network.decode(
            gloss_logits=gloss_logits,
            beam_size=beam_size,
            input_lengths=input_lengths,
        )
