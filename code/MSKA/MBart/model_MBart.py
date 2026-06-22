import torch
from Recognition.recognition import Recognition
from MBart.translation import TranslationNetwork
from MBart.vl_mapper import VLMapper


class SignLanguageModel(torch.nn.Module):
    """
    Modelo completo de reconocimiento/traducción de lenguaje de señas.
    Soporta dos tareas:
      - S2G (Sign-to-Gloss): solo reconocimiento, produce glosas a partir de keypoints.
      - S2T (Sign-to-Text):  reconocimiento + traducción, produce texto natural.

    Arquitectura S2T:
        keypoints → Recognition → VLMapper → TranslationNetwork → texto
    Arquitectura S2G:
        keypoints → Recognition → glosas
    """

    def __init__(self, cfg, args):
        super().__init__()
        self.args   = args
        self.task   = cfg['task']    # 'S2G' o 'S2T'
        self.device = cfg['device']
        model_cfg   = cfg['model']

        # --- Módulo de reconocimiento (compartido por S2G y S2T) ---
        # Recibe keypoints corporales y produce logits de glosas + features visuales
        self.recognition_network = Recognition(
            cfg=model_cfg['RecognitionNetwork'],
            args=self.args,
        )
        self.gloss_tokenizer = self.recognition_network.gloss_tokenizer

        if self.task == 'S2G':
            self.text_tokenizer = None  # no se necesita en reconocimiento puro

        elif self.task == 'S2T':
            # Pesos para ponderar cada loss en la loss total:
            #   total_loss = recognition_weight * recognition_loss
            #              + translation_weight * translation_loss
            self.recognition_weight = model_cfg.get('recognition_weight', 1)
            self.translation_weight = model_cfg.get('translation_weight', 1)

            # --- Módulo de traducción ---
            # Transformer que convierte features visuales (mapeadas) en texto
            self.translation_network = TranslationNetwork(
                cfg=model_cfg['TranslationNetwork'],
            )
            self.text_tokenizer = self.translation_network.text_tokenizer

            # --- VLMapper: puente entre reconocimiento y traducción ---
            # Adapta las features del recognition al espacio vectorial que
            # espera el translation network (dimensión y distribución distintas).
            #
            # Hay dos tipos de mapper según el config:
            #
            #   'projection' (por defecto): opera sobre features visuales continuas.
            #       in_features = hidden_size del encoder visual, salvo que se
            #       especifique explícitamente en el config (en cuyo caso se elimina
            #       del dict con .pop() para no pasarlo dos veces al constructor).
            #
            #   otro tipo (p.ej. embedding de glosas): opera sobre el vocabulario
            #       discreto de glosas, por lo que in_features = nº de glosas.
            #
            mapper_type = model_cfg['VLMapper'].get('type', 'projection')
            if mapper_type == 'projection':
                if 'in_features' in model_cfg['VLMapper']:
                    in_features = model_cfg['VLMapper'].pop('in_features')
                else:
                    in_features = model_cfg['RecognitionNetwork']['visual_head']['hidden_size']
            else:
                in_features = len(self.gloss_tokenizer)

            self.vl_mapper = VLMapper(
                cfg=model_cfg['VLMapper'],
                in_features=in_features,
                out_features=self.translation_network.input_dim,
                gloss_id2str=self.gloss_tokenizer.id2gloss,
                gls2embed=getattr(self.translation_network, 'gls2embed', None),
            )

    def forward(self, src_input, **kwargs):
        """
        Paso forward del modelo.

        Para S2G devuelve solo la loss de reconocimiento.
        Para S2T ejecuta el pipeline completo y combina ambas losses.

        Devuelve un diccionario con losses, logits y features intermedias.
        """

        if self.task == 'S2G':
            recognition_outputs = self.recognition_network(src_input)
            model_outputs = {**recognition_outputs}
            model_outputs['total_loss'] = recognition_outputs['recognition_loss']

        else:  # S2T
            # 1. Extraer features visuales y logits de glosas
            recognition_outputs = self.recognition_network(src_input)

            # 2. Adaptar las features al espacio del translation network
            mapped_feature = self.vl_mapper(visual_outputs=recognition_outputs)

            # 3. Construir las entradas del transformer combinando:
            #    - tokens de texto y glosa (vienen del batch, preparados en collate_fn)
            #    - features visuales mapeadas y sus longitudes
            translation_inputs = {
                **src_input['translation_inputs'],
                'input_feature': mapped_feature,
                'input_lengths': recognition_outputs['input_lengths'],
            }

            # 4. Traducción
            translation_outputs = self.translation_network(**translation_inputs)

            # 5. Combinar outputs de ambos módulos en un único diccionario
            model_outputs = {**translation_outputs, **recognition_outputs}

            # Guardar transformer_inputs para poder usarlos en generate_txt durante evaluación
            model_outputs['transformer_inputs'] = model_outputs['transformer_inputs']

            # Loss total: suma ponderada de recognition loss y translation loss
            model_outputs['total_loss'] = (
                model_outputs['recognition_loss'] +
                model_outputs['translation_loss']
            )

        return model_outputs

    def generate_txt(self, transformer_inputs=None, generate_cfg={}, **kwargs):
        """
        Genera texto en modo evaluación usando beam search.
        Solo se llama desde evaluate() en train.py, nunca durante el entrenamiento.

        transformer_inputs: features y máscaras guardadas en el forward.
        generate_cfg:       parámetros del beam search (tamaño del haz, longitud máxima, etc.)
        """
        return self.translation_network.generate(**transformer_inputs, **generate_cfg)

    def predict_gloss_from_logits(self, gloss_logits, beam_size, input_lengths):
        """
        Decodifica logits de glosas a secuencias de texto mediante CTC beam search.
        Wrapper sobre recognition_network.decode para mantener la interfaz limpia.
        """
        return self.recognition_network.decode(
            gloss_logits=gloss_logits,
            beam_size=beam_size,
            input_lengths=input_lengths,
        )
