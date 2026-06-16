import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from Recognition.Tokenizer import GlossTokenizer_G2T
import math

import csv
import os

class TranslationNetwork(torch.nn.Module):
    """
    Módulo de traducción basado en Qwen2.5-1.5B con LoRA.

    A diferencia de MBart (encoder-decoder), Qwen es decoder-only.
    Las features visuales y las glosas se inyectan como tokens de prefijo
    antes del texto a generar, de forma similar a LLaVA.

    Secuencia que ve el modelo:
        [vis_1...vis_T, gls_1...gls_G, texto_1...texto_N]
         ↑ features      ↑ glosas (opcional)  ↑ traducción alemán

    La loss solo se calcula sobre los tokens de texto (no sobre el prefijo visual/glosa).

    Flags de ablation (controlables desde el config YAML):
        use_visual_features: incluir features visuales del VLMapper (default: True)
        use_gloss_tokens:    incluir embeddings de glosas como contexto (default: False)
        gloss_source:        'ground_truth' usa glosas reales en entrenamiento,
                             'predicted' usa siempre las predichas por CTC (default: 'ground_truth')

    LoRA: solo se entrenan los adaptadores de bajo rango (~1-3% de parámetros).
    El resto de Qwen permanece congelado, reduciendo drásticamente la VRAM y el tiempo necesario.
    """

    def __init__(self, cfg=None, task='S2T') -> None:
        super().__init__()
        self.task = task

        # --- Flags de ablation ---
        self.use_visual_features = cfg.get('use_visual_features', True)
        self.use_gloss_tokens    = cfg.get('use_gloss_tokens', False)
        # 'ground_truth': glosas reales en entrenamiento, predichas en evaluación
        # 'predicted':    siempre usa las predichas por CTC
        self.gloss_source = cfg.get('gloss_source', 'ground_truth')

        if not self.use_visual_features and not self.use_gloss_tokens:
            raise ValueError("Al menos use_visual_features o use_gloss_tokens debe estar activo.")

        # --- Tokenizador de Qwen ---
        model_name = cfg.get('pretrained_model_name_or_path', 'Qwen/Qwen2.5-1.5B')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Qwen no tiene pad_token por defecto, se usa eos_token como pad
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_index = self.tokenizer.pad_token_id
        self.eos_index = self.tokenizer.eos_token_id

        # --- Cargar Qwen en bfloat16 para ahorrar VRAM ---
        # bfloat16 es el formato recomendado para LLMs: misma precisión dinámica que float32
        # pero ocupa la mitad de memoria.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        if "lora" in cfg:
            print("Utilizando LoRA")
            # --- Aplicar LoRA ---
            # LoRA añade matrices de bajo rango (r) a las capas de atención.
            # Solo estas matrices pequeñas se entrenan; el resto de Qwen queda congelado.
            # r=16: rango de las matrices LoRA. Mayor r = más capacidad pero más parámetros.
            # lora_alpha=32: factor de escala. Convención habitual: alpha = 2*r.
            # lora_dropout=0.1: regularización dentro de los adaptadores.
            # target_modules: capas donde se insertan los adaptadores (proyecciones de atención).
            lora_cfg = cfg.get('lora', {})
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_cfg.get('r', 16),
                lora_alpha=lora_cfg.get('alpha', 32),
                lora_dropout=lora_cfg.get('dropout', 0.1),
                target_modules=lora_cfg.get('target_modules', ['q_proj', 'v_proj', 'k_proj', 'o_proj']),
                bias='none',
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()  # para ver cuántos parámetros se entrenan
        else:
            # Fine-tuning: todos los parámetros son entrenables
            for param in self.model.parameters():
                param.requires_grad = True
            n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
            print(f"Fine-tuning: {n_params:.1f}M parámetros entrenables en Qwen")

        # Dimensión oculta de Qwen: es la dimensión que espera el proyector visual
        self.input_dim = self.model.config.hidden_size

        # --- Tokenizador de glosas (opcional) ---
        if self.use_gloss_tokens:
            self.gloss_tokenizer = GlossTokenizer_G2T(tokenizer_cfg=cfg['GlossTokenizer'])
            # Tabla de embeddings de glosas proyectada a hidden_size de Qwen.
            # Se inicializa con embeddings preentrenados si están disponibles.
            self.gloss_embedding = self._build_gloss_embedding(
                gloss2embed_file=cfg['GlossEmbedding']['gloss2embed_file'],
                from_scratch=cfg['GlossEmbedding'].get('from_scratch', False),
            )

        # Cargar checkpoint propio si se especifica
        if 'load_ckpt' in cfg:
            self.load_from_pretrained_ckpt(cfg['load_ckpt'])


        # # --- INICIO DEL DIAGNÓSTICO TEMPORAL ---
        # # Definir la ruta del archivo CSV (puedes ajustarla si quieres)
        # self.diagnostic_file = 'temporal_diagnostics.csv'
        #
        # # Escribir los encabezados si el archivo no existe o queremos empezar de cero
        # # Usamos 'w' la primera vez para crear/limpiar el archivo
        # with open(self.diagnostic_file, mode='w', newline='', encoding='utf-8') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(['Batch_Size', 'Visual_Frames_Max', 'Visual_Frames_Avg', 'Text_Tokens_Max', 'Text_Tokens_Avg'])
        # # --- FIN DEL DIAGNÓSTICO TEMPORAL ---


    def load_from_pretrained_ckpt(self, pretrained_ckpt):
        """Carga pesos del TranslationNetwork desde un checkpoint del proyecto."""
        checkpoint = torch.load(pretrained_ckpt, map_location='cpu')['model_state']
        load_dict = {
            k.replace('translation_network.', ''): v
            for k, v in checkpoint.items()
            if 'translation_network' in k
        }
        self.load_state_dict(load_dict, strict=False)

    def _build_gloss_embedding(self, gloss2embed_file, from_scratch=False):
        """
        Construye la tabla de embeddings de glosas (num_glosas, hidden_size de Qwen).
        Si from_scratch=False, inicializa con embeddings preentrenados.
        """
        gloss_embedding = nn.Embedding(
            num_embeddings=len(self.gloss_tokenizer.id2gloss),
            embedding_dim=self.input_dim,
            padding_idx=self.gloss_tokenizer.gloss2id['<pad>'],
        )
        if not from_scratch:
            gls2embed = torch.load(gloss2embed_file)
            self.gls2embed = gls2embed
            with torch.no_grad():
                for id_, gls in self.gloss_tokenizer.id2gloss.items():
                    if gls in gls2embed:
                        # Los embeddings preentrenados pueden tener distinta dimensión,
                        # se proyectan linealmente si es necesario
                        emb = gls2embed[gls]
                        if emb.shape[0] != self.input_dim:
                            emb = F.interpolate(
                                emb.unsqueeze(0).unsqueeze(0),
                                size=self.input_dim,
                                mode='linear', align_corners=False,
                            ).squeeze()
                        gloss_embedding.weight[id_, :] = emb
        return gloss_embedding

    def _build_prefix_embeds(self, input_feature, input_lengths,
                              gloss_ids=None, gloss_lengths=None):
        """
        Construye el prefijo de embeddings que precede al texto en el decoder.

        El prefijo puede contener (según los flags activos):
            [features_visuales_1...T, glosa_1...G]

        Todo se proyecta al espacio de embeddings de Qwen (hidden_size).
        Se construye la attention_mask correspondiente.

        Args:
            input_feature:  features del VLMapper, shape (B, T, D) o None.
            input_lengths:  longitudes reales de las features, shape (B,).
            gloss_ids:      IDs de glosas, shape (B, G) o None.
            gloss_lengths:  longitudes reales de las glosas, shape (B,).

        Returns:
            prefix_embeds:  (B, prefix_len, hidden_size)
            prefix_mask:    (B, prefix_len) — 1 donde hay tokens reales
        """
        B = input_feature.shape[0] if input_feature is not None else gloss_ids.shape[0]
        device = input_feature.device if input_feature is not None else gloss_ids.device

        parts_per_sample = [[] for _ in range(B)]
        lens_per_sample  = [0] * B

        # --- Features visuales ---
        if self.use_visual_features and input_feature is not None:
            for i in range(B):
                vlen = input_lengths[i]
                parts_per_sample[i].append(input_feature[i, :vlen, :])  # (T, D)
                lens_per_sample[i] += vlen

        # --- Embeddings de glosas ---
        if self.use_gloss_tokens and gloss_ids is not None:
            for i in range(B):
                glen = gloss_lengths[i]
                gls_emb = self.gloss_embedding(gloss_ids[i, :glen].to(device))  # (G, D)
                parts_per_sample[i].append(gls_emb)
                lens_per_sample[i] += glen

        # --- Padding hasta la longitud máxima del batch ---
        max_prefix_len = max(lens_per_sample)
        prefix_embeds = torch.zeros(B, max_prefix_len, self.input_dim,
                                    dtype=torch.bfloat16, device=device)
        prefix_mask   = torch.zeros(B, max_prefix_len, dtype=torch.long, device=device)

        for i in range(B):
            if parts_per_sample[i]:
                cat = torch.cat(parts_per_sample[i], dim=0)  # (prefix_len_i, D)
                # Convertir a bfloat16 para que coincida con Qwen
                prefix_embeds[i, :cat.shape[0], :] = cat.to(torch.bfloat16)
                prefix_mask[i, :lens_per_sample[i]] = 1

        return prefix_embeds, prefix_mask

    def forward(self, input_feature, input_lengths, labels,
                decoder_input_ids, gloss_ids=None, gloss_lengths=None, **kwargs):
        """
        Forward pass en entrenamiento.

        Construye la secuencia completa:
            [prefijo_visual/glosa | tokens_texto]
        y calcula la cross-entropy loss solo sobre los tokens de texto.

        Args:
            input_feature:      features del VLMapper, shape (B, T, D).
            input_lengths:      longitudes de las features, shape (B,).
            labels:             IDs del texto de referencia, shape (B, L).
                                Los tokens de padding ya tienen valor -100.
            decoder_input_ids:  texto desplazado a la derecha para teacher forcing, shape (B, L).
            gloss_ids:          IDs de glosas (ground truth o predichas), shape (B, G).
            gloss_lengths:      longitudes de las glosas, shape (B,).
        """
        B = input_feature.shape[0]
        device = input_feature.device

#         # --- DIAGNÓSTICO TEMPORAL (Escribiendo en CSV) ---
#         # Calculamos los datos reales del lote actual
#         if input_feature is not None:
#             max_visual_frames = input_lengths.max().item()
#             avg_visual_frames = input_lengths.float().mean().item()

#             # Para el texto, contamos los tokens que NO son padding (-100)
#             valid_text_tokens = (labels != -100).sum(dim=1)
#             max_text_tokens = valid_text_tokens.max().item()
#             avg_text_tokens = valid_text_tokens.float().mean().item()

#             # Abrimos el archivo en modo 'a' (append) para añadir la nueva fila
#             with open(self.diagnostic_file, mode='a', newline='', encoding='utf-8') as file:
#                 writer = csv.writer(file)
#                 writer.writerow([
#                     B,
#                     max_visual_frames,
#                     round(avg_visual_frames, 2),
#                     max_text_tokens,
#                     round(avg_text_tokens, 2)
#                 ])
#         # --- FIN DEL DIAGNÓSTICO TEMPORAL ---


        # --- 1. Construir el prefijo visual/glosa ---
        # En entrenamiento con gloss_source='ground_truth' se usan las glosas reales.
        # En evaluación (o si gloss_source='predicted') se usan las predichas por CTC.
        effective_gloss_ids     = gloss_ids     if self.use_gloss_tokens else None
        effective_gloss_lengths = gloss_lengths if self.use_gloss_tokens else None

        prefix_embeds, prefix_mask = self._build_prefix_embeds(
            input_feature, input_lengths,
            effective_gloss_ids, effective_gloss_lengths,
        )
        prefix_len = prefix_embeds.shape[1]

        # --- 2. Embeddings del texto de referencia (para teacher forcing) ---
        # get_input_embeddings() devuelve la tabla de embeddings de Qwen
        # Preparado para LoRA y para FineTunning

        if hasattr(self.model, "get_base_model"):
            base_model = self.model.get_base_model()
        else:
            base_model = self.model

        text_embeds = base_model.model.embed_tokens(
            decoder_input_ids.to(device)
        ).to(torch.bfloat16)  # (B, L, hidden_size)

        # --- 3. Concatenar prefijo + texto ---
        inputs_embeds = torch.cat([prefix_embeds, text_embeds], dim=1)  # (B, prefix+L, D)

        # --- 4. Construir attention_mask completa ---
        text_mask = (decoder_input_ids != self.pad_index).long().to(device)  # (B, L)
        attention_mask = torch.cat([prefix_mask, text_mask], dim=1)           # (B, prefix+L)

        # --- 5. Construir labels: -100 en el prefijo (no supervisar), texto real después ---
        # -100 es el ignore_index estándar de PyTorch: la loss no se calcula en esas posiciones
        prefix_labels = torch.full((B, prefix_len), -100, dtype=torch.long, device=device)
        full_labels   = torch.cat([prefix_labels, labels.to(device)], dim=1)  # (B, prefix+L)

        # --- 6. Forward de Qwen ---
        output = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=full_labels,
            return_dict=True,
        )

        # HuggingFace calcula la cross-entropy internamente cuando se pasan labels.
        # output.loss ya es la media sobre los tokens no enmascarados.
        output_dict = {
            'translation_loss':  output.loss,
            'logits':            output.logits,
            # Guardar para reutilizar en generate() sin recalcular
            'transformer_inputs': {
                'prefix_embeds': prefix_embeds,
                'prefix_mask':   prefix_mask,
            },
        }
        return output_dict

    def generate(self, prefix_embeds, prefix_mask, **kwargs):
        """
        Generación de texto en evaluación mediante beam search.

        Recibe el prefijo visual/glosa ya construido (guardado en forward)
        y genera los tokens de texto de forma autoregresiva.

        Args:
            prefix_embeds:  (B, prefix_len, hidden_size) — prefijo visual/glosa.
            prefix_mask:    (B, prefix_len) — máscara del prefijo.
            num_beams:      tamaño del beam search.
            max_new_tokens: máximo de tokens nuevos a generar (no cuenta el prefijo).
            length_penalty: >1 favorece frases largas, <1 cortas.
        """

        # Para evitar tener que refractorizar
        gen_kwargs = {
            "max_new_tokens": 100,
            "num_beams": 4,
            "length_penalty": 1.0
        }

        #  Sobrescribir con todo lo que llegue desde YAML
        gen_kwargs.update(kwargs)

#         #  La llamada a Qwen (HuggingFace)
#         output = self.model.generate(
#             inputs_embeds=prefix_embeds.to('cuda'),
#             attention_mask=prefix_mask.to('cuda'),
#             eos_token_id=self.eos_index,
#             pad_token_id=self.pad_index,
#             return_dict_in_generate=True,
#             **gen_kwargs
#         )
#         generated_ids = output.sequences


        prompt_str = (
            "<|im_start|>user\n"
            "Übersetze diese Gebärdensprache exakt ins Deutsche.<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        # Tokenizamos el prompt (sin añadir especiales para mantener control total)
        prompt_ids = self.tokenizer(
            prompt_str,
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids.to(prefix_embeds.device)

        # Convertimos los IDs del prompt a vectores usando el cerebro de Qwen
        # Dependiendo de tu wrapper de HuggingFace, suele ser get_input_embeddings()
        prompt_embeds = self.model.get_input_embeddings()(prompt_ids)  # Shape: (1, L_prompt, D)

        # Expandimos el prompt para que coincida con el número de videos en tu batch
        B = prefix_embeds.shape[0]
        prompt_embeds = prompt_embeds.expand(B, -1, -1)  # Shape: (B, L_prompt, D)

        # Creamos una máscara de 1s para el prompt (es información útil)
        prompt_mask = torch.ones(
            B, prompt_embeds.shape[1],
            dtype=prefix_mask.dtype,
            device=prefix_mask.device
        )

        # UNIMOS EL VIDEO Y EL PROMPT
        full_prefix_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
        full_prefix_mask = torch.cat([prefix_mask, prompt_mask], dim=1)
        # ---------------------------------------------------------

        # La llamada a Qwen (HuggingFace) usando el prefix COMPLETO
        output = self.model.generate(
            inputs_embeds=full_prefix_embeds.to('cuda'),
            attention_mask=full_prefix_mask.to('cuda'),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            **gen_kwargs
        )

        # (Opcional) El log de depuración del EOS.
        # Recuerda que HuggingFace SIEMPRE inyecta un falso EOS en la posición 0
        # al usar inputs_embeds. Por eso ignoramos seq[0] en la búsqueda:
        eos_id = self.tokenizer.eos_token_id
        for i, seq in enumerate(output.sequences):
            # Buscamos el EOS a partir de la posición 1
            if len(seq) > 1 and (seq[1:] == eos_id).any().item():
                pass # Aquí sí ha generado el punto final correctamente

        decoded = self.tokenizer.batch_decode(
            output.sequences, skip_special_tokens=True
        )

# # #         # ===== Estadísticas útiles =====

#         eos_id = self.tokenizer.eos_token_id
#         pad_id = self.tokenizer.pad_token_id
#         generated_ids = output.sequences
#         eos_count = 0
#         lengths = []

#         for seq in generated_ids:
#             seq = seq.cpu()

#             # Longitud sin padding
#             length = (seq != pad_id).sum().item()
#             lengths.append(length)

#             # ¿Contiene EOS?
#             if (seq == eos_id).any().item():
#                 eos_count += 1

#         print(f"EOS rate: {eos_count}/{len(generated_ids)} ({100*eos_count/len(generated_ids):.2f}%)")
#         print(f"Avg generated length: {sum(lengths)/len(lengths):.2f}")
#         print(f"Min length: {min(lengths)}")
#         print(f"Max length: {max(lengths)}")

#         # Opcional: mostrar ejemplos
#         for i in range(min(1, len(generated_ids))):
#             print(f"\n--- SAMPLE {i} ---")
#             print("Length:", lengths[i])
#             print("Has EOS:", (generated_ids[i] == eos_id).any().item())
#         print("Sequences shape:", generated_ids.shape)
#         print("Prefix shape:", full_prefix_embeds.shape)

#         for i in range(min(1, len(generated_ids))):
#             seq = generated_ids[i]

#             eos_positions = (seq == eos_id).nonzero(as_tuple=True)[0]

#             print(f"\nSample {i}")
#             print("Seq len:", len(seq))
#             print("EOS positions:", eos_positions.tolist())


#             print(self.tokenizer.decode(
#             generated_ids[i],
#             skip_special_tokens=False))
#         print("="*20)
#         print(output.sequences.shape)
#         print(full_prefix_embeds.shape)
#         print(output.sequences[0][:20])
#         print(output.sequences[0][-20:])
#         # Decodificar los IDs generados a texto legible
#         decoded = self.tokenizer.batch_decode(
#             output.sequences, skip_special_tokens=True
#         )
        return {'sequences': output.sequences, 'decoded_sequences': decoded}
