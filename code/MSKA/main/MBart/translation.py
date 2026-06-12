import torch
from transformers import MBartForConditionalGeneration, MBartTokenizer, MBartConfig
from aux.utils import XentLoss
from Recognition.Tokenizer import GlossTokenizer_G2T, TextTokenizer
import pickle
import math


class TranslationNetwork(torch.nn.Module):
    """
    Módulo de traducción: convierte features visuales en texto natural (alemán en Phoenix14t).

    Envuelve MBart (transformer multilingüe preentrenado de Meta) adaptándolo para
    recibir features visuales como entrada del encoder en lugar de texto.

    Pipeline:
        features visuales (del VLMapper)
            → prepare_feature_inputs (añadir sufijo de idioma + padding)
            → MBart encoder
            → MBart decoder (con el texto de referencia en entrenamiento,
                             con beam search en evaluación)
            → XentLoss → translation_loss
    """

    def __init__(self, input_type='feature', cfg=None, task='S2T') -> None:
        super().__init__()
        self.task = task
        self.input_type = input_type

        # Tokenizador de texto (MBart sentencepiece para Phoenix14t)
        self.text_tokenizer = TextTokenizer(tokenizer_cfg=cfg['TextTokenizer'])

        # Cargar MBart preentrenado. overwrite_cfg permite sobreescribir parámetros
        # del config de MBart (p.ej. forzar un idioma destino específico).
        self.model = MBartForConditionalGeneration.from_pretrained(
            cfg['pretrained_model_name_or_path'],
            **cfg.get('overwrite_cfg', {}),
        )

        # XentLoss: cross-entropy con label smoothing=0.2.
        # El label smoothing evita que el modelo sea demasiado confiado asignando
        # probabilidad 0.8 al token correcto y 0.2/V al resto. Mejora generalización.
        self.translation_loss_fun = XentLoss(
            pad_index=self.text_tokenizer.pad_index,
            smoothing=0.2,
        )

        # Dimensión interna de MBart (d_model). Es la dimensión que espera el encoder.
        self.input_dim = self.model.config.d_model

        # Factor de escala para los embeddings: MBart multiplica internamente sus
        # embeddings de texto por sqrt(d_model). Las features visuales que llegan del
        # VLMapper no tienen esta escala, así que hay que aplicársela manualmente
        # para que estén en el mismo rango numérico que los embeddings de texto.
        self.input_embed_scale = math.sqrt(self.model.config.d_model)

        # Tokenizador de glosas para el encoder (G2T: glosa → texto).
        # Distinto al GlossTokenizer_S2G: este añade </s> y <src_lang> al final,
        # que es el formato que MBart espera como entrada del encoder.
        self.gloss_tokenizer = GlossTokenizer_G2T(tokenizer_cfg=cfg['GlossTokenizer'])

        # Tabla de embeddings de glosas, inicializada con vectores preentrenados.
        # Se usa para construir el sufijo que se añade a las features visuales.
        self.gloss_embedding = self.build_gloss_embedding(**cfg['GlossEmbedding'])

        # gls_eos: qué embedding usar como token de fin de secuencia del encoder.
        #   'gls': usa el </s> del vocabulario de glosas
        #   otro:  usa el </s> del vocabulario de texto de MBart
        self.gls_eos = cfg.get('gls_eos', 'gls')

        # Cargar checkpoint propio si se especifica (distinto del preentrenado de MBart)
        if 'load_ckpt' in cfg:
            self.load_from_pretrained_ckpt(cfg['load_ckpt'])

    def load_from_pretrained_ckpt(self, pretrained_ckpt):
        """
        Carga pesos desde un checkpoint del proyecto (no de HuggingFace).
        Filtra solo las claves que pertenecen a 'translation_network.*'.
        """
        checkpoint = torch.load(pretrained_ckpt, map_location='cpu')['model_state']
        load_dict = {
            k.replace('translation_network.', ''): v
            for k, v in checkpoint.items()
            if 'translation_network' in k
        }
        self.load_state_dict(load_dict)

    def build_gloss_embedding(self, gloss2embed_file, from_scratch=False, freeze=False):
        """
        Construye la tabla de embeddings de glosas (num_glosas, d_model).

        Si from_scratch=False (por defecto): inicializa cada fila con el embedding
        preentrenado de su glosa desde gloss2embed_file. Las glosas sin embedding
        preentrenado conservan la inicialización aleatoria de PyTorch.

        Si from_scratch=True: inicialización aleatoria completa (freeze debe ser False).

        El embedding de <pad> se inicializa a cero (padding_idx) para que el modelo
        aprenda a ignorar las posiciones de padding.
        """
        gloss_embedding = torch.nn.Embedding(
            num_embeddings=len(self.gloss_tokenizer.id2gloss),
            embedding_dim=self.model.config.d_model,
            padding_idx=self.gloss_tokenizer.gloss2id['<pad>'],
        )

        if not from_scratch:
            gls2embed = torch.load(gloss2embed_file)
            self.gls2embed = gls2embed  # guardado para que VLMapper pueda acceder a él

            # Inicializar cada fila con el embedding preentrenado de su glosa
            with torch.no_grad():  # solo inicialización, no un paso de entrenamiento
                for id_, gls in self.gloss_tokenizer.id2gloss.items():
                    if gls in gls2embed:
                        gloss_embedding.weight[id_, :] = gls2embed[gls]
                        # Las glosas no presentes en gls2embed se dejan con init aleatoria
        else:
            assert freeze == False, "No tiene sentido congelar un embedding inicializado desde cero."

        return gloss_embedding

    def prepare_gloss_inputs(self, input_ids):
        """
        Convierte IDs de glosas a embeddings escalados.
        Se usa cuando el encoder recibe glosas directamente (no features visuales).
        """
        return self.gloss_embedding(input_ids) * self.input_embed_scale

    def prepare_feature_inputs(self, input_feature, input_lengths,
                                gloss_embedding=None, gloss_lengths=None):
        """
        Prepara las entradas del encoder de MBart a partir de features visuales.

        MBart espera como entrada del encoder una secuencia de embeddings con un
        sufijo que indica el idioma fuente. Esta función:
          1. Construye el sufijo [</s>, <src_lang>] en el espacio de embeddings.
          2. Concatena las features visuales con el sufijo.
          3. Hace padding hasta la longitud máxima del batch.
          4. Construye la attention_mask.

        El sufijo es necesario para que MBart sepa de qué idioma proviene la entrada
        y en qué idioma debe traducir. Es equivalente a lo que GlossTokenizer_G2T
        añade cuando la entrada es texto de glosas.

        Args:
            input_feature:   features del VLMapper, shape (B, T, D).
            input_lengths:   longitudes reales de cada secuencia, shape (B,).
            gloss_embedding: embeddings de glosas (solo para input_type='gloss+feature').
            gloss_lengths:   longitudes de las glosas (solo para input_type='gloss+feature').

        Returns:
            transformer_inputs: dict con 'inputs_embeds' (B, T+2, D) y 'attention_mask' (B, T+2).
        """

        # --- Construir el sufijo de idioma ---
        # El sufijo indica a MBart el idioma fuente. Por defecto es [</s>, <src_lang>].
        # gls_eos controla qué embedding usar para </s>:
        #   'gls': el </s> del vocabulario de glosas (más cercano al dominio visual)
        #   otro:  el </s> del vocabulario de texto de MBart
        if self.task == 'S2T_glsfree':
            # Sin sufijo de glosa: la entrada son solo features visuales puras
            suffix_len = 0
            suffix_embedding = None
        else:
            if self.gls_eos == 'gls':
                suffix_embedding = [
                    self.gloss_embedding.weight[
                        self.gloss_tokenizer.convert_tokens_to_ids('</s>'), :
                    ]
                ]
            else:
                suffix_embedding = [
                    self.model.model.shared.weight[self.text_tokenizer.eos_index, :]
                ]

            # Añadir el token de idioma fuente (<src_lang>) al sufijo
            if self.task in ['S2T', 'G2T'] and self.gloss_embedding:
                if self.gls_eos == 'gls':
                    src_lang_code_embedding = self.gloss_embedding.weight[
                        self.gloss_tokenizer.convert_tokens_to_ids(self.gloss_tokenizer.src_lang), :
                    ]
                else:
                    # Obtener el embedding del token de idioma desde el vocabulario de MBart
                    src_lang_id = self.text_tokenizer.pruneids[30]
                    assert src_lang_id == 31  # verificación hardcodeada para Phoenix14t
                    src_lang_code_embedding = self.model.model.shared.weight[src_lang_id, :]
                suffix_embedding.append(src_lang_code_embedding)

            suffix_len = len(suffix_embedding)  # normalmente 2: [</s>, <src_lang>]
            suffix_embedding = torch.stack(suffix_embedding, dim=0)  # (2, D)

        # --- Construir inputs_embeds con padding ---
        max_length = torch.max(input_lengths) + suffix_len  # longitud máxima del batch
        inputs_embeds = []
        attention_mask = torch.zeros(
            [input_feature.shape[0], max_length],
            dtype=torch.long, device=input_feature.device,
        )

        for ii, feature in enumerate(input_feature):
            valid_len = input_lengths[ii]

            # Seleccionar las features válidas (sin padding)
            if 'gloss+feature' in self.input_type:
                # Modo híbrido: concatenar embeddings de glosas + features visuales
                valid_feature = torch.cat([
                    gloss_embedding[ii, :gloss_lengths[ii], :],
                    feature[:valid_len - gloss_lengths[ii], :],
                ], dim=0)
            else:
                # Modo estándar: solo features visuales
                valid_feature = feature[:valid_len, :]  # (T, D)

            # Concatenar con el sufijo de idioma
            if suffix_embedding is not None:
                feature_w_suffix = torch.cat([valid_feature, suffix_embedding], dim=0)  # (T+2, D)
            else:
                feature_w_suffix = valid_feature

            # Padding con ceros hasta max_length si la secuencia es más corta
            if feature_w_suffix.shape[0] < max_length:
                pad_len = max_length - feature_w_suffix.shape[0]
                padding = torch.zeros(
                    [pad_len, feature_w_suffix.shape[1]],
                    dtype=feature_w_suffix.dtype,
                    device=feature_w_suffix.device,
                )
                inputs_embeds.append(torch.cat([feature_w_suffix, padding], dim=0))
            else:
                inputs_embeds.append(feature_w_suffix)

            # Marcar como 1 solo las posiciones reales (features + sufijo, sin padding)
            attention_mask[ii, :valid_len + suffix_len] = 1

        return {
            # Escalar por sqrt(d_model) para alinear con el rango de los embeddings de MBart
            'inputs_embeds': torch.stack(inputs_embeds, dim=0) * self.input_embed_scale,  # (B, T+2, D)
            'attention_mask': attention_mask,  # (B, T+2)
        }

    def forward(self, **kwargs):
        """
        Forward pass en entrenamiento.

        Recibe el batch completo de collate_fn via **kwargs y:
          1. Extrae las features visuales y sus longitudes.
          2. Prepara las entradas del encoder (sufijo + padding).
          3. Pasa todo a MBart (encoder + decoder en un solo forward).
          4. Calcula la XentLoss sobre los logits del decoder.

        MBart recibe:
          - inputs_embeds + attention_mask: entrada del encoder (features visuales)
          - decoder_input_ids:             tokens de referencia desplazados a la derecha
          - labels:                        tokens de referencia para calcular la loss

        Guarda transformer_inputs en el output para reutilizarlos en generate()
        sin recalcularlos.
        """
        # Extraer y eliminar de kwargs los argumentos que no acepta MBart directamente
        input_feature = kwargs.pop('input_feature')
        input_lengths = kwargs.pop('input_lengths')
        kwargs.pop('gloss_ids', None)      # no se usa en este forward
        kwargs.pop('gloss_lengths', None)  # no se usa en este forward

        # Preparar inputs_embeds y attention_mask para el encoder
        new_kwargs = self.prepare_feature_inputs(input_feature, input_lengths)
        kwargs = {**kwargs, **new_kwargs}

        # Mover todos los tensores a GPU
        kwargs = {key: value.to('cuda') for key, value in kwargs.items()}

        # Forward de MBart: encoder + decoder en un solo paso
        # kwargs contiene: inputs_embeds, attention_mask, decoder_input_ids, labels
        output_dict = self.model(**kwargs, return_dict=True)

        # Calcular la translation loss sobre los logits del decoder
        # log_softmax convierte logits a log-probabilidades (necesario para XentLoss)
        log_prob = torch.nn.functional.log_softmax(output_dict['logits'], dim=-1)  # (B, T, vocab)
        batch_loss_sum = self.translation_loss_fun(log_probs=log_prob, targets=kwargs['labels'])
        # Normalizar por batch_size para que la loss no dependa del tamaño del batch
        output_dict['translation_loss'] = batch_loss_sum / log_prob.shape[0]

        # Guardar para reutilizar en generate() sin recalcular
        output_dict['transformer_inputs'] = kwargs

        return output_dict

    def generate(self, input_ids=None, attention_mask=None,
                 inputs_embeds=None, input_lengths=None,
                 num_beams=4, max_length=100, length_penalty=1, **kwargs):
        """
        Generación de texto en evaluación mediante beam search.

        Solo se llama desde SignLanguageModel.generate_txt(), nunca durante
        el entrenamiento. Reutiliza los inputs_embeds y attention_mask que
        forward() guardó en transformer_inputs.

        Args:
            inputs_embeds:  features visuales preparadas, shape (B, T+2, D).
            attention_mask: máscara del encoder, shape (B, T+2).
            num_beams:      número de hipótesis del beam search (más = más preciso pero más lento).
            max_length:     longitud máxima de la traducción generada en tokens.
            length_penalty: penalización por longitud. >1 favorece frases largas, <1 cortas.

        Returns:
            dict con 'sequences' (IDs generados) y 'decoded_sequences' (texto decodificado).
        """
        assert attention_mask is not None
        assert inputs_embeds is not None

        batch_size = attention_mask.shape[0]

        # El decoder de MBart empieza con el token de idioma destino (<de_DE> para Phoenix14t).
        # Esto le indica en qué idioma debe generar la traducción.
        decoder_input_ids = torch.ones(
            [batch_size, 1], dtype=torch.long, device=attention_mask.device
        ) * self.text_tokenizer.sos_index  # sos_index = lang_index = <de_DE>

        # Beam search de HuggingFace: genera hasta max_length tokens o hasta </s>
        output_dict = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            num_beams=num_beams,
            length_penalty=length_penalty,
            max_length=max_length,
            return_dict_in_generate=True,
        )

        # Decodificar las secuencias de IDs a texto legible
        # batch_decode deshace el pruneids y elimina tokens especiales
        output_dict['decoded_sequences'] = self.text_tokenizer.batch_decode(output_dict['sequences'])

        return output_dict
