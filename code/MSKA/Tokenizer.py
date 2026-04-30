import pickle
import torch
import json
from collections import defaultdict
from transformers import MBartTokenizer
import os


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, ignore_index: int = -100):
    """
    Prepara las entradas del decoder desplazando los tokens una posición a la derecha.

    El decoder del transformer necesita que en cada posición t la entrada sea el
    token t-1 (lo que ya generó), para predecir el token t.

    MBart no usa un token de inicio fijo (<bos>); usa el token del idioma destino
    (<de_DE> para alemán) como primer token. Por eso esta función busca ese token
    al final de la secuencia, lo extrae y lo coloca al inicio tras el desplazamiento.

    Ejemplo para Phoenix14t (alemán):
        labels:         [HALLO  WELT  </s>  <pad>]
        decoder_input:  [<de_DE> HALLO WELT  </s> ]
    """
    prev_output_tokens = input_ids.clone()

    # Reemplazar posibles -100 (ignore_index) por pad para no confundir el desplazamiento
    prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)

    # Encontrar el índice del último token real (no padding) en cada secuencia
    index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)

    # Marcar como ignore_index todo lo que viene después del token de idioma en input_ids
    # (no queremos que la loss se calcule sobre el padding)
    for ii, ind in enumerate(index_of_eos.squeeze(-1)):
        input_ids[ii, ind:] = ignore_index

    # Extraer el token de idioma (último token real) para ponerlo al inicio
    decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()

    # Desplazar todo a la derecha y colocar el token de idioma en la posición 0
    prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()
    prev_output_tokens[:, 0] = decoder_start_tokens

    return prev_output_tokens


# ---------------------------------------------------------------------------
# Clases base
# ---------------------------------------------------------------------------

class BaseTokenizer(object):
    def __init__(self, tokenizer_cfg):
        self.tokenizer_cfg = tokenizer_cfg

    def __call__(self, input_str):
        pass


# ---------------------------------------------------------------------------
# Tokenizador de texto natural (usado en la tarea S2T)
# ---------------------------------------------------------------------------

class TextTokenizer(BaseTokenizer):
    """
    Tokenizador para texto natural. Soporta dos modos:

      - 'sentencepiece': usa MBart (multilingüe de Meta), divide en subpalabras.
                         Recomendado para Phoenix14t (alemán).
                         Requiere un fichero pruneids para reducir el vocabulario
                         de ~250k tokens a solo los relevantes para el idioma del dataset.
                         Usar generate_pruneids.py para generarlo si no existe.

      - 'word':          vocabulario de palabras construido desde un JSON.
                         Más simple pero peor con palabras raras o morfología compleja.
    """

    def __init__(self, tokenizer_cfg):
        super().__init__(tokenizer_cfg)
        self.level = tokenizer_cfg.get('level', 'sentencepiece')

        if self.level == 'word':
            # --- Construcción del vocabulario desde fichero JSON ---
            self.min_freq = tokenizer_cfg.get('min_freq', 0)
            with open(tokenizer_cfg['tokenizer_file'], 'r') as f:
                tokenizer_info = json.load(f)

            self.word2fre = tokenizer_info['word2fre']
            self.special_tokens = tokenizer_info['special_tokens']

            # Construir id2token: primero tokens especiales, luego palabras por frecuencia desc.
            self.id2token = self.special_tokens[:]
            for w in sorted(self.word2fre.keys(), key=lambda w: self.word2fre[w])[::-1]:
                if self.word2fre[w] >= self.min_freq:
                    self.id2token.append(w)

            self.token2id = {t: id_ for id_, t in enumerate(self.id2token)}
            self.pad_index = self.token2id['<pad>']
            self.eos_index = self.token2id['</s>']
            self.unk_index = self.token2id['<unk>']
            self.sos_index = self.token2id['<s>']

            # Usar defaultdict para devolver <unk> ante palabras fuera de vocabulario
            self.token2id = defaultdict(lambda: self.unk_index, self.token2id)
            self.ignore_index = self.pad_index

        elif self.level == 'sentencepiece':
            # --- Carga del tokenizador MBart ---
            # Eliminamos pruneids_file del cfg antes de pasarlo a from_pretrained
            # porque no es un argumento válido de MBartTokenizer
            temp_cfg = tokenizer_cfg.copy()
            temp_cfg.pop('pruneids_file', None)
            self.tokenizer = MBartTokenizer.from_pretrained(**temp_cfg)
            self.pad_index = self.tokenizer.convert_tokens_to_ids('<pad>')
            self.ignore_index = self.pad_index

            # --- Carga del vocabulario reducido (pruneids) ---
            # pruneids mapea cada ID original de MBart a un ID reducido,
            # eliminando los tokens irrelevantes para el idioma del dataset.
            # Esto reduce la capa de clasificación final de ~250k a solo los tokens necesarios.
            # Si no tienes este fichero, usa generate_pruneids.py para generarlo.
            self.pruneids_file = tokenizer_cfg.get('pruneids_file', None)
            if self.pruneids_file and os.path.exists(self.pruneids_file):
                with open(self.pruneids_file, 'rb') as f:
                    self.pruneids = pickle.load(f)
                # Verificar que los tokens especiales no se han movido tras la poda
                for t in ['<pad>', '<s>', '</s>', '<unk>']:
                    id_ = self.tokenizer.convert_tokens_to_ids(t)
                    assert self.pruneids[id_] == id_, '{}->{}'.format(id_, self.pruneids[id_])
            else:
                # Sin pruneids: mapeo de identidad (vocabulario completo de MBart).
                # Funciona correctamente pero es más lento por el vocabulario grande.
                # Se recomienda generar pruneids con generate_pruneids.py.
                print("Aviso: pruneids_file no encontrado. Usando vocabulario MBart completo.")
                vocab_size = len(self.tokenizer)
                self.pruneids = {i: i for i in range(vocab_size)}

            # Mapeo inverso: ID reducido → ID original de MBart (necesario para batch_decode)
            self.pruneids_reverse = {i2: i1 for i1, i2 in self.pruneids.items()}

            # Índices especiales en el espacio del vocabulario reducido
            target_lang_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tgt_lang)
            self.lang_index = self.pruneids[target_lang_id]
            self.sos_index  = self.lang_index   # MBart usa el token de idioma como inicio
            self.eos_index  = self.pruneids[self.tokenizer.convert_tokens_to_ids('</s>')]
        else:
            raise ValueError(f"Nivel de tokenización desconocido: {self.level}. Usa 'sentencepiece' o 'word'.")

    def generate_decoder_labels(self, input_ids):
        """
        Genera los labels para la loss del decoder.
        El token de idioma (<de_DE>) se reemplaza por ignore_index (-100)
        para que la loss no lo tenga en cuenta (no es una palabra real a predecir).
        """
        return torch.where(
            input_ids == self.lang_index,
            torch.ones_like(input_ids) * self.ignore_index,
            input_ids,
        )

    def generate_decoder_inputs(self, input_ids):
        """
        Genera las entradas del decoder desplazando los tokens a la derecha.
        Ver shift_tokens_right para más detalle.
        """
        return shift_tokens_right(input_ids, pad_token_id=self.pad_index,
                                  ignore_index=self.pad_index)

    def prune(self, input_ids):
        """
        Convierte IDs del vocabulario completo de MBart al vocabulario reducido.
        Si un ID no está en pruneids, se mapea a <unk>.
        """
        unk_id = self.pruneids[self.tokenizer.convert_tokens_to_ids('<unk>')]
        pruned_input_ids = []
        for single_seq in input_ids:
            pruned_single_seq = []
            for id_ in single_seq:
                if id_ not in self.pruneids:
                    print(f"Token fuera de vocabulario: {id_} ({self.tokenizer.convert_ids_to_tokens(id_)})")
                    new_id = unk_id
                else:
                    new_id = self.pruneids[id_]
                pruned_single_seq.append(new_id)
            pruned_input_ids.append(pruned_single_seq)
        return torch.tensor(pruned_input_ids, dtype=torch.long)

    def prune_reverse(self, pruned_input_ids):
        """
        Convierte IDs del vocabulario reducido de vuelta al vocabulario completo de MBart.
        Necesario antes de llamar al decoder de MBart en batch_decode.
        """
        unk_id = self.tokenizer.convert_tokens_to_ids('<unk>')
        batch_size, max_len = pruned_input_ids.shape
        input_ids = pruned_input_ids.clone()
        for b in range(batch_size):
            for i in range(max_len):
                id_ = input_ids[b, i].item()
                input_ids[b, i] = self.pruneids_reverse.get(id_, unk_id)
        return input_ids

    def __call__(self, input_str):
        """
        Tokeniza un batch de frases y devuelve labels y decoder_input_ids.

        labels:             IDs esperados en la salida (para calcular la loss).
        decoder_input_ids:  IDs desplazados a la derecha (entrada del decoder).
        """
        if self.level == 'sentencepiece':
            with self.tokenizer.as_target_tokenizer():
                raw_outputs = self.tokenizer(
                    input_str,
                    return_attention_mask=True,
                    return_length=True,
                    padding='longest',
                )
            pruned_input_ids = self.prune(raw_outputs['input_ids'])
            return {
                'labels':            self.generate_decoder_labels(pruned_input_ids),
                'decoder_input_ids': self.generate_decoder_inputs(pruned_input_ids),
            }

        elif self.level == 'word':
            # Tokenizar cada frase y construir labels + decoder_input_ids
            batch_labels, batch_decoder_input_ids, batch_lengths = [], [], []
            for text in input_str:
                labels = []
                decoder_input_ids = [self.sos_index]  # el decoder empieza con <s>
                for t in text.split():
                    id_ = self.token2id[t]
                    labels.append(id_)
                    decoder_input_ids.append(id_)
                labels.append(self.eos_index)
                batch_labels.append(labels)
                batch_decoder_input_ids.append(decoder_input_ids)
                batch_lengths.append(len(labels))

            # Padding hasta la longitud máxima del batch
            max_length = max(batch_lengths)
            padded_labels, padded_decoder_inputs = [], []
            for labels, decoder_input_ids in zip(batch_labels, batch_decoder_input_ids):
                padded_labels.append(
                    labels + [self.pad_index] * (max_length - len(labels))
                )
                padded_decoder_inputs.append(
                    decoder_input_ids + [self.ignore_index] * (max_length - len(decoder_input_ids))
                )
            return {
                'labels':            torch.tensor(padded_labels, dtype=torch.long),
                'decoder_input_ids': torch.tensor(padded_decoder_inputs, dtype=torch.long),
            }
        else:
            raise ValueError

    def batch_decode(self, sequences):
        """
        Convierte un batch de secuencias de IDs de vuelta a texto legible.
        Se elimina el primer token (token de idioma / inicio de secuencia).
        """
        sequences = sequences[:, 1:]  # eliminar el token de inicio

        if self.level == 'sentencepiece':
            # Deshacer la poda para recuperar los IDs originales de MBart
            sequences_ = self.prune_reverse(sequences)
            decoded_sequences = self.tokenizer.batch_decode(sequences_, skip_special_tokens=True)
            # MBart a veces junta el punto final con la última palabra en alemán ("Wort.")
            # Se añade un espacio para normalizarlo ("Wort .")
            if 'de' in self.tokenizer.tgt_lang:
                for di, d in enumerate(decoded_sequences):
                    if len(d) > 2 and d[-1] == '.' and d[-2] != ' ':
                        decoded_sequences[di] = d[:-1] + ' .'
            return decoded_sequences

        elif self.level == 'word':
            return [' '.join([self.id2token[s] for s in seq]) for seq in sequences]
        else:
            raise ValueError


# ---------------------------------------------------------------------------
# Tokenizadores de glosas
# ---------------------------------------------------------------------------

class BaseGlossTokenizer(BaseTokenizer):
    """
    Tokenizador base para glosas. Carga el vocabulario desde un fichero pickle
    con el mapeo gloss→id y construye el inverso id→gloss.
    Devuelve <unk> para glosas no vistas durante el entrenamiento.
    """

    def __init__(self, tokenizer_cfg):
        super().__init__(tokenizer_cfg)
        with open(tokenizer_cfg['gloss2id_file'], 'rb') as f:
            self.gloss2id = pickle.load(f)

        # defaultdict para devolver <unk> ante glosas desconocidas
        self.gloss2id = defaultdict(lambda: self.gloss2id['<unk>'], self.gloss2id)

        # Construir el mapeo inverso id → glosa
        self.id2gloss = {id_: gls for gls, id_ in self.gloss2id.items()}

        self.lower_case = tokenizer_cfg.get('lower_case', True)

    def convert_tokens_to_ids(self, tokens):
        """Acepta tanto una glosa individual como una lista de glosas."""
        if isinstance(tokens, list):
            return [self.convert_tokens_to_ids(t) for t in tokens]
        return self.gloss2id[tokens]

    def convert_ids_to_tokens(self, ids):
        """Acepta tanto un ID individual como una lista de IDs."""
        if isinstance(ids, list):
            return [self.convert_ids_to_tokens(i) for i in ids]
        return self.id2gloss[ids]

    def __len__(self):
        return len(self.id2gloss)


class GlossTokenizer_S2G(BaseGlossTokenizer):
    """
    Tokenizador de glosas para la tarea de reconocimiento (S2G).

    Añade el concepto de token de silencio (<s> o <si>), que representa
    la ausencia de seña. CTC lo necesita en el índice 0 para modelar
    los espacios entre glosas durante la decodificación.
    """

    def __init__(self, tokenizer_cfg):
        super().__init__(tokenizer_cfg)

        # Detectar el token de silencio (varía según el dataset) # FIXME Averiguar cual es el de Phoenix14t
        if '<s>' in self.gloss2id:
            self.silence_token = '<s>'
        elif '<si>' in self.gloss2id:
            self.silence_token = '<si>'
        else:
            raise ValueError("No se encontró token de silencio (<s> o <si>) en el vocabulario de glosas.")

        self.silence_id = self.convert_tokens_to_ids(self.silence_token)
        # CTC requiere que el token de silencio esté en el índice 0
        assert self.silence_id == 0, f"El token de silencio debe tener id=0, tiene id={self.silence_id}"

        self.pad_token = '<pad>'
        self.pad_id = self.convert_tokens_to_ids(self.pad_token)

    def __call__(self, batch_gls_seq):
        """
        Tokeniza un batch de secuencias de glosas y aplica padding.

        Devuelve:
          gls_lengths:   longitudes reales de cada secuencia (necesarias para CTC).
          gloss_labels:  IDs de glosas con padding hasta la longitud máxima del batch.
        """
        max_length = max(len(gls_seq.split()) for gls_seq in batch_gls_seq)
        gls_lengths, batch_gls_ids = [], []

        for gls_seq in batch_gls_seq:
            gls_ids = [
                self.gloss2id[gls.lower() if self.lower_case else gls]
                for gls in gls_seq.split()
            ]
            gls_lengths.append(len(gls_ids))
            # Padding con pad_id hasta max_length
            gls_ids += [self.pad_id] * (max_length - len(gls_ids))
            batch_gls_ids.append(gls_ids)

        return {
            'gls_lengths':  torch.tensor(gls_lengths),
            'gloss_labels': torch.tensor(batch_gls_ids),
        }


class GlossTokenizer_G2T(BaseGlossTokenizer):
    """
    Tokenizador de glosas para usarlas como entrada del translator (G2T).

    A diferencia de S2G, añade al final de cada secuencia </s> y el token
    de idioma fuente (<src_lang>), que es el formato que MBart espera
    como entrada del encoder. También genera attention_mask para que el
    transformer ignore los tokens de padding.
    """

    def __init__(self, tokenizer_cfg):
        super().__init__(tokenizer_cfg)
        self.src_lang = tokenizer_cfg['src_lang']

    def __call__(self, batch_gls_seq):
        # +2 por </s> y <src_lang> que se añaden al final de cada secuencia
        max_length = max(len(gls_seq.split()) for gls_seq in batch_gls_seq) + 2
        batch_gls_ids = []
        attention_mask = torch.zeros([len(batch_gls_seq), max_length], dtype=torch.long)

        for ii, gls_seq in enumerate(batch_gls_seq):
            gls_ids = [
                self.gloss2id[gls.lower() if self.lower_case else gls]
                for gls in gls_seq.split()
            ]
            # Añadir tokens de fin de secuencia e idioma fuente
            gls_ids += [self.gloss2id['</s>'], self.gloss2id[self.src_lang]]
            # Marcar como 1 solo las posiciones con tokens reales
            attention_mask[ii, :len(gls_ids)] = 1
            # Padding hasta max_length
            gls_ids += [self.gloss2id['<pad>']] * (max_length - len(gls_ids))
            batch_gls_ids.append(gls_ids)

        return {
            'input_ids':      torch.tensor(batch_gls_ids, dtype=torch.long),
            'attention_mask': attention_mask,
        }
