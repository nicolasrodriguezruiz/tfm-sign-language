import random

import torchvision
import torch
import utils as utils
import torch.utils.data.dataset as Dataset
from PIL import Image
import os
from Tokenizer import GlossTokenizer_S2G
from transformers import AutoTokenizer
import numpy as np


class S2T_Dataset(Dataset.Dataset):
    """
    Dataset para reconocimiento/traducción de lenguaje de señas.
    Soporta dos tareas:
      - S2G (Sign-to-Gloss): reconocimiento de señas a glosas.
      - S2T (Sign-to-Text): traducción de señas a texto natural (alemán, Phoenix14t).

    Los datos de entrada son keypoints corporales (coordenadas de articulaciones)
    extraídos de vídeos de lenguaje de señas.

    En S2T el texto se tokeniza con el tokenizador de Qwen2.5-1.5B en lugar de MBart.
    No se necesita shift_tokens_right ni pruneids: Qwen es decoder-only y gestiona
    internamente el teacher forcing a partir de los labels.
    """

    def __init__(self, path, tokenizer, config, args, phase, training_refurbish=False):
        self.config = config
        self.args = args
        self.training_refurbish = training_refurbish
        self.phase = phase

        # Número máximo de frames por muestra
        self.clip_len = 400

        # En entrenamiento se aplica augmentación temporal (acelerar/ralentizar el vídeo).
        # En validación/test se usa la velocidad original (factor 1).
        if phase == 'train':
            self.tmin, self.tmax = 0.5, 1.5
        else:
            self.tmin, self.tmax = 1, 1

        # Dimensiones de los frames según el dataset (Phoenix14t usa 210x260)
        if config['data']['dataset_name'].lower() == 'csl-daily':
            self.w, self.h = 512, 512
        else:
            self.w, self.h = 210, 260

        # Leer config de augmentación de keypoints (solo se aplica en entrenamiento)
        self.augmentation_cfg = config['data'].get('augmentation', {})

        self.raw_data = utils.load_dataset_file(path)
        self.tokenizer = tokenizer  # GlossTokenizer_S2G para las glosas
        self.max_length = config['data']['max_length']
        self.list = [key for key in self.raw_data]

        # En S2T cargamos el tokenizador de Qwen para el texto.
        # A diferencia de MBart, Qwen no necesita pruneids ni shift_tokens_right:
        # se tokeniza directamente y se marcan las posiciones de padding con -100.
        if self.config['task'] == 'S2T':
            qwen_model_name = config['model']['TranslationNetwork'].get(
                'pretrained_model_name_or_path', 'Qwen/Qwen2.5-1.5B'
            )
            self.text_tokenizer = AutoTokenizer.from_pretrained(
                qwen_model_name, trust_remote_code=True
            )
            if self.text_tokenizer.pad_token is None:
                self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
            self.text_max_length = config['data'].get('text_max_length', 128)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, index):
        """
        Devuelve una muestra individual sin procesar.
        El procesamiento pesado (selección de frames, padding, normalización)
        ocurre en collate_fn al construir el batch.
        """
        key = self.list[index]
        sample = self.raw_data[key]

        gloss = sample['gloss']
        length = sample['num_frames']
        text = sample['text'] if self.config['task'] != 'S2G' else None

        # Keypoints: (articulaciones, tiempo, canales) → (canales, tiempo, articulaciones)
        keypoint = sample['keypoint'].permute(2, 0, 1).to(torch.float32)
        name_sample = sample['name']

        return name_sample, keypoint, gloss, text, length

    # ------------------------------------------------------------------ #
    #  Selección de frames                                                 #
    # ------------------------------------------------------------------ #

    def get_selected_index(self, vlen):
        """
        Selecciona qué frames usar de un vídeo de longitud `vlen`.

        En validación/test (tmin == tmax == 1):
          - Si el vídeo cabe en clip_len, se usan todos sus frames.
          - Si es más largo, se recorta centrado.
          - La longitud final se ajusta al múltiplo de 4 más cercano por abajo
            (necesario para las capas de downsampling x2 del modelo).

        En entrenamiento:
          - Se elige aleatoriamente una longitud entre tmin*vlen y tmax*vlen.
          - Si la longitud elegida es menor que vlen, se muestrea sin reemplazo.
          - Si es mayor, se duplican frames aleatorios para completar.

        Devuelve:
          frame_index (np.array): índices de los frames seleccionados.
          valid_len   (int):      número de frames válidos (== len(frame_index)).
        """

        # --- Rama validación/test ---
        if self.tmin == 1 and self.tmax == 1:
            if vlen <= self.clip_len:
                frame_index = np.arange(vlen)
                valid_len = vlen
            else:
                # Recorte centrado
                an = (vlen - self.clip_len) // 2
                en = vlen - self.clip_len - an
                frame_index = np.arange(vlen)[an: -en]
                valid_len = self.clip_len

            # Ajuste al múltiplo de 4 más cercano por abajo
            remainder = valid_len % 4
            if remainder != 0:
                valid_len -= remainder
                frame_index = frame_index[:valid_len]

            return frame_index, valid_len

        # --- Rama entrenamiento (augmentación temporal) ---
        min_len = min(int(self.tmin * vlen), self.clip_len)
        max_len = min(self.clip_len, int(self.tmax * vlen))
        selected_len = np.random.randint(min_len, max_len + 1)

        remainder = selected_len % 4
        if remainder != 0:
            selected_len += (4 - remainder)

        if selected_len <= vlen:
            selected_index = sorted(np.random.permutation(np.arange(vlen))[:selected_len])
        else:
            extra = np.random.randint(0, vlen, selected_len - vlen)
            selected_index = sorted(np.concatenate([np.arange(vlen), extra]))

        return selected_index, selected_len

    # ------------------------------------------------------------------ #
    #  Augmentaciones espaciales sobre keypoints                           #
    # ------------------------------------------------------------------ #

    def rotate_points(self, points, angle):
        """Rota los puntos `angle` radianes alrededor del origen (0, 0)."""
        rotation_matrix = np.array([
            [ np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)],
        ])
        return np.dot(points, rotation_matrix.T)

    def translate_points(self, points, translation):
        """Traslada los puntos sumando el vector `translation`."""
        return points + translation

    def scale_points(self, points, scale_factor):
        """Escala los puntos multiplicando por `scale_factor`."""
        return points * scale_factor

    def random_joint_dropout(self, keypoints, dropout_prob):
        """
        Enmascara articulaciones enteras aleatoriamente poniendo sus coordenadas a 0.
        Simula oclusiones visuales: una mano tapada, keypoints no detectados, etc.

        keypoints shape: (C, T, V)
        dropout_prob: probabilidad de enmascarar cada articulación (p.ej. 0.1 = 10%)
        """
        C, T, V = keypoints.shape
        # Máscara por articulación: misma máscara para todos los frames de esa articulación
        mask = torch.bernoulli(torch.full((V,), 1 - dropout_prob)).bool()
        keypoints[:, :, ~mask] = 0
        return keypoints

    def random_gaussian_noise(self, keypoints, std):
        """
        Añade ruido gaussiano a las coordenadas espaciales (canales x e y).
        Introduce variabilidad sin alterar la estructura global de la pose.
        No se aplica al canal de confianza (índice 2).

        keypoints shape: (C, T, V)
        std: desviación estándar del ruido sobre coordenadas normalizadas [-1, 1]
        """
        noise = torch.randn_like(keypoints[:2, :, :]) * std
        keypoints[:2, :, :] += noise
        return keypoints

    def random_move(self, data_numpy):
        """
        Aplica una rotación aleatoria a los keypoints con probabilidad 0.5.
        El ángulo se elige uniformemente en [-15°, 15°].

        Recibe y devuelve un tensor con shape (T, articulaciones, canales_xy).
        """
        angle = np.radians(np.random.uniform(-15, 15))
        if np.random.uniform(0, 1) >= 0.5:
            data_numpy = self.rotate_points(data_numpy, angle)
        return torch.from_numpy(data_numpy)

    def augment_preprocess_inputs(self, is_train, keypoints=None):
        """
        Normaliza los keypoints al rango [-1, 1] y aplica augmentaciones en entrenamiento.

        Normalización:
          1. Divide X por el ancho del frame.
          2. Invierte Y (en imagen Y crece hacia abajo).
          3. Divide Y por el alto del frame.
          4. Centra ambos ejes: (valor - 0.5) / 0.5 → [-1, 1].

        Augmentaciones (solo en entrenamiento, controladas desde el config YAML):
          - random_move:         rotación aleatoria ±15°
          - joint_dropout:       enmascarado aleatorio de articulaciones
          - gaussian_noise:      ruido gaussiano en coordenadas x, y
        """
        # Normalización (siempre)
        keypoints[:, 0, :, :] /= self.w
        keypoints[:, 1, :, :] = self.h - keypoints[:, 1, :, :]
        keypoints[:, 1, :, :] /= self.h
        keypoints[:, :2, :, :] = (keypoints[:, :2, :, :] - 0.5) / 0.5

        if is_train == 'train':
            # Rotación aleatoria
            keypoints[:, :2, :, :] = self.random_move(
                keypoints[:, :2, :, :].permute(0, 2, 3, 1).numpy()
            ).permute(0, 3, 1, 2)

            # Joint dropout (si está configurado)
            if self.augmentation_cfg.get('joint_dropout', 0) > 0:
                for i in range(keypoints.shape[0]):
                    keypoints[i] = self.random_joint_dropout(
                        keypoints[i],
                        dropout_prob=self.augmentation_cfg['joint_dropout'],
                    )

            # Ruido gaussiano (si está configurado)
            if self.augmentation_cfg.get('gaussian_noise', 0) > 0:
                for i in range(keypoints.shape[0]):
                    keypoints[i] = self.random_gaussian_noise(
                        keypoints[i],
                        std=self.augmentation_cfg['gaussian_noise'],
                    )

        return keypoints

    # ------------------------------------------------------------------ #
    #  Construcción del batch                                              #
    # ------------------------------------------------------------------ #

    def _tokenize_text_for_qwen(self, text_batch):
        """
        Tokeniza un batch de frases en alemán con el tokenizador de Qwen.

        A diferencia de MBart, Qwen es decoder-only y no necesita:
          - shift_tokens_right (lo gestiona HuggingFace internamente con labels)
          - pruneids (Qwen ya tiene un vocabulario apropiado)
          - token de idioma al inicio (no es un modelo multilingüe encoder-decoder)

        Los tokens de padding se marcan con -100 en labels para que la loss
        no se calcule sobre ellos.

        Returns:
            labels:            IDs del texto con -100 en posiciones de padding. (B, L)
            decoder_input_ids: IDs del texto con pad_token_id en posiciones de padding. (B, L)
                               Qwen los usa para teacher forcing internamente.
        """
        encoded = self.text_tokenizer(
            text_batch,
            padding='longest',
            truncation=True,
            max_length=self.text_max_length,
            return_tensors='pt',
        )
        input_ids = encoded['input_ids']  # (B, L)

        # labels: -100 donde hay padding (PyTorch ignora estos en la loss)
        labels = input_ids.clone()
        labels[labels == self.text_tokenizer.pad_token_id] = -100

        return {
            'labels':            labels,
            'decoder_input_ids': input_ids,
        }

    def collate_fn(self, batch):
        """
        Función llamada por el DataLoader para unir muestras en un batch.

        Pasos:
          1. Para cada muestra, seleccionar los frames según get_selected_index.
          2. Hacer padding en la dimensión temporal para igualar longitudes.
          3. Normalizar y augmentar los keypoints.
          4. Construir las máscaras de atención para el recognition transformer
             (1 = posición real, 0 = padding).
          5. Tokenizar glosas y, si la tarea es S2T, también el texto con Qwen.

        Las longitudes se reducen dos veces a la mitad (factor /4 total) porque
        el encoder aplica dos capas de downsampling x2.
        """
        name_batch, keypoint_batch, src_length_batch = [], [], []
        tgt_batch, text_batch = [], []

        # --- 1. Selección de frames por muestra ---
        for name_sample, keypoint_sample, tgt_sample, text, length in batch:
            index, valid_len = self.get_selected_index(length)
            keypoint_batch.append(torch.stack([keypoint_sample[:, i, :] for i in index], dim=1))
            src_length_batch.append(valid_len)
            name_batch.append(name_sample)
            tgt_batch.append(tgt_sample)
            text_batch.append(text)

        # --- 2. Padding temporal: rellenar hasta la longitud máxima del batch ---
        max_length = max(src_length_batch)
        padded_keypoints = []
        for keypoints, len_ in zip(keypoint_batch, src_length_batch):
            if len_ < max_length:
                last_frame = keypoints[:, -1, :].unsqueeze(1)
                padding = torch.tile(last_frame, [1, max_length - len_, 1])
                keypoints = torch.cat([keypoints, padding], dim=1)
            padded_keypoints.append(keypoints)

        # --- 3. Normalización y augmentación ---
        keypoints = torch.stack(padded_keypoints, dim=0)
        keypoints = self.augment_preprocess_inputs(self.phase, keypoints)

        # --- 4. Máscaras de atención para el recognition ---
        src_length_batch = torch.tensor(src_length_batch)

        # El encoder reduce la longitud temporal en dos pasos de /2 → /4 total
        enc_lengths = (((src_length_batch - 1) / 2) + 1).long()
        enc_lengths = (((enc_lengths - 1) / 2) + 1).long()

        max_enc_len = max(enc_lengths)
        mask = torch.zeros(len(enc_lengths), 1, max_enc_len, dtype=torch.bool)
        for i, l in enumerate(enc_lengths):
            mask[i, :, :l] = True

        # --- 5. Tokenización ---
        gloss_input = self.tokenizer(tgt_batch)

        src_input = {
            'name':            name_batch,
            'keypoint':        keypoints,
            'gloss':           tgt_batch,
            'mask':            mask,
            'new_src_lengths': enc_lengths,
            'gloss_input':     gloss_input,
            'src_length':      src_length_batch,
        }

        if self.config['task'] == 'S2T':
            # Tokenizar el texto con Qwen (no MBart)
            t = self._tokenize_text_for_qwen(text_batch)
            src_input['translation_inputs'] = {
                **t,
                # Las glosas ground truth se pasan para el ablation study
                # (use_gloss_tokens: true + gloss_source: ground_truth)
                'gloss_ids':     gloss_input['gloss_labels'],
                'gloss_lengths': gloss_input['gls_lengths'],
            }
            src_input['text'] = text_batch

        return src_input

    # ------------------------------------------------------------------ #
    #  Utilidades                                                          #
    # ------------------------------------------------------------------ #

    def pil_list_to_tensor(self, pil_list, int2float=True):
        """Convierte una lista de imágenes PIL a un tensor (T, C, H, W)."""
        func = torchvision.transforms.PILToTensor()
        tensors = torch.stack([func(img) for img in pil_list], dim=0)
        if int2float:
            tensors = tensors / 255
        return tensors

    def get_seq_frames(self, num_frames):
        """
        Muestrea clip_len frames uniformemente distribuidos a lo largo del vídeo.
        En entrenamiento elige aleatoriamente dentro de cada segmento.
        En validación/test usa el centro de cada segmento.
        """
        seg_size = float(num_frames - 1) / self.clip_len
        seq = []
        for i in range(self.clip_len):
            start = int(np.round(seg_size * i))
            end = int(np.round(seg_size * (i + 1)))
            if self.phase == 'train':
                seq.append(random.randint(start, end))
            else:
                seq.append((start + end) // 2)
        return np.array(seq)

    def apply_spatial_ops(self, x, spatial_ops_func):
        """
        Aplica spatial_ops_func sobre (B, T, C, H, W) en chunks de 16 frames.
        """
        B, T, C_, H, W = x.shape
        x = x.view(-1, C_, H, W)
        chunks = torch.split(x, 16, dim=0)
        transformed = torch.cat([spatial_ops_func(chunk) for chunk in chunks], dim=0)
        _, C_, H_o, W_o = transformed.shape
        return transformed.view(B, T, C_, H_o, W_o)

    def __str__(self):
        return f'#total {self.phase} set: {len(self.list)}.'
