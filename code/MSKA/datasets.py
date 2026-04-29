import random

import torchvision
import torch
import utils as utils
import torch.utils.data.dataset as Dataset
from PIL import Image
import os
from Tokenizer import GlossTokenizer_S2G, TextTokenizer
import numpy as np


CONFIG_SLR = {}
CONFIG_SLT = {}

class S2T_Dataset(Dataset.Dataset):
    """
    Dataset para reconocimiento/traducción de lenguaje de señas.
    Soporta dos tareas:
      - S2G (Sign-to-Gloss): reconocimiento de señas a glosas.
      - S2T (Sign-to-Text): traducción de señas a texto natural.

    Los datos de entrada son keypoints corporales (coordenadas de articulaciones)
    extraídos de vídeos de lenguaje de señas.
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

        # Dimensiones de los frames
        self.w, self.h = 210, 260

        self.raw_data = utils.load_dataset_file(path)
        self.tokenizer = tokenizer
        self.max_length = config['data']['max_length']
        self.list = [key for key in self.raw_data]

        # El tokenizador de texto solo se necesita en la tarea de traducción (S2T)
        if self.config['task'] == 'S2T':
            self.text_tokenizer = TextTokenizer(
                self.config['model']['TranslationNetwork']['TextTokenizer']
            )

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

        # Keypoints con shape original (articulaciones, tiempo, canales) → (canales, tiempo, articulaciones)
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
                # Recorte centrado: descartar an frames al inicio y en frames al final
                an = (vlen - self.clip_len) // 2 # frames a descartar al inicio
                en = vlen - self.clip_len - an # frames a descartar al final
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

        # Redondear al múltiplo de 4 superior
        remainder = selected_len % 4
        if remainder != 0:
            selected_len += (4 - remainder)

        if selected_len <= vlen:
            # Submuestreo: elegir selected_len frames sin reemplazo
            selected_index = sorted(np.random.permutation(np.arange(vlen))[:selected_len])
        else:
            # Sobremuestreo: copiar frames aleatorios para alcanzar selected_len
            extra = np.random.randint(0, vlen, selected_len - vlen)
            selected_index = sorted(np.concatenate([np.arange(vlen), extra]))

        frame_index = selected_index
        valid_len = selected_len

        return frame_index, valid_len

    # ------------------------------------------------------------------ #
    #  Augmentaciones espaciales sobre keypoints                           #
    # ------------------------------------------------------------------ #

    def rotate_points(self, points, angle):
        """Rota los puntos `angle` radianes alrededor del origen (0, 0)."""
        rotation_matrix = np.array([
            [ np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)]
        ])
        return np.dot(points, rotation_matrix.T)

    def translate_points(self, points, translation):
        """Traslada los puntos sumando el vector `translation`."""
        return points + translation

    def scale_points(self, points, scale_factor):
        """Escala los puntos multiplicando por `scale_factor`."""
        return points * scale_factor

    def random_move(self, data_numpy):
        """
        Aplica una rotación aleatoria a los keypoints con probabilidad 0.5.
        El ángulo se elige uniformemente en [-15°, 15°].

        Augmentaciones de traslación y escalado están implementadas pero
        desactivadas.

        Recibe y devuelve un tensor con shape (T, articulaciones, canales_xy).
        """
        angle = np.radians(np.random.uniform(-15, 15))
        if np.random.uniform(0, 1) >= 0.5:
            data_numpy = self.rotate_points(data_numpy, angle)

        # TODO: activar traslación y escalado cuando se complete el augmentado
        # dx = np.random.uniform(-0.21, 0.21)
        # dy = np.random.uniform(-0.26, 0.26)
        # data_numpy = self.translate_points(data_numpy, [dx, dy])
        # scale = np.random.uniform(0.8, 1.2)
        # if np.random.uniform(0, 1) >= 0.5:
        #     data_numpy = self.scale_points(data_numpy, scale)

        return torch.from_numpy(data_numpy)

    def augment_preprocess_inputs(self, is_train, keypoints=None):
        """
        Normaliza los keypoints al rango [-1, 1]:
          1. Divide X por el ancho del frame.
          2. Invierte Y (en imagen Y crece hacia abajo; lo convertimos a convención matemática).
          3. Divide Y por el alto del frame.
          4. Centra ambos ejes: (valor - 0.5) / 0.5  →  [-1, 1].

        En entrenamiento además aplica random_move (rotación aleatoria).

        TODO: el bloque de entrenamiento y el de validación son idénticos salvo
        por random_move; pendiente añadir más augmentaciones espaciales.
        """
        # Normalización X
        keypoints[:, 0, :, :] /= self.w
        # Invertir Y y normalizar
        keypoints[:, 1, :, :] = self.h - keypoints[:, 1, :, :]
        keypoints[:, 1, :, :] /= self.h
        # Centrar ambos ejes en [-1, 1]
        keypoints[:, :2, :, :] = (keypoints[:, :2, :, :] - 0.5) / 0.5

        if is_train == 'train':
            # Augmentación espacial: rotación aleatoria de los keypoints
            keypoints[:, :2, :, :] = self.random_move(
                keypoints[:, :2, :, :].permute(0, 2, 3, 1).numpy()
            ).permute(0, 3, 1, 2)

        return keypoints

    # ------------------------------------------------------------------ #
    #  Construcción del batch                                              #
    # ------------------------------------------------------------------ #

    def collate_fn(self, batch):
        """
        Función llamada por el DataLoader para unir muestras en un batch.

        Pasos:
          1. Para cada muestra, seleccionar los frames según get_selected_index.
          2. Hacer padding en la dimensión temporal para igualar longitudes.
          3. Normalizar y augmentar los keypoints.
          4. Construir las máscaras de atención para el transformer
             (1 = posición real, 0 = padding).
          5. Tokenizar glosas y, si la tarea es S2T, también el texto.

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
                # Repetir el último frame para completar
                last_frame = keypoints[:, -1, :].unsqueeze(1)
                padding = torch.tile(last_frame, [1, max_length - len_, 1])
                keypoints = torch.cat([keypoints, padding], dim=1)
            padded_keypoints.append(keypoints)

        # --- 3. Normalización y augmentación ---
        keypoints = torch.stack(padded_keypoints, dim=0)
        keypoints = self.augment_preprocess_inputs(self.phase, keypoints)

        # --- 4. Máscaras de atención ---
        src_length_batch = torch.tensor(src_length_batch)

        # El encoder reduce la longitud temporal en dos pasos de /2 → longitud / 4 total
        enc_lengths = (((src_length_batch - 1) / 2) + 1).long()
        enc_lengths = (((enc_lengths - 1) / 2) + 1).long()

        max_enc_len = max(enc_lengths)
        # mask shape: (batch, 1, max_enc_len) — True donde hay datos reales
        mask = torch.zeros(len(enc_lengths), 1, max_enc_len, dtype=torch.bool)
        for i, l in enumerate(enc_lengths):
            mask[i, :, :l] = True

        # --- 5. Tokenización ---
        gloss_input = self.tokenizer(tgt_batch)

        # --- Ensamblado del diccionario de entrada al modelo ---
        batch = {
            'name':            name_batch,
            'keypoint':        keypoints,
            'gloss':           tgt_batch,
            'mask':            mask,
            'new_src_lengths': enc_lengths,
            'gloss_input':     gloss_input,
            'src_length':      src_length_batch,
        }

        if self.config['task'] == 'S2T':
            t = self.text_tokenizer(text_batch)
            batch['translation_inputs'] = {
                **t,  # input_ids, attention_mask, labels, etc. del tokenizador de texto
                'gloss_ids':     gloss_input['gloss_labels'],
                'gloss_lengths': gloss_input['gls_lengths'],
            }
            batch['text'] = text_batch

        return batch

    # ------------------------------------------------------------------ #
    #  Utilidades heredadas  (no utilizadas aquí pero repo lo tiene)     #
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
        (Adaptado de SlowFast/SSv2.)
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
        Aplica `spatial_ops_func` sobre un tensor (B, T, C, H, W) procesando
        en chunks de 16 frames para no saturar memoria.
        """
        B, T, C_, H, W = x.shape
        x = x.view(-1, C_, H, W)
        chunks = torch.split(x, 16, dim=0)
        transformed = torch.cat([spatial_ops_func(chunk) for chunk in chunks], dim=0)
        _, C_, H_o, W_o = transformed.shape
        return transformed.view(B, T, C_, H_o, W_o)

    def __str__(self):
        return f'#total {self.phase} set: {len(self.list)}.'



##------------------------##
## collate_fn paso a paso ##
##------------------------##
#
# Paso 1 — Selección de frames
# for name_sample, keypoint_sample, tgt_sample, text, length in batch:
#     index, valid_len = self.get_selected_index(length)
#     keypoint_batch.append(torch.stack([keypoint_sample[:, i, :] for i in index], dim=1))
# Cada muestra llega con todos sus frames. Aquí se decide cuáles usar, aplicando la augmentación temporal en entrenamiento.
# Tras este paso todas las muestras siguen teniendo longitudes distintas, pero ya dentro de [0, clip_len].
#
# Paso 2 — Padding
# max_length = max(src_length_batch)
# for keypoints, len_ in zip(keypoint_batch, src_length_batch):
#     if len_ < max_length:
#         last_frame = keypoints[:, -1, :].unsqueeze(1)
#         padding = torch.tile(last_frame, [1, max_length - len_, 1])
#         keypoints = torch.cat([keypoints, padding], dim=1)
# Se igualan todas las longitudes repitiendo el último frame real hasta alcanzar max_length. Se usa el último frame en lugar de ceros
# para que la normalización posterior no distorsione los valores de padding.
# muestra_A:  [f0, f1, ..., f79, f79, f79, ..., f79]   # 40 frames de padding
# muestra_B:  [f0, f1, ..., f119]                       # sin padding
# muestra_C:  [f0, f1, ..., f94, f94, f94, ..., f94]   # 25 frames de padding
#
# # Paso 3 — Normalización y augmentación
# keypoints = torch.stack(padded_keypoints, dim=0)   # (B, C, T, J)
# keypoints = self.augment_preprocess_inputs(self.phase, keypoints)
# Ahora que todos tienen la misma forma se pueden apilar en un tensor de batch y normalizar de golpe.
#
# Paso 4 — Máscaras de atención
# enc_lengths = (((src_length_batch - 1) / 2) + 1).long()
# enc_lengths = (((enc_lengths - 1) / 2) + 1).long()
#
# mask = torch.zeros(len(enc_lengths), 1, max_enc_len, dtype=torch.bool)
# for i, l in enumerate(enc_lengths):
#     mask[i, :, :l] = True
#
# El transformer no puede distinguir por sí solo qué posiciones son reales y cuáles son padding. La máscara se lo indica explícitamente:
# True donde hay datos reales, False donde hay padding. Se calcula sobre las longitudes después del downsampling (÷4) porque es ahí donde el transformer opera.
# muestra_A  enc_length=20:  [T, T, T, ..., T, F, F, ..., F]
# muestra_B  enc_length=30:  [T, T, T, ..., T, T, T, ..., T]
# muestra_C  enc_length=24:  [T, T, T, ..., T, F, F, ..., F]
#
# Paso 5 — Tokenización
# gloss_input = self.tokenizer(tgt_batch)
#
# if self.config['task'] == 'S2T':
#     t = self.text_tokenizer(text_batch)
#     src_input['translation_inputs'] = {
#         **t,
#         'gloss_ids':     gloss_input['gloss_labels'],
#         'gloss_lengths': gloss_input['gls_lengths'],
#     }
# Las glosas y el texto se convierten a secuencias de IDs numéricos que el modelo puede procesar. En S2T
# se combinan los tokens del texto con los de las glosas en un solo diccionario porque el módulo de traducción necesita ambos.

