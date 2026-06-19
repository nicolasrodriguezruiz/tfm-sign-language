import torch
import torch.nn as nn
import random
import torchaudio # decodeV2 (Full PyT)
from torchaudio.models.decoder import ctc_decoder

# Decode V1 (TF)
import numpy as np
# import tensorflow as tf
from itertools import groupby

from Recognition.Tokenizer import GlossTokenizer_S2G
from Recognition.VisualHead import VisualHead
import math


# ---------------------------------------------------------------------------
# Encoding posicional para grafos espacio-temporales
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """
    Encoding posicional para datos con estructura de grafo (N, C, T, V):
      N = batch, C = canales, T = frames temporales, V = nodos (articulaciones).

    A diferencia del PE estándar (solo temporal), este puede codificar:
      - 'temporal': qué frame es cada posición (mismo valor para todas las articulaciones del frame).
      - 'spatial':  qué articulación es cada nodo (mismo valor para todos los frames).

    Usa la misma fórmula sinusoidal del paper "Attention is All You Need":
      PE(pos, 2i)   = sin(pos / 10000^(2i/C))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/C))
    """

    def __init__(self, channel, joint_num, time_len, domain):
        super(PositionalEncoding, self).__init__()
        self.joint_num = joint_num
        self.time_len = time_len
        self.domain = domain

        # Construir la lista de posiciones según el dominio.
        # Para cada combinación (frame t, articulación j), se asigna una posición:
        # - temporal: la posición es t (todos los nodos del mismo frame comparten posición)
        # - spatial:  la posición es j (todos los frames del mismo nodo comparten posición)
        pos_list = []
        for t in range(self.time_len):
            for j_id in range(self.joint_num):
                if domain == "temporal":
                    pos_list.append(t)
                elif domain == "spatial":
                    pos_list.append(j_id)

        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()  # (T*V, 1)

        # Calcular los vectores sinusoidales para cada posición y canal
        pe = torch.zeros(self.time_len * self.joint_num, channel)  # (T*V, C)
        div_term = torch.exp(
            torch.arange(0, channel, 2).float() * -(math.log(10000.0) / channel)
        )  # factores de escala, shape (C/2,)
        pe[:, 0::2] = torch.sin(position * div_term)  # canales pares: seno
        pe[:, 1::2] = torch.cos(position * div_term)  # canales impares: coseno

        # Reorganizar a (1, C, T, V) para que se pueda sumar directamente a x de shape (N, C, T, V)
        pe = pe.view(time_len, joint_num, channel).permute(2, 0, 1).unsqueeze(0)

        # register_buffer: pe se guarda con el modelo pero NO es un parámetro entrenable
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (N, C, T, V)
        # Se recorta pe a [:, :, :x.size(2)] por si x tiene menos frames que time_len
        x = x + self.pe[:, :, :x.size(2)]
        return x


# ---------------------------------------------------------------------------
# Bloque de atención espacio-temporal
# ---------------------------------------------------------------------------

class STAttentionBlock(nn.Module):
    """
    Bloque que aplica atención espacial (entre articulaciones) y temporal (entre frames).

    La atención espacial modela relaciones entre articulaciones: p.ej. la muñeca
    influye sobre los dedos. Combina una matriz global aprendida (igual para todas
    las muestras) con una atención dinámica calculada desde los datos.

    La atención temporal captura dependencias entre frames consecutivos mediante
    una convolución 1D con kernel t_kernel.

    Ambos bloques tienen conexiones residuales para facilitar el flujo de gradientes.
    """

    def __init__(self, in_channels, out_channels, inter_channels, num_subset=2,
                 num_node=27, num_frame=400, kernel_size=1, stride=1, t_kernel=3,
                 glo_reg_s=True, att_s=True, glo_reg_t=False, att_t=False,
                 use_temporal_att=False, use_spatial_att=True, attentiondrop=0.,
                 use_pes=True, use_pet=False):
        super(STAttentionBlock, self).__init__()
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_subset = num_subset    # número de "cabezas" de atención espacial
        self.glo_reg_s = glo_reg_s      # si True, añade matriz de atención global aprendida
        self.att_s = att_s              # si True, añade atención dinámica por muestra
        self.glo_reg_t = glo_reg_t
        self.att_t = att_t
        self.use_pes = use_pes          # si True, aplica encoding posicional espacial
        self.use_pet = use_pet

        pad = int((kernel_size - 1) / 2)
        self.use_spatial_att = use_spatial_att

        if use_spatial_att:
            # Matriz de atención base: se inicializa a cero y se actualiza con las
            # contribuciones de glo_reg_s y att_s durante el forward.
            # Es un buffer (no parámetro) porque se recalcula en cada forward.
            atts = torch.zeros((1, num_subset, num_node, num_node))
            self.register_buffer('atts', atts)

            # Encoding posicional espacial: indica qué articulación es cada nodo
            self.pes = PositionalEncoding(in_channels, num_node, num_frame, 'spatial')

            # Proyección de salida tras aplicar la atención: consolida los num_subset
            # subgrafos atendidos en un único tensor de out_channels canales
            self.ff_nets = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1, 1, padding=0, bias=True),
                nn.BatchNorm2d(out_channels),
            )

            if att_s:
                # Proyección para calcular queries y keys de la atención dinámica.
                # Produce 2 * num_subset * inter_channels canales: la mitad para Q y la mitad para K.
                self.in_nets = nn.Conv2d(in_channels, 2 * num_subset * inter_channels, 1, bias=True)
                # alpha: escala aprendida que controla cuánto peso tiene la atención dinámica
                # frente a la global. Inicializado a 1 (igual peso inicial).
                self.alphas = nn.Parameter(torch.zeros(1, num_subset, 1, 1), requires_grad=True) # LO PONGO A 0 COMO SUELEN SUGERRIR

            if glo_reg_s:
                # Matriz de atención global aprendida: igual para todas las muestras del batch.
                # Captura relaciones estructurales fijas del cuerpo (p.ej. muñeca→dedos).
                # Inicializada a 1/num_node (distribución uniforme).
                self.attention0s = nn.Parameter(
                    torch.ones(1, num_subset, num_node, num_node) / num_node,
                    requires_grad=True,
                )

            # Proyección final: combina los num_subset grafos atendidos → out_channels
            self.out_nets = nn.Sequential(
                nn.Conv2d(in_channels * num_subset, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else:
            # Sin atención espacial: solo una conv simple entre nodos adyacentes
            self.out_nets = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1), bias=True, stride=1),
                nn.BatchNorm2d(out_channels),
            )

        # Convolución temporal: captura dependencias entre frames consecutivos.
        # Kernel (t_kernel, 1): se mueve solo en la dimensión temporal, no entre nodos.
        # stride controla el downsampling temporal (stride=2 → T/2 frames de salida).
        padd = int(t_kernel / 2)
        self.out_nett = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (t_kernel, 1),
                      padding=(padd, 0), bias=True, stride=(stride, 1)),
            nn.BatchNorm2d(out_channels),
        )

        # Conexiones residuales: si las dimensiones de entrada y salida difieren
        # (in_channels != out_channels o stride != 1), se necesita proyectar la
        # identidad antes de sumarla al resultado.
        if in_channels != out_channels or stride != 1:
            if use_spatial_att:
                self.downs1 = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=True),
                    nn.BatchNorm2d(out_channels),
                )
            self.downs2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            if use_temporal_att:
                self.downt1 = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 1, 1, bias=True),
                    nn.BatchNorm2d(out_channels),
                )
            self.downt2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, (kernel_size, 1),
                          (stride, 1), padding=(pad, 0), bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else:
            # Dimensiones iguales: la identidad no necesita proyección
            if use_spatial_att:
                self.downs1 = lambda x: x
            self.downs2 = lambda x: x
            self.downt2 = lambda x: x

        self.tan = nn.Tanh()
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(attentiondrop)

    def forward(self, x):
        """
        Args:
            x: shape (N, C, T, V) — batch, canales, frames, articulaciones
        Returns:
            z: shape (N, out_channels, T', V) — T' puede ser menor que T si stride > 1
        """
        N, C, T, V = x.size()

        if self.use_spatial_att:
            # --- Atención espacial ---

            # Aplicar encoding posicional espacial (opcional)
            y = self.pes(x) if self.use_pes else x

            # Calcular matriz de atención como suma de componentes:
            attention = self.atts  # base: ceros

            if self.att_s:
                # Atención dinámica: diferente para cada muestra del batch.
                # 1. Proyectar x a queries y keys
                q, k = torch.chunk(
                    self.in_nets(y).view(N, 2 * self.num_subset, self.inter_channels, T, V),
                    2, dim=1,
                )  # q, k: (N, num_subset, inter_channels, T, V)

                # 2. Producto escalar Q·K^T agregado sobre T y C → (N, num_subset, V, V)
                # einsum 'nsctu,nsctv->nsuv': para cada par (u,v) de articulaciones,
                # suma el producto escalar de sus representaciones sobre todos los frames y canales.
                # Se divide por (inter_channels * T) para estabilizar la magnitud.
                attention = attention + self.tan(
                    torch.einsum('nsctu,nsctv->nsuv', [q, k]) / (self.inter_channels * T)
                ) * self.alphas
                # Tanh limita los valores de atención a [-1, 1], evitando explosión de gradientes.
                # alpha escala la contribución de la atención dinámica.

            if self.glo_reg_s:
                # Añadir la matriz global aprendida (replicada para cada muestra del batch)
                attention = attention + self.attention0s.repeat(N, 1, 1, 1)

            attention = self.drop(attention)  # dropout sobre la atención

            # Aplicar la atención: para cada cabeza s, multiplicar x por la matriz attention[s]
            # einsum 'nctu,nsuv->nsctv': pondera la contribución de cada articulación u
            # sobre cada articulación v según los pesos de atención.
            y = torch.einsum('nctu,nsuv->nsctv', [x, attention]).contiguous() \
                .view(N, self.num_subset * self.in_channels, T, V)
            # Proyectar los num_subset grafos atendidos → out_channels
            y = self.out_nets(y)

            # Conexiones residuales espaciales
            y = self.relu(self.downs1(x) + y)  # residual 1
            y = self.ff_nets(y)
            y = self.relu(self.downs2(x) + y)  # residual 2

        else:
            # Sin atención: solo conv entre nodos vecinos + residual
            y = self.out_nets(x)
            y = self.relu(self.downs2(x) + y)

        # --- Atención temporal ---
        # Convolución 1D sobre la dimensión temporal, con posible downsampling (stride)
        z = self.out_nett(y)
        z = self.relu(self.downt2(y) + z)  # residual temporal

        return z


# ---------------------------------------------------------------------------
# Backbone DSTA: procesa keypoints dividiendo el cuerpo en 4 partes
# ---------------------------------------------------------------------------

class DSTA(nn.Module):
    """
    Dual-Stream Temporal Attention backbone.

    Divide el cuerpo en 4 partes y las procesa con ramas independientes de
    STAttentionBlock. Esto permite que el modelo aprenda representaciones
    especializadas para cada parte del cuerpo.

    Partes:
        left:  mano y muñeca izquierda
        right: mano y muñeca derecha
        face:  puntos faciales
        body:  torso y hombros

    Al final agrega los nodos (mean sobre V) y concatena las representaciones:
        output      = [left, face, right, body]  → va al fuse_head
        left_output = [left, face]               → va al left_head
        right_output= [right, face]              → va al right_head
        body        = [body]                     → va al body_head

    La cara se incluye junto a cada mano porque los gestos faciales son
    contexto relevante para interpretar el significado de cada mano.
    """

    def __init__(self, num_frame=400, num_subset=6, dropout=0.1, cfg=None, args=None,
                 num_channel=2, glo_reg_s=True, att_s=True, glo_reg_t=False, att_t=False,
                 use_temporal_att=False, use_spatial_att=True, attentiondrop=0.1,
                 use_pet=False, use_pes=True, mode='SLR'):
        super(DSTA, self).__init__()
        self.mode = mode
        self.cfg = cfg
        self.args = args

        # config es una lista de tuplas (in_ch, out_ch, inter_ch, t_kernel, stride)
        # que define la arquitectura de cada rama de STAttentionBlock
        config = self.cfg['net']
        self.out_channels = config[-1][1]  # canales de salida de la última capa
        in_channels = config[0][0]         # canales de entrada de la primera capa
        self.num_frame = num_frame

        # Parámetros compartidos por todos los bloques de atención
        param = {
            'num_subset': num_subset,
            'glo_reg_s': glo_reg_s,
            'att_s': att_s,
            'glo_reg_t': glo_reg_t,
            'att_t': att_t,
            'use_spatial_att': use_spatial_att,
            'use_temporal_att': use_temporal_att,
            'use_pet': use_pet,
            'use_pes': use_pes,
            'attentiondrop': attentiondrop,
        }

        # Proyecciones de entrada: mapean los canales raw (num_channel=3: x, y, confianza)
        # al espacio de in_channels del backbone. Una por cada parte del cuerpo.
        # Conv2d(num_channel, in_channels, 1) = convolución puntual (kernel 1x1)
        self.left_input_map  = nn.Sequential(nn.Conv2d(num_channel, in_channels, 1), nn.BatchNorm2d(in_channels), nn.LeakyReLU(0.1))
        self.right_input_map = nn.Sequential(nn.Conv2d(num_channel, in_channels, 1), nn.BatchNorm2d(in_channels), nn.LeakyReLU(0.1))
        self.body_input_map  = nn.Sequential(nn.Conv2d(num_channel, in_channels, 1), nn.BatchNorm2d(in_channels), nn.LeakyReLU(0.1))
        self.face_input_map  = nn.Sequential(nn.Conv2d(num_channel, in_channels, 1), nn.BatchNorm2d(in_channels), nn.LeakyReLU(0.1))

        # Construir las 4 ramas de STAttentionBlock.
        # Cada rama tiene el mismo número de capas y arquitectura, pero opera sobre
        # un subconjunto distinto de articulaciones (num_node diferente).
        # num_frame se actualiza tras cada capa porque stride puede reducir T.
        for part, key in [('face', 'face'), ('left', 'left'), ('right', 'right'), ('body', 'body')]:
            num_frame = self.num_frame
            layers = nn.ModuleList()
            for in_ch, out_ch, inter_ch, t_kernel, stride in config:
                layers.append(STAttentionBlock(
                    in_ch, out_ch, inter_ch,
                    stride=stride, t_kernel=t_kernel,
                    num_node=len(self.cfg[key]),
                    num_frame=num_frame,
                    **param,
                ))
                num_frame = int(num_frame / stride + 0.5)
            setattr(self, f'{part}_graph_layers', layers)

        self.drop_out = nn.Dropout(dropout)

    def forward(self, src_input):
        """
        Args:
            src_input: diccionario del batch con clave 'keypoint',
                       shape (N, C, T, V) — batch, canales, frames, articulaciones.

        Returns:
            output:       (N, T', C*4) features de todas las partes concatenadas
            left_output:  (N, T', C*2) features de mano izquierda + cara
            right_output: (N, T', C*2) features de mano derecha + cara
            body:         (N, T', C)   features solo del cuerpo
        """
        x = src_input['keypoint'].cuda()  # (N, C, T, V)
        N, C, T, V = x.shape

        # Seleccionar los nodos de cada parte usando los índices del config
        left  = self.left_input_map(x[:, :, :, self.cfg['left']])   # (N, in_ch, T, V_left)
        right = self.right_input_map(x[:, :, :, self.cfg['right']])
        face  = self.face_input_map(x[:, :, :, self.cfg['face']])
        body  = self.body_input_map(x[:, :, :, self.cfg['body']])

        # Pasar cada parte por su rama de STAttentionBlock
        for m in self.face_graph_layers:
            face = m(face)
        for m in self.left_graph_layers:
            left = m(left)
        for m in self.right_graph_layers:
            right = m(right)
        for m in self.body_graph_layers:
            body = m(body)
        # Tras los bloques: (N, out_channels, T', V_part)

        # Reordenar a (N, T', out_channels, V_part) para hacer mean sobre V
        left  = left.permute(0, 2, 1, 3).contiguous()
        right = right.permute(0, 2, 1, 3).contiguous()
        face  = face.permute(0, 2, 1, 3).contiguous()
        body  = body.permute(0, 2, 1, 3).contiguous()

        # Agregar sobre los nodos (articulaciones): mean sobre V → (N, T', out_channels)
        # Esto colapsa la estructura de grafo en un único vector por frame
        body  = body.mean(3)
        face  = face.mean(3)
        left  = left.mean(3)
        right = right.mean(3)

        # # Concatenar representaciones en la dimensión de canales
        # output       = torch.cat([left, face, right, body], dim=-1)  # todas las partes
        # left_output  = torch.cat([left, face], dim=-1)               # mano izq + cara
        # right_output = torch.cat([right, face], dim=-1)              # mano der + cara

        output       = self.drop_out(torch.cat([left, face, right, body], dim=-1)) # AÑADO DROPOUT
        left_output  = self.drop_out(torch.cat([left, face], dim=-1))
        right_output = self.drop_out(torch.cat([right, face], dim=-1))
        body         = self.drop_out(body)

        return output, left_output, right_output, body


# ---------------------------------------------------------------------------
# Recognition: orquesta DSTA + 4 VisualHeads + losses CTC
# ---------------------------------------------------------------------------

class Recognition(nn.Module):
    """
    Módulo completo de reconocimiento de glosas.

    Pipeline:
        keypoints → DSTA → 4 ramas paralelas de VisualHead → ensemble → CTC loss

    Las 4 cabezas (fuse, body, left, right) se entrenan simultáneamente con
    losses CTC independientes. El ensemble promedia sus probabilidades para
    obtener una predicción más robusta.

    Opcionalmente aplica Knowledge Distillation: el ensemble actúa como
    profesor y cada cabeza individual como alumno (KL divergence loss).
    """

    def __init__(self, cfg, args, input_streams=None):
        super(Recognition, self).__init__()
        self.cfg = cfg
        self.args = args
        self.input_type = cfg['input_type']  # solo 'keypoint' está soportado
        self.gloss_tokenizer = GlossTokenizer_S2G(cfg['GlossTokenizer'])
        self.input_streams = input_streams
        self.fuse_method = cfg.get('fuse_method', 'empty')
        self.heatmap_cfg = cfg.get('heatmap_cfg', {})


        # 'weighted' usa pesos aprendidos, 'uniform' usa el promedio simple original
        self.ensemble_method = cfg.get('ensemble_method', 'uniform')
        if self.ensemble_method == 'weighted':
            self.stream_weights = nn.Parameter(torch.ones(4) / 4)

        if self.input_type == 'keypoint':
            # Backbone DSTA: extrae features espacio-temporales de los keypoints
            self.visual_backbone_keypoint = DSTA(
                cfg=self.cfg['DSTA-Net'],
                num_channel=3,  # x, y, confianza del keypoint
                args=args,
            )
            # 4 VisualHeads en paralelo, uno por cada vista del cuerpo.
            # Cada uno clasifica de forma independiente en el vocabulario de glosas.
            self.fuse_visual_head  = VisualHead(cls_num=len(self.gloss_tokenizer), **cfg['fuse_visual_head'])
            self.body_visual_head  = VisualHead(cls_num=len(self.gloss_tokenizer), **cfg['body_visual_head'])
            self.left_visual_head  = VisualHead(cls_num=len(self.gloss_tokenizer), **cfg['left_visual_head'])
            self.right_visual_head = VisualHead(cls_num=len(self.gloss_tokenizer), **cfg['right_visual_head'])

            # Otros tipos de entrada no están implementados
            self.visual_backbone = None
            self.rgb_visual_head = None
        else:
            raise ValueError(f"input_type '{self.input_type}' no soportado. Usar 'keypoint'.")

        # Cargar pesos preentrenados si se especifica en el config
        if 'pretrained_path' in self.cfg:
            load_dict = torch.load(cfg['pretrained_path'], map_location='cpu')['model']
            # Eliminar el prefijo 'recognition_network.' de las claves del checkpoint
            backbone_dict = {k.replace('recognition_network.', ''): v for k, v in load_dict.items()}
            self.load_state_dict(backbone_dict)

        # CTC Loss: mide la diferencia entre las secuencias de glosas predichas y reales.
        # blank=0: el token de silencio está en el índice 0 (como impone GlossTokenizer_S2G).
        # zero_infinity=True: ignora secuencias donde la loss sería infinita
        #   (ocurre cuando la secuencia objetivo es más larga que la entrada,
        #    lo que matemáticamente CTC no puede resolver).
        # reduction='sum': suma las losses de todas las muestras del batch
        #   (se divide manualmente por batch_size en compute_recognition_loss).
        self.recognition_loss_func = torch.nn.CTCLoss(
            blank=0,
            zero_infinity=True,
            reduction='sum',
        )

    def compute_recognition_loss(self, gloss_labels, gloss_lengths,
                                  gloss_probabilities_log, input_lengths):
        """
        Calcula la CTC loss para una cabeza visual.

        CTC loss mide cuánto cuesta "alinear" la secuencia predicha con la referencia,
        sumando sobre todas las alineaciones válidas. No requiere alineación manual
        entre frames y glosas, lo que la hace ideal para señas donde no sabemos
        exactamente en qué frame empieza/termina cada glosa.

        Args:
            gloss_labels:            IDs de glosas de referencia, shape (B, max_gls_len).
            gloss_lengths:           longitudes reales de cada secuencia de glosas, shape (B,).
            gloss_probabilities_log: log-probabilidades del modelo, shape (B, T, V).
            input_lengths:           longitudes reales de cada secuencia temporal, shape (B,).
        """
        loss = self.recognition_loss_func(
            log_probs=gloss_probabilities_log.permute(1, 0, 2),  # CTC espera (T, N, C)
            targets=gloss_labels,
            input_lengths=input_lengths,
            target_lengths=gloss_lengths,
        )
        # Normalizar por batch_size para que la loss no dependa del tamaño del batch
        loss = loss / gloss_probabilities_log.shape[0]
        return loss


    def decode(self, gloss_logits, beam_size, input_lengths):
        """
        Decodifica logits CTC a secuencias de glosas usando beam search de torchaudio.
        Ya no necesita reordenar el vocabulario ni salir a TensorFlow.
        """
        # torchaudio espera log-probabilidades en formato (T, B, V) .permute(1, 0, 2)
        log_probs = gloss_logits.log_softmax(2).cpu()

        # beam search: blank=0 porque el token de silencio está en el índice 0
        decoder = ctc_decoder( #torchaudio.models.decoder.
            lexicon=None,
            tokens=list(self.gloss_tokenizer.id2gloss.values()),
            blank_token=self.gloss_tokenizer.id2gloss[0],
            sil_token=self.gloss_tokenizer.id2gloss[0], # token de silencio
            beam_size=beam_size,
        )
        hypotheses = decoder(log_probs, input_lengths.cpu())

        # Extraer la mejor hipótesis de cada muestra del batch
        decoded_gloss_sequences = []
        for hyp in hypotheses:
            decoded_gloss_sequences.append([t for t in hyp[0].tokens.tolist()])

        return decoded_gloss_sequences



    def forward(self, src_input):
        """
        Forward pass completo del recognition network.

        Pasos:
          1. DSTA extrae features de las 4 partes del cuerpo.
          2. Cada VisualHead clasifica su vista independientemente.
          3. Se calcula un ensemble promediando las 4 probabilidades.
          4. Se calculan 4 losses CTC independientes (una por cabeza).
          5. Opcionalmente se añade Knowledge Distillation (ensemble→cabezas).

        Returns:
            Diccionario con logits, probabilidades, features y losses.
        """
        if self.input_type != 'keypoint':
            raise ValueError

        # --- 1. Backbone: extraer features por parte del cuerpo ---
        fuse, left_output, right_output, body = self.visual_backbone_keypoint(src_input)

        # --- 2. Clasificación con cada VisualHead ---
        # Todos comparten mask y longitudes del batch
        mask = src_input['mask'].cuda()
        lengths = src_input['new_src_lengths'].cuda()

        body_head  = self.body_visual_head( x=body,         mask=mask, valid_len_in=lengths)
        fuse_head  = self.fuse_visual_head( x=fuse,         mask=mask, valid_len_in=lengths)
        left_head  = self.left_visual_head( x=left_output,  mask=mask, valid_len_in=lengths)
        right_head = self.right_visual_head(x=right_output, mask=mask, valid_len_in=lengths)


        # --- 3. Ensemble: promediar probabilidades de las 4 cabezas ---
        if self.ensemble_method == 'weighted':
            if False:
                temperature = 2.0
                # Podenración por pesos aprendibles
                weights = torch.softmax(self.stream_weights/temperature, dim=0)
            else:
                branch_logits = self.stream_weights.clone()
                # --- DropBranch Universal ---
                if self.training:
                    drop_probs = [0.10, 0.10, 0.10, 0.30] # left, right, body, fuse

                    for i, p in enumerate(drop_probs):
                        if random.random() < p:
                            branch_logits[i] = -1e9

                    # Si se apagaron todas, encendemos UNA al azar
                    if (branch_logits == -1e9).all():
                        random_idx = random.randint(0, 3)
                        branch_logits[random_idx] = self.stream_weights[random_idx].clone()
                # -----------------------------

                weights = torch.softmax(branch_logits, dim=0)

            ensemble_probs = (
                weights[0] * left_head['gloss_probabilities'] +
                weights[1] * right_head['gloss_probabilities'] +
                weights[2] * body_head['gloss_probabilities'] +
                weights[3] * fuse_head['gloss_probabilities']
            ).log()
        else:
            # Promedio simple original
            # Se promedian probabilidades (no log-probabilidades) porque la suma de
            # distribuciones de probabilidad es más estable que la suma de logs.
            # Luego se aplica log para obtener log-probabilidades para CTC.
            ensemble_probs = (
                left_head['gloss_probabilities'] +
                right_head['gloss_probabilities'] +
                body_head['gloss_probabilities'] +
                fuse_head['gloss_probabilities']
            ).log()


        head_outputs = {
            # Ensemble
            'ensemble_last_gloss_logits':            ensemble_probs,
            'ensemble_last_gloss_probabilities_log': ensemble_probs.log_softmax(2),
            'ensemble_last_gloss_probabilities':     ensemble_probs.softmax(2),
            # Features para el VLMapper (configurable, por defecto usa fuse)
            'fuse': fuse,
            # Logits y log-probs de cada cabeza (necesarios para las losses CTC)
            'fuse_gloss_logits':              fuse_head['gloss_logits'],
            'fuse_gloss_probabilities_log':   fuse_head['gloss_probabilities_log'],
            'body_gloss_logits':              body_head['gloss_logits'],
            'body_gloss_probabilities_log':   body_head['gloss_probabilities_log'],
            'left_gloss_logits':              left_head['gloss_logits'],
            'left_gloss_probabilities_log':   left_head['gloss_probabilities_log'],
            'right_gloss_logits':             right_head['gloss_logits'],
            'right_gloss_probabilities_log':  right_head['gloss_probabilities_log'],
        }

        # Features que irán al VLMapper para la traducción.
        # Por defecto usa 'gloss_feature' de fuse_head, pero es configurable.
        self.cfg['gloss_feature_ensemble'] = self.cfg.get('gloss_feature_ensemble', 'gloss_feature')
        head_outputs['gloss_feature'] = fuse_head[self.cfg['gloss_feature_ensemble']]

        outputs = {
            **head_outputs,
            'input_lengths': src_input['new_src_lengths'],
        }

        # --- 4. Losses CTC independientes por cabeza ---
        # Cada cabeza se penaliza por separado para forzarla a ser útil por sí sola.
        # La loss total es la suma de las 4 losses individuales.
        gloss_labels  = src_input['gloss_input']['gloss_labels'].cuda()
        gloss_lengths = src_input['gloss_input']['gls_lengths'].cuda()

        for k in ['left', 'right', 'fuse', 'body']:
            outputs[f'recognition_loss_{k}'] = self.compute_recognition_loss(
                gloss_labels=gloss_labels,
                gloss_lengths=gloss_lengths,
                gloss_probabilities_log=head_outputs[f'{k}_gloss_probabilities_log'],
                input_lengths=src_input['new_src_lengths'].cuda(),
            )

        # Calcular la pérdida CTC para el ensemble mismo
        # Esto forzará a stream_weights a actualizarse basándose en qué cabeza es más útil
        if self.ensemble_method == 'weighted':
            outputs['recognition_loss_ensemble'] = self.compute_recognition_loss(
                gloss_labels=gloss_labels,
                gloss_lengths=gloss_lengths,
                gloss_probabilities_log=outputs['ensemble_last_gloss_probabilities_log'],
                input_lengths=src_input['new_src_lengths'].cuda(),
            )
        else:
            outputs['recognition_loss_ensemble'] = 0.0 # Si es promedio simple, no suma pérdida extra

        # Modificamos la suma total para incluir la pérdida del ensemble
        outputs['recognition_loss'] = (
            outputs['recognition_loss_left'] +
            outputs['recognition_loss_right'] +
            outputs['recognition_loss_fuse'] +
            outputs['recognition_loss_body'] +
            outputs['recognition_loss_ensemble'] # Loss para los pesos "stream_weights"
        )


        # --- 5. Knowledge Distillation (opcional) ---
        # El ensemble actúa como "profesor" y cada cabeza individual como "alumno".
        # KL divergence mide cuánto difiere la distribución del alumno respecto al profesor.
        # detach() en el profesor: sus gradientes no se propagan (es solo una señal de supervisión).
        # Esto transfiere el conocimiento del conjunto al individual, mejorando cada cabeza.
        if 'cross_distillation' in self.cfg:
            loss_func = torch.nn.KLDivLoss(reduction="batchmean")
            teacher_prob = outputs['ensemble_last_gloss_probabilities'].detach()

            for student in ['left', 'right', 'fuse', 'body']:
                student_log_prob = outputs[f'{student}_gloss_probabilities_log']
                outputs[f'{student}_distill_loss'] = loss_func(
                    input=student_log_prob,
                    target=teacher_prob,
                )
                outputs['recognition_loss'] += outputs[f'{student}_distill_loss']

        if self.ensemble_method == 'weighted':
            outputs['stream_weights'] = torch.softmax(self.stream_weights, dim=0).detach()
        return outputs
