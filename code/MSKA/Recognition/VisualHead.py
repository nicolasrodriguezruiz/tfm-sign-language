import torch
import torch.nn as nn
import torch.nn.functional as F
from aux.utils import PositionalEncoding, MaskedNorm, PositionwiseFeedForward, MLPHead


class VisualHead(torch.nn.Module):
    """
    Módulo que transforma features visuales del encoder en probabilidades sobre
    el vocabulario de glosas. Es el último paso antes de la decodificación.

    Parámetros:
        cls_num:            número de glosas en el vocabulario (tamaño de salida).
        input_size:         dimensión de las features que llegan del encoder.
                            Si es None, se asume que ya tienen la dimensión correcta.
        hidden_size:        dimensión interna de todas las capas intermedias.
        ff_size:            dimensión interna del bloque FeedForward.
        pe:                 si True, añade encoding posicional temporal.
        ff_kernelsize:      tamaños de kernel para las convoluciones del FeedForward.
        pretrained_ckpt:    ruta a un checkpoint preentrenado para inicializar pesos.
        is_empty:           si True, omite todas las capas intermedias y solo aplica
                            la clasificación final. Útil cuando las features ya tienen
                            el formato correcto.
        frozen:             si True, congela todas las capas excepto plus_conv y
                            gloss_output_layer. Útil para fine-tuning parcial.
        plus_conv_cfg:      configuración para convoluciones 1D adicionales que hacen
                            downsampling temporal. Si es {}, no se aplica ninguna.
        ssl_projection_cfg: configuración para la cabeza de aprendizaje auto-supervisado
                            (SSL). Si es {}, no se usa.
    """

    def __init__(self,
                 cls_num, input_size=512, hidden_size=1024, ff_size=2048, pe=True,
                 ff_kernelsize=[3, 3], pretrained_ckpt=None, is_empty=False, frozen=False,
                 plus_conv_cfg={},
                 ssl_projection_cfg={}):
        super().__init__()
        self.is_empty = is_empty
        self.plus_conv_cfg = plus_conv_cfg
        self.ssl_projection_cfg = ssl_projection_cfg

        if is_empty == False:
            # --- Modo normal: pipeline completo de transformación ---
            self.frozen = frozen
            self.hidden_size = hidden_size

            # Capa de proyección inicial: mapea input_size → hidden_size.
            # Si input_size es None las features ya tienen hidden_size dimensiones,
            # así que se usa Identity() para no transformarlas.
            if input_size is None:
                self.fc1 = nn.Identity()
            else:
                self.fc1 = torch.nn.Linear(input_size, self.hidden_size)

            # BatchNorm enmascarada: normaliza las features ignorando las posiciones
            # de padding (por eso necesita la máscara). Una BatchNorm normal incluiría
            # los frames de padding en el cálculo de media y varianza, distorsionando
            # la normalización.
            self.bn1 = MaskedNorm(num_features=self.hidden_size, norm_type='batch')

            self.relu1 = torch.nn.ReLU()
            self.dropout1 = torch.nn.Dropout(p=0.1)

            # Encoding posicional: añade información sobre la posición temporal de
            # cada frame. Sin esto el modelo no sabe si un frame es el primero o el
            # último, ya que la atención por sí sola es invariante al orden.
            # Si pe=False se usa Identity() (sin encoding posicional).
            if pe:
                self.pe = PositionalEncoding(self.hidden_size)
            else:
                self.pe = torch.nn.Identity()

            # Bloque FeedForward con conexión residual (skip_connection=True).
            # Equivalente al bloque FFN de un transformer: expande la dimensión a
            # ff_size, aplica no-linealidad, y la reduce de vuelta a hidden_size.
            # La conexión residual suma la entrada al resultado: x = x + FFN(x),
            # lo que facilita el flujo de gradientes en el entrenamiento.
            self.feedforward = PositionwiseFeedForward(
                input_size=self.hidden_size,
                ff_size=ff_size,
                dropout=0.1,
                kernel_size=ff_kernelsize,
                skip_connection=True,
            )

            # LayerNorm tras el FeedForward: estabiliza la distribución de activaciones.
            # eps=1e-6 evita división por cero en la normalización.
            self.layer_norm = torch.nn.LayerNorm(self.hidden_size, eps=1e-6)

            # Convoluciones 1D adicionales opcionales (plus_conv).
            # Cada capa puede hacer downsampling temporal si stride > 1.
            # Esto es lo que produce la reducción de longitud /4 que necesita el
            # transformer (combinado con las dos divisiones de collate_fn).
            # padding_mode='replicate' repite el último valor en los bordes en lugar
            # de rellenar con ceros, lo que es más adecuado para señales temporales.
            # Si plus_conv_cfg == {}, se usa Identity() (sin downsampling).
            if plus_conv_cfg != {}:
                plus_convs = []
                for i in range(plus_conv_cfg['num_layer']):
                    plus_convs.append(nn.Conv1d(
                        self.hidden_size, self.hidden_size,
                        kernel_size=plus_conv_cfg['kernel_size'],
                        stride=plus_conv_cfg['stride'],
                        padding_mode='replicate',
                    ))
                self.plus_conv = nn.Sequential(*plus_convs)
            else:
                self.plus_conv = nn.Identity()

            # Cabeza de proyección SSL (Self-Supervised Learning) opcional.
            # Produce una representación compacta y normalizada para losses contrastivas
            # como SimCLR o MoCo, que entrenan al modelo a distinguir muestras similares
            # de distintas sin etiquetas. Si ssl_projection_cfg == {}, no se usa.
            if ssl_projection_cfg != {}:
                self.ssl_projection = MLPHead(
                    embedding_size=self.hidden_size,
                    projection_hidden_size=ssl_projection_cfg['hidden_size'],
                )

            # Capa de clasificación final: proyecta hidden_size → num_glosas.
            # Produce los logits (puntuaciones sin normalizar) para cada glosa
            # en cada frame temporal. CTC los convierte en secuencias de glosas.
            self.gloss_output_layer = torch.nn.Linear(self.hidden_size, cls_num)

            # Congelar capas para fine-tuning parcial.
            # Al congelar se hace dos cosas:
            #   1. requires_grad=False: PyTorch no calcula gradientes para estos parámetros,
            #      reduciendo memoria y cómputo en el backward pass.
            #   2. layer.eval(): desactiva el comportamiento de entrenamiento de BatchNorm
            #      y Dropout (BatchNorm usa estadísticas acumuladas en lugar del batch actual).
            # plus_conv y gloss_output_layer NO se congelan: son las capas que se
            # reentrenan durante el fine-tuning.
            if self.frozen:
                self.frozen_layers = [
                    self.fc1, self.bn1, self.relu1,
                    self.pe, self.dropout1,
                    self.feedforward, self.layer_norm,
                ]
                for layer in self.frozen_layers:
                    for name, param in layer.named_parameters():
                        param.requires_grad = False
                    layer.eval()

        else:
            # --- Modo vacío (is_empty=True) ---
            # Las features que llegan ya tienen el formato correcto.
            # Solo se aplica la clasificación final directamente sobre input_size.
            self.gloss_output_layer = torch.nn.Linear(input_size, cls_num)

        # Cargar pesos preentrenados si se especifica un checkpoint
        if pretrained_ckpt:
            self.load_from_pretrained_ckpt(pretrained_ckpt)

    def load_from_pretrained_ckpt(self, pretrained_ckpt):
        """
        Carga los pesos del VisualHead desde un checkpoint preentrenado.
        Solo extrae las claves que pertenecen a 'recognition_network.visual_head.*',
        eliminando ese prefijo para que coincidan con los nombres de este módulo.
        """
        logger = get_logger()
        checkpoint = torch.load(pretrained_ckpt, map_location='cpu')['model_state']
        load_dict = {}
        for k, v in checkpoint.items():
            if 'recognition_network.visual_head.' in k:
                load_dict[k.replace('recognition_network.visual_head.', '')] = v
        self.load_state_dict(load_dict)
        logger.info('Load Visual Head from pretrained ckpt {}'.format(pretrained_ckpt))

    def forward(self, x, mask, valid_len_in=None):
        """
        Transforma features visuales en logits de glosas.

        Args:
            x:            features del encoder, shape (B, T, D)
                          B = batch size, T = frames temporales, D = dimensión
            mask:         máscara booleana (B, 1, T), True donde hay frames reales
                          y False donde hay padding. Necesaria para MaskedNorm.
            valid_len_in: longitudes reales de cada secuencia antes de este módulo,
                          shape (B,). Necesarias para ajustar las longitudes tras
                          el downsampling de plus_conv.

        Returns:
            Diccionario con features, logits y longitudes. Ver sección de retorno.
        """
        B, Tin, D = x.shape  # guardar Tin para calcular valid_len_out al final

        if self.is_empty == False:

            if not self.frozen:
                # --- Pipeline completo (capas no congeladas) ---

                # 1. Proyección lineal + normalización + activación
                x = self.fc1(x)          # (B, T, input_size) → (B, T, hidden_size)
                x = self.bn1(x, mask)    # BatchNorm enmascarada: normaliza ignorando padding
                x = self.relu1(x)        # introduce no-linealidad

                # 2. Encoding posicional + dropout
                # El PE suma a cada posición t un vector sinusoidal único,
                # dando al modelo información sobre el orden temporal.
                x = self.pe(x)
                x = self.dropout1(x)     # regularización: apaga neuronas aleatoriamente

                # 3. FeedForward + normalización
                x = self.feedforward(x)  # expande a ff_size y vuelve a hidden_size
                x = self.layer_norm(x)   # estabiliza distribución de activaciones

                # 4. Convoluciones temporales opcionales
                # Conv1d espera (B, D, T), pero x tiene shape (B, T, D),
                # por eso se transpone antes y después.
                x = x.transpose(1, 2)   # (B, T, D) → (B, D, T)
                x = self.plus_conv(x)    # puede reducir T si stride > 1
                x = x.transpose(1, 2)   # (B, D, T) → (B, T', D)  T' <= T

            else:
                # --- Pipeline con capas congeladas ---
                # Se ejecuta con torch.no_grad() para no calcular gradientes
                # en las capas congeladas, ahorrando memoria y tiempo.
                # Además se fuerza eval() en cada paso por si el modo cambió
                # (p.ej. si model.train() se llamó después de congelar).
                with torch.no_grad():
                    for ii, layer in enumerate(self.frozen_layers):
                        layer.eval()
                        if ii == 1:
                            # bn1 (índice 1) necesita la máscara como segundo argumento,
                            # el resto de capas solo reciben x
                            x = layer(x, mask)
                        else:
                            x = layer(x)

                # plus_conv y gloss_output_layer SÍ reciben gradientes (no están congeladas)
                x = x.transpose(1, 2)
                x = self.plus_conv(x)
                x = x.transpose(1, 2)

        # --- Clasificación final (compartida por modo normal, congelado y vacío) ---

        # Proyectar cada frame a un vector de tamaño num_glosas
        logits = self.gloss_output_layer(x)  # (B, T', hidden_size) → (B, T', num_glosas)

        # log_softmax: normaliza los logits a log-probabilidades.
        # CTC loss trabaja en espacio logarítmico por estabilidad numérica.
        gloss_probabilities_log = logits.log_softmax(2)

        # softmax normal: para visualización o análisis, no para la loss.
        gloss_probabilities = logits.softmax(2)

        # --- Ajuste de longitudes válidas tras el posible downsampling ---
        # Si plus_conv redujo la dimensión temporal (Tout < Tin), las longitudes
        # válidas de cada muestra se escalan proporcionalmente.
        # Ejemplo: si Tin=100, Tout=50, valid_len_in=80 → valid_len_out=40
        # Estas longitudes son necesarias para que CTC ignore el padding.
        if self.plus_conv_cfg != {}:
            B, Tout, D = x.shape
            valid_len_out = torch.floor(valid_len_in * Tout / Tin).long()
        else:
            valid_len_out = valid_len_in

        # --- Cabeza SSL opcional ---
        # Proyecta las features a un espacio compacto para losses contrastivas.
        # Si normalize=True, las features se proyectan a la esfera unitaria,
        # lo que es estándar en losses como SimCLR (distancia coseno).
        if self.ssl_projection_cfg != {}:
            x_ssl = self.ssl_projection(x)
            if self.ssl_projection_cfg['normalize'] == True:
                x_ssl = F.normalize(x_ssl, dim=-1)
        else:
            x_ssl = None

        return {
            # Features antes de la clasificación → las usa VLMapper para traducción
            'gloss_feature': x,
            # Features L2-normalizadas → para métricas de similitud coseno
            'gloss_feature_norm': F.normalize(x, dim=-1),
            # Features para loss contrastiva SSL → None si no se usa SSL
            'gloss_feature_ssl': x_ssl,
            # Logits sin normalizar → los usa el CTC decoder
            'gloss_logits': logits,
            # Log-probabilidades → las usa la CTC loss durante entrenamiento
            'gloss_probabilities_log': gloss_probabilities_log,
            # Probabilidades normales → para análisis y visualización
            'gloss_probabilities': gloss_probabilities,
            # Longitudes reales tras downsampling → las usa CTC y el transformer
            'valid_len_out': valid_len_out,
        }
