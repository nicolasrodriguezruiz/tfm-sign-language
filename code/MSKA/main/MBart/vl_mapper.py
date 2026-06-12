"""El recognition network produce features en su propio espacio vectorial (dimensión y distribución propias). El translation network espera features en otro espacio distinto. El VLMapper transforma unas en otras."""
import torch


class VLMapper(torch.nn.Module):
    """
    Puente entre el Recognition network y el Translation network.

    El recognition produce features en su propio espacio vectorial; el translation
    espera features en otro espacio distinto (dimensión y distribución diferentes).
    Este módulo aprende (o inicializa) la transformación entre ambos espacios.

    Soporta dos modos:
      - 'projection': red neuronal pequeña que aprende la transformación.
      - 'embedding':  matriz lineal inicializada con embeddings preentrenados de glosas.
    """

    def __init__(self, cfg, in_features, out_features,
                 gloss_id2str=None,
                 gls2embed=None,
                 freeze=False) -> None:
        super().__init__()
        self.type = cfg.get('type', 'projection')
        if self.type not in ['embedding', 'projection'] : raise ValueError(f"Tipo de VLMapper no reconocido: {self.type}")

        if self.type == 'projection':
            # Red de dos capas con ReLU que aprende la transformación durante el entrenamiento.
            # El tamaño intermedio es igual a out_features (elección del autor, no hay restricción técnica).
            self.hidden_size = out_features
            self.mapping = torch.nn.Sequential(
                torch.nn.Linear(in_features=in_features, out_features=self.hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=self.hidden_size, out_features=out_features),
            )

        elif self.type == 'embedding':
            # Capa lineal sin bias donde cada columna i corresponde a la glosa i.
            # in_features debe ser igual al número de glosas (una entrada por glosa).
            assert in_features == len(gloss_id2str), (in_features, gloss_id2str)
            self.mapping = torch.nn.Linear(
                in_features=in_features,
                out_features=out_features,
                bias=False,
            )

            # Inicializar cada columna con el embedding preentrenado de su glosa.
            # Se hace con no_grad() porque es solo inicialización, no un paso de entrenamiento.
            # Si una glosa no tiene embedding preentrenado, su columna se deja a cero.
            with torch.no_grad():
                for i, s in gloss_id2str.items():
                    if s in gls2embed:
                        self.mapping.weight[:, i] = gls2embed[s]
                    else:
                        self.mapping.weight[:, i] = 0

    def forward(self, visual_outputs, lengths=None):
        """
        Transforma las features visuales del recognition al espacio del translation.

        visual_outputs: diccionario de salidas del recognition network.
                        Se usa la clave 'gloss_feature' (features continuas del encoder).
        lengths:        no se usa, existe por compatibilidad con otras interfaces.
        """
        # Ambos tipos de mapper aplican self.mapping de la misma forma.
        # La distinción solo existe en cómo se construye self.mapping en __init__.
        return self.mapping(visual_outputs['gloss_feature'])
