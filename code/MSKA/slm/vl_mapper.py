import torch
import torch.nn as nn


class VLMapper(torch.nn.Module):
    """
    Puente entre el Recognition network y el Translation network (Qwen).

    Proyecta las features visuales del recognition (hidden_size del VisualHead)
    al espacio de embeddings de Qwen (hidden_size del LLM).

    Con MBart se usaba una proyección simple. Con Qwen se usa un MLP de dos capas
    con GELU, que es lo que usan modelos multimodales como LLaVA y funciona mejor
    para alinear espacios visuales y lingüísticos.

    Soporta dos modos (controlable desde el config):
      - 'projection' (por defecto): MLP que aprende la transformación.
      - 'embedding':  matriz lineal inicializada con embeddings preentrenados de glosas.
    """

    def __init__(self, cfg, in_features, out_features,
                 gloss_id2str=None, gls2embed=None, freeze=False):
        super().__init__()
        self.type = cfg.get('type', 'projection')

        if self.type == 'projection':
            # MLP de dos capas con GELU (estándar en modelos multimodales como LLaVA).
            # La capa intermedia tiene la misma dimensión que la salida.
            # GELU es más suave que ReLU y funciona mejor con LLMs preentrenados.
            self.mapping = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.GELU(),
                nn.Linear(out_features, out_features),
            )

        elif self.type == 'embedding':
            # Matriz lineal sin bias donde cada columna corresponde a una glosa.
            # Se inicializa con embeddings preentrenados de glosas.
            assert in_features == len(gloss_id2str), (in_features, gloss_id2str)
            self.mapping = nn.Linear(in_features, out_features, bias=False)

            with torch.no_grad():
                for i, s in gloss_id2str.items():
                    if s in gls2embed:
                        self.mapping.weight[:, i] = gls2embed[s]
                    else:
                        self.mapping.weight[:, i] = 0

    def forward(self, visual_outputs, lengths=None):
        """
        Proyecta las features del VisualHead al espacio de embeddings de Qwen.

        visual_outputs: dict de salidas del recognition network.
                        Se usa 'gloss_feature', shape (B, T, hidden_size_recognition).
        lengths:        no se usa, existe por compatibilidad.

        Returns:
            features proyectadas, shape (B, T, hidden_size_qwen).
        """
        return self.mapping(visual_outputs['gloss_feature'])
