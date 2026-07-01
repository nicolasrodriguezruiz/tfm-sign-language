# Sign-to-Text via Keypoints · SLM Decoder

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Trabajo de Fin de Máster — Exploración de Modelos de Lenguaje Pequeños (SLMs) como decodificadores de traducción en un sistema de Sign Language Translation (SLT) basado en keypoints.

**Autor:** Nicolás Rodríguez Ruiz
**Tutor:** Miguel Ángel Martínez Del Amor

## Resumen

La Traducción de Lengua de Signos (SLT) es la tarea de convertir automáticamente un vídeo continuo de lengua de signos en texto de lenguaje hablado. La mayoría de los sistemas competitivos dependen de vídeos RGB como entrada y de arquitecturas codificador-decodificador para la traducción, lo que los hace computacionalmente costosos e invasivos para la privacidad.

Este trabajo explora la integración de Modelos de Lenguaje Pequeños (SLMs) con arquitecturas de solo decodificador como reemplazo del módulo clásico de traducción codificador-decodificador en un *pipeline* de SLT basado en puntos clave. La red troncal visual es MSKA-SLR, una red de atención multiflujo que procesa puntos clave anatómicos 2D extraídos con AlphaPose, preservando la privacidad y reduciendo el coste computacional. El módulo de reconocimiento se acopla a modelos Qwen2.5 de escala creciente (1.5B y 7B de parámetros), ajustados finamente mediante Adaptación de Bajo Rango (LoRA) en el conjunto de datos PHOENIX-2014T. Adicionalmente, se evalúa un escenario de traducción *zero-shot* siguiendo el paradigma Spotter+GPT, donde las glosas predichas se pasan directamente a un modelo de 14B ajustado por instrucciones, sin ningún entrenamiento específico para la tarea.

Los resultados muestran que la mejor configuración de Qwen (1.5B, LoRA-All, sin *prompt* de instrucción) logra un BLEU-4 de 20.55 en el conjunto de prueba, reduciendo la brecha con la línea base mBART (22.48) a menos de dos puntos. Escalar a 7B de parámetros produce un BLEU-4 de 20.03 bajo un presupuesto de *finetuning* más restringido. El enfoque *zero-shot* resulta insuficiente para una traducción precisa (BLEU-4 de 0.46), pero demuestra una comprensión semántica parcial medida por BLEURT (0.50). Contrario a las expectativas, se observa que el *prompt* de instrucción resulta perjudicial durante el *finetuning*, ya que los tokens visuales y las instrucciones textuales compiten por la atención del modelo.

Estos hallazgos indican que los SLMs de solo decodificador son alternativas viables, aunque todavía no superiores, a los modelos de traducción especializados en escenarios de SLT con escasez de datos, identificándose la escala del modelo como una de las palancas más prometedoras para futuras mejoras.

## Resultados — PHOENIX-2014T

Comparativa global de todos los sistemas evaluados. B4: BLEU-4 · R: ROUGE-L · Mejor resultado en **negrita**.

| Sistema | Modelo | Entrada | Estrategia | Dev B4 | Dev R | Test B4 | Test R |
|---|---|---|---|:---:|:---:|:---:|:---:|
| MSKA-mBART | mBART | Visual | Full | **21.43** | **45.20** | **22.48** | **45.00** |
| MSKA-Qwen-7B | Qwen2.5-7B | Visual | LoRA-Att | 20.31 | 40.95 | 20.03 | 41.26 |
| MSKA-Qwen-1.5B (sin prompt) | Qwen2.5-1.5B | Visual | LoRA-All | 19.58 | 39.83 | 20.55 | 40.01 |
| MSKA-Qwen-1.5B (full) | Qwen2.5-1.5B | Visual | Full | 18.99 | 38.98 | 18.68 | 37.04 |
| MSKA-Qwen-1.5B (LoRA-All + glosas) | Qwen2.5-1.5B | Glosas | LoRA-All | 16.42 | 36.12 | 16.65 | 35.69 |
| MSKA-Qwen-1.5B (LoRA-Att) | Qwen2.5-1.5B | Visual | LoRA-Att | 16.35 | 37.05 | 16.31 | 36.50 |
| MSKA-Qwen-1.5B (LoRA-Att + glosas) | Qwen2.5-1.5B | Glosas | LoRA-Att | 14.84 | 33.70 | 15.63 | 34.59 |
| MSKA-Qwen-1.5B (full + glosas) | Qwen2.5-1.5B | Glosas | Full | 15.98 | 34.31 | 16.00 | 34.89 |
| Spotter+Qwen (zero-shot) | Qwen2.5-14B | Glosas | — | — | — | 0.46 | 9.85 |

## Estructura del repositorio

```
.
├── code/
│   ├── MSKA/
│   │   ├── train.py                   # Punto de entrada: entrenamiento y evaluación
│   │   ├── eval.py                    # Evaluación independiente con métricas
│   │   ├── pretrain_mapper.py         # Preentrenamiento del VLMapper
│   │   ├── MSKA_Spotter+Qwen.py       # Pipeline zero-shot: SLR → glosas → Qwen (sin finetuning)
│   │   ├── evaluate_oracle_slt.py     # Evaluación oráculo: glosas GT → Qwen (cota superior)
│   │   ├── Recognition/               # Módulo SLR: CTC, tokenizer, visual head
│   │   ├── MBart/                     # Módulo de traducción con mBART
│   │   ├── slm/                       # Módulo de traducción con Qwen (SLM)
│   │   ├── ICL/                       # Experimentos de in-context learning
│   │   ├── aux/                       # Métricas, optimizador, dataset prep, utils
│   │   └── configs/
│   │       ├── SLR_base_config.yaml   # Configuración base para reconocimiento
│   │       └── SLT_base_config.yaml   # Configuración base para traducción
│   ├── GRU/                           # Experimentación preliminar (ver nota)
│   ├── Transformer/                   # Experimentación preliminar (ver nota)
│   └── preprocesing/                  # Extracción y normalización de keypoints
└── Memoria/                           # Memoria del TFM (LaTeX)
```

> **Nota:** los directorios `GRU/` y `Transformer/` contienen experimentación preliminar realizada para familiarizarme con la tarea al inicio del trabajo. No forman parte del sistema descrito en la memoria y se conservan únicamente como registro del proceso de desarrollo.

## Uso

Los archivos `configs/SLR_base_config.yaml` y `configs/SLT_base_config.yaml` deben editarse antes de lanzar cualquier experimento: ajusta las rutas `data.train_label_path`, `data.dev_label_path` y `training.model_dir` a tu entorno.

**Entrenar el módulo de reconocimiento (SLR):**
```bash
cd code/MSKA
python train.py --config configs/SLR_base_config.yaml 
```

**Entrenar la traducción con mBART:**
```bash
python train.py --config configs/SLT_base_config.yaml --epoch 40
```

**Entrenar la traducción con Qwen (SLM, decoder-only):**
```bash
python train.py --config configs/SLT_base_config.yaml --slm --epoch 40
```

**Solo evaluación (sin entrenamiento):**
```bash
python train.py --config configs/SLT_base_config.yaml --slm --eval --resume path/to/checkpoint.pth
```

### Traducción *zero-shot* (Spotter+Qwen)

Pasa las glosas predichas por el módulo SLR directamente a un modelo Qwen sin *finetuning*, siguiendo el paradigma Spotter+GPT:
```bash
python MSKA_Spotter+Qwen.py 
    --config    configs/SLT_base_config.yaml \
    --slr_ckpt  path/to/best_checkpoint_slr.pth \
    --qwen_model Qwen/Qwen2.5-14B-Instruct \
    --split     test
```

### Evaluación oráculo (cota superior)

Evalúa la capacidad de traducción de Qwen partiendo de glosas de referencia perfectas (*ground truth*), sin el módulo SLR. Aísla el error de traducción del error de reconocimiento:
```bash
python evaluate_oracle_slt.py \
    --config     configs/SLT_base_config.yaml \
    --qwen_model Qwen/Qwen2.5-14B-Instruct \
    --split      test
```

## Reconocimientos

**Arquitectura MSKA-SLR** — El módulo de reconocimiento de glosas está basado en la arquitectura propuesta en:
> Mo Guan, Yan Wang, Guangkun Ma, Jiarui Liu y Mingzu Sun. *Multi-Stream Keypoint Attention Network for Sign Language Recognition and Translation*. arXiv:2405.05672, 2024.

**Qwen2.5** — Los modelos de lenguaje empleados en la etapa de traducción son desarrollados por el equipo Qwen de Alibaba Cloud.

**SacreBLEU** — Las métricas BLEU se calculan usando la implementación de referencia:
> Matt Post. *A Call for Clarity in Reporting BLEU Scores*. WMT, 2018.

## Licencia

MIT — consulta el archivo [LICENSE](LICENSE) para más detalles.
