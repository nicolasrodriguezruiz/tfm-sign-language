"""
pretrain_mapper.py
------------------
PROBLEMA QUE RESUELVE:
    Qwen fue preentrenado con texto. Cuando le damos features visuales como prefijo,
    no las "entiende" porque están en un espacio vectorial completamente distinto
    al de los embeddings de texto que conoce. El modelo aprende a ignorarlas.

SOLUCIÓN:
    Antes del entrenamiento completo, preentrenar SOLO el VLMapper para que aprenda
    a proyectar las features visuales al espacio de embeddings de Qwen.

    La señal de supervisión son los embeddings de Qwen correspondientes a las glosas
    ground truth: le decimos al VLMapper "las features visuales de esta seña deben
    parecerse al embedding de texto de la glosa correspondiente".

    Loss: cosine similarity entre:
        - mean pooling de las features visuales proyectadas  (B, D_qwen)
        - mean pooling de los embeddings de glosas de Qwen   (B, D_qwen)

    Durante el preentrenamiento:
        - Recognition network: CONGELADO (no se entrena)
        - Qwen (incluyendo LoRA): CONGELADO (no se entrena)
        - VLMapper: SE ENTRENA

USO:
    TOKENIZERS_PARALLELISM=false python pretrain_mapper.py --config configs/phoenix14t_s2t.yaml --epochs 30 --batch-size 8

    Después del preentrenamiento, el checkpoint del mapper se guarda en:
        {model_dir}/pretrained_mapper.pth

    Para usarlo en el entrenamiento principal, añadir al config:
        model:
          VLMapper:
            pretrained_mapper: /ruta/a/pretrained_mapper.pth
"""

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from Tokenizer import GlossTokenizer_S2G
from model import SignLanguageModel
import utils as utils
from S2T_Dataset import S2T_Dataset
import argparse
import yaml
import numpy as np
import random
from pathlib import Path
import json


import warnings
warnings.filterwarnings("ignore")
# ---------------------------------------------------------------------------
# Argumentos
# ---------------------------------------------------------------------------

def get_args_parser():
    parser = argparse.ArgumentParser('Pretrain VLMapper', add_help=False)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate para el VLMapper. Más alto que en el entrenamiento completo.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--pin-mem', action='store_true')
    parser.set_defaults(pin_mem=True)
    return parser


# ---------------------------------------------------------------------------
# Loss de alineación
# ---------------------------------------------------------------------------

def alignment_loss(visual_features, gloss_embeddings, visual_lengths, gloss_lengths):
    """
    Calcula la loss de alineación entre features visuales y embeddings de glosas.

    Usa mean pooling sobre los tokens válidos (sin padding) para obtener
    un vector representativo de cada secuencia, luego calcula la cosine
    similarity loss entre ambos.

    Args:
        visual_features:  (B, T, D) — features del VLMapper
        gloss_embeddings: (B, G, D) — embeddings de Qwen para las glosas
        visual_lengths:   (B,)      — longitudes reales de las features visuales
        gloss_lengths:    (B,)      — longitudes reales de las glosas

    Returns:
        loss escalar: 1 - cosine_similarity (0 = perfectamente alineado, 2 = opuesto)
    """
    B = visual_features.shape[0]

    # Mean pooling ignorando el padding en cada secuencia
    visual_mean = torch.zeros(B, visual_features.shape[-1], device=visual_features.device)
    gloss_mean  = torch.zeros(B, gloss_embeddings.shape[-1], device=gloss_embeddings.device)

    for i in range(B):
        vlen = visual_lengths[i]
        glen = gloss_lengths[i]
        visual_mean[i] = visual_features[i, :vlen, :].mean(dim=0)
        gloss_mean[i]  = gloss_embeddings[i, :glen, :].mean(dim=0)

    # Cosine similarity loss: queremos que ambos vectores apunten en la misma dirección
    # El valor 1 - cosine_similarity es 0 cuando son idénticos y 2 cuando son opuestos
    cos_sim = F.cosine_similarity(visual_mean, gloss_mean, dim=-1)  # (B,)
    loss = (1 - cos_sim).mean()
    return loss


# ---------------------------------------------------------------------------
# Una época de preentrenamiento
# ---------------------------------------------------------------------------

def pretrain_one_epoch(model, qwen_embeddings, qwen_tokenizer,
                       data_loader, optimizer, device, epoch, args):
    """
    Entrena el VLMapper durante una época.

    Recognition y Qwen están congelados, solo el VLMapper recibe gradientes.
    """
    model.train()
    # Asegurarse de que recognition y Qwen están en eval (afecta a BatchNorm y Dropout)
    model.recognition_network.eval()
    model.translation_network.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'Pretrain Epoch: [{epoch}/{args.epochs}]'

    # total_loss = 0
    # n_batches = 0
    # start = time.time()

    for step, src_input in enumerate(metric_logger.log_every(data_loader, print_freq=50, header=header)):

        # --- 1. Recognition (sin gradientes, está congelado) ---
        with torch.no_grad():
            recognition_outputs = model.recognition_network(src_input)

        # --- 2. VLMapper (CON gradientes) ---
        # Esta es la única parte que se entrena
        visual_features = model.vl_mapper(visual_outputs=recognition_outputs)
        # visual_features: (B, T, D_qwen)

        # --- 3. Embeddings de Qwen para las glosas ground truth ---
        with torch.no_grad():
            # Tokenizar las glosas con Qwen (no con el GlossTokenizer del recognition)
            # Las glosas son strings como "HEUTE WETTER KALT"
            gloss_texts = src_input['gloss']
            encoded = qwen_tokenizer(
                gloss_texts,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors='pt',
            ).to(device)
            # Obtener los embeddings de entrada de Qwen para esos tokens
            gloss_embs = qwen_embeddings(encoded['input_ids'])  # (B, G, D_qwen)
            gloss_lengths = encoded['attention_mask'].sum(dim=1)  # (B,) longitudes reales

        # --- 4. Loss de alineación ---
        visual_lengths = src_input['new_src_lengths'].to(device)
        loss = alignment_loss(
            visual_features=visual_features,
            gloss_embeddings=gloss_embs,
            visual_lengths=visual_lengths,
            gloss_lengths=gloss_lengths,
        )

        # --- 5. Backward ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # total_loss += loss.item()
        # n_batches += 1

        # if step % 50 == 0:
        #     elapsed = time.time() - start
        #     remaining = elapsed / (step + 1) * (len(data_loader) - step - 1)
        #     print(f"Pretrain Epoch [{epoch}]  [{step}/{len(data_loader)}]"
        #           f"  loss: {loss.item():.4f} ({total_loss / n_batches:.4f})"
        #           f"  eta: {datetime.timedelta(seconds=int(remaining))}")

        # --- Logging ---
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    print(f"Pretrain Epoch [{epoch}] completada — Métricas: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# ---------------------------------------------------------------------------
# Validación del preentrenamiento
# ---------------------------------------------------------------------------

def validate_alignment(model, qwen_embeddings, qwen_tokenizer, data_loader, device):
    """
    Calcula la loss de alineación en el conjunto de validación.
    Útil para detectar si el VLMapper está sobreajustando al train.
    """
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Validación Pretrain:'

    with torch.no_grad():
        for step, src_input in enumerate(metric_logger.log_every(data_loader, print_freq=50, header=header)):
            recognition_outputs = model.recognition_network(src_input)
            visual_features = model.vl_mapper(visual_outputs=recognition_outputs)

            gloss_texts = src_input['gloss']
            encoded = qwen_tokenizer(
                gloss_texts, padding=True, truncation=True,
                max_length=64, return_tensors='pt',
            ).to(device)
            gloss_embs    = qwen_embeddings(encoded['input_ids'])
            gloss_lengths = encoded['attention_mask'].sum(dim=1)
            visual_lengths = src_input['new_src_lengths'].to(device)

            loss = alignment_loss(visual_features, gloss_embs, visual_lengths, gloss_lengths)
            metric_logger.update(loss=loss.item())

    print(f"Validación alineación completada — Métricas: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    # Fijar semillas
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.benchmark = False

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device(args.device)
    output_dir = Path(config['training']['model_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- INICIALIZAR WANDB ---
    use_wandb = config.get('training', {}).get('wandb', 'disabled') == 'online'
    if use_wandb:
        import wandb
        wandb.init(project="MSKA-Pretrain-VLMapper", config=config)
        args.run = wandb
    else:
        args.run = None

    # --- Datasets ---
    print("Cargando datasets...")
    tokenizer = GlossTokenizer_S2G(config['gloss'])

    train_data = S2T_Dataset(
        path=config['data']['train_label_path'], tokenizer=tokenizer,
        config=config, args=args, phase='train', training_refurbish=True,
    )
    train_dataloader = DataLoader(
        train_data, batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=train_data.collate_fn, shuffle=True, pin_memory=args.pin_mem,
    )

    dev_data = S2T_Dataset(
        path=config['data']['dev_label_path'], tokenizer=tokenizer,
        config=config, args=args, phase='val', training_refurbish=True,
    )
    dev_dataloader = DataLoader(
        dev_data, batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=dev_data.collate_fn, pin_memory=args.pin_mem,
    )

    # --- Modelo completo ---
    print("Cargando modelo...")
    model = SignLanguageModel(cfg=config, args=args)
    model.to(device)

    # --- Congelar todo excepto el VLMapper ---
    # Recognition: congelado (ya tiene pesos preentrenados del S2G)
    # Qwen + LoRA: congelado (no queremos moverlo antes de la fase 2)
    # VLMapper: SE ENTRENA
    for name, param in model.named_parameters():
        if 'vl_mapper' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros entrenables en preentrenamiento: {n_trainable / 1e6:.2f}M "
          f"(solo VLMapper)")

    # --- Extraer tabla de embeddings de Qwen (congelada) ---
    # get_input_embeddings() devuelve la capa Embedding de Qwen.
    # La usamos para convertir IDs de glosas a vectores densos.
    # No se entrena: es nuestra "diana" fija hacia la que apunta el VLMapper.
    qwen_embeddings = model.translation_network.model.get_base_model().model.embed_tokens
    qwen_embeddings.eval()
    for param in qwen_embeddings.parameters():
        param.requires_grad = False

    qwen_tokenizer = model.translation_network.tokenizer

    # --- Optimizador: solo parámetros del VLMapper ---
    optimizer = torch.optim.Adam(
        [p for p in model.vl_mapper.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )

    # Scheduler coseno para el preentrenamiento
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5,
    )

    # --- Bucle de preentrenamiento ---
    print(f"\nIniciando preentrenamiento del VLMapper durante {args.epochs} épocas...")
    print(f"Objetivo: alinear features visuales con embeddings de glosas de Qwen\n")

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        train_stats = pretrain_one_epoch(
            model=model,
            qwen_embeddings=qwen_embeddings,
            qwen_tokenizer=qwen_tokenizer,
            data_loader=train_dataloader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            args=args,
        )

        test_stats = validate_alignment(
            model=model,
            qwen_embeddings=qwen_embeddings,
            qwen_tokenizer=qwen_tokenizer,
            data_loader=dev_dataloader,
            device=device,
        )
        # --- GUARDAR EN WANDB (Opcional, si está activado) ---
        if args.run:
            args.run.log({
                'epoch': epoch + 1,
                **{f'pretrain/train_{k}': v for k, v in train_stats.items()},
                **{f'pretrain/test_{k}': v for k, v in test_stats.items()},
            })

        scheduler.step()

        # Guardar el mejor mapper según la loss de validación
        # Ahora accedemos a la loss así: test_stats['loss']
        if test_stats['loss'] < best_val_loss:
            best_val_loss = test_stats['loss']
            torch.save(
                model.vl_mapper.state_dict(),
                output_dir / 'pretrained_mapper.pth',
            )
            print(f"  → Nuevo mejor mapper guardado (val_loss={best_val_loss:.4f})")

        # --- EL LOG.TXT COMO LO PEDISTE ---

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,
            'n_parameters': round(n_trainable, 4),
        }

        with (output_dir / "log_pre.txt").open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")

        # Guardar el mejor mapper según la loss de validación
        if test_stats['loss'] < best_val_loss:
            best_val_loss = test_stats['loss']
            torch.save(
                model.vl_mapper.state_dict(),
                output_dir / 'pretrained_mapper.pth',
            )
            print(f"  → Nuevo mejor mapper guardado (val_loss={best_val_loss:.4f})")

        print(f"Época {epoch}: train={train_stats['loss']:.4f}  val={test_stats['loss']:.4f}"
              f"  lr={scheduler.get_last_lr()[0]:.6f}\n")

    print(f"\nPreentrenamiento completado.")
    print(f"Mejor val_loss: {best_val_loss:.4f}")
    print(f"Mapper guardado en: {output_dir}/pretrained_mapper.pth")
    # print(f"\nAhora añade al config YAML:")
    # print(f"  model:")
    # print(f"    VLMapper:")
    # print(f"      pretrained_mapper: {output_dir}/pretrained_mapper.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pretrain VLMapper', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
