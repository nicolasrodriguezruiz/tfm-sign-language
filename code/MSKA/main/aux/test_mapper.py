"""
evaluate_mapper.py
------------------
Script para evaluar el rendimiento del VLMapper preentrenado cargando 
sus pesos desde un archivo .pth.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import yaml
from pathlib import Path

# Importaciones de tu proyecto
from Tokenizer import GlossTokenizer_S2G
from model import SignLanguageModel
import utils as utils
from datasets import S2T_Dataset

def get_args_parser():
    parser = argparse.ArgumentParser('Evaluate pre-trained VLMapper', add_help=False)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--config', type=str, required=True, help='Ruta al archivo YAML de configuración')
    parser.add_argument('--checkpoint', type=str, required=True, help='Ruta al archivo .pth del mapper (ej. pretrained_mapper.pth)')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'], help='Conjunto de datos a evaluar')
    parser.add_argument('--pin-mem', action='store_true')
    parser.set_defaults(pin_mem=True)
    return parser

def alignment_loss(visual_features, gloss_embeddings, visual_lengths, gloss_lengths):
    """Calcula la loss de alineación usando mean pooling."""
    B = visual_features.shape[0]
    visual_mean = torch.zeros(B, visual_features.shape[-1], device=visual_features.device)
    gloss_mean  = torch.zeros(B, gloss_embeddings.shape[-1], device=gloss_embeddings.device)

    for i in range(B):
        vlen = visual_lengths[i]
        glen = gloss_lengths[i]
        visual_mean[i] = visual_features[i, :vlen, :].mean(dim=0)
        gloss_mean[i]  = gloss_embeddings[i, :glen, :].mean(dim=0)

    cos_sim = F.cosine_similarity(visual_mean, gloss_mean, dim=-1)
    loss = (1 - cos_sim).mean()
    return loss

def evaluate_alignment(model, qwen_embeddings, qwen_tokenizer, data_loader, device):
    """Ejecuta la evaluación sobre un dataloader específico."""
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluación VLMapper:'

    with torch.no_grad():
        for step, src_input in enumerate(metric_logger.log_every(data_loader, print_freq=50, header=header)):
            # 1. Obtener features del recognition
            recognition_outputs = model.recognition_network(src_input)
            
            # 2. Proyectar con el VLMapper cargado
            visual_features = model.vl_mapper(visual_outputs=recognition_outputs)

            # 3. Obtener embeddings de glosas objetivo
            gloss_texts = src_input['gloss']
            encoded = qwen_tokenizer(
                gloss_texts, padding=True, truncation=True,
                max_length=64, return_tensors='pt',
            ).to(device)
            gloss_embs    = qwen_embeddings(encoded['input_ids'])
            gloss_lengths = encoded['attention_mask'].sum(dim=1)
            visual_lengths = src_input['new_src_lengths'].to(device)

            # 4. Calcular loss
            loss = alignment_loss(visual_features, gloss_embs, visual_lengths, gloss_lengths)
            metric_logger.update(loss=loss.item())

    print(f"\nEvaluación completada — Loss promedio: {metric_logger.loss.global_avg:.4f}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def main(args):
    device = torch.device(args.device)

    # Cargar configuración
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print("Cargando dataset y tokenizador...")
    tokenizer = GlossTokenizer_S2G(config['gloss'])
    
    # Seleccionar el path según el split (val o test)
    label_path_key = 'dev_label_path' if args.split == 'val' else 'test_label_path'
    
    eval_data = S2T_Dataset(
        path=config['data'][label_path_key], tokenizer=tokenizer,
        config=config, args=args, phase=args.split, training_refurbish=False,
    )
    eval_dataloader = DataLoader(
        eval_data, batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=eval_data.collate_fn, pin_memory=args.pin_mem, shuffle=False
    )

    print("Inicializando arquitectura del modelo...")
    model = SignLanguageModel(cfg=config, args=args)
    model.to(device)

    # --- LA MAGIA OCURRE AQUÍ: Cargar los pesos del mapper ---
    print(f"Cargando pesos del VLMapper desde: {args.checkpoint}")
    mapper_state_dict = torch.load(args.checkpoint, map_location=device)
    
    # Se usa strict=True para asegurar que las llaves del state_dict coinciden exactamente
    model.vl_mapper.load_state_dict(mapper_state_dict, strict=True)
    print("¡Pesos cargados correctamente!")
    
    # Extraer las partes de Qwen necesarias para la evaluación
    qwen_embeddings = model.translation_network.model.get_base_model().model.embed_tokens
    qwen_embeddings.eval()
    qwen_tokenizer = model.translation_network.tokenizer

    print(f"\nIniciando evaluación en el conjunto de '{args.split}'...")
    evaluate_alignment(
        model=model,
        qwen_embeddings=qwen_embeddings,
        qwen_tokenizer=qwen_tokenizer,
        data_loader=eval_dataloader,
        device=device
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate VLMapper', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)