"""
evaluate_oracle_slt.py
--------------------------
Evaluación Oráculo (Upper Bound): Glosas perfectas (Ground Truth) → Qwen → texto alemán.

Pipeline:
              GT Glosses (texto)
              → prompt_str + glosas → Qwen.generate()
              → BLEU-1/2/3/4 + ROUGE

NO usa SLR, ni imágenes, ni VLMapper. Evalúa puramente la capacidad de 
traducción de Qwen en condiciones ideales.

Uso:
    python evaluate_oracle_slt.py \
        --config  configs/phoenix14t_s2t.yaml \
        --qwen_model Qwen/Qwen2.5-1.5B-Instruct \
        --split   test \
        --batch_size 8
"""

import argparse
import os
import sys
import json
import yaml
import random
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

from Recognition.Tokenizer import GlossTokenizer_S2G
from aux.metrics import bleu, rouge
import aux.utils as utils

import warnings
warnings.filterwarnings("ignore")

# ── args ────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',     type=str, required=True)
    # Ya no requerimos el --slr_ckpt
    p.add_argument('--qwen_model', type=str, default=None,
                   help='Ruta o nombre de Qwen. Por defecto usa el del config.')
    p.add_argument('--split',      type=str, default='test',
                   choices=['dev', 'test'])
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--max_new_tokens', type=int, default=100)
    p.add_argument('--num_workers',    type=int, default=4)
    p.add_argument('--seed',       type=int, default=0)
    p.add_argument('--output_dir', type=str, default="./",
                   help='Dónde guardar resultados JSON. Por defecto: model_dir del config.')
    p.add_argument('--device', default='cuda')
    return p.parse_args()


# ── Qwen sin finetuning ──────────────────────────────────────────────────────

def load_qwen(model_name, device):
    """Carga Qwen base sin LoRA, en bfloat16."""
    print(f"Cargando Qwen desde: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.padding_side = 'left'
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print(f"Qwen listo ({sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params)")
    return tokenizer, model


# ── generación glosas → texto ────────────────────────────────────────────────

def generate_from_glosses(
    gloss_strings,   # list[str]  e.g. ["HEUTE WETTER KALT", ...]
    prompt_str,      # str        prefijo del config, e.g. "Translate to German: "
    qwen_tokenizer,
    qwen_model,
    device,
    gen_kwargs,
):
    """
    Construye: prompt_str + glosas → Qwen.generate() → texto alemán.
    """
    inputs_text = []
    for g in gloss_strings:
        # Usamos el formato nativo de mensajes para modelos de chat
        messages = [
            {"role": "system", "content": prompt_str},
            {"role": "user", "content": g}
        ]
        formatted_text = qwen_tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        inputs_text.append(formatted_text)
    
    encoded = qwen_tokenizer(
        inputs_text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=False,
    ).to(device)

    with torch.no_grad():
        output = qwen_model.generate(
            input_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask'],
            eos_token_id=qwen_tokenizer.eos_token_id,
            pad_token_id=qwen_tokenizer.pad_token_id,
            **gen_kwargs
        )

    # Qwen devuelve input_ids + nuevos tokens; nos quedamos solo con los generados
    input_len = encoded['input_ids'].shape[1]
    generated_ids = output[:, input_len:]

    decoded = qwen_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return decoded


# ── evaluación principal ─────────────────────────────────────────────────────

def evaluate(args, config):
    device = torch.device(args.device)

    # Semillas
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ── Dataset ──────────────────────────────────────────────────────────────
    if args.split == 'test':
        from slm.S2T_Dataset import S2T_Dataset
        data_path = config['data']['test_label_path']
    else:
        from slm.S2T_Dataset import S2T_Dataset
        data_path = config['data']['dev_label_path']

    tokenizer_gls = GlossTokenizer_S2G(config['gloss'])

    # Forzamos task=S2T
    tmp_config = {**config, 'task': 'S2T'}
    dataset = S2T_Dataset(
        path=data_path,
        tokenizer=tokenizer_gls,
        config=tmp_config,
        args=argparse.Namespace(batch_size=args.batch_size, num_workers=args.num_workers),
        phase=args.split if args.split != 'dev' else 'test',
        training_refurbish=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        shuffle=False,
        pin_memory=True,
    )
    print(f"Dataset '{args.split}': {len(dataset)} muestras, {len(dataloader)} batches")

    # ── Qwen ─────────────────────────────────────────────────────────────────
    qwen_model_name = (
        args.qwen_model
        or config['model']['TranslationNetwork'].get(
            'pretrained_model_name_or_path', 'Qwen/Qwen2.5-1.5B'
        )
    )
    print(f"Usando modelo: {qwen_model_name}")
    qwen_tokenizer, qwen_model = load_qwen(qwen_model_name, device)

    prompt_str = config['data'].get('prompt', '')
    print(f"Prompt usado: {repr(prompt_str)}")

    # ── Bucle de evaluación ───────────────────────────────────────────────────
    results = {}
    metric_logger = utils.MetricLogger(delimiter="  ")
    
    # Extraer parámetros de generación desde el config YAML
    test_cfg = config.get('testing', {})
    trans_cfg = test_cfg.get('translation', {})
    gen_kwargs = {
        'max_new_tokens': trans_cfg.get('max_new_tokens', 51),
        'num_beams': trans_cfg.get('num_beams', 4),
        'length_penalty': trans_cfg.get('length_penalty', 1.0),
        'repetition_penalty': trans_cfg.get('repetition_penalty', 1.0),
        'no_repeat_ngram_size': trans_cfg.get('no_repeat_ngram_size', 0),
        'early_stopping': trans_cfg.get('early_stopping', False),
        'temperature': trans_cfg.get('temperature', 1.0),
        'do_sample': trans_cfg.get('do_sample', False),
    }    
    
    print(gen_kwargs)

    print(f"\nIniciando Evaluación ORÁCULO (GT Glosas → Qwen)...\n")
    
    with torch.no_grad():
        for step, batch in enumerate(
            metric_logger.log_every(dataloader, print_freq=20, header=f'[{args.split}]')
        ):
            # EXTRAER GLOSAS GROUND TRUTH DIRECTAMENTE DEL BATCH
            # Normalizamos a mayúsculas como haría el tokenizador
            gt_gls_strings = [
                g.upper() if tokenizer_gls.lower_case else g 
                for g in batch['gloss']
            ]

            # Glosas Perfectas → Qwen → texto
            txt_hyp = generate_from_glosses(
                gloss_strings=gt_gls_strings,
                prompt_str=prompt_str,
                qwen_tokenizer=qwen_tokenizer,
                qwen_model=qwen_model,
                device=device,
                gen_kwargs=gen_kwargs,
            )

            for name, hyp, ref, gls_ref in zip(
                batch['name'], txt_hyp, batch['text'], gt_gls_strings
            ):
                results[name] = {
                    'txt_hyp': hyp,
                    'txt_ref': ref,
                    'gls_ref': gls_ref,
                }

    # ── Calcular métricas ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MÉTRICAS ORÁCULO (GT Glosas → Qwen)")
    print("=" * 60)

    names     = list(results.keys())
    txt_refs  = [results[n]['txt_ref']  for n in names]
    txt_hyps  = [results[n]['txt_hyp']  for n in names]

    # BLEU + ROUGE sobre la traducción
    level      = config['data'].get('level', 'word')
    bleu_dict  = bleu(references=txt_refs, hypotheses=txt_hyps, level=level)
    rouge_score = rouge(references=txt_refs, hypotheses=txt_hyps, level=level)

    print(f"\n── Traducción en Condiciones Ideales ──")
    for k in ['bleu1', 'bleu2', 'bleu3', 'bleu4']:
        print(f"  {k.upper()}:  {bleu_dict[k]:.2f}")
    print(f"  ROUGE-L:     {rouge_score:.2f}")

    # ── Guardar resultados ─────────────────────────────────────────────────────
    output_dir = Path(args.output_dir or config['training']['model_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'oracle_gloss2qwen_{args.split}.json'

    summary = {
        'split': args.split,
        'qwen_model': qwen_model_name,
        'prompt': prompt_str,
        'n_samples': len(results),
        'translation': {
            'bleu1':   round(bleu_dict['bleu1'], 2),
            'bleu2':   round(bleu_dict['bleu2'], 2),
            'bleu3':   round(bleu_dict['bleu3'], 2),
            'bleu4':   round(bleu_dict['bleu4'], 2),
            'rouge_l': round(rouge_score, 2),
        },
        'samples': [
            {
                'name':     n,
                'gls_ref':  results[n]['gls_ref'],
                'txt_ref':  results[n]['txt_ref'],
                'txt_hyp':  results[n]['txt_hyp'],
            }
            for n in names
        ],
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        
    # ── Guardar resultados en CSV ─────────────────────────
    csv_file = output_dir / f'oracle_gloss2qwen_{args.split}.csv'
    
    csv_data = []
    for n in names:
        csv_data.append({
            'Video_ID': n,
            'Referencia': results[n]['txt_ref'],
            'Prediccion': results[n]['txt_hyp'],
            'Glosa_GT': results[n]['gls_ref'] # Esta vez la glosa predicha ES la Glosa Referencia
        })
        
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_file, index=False, encoding='utf-8')
    
    print(f"Predicciones listas para BLEURT guardadas en: {csv_file}")
    print(f"Resultados JSON guardados en: {output_file}")
    print("=" * 60)

    return summary

# ── entry point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Evitar deadlocks con tokenizadores HuggingFace
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    args = get_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    evaluate(args, config)