"""
evaluate_gloss_baseline.py
--------------------------
Baseline: glosas predichas por SLR → Qwen (sin finetuning) → texto alemán

Pipeline:
    keypoints → Recognition (pesos S2G preentrenados)
              → CTC decode → glosas predichas (string)
              → prompt_str + glosas → Qwen.generate()
              → BLEU-1/2/3/4 + ROUGE

NO usa VLMapper ni features visuales. Solo texto de glosas como entrada.

Uso:
    python evaluate_gloss_baseline.py \
        --config    configs/phoenix14t_s2t.yaml \
        --slr_ckpt  /ruta/a/best_checkpoint_slr.pth \
        --qwen_model Qwen/Qwen2.5-1.5B \
        --split     test \
        --batch_size 8 \

    # Evaluar en dev:
    python evaluate_gloss_baseline.py \
        --config configs/phoenix14t_s2t.yaml \
        --slr_ckpt /ruta/a/best_checkpoint_slr.pth \
        --split dev
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
from collections import defaultdict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

from Recognition.Tokenizer import GlossTokenizer_S2G
from Recognition.recognition import Recognition
from aux.metrics import bleu, rouge, wer_list
from aux.phoenix_cleanup import clean_phoenix_2014_trans
import aux.utils as utils


import warnings
warnings.filterwarnings("ignore")
# ── args ────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',     type=str, required=True)
    p.add_argument('--slr_ckpt',   type=str, required=True,
                   help='Checkpoint del recognition network (S2G)')
    p.add_argument('--qwen_model', type=str, default=None,
                   help='Ruta o nombre de Qwen. Por defecto usa el del config.')
    p.add_argument('--split',      type=str, default='test',
                   choices=['dev', 'test'])
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--beam_size',  type=int, default=4)
    p.add_argument('--max_new_tokens', type=int, default=100)
    p.add_argument('--num_workers',    type=int, default=4)
    p.add_argument('--seed',       type=int, default=0)
    p.add_argument('--output_dir', type=str, default="./",
                   help='Dónde guardar resultados JSON. Por defecto: model_dir del config.')
    p.add_argument('--device', default='cuda')
    return p.parse_args()


# ── carga del Recognition network ───────────────────────────────────────────

def load_recognition(config, slr_ckpt_path, device):
    """Carga solo el Recognition network con pesos S2G."""
    recognition = Recognition(
        cfg=config['model']['RecognitionNetwork'],
        args=argparse.Namespace(),
    )

    ckpt = torch.load(slr_ckpt_path, map_location='cpu')

    # Los checkpoints pueden venir con prefijo 'model.' o 'recognition_network.'
    state = ckpt.get('model', ckpt)
    # Quitar prefijos comunes
    cleaned = {}
    for k, v in state.items():
        new_k = k
        for prefix in ('recognition_network.', 'module.recognition_network.', 'module.'):
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
        cleaned[new_k] = v

    missing, unexpected = recognition.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[WARN] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    recognition.to(device)
    recognition.eval()
    for p in recognition.parameters():
        p.requires_grad = False

    print(f"Recognition cargado desde: {slr_ckpt_path}")
    return recognition


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
    # beam_size=4,
    # max_new_tokens=51,
    gen_kwargs,
):
    """
    Construye: prompt_str + glosas → Qwen.generate() → texto alemán.
    Las glosas se pasan como texto plano (mismo formato que usa el sistema).
    """
    # Construir los inputs de texto: prompt + glosas
    #inputs_text = [prompt_str + g for g in gloss_strings]
    #inputs_text = [f"{prompt_str}{g}<|im_end|>\n<|im_start|>assistant\n" for g in gloss_strings]
    inputs_text = []
    for g in gloss_strings:
        # Usamos el formato nativo de mensajes para modelos de chat
        messages = [
            {"role": "system", "content": prompt_str},
            {"role": "user", "content": g}
        ]
        # El tokenizador se encarga de poner los <|im_start|> y <|im_end|> de forma perfecta
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
            # max_new_tokens=max_new_tokens,
            # num_beams=beam_size,
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
    # Importación tardía para no requerir Qwen en el import
    if args.split == 'test':
        from slm.S2T_Dataset import S2T_Dataset
        data_path = config['data']['test_label_path']
    else:
        from slm.S2T_Dataset import S2T_Dataset
        data_path = config['data']['dev_label_path']

    tokenizer_gls = GlossTokenizer_S2G(config['gloss'])

    # Forzamos task=S2G para que el dataset no intente cargar Qwen
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

    # ── Recognition ───────────────────────────────────────────────────────────
    recognition = load_recognition(config, args.slr_ckpt, device)

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
    dataset_name = config['data']['dataset_name'].lower()

    print(f"\nIniciando evaluación baseline (glosas → Qwen)...\n")


    # Extraer parámetros de generación desde el config YAML
    test_cfg = config.get('testing', {})

    # Parámetros para el decodificador CTC (reconocimiento)
    rec_beam_size = test_cfg.get('recognition', {}).get('beam_size', args.beam_size)

    # Parámetros para Qwen (traducción)
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

    with torch.no_grad():
        for step, batch in enumerate(
            metric_logger.log_every(dataloader, print_freq=20, header=f'[{args.split}]')
        ):
            # Reconocimiento → glosas predichas
            rec_out = recognition(batch)

            # Usar el ensemble (mejor opción disponible)
            if 'ensemble_last_gloss_logits' in rec_out:
                gls_logits = rec_out['ensemble_last_gloss_logits']
            else:
                # fallback a fuse_head
                gls_logits = rec_out['fuse_gloss_logits']

            ctc_decoded = recognition.decode(
                gloss_logits=gls_logits,
                beam_size=rec_beam_size,
                input_lengths=rec_out['input_lengths'],
            )

            # Convertir IDs → strings de glosas
            # pred_gls_tokens = tokenizer_gls.convert_ids_to_tokens(ctc_decoded)
            # pred_gls_strings = [
            #     (' '.join(g).upper() if tokenizer_gls.lower_case else ' '.join(g))
            #     for g in pred_gls_tokens
            # ]

            # Convertir IDs → tokens de glosas  y limpiar UNK
            pred_gls_tokens = tokenizer_gls.convert_ids_to_tokens(ctc_decoded)

            # --- Limpiar tokens <UNK> (y otros tokens especiales residuales) ---
            cleaned_gls_tokens = [
                [t for t in g if t.upper() not in ['<UNK>', '<PAD>', '<BOS>', '<EOS>']]
                for g in pred_gls_tokens
            ]

            # Unir los tokens limpios en strings
            pred_gls_strings = [
                (' '.join(g).upper() if tokenizer_gls.lower_case else ' '.join(g))
                for g in cleaned_gls_tokens
            ]

            # Glosas → Qwen → texto
            txt_hyp = generate_from_glosses(
                gloss_strings=pred_gls_strings,
                prompt_str=prompt_str,
                qwen_tokenizer=qwen_tokenizer,
                qwen_model=qwen_model,
                device=device,
                gen_kwargs=gen_kwargs,
            )

            for name, hyp, ref, gls_pred, gls_ref in zip(
                batch['name'], txt_hyp, batch['text'],
                pred_gls_strings, batch['gloss'],
            ):
                results[name] = {
                    'txt_hyp': hyp,
                    'txt_ref': ref,
                    'gls_pred': gls_pred,
                    'gls_ref':  gls_ref.upper() if tokenizer_gls.lower_case else gls_ref,
                }

    # ── Calcular métricas ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MÉTRICAS BASELINE (glosas predichas → Qwen base)")
    print("=" * 60)

    names     = list(results.keys())
    txt_refs  = [results[n]['txt_ref']  for n in names]
    txt_hyps  = [results[n]['txt_hyp']  for n in names]
    gls_refs  = [results[n]['gls_ref']  for n in names]
    gls_preds = [results[n]['gls_pred'] for n in names]

    # WER de glosas (para saber qué calidad de reconocimiento entra a Qwen)
    if dataset_name == 'phoenix-2014t':
        gls_refs_clean  = [clean_phoenix_2014_trans(r) for r in gls_refs]
        gls_preds_clean = [clean_phoenix_2014_trans(p) for p in gls_preds]
    else:
        gls_refs_clean  = gls_refs
        gls_preds_clean = gls_preds

    wer_results = wer_list(references=gls_refs_clean, hypotheses=gls_preds_clean)

    # BLEU + ROUGE sobre la traducción
    level      = config['data'].get('level', 'word')
    bleu_dict  = bleu(references=txt_refs, hypotheses=txt_hyps, level=level)
    rouge_score = rouge(references=txt_refs, hypotheses=txt_hyps, level=level)

    print(f"\n── Reconocimiento de glosas ──")
    print(f"  WER:         {wer_results['wer']:.2f}%")
    print(f"  DEL:         {wer_results['del_rate']:.2f}%")
    print(f"  INS:         {wer_results['ins_rate']:.2f}%")
    print(f"  SUB:         {wer_results['sub_rate']:.2f}%")

    print(f"\n── Traducción (glosas → Qwen base) ──")
    for k in ['bleu1', 'bleu2', 'bleu3', 'bleu4']:
        print(f"  {k.upper()}:  {bleu_dict[k]:.2f}")
    print(f"  ROUGE-L:     {rouge_score:.2f}")

    # ── Guardar resultados ─────────────────────────────────────────────────────
    output_dir = Path(args.output_dir or config['training']['model_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'baseline_gloss2qwen_{args.split}.json'

    summary = {
        'split': args.split,
        'slr_ckpt': args.slr_ckpt,
        'qwen_model': qwen_model_name,
        'prompt': prompt_str,
        'n_samples': len(results),
        'recognition': {
            'wer':      round(wer_results['wer'],      2),
            'del_rate': round(wer_results['del_rate'], 2),
            'ins_rate': round(wer_results['ins_rate'], 2),
            'sub_rate': round(wer_results['sub_rate'], 2),
        },
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
                'gls_pred': results[n]['gls_pred'],
                'txt_ref':  results[n]['txt_ref'],
                'txt_hyp':  results[n]['txt_hyp'],
            }
            for n in names
        ],
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ── Guardar resultados en CSV ─────────────────────────
    csv_file = output_dir / f'baseline_gloss2qwen_{args.split}.csv'

    csv_data = []
    for n in names:
        csv_data.append({
            'Video_ID': n,
            'Referencia': results[n]['txt_ref'],
            'Prediccion': results[n]['txt_hyp'],
            'Glosas_Predichas': results[n]['gls_pred'] # Útil para depurar después
        })

    df = pd.DataFrame(csv_data)
    df.to_csv(csv_file, index=False, encoding='utf-8')


    print(f"Predicciones listas para BLEURT guardadas en: {csv_file}")
    #
    print(f"\nResultados guardados en: {output_file}")
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
