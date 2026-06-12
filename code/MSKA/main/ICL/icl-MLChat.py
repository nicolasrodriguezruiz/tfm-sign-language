"""
icl_evaluation.py
-----------------
Evalúa la traducción de lengua de señas usando:
  1. Recognition network entrenado (DSTA + VisualHead + CTC) para obtener glosas.
  2. Qwen2.5-1.5B con in-context learning (sin entrenar) para traducir las glosas.

No requiere entrenamiento del translator. Sirve como baseline para comparar
con el pipeline end-to-end (VLMapper + LoRA).

Uso:
    python icl_evaluation.py \
        --config    configs/phoenix-2014t_s2t.yaml \
        --ckpt      /path/to/Phoenix-2014T_SLR/best_checkpoint.pth \
        --num_shots 5 \
        --beam_size 4 \
        --output    resultados_icl.json
"""

import torch
import argparse
import yaml
import json
import random
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from itertools import groupby

import utils
from datasets import S2T_Dataset
from Tokenizer import GlossTokenizer_S2G
from model import SignLanguageModel
from metrics import wer_list, bleu, rouge
from phoenix_cleanup import clean_phoenix_2014_trans

import time

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# ---------------------------------------------------------------------------
# Construcción del prompt few-shot
# ---------------------------------------------------------------------------
def build_chat_messages(examples, test_gloss):
    """
    Construye un historial de mensajes para el Chat Template.
    Usa los ejemplos few-shot como interacciones previas entre usuario y asistente.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Du bist ein erfahrener Übersetzer für die Deutsche Gebärdensprache (DGS). "
                "Deine Aufgabe ist es, Glossen (Schlüsselwörter in Großbuchstaben) "
                "in natürliches, fließendes Deutsch zu übersetzen."
            )
        }
    ]
    
    # 2. Historial de ejemplos (cambiamos "Glosas" por "Glossen")
    for gloss, text in examples:
        messages.append({"role": "user", "content": f"Glossen: {gloss}"})
        messages.append({"role": "assistant", "content": text})
        
    # 3. La consulta actual
    messages.append({"role": "user", "content": f"Glossen: {test_gloss}"})
    
    return messages


def select_examples(train_data, num_shots, seed=42):
    """
    Selecciona num_shots ejemplos del train set para usar como contexto.
    Se fija la semilla para reproducibilidad entre experimentos.
    """
    random.seed(seed)
    samples = [
        (v['gloss'], v['text'])
        for v in train_data.values()
        if v.get('gloss') and v.get('text')
    ]
    return random.sample(samples, k=min(num_shots, len(samples)))


# ---------------------------------------------------------------------------
# Decodificación CTC (igual que en recognition.py pero standalone)
# ---------------------------------------------------------------------------

def decode_gloss_ids(gloss_ids, gloss_tokenizer):
    """
    Convierte una lista de IDs de glosas a string,
    eliminando repeticiones consecutivas y tokens de silencio/padding.
    """
    # Eliminar repeticiones consecutivas (comportamiento CTC)
    collapsed = [x[0] for x in groupby(gloss_ids)]
    # Filtrar tokens especiales (silencio=0, pad)
    filtered = [
        g for g in collapsed
        if g != 0 and g != gloss_tokenizer.pad_id
    ]
    # Convertir IDs a strings de glosas
    tokens = gloss_tokenizer.convert_ids_to_tokens(filtered)
    return ' '.join(tokens).upper()


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_icl_evaluation(args, config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    # --- 1. Cargar datos ---
    print("Cargando datasets...")
    gloss_tokenizer = GlossTokenizer_S2G(config['gloss'])

    # Train: solo para seleccionar ejemplos few-shot
    train_raw = utils.load_dataset_file(config['data']['train_label_path'])
    examples  = select_examples(train_raw, num_shots=args.num_shots, seed=args.seed)
    print(f"Ejemplos few-shot seleccionados: {len(examples)}")
    for g, t in examples:
        print(f"  Glosas: {g[:50]}...")
        print(f"  Trad:   {t[:50]}...")
        print()

    # Dev y test para evaluación
    dev_data = S2T_Dataset(
        path=config['data']['dev_label_path'],
        tokenizer=gloss_tokenizer,
        config=config, args=args, phase='val',
    )
    test_data = S2T_Dataset(
        path=config['data']['test_label_path'],
        tokenizer=gloss_tokenizer,
        config=config, args=args, phase='test',
    )

    dev_loader = DataLoader(
        dev_data, batch_size=args.batch_size,
        num_workers=4, collate_fn=dev_data.collate_fn,
    )
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size,
        num_workers=4, collate_fn=test_data.collate_fn,
    )

    # --- 2. Cargar recognition network ---
    # Solo cargamos el recognition, no el translator (eso es Qwen ICL)
    print("Cargando recognition network...")
    # Modificamos el config temporalmente para cargar solo S2G
    s2g_config = {**config, 'task': 'S2G'}
    recognition_model = SignLanguageModel(cfg=s2g_config, args=args)

    checkpoint = torch.load(args.ckpt, map_location='cpu')
    # El checkpoint puede tener el modelo completo o solo el recognition
    state_dict = checkpoint.get('model', checkpoint)
    # Filtrar solo las claves del recognition
    recognition_dict = {
        k.replace('recognition_network.', ''): v
        for k, v in state_dict.items()
        if 'recognition_network' in k
    }
    recognition_model.recognition_network.load_state_dict(recognition_dict, strict=True)
    recognition_model.to(device)
    recognition_model.eval()
    # Congelar todos los parámetros
    for param in recognition_model.parameters():
        param.requires_grad = False
    print("Recognition network cargado.")

    # --- 3. Cargar Qwen para ICL ---
    print(f"Cargando Qwen ({args.qwen_model})...")
    qwen_tokenizer = AutoTokenizer.from_pretrained(
        args.qwen_model, trust_remote_code=True
    )
    if qwen_tokenizer.pad_token is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token

    qwen_model = AutoModelForCausalLM.from_pretrained(
        args.qwen_model,
        torch_dtype=torch.bfloat16,
        device_map='cuda',
        trust_remote_code=True,
    )
    qwen_model.eval()
    # Congelar todos los parámetros
    for param in qwen_model.parameters():
        param.requires_grad = False
    print("Qwen cargado.")

    # Token de parada: salto de línea (la traducción termina ahí)
    newline_token_id = qwen_tokenizer.encode('\n', add_special_tokens=False)[0]

    # ---------------------------------------------------------------------------
    # Función de evaluación sobre un dataloader
    # ---------------------------------------------------------------------------

    def evaluate_split(loader, split_name):
        results = defaultdict(dict)

        with torch.no_grad():
            for step, src_input in enumerate(loader):
                if step % 10 == 0:
                    print(f"  [{split_name}] Batch {step}/{len(loader)}")

                # --- Paso 1: Recognition → glosas predichas ---
                recognition_output = recognition_model.recognition_network(src_input)

                # Decodificar con CTC beam search (usando la cabeza fuse)
                ctc_decoded = recognition_model.recognition_network.decode(
                    gloss_logits=recognition_output['fuse_gloss_logits'],
                    beam_size=args.beam_size,
                    input_lengths=recognition_output['input_lengths'],
                )
                # Convertir IDs a strings de glosas
                pred_gloss_batch = gloss_tokenizer.convert_ids_to_tokens(ctc_decoded)

                # --- Paso 2: Qwen ICL → traducción ---
                # --- Paso 2: Qwen ICL → traducción ---
                for name, pred_gloss_ids, ref_gloss, ref_text in zip(src_input['name'],
                                                                     pred_gloss_batch,
                                                                     src_input['gloss'],
                                                                     src_input['text'],
                                                                    ):
                    pred_gloss_str = ' '.join(pred_gloss_ids).upper()

                    # 1. Obtener la lista de mensajes
                    messages = build_chat_messages(examples, pred_gloss_str)

                    # 2. Aplicar el chat template nativo de Qwen
                    # add_generation_prompt=True inyecta el inicio del turno del asistente para que empiece a generar
                    prompt_text = qwen_tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True 
                    )

                    # 3. Tokenizar el prompt estructurado
                    # (Aumenta un poco max_length porque los templates añaden tokens extra)
                    inputs = qwen_tokenizer(
                        prompt_text,
                        return_tensors='pt',
                        truncation=True,
                        max_length=1024,
                    ).to(device)

                    # 4. Generar
                    output_ids = qwen_model.generate(
                        **inputs,
                            max_new_tokens=args.max_new_tokens,
                            # num_beams=args.beam_size y usa esto en su lugar:
                            do_sample=True,
                            temperature=0.3, # Temperatura baja para que sea preciso y no alucine
                            top_p=0.9,
                            repetition_penalty=1.05,
                            eos_token_id=qwen_tokenizer.eos_token_id,
                            pad_token_id=qwen_tokenizer.pad_token_id,
                        )

                    # Extraer solo los tokens generados (sin el prompt)
                    prompt_len = inputs['input_ids'].shape[1]
                    generated_ids = output_ids[0][prompt_len:]
                    translation = qwen_tokenizer.decode(
                        generated_ids, skip_special_tokens=True
                    ).strip()

                    # (Opcional pero recomendado por seguridad) Cortar si hubiera saltos de línea extra
                    translation = translation.split('\n')[0].strip()

                    # Guardar resultados
                    results[name]['txt_hyp']  = translation
                    results[name]['txt_ref']  = ref_text
                    results[name]['gls_hyp']  = pred_gloss_str
                    results[name]['gls_ref']  = ref_gloss
                    results[name]['prompt']   = prompt_text

        # --- Calcular métricas ---
        print(f"\n--- Métricas {split_name} ---")

        # WER sobre glosas
        gls_ref = [clean_phoenix_2014_trans(results[n]['gls_ref']) for n in results]
        gls_hyp = [clean_phoenix_2014_trans(results[n]['gls_hyp']) for n in results]
        wer_results = wer_list(hypotheses=gls_hyp, references=gls_ref)
        print(f"WER: {wer_results['wer']:.2f}")

        # BLEU y ROUGE sobre traducciones
        txt_ref = [results[n]['txt_ref'] for n in results]
        txt_hyp = [results[n]['txt_hyp'] for n in results]
        bleu_dict   = bleu(references=txt_ref, hypotheses=txt_hyp, level=config['data']['level'])
        rouge_score = rouge(references=txt_ref, hypotheses=txt_hyp, level=config['data']['level'])

        for k, v in bleu_dict.items():
            print(f"{k}: {v:.2f}")
        print(f"ROUGE: {rouge_score:.2f}")

        return {
            'wer':     wer_results['wer'],
            'bleu1':   bleu_dict['bleu1'],
            'bleu2':   bleu_dict['bleu2'],
            'bleu3':   bleu_dict['bleu3'],
            'bleu4':   bleu_dict['bleu4'],
            'rouge':   rouge_score,
            'results': dict(results),
        }

    # --- 4. Evaluar en dev y test ---
    print("\nEvaluando en DEV...")
    dev_metrics = evaluate_split(dev_loader, 'DEV')

    print("\nEvaluando en TEST...")
    test_metrics = evaluate_split(test_loader, 'TEST')

    # --- 5. Guardar resultados ---
    output = {
        'config': {
            'num_shots':    args.num_shots,
            'beam_size':    args.beam_size,
            'qwen_model':   args.qwen_model,
            'ckpt':         args.ckpt,
            'examples':     examples,
        },
        'dev':  {k: v for k, v in dev_metrics.items()  if k != 'results'},
        'test': {k: v for k, v in test_metrics.items() if k != 'results'},
        'dev_results':  dev_metrics['results'],
        'test_results': test_metrics['results'],
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResultados guardados en: {args.output}")

    # Resumen final
    print("\n========== RESUMEN ==========")
    print(f"DEV  — WER: {dev_metrics['wer']:.2f}  BLEU-4: {dev_metrics['bleu4']:.2f}  ROUGE: {dev_metrics['rouge']:.2f}")
    print(f"TEST — WER: {test_metrics['wer']:.2f}  BLEU-4: {test_metrics['bleu4']:.2f}  ROUGE: {test_metrics['rouge']:.2f}")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser('ICL evaluation with Recognition + Qwen')
    parser.add_argument('--config',       required=True,  help='Ruta al fichero YAML de config')
    parser.add_argument('--ckpt',         required=True,  help='Checkpoint del recognition network')
    parser.add_argument('--qwen_model',   default='Qwen/Qwen2.5-1.5B', help='Modelo Qwen a usar')
    parser.add_argument('--num_shots',    default=5,   type=int, help='Número de ejemplos few-shot')
    parser.add_argument('--beam_size',    default=4,   type=int, help='Tamaño del beam search')
    parser.add_argument('--max_new_tokens', default=60, type=int, help='Máximo tokens a generar')
    parser.add_argument('--length_penalty', default=1.0, type=float)
    parser.add_argument('--batch_size',   default=8,   type=int)
    parser.add_argument('--seed',         default=42,  type=int, help='Semilla para ejemplos few-shot')
    parser.add_argument('--output',       default='icl_results.json', help='Fichero de salida JSON')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    start_time = time.time()
    run_icl_evaluation(args, config)
    end_time = time.time()
    minutes, seconds = divmod(end_time - start_time, 60)
    print(f"Hemos tardado: {int(minutes)}m {seconds:.2f}s")