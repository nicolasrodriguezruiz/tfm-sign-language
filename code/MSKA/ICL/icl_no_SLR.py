import torch
import random
import yaml
import utils
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Carga de Configuración
# ---------------------------------------------------------------------------
def load_config(config_path='configs/config_icl_noSLR.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------
def build_chat_messages(examples, test_gloss):
    """
    Construye un historial de mensajes para el Chat Template de Qwen.
    Usa los ejemplos few-shot como interacciones previas entre usuario y asistente.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Du bist ein erfahrener Übersetzer für die Deutsche Gebärdensprache (DGS). "
                "Deine Aufgabe ist es, Glossen (Schlüsselwörter in Großbuchstaben) "
                "in natürliches, fließendes Deutsch zu übersetzen. "
                "Wenn die Liste der Glossen leer ist oder keinen Sinn ergibt, "
                "antworte einfach mit 'Keine Übersetzung möglich'."
            )
        }
    ]
    
    for gloss, text in examples:
        messages.append({"role": "user", "content": f"Glossen: {gloss}"})
        messages.append({"role": "assistant", "content": text})
        
    messages.append({"role": "user", "content": f"Glossen: {test_gloss}"})
    
    return messages

# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def main():
    # Cargar config
    print("Cargando configuración...")
    cfg = load_config('configs/config_icl_noSLR.yaml')
    
    # Cargar datos desde las rutas del config
    print("Cargando datasets...")
    train_data = utils.load_dataset_file(cfg['data']['train_label_path'])
    test_data  = utils.load_dataset_file(cfg['data']['test_label_path'])

    # Parámetros del experimento
    fs_cfg = cfg.get('few_shot_g2t', {})
    k_examples = fs_cfg.get('num_examples', 5)
    random.seed(fs_cfg.get('seed', 42))

    # Seleccionar K ejemplos del train como contexto
    examples = random.sample([
        (v['gloss'], v['text']) for v in train_data.values()
        if v.get('gloss') and v.get('text')
    ], k=k_examples)

    # Cargar modelo desde el config
    qwen_path = cfg['model']['Qwen']['pretrained_model_name_or_path']
    print(f"Cargando modelo {qwen_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        qwen_path,
        torch_dtype=torch.bfloat16,
        device_map='cuda',
        trust_remote_code=True,
    )
    model.eval()

    print("Generando traducciones...")
    hypotheses, references = [], []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        for step, sample in enumerate(test_data.values()):
            if step % 10 == 0:
                print(f" Procesando muestra {step}/{len(test_data)}")

            messages = build_chat_messages(examples, sample['gloss'])

            prompt_text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True 
            )

            inputs = tokenizer(
                prompt_text,
                return_tensors='pt',
                truncation=True,
                max_length=1024,
            ).to(device)

            # Extraemos los parámetros de generación del YAML
            gen_kwargs = fs_cfg.get('generation', {})
            
            # Generar usando los parámetros desempaquetados automáticamente
            output_ids = model.generate(
                **inputs,
                **gen_kwargs,  # <-- LA MAGIA ESTÁ AQUÍ
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

            prompt_len = inputs['input_ids'].shape[1]
            generated_ids = output_ids[0][prompt_len:]
            
            generated = tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ).strip()
            
            # Limpieza
            generated = generated.split('\n')[0].strip()
            generated = re.sub(r'\s+[A-Z]{2,}$', '', generated).strip()

            hypotheses.append(generated)
            references.append(sample['text'])

    # Calcular métricas
    from metrics import bleu, rouge
    print("\n--- Resultados Finales ---")
    bleu_scores = bleu(references=references, hypotheses=hypotheses, level='word')
    for k, v in bleu_scores.items():
        print(f"{k}: {v:.2f}")
        
    rouge_score = rouge(references=references, hypotheses=hypotheses, level='word')
    print(f"ROUGE: {rouge_score:.2f}")

if __name__ == "__main__":
    main()