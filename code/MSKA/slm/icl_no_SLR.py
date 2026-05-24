import torch
import random
import utils
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------

def build_chat_messages(examples, test_gloss):
    """
    Construye un historial de mensajes para el Chat Template de Qwen.
    Usa los ejemplos few-shot como interacciones previas entre usuario y asistente.
    """
    # 1. Mensaje de sistema completamente en alemán con regla de escape
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
    
    # 2. Inyectar los ejemplos few-shot como historial de chat
    for gloss, text in examples:
        messages.append({"role": "user", "content": f"Glossen: {gloss}"})
        messages.append({"role": "assistant", "content": text})
        
    # 3. La consulta actual que el modelo debe resolver
    messages.append({"role": "user", "content": f"Glossen: {test_gloss}"})
    
    return messages

# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

# Cargar datos
print("Cargando datasets...")
train_data = utils.load_dataset_file('/home/user/work/Preprocessing/MSKA/Dataset/data/Phoenix-2014T.train.pkl')
test_data  = utils.load_dataset_file('/home/user/work/Preprocessing/MSKA/Dataset/data/Phoenix-2014T.test.pkl')

# Seleccionar K ejemplos del train como contexto (fijando semilla)
random.seed(42)
examples = random.sample([
    (v['gloss'], v['text']) for v in train_data.values()
    if v.get('gloss') and v.get('text')
], k=5)

# Cargar Qwen
print("Cargando modelo Qwen2.5-1.5B...")
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B', trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-1.5B',
    torch_dtype=torch.bfloat16,
    device_map='cuda',
    trust_remote_code=True,
)
model.eval()

# Generar traducciones
print("Generando traducciones...")
hypotheses, references = [], []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with torch.no_grad():
    for step, sample in enumerate(test_data.values()):
        if step % 10 == 0:
            print(f" Procesando muestra {step}/{len(test_data)}")

        # 1. Obtener la lista de mensajes
        messages = build_chat_messages(examples, sample['gloss'])

        # 2. Aplicar el chat template nativo de Qwen
        prompt_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True 
        )

        # 3. Tokenizar el prompt estructurado
        inputs = tokenizer(
            prompt_text,
            return_tensors='pt',
            truncation=True,
            max_length=1024,
        ).to(device)

        # 4. Generar usando parámetros optimizados para LLMs
        output_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,          # Cambiado de num_beams a muestreo
            temperature=0.3,         # Baja temperatura para precisión
            top_p=0.9,
            repetition_penalty=1.05, # Penalización suave de repetición
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        # 5. Extraer solo la parte generada (sin el prompt)
        prompt_len = inputs['input_ids'].shape[1]
        generated_ids = output_ids[0][prompt_len:]
        
        generated = tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()
        
        # Limpieza de seguridad por si escapa saltos de línea o siglas finales
        import re
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
