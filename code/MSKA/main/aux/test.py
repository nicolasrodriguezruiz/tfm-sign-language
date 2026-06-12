from transformers import AutoTokenizer

# 1. Cargamos el tokenizador exacto que estás usando
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)

# 2. Comprobamos el ID del token de fin de texto (eos_token)
eos_id = tokenizer.eos_token_id
eos_text = tokenizer.eos_token

print(f"Token de fin configurado por defecto: {eos_text} -> ID: {eos_id}")

# 3. Comprobamos cuántos tokens hay en total en su vocabulario
vocab_size = len(tokenizer)
print(f"Tamaño total del vocabulario preentrenado: {vocab_size}")

# 4. Buscamos explícitamente el token <|endoftext|> en el diccionario
# convert_tokens_to_ids devolverá el ID numérico si existe, o el ID de <unk> (desconocido) si no.
specific_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
print(f"ID del token literal '<|endoftext|>': {specific_token_id}")