"""
generate_pruneids.py
--------------------
Genera el fichero pruneids.pkl necesario para el TextTokenizer en modo sentencepiece.

¿Qué es pruneids?
    MBart tiene un vocabulario de ~250k tokens (multilingüe).
    Phoenix14t solo usa alemán, así que la mayoría de esos tokens nunca aparecen.
    pruneids es un diccionario {id_original_mbart: id_reducido} que mapea
    solo los tokens que realmente aparecen en el dataset, reduciendo la capa
    de clasificación final del decoder y acelerando el entrenamiento.

    Los tokens especiales (<pad>, <s>, </s>, <unk>) se mantienen en sus
    índices originales para no romper la lógica del modelo.

Uso:
    python generate_pruneids.py \
        --mbart_model  path/to/mbart-large-cc25 \
        --phoenix_path path/to/phoenix14t \
        --output       path/to/pruneids.pkl \
        --tgt_lang     de_DE

    Los ficheros de texto de Phoenix14t deben estar en:
        {phoenix_path}/PHOENIX-2014-T.{train,dev,test}.corpus.csv
    con una columna 'translation' que contiene el texto en alemán.
"""

import pickle
import argparse
from transformers import MBartTokenizer


# ---------------------------------------------------------------------------
# Tokens especiales que deben conservar su índice original
# ---------------------------------------------------------------------------
SPECIAL_TOKENS = ['<pad>', '<s>', '</s>', '<unk>']


def load_phoenix_texts(phoenix_path: str) -> list:
    """
    Carga todas las frases en alemán de los splits train, dev y test de Phoenix14t.
    Se usan los tres splits para cubrir todo el vocabulario del dataset.
    """
    import csv
    import os

    texts = []
    for split in ['train', 'dev', 'test']:
        csv_path = os.path.join(
            phoenix_path,
            f'PHOENIX-2014-T.{split}.corpus.csv'
        )
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='|')
            for row in reader:
                texts.append(row['translation'])

    print(f"Total de frases cargadas: {len(texts)}")
    return texts


def generate_pruneids(mbart_model: str, phoenix_path: str,
                      output: str, tgt_lang: str = 'de_DE') -> None:
    """
    Genera el fichero pruneids.pkl escaneando todos los textos del dataset
    y reteniendo solo los tokens de MBart que aparecen en ellos.

    Pasos:
      1. Tokenizar todos los textos del dataset con MBart.
      2. Recopilar el conjunto de IDs que aparecen al menos una vez.
      3. Añadir siempre los tokens especiales y el token de idioma.
      4. Construir el mapeo {id_original: id_reducido} respetando que
         los tokens especiales mantengan su índice original.
      5. Guardar en disco como pickle.
    """
    print(f"Cargando tokenizador MBart desde: {mbart_model}")
    tokenizer = MBartTokenizer.from_pretrained(mbart_model, tgt_lang=tgt_lang)

    # --- Paso 1 y 2: tokenizar y recopilar IDs usados ---
    print("Tokenizando textos del dataset...")
    texts = load_phoenix_texts(phoenix_path)
    used_ids = set()

    with tokenizer.as_target_tokenizer():
        for text in texts:
            ids = tokenizer.encode(text)
            used_ids.update(ids)

    print(f"Tokens únicos encontrados en el dataset: {len(used_ids)}")

    # --- Paso 3: añadir tokens especiales e idioma siempre ---
    for special in SPECIAL_TOKENS:
        used_ids.add(tokenizer.convert_tokens_to_ids(special))
    # Token del idioma destino (necesario como inicio del decoder)
    used_ids.add(tokenizer.convert_tokens_to_ids(tgt_lang))

    # --- Paso 4: construir el mapeo ---
    # Los tokens especiales mantienen su índice original (requisito del modelo).
    # El resto se reindexan de forma contigua a partir del siguiente índice libre.
    special_ids = {tokenizer.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS}
    non_special_ids = sorted(used_ids - special_ids)

    # Encontrar el primer índice libre después de los especiales
    max_special_id = max(special_ids)
    next_id = max_special_id + 1

    pruneids = {}

    # Tokens especiales: id_reducido == id_original
    for id_ in special_ids:
        pruneids[id_] = id_

    # Resto de tokens: ids contiguos a partir de next_id
    for id_ in non_special_ids:
        if id_ not in pruneids:  # por si algún especial colisiona
            pruneids[id_] = next_id
            next_id += 1

    reduced_vocab_size = len(pruneids)
    print(f"Vocabulario original MBart: {len(tokenizer)}")
    print(f"Vocabulario reducido:       {reduced_vocab_size}")
    print(f"Reducción:                  {(1 - reduced_vocab_size / len(tokenizer)) * 100:.1f}%")

    # --- Verificación de tokens especiales ---
    for t in SPECIAL_TOKENS:
        id_ = tokenizer.convert_tokens_to_ids(t)
        assert pruneids[id_] == id_, f"Token especial {t}: {id_} → {pruneids[id_]} (debería ser {id_})"
    print("Verificación de tokens especiales: OK")

    # --- Paso 5: guardar ---
    with open(output, 'wb') as f:
        pickle.dump(pruneids, f)
    print(f"pruneids guardado en: {output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Genera pruneids.pkl para Phoenix14t')
    parser.add_argument('--mbart_model',  required=True,
                        help='Ruta o nombre del modelo MBart (p.ej. facebook/mbart-large-cc25)')
    parser.add_argument('--phoenix_path', required=True,
                        help='Directorio raíz del dataset Phoenix14t')
    parser.add_argument('--output',       required=True,
                        help='Ruta de salida para pruneids.pkl')
    parser.add_argument('--tgt_lang',     default='de_DE',
                        help='Código de idioma destino para MBart (default: de_DE)')
    args = parser.parse_args()

    generate_pruneids(
        mbart_model=args.mbart_model,
        phoenix_path=args.phoenix_path,
        output=args.output,
        tgt_lang=args.tgt_lang,
    )
