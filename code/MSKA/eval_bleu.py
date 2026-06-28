import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

import evaluate
import pandas as pd

def main():
    csv_path = 'resultados_test_s2t.csv'
    df = pd.read_csv(csv_path)

    # Rellenar posibles NaNs por si el modelo falló generando alguna línea
    df["Prediccion"] = df["Prediccion"].fillna("")

    bleurt = evaluate.load("bleurt", "BLEURT-20")
    resultados = bleurt.compute(predictions=df["Prediccion"].tolist(), references=df["Referencia"].tolist())

    score_final = sum(resultados["scores"]) / len(resultados["scores"])

    df["BLEURT_Score"] = resultados["scores"]
    df.to_csv(csv_path, index=False, encoding='utf-8')

    print(f" SCORE BLEURT GLOBAL: {score_final:.4f}")

if __name__ == "__main__":
    main()
