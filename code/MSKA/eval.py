import os
import subprocess
import json
from pathlib import Path
import yaml

def main():
    experimentos = [
        {
            "config": "temp2/main/configs/abalation/SLT_SLM_LoRA_Att.yaml",
            "resume": "/home/user/work/data/outputs_final/SLR_base/best_checkpoint.pth"
        },
        {
            "config": "configs/abalation/SLR_base_SJD.yaml",
            "resume": "/home/user/work/data/outputs_final/SLR_base_SJD/best_checkpoint.pth"
        },
        {
            "config": "configs/abalation/SLR_Jdropout.yaml",
            "resume": "/home/user/work/data/outputs_final/SLR_Jdropout/best_checkpoint.pth"
        },
        {
            "config": "configs/abalation/SLR_noise_1.yaml",
            "resume": "/home/user/work/data/outputs_final/SLR_noise_1/best_checkpoint.pth"
        },
    ]

    archivo_salida_global = "resumen_todas_las_evaluaciones_SLT.json"
    resultados_globales = []

    print(f"Iniciando evaluación en bucle para {len(experimentos)} configuraciones...\n")

    for idx, exp in enumerate(experimentos):
        config_path = exp["config"]
        ckpt_path = exp["resume"]
        mBart = exp.get("mBART", False)

        print("="*60)
        print(f"Ejecutando [{idx+1}/{len(experimentos)}]: {config_path}")
        print("="*60)

        # Construir el comando de terminal
        comando = [
            "python", "train.py",
            "--config", config_path,
            "--resume", ckpt_path,
            "--eval",
        ]
        if not mBart:
            comando.append("--slm")

        # Ejecutar el subproceso y esperar a que termine
        subprocess.run(comando)

        # Leer la carpeta de salida desde el yaml para buscar el JSON generado
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

        output_dir = Path(cfg['training']['model_dir'])
        eval_json_path = output_dir / 'eval_results.json'

        # Si el subproceso terminó bien, leer su JSON y añadirlo a la lista global
        if eval_json_path.exists():
            with open(eval_json_path, 'r', encoding='utf-8') as f:
                datos = json.load(f)
                # Le añadimos un nombre identificativo para que quede bonito
                datos['nombre_experimento'] = config_path.split('/')[-1]
                resultados_globales.append(datos)
        else:
            print(f"Error: No se generó el archivo {eval_json_path}")

    # Guardar el JSON final
    with open(archivo_salida_global, 'w', encoding='utf-8') as f:
        json.dump(resultados_globales, f, indent=4)

    print("\n" + "★"*60)
    print(f"¡Todas las evaluaciones terminadas! Resumen guardado en: {archivo_salida_global}")
    print("★"*60)

if __name__ == "__main__":
    main()
