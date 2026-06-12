import os
import json
import subprocess
import sys

# SECRETOS wandb Key
WANDB_API_KEY= ''
try:
	with open("key.txt", "r", encoding="utf-8") as f:
	    WANDB_API_KEY = f.read().strip()
	os.environ["WANDB_API_KEY"] = WANDB_API_KEY
except:
	print("[INFO] WANDB_API_KEY no encontrada")

# Configuracion
PROGRESS_FILE = ".SLR_ablation_progress.json"

# Rutas a todos tus archivos de configuracion del ablation study
CONFIG_FILES = [
    "configs/abalation/SLR_base.yaml",
    "configs/abalation/SLR_Jdropout.yaml",
    "configs/abalation/SLR_noise.yaml",
    "configs/abalation/SLR_noise_dropout.yaml",
]

def load_progress():
    """Carga la lista de configuraciones que ya han terminado correctamente."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_progress(completed_configs):
    """Guarda la lista de configuraciones completadas."""
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(completed_configs, f, indent=4)

def main():
    completed_configs = load_progress()
    total = len(CONFIG_FILES)

    print(f"[INFO] Iniciando Ablation Study ({total} configuraciones encontradas)")

    for i, config in enumerate(CONFIG_FILES, 1):
        if config in completed_configs:
            print(f"[{i}/{total}] Saltando {config}: Ya fue completado en una ejecucion anterior.")
            continue

        print(f"\n[{i}/{total}] Iniciando entrenamiento con: {config}")
        print("-" * 60)

        command = ["python", "train.py", "--config", config]

        try:
            subprocess.run(command, check=True)

            # Si llegamos aqui, train.py termino con codigo de salida 0 (sin errores)
            completed_configs.append(config)
            save_progress(completed_configs)

            print("-" * 60)
            print(f"[INFO] Entrenamiento finalizado exitosamente para: {config}")

        except subprocess.CalledProcessError as e:
            # Si train.py lanza una excepcion (OOM, error de sintaxis, etc.), entra aqui
            print("-" * 60)
            print(f"[ERROR] El entrenamiento fallo para {config} (Codigo: {e.returncode}).")
            print("[INFO] Deteniendo el orquestador. Corrige el error y vuelve a lanzar este script.")
            sys.exit(1)

        except KeyboardInterrupt:
            # Ctrl+C, detiene el script limpiamente sin marcar el actual como completado
            print("\n" + "-" * 60)
            print("[INFO] Ejecucion interrumpida por el usuario (Ctrl+C).")
            print(f"[INFO] El progreso hasta antes de {config} ha sido guardado.")
            sys.exit(0)

    print("\n[INFO] Ablation Study completado..")

if __name__ == "__main__":
    main()
