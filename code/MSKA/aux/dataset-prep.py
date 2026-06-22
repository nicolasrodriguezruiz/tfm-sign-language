import os
import json
import glob
import torch
import pickle
import logging
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

# ==========================================
# CONFIGURACIÓN DEL LOGGING
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(processName)s: %(message)s',
    handlers=[
        logging.FileHandler("dataset_builder.log"), # Guarda logs en archivo
        logging.StreamHandler()                     # Muestra logs en consola
    ]
)
logger = logging.getLogger(__name__)

# ==========================================
# FUNCIÓN DE EXTRACCIÓN POR VIDEO (Worker)
# ==========================================
def process_video(args):
    """
    Procesa un solo video. Lee todos sus JSON frame por frame,
    extrae el esqueleto principal y devuelve un diccionario.
    """
    video_name, video_folder, gloss, text = args

    # Buscamos y ordenamos alfabéticamente los JSON para mantener el orden temporal
    json_paths = sorted(glob.glob(os.path.join(video_folder, "*keypoints*.json")))

    if not json_paths:
        logger.warning(f"No se encontraron archivos JSON en {video_folder}. Saltando...")
        return None

    video_keypoints = []
    last_valid_keypoints = np.zeros((136, 3), dtype=np.float32) # Fallback si no hay persona

    for json_file in json_paths:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # AlphaPose guarda las detecciones en la lista 'people'
            people = data.get('people', [])

            if len(people) == 0:
                # Si en este frame no se detectó a nadie, usamos el del frame anterior
                # para no romper la dimensión temporal de la red neuronal.
                video_keypoints.append(last_valid_keypoints)
                continue

            # Si hay varias personas, seleccionamos la principal (la de mayor confianza media)
            best_person = None
            max_conf = -1

            for person in people:
                # AlphaPose exporta en una lista plana: [x1,y1,c1, x2,y2,c2, ...]
                kp_list = person.get('pose_keypoints_2d', [])
                if not kp_list:
                    continue

                # Convertimos a matriz (136 puntos, 3 canales: X, Y, Confianza)
                kp_array = np.array(kp_list, dtype=np.float32).reshape(-1, 3)

                # Calculamos la confianza promedio de este sujeto
                avg_conf = np.mean(kp_array[:, 2])

                if avg_conf > max_conf:
                    max_conf = avg_conf
                    best_person = kp_array

            if best_person is not None:
                video_keypoints.append(best_person)
                last_valid_keypoints = best_person
            else:
                video_keypoints.append(last_valid_keypoints)

        except Exception as e:
            logger.error(f"Error leyendo {json_file}: {e}")
            video_keypoints.append(last_valid_keypoints) # Usar fallback en caso de error de I/O

    # Convertimos la lista de matrices a un Tensor de PyTorch
    # Forma resultante: (Tiempo, Vértices, Canales) -> (T, 136, 3)
    tensor_keypoints = torch.from_numpy(np.stack(video_keypoints))
    num_frames = tensor_keypoints.shape[0]

    return {
        "name": video_name,
        "keypoint": tensor_keypoints,
        "gloss": gloss,
        "text": text,
        "num_frames": num_frames
    }

# ==========================================
# FLUJO PRINCIPAL
# ==========================================
def main():
    # ---------------------------------------------------------
    # RUTAS DE LOS DATOS
    # ---------------------------------------------------------
    # El CSV de Phoenix suele tener un delimitador de barra vertical '|'
    ANNOTATIONS_CSV = "/home/user/work/data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv"
    BASE_JSON_FOLDER = "/home/user/work/data/alphapose_fast_output/train"
    OUTPUT_PKL = "data/Phoenix-2014T.train.pkl"

    # ---------------------------------------------------------
    logger.info("Cargando anotaciones (Glosas y Texto)...")
    try:
        df = pd.read_csv(ANNOTATIONS_CSV, sep='|')
    except Exception as e:
        logger.critical(f"No se pudo cargar el CSV de anotaciones: {e}")
        return

    # Preparamos los argumentos para el multiprocesamiento
    tasks = []
    logger.info("Emparejando videos con sus anotaciones...")
    for index, row in df.iterrows():
        video_name = row['name'] # Ej: '01April_2010_Thursday_heute_default-1'
        gloss = row['orth']      # Ej: 'ICH WETTER HEUTE'
        text = row['translation'] # Ej: 'El tiempo hoy.'

        video_folder = os.path.join(BASE_JSON_FOLDER, video_name)

        if os.path.isdir(video_folder):
            tasks.append((video_name, video_folder, gloss, text))
        else:
            logger.warning(f"Carpeta no encontrada para el video: {video_name}")

    total_videos = len(tasks)
    logger.info(f"Se procesarán {total_videos} videos usando {cpu_count()} núcleos.")

    # ---------------------------------------------------------
    # PROCESAMIENTO EN PARALELO
    # ---------------------------------------------------------
    dataset_dict = {}

    # Usamos todos los núcleos disponibles
    with Pool(processes=cpu_count()) as pool:
        # imap_unordered es más rápido y eficiente con la memoria en Linux
        for i, result in enumerate(pool.imap_unordered(process_video, tasks), 1):
            if result is not None:
                video_name = result['name']
                # Recreamos la estructura exacta que MSKA espera:
                dataset_dict[video_name] = result

            # Log de progreso cada 100 videos
            if i % 100 == 0 or i == total_videos:
                logger.info(f"Progreso: {i}/{total_videos} videos procesados.")

    # ---------------------------------------------------------
    # GUARDADO DEL DATASET
    # ---------------------------------------------------------
    os.makedirs(os.path.dirname(OUTPUT_PKL), exist_ok=True)
    logger.info(f"Guardando dataset empaquetado en {OUTPUT_PKL}...")

    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(dataset_dict, f)

    logger.info(f"¡Proceso completado! Tamaño final del dataset: {len(dataset_dict)} videos.")

if __name__ == "__main__":
    main()
