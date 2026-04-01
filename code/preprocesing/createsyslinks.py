import os
import json
import subprocess
import concurrent.futures

# --- RUTAS PRINCIPALES ---
FRAMES_DIR = "/home/user/work/data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px"
OUTPUT_DIR = "/home/user/work/data/alphapose_fast_output"
ALPHAPOSE_DIR = "/home/user/work/Preprocessing/aplhapose/AlphaPose"
SYMLINK_DIR = "/dev/shm/work_data/temp_links"

def _link_video_frames(video_path, split, video_name):
    """Función 'worker' que procesa todos los frames de un único vídeo."""
    count = 0
    for frame_name in os.listdir(video_path):
        if not frame_name.lower().endswith(('.png', '.jpg', '.jpeg')): 
            continue
        
        src = os.path.join(video_path, frame_name)
        dst_name = f"{split}___{video_name}___{frame_name}"
        dst = os.path.join(SYMLINK_DIR, dst_name)
        
        if not os.path.exists(dst):
            os.symlink(src, dst)
        count += 1
    return count

def step1_create_symlinks():
    print("=== PASO 1: Creando enlaces simbólicos (Multihilo) ===")
    os.makedirs(SYMLINK_DIR, exist_ok=True)
    
    # Preparar la lista de todos los vídeos a procesar
    video_tasks = []
    for split in ["train", "dev", "test"]:
        split_dir = os.path.join(FRAMES_DIR, split)
        if not os.path.isdir(split_dir): continue
        
        for video_name in os.listdir(split_dir):
            video_path = os.path.join(split_dir, video_name)
            if os.path.isdir(video_path):
                video_tasks.append((video_path, split, video_name))
                
    total_frames = 0
    # Lanzar la creación en paralelo usando 16 hilos (ajusta si tienes más/menos núcleos)
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        # Enviar las tareas al pool
        futures = [
            executor.submit(_link_video_frames, path, split, name) 
            for path, split, name in video_tasks
        ]
        
        # Recolectar resultados conforme terminan
        for future in concurrent.futures.as_completed(futures):
            total_frames += future.result()
            
    print(f"Listo. {total_frames} frames enlazados virtualmente a la velocidad de la luz.")

def step2_run_alphapose():
    print("\n=== PASO 2: Ejecutando AlphaPose (Delegando al script oficial) ===")
    os.chdir(ALPHAPOSE_DIR)
    
    # Llamada nativa a consola: a prueba de errores de CUDA multiprocessing
    cmd = [
        "python", "scripts/demo_inference.py",
        "--cfg", "configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml",
        "--checkpoint", "pretrained_models/halpe136_fast50_dcn_combined_256x192_10handweight.pth",
        "--indir", SYMLINK_DIR,
        "--outdir", OUTPUT_DIR,
        "--detbatch", "4",
        "--posebatch", "32"
    ]
    subprocess.run(cmd, check=True)

def step3_split_json():
    print("\n=== PASO 3: Separando resultados a sus carpetas finales ===")
    results_json = os.path.join(OUTPUT_DIR, "alphapose-results.json")
    
    if not os.path.exists(results_json):
        print("ERROR: No se encontró el alphapose-results.json. ¿Falló AlphaPose?")
        return
        
    with open(results_json, 'r') as f:
        data = json.load(f)
        
    frames_dict = {}
    for person in data:
        img_id = person["image_id"]  # ej: train___videoName___images0001.png
        if img_id not in frames_dict:
            frames_dict[img_id] = []
        
        frames_dict[img_id].append({
            "pose_keypoints_2d": person["keypoints"],
            "box": person["box"],
            "score": person["score"]
        })
        
    for img_id, people in frames_dict.items():
        parts = img_id.split("___")
        if len(parts) != 3: continue
        
        split, video_name, frame_name = parts
        frame_stem = os.path.splitext(frame_name)[0]
        
        # Reconstruir la ruta original
        final_dir = os.path.join(OUTPUT_DIR, split, video_name)
        os.makedirs(final_dir, exist_ok=True)
        
        out_file = os.path.join(final_dir, f"{frame_stem}_keypoints.json")
        with open(out_file, 'w') as f:
            json.dump({"version": "alphapose_halpe136", "people": people}, f)
            
    print(f"¡Éxito total! Generados {len(frames_dict)} archivos JSON individuales.")
    
    # Limpieza final
    os.remove(results_json)
    print("Archivo temporal gigante eliminado.")

if __name__ == "__main__":
    step1_create_symlinks()
    step2_run_alphapose()
    step3_split_json()
