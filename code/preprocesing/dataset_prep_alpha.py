"""
dataset_preprocessing_alphapose.py
===================================
Preprocessing pipeline for PHOENIX-2014-T using AlphaPose Halpe-136 keypoints.

Keypoint index source:
    Halpe-FullBody official repo — https://github.com/Fang-Haoshu/Halpe-FullBody
    AlphaPose paper — https://arxiv.org/abs/2211.03375

Halpe-136 layout (per person, flat array of 136*3 = 408 values: x, y, conf):
    0-25  : 26 body keypoints
              0=Nose, 1=LEye, 2=REye, 3=LEar, 4=REar,
              5=LShoulder, 6=RShoulder, 7=LElbow, 8=RElbow,
              9=LWrist, 10=RWrist, 11=LHip, 12=RHip,
              13=LKnee, 14=RKnee, 15=LAnkle, 16=RAnkle,
              17=Head, 18=Neck, 19=Hip,
              20=LBigToe, 21=RBigToe, 22=LSmallToe, 23=RSmallToe,
              24=LHeel, 25=RHeel
    26-93 : 68 face keypoints  (NOT used — too noisy for SLT)
    94-114: 21 left hand keypoints
    115-135: 21 right hand keypoints

Selected keypoints (faithful to Kim et al. 2022 — upper body + both hands):
    FACE_IDX  = [0,1,2,3,4,17]    →  6 kp  (nose, eyes, ears, head)
    UPPER_IDX = [18,5,6,7,8,9,10] →  7 kp  (neck, shoulders, elbows, wrists)
    LARM_IDX  = [7,9]             →  2 kp  (lelbow, lwrist)  — for arm normalisation
    RARM_IDX  = [8,10]            →  2 kp  (relbow, rwrist)
    LHAND_IDX = 94-114            → 21 kp
    RHAND_IDX = 115-135           → 21 kp
    Total stacked: 6+7+2+2+21+21 = 59 kp × 2 coords = 118 dim  (INPUT_DIM=118)

JSON format (from createsyslinks.py step3_split_json):
    {
      "version": "alphapose_halpe136",
      "people": [{
        "pose_keypoints_2d": [x0,y0,c0, x1,y1,c1, ..., x135,y135,c135],  # 408 values
        "box": [...],
        "score": float
      }]
    }

Directory structure (mirrors OpenPose output, compatible with our pipeline):
    {alphapose_dir}/{split}/{video_name}/{frame_stem}_keypoints.json

Usage:
    python dataset_preprocessing_alphapose.py \\
        --train_video /path/to/alphapose_output/train \\
        --val_video   /path/to/alphapose_output/dev   \\
        --train_csv   /path/to/PHOENIX-2014-T.train.corpus.csv \\
        --val_csv     /path/to/PHOENIX-2014-T.dev.corpus.csv   \\
        --train_output phoenix_alphapose_train.pkl \\
        --val_output   phoenix_alphapose_val.pkl
"""

import os
import json
import argparse
import logging
import pickle
import random
import collections
import multiprocessing
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Halpe-136 keypoint index definitions
# Source: https://github.com/Fang-Haoshu/Halpe-FullBody
#     //26 body keypoints
#     {0,  "Nose"},
#     {1,  "LEye"},
#     {2,  "REye"},
#     {3,  "LEar"},
#     {4,  "REar"},
#     {5,  "LShoulder"},
#     {6,  "RShoulder"},
#     {7,  "LElbow"},
#     {8,  "RElbow"},
#     {9,  "LWrist"},
#     {10, "RWrist"},
#     {11, "LHip"},
#     {12, "RHip"},
#     {13, "LKnee"},
#     {14, "Rknee"},
#     {15, "LAnkle"},
#     {16, "RAnkle"},
#     {17,  "Head"},
#     {18,  "Neck"},
#     {19,  "Hip"},
#     {20, "LBigToe"},
#     {21, "RBigToe"},
#     {22, "LSmallToe"},
#     {23, "RSmallToe"},
#     {24, "LHeel"},
#     {25, "RHeel"},
#     //face
#     {26-93, 68 Face Keypoints}
#     //left hand
#     {94-114, 21 Left Hand Keypoints}
#     //right hand
#     {115-135, 21 Right Hand Keypoints}
# # ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Halpe-136 keypoint index definitions (Ajustado a 55 puntos = 110 dims)
# ---------------------------------------------------------------------------

# Face (6 puntos): Nose(0), LEye(1), REye(2), LEar(3), REar(4), Head(17)
FACE_IDX  = [0, 1, 2, 3, 4, 17]

# Upper body (3 puntos): Neck(18), LShoulder(5), RShoulder(6)
UPPER_IDX = [18, 5, 6]

# Arms (2 puntos cada uno): LElbow(7), LWrist(9) | RElbow(8), RWrist(10)
LARM_IDX  = [7, 9]
RARM_IDX  = [8, 10]

# Hands (21 puntos cada una)
LHAND_IDX = list(range(94, 115))
RHAND_IDX = list(range(115, 136))


# Resulting feature dimension
# Dimensiones: (6 + 3 + 2 + 2 + 21 + 21) = 55 kp × 2 coords = 110
KEYPOINT_SIZE = (len(FACE_IDX) + len(UPPER_IDX) + len(LARM_IDX) +
                 len(RARM_IDX) + len(LHAND_IDX) + len(RHAND_IDX)) * 2  # 118

FACE = True

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_TRAIN_VIDEO  = "/home/user/work/data/alphapose_fast_output/train"
DEFAULT_VAL_VIDEO    = "/home/user/work/data/alphapose_fast_output/dev"
DEFAULT_TRAIN_CSV    = "/home/user/work/data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv"
DEFAULT_VAL_CSV      = "/home/user/work/data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.dev.corpus.csv"
DEFAULT_TRAIN_PICKLE = "phoenix_alphapose_train.pkl"
DEFAULT_VAL_PICKLE   = "phoenix_alphapose_val.pkl"
DEFAULT_TOKENIZER    = "phoenix_tokenizer.pkl"
DEFAULT_LOG_DIR      = "./logs"
DEFAULT_NUM_WORKERS  = multiprocessing.cpu_count()
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

log = logging.getLogger("slt_preproc_ap")


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def setup_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = os.path.join(log_dir, f"preprocessing_ap_{timestamp}.log")

    logger = logging.getLogger("slt_preproc_ap")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    fh  = logging.FileHandler(log_path, encoding="utf-8")
    ch  = logging.StreamHandler()
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Log file: {log_path}")
    return logger


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class SignLanguageTokenizer:
    """
    Word-level tokenizer for German (Phoenix) or English (How2Sign).
    Language-agnostic — uses NLTK word_tokenize with language param.
    """
    def __init__(self, language="german"):
        self.language   = language
        self.word2idx   = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.idx2word   = {i: w for w, i in self.word2idx.items()}
        self.vocab_size = 4

    def build_vocab(self, sentences):
        words = []
        for sentence in sentences:
            words.extend(self.tokenize_text(sentence))
        counts = collections.Counter(words)
        for word, _ in counts.most_common():
            if word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1

    def tokenize_text(self, text):
        tokens = word_tokenize(text.lower(), language=self.language)
        return [t for t in tokens if t.isalnum()]

    def encode(self, sentence):
        """Variable-length — collate_fn handles padding per batch."""
        tokens  = self.tokenize_text(sentence)
        encoded = [self.word2idx["<sos>"]]
        encoded.extend([self.word2idx.get(t, self.word2idx["<unk>"]) for t in tokens])
        encoded.append(self.word2idx["<eos>"])
        return encoded


# ---------------------------------------------------------------------------
# Frame — extracts and normalises Halpe-136 keypoints
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Frame — extracts and normalises Halpe-136 keypoints
# ---------------------------------------------------------------------------

class Frame:
    """
    Procesa una detección AlphaPose Halpe-136.
    Lógica original (Kim et al.):
      - Cara: Origen en la Nariz.
      - Cuerpo: Origen en el Cuello.
      - Manos: Lógica original Min-Max (su propio centro de bounding box).
    """

    KEYPOINT_SIZE = KEYPOINT_SIZE + FACE*3 # 55 puntos * 2 + 3*FACE (distancias faciales)

    def __init__(self, json_path: str):
        with open(json_path, "r") as f:
            data = json.load(f)

        if not data.get("people"):
            raise ValueError(f"No person detected: {json_path}")

        # Pick highest-score detection
        person = max(data["people"], key=lambda p: p.get("score", 0.0))

        # Parse flat keypoints → (136, 3)
        kps = np.array(person["pose_keypoints_2d"], dtype=np.float32).reshape(136, 3)

        # Extract groups (x, y only)
        face  = kps[FACE_IDX,  :2]
        upper = kps[UPPER_IDX, :2]
        larm  = kps[LARM_IDX,  :2]
        rarm  = kps[RARM_IDX,  :2]
        lhand = kps[LHAND_IDX, :2]
        rhand = kps[RHAND_IDX, :2]

        # --- DEFINIR CENTROIDES INDEPENDIENTES ---
        centroid_face = kps[0,  :2]   # Punto 0: Nariz
        centroid_body = kps[18, :2]   # Punto 18: Cuello

        # --- DEFINIR REFERENCIAS PARA ESCALADO (Distancia 'd') ---
        ref_face  = kps[18, :2]  # Distancia Nariz-Cuello para escalar cara
        ref_upper = kps[0,  :2]  # Distancia Cuello-Nariz para escalar torso
        ref_larm  = kps[7,  :2]  # Distancia Cuello-Codo Izq para brazo izq
        ref_rarm  = kps[8,  :2]  # Distancia Cuello-Codo Der para brazo der

        # --- NORMALIZAR PASANDO EL CENTROIDE CORRESPONDIENTE ---
        face_norm  = self._normalize_body(face,  centroid_face, ref_face)
        upper_norm = self._normalize_body(upper, centroid_body, ref_upper)
        larm_norm  = self._normalize_body(larm,  centroid_body, ref_larm)
        rarm_norm  = self._normalize_body(rarm,  centroid_body, ref_rarm)

        # Las manos mantienen tu lógica original Min-Max (independiente por naturaleza)
        lhand_norm = self._normalize_hands(lhand)
        rhand_norm = self._normalize_hands(rhand)

        if FACE:
            d_face = get_dist(0, 18)
            # 1. Apertura de boca
            mouth_open = get_dist(77, 83) / d_face # d_face sería la dist nariz-cuello

            # 2. Cejas (Puntos 47 y 48 son los extremos internos de las cejas)
            brows_knit = get_dist(47, 48) / d_face

            # 3. Elevación de cejas (Puntos 71 y 0 - Ceja a Nariz)
            brows_up = (get_dist(47, 0) + get_dist(48, 0)) / (2 * d_face)

            # Creamos un pequeño vector de "features faciales" (4, )
            facial_features = np.array([mouth_open, brows_knit, brows_up], dtype=np.float32)

            # stack final
            self.keypoints_flat = np.concatenate([
                face_norm.flatten(),
                upper_norm.flatten(),
                larm_norm.flatten(),
                rarm_norm.flatten(),
                lhand_norm.flatten(),
                rhand_norm.flatten(),
                facial_features])
        else:
            # Stack → (55, 2) → flatten → (110,)
            self.keypoints_flat = np.vstack([
                face_norm,
                upper_norm,
                larm_norm,
                rarm_norm,
                lhand_norm,
                rhand_norm,
            ]).flatten()

    def _normalize_body(self, keypoints: np.ndarray, centroid: np.ndarray, reference: np.ndarray) -> np.ndarray:
        # Calculamos 'd' usando el centroide específico que le hemos pasado
        d = np.linalg.norm(centroid - reference)
        if d < 1e-6:
            return np.zeros_like(keypoints)
        # Restamos el centroide específico para llevar ese punto exacto al (0,0)
        return (keypoints - centroid) / d

    def _normalize_hands(self, keypoints: np.ndarray) -> np.ndarray:
        v_max = keypoints.max(axis=0)
        v_min = keypoints.min(axis=0)
        d = v_max - v_min
        d = np.where(d < 1e-6, 1e-6, d)
        # Nota: Al restar v_min y dividir por d, las manos ya se anclan matemáticamente
        # al (0,0) de su propio bounding box de forma independiente al cuerpo.
        return (keypoints - v_min) / d - 0.5

    # --- EXTRACCIÓN DE MICRO-EXPRESIONES (Morfemas no manuales) ---
    # Usamos los puntos originales de kps antes de normalizar para consistencia
    def get_dist(p1_idx, p2_idx):
        return np.linalg.norm(kps[p1_idx, :2] - kps[p2_idx, :2])



# ---------------------------------------------------------------------------
# Video — loads all frames from a directory
# ---------------------------------------------------------------------------

class Video:
    """
    Loads all AlphaPose JSON files from a video directory in sorted order.
    Skips corrupt/empty frames with a warning instead of crashing.
    Raises RuntimeError if no valid frames are found.
    """

    def __init__(self, path: str):
        self.path   = path
        self.frames = []

        json_files = sorted([
            f for f in os.listdir(path)
            if f.endswith("_keypoints.json")
        ])

        for fname in json_files:
            fpath = os.path.join(path, fname)
            try:
                self.frames.append(Frame(fpath).keypoints_flat)
            except Exception as e:
                log.warning(f"  Skipping frame {fname}: {e}")

        if not self.frames:
            raise RuntimeError(f"No valid frames in '{path}'")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PhoenixAlphaPoseDataset(Dataset):
    """
    PyTorch Dataset for PHOENIX-2014-T with AlphaPose Halpe-136 keypoints.

    Each sample:
        visual_input: (N, 110/113) float32 — SASS-resampled keypoint frames
        text_target:  variable-length int64 — tokenised German translation
        video_name:   str
    """

    V_STAR = KEYPOINT_SIZE  + 3*FACE

    def __init__(self, sentences: dict, video_keypoints: dict,
                 tokenizer: SignLanguageTokenizer,
                 N: int = 150, training: bool = True):
        valid   = {k: v for k, v in sentences.items() if k in video_keypoints}
        dropped = len(sentences) - len(valid)
        if dropped:
            log.warning(f"  Dropped {dropped} entries with no keypoint data.")

        self.video_names    = list(valid.keys())
        self.sentences      = list(valid.values())
        self.video_kp       = video_keypoints
        self.tokenizer      = tokenizer
        self.N              = N
        self.training       = training

    def __len__(self):
        return len(self.video_names)

    def apply_sass(self, frames: list) -> np.ndarray:
        """
        SASS — Kim et al. 2022, eq. 4.
        T > N: skip-sample N evenly spaced frames (deterministic).
        T < N: duplicate stochastically (train) or linearly (eval).
        """
        T = len(frames)
        if T > self.N:
            indices = np.linspace(0, T - 1, self.N, dtype=int)
        elif T < self.N:
            if self.training:
                indices = np.sort(np.random.choice(T, self.N, replace=True))
            else:
                indices = np.linspace(0, T - 1, self.N, dtype=int)
        else:
            return np.array(frames)
        return np.array([frames[i] for i in indices])

    def __getitem__(self, idx):
        video_name   = self.video_names[idx]
        sentence     = self.sentences[idx]
        frames_array = self.apply_sass(self.video_kp[video_name])

        assert frames_array.shape == (self.N, self.V_STAR), (
            f"Shape mismatch '{video_name}': "
            f"expected ({self.N},{self.V_STAR}), got {frames_array.shape}"
        )

        tokens = self.tokenizer.encode(sentence)
        return {
            "visual_input": torch.tensor(frames_array, dtype=torch.float32),
            "text_target":  torch.tensor(tokens,       dtype=torch.long),
            "video_name":   video_name,
        }


# ---------------------------------------------------------------------------
# load_videos — parallel, one process per video
# ---------------------------------------------------------------------------

def _process_video(entry_path: str) -> tuple:
    """Top-level worker — must be at module level for ProcessPoolExecutor."""
    video_name = os.path.basename(entry_path)
    frames     = Video(entry_path).frames
    return video_name, frames


def load_videos(video_dir: str, num_workers: int = DEFAULT_NUM_WORKERS) -> dict:
    """Load all video subdirectories in parallel."""
    entries = [
        e.path for e in sorted(os.scandir(video_dir), key=lambda e: e.name)
        if e.is_dir()
    ]
    if not entries:
        log.warning(f"No video sub-folders found in {video_dir}")
        return {}

    num_workers = min(num_workers, len(entries))
    log.info(f"  {len(entries)} videos, {num_workers} workers ...")

    videos, skipped = {}, 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_path = {executor.submit(_process_video, p): p for p in entries}
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            name = os.path.basename(path)
            try:
                video_name, frames = future.result()
                videos[video_name] = frames
                log.info(f"  ✓ '{video_name}' ({len(frames)} frames)")
            except Exception as e:
                log.warning(f"  ✗ Skipping '{name}': {e}")
                skipped += 1

    log.info(f"  → {len(videos)} loaded, {skipped} skipped")
    return videos


# ---------------------------------------------------------------------------
# compute_N — paper eq. 5
# ---------------------------------------------------------------------------

def compute_N(videos: dict) -> int:
    """N = round( (1/L) * Σ x_i ) — Kim et al. 2022, eq. 5."""
    counts = [len(f) for f in videos.values()]
    N = int(round(sum(counts) / len(counts)))
    log.info(f"  Frame counts — min:{min(counts)}  max:{max(counts)}  mean:{N}")
    return N


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess PHOENIX-2014-T with AlphaPose Halpe-136."
    )
    parser.add_argument("--train_video",      default=DEFAULT_TRAIN_VIDEO)
    parser.add_argument("--val_video",        default=DEFAULT_VAL_VIDEO)
    parser.add_argument("--train_csv",        default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--val_csv",          default=DEFAULT_VAL_CSV)
    parser.add_argument("--train_output",     default=DEFAULT_TRAIN_PICKLE)
    parser.add_argument("--val_output",       default=DEFAULT_VAL_PICKLE)
    parser.add_argument("--tokenizer_output", default=DEFAULT_TOKENIZER)
    parser.add_argument("--num_workers",      type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--log_dir",          default=DEFAULT_LOG_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    args = parse_args()
    log  = setup_logger(args.log_dir)

    log.info("=== PHOENIX AlphaPose Halpe-136 Preprocessing ===")
    log.info(f"  KEYPOINT_SIZE : {KEYPOINT_SIZE}  (INPUT_DIM in model_train.py)")
    log.info(f"  Face  idx     : {FACE_IDX}")
    log.info(f"  Upper idx     : {UPPER_IDX}")
    log.info(f"  LArm  idx     : {LARM_IDX}")
    log.info(f"  RArm  idx     : {RARM_IDX}")
    log.info(f"  LHand idx     : 94-114")
    log.info(f"  RHand idx     : 115-135")

    # 1. Load videos
    log.info("--- Loading TRAIN videos ---")
    train_videos = load_videos(args.train_video, args.num_workers)
    log.info("--- Loading VAL videos ---")
    val_videos   = load_videos(args.val_video,   args.num_workers)

    # 2. Load CSVs (pipe-separated — Phoenix format)
    log.info("--- Loading annotations ---")
    train_csv = pd.read_csv(args.train_csv, sep="|")
    val_csv   = pd.read_csv(args.val_csv,   sep="|")
    log.info(f"  Train CSV rows: {len(train_csv)}")
    log.info(f"  Val   CSV rows: {len(val_csv)}")

    # Phoenix CSV columns: name|video|start|end|speaker|orth|translation
    train_sentences = dict(zip(train_csv["name"], train_csv["translation"]))
    val_sentences   = dict(zip(val_csv["name"],   val_csv["translation"]))

    # 3. Compute N from train only (eq. 5)
    log.info("--- Computing N (eq. 5) ---")
    N = compute_N(train_videos)
    log.info(f"  N = {N} frames")

    # 4. Tokenizer — train translations only, no leakage
    log.info("--- Building vocabulary ---")
    tokenizer = SignLanguageTokenizer(language="german")
    tokenizer.build_vocab(train_csv["translation"])
    log.info(f"  Vocab size: {tokenizer.vocab_size}")
    with open(args.tokenizer_output, "wb") as f:
        pickle.dump(tokenizer, f)
    log.info(f"  Tokenizer saved → {args.tokenizer_output}")

    # 5. Build datasets
    log.info("--- Building train dataset ---")
    train_dataset = PhoenixAlphaPoseDataset(
        train_sentences, train_videos, tokenizer, N=N, training=True
    )
    log.info(f"  Train samples: {len(train_dataset)}")
    with open(args.train_output, "wb") as f:
        pickle.dump(train_dataset, f)
    log.info(f"  Saved → {args.train_output}")

    log.info("--- Building val dataset ---")
    val_dataset = PhoenixAlphaPoseDataset(
        val_sentences, val_videos, tokenizer, N=N, training=False
    )
    log.info(f"  Val samples: {len(val_dataset)}")
    with open(args.val_output, "wb") as f:
        pickle.dump(val_dataset, f)
    log.info(f"  Saved → {args.val_output}")

    # 6. Summary
    log.info("=== Done ===")
    log.info(f"  Train     : {len(train_dataset)} → {args.train_output}")
    log.info(f"  Val       : {len(val_dataset)}   → {args.val_output}")
    log.info(f"  Tokenizer : {tokenizer.vocab_size} tokens → {args.tokenizer_output}")
    log.info(f"  N         : {N} frames")
    log.info(f"  INPUT_DIM : {KEYPOINT_SIZE}  ← update in model_train.py")
