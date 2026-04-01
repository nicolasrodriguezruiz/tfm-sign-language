import logging
import math
import os
import pickle
import random
import time
import csv
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from torch.nn.utils.rnn import pad_sequence

from dataset_prep_alpha import PhoenixAlphaPoseDataset, SignLanguageTokenizer

# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------
SEED = 48
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark    = False
torch.backends.cudnn.deterministic = True

# ---------------------------------------------------------------------------
# Hyperparameters (Actualizados para Transformer)
# ---------------------------------------------------------------------------
FACE = True
INPUT_DIM    = 110 + 3*FACE  # Ajustar según tu extracción de distancias faciales
D_MODEL      = 256           # Dimensión interna del Transformer
N_HEADS      = 8
N_LAYERS     = 3             # Encoder/Decoder layers
FF_DIM       = 1024          # Feedforward dim
DROPOUT      = 0.1
BATCH_SIZE   = 32
N_EPOCHS     = 100
CLIP         = 1
LR           = 0.0001        # Transformers requieren un LR más bajo que las RNN
MIN_LR       = 1e-7
TRAIN_PICKLE = "./phoenix_alphapose_train.pkl"
VAL_PICKLE   = "./phoenix_alphapose_val.pkl"
SAVE_PATH    = "./models_transformer"
LOG_DIR      = "./logs_transformer"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Componentes del Modelo Transformer
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class FeatureExtractor(nn.Module):
    """Giro Moderno: Conv1D para capturar dinámica local antes del Transformer."""
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        # x: [B, T, INPUT_DIM] -> Conv1d espera [B, F, T]
        x = x.transpose(1, 2)
        x = self.conv(x)
        return x.transpose(1, 2) # [B, T, D_MODEL]

class HybridSignTransformer(nn.Module):
    def __init__(self, input_dim, target_vocab_size, d_model, nhead, num_layers, ff_dim, dropout):
        super().__init__()
        self.d_model = d_model

        # 1. Hybrid Input: Conv + Positional
        self.feat_extractor = FeatureExtractor(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # 2. Text Embedding
        self.tgt_embedding = nn.Embedding(target_vocab_size, d_model)

        # 3. Transformer Core
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )

        self.fc_out = nn.Linear(d_model, target_vocab_size)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt, src_pad_mask=None, tgt_pad_mask=None):
        # Procesar Keypoints
        src_feats = self.feat_extractor(src)
        src_emb = self.pos_encoder(src_feats * math.sqrt(self.d_model))

        # Procesar Texto
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)

        # Máscara causal para el Decoder
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(src.device)

        output = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask
        )
        return self.fc_out(output)

# ---------------------------------------------------------------------------
# Collate & Utils
# ---------------------------------------------------------------------------

def collate_fn(batch):
    visual = torch.stack([item["visual_input"] for item in batch])
    texts  = pad_sequence(
        [item["text_target"] for item in batch],
        batch_first=True, padding_value=0
    )
    return {"visual_input": visual, "text_target": texts}

def epoch_time(start, end):
    e = end - start
    return int(e / 60), int(e % 60)

# ---------------------------------------------------------------------------
# train / evaluate (Adaptados para Transformer)
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, clip, pad_idx):
    model.train()
    epoch_loss = 0
    for batch in loader:
        src = batch["visual_input"].to(device)
        trg = batch["text_target"].to(device)

        # Transformer input/output shift
        tgt_input = trg[:, :-1]
        tgt_expected = trg[:, 1:]

        # Masks
        src_pad_mask = (src.sum(dim=-1) == 0) # Ignora frames vacíos
        tgt_pad_mask = (tgt_input == pad_idx)

        optimizer.zero_grad()
        output = model(src, tgt_input, src_pad_mask, tgt_pad_mask)

        # Reshape para pérdida: [B*T, Vocab]
        output = output.reshape(-1, output.shape[-1])
        tgt_expected = tgt_expected.reshape(-1)

        loss = criterion(output, tgt_expected)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(loader)

@torch.no_grad()
def evaluate_loss(model, loader, criterion, pad_idx):
    model.eval()
    epoch_loss = 0
    for batch in loader:
        src = batch["visual_input"].to(device)
        trg = batch["text_target"].to(device)
        tgt_input = trg[:, :-1]
        tgt_expected = trg[:, 1:]

        src_pad_mask = (src.sum(dim=-1) == 0)
        tgt_pad_mask = (tgt_input == pad_idx)

        output = model(src, tgt_input, src_pad_mask, tgt_pad_mask)
        output = output.reshape(-1, output.shape[-1])
        tgt_expected = tgt_expected.reshape(-1)

        epoch_loss += criterion(output, tgt_expected).item()
    return epoch_loss / len(loader)

@torch.no_grad()
def bleu_evaluate(model, loader, tokenizer, device, max_len=50):
    """Traducción Greedy para el cálculo de BLEU."""
    model.eval()
    references = []
    hypotheses = []

    sos_idx = tokenizer.word2idx["<sos>"]
    eos_idx = tokenizer.word2idx["<eos>"]
    pad_idx = tokenizer.word2idx["<pad>"]

    for batch in loader:
        src = batch["visual_input"].to(device)
        trg = batch["text_target"].to(device)

        for i in range(src.size(0)):
            # Reference
            ref_ids = trg[i, 1:].tolist()
            ref_words = []
            for idx in ref_ids:
                if idx in (eos_idx, pad_idx): break
                ref_words.append(tokenizer.idx2word.get(idx, "<unk>"))
            references.append([ref_words])

            # Greedy Decoding (Hypothesis)
            single_src = src[i:i+1] # [1, T, F]
            ys = torch.ones(1, 1).fill_(sos_idx).type(torch.long).to(device)

            for _ in range(max_len):
                src_mask = (single_src.sum(dim=-1) == 0)
                out = model(single_src, ys, src_mask, None)
                prob = out[0, -1]
                next_word = prob.argmax().item()
                if next_word == eos_idx: break
                ys = torch.cat([ys, torch.ones(1, 1).type(torch.long).to(device).fill_(next_word)], dim=1)

            hyp_words = [tokenizer.idx2word.get(idx.item(), "<unk>") for idx in ys[0, 1:]]
            hypotheses.append(hyp_words)

    smooth = SmoothingFunction().method1
    bleu = corpus_bleu(references, hypotheses, smoothing_function=smooth)
    return bleu * 100

# ---------------------------------------------------------------------------
# Setup & Logger
# ---------------------------------------------------------------------------

def setup_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = os.path.join(log_dir, f"training_{timestamp}.log")
    logger = logging.getLogger("slt_transformer")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    ch = logging.StreamHandler()
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log = setup_logger(LOG_DIR)
    os.makedirs(SAVE_PATH, exist_ok=True)

    log.info("Cargando datasets para Transformer...")
    with open(TRAIN_PICKLE, "rb") as f: train_dataset = pickle.load(f)
    with open(VAL_PICKLE, "rb") as f: val_dataset = pickle.load(f)

    tokenizer = train_dataset.tokenizer
    OUTPUT_DIM = tokenizer.vocab_size
    PAD_IDX = tokenizer.word2idx["<pad>"]

    train_loader = D.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = D.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Construir Modelo
    model = HybridSignTransformer(INPUT_DIM, OUTPUT_DIM, D_MODEL, N_HEADS, N_LAYERS, FF_DIM, DROPOUT).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    csv_file = os.path.join(LOG_DIR, f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train_Loss", "Val_Loss", "BLEU"])

    best_bleu = 0
    log.info(f"Iniciando entrenamiento híbrido (Transformer + Conv1D). Params: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, CLIP, PAD_IDX)
        val_loss = evaluate_loss(model, val_loader, criterion, PAD_IDX)
        bleu = bleu_evaluate(model, val_loader, tokenizer, device)

        scheduler.step()
        mins, secs = epoch_time(t0, time.time())

        if bleu > best_bleu:
            best_bleu = bleu
            torch.save(model.state_dict(), os.path.join(SAVE_PATH, "transformer_best_bleu.pt"))
            log.info(f"Epoch {epoch} | ¡Nuevo récord BLEU!: {bleu:.3f}")

        log.info(f"Epoch {epoch:02d} | {mins}m {secs}s | Loss T: {train_loss:.3f} V: {val_loss:.3f} | BLEU: {bleu:.2f}")

        with open(csv_file, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, bleu])

    log.info(f"Entrenamiento finalizado. Mejor BLEU alcanzado: {best_bleu:.3f}")
