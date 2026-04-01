"""
Sign Language Translation — Seq2Seq GRU + Bahdanau Attention
=============================================================
Basado en:
  model/seq2seq_gru_attention.py  (GRU_AT_Encoder, Attention, GRU_AT_Decoder, GRU_AT_Seq2Seq)
  train.py
  utils/train_utils.py
enlace https://github.com/winston1214/Sign-Language-project

"""

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
# Hyperparameters
# ---------------------------------------------------------------------------
FACE = True
INPUT_DIM    = 110 + 3*FACE
HID_DIM      = 512
EMB_DIM      = 128
N_LAYERS     = 2
DEC_DROPOUT  = 0.5
BATCH_SIZE   = 32
N_EPOCHS     = 100
CLIP         = 1
LR           = 0.001
MIN_LR       = 1e-6
TRAIN_PICKLE = "./phoenix_alphapose_train.pkl"   # preprocessed train split
VAL_PICKLE   = "./phoenix_alphapose_val.pkl"     # preprocessed val split
SAVE_PATH    = "./models"   # directory for saved .pt files and log
LOG_DIR      = "./logs"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

class GRU_AT_Encoder(nn.Module):
    def __init__(self, input_size, hid_dim, n_layers):
        super().__init__()
        self.hid_dim  = hid_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(
            input_size, hid_dim, n_layers,
            batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hid_dim * 2, hid_dim)

    def forward(self, x):
        # x: (B, T, INPUT_DIM)
        h0 = torch.zeros(
            self.n_layers * 2, x.size(0), self.hid_dim
        ).to(device).float()

        out, hidden = self.gru(x, h0)
        # hidden: (n_layers*2, B, hid_dim) → take last layer fwd+bwd
        hidden = torch.tanh(
            self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        )  # (B, hid_dim)
        return out, hidden   # (B, T, hid_dim*2), (B, hid_dim)


class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear((hid_dim * 2) + hid_dim, hid_dim)
        self.v    = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden:          (B, hid_dim)
        # encoder_outputs: (B, T, hid_dim*2)
        src_len = encoder_outputs.shape[1]
        hidden  = hidden.unsqueeze(1).repeat(1, src_len, 1)   # (B, T, hid_dim)
        energy  = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs), dim=2))
        )                                                       # (B, T, hid_dim)
        attention = self.v(energy).squeeze(2)                  # (B, T)
        return F.softmax(attention, dim=1)


class GRU_AT_Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, attention, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim    = hid_dim
        self.attention  = attention
        self.n_layers   = n_layers
        self.embedding  = nn.Embedding(output_dim, emb_dim)

        # CORRECCIÓN: Ahora sí le pasamos num_layers=n_layers
        self.gru    = nn.GRU(hid_dim * 2 + emb_dim, hid_dim, num_layers=n_layers)
        self.fc_out = nn.Linear((hid_dim * 2) + hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input:           (B,)
        # hidden:          (n_layers, B, hid_dim) <-- Ahora recibe todas las capas
        # encoder_outputs: (B, T, hid_dim*2)

        input    = input.unsqueeze(0)                          # (1, B)
        embedded = self.dropout(self.embedding(input))         # (1, B, emb_dim)

        # La atención siempre se calcula usando la última capa oculta (la más profunda)
        top_hidden = hidden[-1]                                # (B, hid_dim)

        attention = self.attention(top_hidden, encoder_outputs)    # (B, T)
        attention = attention.unsqueeze(1)                     # (B, 1, T)
        weighted  = torch.bmm(attention, encoder_outputs)      # (B, 1, hid_dim*2)
        weighted  = weighted.permute(1, 0, 2)                  # (1, B, hid_dim*2)

        rnn_input        = torch.cat((embedded, weighted), dim=2)   # (1, B, emb+hid*2)

        # Le pasamos el hidden multicapa al GRU
        out1, hidden     = self.gru(rnn_input, hidden)         # out1:(1,B,hid), hidden:(n_layers,B,hid)

        embedded = embedded.squeeze(0)   # (B, emb_dim)
        out1     = out1.squeeze(0)       # (B, hid_dim)
        weighted = weighted.squeeze(0)   # (B, hid_dim*2)

        prediction = self.fc_out(
            torch.cat((out1, weighted, embedded), dim=1)
        )  # (B, output_dim)

        return prediction, hidden        # Devolvemos el hidden multicapa para el siguiente paso


class GRU_AT_Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device  = device
        assert encoder.hid_dim  == decoder.hid_dim,   "Hidden dims must match!"
        assert encoder.n_layers == decoder.n_layers,  "Layer counts must match!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: (B, T, INPUT_DIM)
        # trg: (B, seq_len)
        batch_size     = trg.shape[0]
        trg_len        = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(src)

        # CORRECCIÓN: Duplicamos el hidden del Encoder para las N capas del Decoder
        hidden = hidden.unsqueeze(0).repeat(self.decoder.n_layers, 1, 1)

        input = trg[:, 0]   # <sos> tokens, shape (B,)

        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[t]     = output

            teacher_force = random.random() < teacher_forcing_ratio
            top1          = output.argmax(1)
            input         = trg[:, t] if teacher_force else top1

        return outputs   # (trg_len, B, vocab_size)

    @torch.no_grad()
    def translate(self, src, tokenizer, max_len=50):
        """Greedy decode. src: (1, T, INPUT_DIM)"""
        self.eval()
        enc_out, hidden = self.encoder(src)
        dec_input = torch.tensor([tokenizer.word2idx["<sos>"]], device=self.device)
        tokens = []
        for _ in range(max_len):
            pred, hidden = self.decoder(dec_input, hidden, enc_out)
            top1 = pred.argmax(1)
            word = tokenizer.idx2word.get(top1.item(), "<unk>")
            if word == "<eos>":
                break
            tokens.append(word)
            dec_input = top1
        return tokens


# ---------------------------------------------------------------------------
# Weight init
# ---------------------------------------------------------------------------

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


# ---------------------------------------------------------------------------
# Collate — padding dinamico, no truncation
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """
    Dynamic padding: text_target is padded to the longest sequence in the
    batch, not to a fixed global max_length. This means no tokens are ever
    lost — batches with short sentences stay short, and long ones expand
    only as far as needed.
    visual_input is already fixed-length N from SASS, so we just stack it.
    """
    visual = torch.stack([item["visual_input"] for item in batch])
    texts  = pad_sequence(
        [item["text_target"] for item in batch],
        batch_first=True, padding_value=0   # 0 == <pad>
    )
    return {"visual_input": visual, "text_target": texts}


# ---------------------------------------------------------------------------
# train / evaluate
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, output_dim, optimizer, criterion, clip):
    model.train()
    epoch_loss  = 0
    max_seq_len = 0 # track the longest target sequence seen this epoch
    avrg_seq_len = 0 # track the avarfage length of sequences seen this epoch
    for batch in loader:
        src = batch["visual_input"].to(device)
        trg = batch["text_target"].to(device)

        max_seq_len = max(max_seq_len, trg.shape[1])
        avrg_seq_len += trg.shape[1]

        optimizer.zero_grad()
        output = model(src, trg)
        out_dim = output.shape[-1]

        # --- ALINEACIÓN DINÁMICA DE TENSORES ---
        if output.shape[1] == trg.shape[0]:
            # output es [seq_len, batch, dim], trg es [batch, seq_len]
            trg = trg.transpose(0, 1)
            output = output[1:].reshape(-1, out_dim)
            trg    = trg[1:].reshape(-1)
        else:
            output = output[:, 1:, :].reshape(-1, out_dim)
            trg    = trg[:, 1:].reshape(-1)
        # ---------------------------------------

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(loader), max_seq_len, avrg_seq_len/len(loader)


@torch.no_grad()
def evaluate_loss(model, loader, output_dim, criterion):
    model.eval()
    epoch_loss = 0
    for batch in loader:
        src    = batch["visual_input"].to(device)
        trg    = batch["text_target"].to(device)

        output = model(src, trg, teacher_forcing_ratio=0.0)
        out_dim = output.shape[-1]

        # --- ALINEACIÓN DINÁMICA DE TENSORES ---
        if output.shape[1] == trg.shape[0]:
            trg = trg.transpose(0, 1)
            output = output[1:].reshape(-1, out_dim)
            trg    = trg[1:].reshape(-1)
        else:
            output = output[:, 1:, :].reshape(-1, out_dim)
            trg    = trg[:, 1:].reshape(-1)
        # ---------------------------------------

        epoch_loss += criterion(output, trg).item()
    return epoch_loss / len(loader)


@torch.no_grad()
def bleu_evaluate(model, loader, tokenizer, device):
    """Compute corpus BLEU on the val set."""
    model.eval()
    references = []
    hypotheses = []
    correct    = 0
    total      = 0

    pad_idx = tokenizer.word2idx["<pad>"]
    eos_idx = tokenizer.word2idx["<eos>"]
    sos_idx = tokenizer.word2idx["<sos>"]

    for batch in loader:
        src = batch["visual_input"].to(device)
        trg = batch["text_target"].to(device)   # (B, seq_len)

        output = model(src, trg, teacher_forcing_ratio=0.0)

        # --- ALINEACIÓN PARA BLEU ---
        if output.shape[1] == trg.shape[0]: # [seq_len, batch, dim]
            preds = output.argmax(2).transpose(0, 1) # Convertir a [batch, seq_len]
        else: # [batch, seq_len, dim]
            preds = output.argmax(2)

        B = trg.shape[0]
        for i in range(B):
            # Reference: remove <sos>, <eos>, <pad>
            ref_ids = trg[i, 1:].tolist()
            ref_words = []
            for idx in ref_ids:
                if idx in (eos_idx, pad_idx):
                    break
                ref_words.append(tokenizer.idx2word.get(idx, "<unk>"))

            # Hypothesis: preds starts at t=1 (Ahora preds es [batch, seq_len])
            hyp_ids = preds[i, 1:].tolist()
            hyp_words = []
            for idx in hyp_ids:
                if idx in (eos_idx, pad_idx):
                    break
                hyp_words.append(tokenizer.idx2word.get(idx, "<unk>"))

            references.append([ref_words])
            hypotheses.append(hyp_words)

            # Exact-match accuracy
            if ref_words == hyp_words:
                correct += 1
            total += 1

    smooth = SmoothingFunction().method1
    bleu   = corpus_bleu(references, hypotheses, smoothing_function=smooth)
    acc    = correct / total if total > 0 else 0.0
    return bleu * 100, acc * 100   # percentages, same as repo


def epoch_time(start, end):
    e = end - start
    return int(e / 60), int(e % 60)


# ---------------------------------------------------------------------------
# Logger — writes to both stdout and a timestamped .log file
# ---------------------------------------------------------------------------

def setup_logger(log_dir: str) -> logging.Logger:
    """
    Creates a logger that mirrors every message to:
      1. stdout (StreamHandler)
      2. <log_dir>/training_YYYYMMDD_HHMMSS.log (FileHandler)
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = os.path.join(log_dir, f"training_{timestamp}.log")

    logger = logging.getLogger("slt_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()   # avoid duplicate handlers on re-runs in notebooks

    fmt     = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    fh      = logging.FileHandler(log_path, encoding="utf-8")
    ch      = logging.StreamHandler()
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_path}")
    return logger


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log = setup_logger(LOG_DIR)

    # ------------------------------------------------------------------
    # 1. Load the two preprocessed pickles (train + val)
    # ------------------------------------------------------------------
    log.info("Loading datasets …")
    with open(TRAIN_PICKLE, "rb") as f:
        train_dataset = pickle.load(f)
    with open(VAL_PICKLE, "rb") as f:
        val_dataset = pickle.load(f)

    # Val set should use deterministic SASS (no random augmentation)
    if hasattr(val_dataset, "training"):
        val_dataset.training = False

    tokenizer  = train_dataset.tokenizer
    OUTPUT_DIM = tokenizer.vocab_size

    # Verify INPUT_DIM matches what is actually in the data
    actual_dim = train_dataset[0]["visual_input"].shape[-1]
    assert actual_dim == INPUT_DIM, (
        f"INPUT_DIM mismatch: dataset has {actual_dim}, "
        f"model expects {INPUT_DIM}. Fix INPUT_DIM at the top of this file."
    )

    train_loader = D.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                shuffle=True,  collate_fn=collate_fn, drop_last=False)
    val_loader   = D.DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                                shuffle=False, collate_fn=collate_fn, drop_last=False)

    log.info(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} | Vocab: {OUTPUT_DIM}")
    log.info(f"Device: {device}")

    # ------------------------------------------------------------------
    # 2. Build model
    # ------------------------------------------------------------------
    enc = GRU_AT_Encoder(INPUT_DIM, HID_DIM, N_LAYERS)
    att = Attention(HID_DIM)
    dec = GRU_AT_Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, att, DEC_DROPOUT)
    model = GRU_AT_Seq2Seq(enc, dec, device).to(device)
    model.apply(init_weights)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable parameters: {n_params:,}")

    # Shape sanity-check
    with torch.no_grad():
        d_src = torch.zeros(2, 150, INPUT_DIM).to(device)
        d_trg = torch.zeros(2, 10, dtype=torch.long).to(device)
        d_out = model(d_src, d_trg)
        assert d_out.shape == (10, 2, OUTPUT_DIM), f"Bad output shape: {d_out.shape}"
    log.info("Shape check passed ✓")

    # ------------------------------------------------------------------
    # 3. Loss, optimiser, scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.word2idx["<pad>"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=MIN_LR
    )

    # Log hyperparameters
    log.info("--- Hyperparameters ---")
    log.info(f"  INPUT_DIM={INPUT_DIM}  HID_DIM={HID_DIM}  EMB_DIM={EMB_DIM}")
    log.info(f"  N_LAYERS={N_LAYERS}  DROPOUT={DEC_DROPOUT}  CLIP={CLIP}")
    log.info(f"  BATCH={BATCH_SIZE}  LR={LR}  MIN_LR={MIN_LR}  EPOCHS={N_EPOCHS}")
    log.info("-----------------------")

    # Checkpoint paths — one per metric (same as repo)
    save_loss = os.path.join(SAVE_PATH, "loss_best_model.pt")
    save_bleu = os.path.join(SAVE_PATH, "bleu_best_model.pt")
    save_acc  = os.path.join(SAVE_PATH, "acc_best_model.pt")

    best_val_loss = float("inf")
    best_bleu     = 0.0
    best_acc      = 0.0
    # --- AÑADIR ESTE BLOQUE PARA INICIALIZAR EL CSV ---
    csv_file_path = os.path.join(LOG_DIR, f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "LR", "Train_Loss", "Train_PPL", "Val_Loss", "Val_PPL", "Val_BLEU", "Val_Acc"])
    # --------------------------------------------------
    # ------------------------------------------------------------------
    # 4. Training loop
    # ------------------------------------------------------------------
    log.info("Starting training …")
    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()

        train_loss, max_seq, avrg_seq = train_one_epoch(model, train_loader, OUTPUT_DIM,
                                              optimizer, criterion, CLIP)
        val_loss   = evaluate_loss(model, val_loader, OUTPUT_DIM, criterion)
        bleu, acc  = bleu_evaluate(model, val_loader, tokenizer, device)

        scheduler.step(val_loss)
        mins, secs = epoch_time(t0, time.time())
        lr = optimizer.param_groups[0]["lr"]

        # Checkpointing
        saved_tags = []
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_loss)
            saved_tags.append("loss")
        if bleu > best_bleu:
            best_bleu = bleu
            torch.save(model.state_dict(), save_bleu)
            saved_tags.append("bleu")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_acc)
            saved_tags.append("acc")

        saved_str = f"  ✓ saved [{', '.join(saved_tags)}]" if saved_tags else ""

        log.info(
            f"Epoch {epoch:3d}/{N_EPOCHS} | {mins}m {secs}s | "
            f"LR: {lr:.6f} | max_seq: {max_seq} | avrg_seq: {avrg_seq:.6f}{saved_str}"
        )
        log.info(
            f"  Train Loss: {train_loss:.3f} | PPL: {math.exp(train_loss):7.3f}"
        )
        log.info(
            f"  Val   Loss: {val_loss:.3f}   | PPL: {math.exp(val_loss):7.3f}"
        )
        log.info(
            f"  Val   BLEU: {bleu:.3f}       | Acc: {acc:.3f}"
        )
        # --- GUARDAR LA ÉPOCA EN EL CSV ---
        with open(csv_file_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{lr:.12f}",
                f"{train_loss:.12f}",
                f"{math.exp(train_loss):.12f}",
                f"{val_loss:.12f}",
                f"{math.exp(val_loss):.12f}",
                f"{bleu:.12f}",
                f"{acc:.12f}"
            ])
        # ----------------------------------------------------------
    # ------------------------------------------------------------------
    # 5. Final summary
    # ------------------------------------------------------------------
    log.info("=" * 50)
    log.info("Training complete.")
    log.info(f"  Best Val Loss : {best_val_loss:.4f}  → {save_loss}")
    log.info(f"  Best BLEU     : {best_bleu:.3f}      → {save_bleu}")
    log.info(f"  Best Acc      : {best_acc:.3f}       → {save_acc}")
    log.info("=" * 50)
