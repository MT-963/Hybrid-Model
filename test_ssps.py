"""
SSPS (Frame-Level) Training Script - FIXED
=========================================

Fixes:
- Correct loss averaging (no fake 0.000000)
- Stable dev EER computation (no NaN unless your dev split has one class)
- Uses logit-difference score for EER: score = logits[:,0] - logits[:,1]
  (higher => more bonafide)

Assumes:
- SSPSDataset returns (inputs, labels)
- labels: 0=bonafide, 1=spoof
- SSPSModel forward returns (emb, logits) where logits shape [B,2]

You must have:
- train_ssps.py provides SSPSDataset, SSPSModel (or adjust imports)
- eval_metrics.py provides compute_eer(bonafide_scores, spoof_scores)
"""

import os
import math
import random
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_active_config, print_config, PROJECT_ROOT
from train_ssps import SSPSDataset, SSPSModel
import eval_metrics as em


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    """
    Returns:
      avg_loss (float)
      acc (float in [0,1])
      eer (float in [0,1] or nan)
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_n = 0

    scores = []
    labels_all = []

    for x, y in tqdm(loader, desc="Val", ncols=90, leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        _, logits = model(x)  # logits [B,2]
        loss = criterion(logits, y)

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_n += bs

        pred = torch.argmax(logits, dim=1)
        total_correct += (pred == y).sum().item()

        # Stable EER score: logit difference (bonafide - spoof)
        s = (logits[:, 0] - logits[:, 1]).detach().cpu().tolist()
        scores.extend(s)
        labels_all.extend(y.detach().cpu().tolist())

    avg_loss = total_loss / max(total_n, 1)
    acc = total_correct / max(total_n, 1)

    # Build bonafide/spoof score lists
    bon = [s for s, l in zip(scores, labels_all) if l == 0]
    spf = [s for s, l in zip(scores, labels_all) if l == 1]

    # Guard against empty classes -> EER undefined
    if len(bon) == 0 or len(spf) == 0:
        eer = float("nan")
    else:
        eer, _thr = em.compute_eer(bon, spf)

    return avg_loss, acc, eer


def main():
    print_config()
    cfg = get_active_config()

    # -------------------------
    # Settings
    # -------------------------
    seed = cfg.get("seed", 42)
    set_seed(seed)

    batch_size = cfg.get("batch_size", 32)
    num_workers = cfg.get("num_workers", 4)
    max_epochs = cfg.get("epochs", 100)
    lr = cfg.get("lr", 1e-4)
    weight_decay = cfg.get("weight_decay", 1e-4)

    feat_len = cfg["feat_len"]  # your frame length
    padding = cfg.get("padding", "repeat")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -------------------------
    # Datasets / Loaders
    # -------------------------
    train_set = SSPSDataset("train", cfg, max_len=feat_len, padding=padding)
    val_set   = SSPSDataset("dev",   cfg, max_len=feat_len, padding=padding)  # IMPORTANT: dev split for EER

    # Quick label distribution sanity check (helps catch NaN EER early)
    def label_dist(ds, name):
        c = Counter()
        tmp_loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
        for _, y in tmp_loader:
            c.update(y.tolist())
        print(f"{name} label distribution: {dict(c)}")
        return c

    train_dist = label_dist(train_set, "TRAIN")
    val_dist   = label_dist(val_set,   "DEV")

    if 0 not in train_dist or 1 not in train_dist:
        print("WARNING: TRAIN set missing a class -> training is broken.")
    if 0 not in val_dist or 1 not in val_dist:
        print("WARNING: DEV set missing a class -> EER will be NaN.")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # -------------------------
    # Model / Loss / Optim
    # -------------------------
    model = SSPSModel(ssps_dim=3072, emb_dim=256).to(device)

    # For ASVspoof, consider class imbalance.
    # If you want class weights, compute them from TRAIN distribution:
    # weight for class i = N_total / (2 * N_i)
    if 0 in train_dist and 1 in train_dist and train_dist[0] > 0 and train_dist[1] > 0:
        n0 = train_dist[0]
        n1 = train_dist[1]
        n = n0 + n1
        w0 = n / (2.0 * n0)
        w1 = n / (2.0 * n1)
        class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)
        print(f"Using class weights: w0(bonafide)={w0:.3f}, w1(spoof)={w1:.3f}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Optional scheduler (same behavior as your LR drops)
    # This reduces LR on plateau of DEV EER (better than accuracy for spoof detection)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        verbose=True,
        min_lr=1e-6,
    )

    # Mixed precision optional
    use_amp = cfg.get("amp", True) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # -------------------------
    # Saving
    # -------------------------
    model_dir = PROJECT_ROOT / "models" / "ssps_only_skatdnn"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / "best_model.pth"

    best_eer = float("inf")

    # -------------------------
    # Training Loop
    # -------------------------
    for epoch in range(1, max_epochs + 1):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_n = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs} [Train]", ncols=90, leave=False)

        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                _, logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = y.size(0)
            total_loss += loss.item() * bs
            total_n += bs

            pred = torch.argmax(logits, dim=1)
            total_correct += (pred == y).sum().item()

            avg_loss_so_far = total_loss / max(total_n, 1)
            acc_so_far = total_correct / max(total_n, 1)
            pbar.set_postfix(loss=f"{avg_loss_so_far:.6f}", acc=f"{acc_so_far*100:.2f}%")

        train_loss = total_loss / max(total_n, 1)
        train_acc = total_correct / max(total_n, 1)

        # Validate
        val_loss, val_acc, val_eer = evaluate(model, val_loader, device, criterion)

        # Scheduler step uses EER (primary)
        if not math.isnan(val_eer):
            scheduler.step(val_eer)
        else:
            # fallback if EER NaN (still reduce on loss)
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        # Log
        eer_str = "nan" if math.isnan(val_eer) else f"{val_eer*100:.4f}%"
        print(
            f"Epoch {epoch}/{max_epochs} | "
            f"Train Loss: {train_loss:.6f} Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.6f} Acc: {val_acc*100:.2f}% EER: {eer_str} | "
            f"LR: {current_lr:.6f}"
        )

        # Save best by EER (preferred)
        improved = (not math.isnan(val_eer)) and (val_eer < best_eer)
        if improved:
            best_eer = val_eer
            torch.save(model.state_dict(), best_path)
            print(f"--> New best model saved! (EER: {best_eer*100:.4f}%)")

    print("Training finished.")
    print(f"Best DEV EER: {best_eer*100:.4f}%" if best_eer < float("inf") else "Best DEV EER: N/A (never computed)")


if __name__ == "__main__":
    main()
