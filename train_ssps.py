"""
SSPS (Frame-Level) Training Script (Config-driven, FIXED)
=========================================================

- Reads feature settings from: config.get_active_config()
- Reads training params from: config.TRAIN_PARAMS
- Computes DEV EER each epoch using stable score: logits[:,0] - logits[:,1]
- Fixes loss averaging (sample-weighted)
- Saves best model by lowest EER (if EER is available)

Assumptions:
- SSPSDataset(split, cfg, max_len=..., padding=...) returns (inputs, labels)
- labels: 0=bonafide, 1=spoof
- SSPSModel(ssps_dim=3072, emb_dim=TRAIN_PARAMS["emb_dim"]) returns (emb, logits) with logits shape [B,2]
"""

import os
import math
import random
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_active_config, print_config, PROJECT_ROOT, TRAIN_PARAMS, PROTOCOLS
import eval_metrics as em
import torch.nn.functional as F
from torch.utils.data import Dataset
from ska_tdnn_backbone import SKA_TDNN_Backbone


# -------------------------------------------------------------------
# Dataset and Model Classes
# -------------------------------------------------------------------
class SSPSDataset(Dataset):
    def __init__(self, part: str, config: dict, max_len: int = 750, padding: str = "repeat"):
        self.part = part
        self.config = config
        self.max_len = max_len
        self.padding = padding
        self.ssps_path = Path(config["ssps_path"]) / part
        
        # Check if this is ASVspoof 2019 LA dataset
        self.is_2019_la = config.get("dataset") == "asvspoof2019_la" or config.get("dataset_type") == "asvspoof2019_la"
        
        # Load protocol
        if self.is_2019_la:
            # ASVspoof 2019 LA protocol
            protocol_dir = Path("D:/Mahmud/Datasets/asvspoof_2019/LA/LA/ASVspoof2019_LA_cm_protocols")
            # Train partition uses .trn.txt, dev/eval use .trl.txt
            if part == "train":
                self.protocol_path = protocol_dir / f"ASVspoof2019.LA.cm.{part}.trn.txt"
            else:
                self.protocol_path = protocol_dir / f"ASVspoof2019.LA.cm.{part}.trl.txt"
        else:
            # ASVspoof5 protocol
            self.protocol_path = PROTOCOLS[part]
        
        self.samples = self._load_protocol(self.protocol_path)
        print(f"[{part.upper()}] Dataset loaded. Total samples: {len(self.samples)}")

    def _load_protocol(self, path: Path):
        samples = []
        label_counts = {0: 0, 1: 0}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts or len(parts) < 2:
                    continue
                
                if self.is_2019_la:
                    # ASVspoof 2019 LA format: speaker_id, filename, system_id, key, label
                    # Bonafide: LA_0079 LA_T_1138215 - - bonafide
                    # Spoof: LA_0079 LA_T_1004644 - A01 spoof
                    if len(parts) < 5:
                        continue
                    utt_id = parts[1]
                    label_str = parts[-1].lower()
                    if label_str == "bonafide":
                        label = 0
                    elif label_str == "spoof":
                        label = 1
                    else:
                        continue  # Skip unknown labels
                else:
                    # ASVspoof5 format: speaker_id, filename, ..., label
                    if parts[0] == "speaker_id" or "flac" in parts[0]:
                        continue
                    utt_id = parts[1]
                    label_str = parts[-1].lower()
                    if label_str == "spoof":
                        label = 1
                    elif label_str == "bonafide" or label_str == "genuine":
                        label = 0
                    else:
                        continue  # Skip unknown labels
                
                samples.append((utt_id, label))
                label_counts[label] += 1
        
        print(f"  Label distribution: bonafide={label_counts[0]}, spoof={label_counts[1]}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        utt_id, label = self.samples[idx]
        
        # Load SSPS feature
        # SSPS features are extracted as (3072, Time)
        feat_path = self.ssps_path / f"{utt_id}.pt"
        
        if not feat_path.exists():
            # If feature is missing, return a dummy tensor
            return torch.zeros((3072, self.max_len)), torch.tensor(label, dtype=torch.long)
            
        try:
            feat = torch.load(feat_path, map_location="cpu", weights_only=False)  # (C, T)
        except Exception as e:
            print(f"Error loading {feat_path}: {e}")
            return torch.zeros((3072, self.max_len)), torch.tensor(label, dtype=torch.long)

        # Pad/Truncate
        C, T = feat.shape
        if T > self.max_len:
            start = random.randint(0, T - self.max_len)
            feat = feat[:, start : start + self.max_len]
        elif T < self.max_len:
            if self.padding == "repeat":
                n_repeat = self.max_len // T + 1
                feat = feat.repeat(1, n_repeat)[:, :self.max_len]
            else:  # zero padding
                diff = self.max_len - T
                feat = F.pad(feat, (0, diff), "constant", 0)
        
        return feat, torch.tensor(label, dtype=torch.long)


class SSPSModel(nn.Module):
    def __init__(self, ssps_dim: int = 3072, emb_dim: int = 256):
        super().__init__()
        
        # SKA-TDNN Backbone
        # Input channels: ssps_dim (3072 for SSPS Frame-Level)
        self.backbone = SKA_TDNN_Backbone(in_chans=ssps_dim, C=512, out_dim=None)  # C=512 for lighter model
        
        # Output of backbone is (B, 1536, T) -> Pool -> (B, 1536)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_emb = nn.Linear(1536, emb_dim)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 2)
        )

    def forward(self, x):
        # x: (B, C, T)
        x = self.backbone(x)  # (B, 1536, T)
        x = self.pool(x).squeeze(-1)  # (B, 1536)
        emb = self.fc_emb(x)  # (B, emb_dim)
        
        # Normalize embedding
        emb = F.normalize(emb, dim=1)
        
        # Classify
        out = self.classifier(emb)
        return emb, out


# -------------------------------------------------------------------
# Optional OCSoftmax fallback (if your project doesn't provide it)
# -------------------------------------------------------------------
class OCSoftmax(nn.Module):
    """
    Minimal OC-Softmax style loss (fallback).
    If you already have an OCSoftmax implementation in your project,
    replace this class with: from <your_module> import OCSoftmax
    and delete this fallback.
    """
    def __init__(self, feat_dim: int, r_real: float = 0.9, r_fake: float = 0.2, alpha: float = 20.0):
        super().__init__()
        self.r_real = r_real
        self.r_fake = r_fake
        self.alpha = alpha
        self.w = nn.Parameter(torch.randn(feat_dim))
        nn.init.normal_(self.w, mean=0.0, std=0.01)

    def forward(self, emb: torch.Tensor, labels: torch.Tensor):
        """
        emb: [B, D] embedding (assumed L2-normalized or will be normalized)
        labels: [B] 0=bonafide, 1=spoof
        returns: loss (scalar), scores (B,) where higher => more bonafide
        """
        emb = torch.nn.functional.normalize(emb, dim=1)
        w = torch.nn.functional.normalize(self.w, dim=0)
        cos = emb @ w  # [B]

        # score: larger => more bonafide
        scores = self.alpha * cos

        # OC-Softmax margin: push bonafide above r_real and spoof below r_fake
        # This is a simplified hinge-like objective.
        y = labels.float()
        # bonafide (0): want cos >= r_real
        loss_real = torch.relu(self.r_real - cos) * (1.0 - y)
        # spoof (1): want cos <= r_fake
        loss_fake = torch.relu(cos - self.r_fake) * y
        loss = (loss_real + loss_fake).mean()

        return loss, scores


# -------------------------------------------------------------------
# Utils
# -------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def label_distribution(ds, name: str):
    c = Counter()
    tmp = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
    for _, y in tmp:
        c.update(y.tolist())
    print(f"{name} label distribution: {dict(c)}")
    return c


@torch.no_grad()
def eval_dev(model, loader, device, loss_mode: str, criterion_ce=None, criterion_oc=None):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_n = 0

    all_scores = []
    all_labels = []

    for x, y in tqdm(loader, desc="Dev", ncols=90, leave=False):
        x = x.to(device, non_blocking=False)  # non_blocking requires pin_memory=True
        y = y.to(device, non_blocking=False)

        emb, logits = model(x)

        # ---- loss
        if loss_mode == "ocsoftmax":
            # OCSoftmax uses embeddings to compute loss + score
            loss, scores = criterion_oc(emb, y)
            # for accuracy we still use logits argmax (if logits exist)
            pred = torch.argmax(logits, dim=1)
            batch_scores = scores.detach().cpu().tolist()
        else:
            loss = criterion_ce(logits, y)
            pred = torch.argmax(logits, dim=1)
            # Stable score for EER: logit diff (bonafide - spoof)
            scores = (logits[:, 0] - logits[:, 1])
            batch_scores = scores.detach().cpu().tolist()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_n += bs
        total_correct += (pred == y).sum().item()

        all_scores.extend(batch_scores)
        all_labels.extend(y.detach().cpu().tolist())

    avg_loss = total_loss / max(total_n, 1)
    acc = total_correct / max(total_n, 1)

    bon = [s for s, l in zip(all_scores, all_labels) if l == 0]
    spf = [s for s, l in zip(all_scores, all_labels) if l == 1]

    if len(bon) == 0 or len(spf) == 0:
        eer = float("nan")
    else:
        eer, _ = em.compute_eer(bon, spf)

    return avg_loss, acc, eer


def main():
    print_config()

    cfg = get_active_config()   # should be the active FEATURE_CONFIG entry dict
    tp = TRAIN_PARAMS

    # --- device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(tp.get("gpu", "0"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- seed
    set_seed(int(tp.get("seed", 42)))

    # --- read from config ONLY
    feat_len = int(cfg["feat_len"])
    padding = tp["padding"]
    batch_size = int(tp["batch_size"])
    num_workers = int(tp["num_workers"])
    num_epochs = int(tp["num_epochs"])
    lr = float(tp["lr"])
    weight_decay = float(tp["weight_decay"])
    grad_clip = float(tp.get("gradient_clip", 0.0))

    emb_dim = int(tp["emb_dim"])
    loss_mode = str(tp.get("add_loss", "ce")).lower().strip()

    # --- output dir (based on feature config key/name)
    # If cfg has no "key", fallback to a safe folder name
    folder_name = cfg.get("key", "ssps_frame_level_skatdnn")
    model_dir = PROJECT_ROOT / "models" / folder_name
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / "best_model.pth"

    # --- datasets
    # IMPORTANT: use "dev" for EER tracking. Keep "eval" for final evaluation.
    train_set = SSPSDataset("train", cfg, max_len=feat_len, padding=padding)
    dev_set = SSPSDataset("dev", cfg, max_len=feat_len, padding=padding)

    train_dist = label_distribution(train_set, "TRAIN")
    dev_dist = label_distribution(dev_set, "DEV")

    if 0 not in dev_dist or 1 not in dev_dist:
        print("WARNING: DEV split missing a class -> EER will be NaN until fixed.")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # Disabled to reduce CUDA memory usage
        drop_last=False,
    )
    dev_loader = DataLoader(
        dev_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,  # Disabled to reduce CUDA memory usage
        drop_last=False,
    )

    # --- model
    model = SSPSModel(ssps_dim=3072, emb_dim=emb_dim).to(device)

    # --- losses
    criterion_ce = None
    criterion_oc = None

    if loss_mode == "ocsoftmax":
        criterion_oc = OCSoftmax(
            feat_dim=emb_dim,
            r_real=float(tp["r_real"]),
            r_fake=float(tp["r_fake"]),
            alpha=float(tp["alpha"]),
        ).to(device)
        print("Loss: OCSoftmax (config-driven)")
    else:
        # optional class weights (helps imbalance)
        if 0 in train_dist and 1 in train_dist and train_dist[0] > 0 and train_dist[1] > 0:
            n0, n1 = train_dist[0], train_dist[1]
            n = n0 + n1
            w0 = n / (2.0 * n0)
            w1 = n / (2.0 * n1)
            weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)
            criterion_ce = nn.CrossEntropyLoss(weight=weights)
            print(f"Loss: CrossEntropy (weighted) w0={w0:.3f}, w1={w1:.3f}")
        else:
            criterion_ce = nn.CrossEntropyLoss()
            print("Loss: CrossEntropy (unweighted)")

    # --- optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # if you want your original interval-based decay:
    # every TRAIN_PARAMS["interval"] epochs multiply lr by TRAIN_PARAMS["lr_decay"]
    lr_decay = float(tp.get("lr_decay", 1.0))
    interval = int(tp.get("interval", 0))

    best_eer = float("inf")

    # --- training loop
    for epoch in range(1, num_epochs + 1):
        model.train()

        running_loss = 0.0
        running_correct = 0
        running_n = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]", ncols=90, leave=False)

        for x, y in pbar:
            x = x.to(device, non_blocking=False)  # non_blocking requires pin_memory=True
            y = y.to(device, non_blocking=False)

            optimizer.zero_grad(set_to_none=True)

            emb, logits = model(x)

            if loss_mode == "ocsoftmax":
                loss, _scores = criterion_oc(emb, y)
            else:
                loss = criterion_ce(logits, y)

            loss.backward()

            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            bs = y.size(0)
            running_loss += loss.item() * bs
            running_n += bs

            pred = torch.argmax(logits, dim=1)
            running_correct += (pred == y).sum().item()

            avg_loss = running_loss / max(running_n, 1)
            avg_acc = running_correct / max(running_n, 1)

            # important: show real loss (not double-divided)
            pbar.set_postfix(loss=f"{avg_loss:.6f}", acc=f"{avg_acc*100:.2f}%")

        train_loss = running_loss / max(running_n, 1)
        train_acc = running_correct / max(running_n, 1)

        # --- dev eval
        dev_loss, dev_acc, dev_eer = eval_dev(
            model, dev_loader, device,
            loss_mode=loss_mode,
            criterion_ce=criterion_ce,
            criterion_oc=criterion_oc
        )

        # --- LR interval decay (config-driven)
        if interval > 0 and epoch % interval == 0 and lr_decay < 1.0:
            for g in optimizer.param_groups:
                g["lr"] *= lr_decay

        current_lr = optimizer.param_groups[0]["lr"]
        eer_str = "nan" if math.isnan(dev_eer) else f"{dev_eer*100:.4f}%"

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {train_loss:.6f} Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {dev_loss:.6f} Acc: {dev_acc*100:.2f}% EER: {eer_str} | "
            f"LR: {current_lr:.6f}"
        )

        # --- save best by EER (preferred for ASVspoof)
        if not math.isnan(dev_eer) and dev_eer < best_eer:
            best_eer = dev_eer
            torch.save(model.state_dict(), best_path)
            print(f"--> New best model saved! (EER: {best_eer*100:.4f}%)")

    print("Training finished.")
    if best_eer < float("inf"):
        print(f"Best DEV EER: {best_eer*100:.4f}%")
    else:
        print("Best DEV EER: N/A (EER was NaN every epoch)")


if __name__ == "__main__":
    main()
