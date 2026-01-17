"""
ASVspoof5 Hybrid Model Testing - Configurable Version
======================================================

config.py dosyasındaki ayarları kullanarak test yapar.

Kullanım:
    1. config.py'de ACTIVE_CONFIG'u ayarla
    2. python test_hybrid.py
    
    veya komut satırından:
    python test_hybrid.py --config wavlm_fullres_fp16_ssps
"""

from __future__ import annotations
import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import csv
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import eval_metrics as em
from loss import OCSoftmax, setup_seed
from ska_tdnn_backbone import SKA_TDNN_Backbone
try:
    from NeXt_TDNN_ASV.models import NeXt_TDNN_ECA_ilk_ilk_Light
    NEXT_TDNN_AVAILABLE = True
except ImportError:
    NEXT_TDNN_AVAILABLE = False

# Import config
from config import (
    FEATURE_CONFIGS, PROTOCOLS, TRAIN_PARAMS,
    get_output_path, ACTIVE_CONFIG
)

warnings.filterwarnings("ignore")


# =============================================================================
# HYBRID DATASET (Test version)
# =============================================================================
class HybridFeatureDataset(Dataset):
    def __init__(
        self,
        wavlm_root: Path,
        ssps_root: Path,
        protocol_file: Path,
        split: str,
        feat_len: int = 750,
        padding: str = "repeat",
    ) -> None:
        super().__init__()
        self.wavlm_root = Path(wavlm_root)
        self.ssps_root = Path(ssps_root)
        self.split = split
        self.feat_len = int(feat_len)
        self.padding = padding

        if not protocol_file.exists():
            raise FileNotFoundError(f"Protokol bulunamadi: {protocol_file}")

        self.items = self._read_protocol(protocol_file)

        # Check dimensions
        sample_w = torch.load(self._feat_path(self.items[0][0], "wavlm"), map_location="cpu")
        self.wavlm_dim = sample_w.shape[0]
        
        sample_s = torch.load(self._feat_path(self.items[0][0], "ssps"), map_location="cpu")
        self.ssps_dim = sample_s.shape[0] if sample_s.ndim == 1 else sample_s.shape[-1]
        
        print(f"[INFO] WavLM dim: {self.wavlm_dim}, SSPS dim: {self.ssps_dim}, Samples: {len(self.items)}")

    def _read_protocol(self, path: Path):
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        delim = "\t" if "\t" in text else ("," if "," in text.splitlines()[0] else None)

        rows = []
        if delim:
            lines = text.splitlines()
            reader = csv.reader(lines, delimiter=delim)
            first = lines[0].lower()
            if "speaker" in first or "flac" in first or "key" in first:
                next(reader, None)
            for r in reader:
                if any(tok.strip() for tok in r):
                    rows.append([tok.strip() for tok in r])
        else:
            for ln in text.splitlines():
                if ln.strip():
                    rows.append(re.split(r"\s+", ln.strip()))

        uid_idx = self._guess_uid_index(rows)
        lab_idx = self._guess_label_index(rows)

        items = []
        for r in rows:
            uid = r[uid_idx]
            lab_tok = r[lab_idx].lower()
            if lab_tok in ("bonafide", "bona-fide", "genuine", "real", "target"):
                lab = 0
            elif lab_tok in ("spoof", "attack", "non-target", "fake"):
                lab = 1
            else:
                continue
            items.append((uid, lab))
        return items

    def _guess_uid_index(self, rows):
        pat = re.compile(r"^[TDE]_\d{10}$")
        max_cols = max(len(r) for r in rows)
        best_j, best_score = 0, -1
        for j in range(max_cols):
            score = sum(1 for r in rows[:200] if len(r) > j and pat.match(r[j]))
            if score > best_score:
                best_j, best_score = j, score
        return best_j

    def _guess_label_index(self, rows):
        max_cols = max(len(r) for r in rows)
        # Count exact matches for each column
        best_j, best_score = -1, 0
        for j in range(max_cols):
            score = 0
            for r in rows[:500]:
                if len(r) > j:
                    val = r[j].lower().strip()
                    if val in ("bonafide", "bona-fide", "spoof", "attack", "genuine", "fake", "target", "non-target"):
                        score += 1
            if score > best_score:
                best_j, best_score = j, score
        return best_j

    def _feat_path(self, utt_id: str, branch: str) -> Path:
        root = self.wavlm_root if branch == "wavlm" else self.ssps_root
        p = root / self.split / f"{utt_id}.pt"
        if not p.exists():
            alt = list(root.glob(f"**/{self.split}/{utt_id}.pt"))
            if alt:
                return alt[0]
        return p

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        if T == self.feat_len:
            return x
        if T > self.feat_len:
            return x[:, :self.feat_len]
        if self.padding == "zero":
            pad = torch.zeros(x.shape[0], self.feat_len - T, dtype=x.dtype)
        else:
            pad = x.repeat(1, (self.feat_len + T - 1) // T)[:, :self.feat_len - T]
        return torch.cat([x, pad], dim=1)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        utt_id, label = self.items[idx]
        try:
            w = torch.load(self._feat_path(utt_id, "wavlm"), map_location="cpu")
            if w.dtype == torch.float16:
                w = w.float()
            w = self._pad(w)
            
            s = torch.load(self._feat_path(utt_id, "ssps"), map_location="cpu")
            if s.dtype == torch.float16:
                s = s.float()
            if s.ndim == 2:
                s = s.mean(dim=-1)
            
            return w, s, utt_id, int(label)
        except Exception:
            return None

    def collate_fn(self, batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None
        ws, ss, uids, labs = zip(*batch)
        ws = torch.stack(ws, dim=0)
        ss = torch.stack(ss, dim=0)
        labs = torch.as_tensor(labs, dtype=torch.long)
        return ws, ss, list(uids), labs


# =============================================================================
# HYBRID MODEL
# =============================================================================
class HybridModel(nn.Module):
    def __init__(self, wavlm_dim: int, ssps_dim: int, emb_dim: int = 256, feat_len: int = 750, backbone_type: str = "skatdnn"):
        """
        Args:
            wavlm_dim: WavLM/HuBERT feature dimension
            ssps_dim: SSPS feature dimension
            emb_dim: Embedding dimension
            feat_len: Feature sequence length
            backbone_type: "skatdnn" or "next_tdnn"
        """
        super().__init__()
        
        # Select backbone based on type
        if backbone_type == "skatdnn":
            self.wavlm_backbone = SKA_TDNN_Backbone(
                in_chans=wavlm_dim,
                C=1024,
                model_scale=8,
                out_dim=None  # Return (B, 1536, T)
            )
        elif backbone_type == "next_tdnn":
            if not NEXT_TDNN_AVAILABLE:
                raise ImportError("NeXt TDNN not available. Install NeXt_TDNN_ASV or use 'skatdnn' backbone.")
            self.wavlm_backbone = NeXt_TDNN_ECA_ilk_ilk_Light.NeXtTDNN(in_chans=wavlm_dim)
        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}. Use 'skatdnn' or 'next_tdnn'.")
        
        with torch.no_grad():
            # Set to eval mode for BatchNorm during dummy forward pass
            self.wavlm_backbone.eval()
            dummy = torch.randn(1, wavlm_dim, feat_len)
            out = self.wavlm_backbone(dummy)
            wavlm_out_dim = out.shape[1] if out.ndim == 3 else out.shape[-1]
            self.wavlm_backbone.train()  # Set back to train mode
        
        self.wavlm_pool = nn.AdaptiveAvgPool1d(1)
        self.wavlm_fc = nn.Linear(wavlm_out_dim, emb_dim)
        
        self.ssps_fc = nn.Sequential(
            nn.Linear(ssps_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(inplace=True),
        )
        
        self.attention = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(emb_dim, 2),
        )
        
        self._emb_dim = emb_dim

    def forward(self, w: torch.Tensor, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w_out = self.wavlm_backbone(w)
        if w_out.ndim == 3:
            w_out = self.wavlm_pool(w_out).squeeze(-1)
        w_emb = self.wavlm_fc(w_out)
        
        s_emb = self.ssps_fc(s)
        
        concat = torch.cat([w_emb, s_emb], dim=-1)
        attn_weights = self.attention(concat)
        fused = attn_weights[:, 0:1] * w_emb + attn_weights[:, 1:2] * s_emb
        
        emb = F.normalize(fused, dim=1)
        logits = self.classifier(fused)
        
        return emb, logits


# =============================================================================
# TEST FUNCTION
# =============================================================================
def test(config_name: str) -> None:
    # Get config
    if config_name not in FEATURE_CONFIGS:
        raise ValueError(f"Gecersiz config: {config_name}. Secenekler: {list(FEATURE_CONFIGS.keys())}")
    
    cfg = FEATURE_CONFIGS[config_name]
    params = TRAIN_PARAMS
    out_fold = get_output_path(config_name)
    model_path = out_fold / "anti-spoofing_model.pt"
    loss_model_path = out_fold / "anti-spoofing_loss_model.pt"
    
    # Determine audio feature path (WavLM or HuBERT)
    if 'hubert_path' in cfg:
        audio_feat_path = cfg['hubert_path']
        feat_type = "HuBERT"
    elif 'wavlm_path' in cfg:
        audio_feat_path = cfg['wavlm_path']
        feat_type = "WavLM"
    else:
        raise ValueError("Config'de 'wavlm_path' veya 'hubert_path' bulunamadi!")
    
    # Print config
    print("=" * 60)
    print(f"TESTING: {cfg['name']}")
    print("=" * 60)
    print(f"  Model: {model_path}")
    print(f"  Audio Feature: {feat_type}")
    print(f"  Audio Feature Path: {audio_feat_path}")
    print(f"  SSPS: {cfg['ssps_path']}")
    print("=" * 60)
    
    # Check paths
    if not model_path.exists():
        raise FileNotFoundError(f"Model bulunamadi: {model_path}")
    if not audio_feat_path.exists():
        raise FileNotFoundError(f"{feat_type} features bulunamadi: {audio_feat_path}")
    if not cfg['ssps_path'].exists():
        raise FileNotFoundError(f"SSPS features bulunamadi: {cfg['ssps_path']}")

    # Setup
    os.environ["CUDA_VISIBLE_DEVICES"] = params["gpu"]
    setup_seed(params["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    eval_ds = HybridFeatureDataset(
        wavlm_root=audio_feat_path,  # Generic: works for both WavLM and HuBERT
        ssps_root=cfg['ssps_path'],
        protocol_file=PROTOCOLS["eval"],
        split="eval",
        feat_len=cfg['feat_len'],
        padding=params["padding"],
    )

    eval_loader = DataLoader(
        eval_ds, params["batch_size"], False,
        num_workers=params["num_workers"], collate_fn=eval_ds.collate_fn, pin_memory=True
    )

    # Model
    backbone_type = cfg.get('backbone_type', 'skatdnn')  # Default: SKA-TDNN
    print(f"  Backbone: {backbone_type.upper()}")
    model = HybridModel(
        wavlm_dim=eval_ds.wavlm_dim,
        ssps_dim=eval_ds.ssps_dim,
        emb_dim=params["emb_dim"],
        feat_len=cfg['feat_len'],
        backbone_type=backbone_type,
    ).to(device)
    
    # Load weights
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    # Auxiliary loss
    aux = OCSoftmax(params["emb_dim"], r_real=params["r_real"], r_fake=params["r_fake"], alpha=params["alpha"]).to(device)
    if loss_model_path.exists():
        aux.load_state_dict(torch.load(loss_model_path, map_location=device))
    aux.eval()

    print(f"\n  Device: {device}")
    print(f"  Eval samples: {len(eval_ds)}")
    print("=" * 60 + "\n")

    # Evaluate
    scores, labs, uids_all = [], [], []
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Eval"):
            if batch is None:
                continue
            w, s, uids, y = batch
            w, s, y = w.to(device), s.to(device), y.to(device)
            
            emb, logits = model(w, s)
            _, logits = aux(emb, y)
                    
            prob = F.softmax(logits, dim=1)[:, 0] if logits.dim() > 1 else logits
            scores.append(prob.cpu().numpy())
            labs.append(y.cpu().numpy())
            uids_all.extend(uids)

    scores = np.concatenate(scores)
    labs = np.concatenate(labs)
    
    # Compute EER
    eer = em.compute_eer(scores[labs == 0], scores[labs == 1])[0]
    
    print(f"\n{'='*60}")
    print(f"EVAL EER: {eer*100:.4f}%")
    print(f"{'='*60}")
    
    # Save scores
    scores_path = out_fold / "eval_scores.txt"
    with open(scores_path, "w", encoding="utf-8") as f:
        for uid, score, lab in zip(uids_all, scores, labs):
            label_str = "bonafide" if lab == 0 else "spoof"
            f.write(f"{uid}\t{score:.6f}\t{label_str}\n")
    print(f"Scores saved: {scores_path}")
    
    # Stats
    print(f"\nStatistikler:")
    print(f"  Total: {len(scores)}")
    print(f"  Bonafide: {(labs == 0).sum()}")
    print(f"  Spoof: {(labs == 1).sum()}")
    print(f"  Bonafide score mean: {scores[labs == 0].mean():.4f}")
    print(f"  Spoof score mean: {scores[labs == 1].mean():.4f}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hybrid Model Testing")
    parser.add_argument("--config", default=None, help="Config name (opsiyonel, config.py'den alinir)")
    args = parser.parse_args()
    
    config_name = args.config if args.config else ACTIVE_CONFIG
    
    print("\n" + "=" * 60)
    print("HYBRID MODEL TESTING")
    print("=" * 60)
    print(f"Config: {config_name}")
    print("=" * 60 + "\n")
    
    test(config_name)

