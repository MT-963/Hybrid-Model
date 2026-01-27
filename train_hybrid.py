"""
ASVspoof5 Hybrid Model Training - Configurable Version
=======================================================

config.py dosyasındaki ayarları kullanarak eğitim yapar.
Hyperparametreler sabit, sadece feature path'leri değiştirilebilir.

Kullanım:
    1. config.py'de ACTIVE_CONFIG'u ayarla
    2. python train_hybrid.py
    
    veya komut satırından:
    python train_hybrid.py --config wavlm_fullres_fp16_ssps
"""

from __future__ import annotations
import argparse
import os
import shutil
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
from loss import AMSoftmax, OCSoftmax, setup_seed
from ska_tdnn_backbone import SKA_TDNN_Backbone
try:
    from NeXt_TDNN_ASV.models import NeXt_TDNN_ECA_ilk_ilk_Light
    NEXT_TDNN_AVAILABLE = True
except ImportError:
    NEXT_TDNN_AVAILABLE = False

# Import config
from config import (
    FEATURE_CONFIGS, PROTOCOLS, TRAIN_PARAMS,
    get_output_path, get_active_config, print_config, ACTIVE_CONFIG
)

warnings.filterwarnings("ignore")


# =============================================================================
# HYBRID DATASET
# =============================================================================
class HybridFeatureDataset(Dataset):
    def __init__(
        self,
        wavlm_root: Path,
        ssps_root: Path = None,
        protocol_file: Path = None,
        split: str = None,
        feat_len: int = 750,
        padding: str = "repeat",
        use_ssps: bool = True,
    ) -> None:
        super().__init__()
        self.wavlm_root = Path(wavlm_root)
        self.ssps_root = Path(ssps_root) if ssps_root is not None else None
        self.split = split
        self.feat_len = int(feat_len)
        self.padding = padding
        self.use_ssps = use_ssps

        if protocol_file is not None and not protocol_file.exists():
            raise FileNotFoundError(f"Protokol bulunamadi: {protocol_file}")

        if protocol_file is not None:
            self.items = self._read_protocol(protocol_file)
        else:
            self.items = []

        # Check dimensions
        sample_w = torch.load(self._feat_path(self.items[0][0], "wavlm"), map_location="cpu")
        if sample_w.ndim != 2:
            raise ValueError(f"WavLM tensor (C,T) olmali, gelen shape: {tuple(sample_w.shape)}")
        self.wavlm_dim = sample_w.shape[0]
        
        if self.use_ssps and self.ssps_root is not None:
            sample_s = torch.load(self._feat_path(self.items[0][0], "ssps"), map_location="cpu")
            self.ssps_dim = sample_s.shape[0] if sample_s.ndim == 1 else sample_s.shape[-1]
        else:
            self.ssps_dim = None
        
        if self.use_ssps:
            print(f"[INFO] WavLM dim: {self.wavlm_dim}, SSPS dim: {self.ssps_dim}, Samples: {len(self.items)}")
        else:
            print(f"[INFO] WavLM dim: {self.wavlm_dim}, SSPS: DISABLED, Samples: {len(self.items)}")

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
            
            if self.use_ssps and self.ssps_root is not None:
                s = torch.load(self._feat_path(utt_id, "ssps"), map_location="cpu")
                if s.dtype == torch.float16:
                    s = s.float()
                if s.ndim == 2:
                    s = s.mean(dim=-1)
                return w, s, utt_id, int(label)
            else:
                # SSPS olmadan sadece dummy tensor döndür (kullanılmayacak)
                return w, torch.zeros(1), utt_id, int(label)
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
    def __init__(self, wavlm_dim: int, ssps_dim: int = None, emb_dim: int = 256, feat_len: int = 750, backbone_type: str = "skatdnn", use_ssps: bool = True):
        """
        Args:
            wavlm_dim: WavLM/HuBERT feature dimension
            ssps_dim: SSPS feature dimension (None if not using SSPS)
            emb_dim: Embedding dimension
            feat_len: Feature sequence length
            backbone_type: "skatdnn" or "next_tdnn"
            use_ssps: Whether to use SSPS features (default: True)
        """
        super().__init__()
        
        self.use_ssps = use_ssps
        
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
        
        if self.use_ssps:
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
        else:
            self.ssps_fc = None
            self.attention = None
        
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(emb_dim, 2),
        )
        
        self._emb_dim = emb_dim

    def forward(self, w: torch.Tensor, s: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        w_out = self.wavlm_backbone(w)
        if w_out.ndim == 3:
            w_out = self.wavlm_pool(w_out).squeeze(-1)
        w_emb = self.wavlm_fc(w_out)
        
        if self.use_ssps and s is not None:
            s_emb = self.ssps_fc(s)
            concat = torch.cat([w_emb, s_emb], dim=-1)
            attn_weights = self.attention(concat)
            fused = attn_weights[:, 0:1] * w_emb + attn_weights[:, 1:2] * s_emb
        else:
            # SSPS olmadan sadece WavLM/HuBERT kullan
            fused = w_emb
        
        emb = F.normalize(fused, dim=1)
        logits = self.classifier(fused)
        
        return emb, logits


# =============================================================================
# UTILITIES
# =============================================================================
def save_checkpoint(model: nn.Module, aux: Optional[nn.Module], path: Path) -> None:
    ckpt = {"model_state": model.state_dict()}
    if aux:
        ckpt["aux_state"] = aux.state_dict()
    torch.save(ckpt, path)


def adjust_lr(opt, base: float, decay: float, interval: int, epoch: int) -> None:
    lr = base * (decay ** (epoch // interval))
    for g in opt.param_groups:
        g["lr"] = lr


# =============================================================================
# TRAIN FUNCTION
# =============================================================================
def train(config_name: str) -> None:
    # Get config
    if config_name not in FEATURE_CONFIGS:
        raise ValueError(f"Gecersiz config: {config_name}. Secenekler: {list(FEATURE_CONFIGS.keys())}")
    
    cfg = FEATURE_CONFIGS[config_name]
    params = TRAIN_PARAMS
    out_fold = get_output_path(config_name)
    
    # Print config
    print("=" * 60)
    print(f"TRAINING: {cfg['name']}")
    print("=" * 60)
    
    # Determine audio feature path (WavLM, HuBERT, or Wav2Vec2)
    if 'hubert_path' in cfg:
        audio_feat_path = cfg['hubert_path']
        feat_type = "HuBERT"
    elif 'wavlm_path' in cfg:
        audio_feat_path = cfg['wavlm_path']
        feat_type = "WavLM"
    elif 'wav2vec2_path' in cfg:
        audio_feat_path = cfg['wav2vec2_path']
        feat_type = "Wav2Vec2"
    else:
        raise ValueError("Config'de 'wavlm_path', 'hubert_path' veya 'wav2vec2_path' bulunamadi!")
    
    # Check if SSPS is used
    use_ssps = cfg.get('use_ssps', True)  # Default: True for backward compatibility
    ssps_path = cfg.get('ssps_path', None)
    
    print(f"  Audio Feature: {feat_type}")
    print(f"  Audio Feature Path: {audio_feat_path}")
    if use_ssps:
        print(f"  SSPS: {ssps_path}")
    else:
        print(f"  SSPS: DISABLED")
    print(f"  Feat Len: {cfg['feat_len']}")
    print(f"  Output: {out_fold}")
    print("=" * 60)
    
    # Check paths
    if not audio_feat_path.exists():
        raise FileNotFoundError(f"{feat_type} features bulunamadi: {audio_feat_path}")
    if use_ssps and ssps_path is not None and not ssps_path.exists():
        raise FileNotFoundError(f"SSPS features bulunamadi: {ssps_path}")

    # Setup
    os.environ["CUDA_VISIBLE_DEVICES"] = params["gpu"]
    setup_seed(params["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Output folder
    if out_fold.exists():
        shutil.rmtree(out_fold)
    (out_fold / "checkpoint").mkdir(parents=True, exist_ok=True)

    # Datasets
    train_ds = HybridFeatureDataset(
        wavlm_root=audio_feat_path,  # Generic: works for both WavLM and HuBERT
        ssps_root=ssps_path if use_ssps else None,
        protocol_file=PROTOCOLS["train"],
        split="train",
        feat_len=cfg['feat_len'],
        padding=params["padding"],
        use_ssps=use_ssps,
    )
    dev_ds = HybridFeatureDataset(
        wavlm_root=audio_feat_path,  # Generic: works for both WavLM and HuBERT
        ssps_root=ssps_path if use_ssps else None,
        protocol_file=PROTOCOLS["dev"],
        split="dev",
        feat_len=cfg['feat_len'],
        padding=params["padding"],
        use_ssps=use_ssps,
    )

    train_loader = DataLoader(
        train_ds, params["batch_size"], True,
        num_workers=params["num_workers"], collate_fn=train_ds.collate_fn, pin_memory=True
    )
    dev_loader = DataLoader(
        dev_ds, params["batch_size"], False,
        num_workers=params["num_workers"], collate_fn=dev_ds.collate_fn, pin_memory=True
    )

    # Model
    backbone_type = cfg.get('backbone_type', 'skatdnn')  # Default: SKA-TDNN
    print(f"  Backbone: {backbone_type.upper()}")
    model = HybridModel(
        wavlm_dim=train_ds.wavlm_dim,
        ssps_dim=train_ds.ssps_dim if use_ssps else None,
        emb_dim=params["emb_dim"],
        feat_len=cfg['feat_len'],
        backbone_type=backbone_type,
        use_ssps=use_ssps,
    ).to(device)
    
    opt_model = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    ce = nn.CrossEntropyLoss()

    # Auxiliary loss
    aux = OCSoftmax(params["emb_dim"], r_real=params["r_real"], r_fake=params["r_fake"], alpha=params["alpha"]).to(device)
    opt_aux = torch.optim.SGD(aux.parameters(), lr=params["lr"])

    best_eer, early = float("inf"), 0
    ckpt_dir = out_fold / "checkpoint"

    print(f"\n  WavLM dim: {train_ds.wavlm_dim}")
    if use_ssps:
        print(f"  SSPS dim: {train_ds.ssps_dim}")
    print(f"  Train samples: {len(train_ds)}")
    print(f"  Dev samples: {len(dev_ds)}")
    print(f"  Device: {device}")
    print("=" * 60 + "\n")

    for ep in range(params["num_epochs"]):
        # Train
        model.train()
        adjust_lr(opt_model, params["lr"], params["lr_decay"], params["interval"], ep)
        adjust_lr(opt_aux, params["lr"], params["lr_decay"], params["interval"], ep)

        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Train {ep+1}"):
            if batch is None:
                continue
            w, s, _, y = batch
            w, y = w.to(device), y.to(device)
            s = s.to(device) if use_ssps else None

            opt_model.zero_grad()
            opt_aux.zero_grad()

            emb, logits = model(w, s)
            loss, logits = aux(emb, y)
            loss = loss * params["weight_loss"]
                    
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=params["gradient_clip"])
            torch.nn.utils.clip_grad_norm_(aux.parameters(), max_norm=params["gradient_clip"])
            
            opt_model.step()
            opt_aux.step()
            epoch_loss += loss.item()

        # Validation
        model.eval()
        scores, labs = [], []
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc="Dev"):
                if batch is None:
                    continue
                w, s, _, y = batch
                w, y = w.to(device), y.to(device)
                s = s.to(device) if use_ssps else None
                
                emb, logits = model(w, s)
                _, logits = aux(emb, y)
                        
                prob = F.softmax(logits, dim=1)[:, 0] if logits.dim() > 1 else logits
                scores.append(prob.cpu().numpy())
                labs.append(y.cpu().numpy())
                
        scores = np.concatenate(scores)
        labs = np.concatenate(labs)
        eer = em.compute_eer(scores[labs == 0], scores[labs == 1])[0]

        # Log
        with (out_fold / "eer.log").open("a", encoding="utf-8") as fp:
            fp.write(f"{ep+1}\t{eer:.6f}\t{epoch_loss/len(train_loader):.6f}\n")
        print(f"Epoch {ep+1}: EER = {eer*100:.4f}% | Loss = {epoch_loss/len(train_loader):.4f}")

        save_checkpoint(model, aux, ckpt_dir / f"epoch_{ep+1}.pt")
        if eer < best_eer:
            best_eer, early = eer, 0
            save_checkpoint(model, aux, out_fold / "anti-spoofing_model.pt")
            torch.save(aux.state_dict(), out_fold / "anti-spoofing_loss_model.pt")
            print(f"  >> Yeni en iyi EER: {best_eer*100:.4f}%")
        else:
            early += 1
            
        if early >= params["patience"]:
            print(f"Early stop - {params['patience']} epoch iyilesme yok")
            break

    print(f"\n{'='*60}")
    print(f"Egitim tamamlandi. En iyi EER: {best_eer*100:.4f}%")
    print(f"Model: {out_fold / 'anti-spoofing_model.pt'}")
    print(f"{'='*60}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hybrid Model Training")
    parser.add_argument("--config", default=None, help="Config name (opsiyonel, config.py'den alinir)")
    args = parser.parse_args()
    
    config_name = args.config if args.config else ACTIVE_CONFIG
    
    print("\n" + "=" * 60)
    print("HYBRID MODEL TRAINING")
    print("=" * 60)
    print(f"Config: {config_name}")
    print("=" * 60 + "\n")
    
    train(config_name)

