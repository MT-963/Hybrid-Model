"""
ASVspoof 2019 LA Dataset - SSPS Feature Extraction
==================================================

LA dataset için SSPS (SimCLR+ECAPA-TDNN) frame-level feature extraction.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

from config import PROJECT_ROOT

# SSPS model classes (extractFeatures_SSPS_simple_Gul.py'den)
class Conv1dSamePaddingReflect(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        import math
        L_in = x.size(-1)
        L_out = math.floor((L_in - self.dilation * (self.kernel_size - 1) - 1) / self.stride) + 1
        padding = (L_in - L_out) // 2
        x = F.pad(x, (padding, padding), mode="reflect")
        return self.conv(x)

class TDNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, groups=1):
        super().__init__()
        self.conv = Conv1dSamePaddingReflect(in_channels, out_channels, kernel_size, dilation=dilation, groups=groups)
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.norm(self.activation(self.conv(x)))

class Res2NetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale=8, kernel_size=3, dilation=1):
        super().__init__()
        assert in_channels % scale == 0 and out_channels % scale == 0
        in_channel = in_channels // scale
        hidden_channel = out_channels // scale
        self.blocks = nn.ModuleList([TDNNBlock(in_channel, hidden_channel, kernel_size=kernel_size, dilation=dilation) for _ in range(scale - 1)])
        self.scale = scale

    def forward(self, x):
        y = []
        for i, x_i in enumerate(torch.chunk(x, self.scale, dim=1)):
            if i == 0:
                y_i = x_i
            elif i == 1:
                y_i = self.blocks[i - 1](x_i)
            else:
                y_i = self.blocks[i - 1](x_i + y_i)
            y.append(y_i)
        return torch.cat(y, dim=1)

class SEBlock(nn.Module):
    def __init__(self, in_channels, se_channels, out_channels):
        super().__init__()
        self.conv1 = Conv1dSamePaddingReflect(in_channels, se_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv1dSamePaddingReflect(se_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s = x.mean(dim=2, keepdim=True)
        s = self.relu(self.conv1(s))
        s = self.sigmoid(self.conv2(s))
        return s * x

class SERes2NetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, res2net_scale=8, se_channels=128, kernel_size=1, dilation=1, groups=1):
        super().__init__()
        self.tdnn1 = TDNNBlock(in_channels, out_channels, kernel_size=1, dilation=1, groups=groups)
        self.res2net_block = Res2NetBlock(out_channels, out_channels, res2net_scale, kernel_size, dilation)
        self.tdnn2 = TDNNBlock(out_channels, out_channels, kernel_size=1, dilation=1, groups=groups)
        self.se_block = SEBlock(out_channels, se_channels, out_channels)
        self.shortcut = Conv1dSamePaddingReflect(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        residual = self.shortcut(x) if self.shortcut else x
        x = self.tdnn1(x)
        x = self.res2net_block(x)
        x = self.tdnn2(x)
        x = self.se_block(x)
        return x + residual

class ECAPATDNN(nn.Module):
    """ECAPA-TDNN encoder (Pooling İptal Edildi) - Frame-Level"""
    def __init__(self, mel_n_mels=80, encoder_dim=192, channels=[1024, 1024, 1024, 1024, 3072]):
        super().__init__()
        self.encoder_dim = encoder_dim
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=mel_n_mels
        )
        
        self.blocks = nn.ModuleList()
        self.blocks.append(TDNNBlock(mel_n_mels, channels[0], kernel_size=5, dilation=1))
        
        kernel_sizes = [5, 3, 3, 3, 1]
        dilations = [1, 2, 3, 4, 1]
        
        for i in range(1, len(channels) - 1):
            self.blocks.append(SERes2NetBlock(channels[i-1], channels[i], kernel_size=kernel_sizes[i], dilation=dilations[i]))
        
        self.mfa = TDNNBlock(channels[-1], channels[-1], kernel_sizes[-1], dilations[-1])

    def forward(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        
        x = self.mel_transform(x)  # (B, n_mels, T)
        x = (x + 1e-6).log()
        
        feats = []
        for layer in self.blocks:
            x = layer(x)
            feats.append(x)
        
        x = torch.cat(feats[1:], dim=1) 
        x = self.mfa(x)
        
        return x  # (Batch, 3072, Time)

class SimCLRModel(nn.Module):
    """SimCLR wrapper."""
    def __init__(self, encoder_dim=192, channels=[1024, 1024, 1024, 1024, 3072], mel_n_mels=80):
        super().__init__()
        self.encoder = ECAPATDNN(mel_n_mels=mel_n_mels, encoder_dim=encoder_dim, channels=channels)
        
    def forward(self, x):
        return self.encoder(x)

def load_ssps_checkpoint(ckpt_path: str | Path, device: str = "cuda"):
    """Load SSPS checkpoint and return model WITHOUT pooling layers."""
    ckpt_path = Path(ckpt_path)
    
    channels = [1024, 1024, 1024, 1024, 3072]
    encoder_dim = 512 
    mel_n_mels = 40
    
    model = SimCLRModel(encoder_dim=encoder_dim, channels=channels, mel_n_mels=mel_n_mels).to(device)
    
    if ckpt_path.exists():
        print(f">>> Checkpoint yükleniyor: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        
        state_dict = checkpoint.get("model", checkpoint)
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("encoder."):
                new_state_dict[k] = v
            else:
                new_state_dict[k] = v

        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        
        if missing:
            critical_missing = [k for k in missing if "blocks" in k or "mfa" in k]
            if critical_missing:
                print(f"[WARN] Kritik katmanlar eksik: {critical_missing[:3]}...")
        
        print("[OK] Model ağırlıkları yüklendi (Pooling katmanları hariç).")
        
    else:
        print(f"[ERR] Checkpoint bulunamadi: {ckpt_path}")
        sys.exit(1)
    
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    
    return model

def extract_partition(
    *,
    part: str,
    protocol_dir: str | Path,
    audio_root: str | Path,
    output_dir: str | Path,
    checkpoint_path: str | Path,
    device: str = "cuda",
):
    """Extract frame-level SSPS features for ASVspoof 2019 LA dataset."""
    protocol_dir = Path(protocol_dir)
    audio_root = Path(audio_root)
    output_dir = Path(output_dir)

    # ASVspoof 2019 LA protocol file format
    # Train partition uses .trn.txt, dev/eval use .trl.txt
    if part == "train":
        proto_fp = protocol_dir / f"ASVspoof2019.LA.cm.{part}.trn.txt"
    else:
        proto_fp = protocol_dir / f"ASVspoof2019.LA.cm.{part}.trl.txt"
    
    if not proto_fp.exists():
        raise FileNotFoundError(f"Protocol file not found: {proto_fp}")

    # ASVspoof 2019 LA audio directory structure
    audio_dir = audio_root / "LA" / f"ASVspoof2019_LA_{part}" / "flac"
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    out_dir = output_dir / part
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n>>> SSPS (No Pooling) modeli yukleniyor...")
    model = load_ssps_checkpoint(checkpoint_path, device)
    sr_model = 16000

    # Read protocol file
    with proto_fp.open("r", encoding="utf8") as f:
        lines = f.readlines()
        # ASVspoof 2019 format: speaker_id utt_id ... label
        utt_ids: List[str] = [ln.split()[1] for ln in lines if ln.strip()]

    print(f">>> {part} partition: {len(utt_ids)} utterance islenecek")

    for utt_id in tqdm(utt_ids, desc=part, ncols=80):
        out_fp = out_dir / f"{utt_id}.pt"
        if out_fp.is_file():
            continue

        wav_fp = audio_dir / f"{utt_id}.flac"
        if not wav_fp.is_file():
            tqdm.write(f"[MISSING] {wav_fp}")
            continue

        try:
            wav, sr = torchaudio.load(str(wav_fp))
        except Exception as e:
            tqdm.write(f"[ERR] Dosya okunamadi {wav_fp}: {e}")
            continue

        if sr != sr_model:
            wav = torchaudio.functional.resample(wav, sr, sr_model)
        
        # Mono yap
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
            
        wav = wav.squeeze(0).to(device) # (L,)

        with torch.inference_mode():
            # Çıktı: (1, 3072, Time)
            emb = model(wav)
        
        # Kaydederken cpu'ya al ve gereksiz batch boyutunu at
        # Sonuç boyutu: (3072, Time)
        torch.save(emb.squeeze(0).cpu(), out_fp)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SSPS Frame-Level Feature Extraction for ASVspoof 2019 LA")
    parser.add_argument("--part", type=str, choices=["train", "dev", "eval"], required=True,
                        help="Partition to extract: train, dev, or eval")
    parser.add_argument("--checkpoint", type=str, 
                        default="D:/Mahmud/models/ssps_kmeans_25k_uni-1/checkpoints/model_avg.pt",
                        help="Path to SSPS checkpoint")
    parser.add_argument("--protocol_dir", type=str, 
                        default="D:/Mahmud/Datasets/asvspoof_2019/LA/LA/ASVspoof2019_LA_cm_protocols",
                        help="Directory containing protocol files")
    parser.add_argument("--audio_root", type=str, 
                        default="D:/Mahmud/Datasets/asvspoof_2019",
                        help="Root directory of ASVspoof 2019 dataset")
    parser.add_argument("--output_dir", type=str, 
                        default=f"{PROJECT_ROOT}/features/SSPS_2019_LA_FrameLevel",
                        help="Output directory for features")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()
    
    PARAMS = {
        "protocol_dir": args.protocol_dir,
        "audio_root": args.audio_root,
        "output_dir": args.output_dir,
        "checkpoint_path": args.checkpoint,
        "device": args.device,
    }
    
    print(f"Extracting SSPS features for ASVspoof 2019 LA - partition: {args.part}")
    extract_partition(part=args.part, **PARAMS)
    print(f"Done! Features saved to: {args.output_dir}/{args.part}")

