"""
ASVspoof5 Feature Extraction using SSPS (SimCLR+ECAPA-TDNN) - Simplified Version.

sslsv framework'ünün s3prl bağımlılığı olmadan doğrudan ECAPA-TDNN modelini yükler.
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

# ============================================================================
# ECAPA-TDNN Model (sslsv'den alındı, bağımsız çalışır)
# ============================================================================

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


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, channels, attention_channels=128, global_context=True):
        super().__init__()
        self.global_context = global_context
        in_channels = channels * 3 if global_context else channels
        self.tdnn = TDNNBlock(in_channels, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = Conv1dSamePaddingReflect(attention_channels, channels, kernel_size=1)

    def forward(self, x):
        if self.global_context:
            L = x.size(-1)
            mean = x.mean(dim=2, keepdim=True).repeat(1, 1, L)
            std = x.std(dim=2, keepdim=True).clamp(1e-12).repeat(1, 1, L)
            attn = torch.cat([x, mean, std], dim=1)
        else:
            attn = x
        attn = self.conv(self.tanh(self.tdnn(attn)))
        attn = F.softmax(attn, dim=2)
        mean = (attn * x).sum(dim=2)
        std = torch.sqrt(((attn * (x - mean.unsqueeze(2)).pow(2)).sum(dim=2)).clamp(1e-12))
        return torch.cat((mean, std), dim=1).unsqueeze(2)


class ECAPATDNN(nn.Module):
    """ECAPA-TDNN encoder - channels: [1024, 1024, 1024, 1024, 3072]"""
    def __init__(self, mel_n_mels=80, encoder_dim=192, channels=[1024, 1024, 1024, 1024, 3072]):
        super().__init__()
        self.encoder_dim = encoder_dim
        
        # Mel spectrogram
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
        self.asp = AttentiveStatisticsPooling(channels[-1])
        self.asp_bn = nn.BatchNorm1d(channels[-1] * 2)
        self.fc = Conv1dSamePaddingReflect(channels[-1] * 2, encoder_dim, kernel_size=1)

    def forward(self, x):
        # x: (B, L) raw waveform
        if x.ndim == 1:
            x = x.unsqueeze(0)
        
        # Mel spectrogram
        x = self.mel_transform(x)  # (B, n_mels, T)
        x = (x + 1e-6).log()
        
        feats = []
        for layer in self.blocks:
            x = layer(x)
            feats.append(x)
        
        x = torch.cat(feats[1:], dim=1)
        x = self.mfa(x)
        x = self.asp(x)
        x = self.asp_bn(x)
        x = self.fc(x)
        x = x.squeeze(dim=2)
        
        return x


class SimCLRModel(nn.Module):
    """SimCLR wrapper around ECAPA-TDNN encoder."""
    def __init__(self, encoder_dim=192, channels=[1024, 1024, 1024, 1024, 3072], mel_n_mels=80):
        super().__init__()
        self.encoder = ECAPATDNN(mel_n_mels=mel_n_mels, encoder_dim=encoder_dim, channels=channels)
        
    def forward(self, x):
        return self.encoder(x)


def load_ssps_checkpoint(ckpt_path: str | Path, device: str = "cuda"):
    """Load SSPS checkpoint and return model."""
    ckpt_path = Path(ckpt_path)
    
    # Model config from ssps_kmeans_25k_uni-1 (checkpoint'tan alınan değerler)
    channels = [1024, 1024, 1024, 1024, 3072]
    encoder_dim = 512  # Checkpoint'ta 512
    mel_n_mels = 40    # Checkpoint'ta 40
    
    model = SimCLRModel(encoder_dim=encoder_dim, channels=channels, mel_n_mels=mel_n_mels).to(device)
    
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        
        # Extract encoder weights from the checkpoint
        state_dict = checkpoint.get("model", checkpoint)
        
        # Filter only encoder weights
        encoder_state = {}
        for k, v in state_dict.items():
            if k.startswith("encoder."):
                new_key = k.replace("encoder.", "encoder.")
                encoder_state[new_key] = v
        
        if encoder_state:
            model.load_state_dict(encoder_state, strict=False)
            print(f"[OK] Checkpoint yuklendi: {ckpt_path}")
        else:
            # Try loading directly
            model.load_state_dict(state_dict, strict=False)
            print(f"[OK] Checkpoint yuklendi (direct): {ckpt_path}")
    else:
        print(f"[WARN] Checkpoint bulunamadi: {ckpt_path}")
    
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
    """Extract SSPS embeddings for one partition."""
    protocol_dir = Path(protocol_dir)
    audio_root = Path(audio_root)
    output_dir = Path(output_dir)

    proto_map = {
        "train": "ASVspoof5.train.tsv",
        "dev": "ASVspoof5.dev.track_1.tsv",
        "eval": "ASVspoof5.eval.track_1.tsv",
    }
    audio_map = {
        "train": "flac_T",
        "dev": "flac_D",
        "eval": "flac_E_eval",
    }

    proto_fp = protocol_dir / proto_map[part]
    audio_dir = audio_root / audio_map[part]
    
    if not proto_fp.is_file():
        raise FileNotFoundError(f"Protocol not found: {proto_fp}")
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio dir not found: {audio_dir}")

    out_dir = output_dir / part
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n>>> SSPS modeli yukleniyor...")
    model = load_ssps_checkpoint(checkpoint_path, device)
    sr_model = 16000

    with proto_fp.open("r", encoding="utf8") as f:
        lines = f.readlines()
        if lines[0].lower().startswith("speaker") or "flac" in lines[0].lower():
            lines = lines[1:]
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

        wav, sr = torchaudio.load(str(wav_fp))
        if sr != sr_model:
            wav = torchaudio.functional.resample(wav, sr, sr_model)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        wav = wav.squeeze(0).to(device)

        with torch.inference_mode():
            emb = model(wav)  # (1, D)
        
        torch.save(emb.squeeze(0).cpu(), out_fp)


if __name__ == "__main__":
    import argparse
    
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser("SSPS Feature Extractor (Simplified)")
        parser.add_argument("--part", choices=["train", "dev", "eval"], required=True)
        parser.add_argument("--protocol_dir", required=True)
        parser.add_argument("--audio_root", required=True)
        parser.add_argument("--output_dir", required=True)
        parser.add_argument("--checkpoint", required=True)
        parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        args = parser.parse_args()
        
        extract_partition(
            part=args.part,
            protocol_dir=args.protocol_dir,
            audio_root=args.audio_root,
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint,
            device=args.device,
        )
    else:
        # Interactive mode
        CHECKPOINT = "D:/Mahmud/models/ssps/ssps_kmeans_25k_uni-1/checkpoints/model_avg.pt"
        
        # ASVspoof5 dataset lokasyonu
        ASVSPOOF5_ROOT = "D:/Mahmud/asvspoof5"
        
        PARAMS = {
            "protocol_dir": ASVSPOOF5_ROOT,  # Protokol dosyaları burada
            "audio_root": ASVSPOOF5_ROOT,    # Audio dosyaları da burada
            "output_dir": f"{PROJECT_ROOT}/features/SSPS_SimCLR_ECAPA",
            "checkpoint_path": CHECKPOINT,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
        
        # train, dev ve eval için feature çıkar
        for _part in ["train", "dev", "eval"]:
            print(f"\n>>> Processing {_part}…")
            extract_partition(part=_part, **PARAMS)

