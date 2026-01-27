"""
WavLM Feature Extraction - Configurable Version
================================================

Farklı ayarlarla WavLM feature çıkarır.

Kullanım:
    # Full resolution (float16, downsample yok) - ~250 GB
    python extract_wavlm.py --mode fullres --part train
    python extract_wavlm.py --mode fullres --part dev
    python extract_wavlm.py --mode fullres --part eval
    
    # 8x downsampled (float16) - ~96 GB (mevcut)
    python extract_wavlm.py --mode ds8 --part train
    
    # 4x downsampled (float16) - ~130 GB
    python extract_wavlm.py --mode ds4 --part train
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

from config import PROJECT_ROOT

# Paths
ASVSPOOF5_ROOT = Path("D:/Mahmud/Datasets/asvspoof5")

# Audio paths
AUDIO_MAP = {
    "train": ASVSPOOF5_ROOT / "flac_T",
    "dev": ASVSPOOF5_ROOT / "flac_D", 
    "eval": ASVSPOOF5_ROOT / "flac_E_eval",
}

# Protocol paths
PROTOCOL_MAP = {
    "train": ASVSPOOF5_ROOT / "ASVspoof5.train.tsv",
    "dev": ASVSPOOF5_ROOT / "ASVspoof5.dev.track_1.tsv",
    "eval": ASVSPOOF5_ROOT / "ASVspoof5.eval.track_1.tsv",
}

# Mode configurations
MODE_CONFIGS = {
    "fullres": {
        "name": "Full Resolution (float16)",
        "downsample": 1,
        "float16": True,
        "output_dir": "WAVLM_LARGE_L8_fullres_fp16",
        "disk_estimate": "~250 GB",
    },
    "ds8": {
        "name": "8x Downsampled (float16)",
        "downsample": 8,
        "float16": True,
        "output_dir": "WAVLM_LARGE_L8_ds8_fp16",
        "disk_estimate": "~96 GB",
    },
    "ds4": {
        "name": "4x Downsampled (float16)",
        "downsample": 4,
        "float16": True,
        "output_dir": "WAVLM_LARGE_L8_ds4_fp16",
        "disk_estimate": "~130 GB",
    },
    "ds2": {
        "name": "2x Downsampled (float16)",
        "downsample": 2,
        "float16": True,
        "output_dir": "WAVLM_LARGE_L8_ds2_fp16",
        "disk_estimate": "~180 GB",
    },
}


def read_protocol(proto_path: Path):
    """Protokol dosyasından utterance ID'lerini okur."""
    import re
    items = []
    # Pattern: T_0000000000 veya D_0000000000 veya E_0000000000 (10 digit)
    pattern = re.compile(r"[TDE]_\d{10}")
    
    with open(proto_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            # Skip header if exists
            if i == 0 and ("speaker" in line.lower() or "flac" in line.lower() or "key" in line.lower()):
                continue
            
            # Find all matching IDs in line
            matches = pattern.findall(line)
            if matches:
                items.append(matches[0])  # İlk eşleşmeyi al
    
    return items


def extract_features(mode: str, part: str, layer: int = 8):
    """Feature extraction ana fonksiyonu."""
    
    if mode not in MODE_CONFIGS:
        raise ValueError(f"Gecersiz mode: {mode}. Secenekler: {list(MODE_CONFIGS.keys())}")
    if part not in AUDIO_MAP:
        raise ValueError(f"Gecersiz part: {part}. Secenekler: {list(AUDIO_MAP.keys())}")
    
    cfg = MODE_CONFIGS[mode]
    audio_dir = AUDIO_MAP[part]
    proto_path = PROTOCOL_MAP[part]
    output_dir = PROJECT_ROOT / "features" / cfg["output_dir"] / part
    
    print("=" * 60)
    print(f"WAVLM FEATURE EXTRACTION")
    print("=" * 60)
    print(f"  Mode: {cfg['name']}")
    print(f"  Part: {part}")
    print(f"  Layer: {layer}")
    print(f"  Downsample: {cfg['downsample']}x")
    print(f"  Float16: {cfg['float16']}")
    print(f"  Audio Dir: {audio_dir}")
    print(f"  Output Dir: {output_dir}")
    print(f"  Disk Estimate: {cfg['disk_estimate']}")
    print("=" * 60)
    
    # Check paths
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory bulunamadi: {audio_dir}")
    if not proto_path.exists():
        raise FileNotFoundError(f"Protocol file bulunamadi: {proto_path}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\nLoading WavLM LARGE model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torchaudio.pipelines.WAVLM_LARGE
    model = bundle.get_model().to(device)
    model.eval()
    print(f"Model loaded on {device}")
    
    # Read protocol
    items = read_protocol(proto_path)
    print(f"Total utterances: {len(items)}")
    
    # Check existing
    existing = set(f.stem for f in output_dir.glob("*.pt"))
    to_process = [uid for uid in items if uid not in existing]
    print(f"Already extracted: {len(existing)}")
    print(f"To process: {len(to_process)}")
    
    if not to_process:
        print("Nothing to process!")
        return
    
    # Process
    print("\nExtracting features...")
    for uid in tqdm(to_process, desc=part):
        # Find audio file
        audio_path = audio_dir / f"{uid}.flac"
        if not audio_path.exists():
            # Try subdirectories
            candidates = list(audio_dir.glob(f"**/{uid}.flac"))
            if candidates:
                audio_path = candidates[0]
            else:
                continue
        
        try:
            # Load audio
            wav, sr = torchaudio.load(audio_path)
            if sr != bundle.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, bundle.sample_rate)
            
            # Extract features
            with torch.no_grad():
                wav = wav.to(device)
                features, _ = model.extract_features(wav)
                feat = features[layer]  # (1, T, C)
                feat = feat.squeeze(0).transpose(0, 1)  # (C, T)
                
                # Downsample if needed
                if cfg["downsample"] > 1:
                    feat = feat[:, ::cfg["downsample"]]
                
                # Convert to float16 if needed
                if cfg["float16"]:
                    feat = feat.half()
                
                # Move to CPU and save
                feat = feat.cpu()
            
            # Save
            torch.save(feat, output_dir / f"{uid}.pt")
            
        except Exception as e:
            print(f"\nError processing {uid}: {e}")
            continue
    
    # Final stats
    final_count = len(list(output_dir.glob("*.pt")))
    print(f"\nDone! Total files: {final_count}")
    
    # Estimate size
    sample_files = list(output_dir.glob("*.pt"))[:10]
    if sample_files:
        avg_size = sum(f.stat().st_size for f in sample_files) / len(sample_files)
        total_estimate = avg_size * final_count / (1024**3)
        print(f"Estimated total size: {total_estimate:.2f} GB")


def main():
    parser = argparse.ArgumentParser("WavLM Feature Extraction")
    parser.add_argument("--mode", required=True, choices=list(MODE_CONFIGS.keys()),
                        help="Extraction mode: fullres, ds8, ds4, ds2")
    parser.add_argument("--part", required=True, choices=["train", "dev", "eval"],
                        help="Data partition: train, dev, eval")
    parser.add_argument("--layer", type=int, default=8,
                        help="WavLM layer to extract (default: 8)")
    
    args = parser.parse_args()
    
    extract_features(args.mode, args.part, args.layer)


if __name__ == "__main__":
    main()

