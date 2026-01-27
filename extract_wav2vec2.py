"""
Wav2Vec2 Large Feature Extraction - Configurable Version
========================================================

Farklı ayarlarla Wav2Vec2 Large feature çıkarır.
Fairseq kullanarak checkpoint'ten model yükler.

Kullanım:
    # Full resolution (float16, downsample yok) - ~250 GB
    python extract_wav2vec2.py --mode fullres --part train
    python extract_wav2vec2.py --mode fullres --part dev
    python extract_wav2vec2.py --mode fullres --part eval
    
    # 4x downsampled (float16) - ~130 GB
    python extract_wav2vec2.py --mode ds4 --part train
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import torch
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm

# Try transformers first (no fairseq needed!)
USE_TRANSFORMERS = False
USE_FAIRSEQ = False
load_model_ensemble_and_task = None

try:
    from transformers import Wav2Vec2Model, Wav2Vec2Processor
    USE_TRANSFORMERS = True
    print("Transformers kütüphanesi bulundu - fairseq'e gerek yok!")
except ImportError:
    print("Transformers bulunamadı, fairseq deneniyor...")
    # Fairseq import - add fairseq directory to path first
    fairseq_path = Path(__file__).parent / "fairseq"
    if str(fairseq_path) not in sys.path:
        sys.path.insert(0, str(fairseq_path))
    
    try:
        import fairseq
        from fairseq import checkpoint_utils
        load_model_ensemble_and_task = checkpoint_utils.load_model_ensemble_and_task
        USE_FAIRSEQ = True
        print("Fairseq imported successfully!")
    except Exception as e:
        print(f"Fairseq de yüklenemedi: {e}")
        print("\nLütfen şunlardan birini kurun:")
        print("  pip install transformers  (ÖNERİLEN - daha kolay)")
        print("  veya")
        print("  pip install fairseq")

from config import PROJECT_ROOT

# Paths
ASVSPOOF5_ROOT = Path("D:/Mahmud/Datasets/asvspoof5")
CHECKPOINT_PATH = Path("D:/Mahmud/models/WavLarge_960/wav2vec_big_960h.pt")

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
        "output_dir": "WAV2VEC2_LARGE_L8_fullres_fp16",
        "disk_estimate": "~250 GB",
    },
    "ds8": {
        "name": "8x Downsampled (float16)",
        "downsample": 8,
        "float16": True,
        "output_dir": "WAV2VEC2_LARGE_L8_ds8_fp16",
        "disk_estimate": "~96 GB",
    },
    "ds4": {
        "name": "4x Downsampled (float16)",
        "downsample": 4,
        "float16": True,
        "output_dir": "WAV2VEC2_LARGE_L8_ds4_fp16",
        "disk_estimate": "~130 GB",
    },
    "ds2": {
        "name": "2x Downsampled (float16)",
        "downsample": 2,
        "float16": True,
        "output_dir": "WAV2VEC2_LARGE_L8_ds2_fp16",
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


def load_wav2vec2_model(checkpoint_path: Path, device: torch.device):
    """Wav2Vec2 Large modelini yükler - transformers veya fairseq kullanır."""
    print(f"Loading Wav2Vec2 Large model from: {checkpoint_path}")
    
    if USE_TRANSFORMERS:
        # Transformers kullanarak yükle (FAIRSEQ GEREKMEZ!)
        print("Using Transformers library (no fairseq needed)...")
        
        # Checkpoint path'ten model adını çıkar veya varsayılan kullan
        # Fairseq checkpoint'i varsa, transformers'a çevirmeyi dene
        checkpoint_dir = checkpoint_path.parent
        
        # Önce checkpoint dizininden transformers model yüklemeyi dene
        try:
            model = Wav2Vec2Model.from_pretrained(
                str(checkpoint_dir),
                local_files_only=True
            )
            print(f"Loaded from checkpoint directory: {checkpoint_dir}")
        except:
            # Eğer fairseq checkpoint ise, pre-trained model kullan
            # Wav2Vec2 Large 960h modeli
            print("Fairseq checkpoint detected, using pre-trained Wav2Vec2 Large model...")
            model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h")
            print("Note: Using facebook/wav2vec2-large-960h (equivalent to wav2vec_big_960h.pt)")
        
        model.eval()
        model = model.to(device)
        
        print(f"Model loaded on {device}")
        print(f"  Model type: {type(model).__name__}")
        print(f"  Hidden size: {model.config.hidden_size}")
        print(f"  Num layers: {model.config.num_hidden_layers}")
        
        return model, None, None
    
    elif USE_FAIRSEQ and load_model_ensemble_and_task is not None:
        # Fairseq kullanarak yükle
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint bulunamadi: {checkpoint_path}")
        
        print("Loading checkpoint with fairseq...")
        models, cfg, task = load_model_ensemble_and_task(
            [str(checkpoint_path)]
        )
        model = models[0]
        model.eval()
        model = model.to(device)
        
        print(f"Model loaded on {device}")
        print(f"  Model type: {type(model).__name__}")
        
        if hasattr(model, 'cfg'):
            print(f"  Normalize: {getattr(model.cfg, 'normalize', False)}")
        
        return model, cfg, task
    
    else:
        raise RuntimeError(
            "Ne transformers ne de fairseq bulunamadı!\n"
            "Lütfen şunlardan birini kurun:\n"
            "  pip install transformers  (ÖNERİLEN)\n"
            "  veya\n"
            "  pip install fairseq"
        )


def extract_features(mode: str, part: str, layer: int = 8, checkpoint_path: Path = None):
    """Feature extraction ana fonksiyonu."""
    
    if mode not in MODE_CONFIGS:
        raise ValueError(f"Gecersiz mode: {mode}. Secenekler: {list(MODE_CONFIGS.keys())}")
    if part not in AUDIO_MAP:
        raise ValueError(f"Gecersiz part: {part}. Secenekler: {list(AUDIO_MAP.keys())}")
    
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_PATH
    
    cfg = MODE_CONFIGS[mode]
    audio_dir = AUDIO_MAP[part]
    proto_path = PROTOCOL_MAP[part]
    output_dir = PROJECT_ROOT / "features" / cfg["output_dir"] / part
    
    print("=" * 60)
    print(f"WAV2VEC2 LARGE FEATURE EXTRACTION")
    print("=" * 60)
    print(f"  Mode: {cfg['name']}")
    print(f"  Part: {part}")
    print(f"  Layer: {layer}")
    print(f"  Downsample: {cfg['downsample']}x")
    print(f"  Float16: {cfg['float16']}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Audio Dir: {audio_dir}")
    print(f"  Output Dir: {output_dir}")
    print(f"  Disk Estimate: {cfg['disk_estimate']}")
    print("=" * 60)
    
    # Check paths
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory bulunamadi: {audio_dir}")
    if not proto_path.exists():
        raise FileNotFoundError(f"Protocol file bulunamadi: {proto_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint bulunamadi: {checkpoint_path}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_cfg, task = load_wav2vec2_model(checkpoint_path, device)
    
    # Wav2Vec2 sample rate (genellikle 16kHz)
    target_sr = 16000
    
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
    if USE_TRANSFORMERS:
        normalize = False  # Transformers models handle normalization internally
    elif model_cfg is not None:
        normalize = getattr(model_cfg.model, 'normalize', False) if hasattr(model_cfg, 'model') else False
    else:
        normalize = False
    
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
            if sr != target_sr:
                wav = torchaudio.functional.resample(wav, sr, target_sr)
            
            # Normalize audio if needed
            wav = wav.squeeze(0)  # (T,)
            if normalize:
                wav = F.layer_norm(wav, wav.shape)
            wav = wav.view(1, -1)  # (1, T)
            
            # Extract features
            with torch.no_grad():
                wav = wav.to(device)
                
                if USE_TRANSFORMERS:
                    # Transformers Wav2Vec2 forward pass
                    outputs = model(wav, output_hidden_states=True)
                    # Get hidden states from specified layer
                    # transformers: layer 0 is embeddings, layer 1-N are transformer layers
                    # Fairseq: layer 0 is usually first transformer layer
                    # Adjust layer index: transformers has 1 more layer (embeddings)
                    if layer < len(outputs.hidden_states):
                        feat = outputs.hidden_states[layer]  # (1, T, C)
                    else:
                        # If layer out of range, use last layer
                        feat = outputs.last_hidden_state  # (1, T, C)
                
                else:
                    # Fairseq Wav2Vec2 forward pass
                    # features_only=True: sadece feature'ları al, mask=False: masking yapma
                    result = model(source=wav, mask=False, features_only=True, layer=layer)
                    
                    # Extract features from specified layer
                    if isinstance(result, dict):
                        feat = result.get("x", result.get("features", None))
                    else:
                        feat = result
                    
                    if feat is None:
                        # Fallback: tüm layer'ları al
                        result = model(source=wav, mask=False, features_only=True)
                        if isinstance(result, dict) and "layer_results" in result:
                            # Layer results: list of (features, layer_output)
                            feat = result["layer_results"][layer][0]  # features
                        else:
                            feat = result.get("x", None)
                    
                    if feat is None:
                        print(f"\nWarning: Could not extract features for {uid}, skipping...")
                        continue
                
                # feat shape: (1, T, C) -> (C, T)
                if feat.ndim == 3:
                    feat = feat.squeeze(0).transpose(0, 1)  # (C, T)
                elif feat.ndim == 2:
                    feat = feat.transpose(0, 1)  # (C, T)
                
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
            import traceback
            traceback.print_exc()
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
    parser = argparse.ArgumentParser("Wav2Vec2 Large Feature Extraction")
    parser.add_argument("--mode", required=True, choices=list(MODE_CONFIGS.keys()),
                        help="Extraction mode: fullres, ds8, ds4, ds2")
    parser.add_argument("--part", required=True, choices=["train", "dev", "eval"],
                        help="Data partition: train, dev, eval")
    parser.add_argument("--layer", type=int, default=8,
                        help="Wav2Vec2 layer to extract (default: 8)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help=f"Path to Wav2Vec2 checkpoint (default: {CHECKPOINT_PATH})")
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else CHECKPOINT_PATH
    extract_features(args.mode, args.part, args.layer, checkpoint_path)


if __name__ == "__main__":
    main()

