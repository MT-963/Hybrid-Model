"""
ASVspoof5 Hybrid Model - Konfigürasyon Dosyası
===============================================

Bu dosyayı düzenleyerek farklı feature setleri ve pathler ile 
eğitim/test yapabilirsiniz. Hyperparametreler sabit kalır.

Kullanım:
    1. Bu dosyada ACTIVE_CONFIG'u değiştir
    2. python train_asv5_hybrid_configurable.py
    3. python test_asv5_hybrid_configurable.py
"""

from pathlib import Path

# =============================================================================
# BASE PATHS - Sistemine göre güncelle
# =============================================================================
ASVSPOOF5_ROOT = Path("D:/Mahmud/Datasets/asvspoof5")
PROJECT_ROOT = Path("D:/Mahmud")

# =============================================================================
# PROTOCOL FILES
# =============================================================================
PROTOCOLS = {
    "train": ASVSPOOF5_ROOT / "ASVspoof5.train.tsv",
    "dev": ASVSPOOF5_ROOT / "ASVspoof5.dev.track_1.tsv",
    "eval": ASVSPOOF5_ROOT / "ASVspoof5.eval.track_1.tsv",
}

# =============================================================================
# FEATURE CONFIGURATIONS
# =============================================================================
FEATURE_CONFIGS = {
    
    # EN İYİ MODEL: 4x downsample config (Eval EER: 5.37%)
    "wavlm_ds4_fp16_ssps": {
        "name": "WavLM (4x downsample, fp16) + SSPS",
        "wavlm_path": PROJECT_ROOT / "features" / "WAVLM_LARGE_L8_ds4_fp16",
        "ssps_path": PROJECT_ROOT / "features" / "SSPS_SimCLR_ECAPA",
        "feat_len": 187,  # 750/4 ~ 187
        "disk_space": "~130 GB",
        "backbone_type": "skatdnn",  # "skatdnn" veya "next_tdnn"
    },
    
    # Full resolution config (downsample yok, float16) - ihtiyaç olursa
    "wavlm_fullres_fp16_ssps": {
        "name": "WavLM (Full Resolution, fp16) + SSPS",
        "wavlm_path": PROJECT_ROOT / "features" / "WAVLM_LARGE_L8_fullres_fp16",
        "ssps_path": PROJECT_ROOT / "features" / "SSPS_SimCLR_ECAPA",
        "feat_len": 750,  # Full resolution
        "disk_space": "~254 GB",
        "backbone_type": "next_tdnn",  # "skatdnn" veya "next_tdnn"
    },
    
    # HuBERT Full Resolution + SSPS with SKA-TDNN
    "hubert_fullres_fp16_ssps": {
        "name": "HuBERT (Full Resolution, fp16) + SSPS + SKA-TDNN",
        "hubert_path": PROJECT_ROOT / "features" / "HUBERT_LARGE_L8_fullres_fp16",
        "ssps_path": PROJECT_ROOT / "features" / "SSPS_SimCLR_ECAPA",
        "feat_len": 750,  # Full resolution
        "disk_space": "~254 GB",
        "backbone_type": "skatdnn",  # SKA-TDNN kullanıyor
        "use_ssps": True,  # SSPS kullanılıyor
    },
    
    # WavLM + SKA-TDNN (SSPS olmadan)
    "wavlm_ds4_fp16_skatdnn": {
        "name": "WavLM (4x downsample, fp16) + SKA-TDNN (no SSPS)",
        "wavlm_path": PROJECT_ROOT / "features" / "WAVLM_LARGE_L8_ds4_fp16",
        "feat_len": 187,  # 750/4 ~ 187
        "disk_space": "~65 GB",
        "backbone_type": "skatdnn",
        "use_ssps": False,  # SSPS kullanma
    },
    
    # WavLM Full Resolution + SKA-TDNN (SSPS olmadan)
    "wavlm_fullres_fp16_skatdnn": {
        "name": "WavLM (Full Resolution, fp16) + SKA-TDNN (no SSPS)",
        "wavlm_path": PROJECT_ROOT / "features" / "WAVLM_LARGE_L8_fullres_fp16",
        "feat_len": 750,  # Full resolution
        "disk_space": "~127 GB",
        "backbone_type": "skatdnn",
        "use_ssps": False,  # SSPS kullanma
    },
    
    # HuBERT + SKA-TDNN (SSPS olmadan)
    "hubert_fullres_fp16_skatdnn": {
        "name": "HuBERT (Full Resolution, fp16) + SKA-TDNN (no SSPS)",
        "hubert_path": PROJECT_ROOT / "features" / "HUBERT_LARGE_L8_fullres_fp16",
        "feat_len": 750,  # Full resolution
        "disk_space": "~127 GB",
        "backbone_type": "skatdnn",
        "use_ssps": False,  # SSPS kullanma
    },
    
    # Wav2Vec2 Large + SSPS + SKA-TDNN
    "wav2vec2_fullres_fp16_ssps": {
        "name": "Wav2Vec2 Large (Full Resolution, fp16) + SSPS + SKA-TDNN",
        "wav2vec2_path": PROJECT_ROOT / "features" / "WAV2VEC2_LARGE_L8_fullres_fp16",
        "ssps_path": PROJECT_ROOT / "features" / "SSPS_SimCLR_ECAPA",
        "feat_len": 750,  # Full resolution
        "disk_space": "~254 GB",
        "backbone_type": "skatdnn",
        "use_ssps": True,  # SSPS kullanılıyor
    },
    
    # Sadece SSPS Frame Level + SKA-TDNN (ASVspoof5)
    "ssps_frame_level_skatdnn": {
        "name": "SSPS (Frame-Level) Only + SKA-TDNN",
        "ssps_path": PROJECT_ROOT / "features" / "SSPS_FrameLevel",
        "feat_len": 750,  # Full resolution (yaklaşık)
        "disk_space": "~100 GB",
        "backbone_type": "skatdnn",
        "use_ssps": False, # Burada SSPS'i fusion için değil, ana feature olarak kullanacağız.
        "only_ssps": True # Yeni flag: Sadece SSPS kullan
    },
    
    # SSPS Frame Level + SKA-TDNN (ASVspoof 2019 LA)
    "ssps_2019_la_skatdnn": {
        "name": "SSPS (Frame-Level) 2019 LA Only + SKA-TDNN",
        "ssps_path": PROJECT_ROOT / "features" / "SSPS_2019_LA_FrameLevel",
        "feat_len": 750,  # Full resolution (yaklaşık)
        "disk_space": "~50 GB",
        "backbone_type": "skatdnn",
        "use_ssps": False,
        "only_ssps": True,
        "dataset": "asvspoof2019_la"  # Dataset identifier
    },
    


}

# =============================================================================
# ACTIVE CONFIGURATION - Buradan seç!
# =============================================================================
ACTIVE_CONFIG = "ssps_2019_la_skatdnn"  # SSPS 2019 LA Only

# =============================================================================
# TRAINING HYPERPARAMETERS (Anti-Overfitting Config)
# =============================================================================
TRAIN_PARAMS = {
    # Optimizer - düşük LR, yüksek regularization
    "lr": 1e-4,              # Learning rate (düşürüldü)
    "weight_decay": 1e-4,    # L2 regularization (artırıldı)
    "lr_decay": 0.5,         # LR decay factor
    "interval": 20,          # LR decay interval (daha sık)
    
    # Training - erken dur
    "batch_size": 64,        # Batch size
    "num_epochs": 100,        # Max epochs (azaltıldı)
    "patience": 20,           # Early stopping (çok erken dur!)
    "num_workers": 4,        # DataLoader workers
    
    # Model
    "emb_dim": 256,          # Embedding dimension
    "padding": "repeat",     # Padding strategy
    
    # Loss
    "add_loss": "ocsoftmax", # Loss function
    "weight_loss": 1.0,      # Loss weight
    "r_real": 0.9,           # OC-Softmax r_real
    "r_fake": 0.2,           # OC-Softmax r_fake
    "alpha": 20.0,           # OC-Softmax alpha
    
    # Stability
    "gradient_clip": 1.0,    # Gradient clipping max norm
    
    # Misc
    "seed": 598,             # Random seed
    "gpu": "0",              # GPU device
}

# =============================================================================
# OUTPUT PATHS
# =============================================================================
def get_output_path(config_name: str) -> Path:
    """Her config için benzersiz output klasörü oluşturur."""
    return PROJECT_ROOT / "models" / f"hybrid_{config_name}"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_active_config():
    """Aktif konfigürasyonu döndürür."""
    if ACTIVE_CONFIG not in FEATURE_CONFIGS:
        raise ValueError(f"Geçersiz config: {ACTIVE_CONFIG}. Seçenekler: {list(FEATURE_CONFIGS.keys())}")
    return FEATURE_CONFIGS[ACTIVE_CONFIG]

def print_config():
    """Aktif konfigürasyonu yazdırır."""
    cfg = get_active_config()
    print("=" * 60)
    print("AKTİF KONFİGÜRASYON")
    print("=" * 60)
    print(f"  Config: {ACTIVE_CONFIG}")
    print(f"  Name: {cfg['name']}")
    if 'wavlm_path' in cfg:
        print(f"  WavLM Path: {cfg['wavlm_path']}")
    if 'hubert_path' in cfg:
        print(f"  HuBERT Path: {cfg['hubert_path']}")
    if 'wav2vec2_path' in cfg:
        print(f"  Wav2Vec2 Path: {cfg['wav2vec2_path']}")
    if 'ssps_path' in cfg:
        print(f"  SSPS Path: {cfg['ssps_path']}")
    print(f"  Feature Length: {cfg['feat_len']}")
    print(f"  Backbone Type: {cfg.get('backbone_type', 'skatdnn')}")
    print(f"  Disk Space: {cfg['disk_space']}")
    print(f"  Output: {get_output_path(ACTIVE_CONFIG)}")
    print("=" * 60)

if __name__ == "__main__":
    print_config()
    print("\nTRAIN PARAMS:")
    for k, v in TRAIN_PARAMS.items():
        print(f"  {k}: {v}")

