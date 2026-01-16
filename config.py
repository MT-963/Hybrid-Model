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
ASVSPOOF5_ROOT = Path("D:/Mahmud/asvspoof5")
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
        "backbone_type": "skatdnn",  # "skatdnn" veya "next_tdnn"
    },
    


}

# =============================================================================
# ACTIVE CONFIGURATION - Buradan seç!
# =============================================================================
ACTIVE_CONFIG = "wavlm_ds4_fp16_ssps"  # DS4 - best generalization

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
    print(f"  WavLM Path: {cfg['wavlm_path']}")
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

