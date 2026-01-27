"""
Tüm training'lerin minDCF hesaplama scripti
===========================================

Models klasöründeki tüm modeller için eval_scores.txt dosyalarını bulur
ve minDCF hesaplar.
"""

import numpy as np
import eval_metrics as em
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path("D:/Mahmud")
MODELS_DIR = PROJECT_ROOT / "models"

# ASVspoof5 için cost model parametreleri
cost_model = {
    'Ptar': 0.05,      # Prior probability of target speaker
    'Pnon': 0.05,      # Prior probability of nontarget speaker
    'Pspoof': 0.90,    # Prior probability of spoofing attack
    'Cmiss_asv': 1.0,  # Cost of ASV falsely rejecting target
    'Cfa_asv': 1.0,    # Cost of ASV falsely accepting nontarget
    'Cmiss_cm': 1.0,   # Cost of CM falsely rejecting target
    'Cfa_cm': 1.0,     # Cost of CM falsely accepting spoof
}

# ASV error rates
Pfa_asv = 0.05         # 5% false alarm rate
Pmiss_asv = 0.05       # 5% miss rate
Pmiss_spoof_asv = 0.05 # 5% spoof miss rate


def compute_minDCF_from_scores(scores_file: Path):
    """Bir scores dosyasından minDCF hesaplar."""
    try:
        if not scores_file.exists():
            return None, None, None, None
        
        scores = []
        labels = []
        
        with open(scores_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    try:
                        scores.append(float(parts[1]))
                        labels.append(0 if parts[2].lower() == 'bonafide' else 1)
                    except (ValueError, IndexError):
                        continue
        
        if len(scores) == 0:
            return None, None, None, None
        
        scores = np.array(scores)
        labels = np.array(labels)
        
        bonafide_scores = scores[labels == 0]
        spoof_scores = scores[labels == 1]
        
        if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
            return None, None, None, None
        
        # EER hesapla
        eer, threshold_eer = em.compute_eer(bonafide_scores, spoof_scores)
        
        # t-DCF hesapla
        tDCF_norm, thresholds = em.compute_tDCF(
            bonafide_scores,
            spoof_scores,
            Pfa_asv,
            Pmiss_asv,
            Pmiss_spoof_asv,
            cost_model,
            print_cost=False
        )
        
        minDCF = np.min(tDCF_norm)
        minDCF_idx = np.argmin(tDCF_norm)
        minDCF_threshold = thresholds[minDCF_idx]
        
        return eer, minDCF, minDCF_threshold, len(scores)
    
    except Exception as e:
        print(f"  HATA: {e}")
        return None, None, None, None


def find_all_models():
    """Tüm model klasörlerini ve eval_scores.txt dosyalarını bulur."""
    models = []
    
    if not MODELS_DIR.exists():
        print(f"Models klasörü bulunamadı: {MODELS_DIR}")
        return models
    
    for model_dir in MODELS_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        
        # hybrid_ prefix'li klasörleri ara
        if not model_dir.name.startswith('hybrid_'):
            continue
        
        eval_scores = model_dir / "eval_scores.txt"
        if eval_scores.exists():
            models.append({
                'name': model_dir.name,
                'dir': model_dir,
                'scores_file': eval_scores
            })
    
    return sorted(models, key=lambda x: x['name'])


def main():
    """Ana fonksiyon - tüm modeller için minDCF hesaplar."""
    print("=" * 80)
    print("TÜM TRAINING'LER İÇİN minDCF HESAPLAMA")
    print("=" * 80)
    print(f"Models klasörü: {MODELS_DIR}\n")
    
    models = find_all_models()
    
    if len(models) == 0:
        print("Hiç model bulunamadı!")
        return
    
    print(f"Toplam {len(models)} model bulundu.\n")
    
    results = []
    
    for i, model_info in enumerate(models, 1):
        model_name = model_info['name']
        scores_file = model_info['scores_file']
        
        print(f"[{i}/{len(models)}] {model_name}")
        print(f"  Scores: {scores_file}")
        
        eer, minDCF, threshold, num_samples = compute_minDCF_from_scores(scores_file)
        
        if eer is not None:
            results.append({
                'model': model_name,
                'eer': eer * 100,  # Percentage
                'mindcf': minDCF,
                'threshold': threshold,
                'samples': num_samples
            })
            print(f"  [OK] EER: {eer*100:.4f}%")
            print(f"  [OK] minDCF: {minDCF:.6f}")
            print(f"  [OK] Threshold: {threshold:.6f}")
            print(f"  [OK] Samples: {num_samples:,}")
        else:
            print(f"  [FAIL] Hesaplanamadi (dosya bos veya hata)")
        
        print()
    
    # Model isimlerinden bilgileri çıkar
    def parse_model_info(model_name):
        """Model isminden feature, backbone ve fusion bilgilerini çıkarır."""
        parts = model_name.replace('hybrid_', '').split('_')
        
        # Feature type
        if 'hubert' in parts:
            frontend = 'HuBERT Large L8'
        elif 'wav2vec2' in parts:
            frontend = 'Wav2Vec2 Large L8'
        elif 'wavlm' in parts:
            frontend = 'WavLM Large L8'
        else:
            frontend = 'Unknown'
        
        # Resolution
        if 'fullres' in parts:
            resolution = 'Full Resolution (750)'
        elif 'ds4' in parts:
            resolution = '4x Downsample (187)'
        elif 'ds8' in parts:
            resolution = '8x Downsample'
        else:
            resolution = 'Unknown'
        
        frontend_full = f"{frontend} ({resolution})"
        
        # Backend
        backend = 'SKA-TDNN'  # Tüm modeller SKA-TDNN kullanıyor
        
        # Fusion strategy
        if 'ssps' in model_name:
            fusion = 'Attention Fusion (SSPS + Audio)'
        else:
            fusion = 'None (Audio Only)'
        
        return frontend_full, backend, fusion
    
    # minDCF'e göre sırala
    results_sorted = sorted(results, key=lambda x: x['mindcf'])
    
    # Tablo yazdır
    print("\n" + "=" * 140)
    print("DETAYLI SONUÇ TABLOSU")
    print("=" * 140)
    print(f"{'Dataset':<20} {'Front End':<35} {'Back End':<15} {'Fusion Strategy':<35} {'EER (%)':<12} {'min-tDCF':<12} {'Model Klasör':<40}")
    print("-" * 140)
    
    for r in results_sorted:
        frontend, backend, fusion = parse_model_info(r['model'])
        dataset = "ASVspoof5 Eval"
        print(f"{dataset:<20} {frontend:<35} {backend:<15} {fusion:<35} {r['eer']:>10.4f}%  {r['mindcf']:>10.6f}  {r['model']:<40}")
    
    print("=" * 140)
    
    # Eski özet tablo
    print("\n" + "=" * 80)
    print("ÖZET SONUÇLAR (sadece metrikler)")
    print("=" * 80)
    print(f"{'Model Name':<50} {'EER (%)':<12} {'minDCF':<12} {'Samples':<10}")
    print("-" * 80)
    
    for r in results_sorted:
        print(f"{r['model']:<50} {r['eer']:>10.4f}%  {r['mindcf']:>10.6f}  {r['samples']:>10,}")
    
    print("=" * 80)
    print(f"\nToplam {len(results)} model için sonuç hesaplandı.")
    
    # En iyi modelleri göster
    if len(results) > 0:
        print("\n" + "=" * 80)
        print("EN İYİ 3 MODEL (minDCF'e göre)")
        print("=" * 80)
        for i, r in enumerate(results_sorted[:3], 1):
            print(f"{i}. {r['model']}")
            print(f"   EER: {r['eer']:.4f}% | minDCF: {r['mindcf']:.6f}")
    
    # Sonuçları dosyaya kaydet
    output_file = PROJECT_ROOT / "all_mindcf_results.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 140 + "\n")
        f.write("TÜM MODELLER İÇİN DETAYLI SONUÇ TABLOSU\n")
        f.write("=" * 140 + "\n\n")
        f.write(f"{'Dataset':<20} {'Front End':<35} {'Back End':<15} {'Fusion Strategy':<35} {'EER (%)':<12} {'min-tDCF':<12} {'Model Klasör':<40}\n")
        f.write("-" * 140 + "\n")
        
        for r in results_sorted:
            frontend, backend, fusion = parse_model_info(r['model'])
            dataset = "ASVspoof5 Eval"
            f.write(f"{dataset:<20} {frontend:<35} {backend:<15} {fusion:<35} "
                   f"{r['eer']:>10.4f}%  {r['mindcf']:>10.6f}  {r['model']:<40}\n")
        
        f.write("\n" + "=" * 140 + "\n")
        f.write("\nDETAYLI METRİKLER\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Model Name':<50} {'EER (%)':<12} {'minDCF':<12} {'Threshold':<12} {'Samples':<10}\n")
        f.write("-" * 80 + "\n")
        
        for r in results_sorted:
            f.write(f"{r['model']:<50} {r['eer']:>10.4f}%  {r['mindcf']:>10.6f}  "
                   f"{r['threshold']:>10.6f}  {r['samples']:>10,}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("En iyi model (minDCF): " + results_sorted[0]['model'] + "\n")
        f.write(f"  EER: {results_sorted[0]['eer']:.4f}%\n")
        f.write(f"  minDCF: {results_sorted[0]['mindcf']:.6f}\n")
        f.write("=" * 80 + "\n")
    
    print(f"\nSonuçlar kaydedildi: {output_file}")


if __name__ == "__main__":
    main()

