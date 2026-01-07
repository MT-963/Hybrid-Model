"""Compute minDCF for hybrid model"""
import numpy as np
import eval_metrics as em

# Eval scores'ları yükle
scores_file = 'models/hybrid_wavlm_ds4_fp16_ssps/eval_scores.txt'
scores = []
labels = []
with open(scores_file, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 3:
            scores.append(float(parts[1]))
            labels.append(0 if parts[2] == 'bonafide' else 1)

scores = np.array(scores)
labels = np.array(labels)

bonafide_scores = scores[labels == 0]
spoof_scores = scores[labels == 1]

# ASVspoof5 için cost model parametreleri (ASVspoof 2019/2021 standard)
cost_model = {
    'Ptar': 0.05,      # Prior probability of target speaker
    'Pnon': 0.05,      # Prior probability of nontarget speaker
    'Pspoof': 0.90,    # Prior probability of spoofing attack
    'Cmiss_asv': 1.0,  # Cost of ASV falsely rejecting target
    'Cfa_asv': 1.0,    # Cost of ASV falsely accepting nontarget
    'Cmiss_cm': 1.0,   # Cost of CM falsely rejecting target
    'Cfa_cm': 1.0,     # Cost of CM falsely accepting spoof
}

# ASV error rates (ASVspoof5 için resmi parametreler)
# ASVspoof5'te genellikle daha iyi ASV performansı varsayılır
Pfa_asv = 0.05    # 5% false alarm rate
Pmiss_asv = 0.05  # 5% miss rate
Pmiss_spoof_asv = 0.05  # 5% spoof miss rate (ASVspoof5 için daha doğru)
# Not: Bu değer C2/C1 = 19 çarpanını verir (18 yerine)

# t-DCF hesapla
tDCF_norm, thresholds = em.compute_tDCF(
    bonafide_scores, 
    spoof_scores,
    Pfa_asv,
    Pmiss_asv,
    Pmiss_spoof_asv,
    cost_model,
    print_cost=True
)

minDCF = np.min(tDCF_norm)
minDCF_idx = np.argmin(tDCF_norm)
minDCF_threshold = thresholds[minDCF_idx]

print('\n' + '=' * 60)
print('MINIMUM DCF (minDCF) SONUÇLARI')
print('=' * 60)
print(f'minDCF: {minDCF:.6f}')
print(f'Optimal Threshold: {minDCF_threshold:.6f}')
print('=' * 60)

