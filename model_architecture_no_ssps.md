# Model Architecture: WavLM/HuBERT + SKA-TDNN (No SSPS)

## Overview
Bu model **sadece** WavLM veya HuBERT feature'larını kullanır. SSPS (Spectro-Spatial Processing) kullanılmaz.

## Architecture Flow

```
INPUT: WavLM/HuBERT Features
(B, C, T) where:
- B = batch size
- C = feature dimension (e.g., 1024 for WavLM Large L8)
- T = sequence length (e.g., 187 for ds4, 750 for fullres)
│
├─► SKA-TDNN Backbone ─────────────────────────┐
│   │                                           │
│   ├─► Conv1d(C, 1024, k=5, p=2)             │
│   │   └─► ReLU + BatchNorm1d(1024)          │
│   │                                           │
│   ├─► Bottle2neck Block 1                    │
│   │   ├─► Input: (B, 1024, T)                │
│   │   ├─► Dilation: 2                        │
│   │   ├─► SK-Attention                       │
│   │   └─► Output: x1 (B, 1024, T)           │
│   │                                           │
│   ├─► Bottle2neck Block 2                    │
│   │   ├─► Input: x + x1 (residual)           │
│   │   ├─► Dilation: 3                        │
│   │   ├─► SK-Attention                       │
│   │   └─► Output: x2 (B, 1024, T)           │
│   │                                           │
│   ├─► Bottle2neck Block 3                    │
│   │   ├─► Input: x + x1 + x2 (residual)      │
│   │   ├─► Dilation: 4                        │
│   │   ├─► SK-Attention                       │
│   │   └─► Output: x3 (B, 1024, T)           │
│   │                                           │
│   ├─► Concatenate: [x1, x2, x3]             │
│   │   └─► (B, 3072, T)                      │
│   │                                           │
│   └─► Conv1d(3072, 1536, k=1)                │
│       └─► ReLU                                │
│       └─► Output: (B, 1536, T)               │
│                                               │
└─► AdaptiveAvgPool1d(1)                        │
    └─► (B, 1536, 1) → squeeze → (B, 1536)     │
                                               │
└─► Linear(1536, 256) ────────────────────────┐
    └─► w_emb: (B, 256)                        │
                                               │
└─► Classifier ───────────────────────────────┼─►
    ├─► Linear(256, 256)                       │
    ├─► BatchNorm1d(256)                       │
    ├─► ReLU                                   │
    ├─► Dropout(0.3)                           │
    └─► Linear(256, 2)                         │
        └─► logits: (B, 2)                     │
                                               │
└─► L2 Normalize ─────────────────────────────┘
    └─► embeddings: (B, 256)
```

## Detailed Components

### 1. SKA-TDNN Backbone

**Input Shape:** `(B, C, T)` where C=1024 (WavLM) or 768 (HuBERT)

**Structure:**
```
Conv1d(in_chans → 1024, k=5)
  ↓
ReLU + BatchNorm1d(1024)
  ↓
Bottle2neck(dilation=2, scale=8)  → x1
  ↓
Bottle2neck(dilation=3, scale=8)  → x2 (with residual: x + x1)
  ↓
Bottle2neck(dilation=4, scale=8)  → x3 (with residual: x + x1 + x2)
  ↓
Concat([x1, x2, x3])              → (B, 3072, T)
  ↓
Conv1d(3072 → 1536, k=1)
  ↓
ReLU
  ↓
Output: (B, 1536, T)
```

**Bottle2neck Block:**
- Uses SK-Attention (Selective Kernel Attention)
- Multiple parallel convolutions with different kernel sizes (5, 7)
- Attention mechanism selects the best features
- Residual connections for gradient flow

### 2. Temporal Pooling

**AdaptiveAvgPool1d(1):**
- Takes: `(B, 1536, T)` 
- Outputs: `(B, 1536, 1)` → squeeze → `(B, 1536)`
- Aggregates temporal information into a single vector

### 3. Feature Projection

**Linear(1536, 256):**
- Projects backbone output to embedding dimension
- Output: `w_emb` of shape `(B, 256)`

### 4. Classifier

**Sequential:**
```
Linear(256, 256)
  ↓
BatchNorm1d(256)
  ↓
ReLU
  ↓
Dropout(0.3)          # Regularization
  ↓
Linear(256, 2)        # Binary classification (bonafide/spoof)
  ↓
logits: (B, 2)
```

### 5. Embedding Normalization

**L2 Normalize:**
- Normalizes embeddings to unit sphere
- Helps with metric learning

## Forward Pass (No SSPS)

```python
def forward(self, w, s=None):
    # w: (B, C, T) - WavLM/HuBERT features
    
    # Backbone processing
    w_out = self.wavlm_backbone(w)          # (B, 1536, T)
    
    # Temporal pooling
    w_out = self.wavlm_pool(w_out).squeeze(-1)  # (B, 1536)
    
    # Feature projection
    w_emb = self.wavlm_fc(w_out)            # (B, 256)
    
    # NO SSPS processing - use w_emb directly
    fused = w_emb                            # (B, 256)
    
    # Normalize embeddings
    emb = F.normalize(fused, dim=1)          # (B, 256)
    
    # Classification
    logits = self.classifier(fused)          # (B, 2)
    
    return emb, logits
```

## Comparison: With vs Without SSPS

### With SSPS (Hybrid Model):
```
WavLM/HuBERT ──┐
                ├─► Attention Fusion ─► Classifier
SSPS ──────────┘
```

### Without SSPS (Current Model):
```
WavLM/HuBERT ──► Classifier
```

## Key Differences

1. **No Feature Fusion**: SSPS features are not used
2. **No Attention Mechanism**: Simple direct classification from audio features
3. **Simpler Architecture**: Fewer parameters, faster inference
4. **Single Modality**: Only uses audio feature embeddings

## Model Parameters

**SKA-TDNN Backbone:**
- Input: `(B, 1024, T)` for WavLM
- Intermediate: `(B, 1024, T)` per Bottle2neck
- Output: `(B, 1536, T)`

**Projection & Classifier:**
- Pooling → `(B, 1536)`
- FC → `(B, 256)`
- Classifier → `(B, 2)`

## Training

During training, the model:
1. Processes WavLM/HuBERT features through SKA-TDNN
2. Extracts temporal representations
3. Projects to embedding space
4. Classifies as bonafide (0) or spoof (1)

**Loss**: OC-Softmax loss on embeddings + Cross-Entropy on logits

## Advantages of No SSPS Model

1. **Faster Training**: Simpler architecture, fewer parameters
2. **Lower Memory**: No SSPS feature loading/processing
3. **Easier Deployment**: Single feature modality
4. **Baseline Performance**: Good for comparing feature extraction methods

