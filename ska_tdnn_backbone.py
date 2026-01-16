"""
SKA-TDNN Backbone for Feature Processing
=========================================

SKA-TDNN'in sadece TDNN backbone kısmını kullanır.
Feature'ları (B, C, T) formatında alır ve işler.
"""

import torch
import torch.nn as nn
from collections import OrderedDict
import math
import sys
from pathlib import Path

# Import SKA-TDNN components from ska-tdnn/models/SKA_TDNN.py
try:
    # Try importing from ska-tdnn directory
    ska_tdnn_path = Path(__file__).parent / "ska-tdnn"
    if str(ska_tdnn_path) not in sys.path:
        sys.path.insert(0, str(ska_tdnn_path))
    from models.SKA_TDNN import Bottle2neck, SKAttentionModule, SEModule
except ImportError:
    # Fallback: try direct import from root
    try:
        from SKA_TDNN import Bottle2neck, SKAttentionModule, SEModule
    except ImportError:
        # Last resort: import from file directly
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "SKA_TDNN", 
            Path(__file__).parent / "ska-tdnn" / "models" / "SKA_TDNN.py"
        )
        if spec is None:
            spec = importlib.util.spec_from_file_location(
                "SKA_TDNN", 
                Path(__file__).parent / "SKA_TDNN.py"
            )
        ska_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ska_module)
        Bottle2neck = ska_module.Bottle2neck
        SKAttentionModule = ska_module.SKAttentionModule
        SEModule = ska_module.SEModule


class SKA_TDNN_Backbone(nn.Module):
    """
    SKA-TDNN backbone - sadece TDNN kısmı
    Input: (B, in_chans, T) feature tensors
    Output: (B, out_dim, T) veya (B, out_dim) pooled
    """
    
    def __init__(self, in_chans, C=1024, model_scale=8, out_dim=None):
        """
        Args:
            in_chans: Input feature dimension (e.g., HuBERT/WavLM dim)
            C: Base channel dimension (default: 1024)
            model_scale: Scale factor for Bottle2neck (default: 8)
            out_dim: Output dimension after pooling (if None, returns (B, C_out, T))
        """
        super(SKA_TDNN_Backbone, self).__init__()
        
        self.in_chans = in_chans
        self.C = C
        self.model_scale = model_scale
        self.out_dim = out_dim
        
        # Initial projection to C channels
        self.conv1 = nn.Conv1d(in_chans, C, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C)
        
        # SKA-TDNN layers
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=model_scale)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=model_scale)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=model_scale)
        
        # Final projection
        self.layer4 = nn.Conv1d(3 * C, 1536, kernel_size=1)
        
        # Optional pooling
        if out_dim is not None:
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc_out = nn.Linear(1536, out_dim)
        else:
            self.pool = None
            self.fc_out = None
    
    def forward(self, x):
        """
        Args:
            x: (B, in_chans, T) feature tensor
        Returns:
            If out_dim is None: (B, 1536, T)
            If out_dim is set: (B, out_dim)
        """
        # Initial projection
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        
        # SKA-TDNN layers with residual connections
        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)
        
        # Concatenate and project
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.layer4(x)
        x = self.relu(x)
        
        # Pooling and projection if specified
        if self.pool is not None:
            x = self.pool(x).squeeze(-1)  # (B, 1536)
            x = self.fc_out(x)  # (B, out_dim)
        
        return x

