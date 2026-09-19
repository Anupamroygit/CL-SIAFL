"""
dynamic_lora.py
---------------
Dynamic LoRA rank selection based on channel conditions (SNR).

Paper reference:
  Section IV-G: Dual-Level Action Space with Dynamic LoRA Rank
  Eq. (22): r_t = max(1, floor( r_base * (1 + tanh((SNR_min - SNR_t)/σ)) ))
  Eq. (23): W = W_0 + B_t A_t
"""

import torch
import torch.nn as nn
import math

class DynamicLoRA(nn.Module):
    def __init__(self, base_layer, r_base=8, sigma=1.0, snr_min=-5.0):
        """
        Args:
            base_layer: pre-trained linear layer (frozen)
            r_base: base LoRA rank
            sigma: scaling parameter
            snr_min: minimum SNR threshold (dB)
        """
        super().__init__()
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False
        self.r_base = r_base
        self.sigma = sigma
        self.snr_min = snr_min
        # We'll create A and B dynamically in forward pass based on rank.

    def compute_rank(self, snr):
        """Eq. (22) – higher rank for lower SNR (more challenging)."""
        # Note: paper originally had sign error; we correct it here.
        rank = self.r_base * (1 + math.tanh((self.snr_min - snr) / self.sigma))
        return max(1, int(rank))

    def forward(self, x, snr):
        """
        Args:
            x: input tensor (B, in_features)
            snr: current SNR (scalar or per-batch)
        Returns:
            output: (B, out_features)
        """
        # Get base output
        base_out = self.base_layer(x)
        # Compute dynamic rank
        r = self.compute_rank(snr)
        in_features = x.size(-1)
        out_features = base_out.size(-1)
        # Create low-rank matrices A and B on the fly
        # In practice, these would be learnable parameters.
        # For simplicity, we use random initialization here.
        A = nn.Parameter(torch.randn(r, in_features, device=x.device) * 0.01)
        B = nn.Parameter(torch.randn(out_features, r, device=x.device) * 0.01)
        lora_out = (x @ A.T) @ B.T
        return base_out + lora_out
