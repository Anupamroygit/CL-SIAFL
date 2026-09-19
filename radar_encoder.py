"""
radar_encoder.py
----------------
2D CNN encoder for radar range-Doppler maps.
Produces a sequence of tokens for transformer input.

Paper reference:
  Section IV-A: Sensor-Informed Multi-Modal Tokenization
"""

import torch
import torch.nn as nn

class RadarEncoder(nn.Module):
    def __init__(self, in_channels=2, d_model=256, Nr=256, Nd=128, patch_size=16):
        """
        Args:
            in_channels: 2 (real & imag) or 1 (magnitude)
            d_model: embedding dimension
            Nr, Nd: input dimensions
            patch_size: size of patches for tokenization
        """
        super().__init__()
        self.patch_size = patch_size
        # Simple CNN to extract spatial features
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # Learnable projection to d_model (if needed)
        self.proj = nn.Linear(d_model, d_model)
        # Number of patches after downsampling
        self.num_patches = (Nr // 4) * (Nd // 4)

    def forward(self, x):
        """
        Args:
            x: (B, C, Nr, Nd) range-Doppler map (C=2 for real/imag)
        Returns:
            tokens: (B, num_patches, d_model)
        """
        # x: (B, C, Nr, Nd)
        features = self.conv(x)  # (B, d_model, H', W')
        B, D, H, W = features.shape
        # Flatten spatial dimensions to sequence
        tokens = features.flatten(2).transpose(1, 2)  # (B, H*W, D)
        tokens = self.proj(tokens)
        return tokens