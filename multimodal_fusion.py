"""
multimodal_fusion.py
--------------------
Cross-attention fusion of radar tokens with IMU and GPS embeddings.

Paper reference:
  Section IV-A: Multi-Modal Tokenization
  Eq. (5): E_fusion = CrossAttn(E_radar, [E_imu; E_gps])
  Eq. (6): CrossAttn(Q,K,V) = softmax(QK^T/√d) V
"""

import torch
import torch.nn as nn

class MultimodalFusion(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=2):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, radar_tokens, imu_tokens, gps_tokens):
        """
        Args:
            radar_tokens: (B, N_r, d)
            imu_tokens: (B, N_i, d)
            gps_tokens: (B, N_g, d)
        Returns:
            fused: (B, N_r, d) – radar tokens enriched with motion context
        """
        # Concatenate IMU and GPS as key/value
        context = torch.cat([imu_tokens, gps_tokens], dim=1)  # (B, N_i+N_g, d)
        # Cross-attention: query=radar, key=value=context
        attn_out, _ = self.cross_attn(radar_tokens, context, context)
        x = self.norm1(radar_tokens + self.dropout(attn_out))
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x