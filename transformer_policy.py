"""
transformer_policy.py
---------------------
Transformer-based policy network for closed-loop agentic decisions.

Paper reference:
  Section IV-F: Transformer-Attention Policy Network
  Eq. (20): π_θ(a|s) = Softmax( Q(s) K(a)^T / √d ) V(a)
  Eq. (21): state encoding with Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.state_proj = nn.Linear(state_dim, d_model)
        self.action_proj = nn.Linear(action_dim, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, batch_first=True),
            num_layers=num_layers
        )
        # Output head for action distribution (discrete actions)
        self.action_head = nn.Linear(d_model, action_dim)

    def forward(self, state, action_embeddings=None):
        """
        Args:
            state: (B, T, state_dim) – history of states
            action_embeddings: (B, A, d) – optional action embeddings
        Returns:
            action_logits: (B, action_dim)
        """
        # Encode state sequence
        x = self.state_proj(state)  # (B, T, d)
        x = self.transformer(x)     # (B, T, d)
        # Use last token as summary
        summary = x[:, -1, :]       # (B, d)
        # Compute action logits (Eq. 20 simplified)
        logits = self.action_head(summary)  # (B, action_dim)
        return logits

    def get_action(self, state, deterministic=False):
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        if deterministic:
            return torch.argmax(probs, dim=-1)
        else:
            return torch.multinomial(probs, 1).squeeze(-1)
