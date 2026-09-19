"""
contrastive_loss.py
-------------------
Standard InfoNCE contrastive loss and combination with quantum loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.tau = temperature

    def forward(self, anchor, positive, negatives):
        """
        Args:
            anchor: (B, d)
            positive: (B, d)
            negatives: (B, N, d)
        """
        # Normalize
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives, dim=-1)
        # Positive similarity
        sim_pos = torch.sum(anchor * positive, dim=-1, keepdim=True)  # (B,1)
        # Negative similarities
        sim_neg = torch.bmm(negatives, anchor.unsqueeze(-1)).squeeze(-1)  # (B,N)
        logits = torch.cat([sim_pos, sim_neg], dim=1) / self.tau
        labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
        return F.cross_entropy(logits, labels)