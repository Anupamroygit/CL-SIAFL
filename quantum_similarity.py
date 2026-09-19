"""
quantum_similarity.py
---------------------
Quantum-inspired similarity measure for radar scenes.
Encodes scenes as quantum states with phase information and computes
similarity via Born rule.

Paper reference:
  Section IV-C: Quantum-Inspired Probabilistic Similarity Learning
  Eq. (10): |ψ_i> = Σ α_ik |k>
  Eq. (11): S_quantum(i,j) = |<ψ_i|ψ_j>|² / Z
  Eq. (12): <ψ_i|ψ_j> = Σ α_ik* α_jk
  Eq. (13): Quantum contrastive loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumSimilarity(nn.Module):
    def __init__(self, K=64, temperature=0.1):
        """
        Args:
            K: number of basis states (dimension of quantum state)
            temperature: softmax temperature τ
        """
        super().__init__()
        self.K = K
        self.tau = temperature

    def encode_phase(self, features):
        """
        Convert features to complex probability amplitudes.
        We assume features contain phase information (e.g., from radar).
        Here we use a learnable linear layer to produce phases.
        Args:
            features: (B, d) – radar embeddings
        Returns:
            alpha: (B, K) complex amplitudes
        """
        B, d = features.shape
        # Learnable projection to K phases
        phase_proj = nn.Linear(d, self.K).to(features.device)
        phases = phase_proj(features)  # (B, K)
        # Normalize amplitudes: α_ik = exp(j φ_ik) / √K
        alpha = torch.exp(1j * phases) / (self.K ** 0.5)
        return alpha

    def forward(self, emb_i, emb_j):
        """
        Compute quantum similarity between two sets of embeddings.
        Args:
            emb_i: (B, d) – anchor embeddings
            emb_j: (B, d) – positive/negative embeddings
        Returns:
            sim: (B,) similarity scores
        """
        alpha_i = self.encode_phase(emb_i)  # (B, K)
        alpha_j = self.encode_phase(emb_j)  # (B, K)
        # Inner product <ψ_i|ψ_j> = Σ α_i* α_j
        inner = torch.sum(torch.conj(alpha_i) * alpha_j, dim=-1)  # (B,)
        # Born rule: |<ψ_i|ψ_j>|²
        sim = torch.abs(inner) ** 2
        # Z is a global normalization – cancels in contrastive loss, so we omit.
        return sim

    def contrastive_loss(self, anchor, positive, negatives):
        """
        Quantum-inspired contrastive loss (Eq. 13).
        Args:
            anchor: (B, d)
            positive: (B, d)
            negatives: (B, N, d)
        Returns:
            loss: scalar
        """
        sim_pos = self.forward(anchor, positive)  # (B,)
        # Expand negatives: (B, N, d) -> compute similarity for each
        B, N, d = negatives.shape
        anchor_exp = anchor.unsqueeze(1).expand(B, N, d)
        sim_neg = self.forward(anchor_exp.reshape(-1, d),
                               negatives.reshape(-1, d)).reshape(B, N)  # (B, N)
        # Cosine similarity for text alignment (as in Eq. 13)
        # Here we assume we have text embeddings as well; simplified.
        # We'll just use the quantum similarity as the weight.
        logits = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1) / self.tau
        labels = torch.zeros(B, dtype=torch.long, device=anchor.device)
        loss = F.cross_entropy(logits, labels)
        return loss
