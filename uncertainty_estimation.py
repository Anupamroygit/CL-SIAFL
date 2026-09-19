"""
uncertainty_estimation.py
-------------------------
Monte Carlo Dropout for predictive uncertainty.

Paper reference:
  Section IV-G: Uncertainty-Aware Decision Making
  Eq. (24): y_hat = (1/T) Σ f_θt(x),  σ² = (1/T) Σ (f_θt(x) - y_hat)²
  Eq. (25): action selection based on σ²
"""

import torch
import torch.nn as nn

class MCDropout(nn.Module):
    def __init__(self, model, dropout_rate=0.1, T=20):
        """
        Args:
            model: base model (must have dropout layers)
            dropout_rate: dropout probability
            T: number of stochastic forward passes
        """
        super().__init__()
        self.model = model
        self.T = T
        self.dropout = nn.Dropout(dropout_rate)

    def enable_dropout(self):
        """Enable dropout during inference."""
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def forward(self, x):
        """
        Returns:
            mean: (B, ...) predictive mean
            var: (B, ...) predictive variance
        """
        self.model.eval()
        self.enable_dropout()
        preds = []
        for _ in range(self.T):
            preds.append(self.model(x))
        preds = torch.stack(preds, dim=0)  # (T, B, ...)
        mean = preds.mean(dim=0)
        var = preds.var(dim=0)
        return mean, var

    def select_action(self, mean, var, tau_low=0.01, tau_high=0.1):
        """Eq. (25) – choose aggressive or conservative action."""
        if var < tau_low:
            return "aggressive"
        elif var > tau_high:
            return "conservative"
        else:
            return "normal"
