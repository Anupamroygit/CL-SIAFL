"""
maml.py
-------
MAML implementation for fast adaptation to new mobility scenarios.

Paper reference:
  Section IV-E: Meta-Learned Initialization for Rapid Adaptation
  Eq. (17): θ* = argmin_θ Σ L_Ti(f_θi')
  Eq. (18): θi' = θ - α ∇_θ L_Ti(f_θ)
  Eq. (19): θ_new = θ* - α ∇_θ* L_Tnew(f_θ*)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

class MAML:
    def __init__(self, model, inner_lr=0.01, outer_lr=0.001, inner_steps=1):
        """
        Args:
            model: PyTorch model to meta-learn
            inner_lr: learning rate for task adaptation (α)
            outer_lr: learning rate for meta-update
            inner_steps: number of gradient steps per task
        """
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=outer_lr)

    def inner_update(self, task_data, loss_fn):
        """Perform inner-loop adaptation on a single task."""
        # Clone model for task-specific update
        fast_model = deepcopy(self.model)
        fast_optimizer = optim.SGD(fast_model.parameters(), lr=self.inner_lr)
        for _ in range(self.inner_steps):
            x, y = task_data
            pred = fast_model(x)
            loss = loss_fn(pred, y)
            fast_optimizer.zero_grad()
            loss.backward()
            fast_optimizer.step()
        return fast_model

    def meta_train(self, tasks, loss_fn):
        """
        Meta-training loop.
        Args:
            tasks: list of tasks, each is a tuple (support_x, support_y, query_x, query_y)
            loss_fn: loss function
        """
        meta_loss = 0.0
        for task in tasks:
            support_x, support_y, query_x, query_y = task
            # Inner update on support set
            fast_model = self.inner_update((support_x, support_y), loss_fn)
            # Evaluate on query set
            pred = fast_model(query_x)
            loss = loss_fn(pred, query_y)
            meta_loss += loss
        # Meta-update
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        return meta_loss.item()

    def adapt(self, support_x, support_y, loss_fn, steps=5):
        """Fast adaptation to a new task (Eq. 19)."""
        fast_model = deepcopy(self.model)
        fast_optimizer = optim.SGD(fast_model.parameters(), lr=self.inner_lr)
        for _ in range(steps):
            pred = fast_model(support_x)
            loss = loss_fn(pred, support_y)
            fast_optimizer.zero_grad()
            loss.backward()
            fast_optimizer.step()
        return fast_model
