"""Elastic Weight Consolidation (EWC) and Replay Buffer for incremental learning."""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import DEVICE, REPLAY_RATIO


class EWC:
    """Elastic Weight Consolidation.

    Computes the diagonal Fisher Information Matrix from a data loader,
    then provides a penalty term that discourages the model from
    deviating too far from previously learned parameters.
    """

    def __init__(self):
        self.fisher: dict[str, torch.Tensor] = {}
        self.optimal_params: dict[str, torch.Tensor] = {}

    def compute_fisher(self, model: nn.Module, data_loader: DataLoader):
        """Compute diagonal Fisher Information via squared gradients."""
        model.train()
        fisher = {n: torch.zeros_like(p, device=DEVICE)
                  for n, p in model.named_parameters() if p.requires_grad}

        total_samples = 0
        criterion = nn.MSELoss()

        for xb, yb in data_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            model.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()

            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2 * xb.size(0)
            total_samples += xb.size(0)

        for n in fisher:
            fisher[n] /= total_samples

        self.fisher = fisher
        self.optimal_params = {
            n: p.detach().clone()
            for n, p in model.named_parameters() if p.requires_grad
        }

    def update(self, model: nn.Module, data_loader: DataLoader):
        """Update Fisher matrix after learning new data (running average)."""
        old_fisher = self.fisher
        self.compute_fisher(model, data_loader)

        if old_fisher:
            for n in self.fisher:
                self.fisher[n] = 0.5 * old_fisher[n] + 0.5 * self.fisher[n]

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """EWC penalty: sum(F_i * (theta_i - theta*_i)^2)."""
        if not self.fisher:
            return torch.tensor(0.0, device=DEVICE)

        loss = torch.tensor(0.0, device=DEVICE)
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.optimal_params[n]) ** 2).sum()
        return loss


class ReplayBuffer:
    """Reservoir-sampling replay buffer.

    Keeps a fixed-size subset of past training data to mix in
    during incremental updates, preventing catastrophic forgetting.
    """

    def __init__(self, max_size: int | None = None):
        self.max_size = max_size
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.n_seen = 0

    @classmethod
    def from_initial_data(cls, X: np.ndarray, y: np.ndarray,
                          ratio: float = REPLAY_RATIO) -> "ReplayBuffer":
        """Create buffer with a random subset of the initial training data."""
        size = max(1, int(len(X) * ratio))
        buf = cls(max_size=size)
        indices = np.random.choice(len(X), size=size, replace=False)
        buf.X = X[indices].copy()
        buf.y = y[indices].copy()
        buf.n_seen = len(X)
        return buf

    def add(self, X_new: np.ndarray, y_new: np.ndarray):
        """Add new samples using reservoir sampling."""
        if self.X is None:
            self.X = X_new.copy()
            self.y = y_new.copy()
            self.n_seen = len(X_new)
            return

        for i in range(len(X_new)):
            self.n_seen += 1
            if len(self.X) < self.max_size:
                self.X = np.concatenate([self.X, X_new[i:i+1]])
                self.y = np.concatenate([self.y, y_new[i:i+1]])
            else:
                j = np.random.randint(0, self.n_seen)
                if j < self.max_size:
                    self.X[j] = X_new[i]
                    self.y[j] = y_new[i]

    def sample(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return all data in the buffer (used as replay during training)."""
        if self.X is None:
            return None, None
        return self.X, self.y
