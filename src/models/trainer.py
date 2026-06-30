"""Batch and incremental trainers for the LSTM model."""

import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    BATCH_SIZE, LR, INITIAL_EPOCHS, EARLY_STOPPING_PATIENCE,
    INCREMENTAL_EPOCHS, INCREMENTAL_LR, EWC_LAMBDA, REPLAY_ALPHA, DEVICE,
    GRAD_CLIP, SCHEDULER_PATIENCE,
)
from src.incremental.incremental_learner import EWC, ReplayBuffer


class BatchTrainer:
    """Train the model from scratch on the initial data (Jan-May + Jun val)."""

    def __init__(self, model: nn.Module):
        self.model = model.to(DEVICE)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5,
            patience=SCHEDULER_PATIENCE, min_lr=1e-6,
        )

    def fit(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
        epochs: int = INITIAL_EPOCHS,
    ) -> dict:
        """Train with early stopping. Returns training history dict."""
        train_ds = TensorDataset(
            torch.tensor(X_train, device=DEVICE),
            torch.tensor(y_train, device=DEVICE),
        )
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        X_val_t = torch.tensor(X_val, device=DEVICE)
        y_val_t = torch.tensor(y_val, device=DEVICE)

        history = {"train_loss": [], "val_loss": [], "epoch_time": [], "lr": []}
        best_val = float("inf")
        best_state = None
        patience_counter = 0

        start_total = time.time()
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            self.model.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                self.optimizer.zero_grad()
                pred = self.model(xb)
                loss = self.criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
                self.optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(train_ds)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val_t)
                val_loss = self.criterion(val_pred, y_val_t).item()

            # Step LR scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            elapsed = time.time() - t0
            history["train_loss"].append(epoch_loss)
            history["val_loss"].append(val_loss)
            history["epoch_time"].append(elapsed)
            history["lr"].append(current_lr)

            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d} | train={epoch_loss:.6f} | val={val_loss:.6f} | lr={current_lr:.1e}")

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

        history["total_time"] = time.time() - start_total
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return history


class IncrementalTrainer:
    """Update the model incrementally using EWC + replay buffer."""

    def __init__(self, model: nn.Module, ewc: EWC, replay_buffer: ReplayBuffer,
                 ewc_lambda: float = EWC_LAMBDA, replay_alpha: float = REPLAY_ALPHA):
        self.model = model.to(DEVICE)
        self.ewc = ewc
        self.replay_buffer = replay_buffer
        self.ewc_lambda = ewc_lambda
        self.replay_alpha = replay_alpha
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=INCREMENTAL_LR
        )

    def update(
        self,
        X_new: np.ndarray, y_new: np.ndarray,
        epochs: int = INCREMENTAL_EPOCHS,
        refresh_ewc: bool = True,
        update_replay: bool = True,
    ) -> dict:
        """Incrementally update on new data. Returns history.

        Args:
            refresh_ewc: recompute the Fisher matrix after this update. For
                per-tick streaming this is expensive, so callers may refresh
                only periodically (see ``EWC_REFRESH_EVERY``).
            update_replay: add the new samples to the replay buffer.
        """
        new_ds = TensorDataset(
            torch.tensor(X_new, device=DEVICE),
            torch.tensor(y_new, device=DEVICE),
        )
        new_loader = DataLoader(new_ds, batch_size=BATCH_SIZE, shuffle=True)

        # Get replay data
        X_replay, y_replay = self.replay_buffer.sample()
        if X_replay is not None:
            X_rep_t = torch.tensor(X_replay, device=DEVICE)
            y_rep_t = torch.tensor(y_replay, device=DEVICE)
        else:
            X_rep_t = y_rep_t = None

        history = {"train_loss": [], "epoch_time": []}
        start = time.time()

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            self.model.train()
            epoch_loss = 0.0
            for xb, yb in new_loader:
                self.optimizer.zero_grad()

                # New data loss
                pred = self.model(xb)
                loss_new = self.criterion(pred, yb)

                # EWC penalty
                loss_ewc = self.ewc.penalty(self.model)

                # Replay loss
                loss_replay = torch.tensor(0.0, device=DEVICE)
                if X_rep_t is not None:
                    rep_pred = self.model(X_rep_t)
                    loss_replay = self.criterion(rep_pred, y_rep_t)

                loss = loss_new + self.ewc_lambda * loss_ewc + self.replay_alpha * loss_replay
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
                self.optimizer.step()
                epoch_loss += loss.item() * xb.size(0)

            epoch_loss /= len(new_ds)
            history["train_loss"].append(epoch_loss)
            history["epoch_time"].append(time.time() - t0)

        history["total_time"] = time.time() - start

        # Update EWC Fisher matrix with new data
        if refresh_ewc:
            self.ewc.update(self.model, new_loader)

        # Add new data to replay buffer
        if update_replay:
            self.replay_buffer.add(X_new, y_new)

        return history
