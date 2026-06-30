"""Phase 2 tick-based retraining loop.

A *tick* is defined precisely as every 15 minutes: each new 15-min OHLCV bar
yields exactly one new sliding-window sequence. On every tick we run a
prequential (test-then-train) step:

    1. Predict the new sequence with the current model (online test).
    2. Run a single incremental update on that one sequence (retrain).

This module is shared by the main experiment runner and the ablation study so
that both compare methods under the identical streaming regime.
"""

import time

import numpy as np
import torch

from src.config import DEVICE, TICK_EPOCHS, EWC_REFRESH_EVERY


def _predict(model, X: np.ndarray) -> np.ndarray:
    """Run inference on a batch of sequences, return raw model output."""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, device=DEVICE)
        return model(X_t).cpu().numpy().ravel()


def run_tick_retraining(
    inc_trainer,
    X_ticks: np.ndarray,
    y_ticks: np.ndarray,
    *,
    y_scaler=None,
    max_ticks: int | None = None,
    tick_epochs: int = TICK_EPOCHS,
    ewc_refresh_every: int = EWC_REFRESH_EVERY,
    log_every: int = 500,
    label: str = "",
) -> dict:
    """Stream every 15-min tick, retraining once per tick (prequential).

    Args:
        inc_trainer: an ``IncrementalTrainer`` wrapping the model to retrain.
        X_ticks: (N, lookback, n_features) — one sequence per tick, in order.
        y_ticks: (N,) — target (price change) per tick, in the ORIGINAL scale.
        y_scaler: optional scaler mapping original targets to model space. If
            given, the model is trained on scaled targets and predictions are
            inverse-transformed back before being compared with ``y_ticks``.
        max_ticks: cap the number of ticks processed (handy for quick runs).
        tick_epochs: gradient passes per tick.
        ewc_refresh_every: recompute the Fisher matrix every N ticks.
        log_every: print a progress line every N ticks.
        label: tag printed in progress lines.

    Returns:
        dict with y_true, y_pred (both original scale), total_time, n_ticks.
    """
    model = inc_trainer.model
    n = len(X_ticks)
    if max_ticks is not None:
        n = min(n, max_ticks)

    y_true: list[float] = []
    y_pred: list[float] = []

    tag = f"[{label}] " if label else ""
    print(f"  {tag}streaming {n} ticks (1 tick = 15 min)...")

    start = time.time()
    for i in range(n):
        x_i = X_ticks[i : i + 1]          # (1, lookback, n_features)
        y_i = y_ticks[i : i + 1]          # (1,) original scale

        # ── 1. Prequential test: predict BEFORE retraining on this tick ──
        raw_pred = _predict(model, x_i)[0]
        pred = (
            y_scaler.inverse_transform([[raw_pred]])[0, 0]
            if y_scaler is not None else raw_pred
        )
        y_true.append(float(y_i[0]))
        y_pred.append(float(pred))

        # ── 2. Retrain on this single tick ──────────────────────────────
        y_train = (
            y_scaler.transform(y_i.reshape(-1, 1)).ravel().astype(np.float32)
            if y_scaler is not None else y_i.astype(np.float32)
        )
        refresh = (i % ewc_refresh_every == 0)
        inc_trainer.update(
            x_i.astype(np.float32), y_train,
            epochs=tick_epochs, refresh_ewc=refresh, update_replay=True,
        )

        if log_every and (i + 1) % log_every == 0:
            print(f"    {tag}tick {i + 1}/{n} done")

    total_time = time.time() - start
    print(f"  {tag}finished {n} ticks in {total_time:.1f}s "
          f"({total_time / max(n, 1) * 1000:.1f} ms/tick)")

    return {
        "y_true": np.array(y_true, dtype=np.float32),
        "y_pred": np.array(y_pred, dtype=np.float32),
        "total_time": total_time,
        "n_ticks": n,
    }
