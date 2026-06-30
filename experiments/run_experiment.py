"""End-to-end experiment runner: two-phase training pipeline.

Phase 1 — Offline base training:
    2022 Jan-May : batch training (offline)
    2022 Jun     : validation (early stopping)

Phase 2 — Tick-based retraining:
    2022 Jul-Dec : retrain on every tick (1 tick = 15 min = 1 new bar).
                   Each tick is evaluated prequentially (predict-then-train).

A batch-retrain baseline (all of 2022) is trained for a speed/accuracy
comparison against the incremental tick pipeline.

Usage:
    python -m experiments.run_experiment                    # all tickers
    python -m experiments.run_experiment --ticker AAPL      # single ticker
    python -m experiments.run_experiment --max-ticks 500    # quick run
"""

import argparse
import os
import sys
import copy
import joblib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    TICKERS, INITIAL_TRAIN_YEAR, INITIAL_TRAIN_MONTHS, VALIDATION_MONTH,
    PHASE2_YEAR, PHASE2_MONTHS, PHASE1_MONTHS, TICK_INTERVAL_MINUTES,
    LOOKBACK_WINDOW, FORECAST_HORIZON, BATCH_SIZE, DEVICE, OUTPUT_DIR,
)
from src.data.loader import load_monthly_featured, load_months_featured
from src.data.features import create_sequences, normalize_data, scale_with_scaler
from src.models.lstm_model import StockLSTM
from src.models.trainer import BatchTrainer, IncrementalTrainer
from src.incremental.incremental_learner import EWC, ReplayBuffer
from src.incremental.tick_pipeline import run_tick_retraining
from src.evaluation.metrics import compute_metrics
from src.utils.plotting import (
    plot_predictions, plot_metrics_over_months,
    plot_training_time_comparison, plot_forgetting_analysis, plot_loss_curves,
)


def predict(model, X):
    """Run inference and return numpy predictions (in y_scaler space)."""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, device=DEVICE)
        return model(X_t).cpu().numpy()


def predict_inv(model, X, y_scaler):
    """Run inference and inverse-transform predictions to original y scale."""
    raw = predict(model, X)
    return y_scaler.inverse_transform(raw.reshape(-1, 1)).ravel()


def tick_timestamps(feat: pd.DataFrame, n_ticks: int):
    """Timestamps of the 'current' bar for each tick/sequence (see create_sequences)."""
    return feat.index[LOOKBACK_WINDOW - 1: LOOKBACK_WINDOW - 1 + n_ticks]


def monthly_metrics_from_ticks(y_true, y_pred, timestamps):
    """Aggregate prequential tick errors into per-month metrics."""
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}, index=timestamps)
    rows = []
    for period, grp in df.groupby(df.index.to_period("M")):
        m = compute_metrics(grp["y_true"].values, grp["y_pred"].values)
        m["month"] = str(period)
        rows.append(m)
    return rows


def run_single_ticker(ticker: str, max_ticks: int | None = None) -> dict:
    """Run the full two-phase pipeline for one ticker."""
    print(f"\n{'='*60}")
    print(f"  TICKER: {ticker}")
    print(f"{'='*60}")

    results = {"ticker": ticker}

    # ════════════════════════════════════════════════════════════════
    #  PHASE 1 — Offline base training (static 6-month block of 2022)
    # ════════════════════════════════════════════════════════════════
    print(f"\n[PHASE 1] Offline base training on {INITIAL_TRAIN_YEAR} "
          f"{INITIAL_TRAIN_MONTHS[0]}-{VALIDATION_MONTH} (6-month block)")

    # ── 1. Load & feature-engineer the base block ───────────────────
    train_feat = load_months_featured(ticker, INITIAL_TRAIN_MONTHS, year=INITIAL_TRAIN_YEAR)
    val_feat = load_monthly_featured(ticker, VALIDATION_MONTH, year=INITIAL_TRAIN_YEAR)
    n_features = train_feat.shape[1]
    print(f"    Features: {n_features}, Train rows: {len(train_feat)}, Val rows: {len(val_feat)}")

    # ── 2. Normalize (scaler frozen after Phase 1) ──────────────────
    train_scaled, val_scaled, scaler = normalize_data(train_feat, val_feat)

    # ── 3. Sequences ────────────────────────────────────────────────
    X_train, y_train, ref_train = create_sequences(train_scaled, LOOKBACK_WINDOW, FORECAST_HORIZON)
    X_val, y_val, ref_val = create_sequences(val_scaled, LOOKBACK_WINDOW, FORECAST_HORIZON)
    print(f"    X_train: {X_train.shape}, X_val: {X_val.shape}")

    # Scale y so the model sees meaningful variance (frozen after Phase 1)
    y_scaler = StandardScaler()
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel().astype(np.float32)
    y_val_s   = y_scaler.transform(y_val.reshape(-1, 1)).ravel().astype(np.float32)

    # ── 4. Batch train (offline) ────────────────────────────────────
    print("\n[PHASE 1] Batch training LSTM (offline)...")
    model = StockLSTM(n_features=n_features)
    batch_trainer = BatchTrainer(model)
    batch_history = batch_trainer.fit(X_train, y_train_s, X_val, y_val_s)
    batch_time = batch_history["total_time"]
    results["batch_time"] = batch_time
    print(f"    Base training time: {batch_time:.1f}s")

    y_val_pred = predict_inv(model, X_val, y_scaler)
    batch_val_metrics = compute_metrics(y_val, y_val_pred)
    results["batch_val_metrics"] = batch_val_metrics
    print(f"    Base val metrics: {batch_val_metrics}")

    plot_loss_curves(batch_history, ticker, title_suffix="batch")
    plot_predictions(y_val, y_val_pred, ticker, title_suffix="batch_val")

    # ── 5. EWC + replay seeded from the base block ──────────────────
    print("\n[PHASE 1] Setting up EWC and replay buffer...")
    ewc = EWC()
    train_ds = TensorDataset(
        torch.tensor(X_train, device=DEVICE),
        torch.tensor(y_train_s, device=DEVICE),
    )
    ewc.compute_fisher(model, DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False))
    replay_buffer = ReplayBuffer.from_initial_data(X_train, y_train_s)
    print(f"    Replay buffer size: {len(replay_buffer.X)}")

    # Forgetting probe: Jan 2022 (start of the base block)
    jan_feat = load_monthly_featured(ticker, "01", year=INITIAL_TRAIN_YEAR)
    jan_scaled = scale_with_scaler(scaler, jan_feat)
    X_jan, y_jan, _ = create_sequences(jan_scaled, LOOKBACK_WINDOW, FORECAST_HORIZON)
    base_jan_metrics = compute_metrics(y_jan, predict_inv(model, X_jan, y_scaler))

    # ════════════════════════════════════════════════════════════════
    #  PHASE 2 — Tick-based retraining (subsequent 6 months of 2022)
    # ════════════════════════════════════════════════════════════════
    print(f"\n[PHASE 2] Tick-based retraining on {PHASE2_YEAR} "
          f"{PHASE2_MONTHS[0]}-{PHASE2_MONTHS[-1]} "
          f"(1 tick = {TICK_INTERVAL_MINUTES} min)")

    phase2_feat = load_months_featured(ticker, PHASE2_MONTHS, year=PHASE2_YEAR)
    phase2_scaled = scale_with_scaler(scaler, phase2_feat)
    X_ticks, y_ticks, ref_ticks = create_sequences(phase2_scaled, LOOKBACK_WINDOW, FORECAST_HORIZON)
    print(f"    Phase 2 stream: {len(X_ticks)} ticks "
          f"({len(X_ticks) * TICK_INTERVAL_MINUTES / 60:.0f} market-hours of data)")

    inc_trainer = IncrementalTrainer(model, ewc, replay_buffer)
    tick_result = run_tick_retraining(
        inc_trainer, X_ticks, y_ticks,
        y_scaler=y_scaler, max_ticks=max_ticks, label=ticker,
    )
    tick_time = tick_result["total_time"]
    results["tick_time"] = tick_time
    results["n_ticks"] = tick_result["n_ticks"]

    # Prequential metrics over the whole Phase 2 stream
    y_true_p2 = tick_result["y_true"]
    y_pred_p2 = tick_result["y_pred"]
    phase2_metrics = compute_metrics(y_true_p2, y_pred_p2)
    results["phase2_metrics"] = phase2_metrics
    print(f"    Phase 2 prequential: RMSE={phase2_metrics['RMSE']:.6f}, "
          f"R²={phase2_metrics['R2']:.4f}, DirAcc={phase2_metrics['DirAcc']:.1f}%")

    # Per-month aggregation + plots
    ts = tick_timestamps(phase2_feat, len(y_true_p2))
    monthly_rows = monthly_metrics_from_ticks(y_true_p2, y_pred_p2, ts)
    results["incremental_metrics"] = monthly_rows
    plot_metrics_over_months(pd.DataFrame(monthly_rows), ticker)
    plot_predictions(y_true_p2, y_pred_p2, ticker, title_suffix="phase2_ticks", ref_close=ref_ticks[:len(y_true_p2)])

    # Forgetting test after Phase 2
    jan_after = compute_metrics(y_jan, predict_inv(model, X_jan, y_scaler))
    forgetting_metrics = [
        {"month": "2022-06 (base)", **base_jan_metrics},
        {"month": f"after Phase 2", **jan_after},
    ]
    results["forgetting_metrics"] = forgetting_metrics
    plot_forgetting_analysis(forgetting_metrics, ticker)
    print(f"    Forgetting (Jan 2022 R²): base={base_jan_metrics['R2']:.4f} "
          f"-> after={jan_after['R2']:.4f}")

    # ════════════════════════════════════════════════════════════════
    #  Batch-retrain baseline (full 2022) for comparison
    # ════════════════════════════════════════════════════════════════
    print("\n[BASELINE] Batch retrain on all of 2022 (Jan-Dec)...")
    retrain_feat = load_months_featured(ticker, INITIAL_TRAIN_MONTHS + [VALIDATION_MONTH] + PHASE2_MONTHS,
                                        year=INITIAL_TRAIN_YEAR)
    retrain_scaled, _retrain_scaler = normalize_data(retrain_feat)
    X_full, y_full, _ = create_sequences(retrain_scaled, LOOKBACK_WINDOW, FORECAST_HORIZON)
    split_idx = int(len(X_full) * 0.8)
    X_full_train, y_full_train = X_full[:split_idx], y_full[:split_idx]
    X_full_val, y_full_val = X_full[split_idx:], y_full[split_idx:]

    y_full_scaler = StandardScaler()
    y_full_train_s = y_full_scaler.fit_transform(y_full_train.reshape(-1, 1)).ravel().astype(np.float32)
    y_full_val_s   = y_full_scaler.transform(y_full_val.reshape(-1, 1)).ravel().astype(np.float32)

    retrain_model = StockLSTM(n_features=n_features)
    retrain_history = BatchTrainer(retrain_model).fit(
        X_full_train, y_full_train_s, X_full_val, y_full_val_s)
    retrain_time = retrain_history["total_time"]
    retrain_metrics = compute_metrics(
        y_full_val, predict_inv(retrain_model, X_full_val, y_full_scaler))
    results["retrain_time"] = retrain_time
    results["retrain_metrics"] = retrain_metrics
    print(f"    Retrain time: {retrain_time:.1f}s, metrics: {retrain_metrics}")

    # ── Efficiency comparison ───────────────────────────────────────
    inc_total = batch_time + tick_time
    speedup = retrain_time / inc_total if inc_total > 0 else float("nan")
    print(f"\n[SUMMARY] {ticker}")
    print(f"    Batch retrain (full 2022): {retrain_time:.1f}s")
    print(f"    Incremental (base + ticks): {inc_total:.1f}s "
          f"(base={batch_time:.1f} + ticks={tick_time:.1f})")
    print(f"    Speedup: {speedup:.2f}x")
    plot_training_time_comparison(batch_time, tick_time, retrain_time, ticker)

    # ── Save model + frozen scalers ─────────────────────────────────
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"{ticker}_incremental_model.pt"))
    joblib.dump(scaler,   os.path.join(OUTPUT_DIR, f"{ticker}_feature_scaler.pkl"))
    joblib.dump(y_scaler, os.path.join(OUTPUT_DIR, f"{ticker}_y_scaler.pkl"))
    print(f"    Saved model + scalers to {OUTPUT_DIR}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Two-phase stock prediction experiment")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Run for a single ticker (default: all)")
    parser.add_argument("--max-ticks", type=int, default=None,
                        help="Cap Phase 2 ticks (for quick runs; default: all)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tickers = [args.ticker] if args.ticker else TICKERS
    all_results = []
    for ticker in tickers:
        all_results.append(run_single_ticker(ticker, max_ticks=args.max_ticks))

    # Save summary
    summary_rows = []
    for r in all_results:
        summary_rows.append({
            "ticker": r["ticker"],
            "n_ticks": r["n_ticks"],
            "batch_time": r["batch_time"],
            "tick_time": r["tick_time"],
            "inc_total_time": r["batch_time"] + r["tick_time"],
            "retrain_time": r["retrain_time"],
            "batch_val_RMSE": r["batch_val_metrics"]["RMSE"],
            "batch_val_R2": r["batch_val_metrics"]["R2"],
            "phase2_RMSE": r["phase2_metrics"]["RMSE"],
            "phase2_R2": r["phase2_metrics"]["R2"],
            "phase2_DirAcc": r["phase2_metrics"]["DirAcc"],
            "retrain_RMSE": r["retrain_metrics"]["RMSE"],
            "retrain_R2": r["retrain_metrics"]["R2"],
            "jan_forgetting_R2": r["forgetting_metrics"][-1]["R2"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "experiment_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n\nExperiment summary saved to {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
