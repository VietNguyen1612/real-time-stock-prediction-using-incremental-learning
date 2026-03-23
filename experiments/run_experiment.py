"""End-to-end experiment runner: batch train -> incremental updates -> comparison.

Pipeline:
  2022 Jan-May:  Batch training
  2022 Jun:      Validation (early stopping)
  2022 Jul-Dec → 2023 → 2024 → 2025 Jan-Nov:  Incremental learning (80/20 split)
  2025 Dec:      Final unseen test

Usage:
    python -m experiments.run_experiment              # all tickers
    python -m experiments.run_experiment --ticker AAPL # single ticker
"""

import argparse
import os
import sys
import time
import json
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
    INCREMENTAL_SCHEDULE, INC_TEST_RATIO, TEST_YEAR, TEST_MONTH,
    LOOKBACK_WINDOW, FORECAST_HORIZON, BATCH_SIZE, DEVICE, OUTPUT_DIR,
)
from src.data.loader import load_months, load_monthly_csv
from src.data.features import compute_features, create_sequences, normalize_data, scale_with_scaler, returns_to_prices
from src.models.lstm_model import StockLSTM
from src.models.trainer import BatchTrainer, IncrementalTrainer
from src.incremental.incremental_learner import EWC, ReplayBuffer
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


def run_single_ticker(ticker: str) -> dict:
    """Run the full pipeline for one ticker."""
    print(f"\n{'='*60}")
    print(f"  TICKER: {ticker}")
    print(f"{'='*60}")

    results = {"ticker": ticker}

    # ── 1. Load & feature-engineer initial data ─────────────────────
    print(f"\n[1] Loading initial training data ({INITIAL_TRAIN_YEAR} Jan-May)...")
    train_raw = load_months(ticker, INITIAL_TRAIN_MONTHS, year=INITIAL_TRAIN_YEAR)
    val_raw = load_monthly_csv(ticker, VALIDATION_MONTH, year=INITIAL_TRAIN_YEAR)

    train_feat = compute_features(train_raw)
    val_feat = compute_features(val_raw)

    n_features = train_feat.shape[1]
    print(f"    Features: {n_features}, Train rows: {len(train_feat)}, Val rows: {len(val_feat)}")

    # ── 2. Normalize ────────────────────────────────────────────────
    print("[2] Normalizing data...")
    train_scaled, val_scaled, scaler = normalize_data(train_feat, val_feat)

    # Keep track of all data seen so far (for scaler refitting)
    all_feat_seen = pd.concat([train_feat, val_feat])

    # ── 3. Create sequences ─────────────────────────────────────────
    print("[3] Creating sequences (target = returns)...")
    X_train, y_train, ref_train = create_sequences(train_scaled, LOOKBACK_WINDOW, FORECAST_HORIZON)
    X_val, y_val, ref_val = create_sequences(val_scaled, LOOKBACK_WINDOW, FORECAST_HORIZON)
    print(f"    X_train: {X_train.shape}, X_val: {X_val.shape}")

    # Scale y to zero-mean unit-variance so the model sees meaningful variance
    y_scaler = StandardScaler()
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel().astype(np.float32)
    y_val_s   = y_scaler.transform(y_val.reshape(-1, 1)).ravel().astype(np.float32)

    # ── 4. Batch train ──────────────────────────────────────────────
    print("\n[4] Batch training LSTM...")
    model = StockLSTM(n_features=n_features)
    batch_trainer = BatchTrainer(model)
    batch_history = batch_trainer.fit(X_train, y_train_s, X_val, y_val_s)
    batch_time = batch_history["total_time"]
    print(f"    Batch training time: {batch_time:.1f}s")
    results["batch_time"] = batch_time

    # Evaluate batch model on validation
    y_val_pred = predict_inv(model, X_val, y_scaler)
    batch_val_metrics = compute_metrics(y_val, y_val_pred)
    print(f"    Batch val metrics: {batch_val_metrics}")
    results["batch_val_metrics"] = batch_val_metrics

    plot_loss_curves(batch_history, ticker, title_suffix="batch")
    plot_predictions(y_val, y_val_pred, ticker, title_suffix="batch_val")

    # ── 5. Prepare for incremental learning ─────────────────────────
    print("\n[5] Setting up EWC and replay buffer...")
    ewc = EWC()
    train_ds = TensorDataset(
        torch.tensor(X_train, device=DEVICE),
        torch.tensor(y_train_s, device=DEVICE),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    ewc.compute_fisher(model, train_loader)

    replay_buffer = ReplayBuffer.from_initial_data(X_train, y_train_s)
    print(f"    Replay buffer size: {len(replay_buffer.X)}")

    # Save a copy of the model for forgetting test baseline
    baseline_model_state = copy.deepcopy(model.state_dict())

    # Prepare forgetting test data (use Jan 2022 data)
    jan_raw = load_monthly_csv(ticker, "01", year=INITIAL_TRAIN_YEAR)
    jan_feat = compute_features(jan_raw)
    jan_scaled = scale_with_scaler(scaler, jan_feat)
    X_jan, y_jan, ref_jan = create_sequences(jan_scaled, LOOKBACK_WINDOW, FORECAST_HORIZON)

    # Baseline forgetting metric
    y_jan_pred_baseline = predict_inv(model, X_jan, y_scaler)
    baseline_jan_metrics = compute_metrics(y_jan, y_jan_pred_baseline)

    # ── 6. Incremental updates (2022 Jul → 2025 Nov) ────────────────
    print(f"\n[6] Incremental updates ({len(INCREMENTAL_SCHEDULE)} months, 80/20 train/test split)...")
    inc_trainer = IncrementalTrainer(model, ewc, replay_buffer)

    incremental_metrics = []
    incremental_times = []
    forgetting_metrics = [{"month": "2022-06 (baseline)", **baseline_jan_metrics}]

    for year, month in INCREMENTAL_SCHEDULE:
        label = f"{year}-{month}"
        print(f"\n  --- {label} ---")

        # Load and process month data
        month_raw = load_monthly_csv(ticker, month, year=year)
        month_feat = compute_features(month_raw)

        # Refit scaler with all data seen so far (expanding window)
        all_feat_seen = pd.concat([all_feat_seen, month_feat])
        scaler = normalize_data(all_feat_seen)[-1]

        month_scaled = scale_with_scaler(scaler, month_feat)
        X_month, y_month, ref_month = create_sequences(month_scaled, LOOKBACK_WINDOW, FORECAST_HORIZON)

        # Split: first 80% for training, last 20% for testing (temporal order)
        split = int(len(X_month) * (1 - INC_TEST_RATIO))
        X_inc_train, X_inc_test = X_month[:split], X_month[split:]
        y_inc_train, y_inc_test = y_month[:split], y_month[split:]
        print(f"    Samples: {len(X_month)} (train={split}, test={len(X_month)-split})")

        # Train only on the training portion
        y_inc_train_s = y_scaler.transform(y_inc_train.reshape(-1, 1)).ravel().astype(np.float32)
        inc_history = inc_trainer.update(X_inc_train, y_inc_train_s)
        inc_time = inc_history["total_time"]
        incremental_times.append(inc_time)
        print(f"    Inc. training time: {inc_time:.1f}s")

        # Evaluate on held-out test portion (never trained on)
        y_test_pred = predict_inv(model, X_inc_test, y_scaler)
        month_metrics = compute_metrics(y_inc_test, y_test_pred)
        month_metrics["month"] = label
        incremental_metrics.append(month_metrics)
        print(f"    Test metrics: RMSE={month_metrics['RMSE']:.6f}, R²={month_metrics['R2']:.4f}, DirAcc={month_metrics['DirAcc']:.1f}%")

        # Forgetting test on Jan 2022 data (re-scale with current scaler)
        jan_scaled = scale_with_scaler(scaler, jan_feat)
        X_jan, y_jan, ref_jan = create_sequences(jan_scaled, LOOKBACK_WINDOW, FORECAST_HORIZON)
        y_jan_pred = predict_inv(model, X_jan, y_scaler)
        jan_metrics = compute_metrics(y_jan, y_jan_pred)
        jan_metrics["month"] = label
        forgetting_metrics.append(jan_metrics)
        print(f"    Jan forgetting: RMSE={jan_metrics['RMSE']:.6f}, R²={jan_metrics['R2']:.4f}, DirAcc={jan_metrics['DirAcc']:.1f}%")

    results["incremental_metrics"] = incremental_metrics
    results["incremental_times"] = incremental_times
    results["forgetting_metrics"] = forgetting_metrics

    # ── 7. Batch retrain baseline (for comparison) ──────────────────
    print("\n[7] Batch retrain baseline (all data 2022-2025 Nov)...")
    # Load all data that incremental model has seen
    retrain_feat = all_feat_seen.copy()
    retrain_scaled_result = normalize_data(retrain_feat)
    retrain_scaled = retrain_scaled_result[0]
    retrain_scaler = retrain_scaled_result[-1]
    X_full, y_full, ref_full = create_sequences(retrain_scaled, LOOKBACK_WINDOW, FORECAST_HORIZON)

    # Split: use last 20% as validation
    split_idx = int(len(X_full) * 0.8)
    X_full_train, y_full_train = X_full[:split_idx], y_full[:split_idx]
    X_full_val, y_full_val = X_full[split_idx:], y_full[split_idx:]

    y_full_scaler = StandardScaler()
    y_full_train_s = y_full_scaler.fit_transform(y_full_train.reshape(-1, 1)).ravel().astype(np.float32)
    y_full_val_s   = y_full_scaler.transform(y_full_val.reshape(-1, 1)).ravel().astype(np.float32)

    retrain_model = StockLSTM(n_features=n_features)
    retrain_trainer = BatchTrainer(retrain_model)
    retrain_history = retrain_trainer.fit(X_full_train, y_full_train_s, X_full_val, y_full_val_s)
    retrain_time = retrain_history["total_time"]
    results["retrain_time"] = retrain_time

    y_retrain_pred = predict_inv(retrain_model, X_full_val, y_full_scaler)
    retrain_metrics = compute_metrics(y_full_val, y_retrain_pred)
    results["retrain_metrics"] = retrain_metrics
    print(f"    Retrain time: {retrain_time:.1f}s")
    print(f"    Retrain metrics: {retrain_metrics}")

    # ── 8. Comparison summary ───────────────────────────────────────
    total_inc_time = sum(incremental_times) + batch_time
    print(f"\n[8] Summary for {ticker}:")
    print(f"    Batch retrain total time: {retrain_time:.1f}s")
    print(f"    Incremental total time:   {total_inc_time:.1f}s  (batch={batch_time:.1f} + inc={sum(incremental_times):.1f})")
    print(f"    Speedup: {retrain_time / total_inc_time:.2f}x" if total_inc_time > 0 else "    N/A")

    # ── 9. Plots ────────────────────────────────────────────────────
    print("\n[9] Generating plots...")
    metrics_df = pd.DataFrame(incremental_metrics)
    plot_metrics_over_months(metrics_df, ticker)
    plot_training_time_comparison(batch_time, incremental_times, ticker)
    plot_forgetting_analysis(forgetting_metrics, ticker)

    # Plot predictions on last incremental month (2025-11)
    last_year, last_month = INCREMENTAL_SCHEDULE[-1]
    last_raw = load_monthly_csv(ticker, last_month, year=last_year)
    last_feat = compute_features(last_raw)
    last_scaled = scale_with_scaler(scaler, last_feat)
    X_last, y_last, ref_last = create_sequences(last_scaled, LOOKBACK_WINDOW, FORECAST_HORIZON)
    y_last_pred = predict_inv(model, X_last, y_scaler)
    plot_predictions(y_last, y_last_pred, ticker, title_suffix="inc_2025_11", ref_close=ref_last)

    # ── 10. Final test on 2025 Dec (completely unseen) ──────────────
    print(f"\n[10] Final test on {TEST_YEAR}-{TEST_MONTH} (never seen during training)...")
    test_raw = load_monthly_csv(ticker, TEST_MONTH, year=TEST_YEAR)
    test_feat = compute_features(test_raw)
    test_scaled = scale_with_scaler(scaler, test_feat)
    X_test, y_test, ref_test = create_sequences(test_scaled, LOOKBACK_WINDOW, FORECAST_HORIZON)
    print(f"    Test samples: {len(X_test)}")

    # Incremental model on 2025 Dec
    y_test_pred_inc = predict_inv(model, X_test, y_scaler)
    test_inc_metrics = compute_metrics(y_test, y_test_pred_inc)
    results["test_2025_inc_metrics"] = test_inc_metrics
    print(f"    Incremental model: RMSE={test_inc_metrics['RMSE']:.6f}, R²={test_inc_metrics['R2']:.4f}, DirAcc={test_inc_metrics['DirAcc']:.1f}%")

    # Retrain model on 2025 Dec
    y_test_pred_retrain = predict_inv(retrain_model, X_test, y_full_scaler)
    test_retrain_metrics = compute_metrics(y_test, y_test_pred_retrain)
    results["test_2025_retrain_metrics"] = test_retrain_metrics
    print(f"    Retrain model:     RMSE={test_retrain_metrics['RMSE']:.6f}, R²={test_retrain_metrics['R2']:.4f}, DirAcc={test_retrain_metrics['DirAcc']:.1f}%")

    plot_predictions(y_test, y_test_pred_inc, ticker, title_suffix="test_2025_dec", ref_close=ref_test)

    # Save model and scalers
    model_path = os.path.join(OUTPUT_DIR, f"{ticker}_incremental_model.pt")
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler,   os.path.join(OUTPUT_DIR, f"{ticker}_feature_scaler.pkl"))
    joblib.dump(y_scaler, os.path.join(OUTPUT_DIR, f"{ticker}_y_scaler.pkl"))
    print(f"    Model saved to {model_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Stock prediction experiment")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Run for a single ticker (default: all)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tickers = [args.ticker] if args.ticker else TICKERS
    all_results = []

    for ticker in tickers:
        result = run_single_ticker(ticker)
        all_results.append(result)

    # Save summary
    summary_rows = []
    for r in all_results:
        row = {
            "ticker": r["ticker"],
            "batch_time": r["batch_time"],
            "retrain_time": r["retrain_time"],
            "inc_total_time": r["batch_time"] + sum(r["incremental_times"]),
            "batch_val_RMSE": r["batch_val_metrics"]["RMSE"],
            "batch_val_R2": r["batch_val_metrics"]["R2"],
            "retrain_RMSE": r["retrain_metrics"]["RMSE"],
            "retrain_R2": r["retrain_metrics"]["R2"],
            "final_inc_RMSE": r["incremental_metrics"][-1]["RMSE"],
            "final_inc_R2": r["incremental_metrics"][-1]["R2"],
            "final_inc_DirAcc": r["incremental_metrics"][-1]["DirAcc"],
            "final_jan_forgetting_RMSE": r["forgetting_metrics"][-1]["RMSE"],
            "final_jan_forgetting_R2": r["forgetting_metrics"][-1]["R2"],
            "test_2025dec_inc_RMSE": r["test_2025_inc_metrics"]["RMSE"],
            "test_2025dec_inc_R2": r["test_2025_inc_metrics"]["R2"],
            "test_2025dec_inc_DirAcc": r["test_2025_inc_metrics"]["DirAcc"],
            "test_2025dec_retrain_RMSE": r["test_2025_retrain_metrics"]["RMSE"],
            "test_2025dec_retrain_R2": r["test_2025_retrain_metrics"]["R2"],
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "experiment_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n\nExperiment summary saved to {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
