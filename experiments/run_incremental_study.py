"""Incremental Learning Study: ablation, multi-year, and forgetting analysis.

This script focuses on the incremental learning component:
1. Ablation study: Fine-tune vs EWC only vs Replay only vs EWC+Replay
2. Extended incremental learning across 2022-2024 (30 months)
3. Forgetting heatmap across all past months
4. Unseen year evaluation (2023/2024 as test data)

Usage:
    python -m experiments.run_incremental_study                # all tickers
    python -m experiments.run_incremental_study --ticker AAPL  # single ticker
"""

import argparse
import os
import sys
import time
import copy

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    TICKERS, INITIAL_TRAIN_MONTHS, VALIDATION_MONTH,
    LOOKBACK_WINDOW, FORECAST_HORIZON, BATCH_SIZE, DEVICE, OUTPUT_DIR,
    EWC_LAMBDA, REPLAY_ALPHA,
)
from src.data.loader import load_months, load_monthly_csv, load_monthly_featured, load_months_featured
from src.data.features import compute_features, create_sequences, normalize_data, scale_with_scaler, returns_to_prices
from src.models.lstm_model import StockLSTM
from src.models.trainer import BatchTrainer, IncrementalTrainer
from src.incremental.incremental_learner import EWC, ReplayBuffer
from src.evaluation.metrics import compute_metrics
from src.utils.plotting import (
    plot_predictions, plot_loss_curves,
    plot_ablation_comparison, plot_ablation_forgetting, plot_forgetting_heatmap,
)


ALL_MONTHS = [f"{m:02d}" for m in range(1, 13)]
INCREMENTAL_MONTHS_2022 = ["07", "08", "09", "10", "11", "12"]


def predict(model, X):
    """Run inference and return numpy predictions."""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, device=DEVICE)
        return model(X_t).cpu().numpy()


def load_and_prepare(ticker, month, year, scaler):
    """Load a month's data with features, scale, create sequences."""
    feat = load_monthly_featured(ticker, month, year=year)
    scaled = scale_with_scaler(scaler, feat)
    X, y, ref = create_sequences(scaled, LOOKBACK_WINDOW, FORECAST_HORIZON)
    return X, y, ref


def train_batch_model(ticker, n_features, X_train, y_train, X_val, y_val):
    """Train the base LSTM model on initial data. Returns model and history."""
    model = StockLSTM(n_features=n_features)
    trainer = BatchTrainer(model)
    history = trainer.fit(X_train, y_train, X_val, y_val)
    return model, history


def run_incremental_variant(
    base_state_dict, n_features, ticker, scaler,
    incremental_schedule, jan_data,
    use_ewc, use_replay,
    X_train, y_train,
    method_name,
):
    """Run one incremental learning variant (for ablation).

    Args:
        base_state_dict: initial model weights (after batch training)
        incremental_schedule: list of (year, month) tuples
        jan_data: (X_jan, y_jan) for forgetting test
        use_ewc: whether to use EWC penalty
        use_replay: whether to use replay buffer

    Returns:
        dict with incremental_metrics, forgetting_metrics, times
    """
    # Fresh model from batch checkpoint
    model = StockLSTM(n_features=n_features)
    model.load_state_dict(copy.deepcopy(base_state_dict))
    model.to(DEVICE)

    # Setup EWC
    ewc = EWC()
    if use_ewc:
        train_ds = TensorDataset(
            torch.tensor(X_train, device=DEVICE),
            torch.tensor(y_train, device=DEVICE),
        )
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
        ewc.compute_fisher(model, train_loader)

    # Setup replay buffer
    replay_buffer = ReplayBuffer(max_size=1)  # dummy
    if use_replay:
        replay_buffer = ReplayBuffer.from_initial_data(X_train, y_train)

    # Set lambda/alpha based on variant
    ewc_lambda = EWC_LAMBDA if use_ewc else 0.0
    replay_alpha = REPLAY_ALPHA if use_replay else 0.0

    inc_trainer = IncrementalTrainer(model, ewc, replay_buffer,
                                     ewc_lambda=ewc_lambda,
                                     replay_alpha=replay_alpha)

    X_jan, y_jan = jan_data
    y_jan_pred_baseline = predict(model, X_jan)
    baseline_jan = compute_metrics(y_jan, y_jan_pred_baseline)

    incremental_metrics = []
    forgetting_metrics = [{"month": "baseline", **baseline_jan}]
    times = []

    for year, month in incremental_schedule:
        label = f"{year}-{month}"
        X_m, y_m, _ = load_and_prepare(ticker, month, year, scaler)

        inc_history = inc_trainer.update(X_m, y_m)

        inc_time = inc_history["total_time"]
        times.append(inc_time)

        # Evaluate on current month
        y_pred = predict(model, X_m)
        month_metrics = compute_metrics(y_m, y_pred)
        month_metrics["month"] = label
        incremental_metrics.append(month_metrics)

        # Forgetting test on Jan 2022
        y_jan_pred = predict(model, X_jan)
        jan_metrics = compute_metrics(y_jan, y_jan_pred)
        jan_metrics["month"] = label
        forgetting_metrics.append(jan_metrics)

        print(f"    [{method_name}] {label}: RMSE={month_metrics['RMSE']:.6f}, "
              f"R²={month_metrics['R2']:.4f}, DirAcc={month_metrics['DirAcc']:.1f}%, "
              f"Jan R²={jan_metrics['R2']:.4f}")

    return {
        "model": model,
        "incremental_metrics": incremental_metrics,
        "forgetting_metrics": forgetting_metrics,
        "times": times,
    }


def run_forgetting_heatmap(model_state, n_features, ticker, scaler,
                           incremental_schedule, eval_months,
                           X_train, y_train):
    """Run EWC+Replay and evaluate on ALL past months after each update.

    Args:
        eval_months: list of (year, month) to evaluate on after each update

    Returns:
        pd.DataFrame heatmap (rows=eval_month, cols=after_update_month)
    """
    model = StockLSTM(n_features=n_features)
    model.load_state_dict(copy.deepcopy(model_state))
    model.to(DEVICE)

    ewc = EWC()
    train_ds = TensorDataset(
        torch.tensor(X_train, device=DEVICE),
        torch.tensor(y_train, device=DEVICE),
    )
    ewc.compute_fisher(model, DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False))
    replay_buffer = ReplayBuffer.from_initial_data(X_train, y_train)
    inc_trainer = IncrementalTrainer(model, ewc, replay_buffer)

    # Preload all eval month data
    eval_data = {}
    for yr, mo in eval_months:
        label = f"{yr}-{mo}"
        try:
            X_e, y_e, _ = load_and_prepare(ticker, mo, yr, scaler)
            eval_data[label] = (X_e, y_e)
        except Exception:
            pass

    # Columns = after training on month, Rows = eval month
    update_labels = [f"{yr}-{mo}" for yr, mo in incremental_schedule]
    heatmap = pd.DataFrame(index=list(eval_data.keys()), columns=update_labels, dtype=float)

    for year, month in incremental_schedule:
        update_label = f"{year}-{month}"
        X_m, y_m, _ = load_and_prepare(ticker, month, year, scaler)
        inc_trainer.update(X_m, y_m)

        # Evaluate on all past months
        for eval_label, (X_e, y_e) in eval_data.items():
            y_pred = predict(model, X_e)
            metrics = compute_metrics(y_e, y_pred)
            heatmap.loc[eval_label, update_label] = metrics["R2"]

    return heatmap


def run_ticker_study(ticker: str) -> dict:
    """Run the full incremental learning study for one ticker."""
    print(f"\n{'='*70}")
    print(f"  INCREMENTAL LEARNING STUDY: {ticker}")
    print(f"{'='*70}")

    # ── 1. Load & prepare initial data (2022 Jan-May train, Jun val) ──
    print("\n[1] Loading initial data (2022 Jan-May + Jun)...")
    train_feat = load_months_featured(ticker, INITIAL_TRAIN_MONTHS, year=2022)
    val_feat = load_monthly_featured(ticker, VALIDATION_MONTH, year=2022)
    n_features = train_feat.shape[1]

    train_scaled, val_scaled, scaler = normalize_data(train_feat, val_feat)
    X_train, y_train, ref_train = create_sequences(train_scaled, LOOKBACK_WINDOW, FORECAST_HORIZON)
    X_val, y_val, ref_val = create_sequences(val_scaled, LOOKBACK_WINDOW, FORECAST_HORIZON)
    print(f"    Features: {n_features}, Train: {X_train.shape}, Val: {X_val.shape}")

    # ── 2. Batch training ─────────────────────────────────────────────
    print("\n[2] Batch training LSTM on 2022 Jan-May...")
    model, batch_history = train_batch_model(ticker, n_features, X_train, y_train, X_val, y_val)
    base_state = copy.deepcopy(model.state_dict())

    y_val_pred = predict(model, X_val)
    batch_metrics = compute_metrics(y_val, y_val_pred)
    print(f"    Batch val: RMSE={batch_metrics['RMSE']:.6f}, R²={batch_metrics['R2']:.4f}")
    plot_loss_curves(batch_history, ticker, title_suffix="batch_inc_study")

    # ── 3. Prepare Jan 2022 for forgetting test ───────────────────────
    X_jan, y_jan, _ = load_and_prepare(ticker, "01", 2022, scaler)

    # ── 4. Build incremental schedule ─────────────────────────────────
    # 2022 Jul-Dec + 2023 Jan-Dec + 2024 Jan-Dec = 30 months
    schedule = []
    for mo in INCREMENTAL_MONTHS_2022:
        schedule.append((2022, mo))
    for mo in ALL_MONTHS:
        schedule.append((2023, mo))
    for mo in ALL_MONTHS:
        schedule.append((2024, mo))

    print(f"\n[3] Incremental schedule: {len(schedule)} months "
          f"(2022-07 → 2024-12)")

    # ── 5. Ablation study ─────────────────────────────────────────────
    print("\n[4] Running ablation study...")
    ablation_configs = {
        "Fine-tune":    {"use_ewc": False, "use_replay": False},
        "EWC only":     {"use_ewc": True,  "use_replay": False},
        "Replay only":  {"use_ewc": False, "use_replay": True},
        "EWC + Replay": {"use_ewc": True,  "use_replay": True},
    }

    ablation_results = {}
    ablation_forgetting = {}
    ablation_times = {}

    for method, cfg in ablation_configs.items():
        print(f"\n  --- {method} ---")
        result = run_incremental_variant(
            base_state_dict=base_state,
            n_features=n_features,
            ticker=ticker,
            scaler=scaler,
            incremental_schedule=schedule,
            jan_data=(X_jan, y_jan),
            X_train=X_train,
            y_train=y_train,
            method_name=method,
            **cfg,
        )
        ablation_results[method] = result["incremental_metrics"]
        ablation_forgetting[method] = result["forgetting_metrics"]
        ablation_times[method] = sum(result["times"])

    # Plot ablation
    print("\n[5] Plotting ablation results...")
    plot_ablation_comparison(ablation_results, ticker)
    plot_ablation_forgetting(ablation_forgetting, ticker)

    # ── 6. Forgetting heatmap (EWC+Replay, sample of months) ──────────
    print("\n[6] Computing forgetting heatmap...")
    # Evaluate on a subset of past months to keep it manageable
    eval_months = [
        (2022, "01"), (2022, "03"), (2022, "05"), (2022, "06"),
        (2022, "09"), (2022, "12"),
        (2023, "03"), (2023, "06"), (2023, "09"), (2023, "12"),
        (2024, "03"), (2024, "06"), (2024, "09"), (2024, "12"),
    ]
    # Use a shorter schedule for heatmap (every 3 months to keep readable)
    heatmap_schedule = [(y, m) for y, m in schedule if m in ("01", "04", "07", "10")]
    if not heatmap_schedule:
        heatmap_schedule = schedule[::3]  # every 3rd month

    heatmap_df = run_forgetting_heatmap(
        base_state, n_features, ticker, scaler,
        heatmap_schedule, eval_months,
        X_train, y_train,
    )
    plot_forgetting_heatmap(heatmap_df, ticker)

    # ── 7. Final predictions on Dec 2024 ──────────────────────────────
    print("\n[7] Final predictions on Dec 2024...")
    best_model = StockLSTM(n_features=n_features)
    # Use EWC+Replay variant for final prediction
    best_result = run_incremental_variant(
        base_state, n_features, ticker, scaler,
        schedule, (X_jan, y_jan),
        use_ewc=True, use_replay=True,
        X_train=X_train, y_train=y_train,
        method_name="final",
    )
    X_dec24, y_dec24, ref_dec24 = load_and_prepare(ticker, "12", 2024, scaler)
    y_dec24_pred = predict(best_result["model"], X_dec24)
    final_metrics = compute_metrics(y_dec24, y_dec24_pred)
    plot_predictions(y_dec24, y_dec24_pred, ticker, title_suffix="inc_dec_2024",
                     ref_close=ref_dec24)
    print(f"    Dec 2024: RMSE={final_metrics['RMSE']:.6f}, R²={final_metrics['R2']:.4f}")

    # Save model and feature scaler for real-time prediction
    torch.save(best_result["model"].state_dict(),
               os.path.join(OUTPUT_DIR, f"{ticker}_incremental_model.pt"))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, f"{ticker}_feature_scaler.pkl"))
    print(f"    Saved model + feature scaler to {OUTPUT_DIR}")

    # ── 8. Summary table ──────────────────────────────────────────────
    print(f"\n[8] Summary for {ticker}:")
    print(f"    {'Method':<16} {'Final RMSE':>12} {'Final R²':>10} "
          f"{'DirAcc':>8} {'Jan Forg. R²':>14} {'Total Time':>12}")
    print(f"    {'-'*78}")
    for method in ablation_configs:
        inc_m = ablation_results[method]
        forg_m = ablation_forgetting[method]
        print(f"    {method:<16} {inc_m[-1]['RMSE']:>12.6f} {inc_m[-1]['R2']:>10.4f} "
              f"{inc_m[-1]['DirAcc']:>7.1f}% {forg_m[-1]['R2']:>14.4f} "
              f"{ablation_times[method]:>10.1f}s")

    # ── 9. Save results ───────────────────────────────────────────────
    rows = []
    for method in ablation_configs:
        inc_m = ablation_results[method]
        forg_m = ablation_forgetting[method]
        rows.append({
            "ticker": ticker,
            "method": method,
            "final_RMSE": inc_m[-1]["RMSE"],
            "final_MAE": inc_m[-1]["MAE"],
            "final_R2": inc_m[-1]["R2"],
            "jan_forgetting_R2": forg_m[-1]["R2"],
            "total_inc_time": ablation_times[method],
        })

    return rows


def main():
    parser = argparse.ArgumentParser(description="Incremental Learning Study")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Run for a single ticker (default: all)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tickers = [args.ticker] if args.ticker else TICKERS

    all_rows = []
    for ticker in tickers:
        rows = run_ticker_study(ticker)
        all_rows.extend(rows)

    # Save summary
    summary_df = pd.DataFrame(all_rows)
    summary_path = os.path.join(OUTPUT_DIR, "incremental_study_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n\nStudy summary saved to {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
