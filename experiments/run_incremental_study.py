"""Incremental Learning Study under tick-based retraining (Phase 2).

This script focuses on the incremental learning component, now driven by the
two-phase pipeline:
  Phase 1: offline base training on a static 6-month block of 2022 (Jan-Jun).
  Phase 2: tick-based retraining over the subsequent 6 months (Jul-Dec 2022),
           where 1 tick == 15 min == 1 new bar.

It compares four variants under that streaming regime:
  1. Ablation: Fine-tune vs EWC only vs Replay only vs EWC+Replay
  2. Forgetting heatmap across past months after each Phase 2 month
  3. Final-month (Dec 2022) evaluation

Prequential tick errors are aggregated per Phase 2 month so the ablation /
forgetting plots remain readable.

Usage:
    python -m experiments.run_incremental_study                # all tickers
    python -m experiments.run_incremental_study --ticker AAPL  # single ticker
    python -m experiments.run_incremental_study --max-ticks 100  # quick run
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    TICKERS, INITIAL_TRAIN_MONTHS, VALIDATION_MONTH,
    PHASE2_YEAR, PHASE2_MONTHS, PHASE2_SCHEDULE, TICK_INTERVAL_MINUTES,
    LOOKBACK_WINDOW, FORECAST_HORIZON, BATCH_SIZE, DEVICE, OUTPUT_DIR,
    EWC_LAMBDA, REPLAY_ALPHA,
)
from src.data.loader import load_monthly_featured, load_months_featured
from src.data.features import create_sequences, normalize_data, scale_with_scaler
from src.models.lstm_model import StockLSTM
from src.models.trainer import BatchTrainer, IncrementalTrainer
from src.incremental.incremental_learner import EWC, ReplayBuffer
from src.incremental.tick_pipeline import run_tick_retraining
from src.evaluation.metrics import compute_metrics
from src.utils.plotting import (
    plot_predictions, plot_loss_curves,
    plot_ablation_comparison, plot_ablation_forgetting, plot_forgetting_heatmap,
)


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
    """Train the base LSTM model offline. Returns model and history."""
    model = StockLSTM(n_features=n_features)
    trainer = BatchTrainer(model)
    history = trainer.fit(X_train, y_train, X_val, y_val)
    return model, history


def run_incremental_variant(
    base_state_dict, n_features, ticker, scaler,
    schedule, jan_data,
    use_ewc, use_replay,
    X_train, y_train,
    method_name, max_ticks=None,
):
    """Run one tick-based incremental variant (for ablation).

    Streams the Phase 2 months tick by tick (1 tick = 15 min). Prequential
    errors are aggregated per Phase 2 month, and a forgetting probe (Jan 2022)
    is taken after each month.

    Returns:
        dict with incremental_metrics, forgetting_metrics, times.
    """
    # Fresh model from the offline base checkpoint
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
        ewc.compute_fisher(model, DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False))

    # Setup replay buffer
    replay_buffer = ReplayBuffer(max_size=1)  # dummy
    if use_replay:
        replay_buffer = ReplayBuffer.from_initial_data(X_train, y_train)

    ewc_lambda = EWC_LAMBDA if use_ewc else 0.0
    replay_alpha = REPLAY_ALPHA if use_replay else 0.0

    inc_trainer = IncrementalTrainer(model, ewc, replay_buffer,
                                     ewc_lambda=ewc_lambda,
                                     replay_alpha=replay_alpha)

    X_jan, y_jan = jan_data
    baseline_jan = compute_metrics(y_jan, predict(model, X_jan))

    incremental_metrics = []
    forgetting_metrics = [{"month": "baseline", **baseline_jan}]
    times = []

    for year, month in schedule:
        label = f"{year}-{month}"
        X_m, y_m, _ = load_and_prepare(ticker, month, year, scaler)

        # Stream this month's ticks (continuous: same inc_trainer/model)
        res = run_tick_retraining(
            inc_trainer, X_m, y_m,
            y_scaler=None, max_ticks=max_ticks,
            label=f"{method_name} {label}", log_every=0,
        )
        times.append(res["total_time"])

        # Prequential metrics for this month
        month_metrics = compute_metrics(res["y_true"], res["y_pred"])
        month_metrics["month"] = label
        incremental_metrics.append(month_metrics)

        # Forgetting probe on Jan 2022
        jan_metrics = compute_metrics(y_jan, predict(model, X_jan))
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
                           schedule, eval_months,
                           X_train, y_train, max_ticks=None):
    """Run EWC+Replay tick stream, evaluating on past months after each month.

    Returns:
        pd.DataFrame heatmap (rows=eval_month, cols=after_phase2_month)
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

    update_labels = [f"{yr}-{mo}" for yr, mo in schedule]
    heatmap = pd.DataFrame(index=list(eval_data.keys()), columns=update_labels, dtype=float)

    for year, month in schedule:
        update_label = f"{year}-{month}"
        X_m, y_m, _ = load_and_prepare(ticker, month, year, scaler)
        run_tick_retraining(inc_trainer, X_m, y_m, y_scaler=None,
                            max_ticks=max_ticks, label=update_label, log_every=0)

        for eval_label, (X_e, y_e) in eval_data.items():
            metrics = compute_metrics(y_e, predict(model, X_e))
            heatmap.loc[eval_label, update_label] = metrics["R2"]

    return heatmap


def run_ticker_study(ticker: str, max_ticks=None) -> list:
    """Run the full tick-based incremental learning study for one ticker."""
    print(f"\n{'='*70}")
    print(f"  INCREMENTAL LEARNING STUDY (tick-based): {ticker}")
    print(f"{'='*70}")

    # ── 1. Phase 1: load & prepare the offline base block (Jan-May + Jun) ──
    print("\n[1] Loading base block (2022 Jan-May + Jun)...")
    train_feat = load_months_featured(ticker, INITIAL_TRAIN_MONTHS, year=2022)
    val_feat = load_monthly_featured(ticker, VALIDATION_MONTH, year=2022)
    n_features = train_feat.shape[1]

    train_scaled, val_scaled, scaler = normalize_data(train_feat, val_feat)
    X_train, y_train, _ = create_sequences(train_scaled, LOOKBACK_WINDOW, FORECAST_HORIZON)
    X_val, y_val, _ = create_sequences(val_scaled, LOOKBACK_WINDOW, FORECAST_HORIZON)
    print(f"    Features: {n_features}, Train: {X_train.shape}, Val: {X_val.shape}")

    # ── 2. Phase 1: offline batch training ────────────────────────────
    print("\n[2] Offline base training (2022 Jan-May)...")
    model, batch_history = train_batch_model(ticker, n_features, X_train, y_train, X_val, y_val)
    base_state = copy.deepcopy(model.state_dict())

    batch_metrics = compute_metrics(y_val, predict(model, X_val))
    print(f"    Base val: RMSE={batch_metrics['RMSE']:.6f}, R²={batch_metrics['R2']:.4f}")
    plot_loss_curves(batch_history, ticker, title_suffix="batch_inc_study")

    # ── 3. Jan 2022 forgetting probe data ─────────────────────────────
    X_jan, y_jan, _ = load_and_prepare(ticker, "01", 2022, scaler)

    # ── 4. Phase 2 schedule (tick-based, Jul-Dec 2022) ────────────────
    schedule = PHASE2_SCHEDULE
    print(f"\n[3] Phase 2 tick schedule: {len(schedule)} months "
          f"({schedule[0][0]}-{schedule[0][1]} → {schedule[-1][0]}-{schedule[-1][1]}), "
          f"1 tick = {TICK_INTERVAL_MINUTES} min")

    # ── 5. Ablation study (tick-based) ────────────────────────────────
    print("\n[4] Running ablation study under tick-based retraining...")
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
            schedule=schedule,
            jan_data=(X_jan, y_jan),
            X_train=X_train,
            y_train=y_train,
            method_name=method,
            max_ticks=max_ticks,
            **cfg,
        )
        ablation_results[method] = result["incremental_metrics"]
        ablation_forgetting[method] = result["forgetting_metrics"]
        ablation_times[method] = sum(result["times"])

    # Plot ablation
    print("\n[5] Plotting ablation results...")
    plot_ablation_comparison(ablation_results, ticker)
    plot_ablation_forgetting(ablation_forgetting, ticker)

    # ── 6. Forgetting heatmap (EWC+Replay) ────────────────────────────
    print("\n[6] Computing forgetting heatmap...")
    eval_months = [
        (2022, "01"), (2022, "02"), (2022, "03"), (2022, "04"),
        (2022, "05"), (2022, "06"), (2022, "07"), (2022, "08"),
        (2022, "09"), (2022, "10"), (2022, "11"), (2022, "12"),
    ]
    heatmap_df = run_forgetting_heatmap(
        base_state, n_features, ticker, scaler,
        schedule, eval_months,
        X_train, y_train, max_ticks=max_ticks,
    )
    plot_forgetting_heatmap(heatmap_df, ticker)

    # ── 7. Final predictions on Dec 2022 (EWC+Replay) ─────────────────
    print("\n[7] Final predictions on Dec 2022...")
    best_result = run_incremental_variant(
        base_state, n_features, ticker, scaler,
        schedule, (X_jan, y_jan),
        use_ewc=True, use_replay=True,
        X_train=X_train, y_train=y_train,
        method_name="final", max_ticks=max_ticks,
    )
    X_dec, y_dec, ref_dec = load_and_prepare(ticker, "12", PHASE2_YEAR, scaler)
    y_dec_pred = predict(best_result["model"], X_dec).ravel()
    final_metrics = compute_metrics(y_dec, y_dec_pred)
    plot_predictions(y_dec, y_dec_pred, ticker, title_suffix="inc_dec_2022",
                     ref_close=ref_dec)
    print(f"    Dec 2022: RMSE={final_metrics['RMSE']:.6f}, R²={final_metrics['R2']:.4f}")

    # Save model and feature scaler
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

    # ── 9. Build result rows ──────────────────────────────────────────
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
    parser = argparse.ArgumentParser(description="Tick-based Incremental Learning Study")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Run for a single ticker (default: all)")
    parser.add_argument("--max-ticks", type=int, default=None,
                        help="Cap ticks per Phase 2 month (quick runs; default: all)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tickers = [args.ticker] if args.ticker else TICKERS

    all_rows = []
    for ticker in tickers:
        all_rows.extend(run_ticker_study(ticker, max_ticks=args.max_ticks))

    summary_df = pd.DataFrame(all_rows)
    summary_path = os.path.join(OUTPUT_DIR, "incremental_study_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n\nStudy summary saved to {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
