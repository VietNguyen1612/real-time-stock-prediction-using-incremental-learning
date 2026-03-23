"""Real-time stock price prediction using a trained incremental LSTM model.

Modes:
  - Backtest (default): evaluate model on recent historical data
  - Live (--live): forward prediction every 15 min, verify previous predictions

Usage:
    # Backtest on recent data
    python predict_realtime.py --ticker AAPL --api_key YOUR_KEY

    # Live forward prediction for 3 hours
    python predict_realtime.py --ticker AAPL --api_key YOUR_KEY --live --duration 180
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import OUTPUT_DIR, LOOKBACK_WINDOW, FORECAST_HORIZON, DEVICE
from src.data.features import compute_features, create_sequences, scale_with_scaler
from src.data.fetch_alphavantage import fetch_intraday_extended
from src.models.lstm_model import StockLSTM


# ── Shared helpers ────────────────────────────────────────────────────────────

def fetch_recent_bars(ticker: str, api_key: str, months: int = 2) -> pd.DataFrame:
    """Fetch the last N months of 15-min bars from AlphaVantage."""
    now = datetime.now()
    frames = []

    for i in range(months - 1, -1, -1):
        month_num = now.month - i
        year = now.year
        while month_num <= 0:
            month_num += 12
            year -= 1

        print(f"  Fetching {ticker} {year}-{month_num:02d} from AlphaVantage...")
        df = fetch_intraday_extended(ticker, api_key, year=year, month_slot=month_num)
        if not df.empty:
            frames.append(df)

    if not frames:
        raise ValueError(f"No data returned for {ticker}. Check your API key and ticker symbol.")

    raw = pd.concat(frames).sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]
    print(f"  Total bars fetched: {len(raw)}  ({raw.index[0]} -> {raw.index[-1]})")
    return raw


def load_model_and_scalers(ticker: str):
    """Load trained model + feature scaler from outputs/."""
    model_path       = os.path.join(OUTPUT_DIR, f"{ticker}_incremental_model.pt")
    feat_scaler_path = os.path.join(OUTPUT_DIR, f"{ticker}_feature_scaler.pkl")
    y_scaler_path    = os.path.join(OUTPUT_DIR, f"{ticker}_y_scaler.pkl")

    for p in (model_path, feat_scaler_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing file: {p}\n"
                "Run 'python -m experiments.run_incremental_study --ticker <TICKER>' first."
            )

    feature_scaler = joblib.load(feat_scaler_path)
    y_scaler = joblib.load(y_scaler_path) if os.path.exists(y_scaler_path) else None

    n_features = feature_scaler.n_features_in_
    model = StockLSTM(n_features=n_features)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"  Loaded model ({n_features} features) from {model_path}")
    return model, feature_scaler, y_scaler


def predict_one(model, x_seq: np.ndarray, y_scaler) -> float:
    """Run inference on a single sequence (1, LOOKBACK, n_features)."""
    with torch.no_grad():
        X_t = torch.tensor(x_seq, device=DEVICE)
        pred = model(X_t).cpu().numpy().item()
    if y_scaler is not None:
        pred = y_scaler.inverse_transform([[pred]])[0, 0]
    return pred


# ── Backtest mode ─────────────────────────────────────────────────────────────

def run_backtest(ticker: str, api_key: str, months: int = 2):
    """Backtest: evaluate model on all recent historical sequences."""
    print(f"\n{'='*55}")
    print(f"  Backtest: {ticker}")
    print(f"{'='*55}\n")

    raw = fetch_recent_bars(ticker, api_key, months)

    min_bars = LOOKBACK_WINDOW + FORECAST_HORIZON + 30
    if len(raw) < min_bars:
        raise ValueError(f"Only {len(raw)} bars — need at least {min_bars}. Try --months.")

    feat = compute_features(raw)
    print(f"  After feature engineering: {len(feat)} bars, {feat.shape[1]} features")

    model, feature_scaler, y_scaler = load_model_and_scalers(ticker)

    scaled = scale_with_scaler(feature_scaler, feat)
    X, y_true_diff, ref_close = create_sequences(scaled, LOOKBACK_WINDOW, FORECAST_HORIZON)

    if len(X) == 0:
        raise ValueError("Not enough data to create sequences.")

    # Batch inference
    with torch.no_grad():
        X_t = torch.tensor(X, device=DEVICE)
        y_pred_diff = model(X_t).cpu().numpy().ravel()
    if y_scaler is not None:
        y_pred_diff = y_scaler.inverse_transform(y_pred_diff.reshape(-1, 1)).ravel()

    y_true_price = ref_close + y_true_diff
    y_pred_price = ref_close + y_pred_diff

    from sklearn.metrics import mean_squared_error, r2_score
    rmse = np.sqrt(mean_squared_error(y_true_diff, y_pred_diff))
    r2   = r2_score(y_true_diff, y_pred_diff)
    moved = y_true_diff != 0
    dir_acc = np.mean(np.sign(y_true_diff[moved]) == np.sign(y_pred_diff[moved])) * 100

    print(f"\n  Metrics on {len(X)} sequences:")
    print(f"    RMSE (scaled diff): {rmse:.6f}")
    print(f"    R2:                 {r2:.4f}")
    print(f"    Directional Acc:    {dir_acc:.1f}%")

    last_diff = y_pred_diff[-1]
    direction = "UP" if last_diff > 0 else "DOWN"
    print(f"\n  Latest signal ({feat.index[-1]}): {direction} (diff = {last_diff:+.6f})")

    # Plot
    timestamps = feat.index[LOOKBACK_WINDOW: LOOKBACK_WINDOW + len(X)]
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    ax = axes[0]
    ax.plot(timestamps, y_true_price, label="Actual", alpha=0.8, linewidth=1)
    ax.plot(timestamps, y_pred_price, label="Predicted", alpha=0.8, linewidth=1, linestyle="--")
    ax.set_title(f"{ticker} — Backtest: Predicted vs Actual Price (scaled)")
    ax.set_ylabel("Price (scaled)")
    ax.legend()
    ax.tick_params(axis="x", rotation=30)

    ax = axes[1]
    correct = (np.sign(y_true_diff) == np.sign(y_pred_diff)).astype(float)
    window = min(20, len(correct) // 4)
    if window > 0:
        rolling = pd.Series(correct).rolling(window, center=True).mean() * 100
        ax.plot(timestamps, rolling, color="teal", linewidth=1)
    ax.axhline(50, color="red", linewidth=1, linestyle="--", label="Random (50%)")
    ax.set_title(f"Rolling Directional Accuracy (window={window})")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.tick_params(axis="x", rotation=30)

    plt.suptitle(f"{ticker} — Backtest (AlphaVantage)", fontsize=14)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, f"{ticker}_realtime_prediction.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved to {out_path}")


# ── Live forward prediction mode ─────────────────────────────────────────────

def forward_predict(model, feature_scaler, y_scaler, feat_df: pd.DataFrame):
    """Use the last LOOKBACK_WINDOW bars to predict the NEXT price change.

    Returns (pred_diff, ref_close_scaled) in scaled space.
    """
    scaled = scale_with_scaler(feature_scaler, feat_df)
    data = scaled.values
    close_idx = list(scaled.columns).index("Close")

    x = data[-LOOKBACK_WINDOW:]
    ref_close = x[-1, close_idx]
    X = np.expand_dims(x, axis=0).astype(np.float32)

    pred_diff = predict_one(model, X, y_scaler)
    return pred_diff, ref_close


def verify_previous(prev, raw):
    """Check the previous prediction against what actually happened."""
    future_bars = raw[raw.index > prev["timestamp"]]
    if len(future_bars) < FORECAST_HORIZON:
        return False  # not enough new bars yet

    actual_close = future_bars["Close"].iloc[FORECAST_HORIZON - 1]
    actual_diff = actual_close - prev["raw_close"]
    prev["actual_close"] = actual_close
    prev["actual_diff"] = actual_diff
    prev["correct"] = (
        (prev["pred_diff"] > 0 and actual_diff > 0) or
        (prev["pred_diff"] <= 0 and actual_diff <= 0)
    )
    return True


def save_log(entry, ticker):
    """Append one prediction to CSV log."""
    csv_path = os.path.join(OUTPUT_DIR, f"{ticker}_realtime_log.csv")
    fields = ["timestamp", "raw_close", "direction", "pred_diff",
              "actual_close", "actual_diff", "correct"]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow({k: entry.get(k, "") for k in fields})
    return csv_path


def plot_live(history, ticker):
    """Plot accumulated live predictions."""
    verified = [h for h in history if h.get("actual_diff") is not None]
    if len(verified) < 2:
        return

    times   = [h["timestamp"] for h in verified]
    closes  = [h["raw_close"] for h in verified]
    actuals = [h["actual_close"] for h in verified]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    # Price chart with direction markers
    ax = axes[0]
    ax.plot(times, closes, "o-", label="Close at prediction", markersize=5, alpha=0.7)
    ax.plot(times, actuals, "s-", label=f"Actual {FORECAST_HORIZON} bars later", markersize=5, alpha=0.7)
    for i, h in enumerate(verified):
        color = "green" if h["correct"] else "red"
        marker = "^" if h["direction"] == "UP" else "v"
        ax.plot(times[i], closes[i], marker=marker, color=color, markersize=12, zorder=5)
    ax.set_title(f"{ticker} — Live Forward Predictions")
    ax.set_ylabel("Close Price ($)")
    ax.legend()
    ax.tick_params(axis="x", rotation=30)

    # Cumulative accuracy
    ax = axes[1]
    correct_list = [h["correct"] for h in verified]
    cum_acc = [sum(correct_list[:i+1]) / (i+1) * 100 for i in range(len(correct_list))]
    ax.plot(times, cum_acc, "o-", color="teal", markersize=5)
    ax.axhline(50, color="red", linestyle="--", label="Random (50%)")
    ax.set_title("Cumulative Directional Accuracy")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"{ticker}_live_prediction.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to {out_path}")


def run_live(ticker, api_key, months, interval, duration):
    """Live forward prediction loop."""
    print(f"\n{'='*60}")
    print(f"  Loading model for {ticker}...")
    print(f"{'='*60}")
    model, feature_scaler, y_scaler = load_model_and_scalers(ticker)

    dur_msg = f" for {duration} min" if duration > 0 else ""
    print(f"\n  Live mode ON{dur_msg} — predicting every {interval} min")
    print(f"  Forecast: {FORECAST_HORIZON} bars ahead ({FORECAST_HORIZON * 15} min)")
    print(f"  Press Ctrl+C to stop.\n")

    history = []
    start_time = time.time()

    while True:
        try:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n{'='*60}")
            print(f"  Cycle {len(history) + 1} -- {now_str}")
            print(f"{'='*60}")

            # Fetch latest data
            raw = fetch_recent_bars(ticker, api_key, months)
            latest_bar = raw.index[-1]
            latest_close = raw["Close"].iloc[-1]
            print(f"  Latest bar: {latest_bar}  Close: ${latest_close:.2f}")

            # Verify previous prediction
            if history and history[-1].get("actual_diff") is None:
                prev = history[-1]
                if verify_previous(prev, raw):
                    tag = "CORRECT" if prev["correct"] else "WRONG"
                    symbol = "O" if prev["correct"] else "X"
                    print(f"\n  Previous prediction ({prev['timestamp']}):")
                    print(f"    Predicted: {prev['direction']}")
                    print(f"    Actual close: ${prev['actual_close']:.2f} "
                          f"(diff: {prev['actual_diff']:+.2f})")
                    print(f"    Result:    [{symbol}] {tag}")
                    save_log(prev, ticker)
                else:
                    print(f"\n  Previous prediction not yet verifiable "
                          f"(need {FORECAST_HORIZON} new bars)")

            # Make new forward prediction
            feat = compute_features(raw)
            pred_diff, ref_close = forward_predict(model, feature_scaler, y_scaler, feat)
            direction = "UP" if pred_diff > 0 else "DOWN"

            entry = {
                "timestamp": latest_bar,
                "raw_close": latest_close,
                "pred_diff": pred_diff,
                "direction": direction,
                "actual_close": None,
                "actual_diff": None,
                "correct": None,
            }
            history.append(entry)

            arrow = "^" if direction == "UP" else "v"
            print(f"\n  >> PREDICTION: {direction} {arrow} "
                  f"(next {FORECAST_HORIZON * 15} min)")
            print(f"     Current close: ${latest_close:.2f}")
            print(f"     Predicted diff: {pred_diff:+.6f} (scaled)")

            # Running accuracy
            verified = [h for h in history if h.get("correct") is not None]
            if verified:
                n_correct = sum(1 for h in verified if h["correct"])
                acc = n_correct / len(verified) * 100
                print(f"\n  Running accuracy: {acc:.0f}% "
                      f"({n_correct}/{len(verified)} correct)")

            plot_live(history, ticker)

        except Exception as e:
            print(f"  [ERROR] {e}")

        # Check duration limit
        if duration > 0:
            elapsed = (time.time() - start_time) / 60
            if elapsed >= duration:
                print(f"\n  Duration limit ({duration} min) reached. Stopping.")
                break
            print(f"  Elapsed: {elapsed:.1f} / {duration} min")

        print(f"\n  Waiting {interval} min for next cycle...")
        time.sleep(interval * 60)

    # Final summary
    verified = [h for h in history if h.get("correct") is not None]
    if verified:
        n_correct = sum(1 for h in verified if h["correct"])
        acc = n_correct / len(verified) * 100
        print(f"\n{'='*60}")
        print(f"  FINAL SUMMARY -- {ticker}")
        print(f"{'='*60}")
        print(f"  Total predictions: {len(history)}")
        print(f"  Verified:          {len(verified)}")
        print(f"  Correct:           {n_correct}")
        print(f"  Accuracy:          {acc:.1f}%")
        csv_path = os.path.join(OUTPUT_DIR, f"{ticker}_realtime_log.csv")
        print(f"  Log:               {csv_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Real-time stock prediction via AlphaVantage")
    parser.add_argument("--ticker",   type=str, required=True, help="e.g. AAPL")
    parser.add_argument("--api_key",  type=str, required=True, help="AlphaVantage API key")
    parser.add_argument("--months",   type=int, default=2,
                        help="Months of recent data to fetch (default: 2)")
    parser.add_argument("--live",     action="store_true",
                        help="Live forward prediction mode")
    parser.add_argument("--interval", type=int, default=15,
                        help="Minutes between predictions in live mode (default: 15)")
    parser.add_argument("--duration", type=int, default=0,
                        help="Total minutes for live mode (default: 0 = unlimited)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.live:
        run_live(args.ticker, args.api_key, args.months, args.interval, args.duration)
    else:
        run_backtest(args.ticker, args.api_key, args.months)


if __name__ == "__main__":
    main()
