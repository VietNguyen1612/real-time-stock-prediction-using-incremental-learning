"""Classic-ML baselines (scikit-learn) for stock price-change prediction.

Trains LinearRegression and RandomForestRegressor on the same temporal split
and target variable as the LSTM pipeline, giving a fair "classic ML vs deep
learning" comparison without any Big Data tooling.

For each ticker:
  Train: INITIAL_TRAIN_MONTHS of 2022 (Jan-May)
  Test:  TEST_MONTH of TEST_YEAR (2022 Dec — last Phase 2 month)
  Target: price change = Close[t + FORECAST_HORIZON] - Close[t]

Usage:
    python -m src.models.sklearn_baseline                 # all tickers
    python -m src.models.sklearn_baseline --ticker AAPL   # single ticker
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.config import (
    TICKERS, OUTPUT_DIR, INITIAL_TRAIN_YEAR, INITIAL_TRAIN_MONTHS,
    TEST_YEAR, TEST_MONTH, FORECAST_HORIZON,
)
from src.data.loader import load_monthly_featured, load_months_featured
from src.evaluation.metrics import compute_metrics


FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_20", "EMA_12", "MACD", "MACD_signal", "RSI_14",
    "BB_upper", "BB_lower", "ATR_14", "Volume_pct", "Returns",
]


def build_xy(feat: pd.DataFrame, horizon: int = FORECAST_HORIZON):
    """Build (X, y) where y = Close[t+horizon] - Close[t] (raw price change)."""
    target = feat["Close"].shift(-horizon) - feat["Close"]
    valid = target.notna()
    X = feat.loc[valid, FEATURE_COLS].values.astype(np.float64)
    y = target[valid].values.astype(np.float64)
    return X, y


def run_ticker_baseline(ticker: str) -> list[dict]:
    """Train LR + RF for one ticker and return a metrics row per model."""
    train_feat = load_months_featured(ticker, INITIAL_TRAIN_MONTHS, year=INITIAL_TRAIN_YEAR)
    test_feat = load_monthly_featured(ticker, TEST_MONTH, year=TEST_YEAR)

    X_train, y_train = build_xy(train_feat)
    X_test, y_test = build_xy(test_feat)

    # Standardize features (fit on train) so LinearRegression is well-conditioned
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
    }

    rows = []
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train_s, y_train)
        train_time = time.time() - t0
        metrics = compute_metrics(y_test, model.predict(X_test_s))
        metrics["train_time"] = train_time
        rows.append({"ticker": ticker, "model": name, **metrics})
        print(f"  {ticker:<6} {name:<18} RMSE={metrics['RMSE']:.6f} "
              f"R²={metrics['R2']:.4f} DirAcc={metrics['DirAcc']:.1f}% "
              f"({train_time:.2f}s)")
    return rows


def run_baselines(tickers: list[str] | None = None) -> pd.DataFrame:
    """Run sklearn baselines for the given tickers (default: all)."""
    if tickers is None:
        tickers = TICKERS

    print(f"[baseline] Training LinearRegression + RandomForest for "
          f"{len(tickers)} ticker(s)...")
    all_rows = []
    for ticker in tickers:
        all_rows.extend(run_ticker_baseline(ticker))

    summary = pd.DataFrame(all_rows)

    # Mean per model across tickers
    print("\nMean across tickers:")
    print(summary.groupby("model")[["RMSE", "MAE", "R2", "DirAcc", "train_time"]]
          .mean().to_string())

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_path = os.path.join(OUTPUT_DIR, "baseline_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved to {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Classic-ML (sklearn) baselines")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Run for a single ticker (default: all)")
    args = parser.parse_args()
    tickers = [args.ticker] if args.ticker else None
    run_baselines(tickers)


if __name__ == "__main__":
    main()
