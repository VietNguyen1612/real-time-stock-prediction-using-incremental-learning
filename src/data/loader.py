"""Data loading utilities for NASDAQ 15-min OHLCV data.

Supports two backends:
  - Legacy: read raw CSVs + compute features via ta library
  - Spark:  read pre-computed features from Parquet (USE_SPARK_FEATURES=True)
"""

import os
import pandas as pd
from src.config import DATA_DIR, PROJECT_ROOT, PARQUET_DIR, USE_SPARK_FEATURES


def load_monthly_csv(ticker: str, month: str, year: int = 2022) -> pd.DataFrame:
    """Load a single monthly CSV for a ticker.

    Args:
        ticker: e.g. "AAPL"
        month: two-digit month string, e.g. "01"
        year: data year (default 2022)
    """
    data_dir = os.path.join(PROJECT_ROOT, f"NASDAQ_{year}")
    path = os.path.join(data_dir, ticker, f"{ticker}_{year}-{month}_15min.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def load_months(ticker: str, months: list[str], year: int = 2022) -> pd.DataFrame:
    """Load and concatenate multiple monthly CSVs for a ticker."""
    dfs = [load_monthly_csv(ticker, m, year=year) for m in months]
    return pd.concat(dfs).sort_index()


def load_full_year(ticker: str, year: int = 2022) -> pd.DataFrame:
    """Load the full-year 15-min CSV for a ticker."""
    data_dir = os.path.join(PROJECT_ROOT, f"NASDAQ_{year}")
    path = os.path.join(data_dir, ticker, f"{ticker}_{year}_full_15min.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


# ── Parquet loaders (Spark-produced features) ────────────────────────────────

KEEP_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_20", "EMA_12", "MACD", "MACD_signal", "RSI_14",
    "BB_upper", "BB_lower", "ATR_14", "Volume_pct", "Returns",
]


def load_monthly_parquet(ticker: str, month: str, year: int = 2022) -> pd.DataFrame:
    """Load pre-computed features from Spark-produced Parquet for one month."""
    path = os.path.join(
        PARQUET_DIR, f"ticker={ticker}", f"year={year}", f"month={int(month)}"
    )
    df = pd.read_parquet(path)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.set_index("Datetime").sort_index()
    return df[KEEP_COLS]


def load_months_parquet(
    ticker: str, months: list[str], year: int = 2022
) -> pd.DataFrame:
    """Load and concatenate multiple months from Spark Parquet."""
    dfs = [load_monthly_parquet(ticker, m, year=year) for m in months]
    return pd.concat(dfs).sort_index()


# ── Facade functions (auto-select backend) ───────────────────────────────────

def load_monthly_featured(ticker: str, month: str, year: int = 2022) -> pd.DataFrame:
    """Load one month with all features already computed.

    Uses Spark Parquet when USE_SPARK_FEATURES=True, otherwise falls back
    to CSV + ta library.
    """
    if USE_SPARK_FEATURES:
        return load_monthly_parquet(ticker, month, year)
    from src.data.features import compute_features
    raw = load_monthly_csv(ticker, month, year)
    return compute_features(raw)


def load_months_featured(
    ticker: str, months: list[str], year: int = 2022
) -> pd.DataFrame:
    """Load multiple months with all features already computed."""
    if USE_SPARK_FEATURES:
        return load_months_parquet(ticker, months, year)
    from src.data.features import compute_features
    raw = load_months(ticker, months, year)
    return compute_features(raw)
