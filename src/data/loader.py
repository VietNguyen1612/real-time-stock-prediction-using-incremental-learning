"""Data loading utilities for NASDAQ 15-min OHLCV data.

Reads raw monthly CSVs and computes technical-indicator features on the fly
via the ``ta`` library (see ``src/data/features.py``).
"""

import os
import pandas as pd

from src.config import PROJECT_ROOT
from src.data.features import compute_features


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


# ── Feature-enriched loaders ─────────────────────────────────────────────────

def load_monthly_featured(ticker: str, month: str, year: int = 2022) -> pd.DataFrame:
    """Load one month with all technical-indicator features computed."""
    return compute_features(load_monthly_csv(ticker, month, year))


def load_months_featured(
    ticker: str, months: list[str], year: int = 2022
) -> pd.DataFrame:
    """Load multiple months (concatenated) with all features computed.

    Indicators are computed over the concatenated series so values are
    continuous across month boundaries.
    """
    return compute_features(load_months(ticker, months, year))
