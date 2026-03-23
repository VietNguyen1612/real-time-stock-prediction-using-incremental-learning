"""Fetch 2023 intraday 15-min OHLCV data from AlphaVantage.

Usage:
    python -m src.data.fetch_alphavantage --api_key YOUR_KEY
    python -m src.data.fetch_alphavantage --api_key YOUR_KEY --tickers AAPL MSFT
    python -m src.data.fetch_alphavantage --api_key YOUR_KEY --year 2024
"""

import os
import time
import argparse
import requests
import pandas as pd

TICKERS = ["AAPL", "AMZN", "BRK-B", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]
BASE_URL = "https://www.alphavantage.co/query"


def fetch_intraday_extended(ticker: str, api_key: str, year: int, month_slot: int) -> pd.DataFrame:
    """Fetch one month-slot of 15-min intraday data.

    AlphaVantage TIME_SERIES_INTRADAY uses 'month' param as YYYY-MM.
    """
    month_str = f"{year}-{month_slot:02d}"
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": ticker,
        "interval": "15min",
        "month": month_str,
        "outputsize": "full",
        "apikey": api_key,
        "datatype": "csv",
    }

    resp = requests.get(BASE_URL, params=params)
    resp.raise_for_status()

    # Check if response is an error message (JSON) instead of CSV
    if resp.text.startswith("{"):
        print(f"  API error for {ticker} {month_str}: {resp.text.strip()}")
        return pd.DataFrame()

    from io import StringIO
    df = pd.read_csv(StringIO(resp.text))

    if df.empty:
        print(f"  No data for {ticker} {month_str}")
        return df

    # Rename columns to match existing format
    df = df.rename(columns={
        "timestamp": "Datetime",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })

    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.set_index("Datetime").sort_index()
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    return df


def fetch_ticker_year(ticker: str, api_key: str, year: int, output_dir: str):
    """Fetch all 12 months for a ticker and save as monthly CSVs."""
    ticker_dir = os.path.join(output_dir, ticker)
    os.makedirs(ticker_dir, exist_ok=True)

    all_months = []

    for month in range(1, 13):
        out_path = os.path.join(ticker_dir, f"{ticker}_{year}-{month:02d}_15min.csv")

        # Skip if already downloaded
        if os.path.exists(out_path):
            print(f"  {ticker} {year}-{month:02d} already exists, skipping")
            df = pd.read_csv(out_path, index_col=0, parse_dates=True)
            all_months.append(df)
            continue

        print(f"  Fetching {ticker} {year}-{month:02d}...")
        df = fetch_intraday_extended(ticker, api_key, year, month)

        if df.empty:
            continue

        df.to_csv(out_path)
        all_months.append(df)
        print(f"  Saved {len(df)} rows to {out_path}")

        # Premium subscription: 75 req/min — no sleep needed
        pass

    # Save full-year combined file
    if all_months:
        full = pd.concat(all_months).sort_index()
        full_path = os.path.join(ticker_dir, f"{ticker}_{year}_full_15min.csv")
        full.to_csv(full_path)
        print(f"  Saved full year: {len(full)} rows to {full_path}")


def main():
    parser = argparse.ArgumentParser(description="Fetch AlphaVantage intraday data")
    parser.add_argument("--api_key", required=True, help="AlphaVantage API key")
    parser.add_argument("--tickers", nargs="+", default=TICKERS, help="Tickers to fetch")
    parser.add_argument("--year", type=int, default=2023, help="Year to fetch (default: 2023)")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    args = parser.parse_args()

    output_dir = args.output_dir or f"NASDAQ_{args.year}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Fetching {args.year} data for {len(args.tickers)} tickers into {output_dir}/")
    print(f"Note: Free API key = 25 requests/day. 8 tickers × 12 months = 96 requests.\n")

    for ticker in args.tickers:
        print(f"\n=== {ticker} ===")
        fetch_ticker_year(ticker, args.api_key, args.year, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
