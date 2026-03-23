"""Feature engineering: technical indicators, sliding windows, normalization."""

import numpy as np
import pandas as pd
import ta
from sklearn.preprocessing import MinMaxScaler


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators from OHLCV data.

    Returns a DataFrame with original OHLCV + derived features, NaN rows dropped.
    """
    df = df.copy()

    # Trend indicators
    df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)
    df["EMA_12"] = ta.trend.ema_indicator(df["Close"], window=12)

    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()

    # Momentum
    df["RSI_14"] = ta.momentum.rsi(df["Close"], window=14)

    # Volatility
    bb = ta.volatility.BollingerBands(df["Close"], window=20)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()
    df["ATR_14"] = ta.volatility.average_true_range(
        df["High"], df["Low"], df["Close"], window=14
    )

    # Volume & price changes
    df["Volume_pct"] = df["Volume"].pct_change()
    df["Returns"] = df["Close"].pct_change()

    df.dropna(inplace=True)
    return df


def create_sequences(
    df: pd.DataFrame, lookback: int, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create sliding-window sequences for LSTM.

    Args:
        df: feature DataFrame (all columns used as input features)
        lookback: number of past time steps
        horizon: number of steps ahead to predict

    Returns:
        X: shape (N, lookback, n_features)
        y: shape (N,) -- price change (future Close - last input Close) on scaled data
        ref_close: shape (N,) -- last input Close (for converting back to price levels)
    """
    data = df.values
    close_idx = list(df.columns).index("Close")

    X, y, ref = [], [], []
    for i in range(lookback, len(data) - horizon + 1):
        X.append(data[i - lookback : i])
        last_close = data[i - 1, close_idx]
        future_close = data[i + horizon - 1, close_idx]
        y.append(future_close - last_close)
        ref.append(last_close)

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
        np.array(ref, dtype=np.float32),
    )


def returns_to_prices(y_diff: np.ndarray, ref_close: np.ndarray) -> np.ndarray:
    """Convert price-change predictions back to (scaled) price levels."""
    return ref_close + y_diff


def normalize_data(
    train_df: pd.DataFrame, *other_dfs: pd.DataFrame
) -> tuple[pd.DataFrame, ...]:
    """Fit MinMaxScaler on train_df, transform all DataFrames.

    Returns:
        (scaled_train, *scaled_others, scaler)
        The last element is always the fitted scaler (for reuse in incremental months).
    """
    scaler = MinMaxScaler()
    cols = train_df.columns

    scaled_train = pd.DataFrame(
        scaler.fit_transform(train_df), columns=cols, index=train_df.index
    )

    results = [scaled_train]
    for odf in other_dfs:
        scaled = pd.DataFrame(
            scaler.transform(odf), columns=cols, index=odf.index
        )
        results.append(scaled)

    results.append(scaler)
    return tuple(results)


def scale_with_scaler(
    scaler: MinMaxScaler, df: pd.DataFrame
) -> pd.DataFrame:
    """Transform a DataFrame using a pre-fitted scaler."""
    return pd.DataFrame(
        scaler.transform(df), columns=df.columns, index=df.index
    )
