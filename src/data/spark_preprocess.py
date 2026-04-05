"""PySpark preprocessing pipeline for NASDAQ stock data (2022-2025).

Computes all technical indicators used by the LSTM pipeline:
  - Window functions: SMA_20, BB_upper, BB_lower, Volume_pct, Returns
  - Grouped-map UDF (applyInPandas): EMA_12, MACD, MACD_signal, RSI_14, ATR_14

Outputs feature-enriched Parquet partitioned by (ticker, year, month).
"""

import os
import re
import time
import glob as globmod
from functools import reduce

import pandas as pd
import ta
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType, TimestampType,
)
from pyspark.sql.window import Window

from src.config import PROJECT_ROOT, OUTPUT_DIR, PARQUET_DIR, DATA_YEARS, TICKERS


# ── Column order expected by the training pipeline ──────────────────────────
FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_20", "EMA_12", "MACD", "MACD_signal", "RSI_14",
    "BB_upper", "BB_lower", "ATR_14", "Volume_pct", "Returns",
]


def create_spark_session() -> SparkSession:
    """Create a local Spark session optimised for this workload."""
    return (
        SparkSession.builder
        .appName("NASDAQ_FeatureEngineering")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "32")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )


# ── 1. Data Loading ─────────────────────────────────────────────────────────

def load_all_csvs(spark: SparkSession) -> DataFrame:
    """Load all monthly 15-min CSVs across all years into one DataFrame.

    Adds columns: ticker (str), year (int), month (int).
    """
    schema = StructType([
        StructField("Datetime", StringType(), True),
        StructField("Open", DoubleType(), True),
        StructField("High", DoubleType(), True),
        StructField("Low", DoubleType(), True),
        StructField("Close", DoubleType(), True),
        StructField("Volume", DoubleType(), True),
    ])

    dfs = []
    for year in DATA_YEARS:
        data_dir = os.path.join(PROJECT_ROOT, f"NASDAQ_{year}")
        if not os.path.isdir(data_dir):
            print(f"  [skip] {data_dir} not found")
            continue

        all_files = globmod.glob(os.path.join(data_dir, "*", "*_15min.csv"))
        monthly_files = [f for f in all_files if "full" not in f]

        for fpath in monthly_files:
            ticker = os.path.basename(os.path.dirname(fpath))
            # Extract month from filename: {TICKER}_{YEAR}-{MM}_15min.csv
            # Use regex to handle tickers with dashes (e.g. BRK-B)
            fname = os.path.basename(fpath)
            m = re.search(r'(\d{4})-(\d{2})_15min', fname)
            month_str = m.group(2)

            df = (
                spark.read.csv(fpath, header=True, schema=schema)
                .withColumn("ticker", F.lit(ticker))
                .withColumn("year", F.lit(year).cast(IntegerType()))
                .withColumn("month", F.lit(int(month_str)).cast(IntegerType()))
                .withColumn("Datetime", F.to_timestamp("Datetime"))
            )
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No CSV files found in any NASDAQ data directory")

    combined = reduce(DataFrame.union, dfs)
    return combined


# ── 2. Window-based features ────────────────────────────────────────────────

def compute_window_features(df: DataFrame) -> DataFrame:
    """Compute features expressible as pure Spark Window functions.

    Partitioned by ticker (continuous across months) and ordered by Datetime.
    """
    w = Window.partitionBy("ticker").orderBy("Datetime")
    w20 = w.rowsBetween(-19, 0)   # 20-period window

    prev_close = F.lag("Close", 1).over(w)
    prev_vol = F.lag("Volume", 1).over(w)

    sma_20 = F.avg("Close").over(w20)
    std_20 = F.stddev_pop("Close").over(w20)  # stddev_pop to match ta library

    df = (
        df
        .withColumn("SMA_20", sma_20)
        .withColumn("BB_upper", sma_20 + 2 * std_20)
        .withColumn("BB_lower", sma_20 - 2 * std_20)
        .withColumn("Volume_pct", (F.col("Volume") - prev_vol) / prev_vol)
        .withColumn("Returns", (F.col("Close") - prev_close) / prev_close)
    )
    return df


# ── 3. Recursive features via applyInPandas ────────────────────────────────

def _compute_recursive_indicators(pdf: pd.DataFrame) -> pd.DataFrame:
    """Grouped-map UDF: receives one ticker's full history, returns with indicators.

    Uses the same `ta` library as features.py to guarantee numerical parity.
    """
    pdf = pdf.sort_values("Datetime")

    # EMA
    pdf["EMA_12"] = ta.trend.ema_indicator(pdf["Close"], window=12)

    # MACD
    macd_obj = ta.trend.MACD(pdf["Close"])
    pdf["MACD"] = macd_obj.macd()
    pdf["MACD_signal"] = macd_obj.macd_signal()

    # RSI
    pdf["RSI_14"] = ta.momentum.rsi(pdf["Close"], window=14)

    # ATR (Wilder's smoothing)
    pdf["ATR_14"] = ta.volatility.average_true_range(
        pdf["High"], pdf["Low"], pdf["Close"], window=14
    )

    return pdf


def compute_recursive_features(df: DataFrame) -> DataFrame:
    """Compute EMA_12, MACD, MACD_signal, RSI_14, ATR_14 via applyInPandas.

    Spark distributes one group per ticker; pandas computes within each group.
    """
    # Build output schema: input columns + 5 new indicator columns
    new_cols = [
        StructField("EMA_12", DoubleType(), True),
        StructField("MACD", DoubleType(), True),
        StructField("MACD_signal", DoubleType(), True),
        StructField("RSI_14", DoubleType(), True),
        StructField("ATR_14", DoubleType(), True),
    ]
    output_schema = StructType(list(df.schema.fields) + new_cols)

    df = df.groupBy("ticker").applyInPandas(
        _compute_recursive_indicators, schema=output_schema
    )
    return df


# ── 4. Assembly & output ────────────────────────────────────────────────────

def compute_all_features(df: DataFrame) -> DataFrame:
    """Compute all 10 technical indicators (window + recursive), drop nulls."""
    print("  Computing window features (SMA, BB, Volume_pct, Returns)...")
    df = compute_window_features(df)

    print("  Computing recursive features (EMA, MACD, RSI, ATR) via applyInPandas...")
    df = compute_recursive_features(df)

    # Drop rows with NaN indicators (leading warmup period per ticker)
    indicator_cols = [c for c in FEATURE_COLUMNS if c not in ("Open", "High", "Low", "Close", "Volume")]
    for col in indicator_cols:
        df = df.filter(F.col(col).isNotNull())

    return df


def write_feature_parquet(df: DataFrame) -> str:
    """Write feature-enriched data as Parquet partitioned by (ticker, year, month)."""
    os.makedirs(PARQUET_DIR, exist_ok=True)

    # Cast Datetime to string to avoid Spark UTC timezone conversion issues
    df = df.withColumn("Datetime", F.date_format("Datetime", "yyyy-MM-dd HH:mm:ss"))

    # Select columns in the expected order + partition keys
    output_cols = ["Datetime"] + FEATURE_COLUMNS + ["ticker", "year", "month"]
    df = df.select(*output_cols)

    (
        df.write
        .mode("overwrite")
        .partitionBy("ticker", "year", "month")
        .parquet(PARQUET_DIR)
    )
    print(f"  Parquet written to {PARQUET_DIR}")
    return PARQUET_DIR


def compute_aggregates(df: DataFrame) -> DataFrame:
    """Legacy: per-ticker, per-month aggregate statistics."""
    return (
        df.groupBy("ticker", "year", "month")
        .agg(
            F.mean("Close").alias("mean_close"),
            F.stddev("Close").alias("std_close"),
            F.sum("Volume").alias("total_volume"),
            F.max("High").alias("max_high"),
            F.min("Low").alias("min_low"),
            ((F.max("High") - F.min("Low")) / F.mean("Close")).alias("volatility"),
        )
        .orderBy("ticker", "year", "month")
    )


# ── 5. Pipeline orchestrator ────────────────────────────────────────────────

def run_spark_pipeline():
    """Execute the full PySpark feature engineering pipeline.

    Returns:
        Path to the output Parquet directory.
    """
    spark = create_spark_session()
    try:
        t0 = time.time()

        # Load
        print("[Spark] Loading all CSVs across years", DATA_YEARS, "...")
        df_raw = load_all_csvs(spark)
        n_raw = df_raw.count()
        t_load = time.time() - t0
        print(f"  Loaded {n_raw:,} rows in {t_load:.1f}s")

        # Features
        print("[Spark] Computing technical indicators...")
        t1 = time.time()
        df_feat = compute_all_features(df_raw)
        n_feat = df_feat.count()
        t_feat = time.time() - t1
        print(f"  {n_feat:,} rows after dropping indicator warmup ({t_feat:.1f}s)")

        # Write Parquet
        print("[Spark] Writing feature Parquet...")
        t2 = time.time()
        parquet_path = write_feature_parquet(df_feat)
        t_write = time.time() - t2
        print(f"  Write complete ({t_write:.1f}s)")

        # Aggregates (legacy compat)
        print("[Spark] Computing aggregate statistics...")
        agg_df = compute_aggregates(df_raw)
        agg_path = os.path.join(OUTPUT_DIR, "nasdaq_aggregates")
        agg_df.write.mode("overwrite").parquet(agg_path)
        agg_df.show(10)

        # Summary
        t_total = time.time() - t0
        print(f"\n[Spark] Pipeline complete in {t_total:.1f}s")
        print(f"  Raw rows:     {n_raw:,}")
        print(f"  Feature rows: {n_feat:,}")
        print(f"  Output:       {parquet_path}")

        return parquet_path

    finally:
        spark.stop()


if __name__ == "__main__":
    run_spark_pipeline()
