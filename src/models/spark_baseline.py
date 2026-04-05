"""Spark MLlib baseline models for stock price-change prediction.

Trains LinearRegression and RandomForestRegressor on the same temporal
split and target variable as the LSTM pipeline, enabling a fair comparison
between distributed ML (Spark) and single-GPU deep learning.
"""

import os
import time

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

from src.config import (
    PARQUET_DIR, TICKERS, OUTPUT_DIR,
    INITIAL_TRAIN_YEAR, INITIAL_TRAIN_MONTHS, VALIDATION_MONTH,
    TEST_YEAR, TEST_MONTH, FORECAST_HORIZON,
)
from src.evaluation.metrics import compute_metrics


FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_20", "EMA_12", "MACD", "MACD_signal", "RSI_14",
    "BB_upper", "BB_lower", "ATR_14", "Volume_pct", "Returns",
]


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("NASDAQ_MLlib_Baselines")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )


def load_features(spark: SparkSession, tickers: list[str] = None) -> DataFrame:
    """Load Spark-produced feature Parquet for given tickers."""
    df = spark.read.parquet(PARQUET_DIR)
    if tickers:
        df = df.filter(F.col("ticker").isin(tickers))
    return df


def add_target(df: DataFrame, horizon: int = FORECAST_HORIZON) -> DataFrame:
    """Add target column: price change = Close[t+horizon] - Close[t].

    Same target as the LSTM pipeline (features.py create_sequences).
    """
    w = Window.partitionBy("ticker").orderBy("Datetime")
    df = df.withColumn(
        "target",
        F.lead("Close", horizon).over(w) - F.col("Close")
    )
    # Drop rows where target is null (last `horizon` rows per ticker)
    df = df.filter(F.col("target").isNotNull())
    return df


def split_by_time(df: DataFrame) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Split into train/val/test using the same temporal split as the LSTM pipeline.

    Train: INITIAL_TRAIN_MONTHS of INITIAL_TRAIN_YEAR (2022 Jan-May)
    Val:   VALIDATION_MONTH of INITIAL_TRAIN_YEAR (2022 Jun)
    Test:  TEST_MONTH of TEST_YEAR (2025 Dec)
    """
    train_months = [int(m) for m in INITIAL_TRAIN_MONTHS]
    val_month = int(VALIDATION_MONTH)
    test_month = int(TEST_MONTH)

    df_train = df.filter(
        (F.col("year") == INITIAL_TRAIN_YEAR) & (F.col("month").isin(train_months))
    )
    df_val = df.filter(
        (F.col("year") == INITIAL_TRAIN_YEAR) & (F.col("month") == val_month)
    )
    df_test = df.filter(
        (F.col("year") == TEST_YEAR) & (F.col("month") == test_month)
    )
    return df_train, df_val, df_test


def train_and_evaluate(
    df_train: DataFrame, df_test: DataFrame, assembler: VectorAssembler
) -> dict:
    """Train LR and RF, return metrics for both."""
    train_assembled = assembler.transform(df_train).select("features", "target")
    test_assembled = assembler.transform(df_test).select("features", "target")

    results = {}

    # ── Linear Regression ────────────────────────────────────────────
    t0 = time.time()
    lr = LinearRegression(
        featuresCol="features", labelCol="target",
        maxIter=100, regParam=0.01, elasticNetParam=0.0,
    )
    lr_model = lr.fit(train_assembled)
    lr_time = time.time() - t0

    lr_preds = lr_model.transform(test_assembled)
    lr_pdf = lr_preds.select("target", "prediction").toPandas()
    lr_metrics = compute_metrics(
        lr_pdf["target"].values, lr_pdf["prediction"].values
    )
    lr_metrics["train_time"] = lr_time
    results["LinearRegression"] = lr_metrics

    # ── Random Forest ────────────────────────────────────────────────
    t0 = time.time()
    rf = RandomForestRegressor(
        featuresCol="features", labelCol="target",
        numTrees=100, maxDepth=10, seed=42,
    )
    rf_model = rf.fit(train_assembled)
    rf_time = time.time() - t0

    rf_preds = rf_model.transform(test_assembled)
    rf_pdf = rf_preds.select("target", "prediction").toPandas()
    rf_metrics = compute_metrics(
        rf_pdf["target"].values, rf_pdf["prediction"].values
    )
    rf_metrics["train_time"] = rf_time
    results["RandomForest"] = rf_metrics

    return results


def run_spark_baselines(tickers: list[str] = None) -> pd.DataFrame:
    """Run Spark MLlib baselines for given tickers (default: all).

    Trains across ALL tickers simultaneously — demonstrating distributed ML.

    Returns:
        DataFrame with columns: model, RMSE, MAE, MAPE, R2, DirAcc, train_time
    """
    if tickers is None:
        tickers = TICKERS

    spark = create_spark_session()
    try:
        print(f"[MLlib] Loading features for {len(tickers)} tickers...")
        df = load_features(spark, tickers)
        df = add_target(df)

        df_train, df_val, df_test = split_by_time(df)
        n_train = df_train.count()
        n_test = df_test.count()
        print(f"  Train: {n_train:,} rows ({len(tickers)} tickers)")
        print(f"  Test:  {n_test:,} rows")

        assembler = VectorAssembler(
            inputCols=FEATURE_COLS, outputCol="features", handleInvalid="skip"
        )

        print("[MLlib] Training LinearRegression + RandomForest...")
        results = train_and_evaluate(df_train, df_test, assembler)

        # Print results
        print(f"\n{'Model':<22} {'RMSE':>10} {'MAE':>10} {'R2':>8} {'DirAcc':>8} {'Time':>8}")
        print("-" * 70)
        rows = []
        for model_name, metrics in results.items():
            print(f"{model_name:<22} {metrics['RMSE']:>10.6f} {metrics['MAE']:>10.6f} "
                  f"{metrics['R2']:>8.4f} {metrics['DirAcc']:>7.1f}% "
                  f"{metrics['train_time']:>7.1f}s")
            rows.append({"model": model_name, **metrics})

        summary = pd.DataFrame(rows)
        summary_path = os.path.join(OUTPUT_DIR, "spark_baseline_summary.csv")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        print(f"\nSaved to {summary_path}")

        return summary

    finally:
        spark.stop()


if __name__ == "__main__":
    run_spark_baselines()
