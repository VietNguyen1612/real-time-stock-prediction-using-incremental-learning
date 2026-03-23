"""PySpark preprocessing pipeline for NASDAQ 2022 data.

Demonstrates Big Data processing: loads all 96 monthly CSVs into a single
Spark DataFrame, computes aggregate statistics, and writes enriched data
as Parquet partitioned by ticker.
"""

import os
import glob as globmod
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType

from src.config import DATA_DIR, OUTPUT_DIR


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("NASDAQ_2022_Preprocessing")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )


def load_all_csvs(spark: SparkSession):
    """Load all monthly CSVs into one DataFrame with a `ticker` column."""
    schema = StructType([
        StructField("Datetime", StringType(), True),
        StructField("Open", DoubleType(), True),
        StructField("High", DoubleType(), True),
        StructField("Low", DoubleType(), True),
        StructField("Close", DoubleType(), True),
        StructField("Volume", DoubleType(), True),
    ])

    all_files = globmod.glob(os.path.join(DATA_DIR, "*", "*_15min.csv"))
    # Exclude full-year files
    monthly_files = [f for f in all_files if "full" not in f]

    dfs = []
    for fpath in monthly_files:
        ticker = os.path.basename(os.path.dirname(fpath))
        df = (
            spark.read.csv(fpath, header=True, schema=schema)
            .withColumn("ticker", F.lit(ticker))
            .withColumn("Datetime", F.to_timestamp("Datetime"))
        )
        dfs.append(df)

    combined = dfs[0]
    for df in dfs[1:]:
        combined = combined.union(df)

    return combined


def compute_aggregates(df):
    """Compute per-ticker, per-month aggregate statistics."""
    df_with_month = df.withColumn("month", F.month("Datetime"))

    agg_df = (
        df_with_month.groupBy("ticker", "month")
        .agg(
            F.mean("Close").alias("mean_close"),
            F.stddev("Close").alias("std_close"),
            F.sum("Volume").alias("total_volume"),
            F.max("High").alias("max_high"),
            F.min("Low").alias("min_low"),
            # Volatility: std of returns approximated by (max-min)/mean
            ((F.max("High") - F.min("Low")) / F.mean("Close")).alias("volatility"),
        )
        .orderBy("ticker", "month")
    )
    return agg_df


def write_parquet(df, name="nasdaq_2022_enriched"):
    """Write DataFrame as Parquet partitioned by ticker."""
    output_path = os.path.join(OUTPUT_DIR, name)
    (
        df.write
        .mode("overwrite")
        .partitionBy("ticker")
        .parquet(output_path)
    )
    print(f"Written Parquet to {output_path}")
    return output_path


def run_spark_pipeline():
    """Execute the full PySpark preprocessing pipeline."""
    spark = create_spark_session()
    try:
        print("Loading all CSVs into Spark...")
        df = load_all_csvs(spark)
        total_rows = df.count()
        print(f"Total rows: {total_rows}")

        print("Computing aggregate statistics...")
        agg_df = compute_aggregates(df)
        agg_df.show(10)

        print("Writing enriched data as Parquet...")
        write_parquet(df)

        print("Writing aggregate stats as Parquet...")
        write_parquet(agg_df, name="nasdaq_2022_aggregates")

        print("PySpark pipeline complete.")
        return df, agg_df
    finally:
        spark.stop()


if __name__ == "__main__":
    run_spark_pipeline()
