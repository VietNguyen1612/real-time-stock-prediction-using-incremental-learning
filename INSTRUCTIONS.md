# How to Run This Project

## 1. Clone & Setup Environment

```bash
git clone <your-repo-url>
cd big-data
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Prepare Data

The project expects NASDAQ 15-min intraday CSVs in `NASDAQ_YYYY/` folders (2022-2025).

**Option A** — Use the existing 2022 zip:

```bash
unzip stock_data_NASDAQ_2022-*.zip -d NASDAQ_2022/
```

**Option B** — Fetch from AlphaVantage API (for 2023, 2024, 2025):

```bash
# Get a free API key from https://www.alphavantage.co/support/#api-key
python -m src.data.fetch_alphavantage --api_key YOUR_KEY --year 2023
python -m src.data.fetch_alphavantage --api_key YOUR_KEY --year 2024
python -m src.data.fetch_alphavantage --api_key YOUR_KEY --year 2025
```

## 3. (Optional) PySpark Preprocessing

```bash
python -m src.data.spark_preprocess
```

This loads all CSVs into Spark, computes aggregate stats, and writes Parquet files. Demonstrates the Big Data pipeline component.

## 4. Run Main Experiment

```bash
# All 8 tickers (AAPL, AMZN, BRK-B, GOOGL, META, MSFT, NVDA, TSLA)
python -m experiments.run_experiment

# Or a single ticker
python -m experiments.run_experiment --ticker AAPL
```

**What it does:**

- Batch trains LSTM on 2022 Jan-May
- Validates on 2022 Jun (early stopping)
- Incrementally updates (EWC + Replay) from 2022 Jul through 2025 Nov
- Tests on 2025 Dec (unseen)
- Saves plots and metrics to `outputs/`

## 5. Run Incremental Learning Study

```bash
python -m experiments.run_incremental_study
python -m experiments.run_incremental_study --ticker AAPL
```

**What it does:**

- Ablation study: Fine-tune vs EWC-only vs Replay-only vs EWC+Replay
- Forgetting heatmap analysis
- Saves comparison plots to `outputs/`

## 6. Real-Time Prediction

```bash
# Backtest on recent historical data
python predict_realtime.py --ticker AAPL --api_key YOUR_KEY

# Live forward prediction (every 15 min for 3 hours)
python predict_realtime.py --ticker AAPL --api_key YOUR_KEY --live --duration 180
```

Requires a trained model (`outputs/AAPL_incremental_model.pt`) from step 4.

## 7. Outputs

All results go to `outputs/`:

- `*_lossbatch.png` — training loss curves
- `*_predictionsbatch_val.png` — validation predictions
- `*_predictionsinc_*.png` — incremental predictions
- `*_forgetting.png` — forgetting analysis
- `*_metrics_over_months.png` — metrics over time
- `*_training_time.png` — training time comparison
- `experiment_summary.csv` — metrics summary table
