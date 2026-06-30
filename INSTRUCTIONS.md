# How to Run This Project

A two-phase stock-prediction pipeline on 2022 NASDAQ 15-min data:

- **Phase 1 — Offline base training:** an LSTM is trained once, offline, on a
  static 6-month block of 2022 (Jan–Jun).
- **Phase 2 — Tick-based retraining:** the next 6 months (Jul–Dec 2022) are
  streamed and the model is retrained on **every tick**, where a tick is
  defined precisely as every **15 minutes** (one new bar → one update).

Forgetting is mitigated with **EWC + Experience Replay**. See `README.md` for
architecture and the full project structure.

## 1. Clone & Setup Environment

```bash
git clone <your-repo-url>
cd big-data
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Prepare Data

The pipeline runs on the 2022 NASDAQ 15-min intraday CSVs in `NASDAQ_2022/`.

```bash
unzip stock_data_NASDAQ_2022-*.zip -d NASDAQ_2022/
```

Technical-indicator features are computed on the fly from the raw CSVs when the
experiments run — there is no separate preprocessing step.

## 3. Run the Main Experiment (two-phase pipeline)

```bash
# All 8 tickers (AAPL, AMZN, BRK-B, GOOGL, META, MSFT, NVDA, TSLA)
python -m experiments.run_experiment

# Single ticker
python -m experiments.run_experiment --ticker AAPL

# Quick run: cap the number of Phase 2 ticks
python -m experiments.run_experiment --ticker AAPL --max-ticks 500
```

**What it does:**

1. **Phase 1 (offline base training):** batch-trains the LSTM on 2022 Jan–May
   (200 epochs, early stopping), validates on 2022 Jun.
2. **Phase 2 (tick-based retraining):** streams 2022 Jul–Dec and retrains on
   every tick (1 tick = 15 min = 1 new bar). Each tick is evaluated
   prequentially (predict-then-train) with EWC + Replay.
3. **Baseline:** batch-retrains the LSTM from scratch on all of 2022 to compare
   accuracy and training time (speedup) against the incremental approach.

## 4. Run the Incremental Learning Study (ablation)

```bash
python -m experiments.run_incremental_study
python -m experiments.run_incremental_study --ticker AAPL
python -m experiments.run_incremental_study --ticker AAPL --max-ticks 100   # quick
```

**What it does:** runs the Phase 2 tick stream four times — Fine-tune vs
EWC-only vs Replay-only vs EWC+Replay — and produces ablation comparison plots
and a forgetting heatmap over past months.

## 5. Run the Classic-ML Baselines

```bash
python -m src.models.sklearn_baseline
python -m src.models.sklearn_baseline --ticker AAPL
```

Trains scikit-learn LinearRegression + RandomForest on the same temporal split
and target as the LSTM, for a "classic ML vs deep learning" comparison.

## 6. Outputs

Everything is written to `outputs/`.

**Summary tables (CSV)**

| File | From |
|------|------|
| `experiment_summary.csv` | main experiment (per ticker) |
| `incremental_study_summary.csv` | ablation study (per ticker × method) |
| `baseline_summary.csv` | classic-ML baselines (per ticker × model) |

**Saved models / scalers (per ticker)**

- `{TICKER}_incremental_model.pt` — trained LSTM weights
- `{TICKER}_feature_scaler.pkl`, `{TICKER}_y_scaler.pkl` — frozen scalers

**Plots (per ticker, PNG)**

- `{TICKER}_lossbatch.png` — Phase 1 training loss curve
- `{TICKER}_predictionsbatch_val.png` — Phase 1 validation predictions
- `{TICKER}_predictionsphase2_ticks.png` — Phase 2 prequential predictions
- `{TICKER}_metrics_over_months.png` — Phase 2 metrics aggregated per month
- `{TICKER}_forgetting.png` — accuracy on Jan 2022 before vs after Phase 2
- `{TICKER}_training_time.png` — incremental vs full-retrain training time
- `{TICKER}_ablation_comparison.png`, `{TICKER}_ablation_forgetting.png`,
  `{TICKER}_forgetting_heatmap.png` — from the ablation study
