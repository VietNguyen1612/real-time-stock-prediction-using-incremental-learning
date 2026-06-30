# Stock Prediction with Incremental Learning

NASDAQ stock prediction using an **LSTM** for deep learning and **EWC + Experience Replay** for incremental learning — on 15-minute intraday data.

The training pipeline runs in **two phases** on 2022 data:

- **Phase 1 — Offline base training:** train the LSTM once, offline, on a static 6-month block of 2022 (Jan–Jun).
- **Phase 2 — Tick-based retraining:** stream the subsequent 6 months (Jul–Dec 2022) and retrain on **every tick**, where a tick is defined precisely as every **15 minutes** (one new 15-min bar → one incremental update).

## Prerequisites

- Python 3.9+
- CUDA-capable GPU (optional, auto-detected)

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `torch`, `pandas`, `numpy`, `scikit-learn`, `ta`, `matplotlib`, `seaborn`

## Project Structure

```
.
├── src/
│   ├── config.py                    # Hyperparameters, paths, schedule
│   ├── data/
│   │   ├── loader.py                # CSV loading + feature enrichment
│   │   └── features.py              # Feature engineering (technical indicators)
│   ├── models/
│   │   ├── lstm_model.py            # StockLSTM architecture
│   │   ├── trainer.py               # Batch & incremental trainers
│   │   └── sklearn_baseline.py      # Classic-ML baselines (LR, RF)
│   ├── incremental/
│   │   ├── incremental_learner.py   # EWC & ReplayBuffer
│   │   └── tick_pipeline.py         # Phase 2 tick-based retraining loop
│   ├── evaluation/
│   │   └── metrics.py               # RMSE, MAE, MAPE, R², DirAcc
│   └── utils/
│       └── plotting.py              # Visualization functions
├── experiments/
│   ├── run_experiment.py            # Two-phase pipeline (offline base + tick retraining)
│   └── run_incremental_study.py     # Tick-based ablation study (4 methods compared)
├── NASDAQ_2022/                     # 2022 data (8 tickers × 12 months) — used by the pipeline
├── NASDAQ_2023..2025/               # Additional years on disk (not used by the 2022 pipeline)
├── outputs/                         # Models, scalers, plots, summaries
├── requirements.txt
└── README.md
```

## Data

Pre-downloaded NASDAQ 15-min OHLCV data:

- **Tickers**: AAPL, AMZN, BRK-B, GOOGL, META, MSFT, NVDA, TSLA
- **Year used by the pipeline**: 2022 (2023–2025 are also on disk but unused)
- **Format**: 15-minute bars, one CSV per month per ticker
- **Path pattern**: `NASDAQ_{YEAR}/{TICKER}/{TICKER}_{YEAR}-{MM}_15min.csv`

Technical-indicator features (SMA_20, EMA_12, MACD, MACD_signal, RSI_14, BB_upper,
BB_lower, ATR_14, Volume_pct, Returns) are computed on the fly from the raw OHLCV
CSVs by `src/data/features.py` (via the `ta` library) — 15 features total.

## How to Run

### Main Experiment (two-phase pipeline)

```bash
# All 8 tickers
python -m experiments.run_experiment

# Single ticker
python -m experiments.run_experiment --ticker AAPL

# Quick run: cap the number of Phase 2 ticks
python -m experiments.run_experiment --ticker AAPL --max-ticks 500
```

**Pipeline:**
1. **Phase 1 (offline base training):** batch-train LSTM on Jan–May 2022 (200 epochs, early stopping), validate on Jun 2022.
2. **Phase 2 (tick-based retraining):** stream Jul–Dec 2022 and retrain on **every tick** (1 tick = 15 min = 1 new bar) with EWC + Replay. Each tick is evaluated prequentially (predict-then-train).
3. Batch-retrain baseline on all of 2022 for a speed/accuracy comparison.

**Output:**
- `outputs/experiment_summary.csv` — metrics for all tickers
- `outputs/{TICKER}_incremental_model.pt` — saved model
- `outputs/*.png` — prediction plots, loss curves, forgetting analysis

### Ablation Study (tick-based)

Compares Fine-tune only vs EWC only vs Replay only vs EWC+Replay, all under the
Phase 2 tick-based retraining regime (Jul–Dec 2022).

```bash
python -m experiments.run_incremental_study --ticker AAPL
python -m experiments.run_incremental_study --ticker AAPL --max-ticks 100  # quick run
```

**Output:**
- `outputs/incremental_study_summary.csv`
- `outputs/*_ablation_*.png`, `outputs/*_forgetting_heatmap.png`

### Classic-ML Baselines

Compare scikit-learn LinearRegression and RandomForest against the LSTM, on the
same temporal split and target:

```bash
python -m src.models.sklearn_baseline                 # all tickers
python -m src.models.sklearn_baseline --ticker AAPL   # single ticker
```

Outputs `outputs/baseline_summary.csv`.

## Data Flow

```
NASDAQ_2022/ (raw 15-min CSVs)
        │
        ▼
  Feature engineering (features.py, via `ta`)
  └─ SMA, EMA, MACD, RSI, Bollinger Bands, ATR, Volume %, Returns → 15 features
        │
        ├──► LSTM + EWC + Replay (trainer.py, incremental_learner.py, tick_pipeline.py)
        │    ┌─ Phase 1 (offline base): batch train Jan–May 2022, validate Jun 2022
        │    └─ Phase 2 (tick retraining): Jul–Dec 2022, retrain every 15-min tick
        │
        └──► Classic-ML baselines (sklearn_baseline.py)
             ┌─ LinearRegression
             └─ RandomForestRegressor
```

## Configuration

All hyperparameters in `src/config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `LOOKBACK_WINDOW` | 78 | Input sequence (78 bars = 3 trading days) |
| `FORECAST_HORIZON` | 26 | Prediction horizon (26 bars = 1 trading day) |
| `HIDDEN_SIZE` | 128 | LSTM hidden units |
| `NUM_LAYERS` | 2 | LSTM layers |
| `DROPOUT` | 0.2 | Dropout rate |
| `INITIAL_EPOCHS` | 200 | Max batch training epochs |
| `BATCH_SIZE` | 64 | Training batch size |
| `LR` | 1e-3 | Batch learning rate |
| `EARLY_STOPPING_PATIENCE` | 20 | Early stopping patience |
| `INCREMENTAL_EPOCHS` | 5 | Epochs per incremental update (batch mode) |
| `INCREMENTAL_LR` | 1e-4 | Incremental learning rate |
| `EWC_LAMBDA` | 0.4 | EWC regularization weight |
| `REPLAY_ALPHA` | 0.2 | Replay loss weight |
| `REPLAY_RATIO` | 0.1 | Fraction of initial data in replay buffer |
| `PHASE1_MONTHS` | Jan–Jun 2022 | Static 6-month block for offline base training |
| `PHASE2_MONTHS` | Jul–Dec 2022 | Subsequent 6 months streamed for tick retraining |
| `TICK_INTERVAL_MINUTES` | 15 | One tick = one 15-min bar |
| `TICK_EPOCHS` | 1 | Gradient passes per tick (online retraining) |
| `EWC_REFRESH_EVERY` | 26 | Refresh Fisher every N ticks (≈ 1 trading day) |

## How to Predict

After training:

```python
import torch
from src.config import DEVICE, LOOKBACK_WINDOW, FORECAST_HORIZON
from src.models.lstm_model import StockLSTM
from src.data.loader import load_monthly_featured
from src.data.features import create_sequences, scale_with_scaler
import pickle

# 1. Load data (features computed on the fly from the raw CSV)
feat = load_monthly_featured("AAPL", "12", year=2022)

# 2. Load scaler and scale
with open("outputs/AAPL_feature_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
scaled = scale_with_scaler(scaler, feat)
X, y, ref_close = create_sequences(scaled, LOOKBACK_WINDOW, FORECAST_HORIZON)

# 3. Load model and predict
model = StockLSTM(n_features=X.shape[2])
model.load_state_dict(torch.load("outputs/AAPL_incremental_model.pt", map_location=DEVICE))
model.to(DEVICE)
model.eval()

with torch.no_grad():
    predictions = model(torch.tensor(X, device=DEVICE)).cpu().numpy()

print(f"Predictions: {predictions.shape}")  # (N,) predicted price changes
```
