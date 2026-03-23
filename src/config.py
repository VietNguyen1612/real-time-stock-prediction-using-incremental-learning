"""Hyperparameters, paths, and ticker list for the stock prediction project."""

import os

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "NASDAQ_2022")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# ── Tickers ──────────────────────────────────────────────────────────────────
TICKERS = ["AAPL", "AMZN", "BRK-B", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]

# ── Data split ──────────────────────────────────────────────────────────────
INITIAL_TRAIN_YEAR = 2022
INITIAL_TRAIN_MONTHS = ["01", "02", "03", "04", "05"]  # 2022 Jan-May batch train
VALIDATION_MONTH = "06"                                  # 2022 Jun validation

# ── Incremental learning schedule ───────────────────────────────────────────
# 2022 Jul-Dec → 2023 full → 2024 full → 2025 Jan-Nov = 41 months
ALL_MONTHS = [f"{m:02d}" for m in range(1, 13)]
INC_TEST_RATIO = 0.2  # hold out 20% of each incremental month for testing

def build_incremental_schedule():
    """Build list of (year, month) for incremental learning."""
    schedule = []
    # 2022 Jul-Dec
    for m in ["07", "08", "09", "10", "11", "12"]:
        schedule.append((2022, m))
    # 2023 full year
    for m in ALL_MONTHS:
        schedule.append((2023, m))
    # 2024 full year
    for m in ALL_MONTHS:
        schedule.append((2024, m))
    # 2025 Jan-Nov (learn)
    for m in [f"{i:02d}" for i in range(1, 12)]:
        schedule.append((2025, m))
    return schedule

INCREMENTAL_SCHEDULE = build_incremental_schedule()

# ── Final unseen test ───────────────────────────────────────────────────────
TEST_YEAR = 2025
TEST_MONTH = "12"  # 2025 Dec — never seen during any training

# ── Sliding window ───────────────────────────────────────────────────────────
LOOKBACK_WINDOW = 78   # 78 bars × 15 min = 3 trading days of context
FORECAST_HORIZON = 26  # 26 bars × 15 min = 1 trading day ahead

# ── LSTM architecture ────────────────────────────────────────────────────────
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2

# ── Batch training ───────────────────────────────────────────────────────────
INITIAL_EPOCHS = 200
BATCH_SIZE = 64
LR = 1e-3
EARLY_STOPPING_PATIENCE = 20
GRAD_CLIP = 1.0        # max gradient norm for clipping
SCHEDULER_PATIENCE = 7  # epochs before LR reduction

# ── Incremental training ────────────────────────────────────────────────────
INCREMENTAL_EPOCHS = 5
INCREMENTAL_LR = 1e-4
EWC_LAMBDA = 0.4
REPLAY_ALPHA = 0.2       # weight for replay loss
REPLAY_RATIO = 0.1       # fraction of initial data kept in replay buffer
REPLAY_MIX = 0.2         # 20% replay, 80% new data in each batch

# ── Device ───────────────────────────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
