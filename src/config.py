"""Hyperparameters, paths, and ticker list for the stock prediction project."""

import os

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "NASDAQ_2022")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# ── Tickers ──────────────────────────────────────────────────────────────────
TICKERS = ["AAPL", "AMZN", "BRK-B", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]

# ── Phase 1: Offline base training (static 6-month block of 2022) ───────────
# The base LSTM is trained once, offline, on a fixed 6-month block of 2022.
# Internally that block is split into 5 training months + 1 validation month.
INITIAL_TRAIN_YEAR = 2022
INITIAL_TRAIN_MONTHS = ["01", "02", "03", "04", "05"]  # 2022 Jan-May (train)
VALIDATION_MONTH = "06"                                  # 2022 Jun (validation)
PHASE1_YEAR = INITIAL_TRAIN_YEAR
PHASE1_MONTHS = INITIAL_TRAIN_MONTHS + [VALIDATION_MONTH]  # full Jan-Jun block

# ── Phase 2: Tick-based retraining (subsequent 6 months of 2022) ────────────
# After the offline base model is trained, the system streams the next
# 6 months of 2022 (Jul-Dec) and retrains on every "tick". A tick is defined
# precisely as every 15 minutes — i.e. each new 15-min OHLCV bar produces one
# new sliding-window sequence that triggers a single incremental update.
ALL_MONTHS = [f"{m:02d}" for m in range(1, 13)]
PHASE2_YEAR = 2022
PHASE2_MONTHS = ["07", "08", "09", "10", "11", "12"]  # 2022 Jul-Dec

TICK_INTERVAL_MINUTES = 15   # one tick == one 15-min bar
BARS_PER_TICK = 1            # bars consumed per tick (data cadence is 15 min)
TICK_EPOCHS = 1              # gradient passes per tick (online retraining)
EWC_REFRESH_EVERY = 26       # refresh Fisher every N ticks (26 bars ≈ 1 day)


def build_tick_schedule():
    """Build the list of (year, month) blocks streamed during Phase 2.

    The actual retraining cadence is per-tick (every 15 minutes); this only
    enumerates which monthly data blocks make up the Phase 2 stream.
    """
    return [(PHASE2_YEAR, m) for m in PHASE2_MONTHS]


PHASE2_SCHEDULE = build_tick_schedule()

# ── Final unseen test ───────────────────────────────────────────────────────
# Held-out month used by the classic-ML (sklearn) baselines for comparison.
TEST_YEAR = 2022
TEST_MONTH = "12"  # 2022 Dec — last month of the Phase 2 stream

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

# ── Device ───────────────────────────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
