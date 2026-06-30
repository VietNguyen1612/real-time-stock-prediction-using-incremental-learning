"""Visualization utilities for the stock prediction project."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import OUTPUT_DIR

sns.set_theme(style="whitegrid")


def plot_predictions(y_true_diff, y_pred_diff, ticker, title_suffix="", save=True,
                     ref_close=None):
    """Plot actual vs predicted price levels/changes and directional accuracy.

    Args:
        y_true_diff: actual price changes (scaled differences)
        y_pred_diff: predicted price changes (scaled differences)
        ref_close: optional array of last-known close prices (scaled).
                   When provided, plots show price levels (ref + diff) instead
                   of raw differences, making the chart much easier to read.
    """
    # Decide whether to plot price levels or raw differences
    if ref_close is not None:
        y_true_plot = ref_close + y_true_diff
        y_pred_plot = ref_close + y_pred_diff
        y_label = "Price (scaled)"
        price_title = "Price Level"
    else:
        y_true_plot = y_true_diff
        y_pred_plot = y_pred_diff
        y_label = "Price change (scaled)"
        price_title = "Price Change"

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. Actual vs predicted price over time
    ax = axes[0, 0]
    ax.plot(y_true_plot, label="Actual", alpha=0.7, linewidth=0.8)
    ax.plot(y_pred_plot, label="Predicted", alpha=0.7, linewidth=0.8)
    if ref_close is None:
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title(f"{ticker} — {price_title}: Actual vs Predicted")
    ax.set_xlabel("Time step")
    ax.set_ylabel(y_label)
    ax.legend()

    # 2. Scatter plot: predicted vs actual
    ax = axes[0, 1]
    ax.scatter(y_true_plot, y_pred_plot, alpha=0.3, s=8)
    lims = [min(y_true_plot.min(), y_pred_plot.min()),
            max(y_true_plot.max(), y_pred_plot.max())]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    if ref_close is None:
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_title(f"{ticker} — Scatter: Predicted vs Actual")
    ax.set_xlabel(f"Actual {y_label.lower()}")
    ax.set_ylabel(f"Predicted {y_label.lower()}")
    ax.legend()

    # 3. Rolling directional accuracy (window=50)
    ax = axes[1, 0]
    correct = (np.sign(y_true_diff) == np.sign(y_pred_diff)).astype(float)
    window = min(50, len(correct) // 4)
    if window > 0:
        rolling_acc = pd.Series(correct).rolling(window, center=True).mean() * 100
        ax.plot(rolling_acc, color="teal", linewidth=1)
    ax.axhline(50, color="red", linewidth=1, linestyle="--", label="Random (50%)")
    ax.set_title(f"{ticker} — Rolling Directional Accuracy (window={window})")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend()

    # 4. Cumulative return: model vs actual
    ax = axes[1, 1]
    # If model predicts direction correctly, we gain; otherwise we lose
    model_returns = np.sign(y_pred_diff) * y_true_diff
    cum_model = np.cumsum(model_returns)
    cum_actual = np.cumsum(np.abs(y_true_diff))
    ax.plot(cum_model, label="Model strategy", color="teal")
    ax.axhline(0, color="red", linewidth=1, linestyle="--", label="Break-even")
    ax.set_title(f"{ticker} — Cumulative Return (model direction x actual move)")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Cumulative return (scaled)")
    ax.legend()

    plt.suptitle(f"{ticker} — Prediction Analysis {title_suffix}", fontsize=14, y=1.01)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, f"{ticker}_predictions{title_suffix.replace(' ', '_')}.png"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_over_months(metrics_df: pd.DataFrame, ticker: str, save=True):
    """Line chart of RMSE/MAE over incremental months."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = range(len(metrics_df))
    labels = metrics_df["month"].tolist()
    for ax, metric in zip(axes, ["RMSE", "MAE"]):
        ax.plot(x, metrics_df[metric], marker="o", markersize=3, linewidth=1)
        ax.set_title(f"{ticker} — {metric} over incremental months")
        ax.set_xlabel("Month")
        ax.set_ylabel(metric)
        # Show every Nth label to avoid clutter
        step = max(1, len(labels) // 10)
        ax.set_xticks([i for i in x if i % step == 0])
        ax.set_xticklabels([labels[i] for i in x if i % step == 0], rotation=45, ha="right")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, f"{ticker}_metrics_over_months.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_time_comparison(phase1_time, phase2_time, retrain_time, ticker, save=True):
    """Bar chart: incremental cost (Phase 1 base + Phase 2 ticks) vs full retrain.

    Args:
        phase1_time: offline base training time (seconds)
        phase2_time: total Phase 2 tick-retraining time (seconds)
        retrain_time: time to batch-retrain from scratch on all data (baseline)
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    categories = ["Phase 1\n(offline base)", "Phase 2\n(tick retraining)",
                  "Full retrain\n(baseline)"]
    values = [phase1_time, phase2_time, retrain_time]
    colors = ["#4C72B0", "#55A868", "#DD8452"]
    bars = ax.bar(categories, values, color=colors)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f}s", ha="center", va="bottom", fontsize=10)
    inc_total = phase1_time + phase2_time
    speedup = retrain_time / inc_total if inc_total > 0 else float("nan")
    ax.set_title(f"{ticker} — Training time: incremental vs full retrain "
                 f"(speedup {speedup:.2f}x)")
    ax.set_ylabel("Time (seconds)")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, f"{ticker}_training_time.png"), dpi=150)
    plt.close(fig)


def plot_forgetting_analysis(forgetting_metrics: list[dict], ticker: str, save=True):
    """Plot old-data (Jan 2022) performance after each incremental update."""
    months = [d["month"] for d in forgetting_metrics]
    rmses = [d["RMSE"] for d in forgetting_metrics]
    r2s = [d["R2"] for d in forgetting_metrics]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = range(len(months))

    axes[0].plot(x, rmses, marker="s", markersize=3, color="crimson", linewidth=1)
    axes[0].set_title(f"{ticker} — Forgetting: RMSE on Jan 2022 data")
    axes[0].set_xlabel("After training on month")
    axes[0].set_ylabel("RMSE")

    axes[1].plot(x, r2s, marker="s", markersize=3, color="teal", linewidth=1)
    axes[1].set_title(f"{ticker} — Forgetting: R² on Jan 2022 data")
    axes[1].set_xlabel("After training on month")
    axes[1].set_ylabel("R²")

    # Show every Nth label
    step = max(1, len(months) // 10)
    for ax in axes:
        ax.set_xticks([i for i in x if i % step == 0])
        ax.set_xticklabels([months[i] for i in x if i % step == 0], rotation=45, ha="right")

    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, f"{ticker}_forgetting.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_comparison(ablation_results: dict, ticker: str, save=True):
    """Compare RMSE/R² across ablation methods over incremental months.

    Args:
        ablation_results: {method_name: [{"month": ..., "RMSE": ..., "R2": ...}, ...]}
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"Fine-tune": "#e74c3c", "EWC only": "#3498db",
              "Replay only": "#2ecc71", "EWC + Replay": "#9b59b6"}

    for method, metrics_list in ablation_results.items():
        months = [m["month"] for m in metrics_list]
        rmses = [m["RMSE"] for m in metrics_list]
        r2s = [m["R2"] for m in metrics_list]
        color = colors.get(method, None)
        axes[0].plot(months, rmses, marker="o", label=method, color=color)
        axes[1].plot(months, r2s, marker="o", label=method, color=color)

    axes[0].set_title(f"{ticker} — Ablation: RMSE over months")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("RMSE")
    axes[0].legend()
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].set_title(f"{ticker} — Ablation: R² over months")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("R²")
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, f"{ticker}_ablation_comparison.png"), dpi=150)
    plt.close(fig)


def plot_ablation_forgetting(ablation_forgetting: dict, ticker: str, save=True):
    """Compare forgetting (Jan R²) across ablation methods.

    Args:
        ablation_forgetting: {method_name: [{"month": ..., "R2": ...}, ...]}
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"Fine-tune": "#e74c3c", "EWC only": "#3498db",
              "Replay only": "#2ecc71", "EWC + Replay": "#9b59b6"}

    for method, metrics_list in ablation_forgetting.items():
        months = [m["month"] for m in metrics_list]
        r2s = [m["R2"] for m in metrics_list]
        color = colors.get(method, None)
        ax.plot(months, r2s, marker="s", label=method, color=color)

    ax.set_title(f"{ticker} — Forgetting Analysis: R² on Jan 2022 data")
    ax.set_xlabel("After training on month")
    ax.set_ylabel("R² on Jan 2022")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, f"{ticker}_ablation_forgetting.png"), dpi=150)
    plt.close(fig)


def plot_forgetting_heatmap(heatmap_data: pd.DataFrame, ticker: str, save=True):
    """Heatmap of R² on past months after each incremental update.

    Args:
        heatmap_data: DataFrame with rows=eval_month, cols=after_training_month, values=R²
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=0.8, vmax=1.0, ax=ax, linewidths=0.5)
    ax.set_title(f"{ticker} — Forgetting Heatmap: R² on past months")
    ax.set_xlabel("After incremental update on month")
    ax.set_ylabel("Evaluated on month")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, f"{ticker}_forgetting_heatmap.png"), dpi=150)
    plt.close(fig)


def plot_loss_curves(history: dict, ticker: str, title_suffix="", save=True):
    """Plot training (and optionally validation) loss curves."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train_loss"], label="Train loss")
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="Val loss")
    ax.set_title(f"{ticker} — Loss curve {title_suffix}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, f"{ticker}_loss{title_suffix.replace(' ', '_')}.png"), dpi=150)
    plt.close(fig)
