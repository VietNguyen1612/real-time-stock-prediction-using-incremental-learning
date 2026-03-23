"""Evaluation metrics for regression."""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute RMSE, MAE, MAPE, R², and Directional Accuracy."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    # Avoid division by zero in MAPE
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    r2 = r2_score(y_true, y_pred)

    # Directional accuracy: did we predict up/down correctly?
    # Only count samples where price actually moved
    moved = y_true != 0
    if moved.sum() > 0:
        dir_acc = np.mean(np.sign(y_true[moved]) == np.sign(y_pred[moved])) * 100
    else:
        dir_acc = 50.0

    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2, "DirAcc": dir_acc}
