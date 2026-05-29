"""
Deterministic baseline evaluators for time-series model comparison.

These are not trained ML models. They provide simple, auditable reference
points so advanced models must prove they beat reasonable alternatives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config.settings import DATA_SETTINGS


def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate the same core metrics used by trained models."""
    if len(y_true) == 0 or len(y_pred) == 0:
        return {}

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else 0.0

    if len(y_true) > 1:
        true_dir = np.diff(y_true) > 0
        pred_dir = np.diff(y_pred) > 0
        directional_accuracy = float(np.mean(true_dir == pred_dir) * 100)
    else:
        directional_accuracy = 0.0

    r_squared = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "mape": round(mape, 2),
        "directional_accuracy": round(directional_accuracy, 2),
        "r_squared": round(r_squared, 4),
    }


def _max_drawdown(prices: np.ndarray) -> float:
    """Calculate max drawdown for a price path."""
    if len(prices) == 0:
        return 0.0

    prices = np.asarray(prices, dtype=float)
    running_peak = np.maximum.accumulate(prices)
    drawdowns = np.where(running_peak > 0, (running_peak - prices) / running_peak, 0.0)
    return float(np.max(drawdowns)) if len(drawdowns) else 0.0


def evaluate_price_baselines(
    df: pd.DataFrame,
    benchmark_df: pd.DataFrame | None = None,
    train_ratio: float | None = None,
    val_ratio: float | None = None,
) -> dict[str, dict]:
    """
    Evaluate simple baselines on the chronological test segment.

    Baselines:
    - naive: predicts tomorrow's close equals the latest known close.
    - mean_return: applies the training-window mean daily return.
    - buy_hold: same zero-return price persistence plus portfolio-style fields.
    - benchmark: optional same-period benchmark buy-and-hold return if provided.
    """
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column")

    clean = df.copy().sort_index()
    close = clean["close"].astype(float).dropna()

    if train_ratio is None:
        train_ratio = DATA_SETTINGS["train_ratio"]
    if val_ratio is None:
        val_ratio = DATA_SETTINGS["val_ratio"]

    n = len(close)
    train_end = int(n * train_ratio)
    test_start = int(n * (train_ratio + val_ratio))

    if n < 3 or train_end < 2 or test_start <= 0 or test_start >= n:
        return {}

    y_true = close.iloc[test_start:].values
    previous_close = close.iloc[test_start - 1 : n - 1].values

    train_returns = close.iloc[:train_end].pct_change().dropna()
    mean_return = float(train_returns.mean()) if len(train_returns) else 0.0

    baselines = {
        "naive": {
            **_calculate_metrics(y_true, previous_close),
            "baseline_type": "last_close",
            "description": "Predict next close as the latest known close.",
        },
        "mean_return": {
            **_calculate_metrics(y_true, previous_close * (1.0 + mean_return)),
            "baseline_type": "train_mean_return",
            "mean_daily_return": round(mean_return, 8),
            "description": "Apply the train-window mean daily return to the latest known close.",
        },
    }

    test_path = close.iloc[test_start - 1 :].values
    buy_hold_return = float((test_path[-1] - test_path[0]) / test_path[0]) if test_path[0] else 0.0
    baselines["buy_hold"] = {
        **_calculate_metrics(y_true, previous_close),
        "baseline_type": "buy_and_hold",
        "total_return": round(buy_hold_return, 4),
        "max_drawdown": round(_max_drawdown(test_path), 4),
        "description": "Hold the asset through the test window; price metrics use zero-return persistence.",
    }

    if benchmark_df is not None and not benchmark_df.empty and "close" in benchmark_df.columns:
        aligned = pd.DataFrame({"asset": close}).join(
            benchmark_df[["close"]].rename(columns={"close": "benchmark"}),
            how="inner",
        )
        bench_path = aligned["benchmark"].iloc[test_start - 1 :] if len(aligned) > test_start else pd.Series(dtype=float)
        if len(bench_path) >= 2 and bench_path.iloc[0] != 0:
            benchmark_return = float((bench_path.iloc[-1] - bench_path.iloc[0]) / bench_path.iloc[0])
            baselines["benchmark"] = {
                "rmse": 0.0,
                "mae": 0.0,
                "mape": 0.0,
                "directional_accuracy": 0.0,
                "r_squared": 0.0,
                "baseline_type": "benchmark_buy_and_hold",
                "total_return": round(benchmark_return, 4),
                "max_drawdown": round(_max_drawdown(bench_path.values), 4),
                "description": "Same-period benchmark buy-and-hold return; price-error metrics are not applicable.",
            }

    return baselines
