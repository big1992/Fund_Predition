"""
Validation report-card helpers for per-symbol model quality summaries.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


BASELINE_PRIORITY = ("naive", "mean_return", "buy_hold", "benchmark")
MODEL_PRIORITY = ("lstm", "xgboost", "autogluon", "ensemble")


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _pick_baseline_mape(metrics_by_name: dict[str, dict]) -> tuple[float | None, str | None]:
    for name in BASELINE_PRIORITY:
        row = metrics_by_name.get(name)
        if isinstance(row, dict):
            mape = _to_float(row.get("mape"))
            if mape is not None and mape >= 0:
                return mape, name
    return None, None


def _grade_model(mape: float | None, directional_accuracy: float | None, improvement_pct: float | None) -> tuple[str, list[str]]:
    caveats: list[str] = []

    if mape is None or directional_accuracy is None:
        return "fail", ["Missing core metrics (MAPE/directional_accuracy)."]

    if improvement_pct is None:
        caveats.append("No baseline comparison available.")
    elif improvement_pct < 0:
        caveats.append("Model underperforms selected baseline on MAPE.")

    if mape <= 5.0 and directional_accuracy >= 52.0 and (improvement_pct is None or improvement_pct >= 0):
        status = "pass"
    elif mape <= 10.0 and directional_accuracy >= 48.0 and (improvement_pct is None or improvement_pct >= -5.0):
        status = "warn"
    else:
        status = "fail"

    return status, caveats


def build_validation_report(symbol: str, metrics_by_name: dict[str, dict]) -> dict:
    """
    Build a per-symbol validation report card.

    metrics_by_name expects keys like:
    - lstm / xgboost / autogluon / ensemble
    - naive / mean_return / buy_hold / benchmark (optional)
    """
    baseline_mape, baseline_name = _pick_baseline_mape(metrics_by_name)
    cards = []

    for model_name in MODEL_PRIORITY:
        row = metrics_by_name.get(model_name)
        if not isinstance(row, dict) or "error" in row:
            continue

        mape = _to_float(row.get("mape"))
        da = _to_float(row.get("directional_accuracy"))
        rmse = _to_float(row.get("rmse"))
        mae = _to_float(row.get("mae"))
        r2 = _to_float(row.get("r_squared"))

        improvement_pct = None
        if baseline_mape is not None and mape is not None and baseline_mape > 0:
            improvement_pct = ((baseline_mape - mape) / baseline_mape) * 100.0

        status, caveats = _grade_model(mape, da, improvement_pct)
        if "n_folds" in row and int(row.get("n_folds") or 0) < 3:
            caveats.append("Validation folds are limited; confidence may be unstable.")

        cards.append(
            {
                "model_name": model_name,
                "status": status,
                "rmse": rmse,
                "mae": mae,
                "mape": mape,
                "directional_accuracy": da,
                "r_squared": r2,
                "baseline_name": baseline_name,
                "baseline_mape": baseline_mape,
                "mape_improvement_pct": round(improvement_pct, 2) if improvement_pct is not None else None,
                "caveats": caveats,
            }
        )

    for baseline_name_key in BASELINE_PRIORITY:
        row = metrics_by_name.get(baseline_name_key)
        if not isinstance(row, dict):
            continue
        cards.append(
            {
                "model_name": baseline_name_key,
                "status": "reference",
                "rmse": _to_float(row.get("rmse")),
                "mae": _to_float(row.get("mae")),
                "mape": _to_float(row.get("mape")),
                "directional_accuracy": _to_float(row.get("directional_accuracy")),
                "r_squared": _to_float(row.get("r_squared")),
                "baseline_name": None,
                "baseline_mape": None,
                "mape_improvement_pct": None,
                "caveats": [row.get("description")] if row.get("description") else [],
            }
        )

    return {
        "symbol": symbol,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "cards": cards,
    }
