"""
Ensemble model combining LSTM, XGBoost, and AutoGluon predictions.
Supports static and dynamic weighting for 2 or 3 models.
"""

import numpy as np
import logging
from typing import Optional

from config.settings import ENSEMBLE_PARAMS

logger = logging.getLogger(__name__)


class EnsembleModel:
    """Weighted ensemble of LSTM, XGBoost, and AutoGluon models."""

    def __init__(self, params: dict = None):
        self.params = params or ENSEMBLE_PARAMS
        self.lstm_weight = self.params["lstm_weight"]
        self.xgboost_weight = self.params["xgboost_weight"]
        self.autogluon_weight = self.params.get("autogluon_weight", 0.0)

    def predict(
        self,
        lstm_predictions: np.ndarray,
        xgboost_predictions: np.ndarray,
        autogluon_predictions: np.ndarray = None,
    ) -> np.ndarray:
        """
        Combine predictions using weighted average.
        Supports 2-model (LSTM+XGB) or 3-model (LSTM+XGB+AG) ensembles.
        Automatically normalizes weights based on available models.
        """
        available = []
        weights = []

        if lstm_predictions is not None and len(lstm_predictions) > 0:
            available.append(("LSTM", lstm_predictions, self.lstm_weight))
        if xgboost_predictions is not None and len(xgboost_predictions) > 0:
            available.append(("XGB", xgboost_predictions, self.xgboost_weight))
        if autogluon_predictions is not None and len(autogluon_predictions) > 0:
            available.append(("AG", autogluon_predictions, self.autogluon_weight))

        if not available:
            return np.array([])

        if len(available) == 1:
            name, preds, _ = available[0]
            logger.info("Single model ensemble: %s, %d points", name, len(preds))
            return preds

        # Find minimum length across all available predictions
        min_len = min(len(item[1]) for item in available)

        # Truncate and normalize weights
        total_weight = sum(item[2] for item in available)
        ensemble = np.zeros(min_len)

        for name, preds, weight in available:
            normalized_weight = weight / total_weight if total_weight > 0 else 1.0 / len(available)
            ensemble += normalized_weight * preds[:min_len]

        model_names = "+".join(f"{item[0]}({item[2]/total_weight:.2f})" for item in available)
        logger.info("Ensemble prediction: %s, %d points", model_names, len(ensemble))
        return ensemble

    def update_weights_dynamic(
        self,
        lstm_errors: np.ndarray = None,
        xgboost_errors: np.ndarray = None,
        autogluon_errors: np.ndarray = None,
        window: int = 30,
    ):
        """
        Dynamically adjust weights based on recent RMSE performance.
        Model with lower error gets higher weight.
        Supports 2 or 3 models.
        """
        models = []

        if lstm_errors is not None and len(lstm_errors) > 0:
            recent = lstm_errors[-window:] if len(lstm_errors) >= window else lstm_errors
            models.append(("lstm", np.sqrt(np.mean(recent ** 2))))

        if xgboost_errors is not None and len(xgboost_errors) > 0:
            recent = xgboost_errors[-window:] if len(xgboost_errors) >= window else xgboost_errors
            models.append(("xgboost", np.sqrt(np.mean(recent ** 2))))

        if autogluon_errors is not None and len(autogluon_errors) > 0:
            recent = autogluon_errors[-window:] if len(autogluon_errors) >= window else autogluon_errors
            models.append(("autogluon", np.sqrt(np.mean(recent ** 2))))

        if len(models) < 2:
            return

        total_rmse = sum(rmse for _, rmse in models)
        if total_rmse == 0:
            return

        # Inverse RMSE weighting: lower error = higher weight
        raw_weights = {name: 1 - (rmse / total_rmse) for name, rmse in models}
        total_raw = sum(raw_weights.values())

        for name in raw_weights:
            raw_weights[name] /= total_raw

        self.lstm_weight = raw_weights.get("lstm", 0.0)
        self.xgboost_weight = raw_weights.get("xgboost", 0.0)
        self.autogluon_weight = raw_weights.get("autogluon", 0.0)

        log_parts = ", ".join(
            f"{name.upper()}={raw_weights[name]:.3f} (RMSE={rmse:.4f})"
            for name, rmse in models
        )
        logger.info("Dynamic weights updated: %s", log_parts)

    def get_weights(self) -> dict:
        """Return current weights."""
        return {
            "lstm": round(self.lstm_weight, 4),
            "xgboost": round(self.xgboost_weight, 4),
            "autogluon": round(self.autogluon_weight, 4),
        }
