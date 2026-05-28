"""
XGBoost model for structured stock price prediction.
Uses technical indicators as features.
"""

import numpy as np
import os
import logging
from pathlib import Path
from typing import Optional
import joblib

logger = logging.getLogger(__name__)


class XGBoostModel:
    """XGBoost regressor for stock price prediction."""

    def __init__(self, params: dict = None):
        from config.settings import XGBOOST_PARAMS
        self.params = params or XGBOOST_PARAMS
        self.model = None
        self.feature_names = None
        self.feature_importances = None

    def build(self):
        """Build XGBoost model."""
        from xgboost import XGBRegressor

        self.model = XGBRegressor(
            n_estimators=self.params["n_estimators"],
            max_depth=self.params["max_depth"],
            learning_rate=self.params["learning_rate"],
            subsample=self.params["subsample"],
            colsample_bytree=self.params["colsample_bytree"],
            reg_alpha=self.params["reg_alpha"],
            reg_lambda=self.params["reg_lambda"],
            objective=self.params["objective"],
            random_state=42,
            n_jobs=-1,
        )
        logger.info("XGBoost model built")
        return self

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        feature_names: list[str] = None,
    ) -> dict:
        """Train the model."""
        if self.model is None:
            self.build()

        self.feature_names = feature_names

        eval_set = [(X_train, y_train)]
        if X_val is not None and len(X_val) > 0:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=50,
        )

        # Feature importance
        if feature_names:
            importances = self.model.feature_importances_
            self.feature_importances = dict(zip(feature_names, importances.tolist()))

        # Training metrics
        result = {
            "n_estimators": self.model.n_estimators,
            "best_iteration": getattr(self.model, 'best_iteration', self.params["n_estimators"]),
        }

        logger.info("XGBoost training complete: %s", result)
        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not built/loaded. Train or load first.")
        return self.model.predict(X)

    def get_feature_importance(self, top_n: int = 15) -> list[dict]:
        """Get top N most important features."""
        if self.feature_importances is None:
            return []

        sorted_features = sorted(
            self.feature_importances.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return [
            {"feature": name, "importance": float(imp)}
            for name, imp in sorted_features[:top_n]
        ]

    def save(self, path: str):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        save_data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "feature_importances": self.feature_importances,
        }
        joblib.dump(save_data, path)
        logger.info("XGBoost model saved to %s", path)

    def load(self, path: str):
        """Load model from disk."""
        if os.path.exists(path):
            save_data = joblib.load(path)
            self.model = save_data["model"]
            self.feature_names = save_data.get("feature_names")
            self.feature_importances = save_data.get("feature_importances")
            logger.info("XGBoost model loaded from %s", path)
        else:
            logger.warning("Model file not found: %s", path)
