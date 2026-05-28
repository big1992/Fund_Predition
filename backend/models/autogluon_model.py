"""
AutoGluon TabularPredictor for stock return prediction.
Uses AutoML to automatically select and ensemble the best models
from LightGBM, CatBoost, XGBoost, RandomForest, Neural Net, etc.
"""

import numpy as np
import os
import shutil
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class AutoGluonModel:
    """AutoGluon TabularPredictor for stock return prediction."""

    def __init__(self, params: dict = None):
        from config.settings import AUTOGLUON_PARAMS
        self.params = params or AUTOGLUON_PARAMS
        self.predictor = None
        self.feature_names = None
        self.leaderboard_df = None

    def build(self):
        """No-op: AutoGluon builds during fit."""
        logger.info("AutoGluon model ready (builds during training)")
        return self

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        feature_names: list[str] = None,
    ) -> dict:
        """Train AutoGluon TabularPredictor."""
        import pandas as pd
        from autogluon.tabular import TabularPredictor

        self.feature_names = feature_names or [f"f_{i}" for i in range(X_train.shape[1])]
        target_col = "__target__"

        # Build training DataFrame
        train_df = pd.DataFrame(X_train, columns=self.feature_names)
        train_df[target_col] = y_train

        # Build validation (tuning) DataFrame
        tuning_df = None
        if X_val is not None and len(X_val) > 0:
            tuning_df = pd.DataFrame(X_val, columns=self.feature_names)
            tuning_df[target_col] = y_val

        # Temporary save path (AutoGluon needs a directory during fit)
        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix="ag_train_")

        try:
            self.predictor = TabularPredictor(
                label=target_col,
                eval_metric=self.params.get("eval_metric", "root_mean_squared_error"),
                path=tmp_dir,
                problem_type="regression",
                verbosity=1,
            )

            fit_kwargs = {
                "train_data": train_df,
                "time_limit": self.params.get("time_limit", 120),
                "presets": self.params.get("preset", "medium_quality"),
            }

            if tuning_df is not None:
                fit_kwargs["tuning_data"] = tuning_df

            excluded = self.params.get("excluded_model_types", [])
            if excluded:
                fit_kwargs["excluded_model_types"] = excluded

            self.predictor.fit(**fit_kwargs)

            # Get leaderboard
            self.leaderboard_df = self.predictor.leaderboard(silent=True)

            # Training result summary
            best_model = self.predictor.model_best
            best_score = float(self.predictor.info()["best_model_score_val"])

            result = {
                "best_model": best_model,
                "best_score": best_score,
                "models_trained": len(self.leaderboard_df),
                "leaderboard": self.leaderboard_df[["model", "score_val", "fit_time", "pred_time_val"]].to_dict("records")
                    if self.leaderboard_df is not None else [],
            }

            logger.info(
                "AutoGluon training complete: best=%s, score=%.6f, models=%d",
                best_model, best_score, len(self.leaderboard_df),
            )
            return result

        except Exception as e:
            logger.error("AutoGluon training failed: %s", e)
            raise
        finally:
            # tmp_dir cleanup is handled by save/load — keep it for now
            pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        import pandas as pd

        if self.predictor is None:
            raise ValueError("Model not built/loaded. Train or load first.")

        feature_names = self.feature_names or [f"f_{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        preds = self.predictor.predict(df)
        return preds.values

    def get_feature_importance(self, top_n: int = 15) -> list[dict]:
        """Get top N most important features."""
        if self.predictor is None:
            return []

        try:
            importance = self.predictor.feature_importance(silent=True)
            if importance is None or len(importance) == 0:
                return []

            # AutoGluon returns importance as a DataFrame with 'importance' column
            sorted_imp = importance.sort_values("importance", ascending=False).head(top_n)
            return [
                {"feature": str(name), "importance": float(imp)}
                for name, imp in zip(sorted_imp.index, sorted_imp["importance"])
            ]
        except Exception as e:
            logger.warning("Could not get AG feature importance: %s", e)
            return []

    def get_leaderboard(self) -> list[dict]:
        """Return AutoGluon leaderboard as list of dicts."""
        if self.leaderboard_df is None:
            return []
        try:
            cols = [c for c in ["model", "score_val", "fit_time", "pred_time_val", "stack_level"]
                    if c in self.leaderboard_df.columns]
            return self.leaderboard_df[cols].to_dict("records")
        except Exception:
            return []

    def save(self, path: str):
        """Save model to disk (directory-based)."""
        if self.predictor is None:
            raise ValueError("No model to save")

        save_dir = Path(path)
        # If the target directory exists, remove it first to avoid conflicts
        if save_dir.exists():
            shutil.rmtree(save_dir)

        # Copy the entire predictor directory (tmp_dir) to save_dir
        # This is more reliable than predictor.save(path) which may not copy all files
        src_path = Path(self.predictor.path)
        if src_path.exists():
            shutil.copytree(str(src_path), str(save_dir))
        else:
            # Fallback: use predictor.save()
            self.predictor.save(str(save_dir))
            save_dir.mkdir(parents=True, exist_ok=True)

        # Also save feature names alongside
        import joblib
        meta_path = save_dir / "_ag_meta.joblib"
        joblib.dump({
            "feature_names": self.feature_names,
            "leaderboard": self.get_leaderboard(),
        }, str(meta_path))

        logger.info("AutoGluon model saved to %s", save_dir)

    def load(self, path: str):
        """Load model from disk. Silently skips if autogluon is not installed."""
        try:
            from autogluon.tabular import TabularPredictor
        except ImportError:
            logger.debug(
                "autogluon package is not installed — AutoGluon model will not be loaded. "
                "Install with: pip install autogluon.tabular"
            )
            return  # predictor stays None; ensemble will skip AG

        save_dir = Path(path)
        if save_dir.exists() and save_dir.is_dir():
            self.predictor = TabularPredictor.load(str(save_dir))

            # Load metadata
            import joblib
            meta_path = save_dir / "_ag_meta.joblib"
            if meta_path.exists():
                meta = joblib.load(str(meta_path))
                self.feature_names = meta.get("feature_names")
                # Reconstruct leaderboard_df from saved data
                lb_data = meta.get("leaderboard", [])
                if lb_data:
                    import pandas as pd
                    self.leaderboard_df = pd.DataFrame(lb_data)

            logger.info("AutoGluon model loaded from %s", save_dir)
        else:
            logger.warning("AutoGluon model directory not found: %s", save_dir)

