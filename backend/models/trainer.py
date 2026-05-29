"""
Training and evaluation pipeline for ML models.
Supports walk-forward validation and comprehensive metrics.
Models: LSTM, XGBoost, AutoGluon (TabularPredictor).
"""

import numpy as np
import pandas as pd
import random
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
from datetime import datetime, timezone
import logging
from typing import Optional

from config.settings import LSTM_PARAMS, XGBOOST_PARAMS, AUTOGLUON_PARAMS, DATA_SETTINGS, STOCK_UNIVERSE, settings
from data.preprocessor import DataPreprocessor
from features.technical_indicators import TechnicalIndicators
from models.baselines import evaluate_price_baselines
from models.lstm_model import LSTMModel
from models.xgboost_model import XGBoostModel
from models.autogluon_model import AutoGluonModel
from models.ensemble import EnsembleModel

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Unified training pipeline for LSTM, XGBoost, AutoGluon, and Ensemble models."""

    def __init__(self):
        # Reproducibility baseline for all training flows.
        self.seed = 42
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.preprocessor = DataPreprocessor()
        self.lstm_model = None
        self.xgb_model = None
        self.ag_model = None
        self.ensemble_model = EnsembleModel()
        self.metrics = {}
        self.model_dir = Path(settings.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def _save_model_metadata(
        self,
        symbol: str,
        model_name: str,
        df: pd.DataFrame,
        metrics: dict,
        extra: Optional[dict] = None,
    ) -> None:
        """Persist per-run model metadata for traceability."""
        safe_symbol = symbol.replace(".", "_")
        out_path = self.model_dir / f"metadata_{model_name}_{safe_symbol}.json"

        payload = {
            "symbol": symbol,
            "model_name": model_name,
            "trained_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "seed": self.seed,
            "row_count": int(len(df)),
            "date_start": str(df.index.min()) if len(df) > 0 else None,
            "date_end": str(df.index.max()) if len(df) > 0 else None,
            "metrics": {k: v for k, v in metrics.items() if k != "training"},
        }
        if extra:
            payload["extra"] = extra

        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved model metadata to %s", out_path)

    def _get_sentiment_df(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load daily sentiment scores for a symbol from DB."""
        try:
            from data.storage import DatabaseManager
            db = DatabaseManager(settings.db_path)
            rows = db.get_daily_sentiment(symbol, days=9999)
            if rows:
                return pd.DataFrame(rows)
        except Exception as e:
            logger.warning("Could not load sentiment for %s: %s", symbol, e)
        return None

    def _get_benchmark_df(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load a stored benchmark series when one is available in the local DB."""
        candidates = []
        market = STOCK_UNIVERSE.get(symbol, {}).get("market", "")

        if symbol.endswith(".BK") or market == "SET":
            candidates = ["SET.BK", "SET50.BK", "^SET.BK", "^SET50.BK"]
        elif market in ("NASDAQ", "NYSE") or not symbol.endswith(".BK"):
            candidates = ["SPY", "QQQ", "^GSPC", "^IXIC"]

        try:
            from data.storage import DatabaseManager
            db = DatabaseManager(settings.db_path)
            for candidate in candidates:
                if candidate == symbol:
                    continue
                df = db.get_stock_prices(candidate)
                if not df.empty and "close" in df.columns:
                    logger.info("Using %s as benchmark for %s", candidate, symbol)
                    return df
        except Exception as e:
            logger.warning("Could not load benchmark for %s: %s", symbol, e)

        return None

    def evaluate_baselines(self, df: pd.DataFrame, symbol: str) -> dict:
        """Evaluate simple baseline rows and store them with model metrics."""
        try:
            baselines = evaluate_price_baselines(
                df,
                benchmark_df=self._get_benchmark_df(symbol),
            )
        except Exception as e:
            logger.warning("Baseline evaluation failed for %s: %s", symbol, e)
            return {}

        for name, metrics in baselines.items():
            self.metrics[f"{name}_{symbol}"] = metrics

        if baselines:
            logger.info("Baseline %s metrics: %s", symbol, baselines)

        return baselines

    def train_lstm(self, df: pd.DataFrame, symbol: str, sentiment_df=None) -> dict:
        """Train LSTM model on price data."""
        logger.info("Training LSTM for %s...", symbol)

        # Prepare data (with sentiment if available)
        data = self.preprocessor.prepare_lstm_data(df, target_col="close", sentiment_df=sentiment_df)

        # Build and train
        self.lstm_model = LSTMModel()
        self.lstm_model.build(input_shape=(data["X_train"].shape[1], data["X_train"].shape[2]))

        # Augment training data (noise injection for robustness)
        X_train_aug, y_train_aug = self.preprocessor.augment_training_data(
            data["X_train"], data["y_train"], noise_factor=0.005, n_augments=1
        )
        logger.info("Augmented LSTM data: %d → %d samples", len(data["X_train"]), len(X_train_aug))

        train_result = self.lstm_model.train(
            X_train_aug, y_train_aug,
            data["X_val"], data["y_val"],
        )

        # Evaluate on test set
        if len(data["X_test"]) > 0:
            y_pred_norm = self.lstm_model.predict(data["X_test"])
            y_pred = self.preprocessor.inverse_normalize_prices(y_pred_norm)
            y_true = self.preprocessor.inverse_normalize_prices(data["y_test"])

            metrics = self._calculate_metrics(y_true, y_pred)
            metrics["training"] = train_result
        else:
            metrics = {"training": train_result}

        # Save model
        model_path = str(self.model_dir / f"lstm_{symbol.replace('.', '_')}.keras")
        self.lstm_model.save(model_path)

        # Save scalers alongside model for consistent inference
        import joblib
        scaler_path = str(self.model_dir / f"lstm_scalers_{symbol.replace('.', '_')}.joblib")
        joblib.dump({
            'price_scaler': self.preprocessor.price_scaler,
            'feature_scaler': getattr(self.preprocessor, 'lstm_feature_scaler', None),
            'feature_cols': getattr(self.preprocessor, 'lstm_feature_cols', None),
        }, scaler_path)
        logger.info("Saved LSTM scalers to %s", scaler_path)

        self.metrics[f"lstm_{symbol}"] = metrics
        logger.info("LSTM %s metrics: %s", symbol, {k: v for k, v in metrics.items() if k != "training"})
        self._save_model_metadata(
            symbol=symbol,
            model_name="lstm",
            df=df,
            metrics=metrics,
            extra={
                "params": {
                    "sequence_length": LSTM_PARAMS.get("sequence_length"),
                    "epochs": LSTM_PARAMS.get("epochs"),
                    "batch_size": LSTM_PARAMS.get("batch_size"),
                    "validation_gap_days": DATA_SETTINGS.get("validation_gap_days", 0),
                },
                "feature_cols": data.get("feature_cols", []),
                "split_meta": data.get("split_meta", {}),
            },
        )
        return metrics

    def train_xgboost(self, df: pd.DataFrame, symbol: str, sentiment_df=None) -> dict:
        """Train XGBoost model on feature data."""
        logger.info("Training XGBoost for %s...", symbol)

        # Calculate technical indicators with sentiment
        df_features = TechnicalIndicators.calculate_all(df, sentiment_df=sentiment_df)

        # Prepare data
        data = self.preprocessor.prepare_xgboost_data(df_features, target_col="close")

        # Build and train
        self.xgb_model = XGBoostModel()
        self.xgb_model.build()
        train_result = self.xgb_model.train(
            data["X_train"], data["y_train"],
            data["X_val"], data["y_val"],
            feature_names=data["feature_names"],
        )

        # Evaluate on test set — convert returns to prices for meaningful metrics
        if len(data["X_test"]) > 0:
            y_pred_returns = self.xgb_model.predict(data["X_test"])
            y_true_returns = data["y_test"]
            close_test = data.get("close_test")

            if close_test is not None and len(close_test) == len(y_true_returns):
                # Convert returns to next-day prices
                y_true_prices = close_test * (1 + y_true_returns)
                y_pred_prices = close_test * (1 + y_pred_returns)
                metrics = self._calculate_metrics(y_true_prices, y_pred_prices)
            else:
                # Fallback: metrics on returns
                metrics = self._calculate_metrics(y_true_returns, y_pred_returns)
            metrics["training"] = train_result
        else:
            metrics = {"training": train_result}

        # Save model
        model_path = str(self.model_dir / f"xgb_{symbol.replace('.', '_')}.joblib")
        self.xgb_model.save(model_path)

        self.metrics[f"xgboost_{symbol}"] = metrics
        logger.info("XGBoost %s metrics: %s", symbol, {k: v for k, v in metrics.items() if k != "training"})
        self._save_model_metadata(
            symbol=symbol,
            model_name="xgboost",
            df=df,
            metrics=metrics,
            extra={
                "params": {
                    "n_estimators": XGBOOST_PARAMS.get("n_estimators"),
                    "max_depth": XGBOOST_PARAMS.get("max_depth"),
                    "learning_rate": XGBOOST_PARAMS.get("learning_rate"),
                    "validation_gap_days": DATA_SETTINGS.get("validation_gap_days", 0),
                },
                "feature_count": len(data.get("feature_names", [])),
                "split_meta": data.get("split_meta", {}),
            },
        )
        return metrics

    def train_autogluon(self, df: pd.DataFrame, symbol: str, sentiment_df=None) -> dict:
        """Train AutoGluon TabularPredictor on feature data."""
        logger.info("Training AutoGluon for %s...", symbol)

        # Calculate technical indicators (same as XGBoost)
        df_features = TechnicalIndicators.calculate_all(df, sentiment_df=sentiment_df)

        # Prepare data (reuse XGBoost pipeline — same features, same target)
        data = self.preprocessor.prepare_xgboost_data(df_features, target_col="close")

        # Build and train
        self.ag_model = AutoGluonModel()
        train_result = self.ag_model.train(
            data["X_train"], data["y_train"],
            data["X_val"], data["y_val"],
            feature_names=data["feature_names"],
        )

        # Evaluate on test set — convert returns to prices
        if len(data["X_test"]) > 0:
            y_pred_returns = self.ag_model.predict(data["X_test"])
            y_true_returns = data["y_test"]
            close_test = data.get("close_test")

            if close_test is not None and len(close_test) == len(y_true_returns):
                y_true_prices = close_test * (1 + y_true_returns)
                y_pred_prices = close_test * (1 + y_pred_returns)
                metrics = self._calculate_metrics(y_true_prices, y_pred_prices)
            else:
                metrics = self._calculate_metrics(y_true_returns, y_pred_returns)
            metrics["training"] = train_result
        else:
            metrics = {"training": train_result}

        # Save model (directory-based)
        model_path = str(self.model_dir / f"ag_{symbol.replace('.', '_')}")
        self.ag_model.save(model_path)

        self.metrics[f"autogluon_{symbol}"] = metrics
        logger.info("AutoGluon %s metrics: %s", symbol, {k: v for k, v in metrics.items() if k != "training"})
        self._save_model_metadata(
            symbol=symbol,
            model_name="autogluon",
            df=df,
            metrics=metrics,
            extra={
                "params": {
                    "time_limit": AUTOGLUON_PARAMS.get("time_limit"),
                    "preset": AUTOGLUON_PARAMS.get("preset"),
                    "validation_gap_days": DATA_SETTINGS.get("validation_gap_days", 0),
                },
                "feature_count": len(data.get("feature_names", [])),
                "split_meta": data.get("split_meta", {}),
            },
        )
        return metrics

    def train_all(self, df: pd.DataFrame, symbol: str) -> dict:
        """Train LSTM, XGBoost, and AutoGluon models, then auto-adjust ensemble weights."""
        sentiment_df = self._get_sentiment_df(symbol)
        results = self.evaluate_baselines(df, symbol)

        try:
            results["lstm"] = self.train_lstm(df, symbol, sentiment_df=sentiment_df)
        except Exception as e:
            logger.error("LSTM training failed for %s: %s", symbol, e)
            results["lstm"] = {"error": str(e)}

        try:
            results["xgboost"] = self.train_xgboost(df, symbol, sentiment_df=sentiment_df)
        except Exception as e:
            logger.error("XGBoost training failed for %s: %s", symbol, e)
            results["xgboost"] = {"error": str(e)}

        try:
            results["autogluon"] = self.train_autogluon(df, symbol, sentiment_df=sentiment_df)
        except Exception as e:
            logger.error("AutoGluon training failed for %s: %s", symbol, e)
            results["autogluon"] = {"error": str(e)}

        # Auto-adjust ensemble weights based on validation performance
        self._update_ensemble_weights(results, symbol)

        return results

    def _update_ensemble_weights(self, results: dict, symbol: str):
        """Auto-adjust ensemble weights based on model RMSE and persist them."""
        import joblib

        lstm_rmse = results.get("lstm", {}).get("rmse")
        xgb_rmse = results.get("xgboost", {}).get("rmse")
        ag_rmse = results.get("autogluon", {}).get("rmse")

        models_with_rmse = {}
        if lstm_rmse and lstm_rmse > 0:
            models_with_rmse["lstm"] = lstm_rmse
        if xgb_rmse and xgb_rmse > 0:
            models_with_rmse["xgboost"] = xgb_rmse
        if ag_rmse and ag_rmse > 0:
            models_with_rmse["autogluon"] = ag_rmse

        if len(models_with_rmse) >= 2:
            # Inverse RMSE weighting: lower error = higher weight
            total = sum(models_with_rmse.values())
            raw_weights = {name: 1 - (rmse / total) for name, rmse in models_with_rmse.items()}
            w_sum = sum(raw_weights.values())
            normalized = {name: w / w_sum for name, w in raw_weights.items()}

            self.ensemble_model.lstm_weight = normalized.get("lstm", 0.0)
            self.ensemble_model.xgboost_weight = normalized.get("xgboost", 0.0)
            self.ensemble_model.autogluon_weight = normalized.get("autogluon", 0.0)

            log_parts = ", ".join(f"{k.upper()}={v:.3f}" for k, v in normalized.items())
            rmse_parts = ", ".join(f"{k.upper()}={v:.4f}" for k, v in models_with_rmse.items())
            logger.info("Dynamic ensemble weights: %s (RMSE: %s)", log_parts, rmse_parts)
        else:
            logger.info("Using default ensemble weights (insufficient metrics)")

        # Save ensemble weights
        weights_path = str(self.model_dir / f"ensemble_weights_{symbol.replace('.', '_')}.joblib")
        joblib.dump(self.ensemble_model.get_weights(), weights_path)
        logger.info("Saved ensemble weights to %s", weights_path)

    def train_walk_forward(self, df: pd.DataFrame, symbol: str) -> dict:
        """
        Walk-forward validation: expanding window training.
        Each fold uses more data for training, tests on the next segment.
        Returns averaged metrics and keeps the model from the last (best) fold.
        """
        sentiment_df = self._get_sentiment_df(symbol)
        n_folds = DATA_SETTINGS.get("walk_forward_folds", 5)
        window_size = DATA_SETTINGS.get("walk_forward_window", 30)
        n = len(df)

        # Minimum training size = 60% of data
        min_train_size = int(n * 0.6)
        test_size = window_size
        val_size = int(test_size * 0.5)

        if n < min_train_size + val_size + test_size:
            logger.warning("Not enough data for walk-forward, falling back to standard train")
            return self.train_all(df, symbol)

        logger.info("Walk-forward validation for %s: %d folds, window=%d, total=%d rows",
                     symbol, n_folds, window_size, n)

        all_lstm_metrics = []
        all_xgb_metrics = []
        all_ag_metrics = []

        # Calculate fold boundaries (expanding window)
        # Last fold ends at data end, others step back by window_size
        fold_ends = []
        for i in range(n_folds):
            end_idx = n - (n_folds - 1 - i) * window_size
            if end_idx < min_train_size + val_size + test_size:
                continue
            fold_ends.append(end_idx)

        if not fold_ends:
            logger.warning("Could not create any valid folds, falling back to standard train")
            return self.train_all(df, symbol)

        for fold_i, end_idx in enumerate(fold_ends):
            fold_df = df.iloc[:end_idx].copy()
            logger.info("Fold %d/%d: using %d rows (up to index %d)",
                        fold_i + 1, len(fold_ends), len(fold_df), end_idx)

            # Train LSTM
            try:
                lstm_result = self.train_lstm(fold_df, symbol, sentiment_df=sentiment_df)
                if "error" not in lstm_result:
                    all_lstm_metrics.append({
                        k: v for k, v in lstm_result.items() if k != "training"
                    })
            except Exception as e:
                logger.error("Fold %d LSTM failed: %s", fold_i + 1, e)

            # Train XGBoost
            try:
                xgb_result = self.train_xgboost(fold_df, symbol, sentiment_df=sentiment_df)
                if "error" not in xgb_result:
                    all_xgb_metrics.append({
                        k: v for k, v in xgb_result.items() if k != "training"
                    })
            except Exception as e:
                logger.error("Fold %d XGBoost failed: %s", fold_i + 1, e)

            # Train AutoGluon
            try:
                ag_result = self.train_autogluon(fold_df, symbol, sentiment_df=sentiment_df)
                if "error" not in ag_result:
                    all_ag_metrics.append({
                        k: v for k, v in ag_result.items() if k != "training"
                    })
            except Exception as e:
                logger.error("Fold %d AutoGluon failed: %s", fold_i + 1, e)

        # Average metrics across folds
        def avg_metrics(metrics_list):
            if not metrics_list:
                return {}
            keys = metrics_list[0].keys()
            averaged = {}
            for k in keys:
                vals = [m[k] for m in metrics_list if k in m and isinstance(m[k], (int, float))]
                if vals:
                    averaged[k] = round(sum(vals) / len(vals), 4)
            averaged["n_folds"] = len(metrics_list)
            return averaged

        results = {
            "lstm": avg_metrics(all_lstm_metrics),
            "xgboost": avg_metrics(all_xgb_metrics),
            "autogluon": avg_metrics(all_ag_metrics),
            "walk_forward": True,
            "total_folds": len(fold_ends),
        }
        results.update(self.evaluate_baselines(df, symbol))

        # Log per-fold metrics for analysis
        if all_lstm_metrics:
            logger.info("Walk-forward LSTM avg metrics: %s", results["lstm"])
        if all_xgb_metrics:
            logger.info("Walk-forward XGB avg metrics: %s", results["xgboost"])
        if all_ag_metrics:
            logger.info("Walk-forward AG avg metrics: %s", results["autogluon"])

        # The last fold's model (trained on most data) is already saved
        logger.info("Walk-forward complete for %s. Final model from fold %d/%d",
                     symbol, len(fold_ends), len(fold_ends))
        return results

    def predict(
        self,
        df: pd.DataFrame,
        symbol: str,
        model_type: str = "ensemble",
        days: Optional[int] = None,
    ) -> dict:
        """
        Generate predictions for a symbol.
        Returns predictions, signal, and confidence.
        """
        result = {
            "symbol": symbol,
            "model": model_type,
            "predictions": [],
            "signal": "HOLD",
            "signal_reason": "",
            "confidence": 0.0,
            "model_disagreement_pct": None,
            "confidence_band": None,
            "current_price": float(df["close"].iloc[-1]) if "close" in df.columns else 0.0,
        }

        try:
            if model_type in ("lstm", "ensemble"):
                self._load_lstm(symbol)

            if model_type in ("xgboost", "ensemble"):
                self._load_xgboost(symbol)

            if model_type in ("autogluon", "ensemble"):
                self._load_autogluon(symbol)


            # Load saved ensemble weights if available
            if model_type == "ensemble":
                import joblib
                weights_path = str(self.model_dir / f"ensemble_weights_{symbol.replace('.', '_')}.joblib")
                try:
                    saved_w = joblib.load(weights_path)
                    self.ensemble_model.lstm_weight = saved_w.get("lstm", 0.3)
                    self.ensemble_model.xgboost_weight = saved_w.get("xgboost", 0.35)
                    self.ensemble_model.autogluon_weight = saved_w.get("autogluon", 0.35)
                    logger.info("Loaded ensemble weights: LSTM=%.3f, XGB=%.3f, AG=%.3f",
                               saved_w.get("lstm", 0), saved_w.get("xgboost", 0), saved_w.get("autogluon", 0))
                except FileNotFoundError:
                    logger.info("Using default ensemble weights")

            # Load sentiment data for predictions
            sentiment_df = self._get_sentiment_df(symbol)

            # Get predictions from each model
            lstm_preds = None
            xgb_preds = None
            ag_preds = None
            n_days = int(days) if days is not None else int(LSTM_PARAMS["prediction_days"])

            if model_type in ("lstm", "ensemble") and self.lstm_model and self.lstm_model.model:
                lstm_preds = self._predict_lstm(df, n_days, sentiment_df=sentiment_df)

            if model_type in ("xgboost", "ensemble") and self.xgb_model and self.xgb_model.model:
                xgb_preds = self._predict_xgboost(df, n_days, sentiment_df=sentiment_df)

            if model_type in ("autogluon", "ensemble") and self.ag_model and self.ag_model.predictor:
                ag_preds = self._predict_autogluon(df, n_days, sentiment_df=sentiment_df)

            # Log predictions
            if lstm_preds is not None:
                logger.info("LSTM raw preds: %s", [f"{p:.2f}" for p in lstm_preds])
            if xgb_preds is not None:
                logger.info("XGBoost raw preds: %s", [f"{p:.2f}" for p in xgb_preds])
            if ag_preds is not None:
                logger.info("AutoGluon raw preds: %s", [f"{p:.2f}" for p in ag_preds])

            # Combine predictions
            if model_type == "ensemble":
                final_preds = self.ensemble_model.predict(lstm_preds, xgb_preds, ag_preds)
            elif model_type == "lstm" and lstm_preds is not None:
                final_preds = lstm_preds
            elif model_type == "xgboost" and xgb_preds is not None:
                final_preds = xgb_preds
            elif model_type == "autogluon" and ag_preds is not None:
                final_preds = ag_preds
            else:
                # Fallback: use whatever is available
                if lstm_preds is not None:
                    final_preds = lstm_preds
                elif xgb_preds is not None:
                    final_preds = xgb_preds
                elif ag_preds is not None:
                    final_preds = ag_preds
                else:
                    return result

            if len(final_preds) == 0:
                return result

            # Generate signal and confidence
            current_price = result["current_price"]
            if len(final_preds) > 0 and current_price > 0:
                predicted_change = (final_preds[-1] - current_price) / current_price

                if predicted_change > 0.02:
                    result["signal"] = "BUY"
                    result["signal_reason"] = f"คาดว่าจะขึ้น {predicted_change*100:.1f}% ใน {n_days} วัน"
                elif predicted_change < -0.02:
                    result["signal"] = "SELL"
                    result["signal_reason"] = f"คาดว่าจะลง {predicted_change*100:.1f}% ใน {n_days} วัน"
                else:
                    result["signal"] = "HOLD"
                    result["signal_reason"] = f"คาดว่าจะเปลี่ยนแปลง {predicted_change*100:.1f}% (ไม่มีนัยสำคัญ)"

                # --- Confidence calculation ---
                # 1) Base confidence from model accuracy (MAPE)
                #    MAPE < 1% → 95%, MAPE 1-3% → 85-90%, MAPE 3-5% → 70-85%, MAPE > 10% → 40%
                model_key = f"{model_type}_{symbol}"
                mape = self.metrics.get(model_key, {}).get("mape", None)
                if mape is None:
                    # Try to find any recent metrics for this symbol
                    for key in [f"autogluon_{symbol}", f"xgboost_{symbol}", f"lstm_{symbol}"]:
                        m = self.metrics.get(key, {}).get("mape")
                        if m is not None:
                            mape = m
                            break
                if mape is not None and mape > 0:
                    base_confidence = max(20.0, min(95.0, 100.0 - mape * 5))
                else:
                    base_confidence = 65.0  # default if no metrics available

                # 2) Signal strength bonus: stronger moves get slight boost
                signal_bonus = min(abs(predicted_change) * 200, 10.0)  # up to +10%

                # 3) Model agreement bonus (for ensemble): if models agree on direction, boost confidence
                agreement_bonus = 0.0
                disagreement_pct = None
                if model_type == "ensemble":
                    directional_preds = []
                    if lstm_preds is not None and len(lstm_preds) > 0:
                        directional_preds.append(float(lstm_preds[-1]))
                    if xgb_preds is not None and len(xgb_preds) > 0:
                        directional_preds.append(float(xgb_preds[-1]))
                    if ag_preds is not None and len(ag_preds) > 0:
                        directional_preds.append(float(ag_preds[-1]))
                    directions = [1 if p > current_price else -1 for p in directional_preds]
                    if len(directions) >= 2:
                        # All agree = +10%, 2 out of 3 agree = +5%
                        if all(d == directions[0] for d in directions):
                            agreement_bonus = 10.0
                        elif sum(d == directions[0] for d in directions) >= 2:
                            agreement_bonus = 5.0
                    if len(directional_preds) >= 2:
                        spread_pct = (max(directional_preds) - min(directional_preds)) / max(current_price, 1e-9) * 100.0
                        disagreement_pct = round(max(0.0, spread_pct), 2)
                        agreement_bonus -= min(12.0, disagreement_pct * 0.8)

                calibrated_conf = min(95.0, max(15.0, base_confidence + signal_bonus + agreement_bonus))
                result["confidence"] = round(calibrated_conf, 1)
                result["model_disagreement_pct"] = disagreement_pct
                if calibrated_conf >= 75:
                    result["confidence_band"] = "high"
                elif calibrated_conf >= 50:
                    result["confidence_band"] = "medium"
                else:
                    result["confidence_band"] = "low"

            # Format predictions
            last_date = df.index[-1] if hasattr(df.index[-1], 'strftime') else pd.Timestamp.now()
            for i, pred in enumerate(final_preds):
                from datetime import timedelta
                target_date = last_date + timedelta(days=i + 1)
                # Skip weekends
                while target_date.weekday() >= 5:
                    target_date += timedelta(days=1)

                # Per-day confidence decreases slightly for farther predictions
                day_confidence = round(result["confidence"] * (1 - i * 0.03), 1)
                uncertainty_pct = self._estimate_uncertainty_pct(symbol, model_type, day_index=i)
                predicted_low = float(pred) * (1.0 - uncertainty_pct)
                predicted_high = float(pred) * (1.0 + uncertainty_pct)

                result["predictions"].append({
                    "date": target_date.strftime("%Y-%m-%d"),
                    "predicted_price": round(float(pred), 2),
                    "predicted_low": round(float(predicted_low), 2),
                    "predicted_high": round(float(predicted_high), 2),
                    "uncertainty_pct": round(float(uncertainty_pct) * 100.0, 2),
                    "confidence": day_confidence,
                })

        except Exception as e:
            logger.error("Prediction error for %s: %s", symbol, e)
            result["signal_reason"] = f"Error: {str(e)}"

        return result

    def _estimate_uncertainty_pct(self, symbol: str, model_type: str, day_index: int) -> float:
        """
        Estimate forecast uncertainty from available validation error.
        Uses MAPE-derived uncertainty scaled by horizon growth.
        """
        metric_keys = [f"{model_type}_{symbol}", f"autogluon_{symbol}", f"xgboost_{symbol}", f"lstm_{symbol}"]
        mape = None
        for key in metric_keys:
            val = self.metrics.get(key, {}).get("mape")
            if isinstance(val, (int, float)) and val > 0:
                mape = float(val)
                break

        base_uncertainty = (mape / 100.0) if mape is not None else 0.03
        base_uncertainty = max(0.01, min(0.20, base_uncertainty))

        horizon_scale = (day_index + 1) ** 0.5
        uncertainty = base_uncertainty * horizon_scale
        return max(0.01, min(0.35, uncertainty))

    def _predict_lstm(self, df: pd.DataFrame, n_days: int, sentiment_df=None) -> np.ndarray:
        """Generate LSTM predictions using multi-feature input.
        Always converts model output to returns and anchors to current price.
        """
        from data.preprocessor import DataPreprocessor

        # Calculate technical indicators with sentiment
        df_clean = self.preprocessor.clean(df)
        df_feat = TechnicalIndicators.calculate_all(df_clean, sentiment_df=sentiment_df)
        df_feat = df_feat.dropna()

        # Use the same feature columns as training (loaded from saved scalers)
        feature_cols = getattr(self.preprocessor, 'lstm_feature_cols', None)
        if feature_cols is None:
            feature_cols = [c for c in DataPreprocessor.LSTM_FEATURE_COLS if c in df_feat.columns]

        seq_len = LSTM_PARAMS["sequence_length"]

        if len(df_feat) < seq_len:
            return np.array([])

        # Use loaded scalers if available (saved during training)
        feat_scaler = getattr(self.preprocessor, 'lstm_feature_scaler', None)
        has_saved_scalers = feat_scaler is not None and hasattr(feat_scaler, 'data_min_') and feat_scaler.data_min_ is not None

        if not has_saved_scalers:
            # Fallback: re-fit scalers on all data
            from sklearn.preprocessing import MinMaxScaler
            feat_scaler = MinMaxScaler(feature_range=(0, 1))
            feat_scaler.fit(df_feat[feature_cols].values)
            self.preprocessor.price_scaler.fit(df_feat["close"].values.reshape(-1, 1))
            logger.warning("Using re-fitted scalers (no saved scalers found)")

        # Normalize the last seq_len rows
        last_rows = df_feat[feature_cols].iloc[-seq_len:].values
        last_rows_norm = feat_scaler.transform(last_rows)

        # Shape: (seq_len, n_features) → predict
        preds_norm = self.lstm_model.predict_future(last_rows_norm, n_days)
        raw_preds = self.preprocessor.inverse_normalize_prices(preds_norm)

        # Always anchor to current price using returns
        # This eliminates scaler mismatch — we only use the model's predicted TREND
        current_price = float(df_feat["close"].iloc[-1])
        if len(raw_preds) > 0 and raw_preds[0] != 0:
            # Extract the inter-day returns from the model's raw predictions
            all_prices = np.concatenate([[raw_preds[0]], raw_preds])
            returns = np.diff(all_prices) / all_prices[:-1]

            # Also compute initial offset as a return
            # Use the model's predicted change from its "current" reference
            last_norm_close = last_rows_norm[-1, 0]  # normalized current close
            initial_return = (preds_norm[0] - last_norm_close) / max(last_norm_close, 1e-8)
            # Clamp initial return to reasonable range
            initial_return = max(-0.05, min(0.05, initial_return))

            # Build anchored predictions
            anchored = [current_price * (1 + initial_return)]
            for r in returns:
                r = max(-0.05, min(0.05, r))  # clamp daily change
                anchored.append(anchored[-1] * (1 + r))
            raw_preds = np.array(anchored[:n_days])

            logger.info("LSTM anchored preds: %s (current=%.2f)", 
                       [f"{p:.2f}" for p in raw_preds], current_price)

        return raw_preds

    def _predict_xgboost(self, df: pd.DataFrame, n_days: int, sentiment_df=None) -> np.ndarray:
        """Generate XGBoost predictions (iterative, return-based).
        XGBoost now predicts percentage returns, not absolute prices.
        """
        df_features = TechnicalIndicators.calculate_all(df, sentiment_df=sentiment_df)
        df_features = df_features.dropna()

        # Use the exact feature names the model was trained on
        if self.xgb_model.feature_names:
            feature_cols = [
                c for c in self.xgb_model.feature_names
                if c in df_features.columns
            ]
        else:
            feature_cols = [
                c for c in df_features.select_dtypes(include=[np.number]).columns
                if c not in ("close", "open", "high", "low", "volume", "adj_close")
            ]

        current_price = float(df["close"].iloc[-1])
        predictions = []
        raw_df = df.copy()

        for _ in range(n_days):
            # Recalculate all indicators each iteration
            iter_features = TechnicalIndicators.calculate_all(raw_df, sentiment_df=sentiment_df).dropna()
            if len(iter_features) == 0:
                break

            last_row = iter_features[feature_cols].iloc[-1:].values
            predicted_return = float(self.xgb_model.predict(last_row)[0])

            # Clamp return to reasonable range (-10% to +10% per day)
            predicted_return = max(-0.10, min(0.10, predicted_return))

            # Apply predicted return to running price
            anchor = predictions[-1] if predictions else current_price
            next_price = anchor * (1 + predicted_return)
            predictions.append(next_price)

            # Append predicted price as a new row in raw data
            last_date = raw_df.index[-1] if isinstance(raw_df.index, pd.DatetimeIndex) else pd.Timestamp('now')
            new_row = raw_df.iloc[-1:].copy()
            try:
                new_row.index = [last_date + pd.Timedelta(days=1)]
            except Exception:
                new_row.index = new_row.index + pd.Timedelta(days=1)
            new_row["close"] = next_price
            new_row["open"] = next_price
            new_row["high"] = next_price
            new_row["low"] = next_price
            raw_df = pd.concat([raw_df, new_row])

        return np.array(predictions)

    def _predict_autogluon(self, df: pd.DataFrame, n_days: int, sentiment_df=None) -> np.ndarray:
        """Generate AutoGluon predictions (iterative, return-based).
        Same approach as XGBoost: predict return → apply to price → recalculate → repeat.
        """
        df_features = TechnicalIndicators.calculate_all(df, sentiment_df=sentiment_df)
        df_features = df_features.dropna()

        # Use the exact feature names the model was trained on
        if self.ag_model.feature_names:
            feature_cols = [
                c for c in self.ag_model.feature_names
                if c in df_features.columns
            ]
        else:
            feature_cols = [
                c for c in df_features.select_dtypes(include=[np.number]).columns
                if c not in ("close", "open", "high", "low", "volume", "adj_close")
            ]

        current_price = float(df["close"].iloc[-1])
        predictions = []
        raw_df = df.copy()

        for _ in range(n_days):
            # Recalculate all indicators each iteration
            iter_features = TechnicalIndicators.calculate_all(raw_df, sentiment_df=sentiment_df).dropna()
            if len(iter_features) == 0:
                break

            last_row = iter_features[feature_cols].iloc[-1:].values
            predicted_return = float(self.ag_model.predict(last_row)[0])

            # Clamp return to reasonable range (-10% to +10% per day)
            predicted_return = max(-0.10, min(0.10, predicted_return))

            # Apply predicted return to running price
            anchor = predictions[-1] if predictions else current_price
            next_price = anchor * (1 + predicted_return)
            predictions.append(next_price)

            # Append predicted price as a new row in raw data
            last_date = raw_df.index[-1] if isinstance(raw_df.index, pd.DatetimeIndex) else pd.Timestamp('now')
            new_row = raw_df.iloc[-1:].copy()
            try:
                new_row.index = [last_date + pd.Timedelta(days=1)]
            except Exception:
                new_row.index = new_row.index + pd.Timedelta(days=1)
            new_row["close"] = next_price
            new_row["open"] = next_price
            new_row["high"] = next_price
            new_row["low"] = next_price
            raw_df = pd.concat([raw_df, new_row])

        return np.array(predictions)

    def _load_lstm(self, symbol: str):
        """Load LSTM model and scalers for a symbol."""
        model_path = str(self.model_dir / f"lstm_{symbol.replace('.', '_')}.keras")
        self.lstm_model = LSTMModel()
        self.lstm_model.load(model_path)

        # Load saved scalers if available
        import joblib
        scaler_path = str(self.model_dir / f"lstm_scalers_{symbol.replace('.', '_')}.joblib")
        try:
            scalers = joblib.load(scaler_path)
            self.preprocessor.price_scaler = scalers['price_scaler']
            self.preprocessor.lstm_feature_scaler = scalers.get('feature_scaler')
            self.preprocessor.lstm_feature_cols = scalers.get('feature_cols')
            logger.info("Loaded LSTM scalers from %s", scaler_path)
        except FileNotFoundError:
            logger.warning("No saved scalers found for %s, will re-fit during inference", symbol)

    def _load_xgboost(self, symbol: str):
        """Load XGBoost model for a symbol."""
        model_path = str(self.model_dir / f"xgb_{symbol.replace('.', '_')}.joblib")
        self.xgb_model = XGBoostModel()
        self.xgb_model.load(model_path)

    def _load_autogluon(self, symbol: str):
        """Load AutoGluon model for a symbol. Skips silently if autogluon is not installed."""
        try:
            model_path = str(self.model_dir / f"ag_{symbol.replace('.', '_')}")
            self.ag_model = AutoGluonModel()
            self.ag_model.load(model_path)
        except ImportError:
            logger.warning("AutoGluon package not installed — skipping AG model for %s", symbol)
            self.ag_model = None
        except Exception as e:
            logger.warning("Could not load AutoGluon model for %s: %s", symbol, e)
            self.ag_model = None

    @staticmethod
    def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calculate evaluation metrics."""
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))

        # MAPE (avoid division by zero)
        mask = y_true != 0
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

        # Directional Accuracy
        if len(y_true) > 1:
            true_dir = np.diff(y_true) > 0
            pred_dir = np.diff(y_pred) > 0
            da = float(np.mean(true_dir == pred_dir) * 100)
        else:
            da = 0.0

        r2 = float(r2_score(y_true, y_pred))

        return {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "mape": round(mape, 2),
            "directional_accuracy": round(da, 2),
            "r_squared": round(r2, 4),
        }

    def get_all_metrics(self) -> dict:
        """Return all stored metrics."""
        return self.metrics
