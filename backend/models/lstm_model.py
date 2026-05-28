"""
LSTM model for time-series stock price prediction.
Stacked LSTM with dropout for overfitting prevention.
"""

import numpy as np
import os
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Suppress TF warnings (set before TF import so it takes effect)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf


class LSTMModel:
    """Stacked LSTM model for stock price prediction."""

    def __init__(self, params: dict = None):
        from config.settings import LSTM_PARAMS
        self.params = params or LSTM_PARAMS
        self.model = None
        self.history = None

    def build(self, input_shape: Tuple[int, int] = None):
        """Build the LSTM architecture with Attention and BatchNorm."""
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import (
            LSTM, Dense, Dropout, Input, BatchNormalization,
            Layer, Multiply, Permute, Reshape, Lambda
        )

        if input_shape is None:
            input_shape = (self.params["sequence_length"], 1)

        # Attention layer implementation
        class AttentionLayer(Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

            def build(self, input_shape):
                self.W = self.add_weight(name="att_weight",
                    shape=(input_shape[-1], input_shape[-1]),
                    initializer="glorot_uniform", trainable=True)
                self.b = self.add_weight(name="att_bias",
                    shape=(input_shape[-1],),
                    initializer="zeros", trainable=True)
                super().build(input_shape)

            def call(self, x):
                # x shape: (batch, timesteps, features)
                e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
                a = tf.nn.softmax(e, axis=1)  # attention weights over timesteps
                output = x * a
                return tf.reduce_sum(output, axis=1)  # weighted sum

        inputs = Input(shape=input_shape)

        # LSTM Block 1
        x = LSTM(self.params["lstm_units_1"], return_sequences=True)(inputs)
        x = BatchNormalization()(x)
        x = Dropout(self.params["dropout"])(x)

        # LSTM Block 2
        x = LSTM(self.params["lstm_units_2"], return_sequences=True)(x)
        x = BatchNormalization()(x)
        x = Dropout(self.params["dropout"])(x)

        # Attention — focus on important time steps
        x = AttentionLayer(name="attention")(x)

        # Dense head
        x = Dense(self.params["dense_units"], activation="relu")(x)
        outputs = Dense(1)(x)

        self.model = Model(inputs=inputs, outputs=outputs)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.params["learning_rate"]
            ),
            loss="huber",
            metrics=["mae"],
        )

        logger.info("LSTM+Attention model built: %s", self.model.summary(print_fn=lambda x: None))
        return self

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ) -> dict:
        """Train the model."""
        if self.model is None:
            self.build(input_shape=(X_train.shape[1], X_train.shape[2]))

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss" if X_val is not None else "loss",
                patience=self.params["early_stopping_patience"],
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss" if X_val is not None else "loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
            ),
        ]

        validation_data = (X_val, y_val) if X_val is not None and len(X_val) > 0 else None

        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.params["epochs"],
            batch_size=self.params["batch_size"],
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1,
        )

        result = {
            "epochs_trained": len(self.history.history["loss"]),
            "final_loss": float(self.history.history["loss"][-1]),
            "final_mae": float(self.history.history["mae"][-1]),
            "loss_history": [float(v) for v in self.history.history["loss"]],
            "mae_history": [float(v) for v in self.history.history["mae"]],
        }

        if validation_data:
            result["val_loss"] = float(self.history.history["val_loss"][-1])
            result["val_mae"] = float(self.history.history["val_mae"][-1])
            result["val_loss_history"] = [float(v) for v in self.history.history["val_loss"]]

        logger.info("LSTM training complete: epochs=%d, loss=%.6f",
                     result["epochs_trained"], result["final_loss"])
        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not built/loaded. Train or load first.")
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()

    def predict_future(self, last_sequence: np.ndarray, n_days: int = 5) -> np.ndarray:
        """
        Predict N days into the future.
        Uses iterative single-step prediction.
        last_sequence: shape (seq_length, n_features)
        """
        if self.model is None:
            raise ValueError("Model not built/loaded.")

        predictions = []
        current_seq = last_sequence.copy()

        for _ in range(n_days):
            x = current_seq.reshape(1, current_seq.shape[0], current_seq.shape[1])
            pred = self.model.predict(x, verbose=0)
            # Handle both single-output and multi-output models
            if pred.ndim > 1 and pred.shape[-1] > 1:
                pred_val = float(pred[0, 0])  # take first day from multi-step
            else:
                pred_val = float(pred.flatten()[0])
            predictions.append(pred_val)

            # Slide window
            new_row = current_seq[-1:].copy()
            new_row[0, 0] = pred_val
            current_seq = np.vstack([current_seq[1:], new_row])

        return np.array(predictions)

    def save(self, path: str):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        logger.info("LSTM model saved to %s", path)

    def load(self, path: str):
        """Load model from disk (handles custom AttentionLayer)."""
        if os.path.exists(path):
            try:
                self.model = tf.keras.models.load_model(path)
            except Exception:
                # Fallback: register custom objects for AttentionLayer
                from tensorflow.keras.layers import Layer

                class AttentionLayer(Layer):
                    def __init__(self, **kwargs):
                        super().__init__(**kwargs)

                    def build(self, input_shape):
                        self.W = self.add_weight(name="att_weight",
                            shape=(input_shape[-1], input_shape[-1]),
                            initializer="glorot_uniform", trainable=True)
                        self.b = self.add_weight(name="att_bias",
                            shape=(input_shape[-1],),
                            initializer="zeros", trainable=True)
                        super().build(input_shape)

                    def call(self, x):
                        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
                        a = tf.nn.softmax(e, axis=1)
                        output = x * a
                        return tf.reduce_sum(output, axis=1)

                self.model = tf.keras.models.load_model(
                    path, custom_objects={"AttentionLayer": AttentionLayer}
                )
            logger.info("LSTM model loaded from %s", path)
        else:
            logger.warning("Model file not found: %s", path)

    def get_training_history(self) -> Optional[dict]:
        """Return training history for plotting."""
        if self.history is None:
            return None
        return {k: [float(v) for v in vals] for k, vals in self.history.history.items()}
