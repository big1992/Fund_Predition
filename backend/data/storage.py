"""
SQLite storage layer for stock prices, fund NAV, predictions, and indicators.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from datetime import date, datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    """SQLite database manager with schema auto-creation."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        """Create tables if they don't exist."""
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS stock_prices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date DATE NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    adj_close REAL,
                    volume INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, date)
                );

                CREATE TABLE IF NOT EXISTS fund_nav (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fund_name TEXT NOT NULL,
                    date DATE NOT NULL,
                    nav REAL,
                    daily_return REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(fund_name, date)
                );

                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    prediction_date DATE NOT NULL,
                    target_date DATE NOT NULL,
                    predicted_price REAL,
                    actual_price REAL,
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS technical_indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date DATE NOT NULL,
                    sma_20 REAL, sma_50 REAL, sma_200 REAL,
                    ema_12 REAL, ema_26 REAL,
                    rsi_14 REAL,
                    macd REAL, macd_signal REAL, macd_histogram REAL,
                    bb_upper REAL, bb_middle REAL, bb_lower REAL,
                    atr_14 REAL, obv REAL, adx REAL,
                    stoch_k REAL, stoch_d REAL,
                    williams_r REAL,
                    daily_return REAL, log_return REAL,
                    volume_sma_20 REAL, volume_ratio REAL,
                    vwap REAL,
                    roc_10 REAL,
                    hist_volatility_20 REAL,
                    price_sma50_ratio REAL,
                    UNIQUE(symbol, date)
                );

                CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol_date
                    ON stock_prices(symbol, date);
                CREATE INDEX IF NOT EXISTS idx_fund_nav_name_date
                    ON fund_nav(fund_name, date);
                CREATE INDEX IF NOT EXISTS idx_predictions_symbol
                    ON predictions(symbol, model_name);
                CREATE INDEX IF NOT EXISTS idx_indicators_symbol_date
                    ON technical_indicators(symbol, date);
            """)
            conn.commit()
            logger.info("Database initialized at %s", self.db_path)
        finally:
            conn.close()

    # ========== Stock Prices ==========

    def store_stock_prices(self, symbol: str, df: pd.DataFrame) -> int:
        """Store OHLCV data. Returns number of rows inserted."""
        conn = self._get_conn()
        count = 0
        try:
            for _, row in df.iterrows():
                try:
                    conn.execute(
                        """INSERT OR REPLACE INTO stock_prices
                           (symbol, date, open, high, low, close, adj_close, volume)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            symbol,
                            row.name.strftime("%Y-%m-%d") if hasattr(row.name, 'strftime') else str(row.name),
                            float(row.get("Open", 0)),
                            float(row.get("High", 0)),
                            float(row.get("Low", 0)),
                            float(row.get("Close", 0)),
                            float(row.get("Adj Close", row.get("Close", 0))),
                            int(row.get("Volume", 0)),
                        ),
                    )
                    count += 1
                except Exception as e:
                    logger.warning("Error storing row for %s: %s", symbol, e)
            conn.commit()
        finally:
            conn.close()
        return count

    def get_stock_prices(
        self, symbol: str, start: Optional[str] = None, end: Optional[str] = None
    ) -> pd.DataFrame:
        """Get stock prices as DataFrame."""
        conn = self._get_conn()
        try:
            query = "SELECT date, open, high, low, close, adj_close, volume FROM stock_prices WHERE symbol = ?"
            params = [symbol]
            if start:
                query += " AND date >= ?"
                params.append(start)
            if end:
                query += " AND date <= ?"
                params.append(end)
            query += " ORDER BY date"
            df = pd.read_sql_query(query, conn, params=params, parse_dates=["date"])
            if not df.empty:
                df.set_index("date", inplace=True)
            return df
        finally:
            conn.close()

    def get_available_symbols(self) -> list[str]:
        """Get list of symbols with data in DB."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT DISTINCT symbol FROM stock_prices ORDER BY symbol"
            ).fetchall()
            return [r["symbol"] for r in rows]
        finally:
            conn.close()

    def get_latest_date(self, symbol: str) -> Optional[str]:
        """Get the latest date for a symbol."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT MAX(date) as max_date FROM stock_prices WHERE symbol = ?",
                (symbol,),
            ).fetchone()
            return row["max_date"] if row else None
        finally:
            conn.close()

    # ========== Fund NAV ==========

    def store_fund_nav(self, fund_name: str, df: pd.DataFrame) -> int:
        """Store fund NAV data."""
        conn = self._get_conn()
        count = 0
        try:
            for _, row in df.iterrows():
                try:
                    conn.execute(
                        """INSERT OR REPLACE INTO fund_nav
                           (fund_name, date, nav, daily_return)
                           VALUES (?, ?, ?, ?)""",
                        (
                            fund_name,
                            row["date"] if isinstance(row["date"], str) else row["date"].strftime("%Y-%m-%d"),
                            float(row["nav"]),
                            float(row.get("daily_return", 0)),
                        ),
                    )
                    count += 1
                except Exception as e:
                    logger.warning("Error storing NAV for %s: %s", fund_name, e)
            conn.commit()
        finally:
            conn.close()
        return count

    def get_fund_nav(
        self, fund_name: str, start: Optional[str] = None, end: Optional[str] = None
    ) -> pd.DataFrame:
        """Get fund NAV history."""
        conn = self._get_conn()
        try:
            query = "SELECT date, nav, daily_return FROM fund_nav WHERE fund_name = ?"
            params = [fund_name]
            if start:
                query += " AND date >= ?"
                params.append(start)
            if end:
                query += " AND date <= ?"
                params.append(end)
            query += " ORDER BY date"
            df = pd.read_sql_query(query, conn, params=params, parse_dates=["date"])
            return df
        finally:
            conn.close()

    # ========== Technical Indicators ==========

    def store_indicators(self, symbol: str, df: pd.DataFrame) -> int:
        """Store technical indicators."""
        conn = self._get_conn()
        count = 0
        try:
            indicator_cols = [
                "sma_20", "sma_50", "sma_200", "ema_12", "ema_26",
                "rsi_14", "macd", "macd_signal", "macd_histogram",
                "bb_upper", "bb_middle", "bb_lower", "atr_14", "obv", "adx",
                "stoch_k", "stoch_d", "williams_r",
                "daily_return", "log_return",
                "volume_sma_20", "volume_ratio", "vwap", "roc_10",
                "hist_volatility_20", "price_sma50_ratio",
            ]
            for idx, row in df.iterrows():
                try:
                    date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, 'strftime') else str(idx)
                    values = [symbol, date_str]
                    for col in indicator_cols:
                        val = row.get(col)
                        values.append(float(val) if pd.notna(val) else None)

                    placeholders = ", ".join(["?"] * (2 + len(indicator_cols)))
                    cols = ", ".join(["symbol", "date"] + indicator_cols)
                    conn.execute(
                        f"INSERT OR REPLACE INTO technical_indicators ({cols}) VALUES ({placeholders})",
                        values,
                    )
                    count += 1
                except Exception as e:
                    logger.warning("Error storing indicators for %s: %s", symbol, e)
            conn.commit()
        finally:
            conn.close()
        return count

    def get_indicators(
        self, symbol: str, start: Optional[str] = None, end: Optional[str] = None
    ) -> pd.DataFrame:
        """Get technical indicators."""
        conn = self._get_conn()
        try:
            query = "SELECT * FROM technical_indicators WHERE symbol = ?"
            params = [symbol]
            if start:
                query += " AND date >= ?"
                params.append(start)
            if end:
                query += " AND date <= ?"
                params.append(end)
            query += " ORDER BY date"
            df = pd.read_sql_query(query, conn, params=params, parse_dates=["date"])
            if not df.empty and "date" in df.columns:
                df.set_index("date", inplace=True)
            return df
        finally:
            conn.close()

    # ========== Predictions ==========

    def store_prediction(
        self, symbol: str, model_name: str, prediction_date: str,
        target_date: str, predicted_price: float, confidence: float,
        actual_price: Optional[float] = None,
    ):
        """Store a single prediction."""
        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT INTO predictions
                   (symbol, model_name, prediction_date, target_date,
                    predicted_price, actual_price, confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (symbol, model_name, prediction_date, target_date,
                 predicted_price, actual_price, confidence),
            )
            conn.commit()
        finally:
            conn.close()

    def get_data_stats(self) -> dict:
        """Get database statistics."""
        conn = self._get_conn()
        try:
            stock_count = conn.execute("SELECT COUNT(DISTINCT symbol) FROM stock_prices").fetchone()[0]
            total_prices = conn.execute("SELECT COUNT(*) FROM stock_prices").fetchone()[0]
            fund_count = conn.execute("SELECT COUNT(DISTINCT fund_name) FROM fund_nav").fetchone()[0]
            return {
                "stock_symbols": stock_count,
                "total_price_records": total_prices,
                "fund_count": fund_count,
            }
        finally:
            conn.close()

    # ========== Training History ==========

    def _ensure_training_history_table(self):
        """Create training_history table if not exists."""
        conn = self._get_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    lstm_rmse REAL, lstm_mae REAL, lstm_mape REAL,
                    lstm_directional_accuracy REAL, lstm_r_squared REAL,
                    xgb_rmse REAL, xgb_mae REAL, xgb_mape REAL,
                    xgb_directional_accuracy REAL, xgb_r_squared REAL,
                    params_json TEXT,
                    metrics_json TEXT,
                    satisfaction_score INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def save_training_run(self, symbol: str, model_type: str,
                          metrics: dict, params: dict,
                          satisfaction_score: int = 0) -> int:
        """Save a training run. Returns the inserted row ID."""
        import json
        self._ensure_training_history_table()

        lstm = metrics.get("lstm", {})
        xgb = metrics.get("xgboost", {})

        conn = self._get_conn()
        try:
            cur = conn.execute("""
                INSERT INTO training_history
                (symbol, model_type,
                 lstm_rmse, lstm_mae, lstm_mape, lstm_directional_accuracy, lstm_r_squared,
                 xgb_rmse, xgb_mae, xgb_mape, xgb_directional_accuracy, xgb_r_squared,
                 params_json, metrics_json, satisfaction_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol, model_type,
                lstm.get("rmse"), lstm.get("mae"), lstm.get("mape"),
                lstm.get("directional_accuracy"), lstm.get("r_squared"),
                xgb.get("rmse"), xgb.get("mae"), xgb.get("mape"),
                xgb.get("directional_accuracy"), xgb.get("r_squared"),
                json.dumps(params, default=str),
                json.dumps(metrics, default=str),
                satisfaction_score,
            ))
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def get_training_history(self, symbol: str, limit: int = 10) -> list[dict]:
        """Get recent training history for a symbol."""
        import json
        self._ensure_training_history_table()

        conn = self._get_conn()
        try:
            rows = conn.execute("""
                SELECT * FROM training_history
                WHERE symbol = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (symbol, limit)).fetchall()

            history = []
            for r in rows:
                entry = dict(r)
                try:
                    entry["params"] = json.loads(entry.pop("params_json", "{}"))
                except Exception:
                    entry["params"] = {}
                try:
                    entry["metrics_detail"] = json.loads(entry.pop("metrics_json", "{}"))
                except Exception:
                    entry["metrics_detail"] = {}
                history.append(entry)
            return history
        finally:
            conn.close()

    def get_best_training_run(self, symbol: str) -> Optional[dict]:
        """Get the best training run based on composite score (DirAcc 60% + MAPE 40%)."""
        import json
        self._ensure_training_history_table()

        conn = self._get_conn()
        try:
            # Composite: lower = better
            # score = 0.6 * (100 - avg_dir_acc) + 0.4 * avg_mape
            row = conn.execute("""
                SELECT *,
                    (
                        0.6 * (100.0 - COALESCE(
                            CASE
                                WHEN lstm_directional_accuracy IS NOT NULL AND xgb_directional_accuracy IS NOT NULL
                                    THEN (lstm_directional_accuracy + xgb_directional_accuracy) / 2.0
                                WHEN lstm_directional_accuracy IS NOT NULL THEN lstm_directional_accuracy
                                ELSE COALESCE(xgb_directional_accuracy, 50.0)
                            END, 50.0))
                        +
                        0.4 * COALESCE(
                            CASE
                                WHEN lstm_mape IS NOT NULL AND xgb_mape IS NOT NULL
                                    THEN (lstm_mape + xgb_mape) / 2.0
                                WHEN lstm_mape IS NOT NULL THEN lstm_mape
                                ELSE xgb_mape
                            END, 999)
                    ) AS composite_score
                FROM training_history
                WHERE symbol = ?
                  AND (lstm_mape IS NOT NULL OR xgb_mape IS NOT NULL)
                ORDER BY composite_score ASC
                LIMIT 1
            """, (symbol,)).fetchone()

            if not row:
                return None

            entry = dict(row)
            try:
                entry["params"] = json.loads(entry.pop("params_json", "{}"))
            except Exception:
                entry["params"] = {}
            try:
                entry["metrics_detail"] = json.loads(entry.pop("metrics_json", "{}"))
            except Exception:
                entry["metrics_detail"] = {}
            return entry
        finally:
            conn.close()

    # ========== Model Registry ==========

    def _ensure_model_registry_table(self):
        conn = self._get_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_registry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    training_run_id INTEGER,
                    train_start_date TEXT,
                    train_end_date TEXT,
                    feature_version TEXT,
                    artifact_path TEXT,
                    params_json TEXT,
                    metrics_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, model_name, version)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_registry_symbol_model
                    ON model_registry(symbol, model_name)
            """)
            conn.commit()
        finally:
            conn.close()

    def save_model_registry_entries(
        self,
        *,
        symbol: str,
        metrics_by_model: dict,
        params: dict,
        training_run_id: Optional[int] = None,
        train_start_date: Optional[str] = None,
        train_end_date: Optional[str] = None,
        feature_version: str = "v1",
        artifact_paths: Optional[dict] = None,
    ) -> int:
        import json
        self._ensure_model_registry_table()
        artifact_paths = artifact_paths or {}
        conn = self._get_conn()
        inserted = 0
        try:
            for model_name, model_metrics in metrics_by_model.items():
                if not isinstance(model_metrics, dict):
                    continue
                if model_metrics.get("error"):
                    continue
                row = conn.execute(
                    "SELECT COALESCE(MAX(version), 0) AS v FROM model_registry WHERE symbol = ? AND model_name = ?",
                    (symbol, model_name),
                ).fetchone()
                next_version = int((dict(row).get("v") if row else 0) or 0) + 1
                conn.execute(
                    """
                    INSERT INTO model_registry
                    (symbol, model_name, version, training_run_id, train_start_date, train_end_date,
                     feature_version, artifact_path, params_json, metrics_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        symbol,
                        model_name,
                        next_version,
                        training_run_id,
                        train_start_date,
                        train_end_date,
                        feature_version,
                        artifact_paths.get(model_name),
                        json.dumps(params, default=str),
                        json.dumps(model_metrics, default=str),
                    ),
                )
                inserted += 1
            conn.commit()
            return inserted
        finally:
            conn.close()

    def get_model_registry(self, symbol: str, limit: int = 100) -> list[dict]:
        import json
        self._ensure_model_registry_table()
        self._ensure_training_history_table()
        best = self.get_best_training_run(symbol)
        best_run_id = best.get("id") if best else None

        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT mr.*
                FROM model_registry mr
                WHERE mr.symbol = ?
                ORDER BY mr.created_at DESC, mr.id DESC
                LIMIT ?
                """,
                (symbol, limit),
            ).fetchall()

            latest_by_model = {}
            for r in rows:
                entry = dict(r)
                model_name = entry.get("model_name")
                if model_name and model_name not in latest_by_model:
                    latest_by_model[model_name] = entry.get("id")

            out = []
            for r in rows:
                entry = dict(r)
                try:
                    entry["params"] = json.loads(entry.pop("params_json", "{}"))
                except Exception:
                    entry["params"] = {}
                try:
                    entry["metrics"] = json.loads(entry.pop("metrics_json", "{}"))
                except Exception:
                    entry["metrics"] = {}
                entry["is_deployed"] = latest_by_model.get(entry.get("model_name")) == entry.get("id")
                entry["is_best_run"] = best_run_id is not None and entry.get("training_run_id") == best_run_id
                out.append(entry)
            return out
        finally:
            conn.close()

    # ========== Drift Status ==========

    def _ensure_drift_status_table(self):
        conn = self._get_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS drift_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    drift_score REAL,
                    should_retrain INTEGER DEFAULT 0,
                    reasons_json TEXT,
                    metrics_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_drift_status_symbol_model
                    ON drift_status(symbol, model_name, created_at DESC)
            """)
            conn.commit()
        finally:
            conn.close()

    def save_drift_status(
        self,
        *,
        symbol: str,
        model_name: str,
        status: str,
        drift_score: float,
        should_retrain: bool,
        reasons: list[str],
        metrics: dict,
    ) -> int:
        import json
        self._ensure_drift_status_table()
        conn = self._get_conn()
        try:
            cur = conn.execute(
                """
                INSERT INTO drift_status
                (symbol, model_name, status, drift_score, should_retrain, reasons_json, metrics_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    model_name,
                    status,
                    float(drift_score),
                    1 if should_retrain else 0,
                    json.dumps(reasons, default=str),
                    json.dumps(metrics, default=str),
                ),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()

    def get_latest_drift_status(self, symbol: str) -> list[dict]:
        import json
        self._ensure_drift_status_table()
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT d1.*
                FROM drift_status d1
                INNER JOIN (
                    SELECT symbol, model_name, MAX(id) AS max_id
                    FROM drift_status
                    WHERE symbol = ?
                    GROUP BY symbol, model_name
                ) d2 ON d1.id = d2.max_id
                ORDER BY d1.model_name
                """,
                (symbol,),
            ).fetchall()
            out = []
            for r in rows:
                entry = dict(r)
                entry["should_retrain"] = bool(entry.get("should_retrain"))
                try:
                    entry["reasons"] = json.loads(entry.pop("reasons_json", "[]"))
                except Exception:
                    entry["reasons"] = []
                try:
                    entry["metrics"] = json.loads(entry.pop("metrics_json", "{}"))
                except Exception:
                    entry["metrics"] = {}
                out.append(entry)
            return out
        finally:
            conn.close()

    # ========== Alert Events ==========

    def _ensure_alert_events_table(self):
        conn = self._get_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alert_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    payload_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alert_events_symbol_created
                    ON alert_events(symbol, created_at DESC)
            """)
            conn.commit()
        finally:
            conn.close()

    def save_alert_event(
        self,
        *,
        symbol: str,
        alert_type: str,
        severity: str,
        message: str,
        payload: dict,
    ) -> int:
        import json
        self._ensure_alert_events_table()
        conn = self._get_conn()
        try:
            cur = conn.execute(
                """
                INSERT INTO alert_events (symbol, alert_type, severity, message, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (symbol, alert_type, severity, message, json.dumps(payload, default=str)),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()

    def get_alert_events(self, symbol: str, limit: int = 20) -> list[dict]:
        import json
        self._ensure_alert_events_table()
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT * FROM alert_events
                WHERE symbol = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (symbol, limit),
            ).fetchall()
            out = []
            for r in rows:
                entry = dict(r)
                try:
                    entry["payload"] = json.loads(entry.pop("payload_json", "{}"))
                except Exception:
                    entry["payload"] = {}
                out.append(entry)
            return out
        finally:
            conn.close()

    # ======================== NEWS SENTIMENT ========================

    def _ensure_news_sentiment_table(self):
        """Create news_sentiment table if it doesn't exist."""
        conn = self._get_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS news_sentiment (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT,
                    source TEXT,
                    source_type TEXT,
                    url TEXT,
                    published_at TEXT,
                    sentiment_score REAL DEFAULT 0,
                    sentiment_label TEXT DEFAULT 'neutral',
                    key_topics TEXT,
                    summary_th TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def save_news_sentiment(self, items: list[dict]) -> int:
        """Save news articles with sentiment to database. Returns count saved."""
        import json
        self._ensure_news_sentiment_table()

        conn = self._get_conn()
        try:
            count = 0
            for item in items:
                # Skip if same title+symbol exists (dedup)
                exists = conn.execute(
                    "SELECT 1 FROM news_sentiment WHERE title = ? AND symbol = ? LIMIT 1",
                    (item.get("title", ""), item.get("symbol", "")),
                ).fetchone()
                if exists:
                    continue

                conn.execute("""
                    INSERT INTO news_sentiment
                    (symbol, title, summary, source, source_type, url, published_at,
                     sentiment_score, sentiment_label, key_topics, summary_th)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    item.get("symbol", ""),
                    item.get("title", ""),
                    item.get("summary", ""),
                    item.get("source", ""),
                    item.get("source_type", ""),
                    item.get("url", ""),
                    item.get("published_at", ""),
                    item.get("sentiment_score", 0),
                    item.get("sentiment_label", "neutral"),
                    json.dumps(item.get("key_topics", []), ensure_ascii=False),
                    item.get("summary_th", ""),
                ))
                count += 1

            conn.commit()
            return count
        finally:
            conn.close()

    def get_news_sentiment(self, symbol: str, limit: int = 50, period: str = "30d") -> list[dict]:
        """Get recent news with sentiment for a symbol.
        period: '1d', '7d', '30d', 'all'
        """
        import json
        self._ensure_news_sentiment_table()

        # Map period to days
        days_map = {"1d": 1, "7d": 7, "30d": 30, "all": 9999}
        days = days_map.get(period, 30)

        conn = self._get_conn()
        try:
            rows = conn.execute("""
                SELECT * FROM news_sentiment
                WHERE (symbol = ? OR symbol = 'MARKET')
                  AND created_at >= datetime('now', ?)
                ORDER BY created_at DESC
                LIMIT ?
            """, (symbol, f"-{days} days", limit)).fetchall()

            results = []
            for row in rows:
                entry = dict(row)
                try:
                    entry["key_topics"] = json.loads(entry.get("key_topics", "[]"))
                except Exception:
                    entry["key_topics"] = []
                results.append(entry)
            return results
        finally:
            conn.close()

    def get_daily_sentiment(self, symbol: str, days: int = 30) -> list[dict]:
        """Get daily average sentiment score for a symbol (for features)."""
        self._ensure_news_sentiment_table()

        conn = self._get_conn()
        try:
            rows = conn.execute("""
                SELECT
                    DATE(published_at) as date,
                    AVG(sentiment_score) as avg_sentiment,
                    COUNT(*) as news_count,
                    SUM(CASE WHEN sentiment_label = 'bullish' THEN 1 ELSE 0 END) as bullish_count,
                    SUM(CASE WHEN sentiment_label = 'bearish' THEN 1 ELSE 0 END) as bearish_count
                FROM news_sentiment
                WHERE (symbol = ? OR symbol = 'MARKET')
                  AND published_at IS NOT NULL
                  AND published_at != ''
                GROUP BY DATE(published_at)
                ORDER BY date DESC
                LIMIT ?
            """, (symbol, days)).fetchall()

            return [dict(row) for row in rows]
        finally:
            conn.close()

