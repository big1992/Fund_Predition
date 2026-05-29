"""
AI Explanation service using OpenAI GPT.
Provides intelligent analysis of predictions and model performance.
"""

import os
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AIExplainer:
    """Generate AI-powered explanations for predictions and model metrics."""

    def __init__(self):
        self.client = None
        self.model = "gpt-4o-mini"
        self.unavailable_reason = "not_initialized"
        self._init_client()

    def _init_client(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                # Support keys persisted in app settings (e.g., overrides file)
                try:
                    from config.settings import settings
                    api_key = settings.openai_api_key
                except Exception:
                    api_key = None
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                self.client = OpenAI(api_key=api_key)
                self.unavailable_reason = ""
                logger.info("OpenAI client initialized successfully")
            else:
                self.unavailable_reason = "missing_api_key"
                logger.warning("OPENAI_API_KEY not set, AI explanations disabled")
        except ImportError as e:
            self.unavailable_reason = "missing_dependency"
            logger.error("Failed to init OpenAI (dependency missing): %s", e)
        except Exception as e:
            self.unavailable_reason = "openai_init_failed"
            logger.error("Failed to init OpenAI: %s", e)

    def is_available(self) -> bool:
        return self.client is not None

    def get_unavailable_message(self) -> str:
        """Return a user-facing message describing why AI explainer is unavailable."""
        if self.is_available():
            return ""
        if self.unavailable_reason == "missing_api_key":
            return "AI explanation unavailable — OPENAI_API_KEY not configured"
        if self.unavailable_reason == "missing_dependency":
            return "AI explanation unavailable — backend dependency `openai` is not installed"
        if self.unavailable_reason == "openai_init_failed":
            return "AI explanation unavailable — OpenAI client initialization failed"
        return "AI explanation unavailable — service is not ready"

    async def explain_prediction(
        self,
        symbol: str,
        current_price: float,
        predictions: list[dict],
        signal: str,
        signal_reason: str,
        model_type: str,
        indicators: Optional[dict] = None,
    ) -> str:
        """Generate AI explanation for a stock prediction."""
        if not self.client:
            return "⚠️ AI explanation unavailable (API key not configured)"

        # Build indicator context
        ind_text = ""
        if indicators:
            ind_text = f"""
Technical Indicators:
- RSI(14): {indicators.get('rsi_14', 'N/A')}
- MACD: {indicators.get('macd', 'N/A')}, Signal: {indicators.get('macd_signal', 'N/A')}
- SMA20: {indicators.get('sma_20', 'N/A')}, SMA50: {indicators.get('sma_50', 'N/A')}
- Bollinger Upper: {indicators.get('bb_upper', 'N/A')}, Lower: {indicators.get('bb_lower', 'N/A')}
- ATR: {indicators.get('atr', 'N/A')}
- Volume Ratio: {indicators.get('volume_ratio', 'N/A')}
"""

        pred_text = "\n".join(
            [f"  Day {i+1}: ฿{p['predicted_price']:.2f} (confidence: {p['confidence']:.1f}%)"
             for i, p in enumerate(predictions)]
        )

        prompt = f"""คุณเป็นนักวิเคราะห์หุ้นไทยมืออาชีพ กรุณาวิเคราะห์สัญญาณการลงทุนนี้:

หุ้น: {symbol}
ราคาปัจจุบัน: ฿{current_price:.2f}
Model ที่ใช้: {model_type}
สัญญาณ: {signal}
เหตุผลเบื้องต้น: {signal_reason}

การพยากรณ์ 5 วัน:
{pred_text}
{ind_text}

กรุณาให้:
1. สรุปสั้นๆ ว่าทำไม Model ถึงให้สัญญาณนี้
2. วิเคราะห์ Technical Indicators ที่สนับสนุน/ขัดแย้ง
3. ระดับความเสี่ยง (ต่ำ/กลาง/สูง)
4. คำแนะนำ (2-3 ประโยค)

ตอบเป็นภาษาไทย กระชับ ไม่เกิน 200 คำ ใช้ emoji ให้อ่านง่าย"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "คุณเป็น AI ที่เชี่ยวชาญด้านวิเคราะห์หุ้นไทย ตอบกระชับ ใช้ภาษาที่เข้าใจง่าย"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error("OpenAI prediction explanation error: %s", e)
            return f"⚠️ AI explanation error: {str(e)}"

    async def explain_performance(
        self,
        models: list[dict],
        backtest: Optional[dict] = None,
    ) -> str:
        """Generate AI explanation for model performance metrics."""
        if not self.client:
            return "⚠️ AI explanation unavailable (API key not configured)"

        models_text = ""
        for m in models:
            models_text += f"""
- {m.get('model_name', 'unknown').upper()} ({m.get('symbol', '')}):
  RMSE: {m.get('rmse', 'N/A')}, MAE: {m.get('mae', 'N/A')}, MAPE: {m.get('mape', 'N/A')}%
  Direction Accuracy: {m.get('directional_accuracy', 'N/A')}%, R²: {m.get('r_squared', 'N/A')}
"""

        bt_text = ""
        if backtest:
            bt_text = f"""
Backtest Results ({backtest.get('symbol', '')}):
- Total Return: {backtest.get('total_return', 0)*100:.2f}%
- Sharpe Ratio: {backtest.get('sharpe_ratio', 0):.3f}
- Max Drawdown: {backtest.get('max_drawdown', 0)*100:.2f}%
- Win Rate: {backtest.get('win_rate', 0):.1f}%
- Total Trades: {backtest.get('total_trades', 0)}
- Profit Factor: {backtest.get('profit_factor', 0):.2f}
"""

        prompt = f"""คุณเป็นผู้เชี่ยวชาญ Machine Learning สำหรับการเงิน กรุณาวิเคราะห์ผลการทำงานของ Models:

{models_text}
{bt_text}

กรุณาให้:
1. สรุปภาพรวมว่า Model ไหนดีที่สุดและทำไม
2. วิเคราะห์จุดแข็ง/จุดอ่อนของแต่ละ Model
3. MAPE ต่ำกว่า 5% = ดีมาก, 5-10% = พอใช้, >10% = ต้องปรับปรุง
4. Direction Accuracy > 55% ถือว่าใช้งานได้
5. ถ้ามี Backtest ให้วิเคราะห์ Sharpe Ratio (>1 = ดี) และ Max Drawdown
6. คำแนะนำในการปรับปรุง

ตอบเป็นภาษาไทย กระชับ ไม่เกิน 250 คำ ใช้ emoji ให้อ่านง่าย"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "คุณเป็น AI ผู้เชี่ยวชาญ ML/Finance ตอบกระชับ ใช้ภาษาไทยที่เข้าใจง่าย"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=600,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error("OpenAI performance explanation error: %s", e)
            return f"⚠️ AI explanation error: {str(e)}"

    async def explain_training(
        self,
        symbol: str,
        metrics: dict,
        current_params: dict,
        previous_metrics: dict = None,
    ) -> dict:
        """Analyze training results and recommend hyperparameter adjustments.
        Returns dict with 'explanation' (str) and 'recommended_params' (dict).
        previous_metrics: optional dict with 'metrics' and 'params' from a prior round.
        """
        if not self.client:
            return {
                "explanation": "⚠️ AI explanation unavailable (API key not configured)",
                "recommended_params": {},
                "satisfaction_score": 5,
            }

        import json

        # Build LSTM metrics text
        lstm_text = ""
        if "lstm" in metrics and "error" not in metrics.get("lstm", {}):
            lm = metrics["lstm"]
            tr = lm.get("training", {})
            lstm_text = f"""
LSTM Results:
- Epochs trained: {tr.get('epochs_trained', '?')} / {current_params.get('lstm_epochs', '?')} (max)
- Final Loss: {tr.get('final_loss', '?')}
- Val Loss: {tr.get('val_loss', '?')}
- Final MAE: {tr.get('final_mae', '?')}
- Test RMSE: {lm.get('rmse', '?')}
- Test MAE: {lm.get('mae', '?')}
- Test MAPE: {lm.get('mape', '?')}%
- Direction Accuracy: {lm.get('directional_accuracy', '?')}%
- R²: {lm.get('r_squared', '?')}
- Last 5 train losses: {tr.get('loss_history', [])[-5:] if tr.get('loss_history') else 'N/A'}
- Last 5 val losses: {tr.get('val_loss_history', [])[-5:] if tr.get('val_loss_history') else 'N/A'}
"""

        # Build XGBoost metrics text
        xgb_text = ""
        if "xgboost" in metrics and "error" not in metrics.get("xgboost", {}):
            xm = metrics["xgboost"]
            xgb_text = f"""
XGBoost Results:
- Test RMSE: {xm.get('rmse', '?')}
- Test MAE: {xm.get('mae', '?')}
- Test MAPE: {xm.get('mape', '?')}%
- Direction Accuracy: {xm.get('directional_accuracy', '?')}%
- R²: {xm.get('r_squared', '?')}
"""

        # Build current params text
        params_text = f"""
Current Hyperparameters:
LSTM: seq_length={current_params.get('lstm_sequence_length', '?')}, units=[{current_params.get('lstm_units_1', '?')},{current_params.get('lstm_units_2', '?')}], dropout={current_params.get('lstm_dropout', '?')}, lr={current_params.get('lstm_learning_rate', '?')}, epochs={current_params.get('lstm_epochs', '?')}, batch_size={current_params.get('lstm_batch_size', '?')}, early_stop={current_params.get('lstm_early_stopping', '?')}
XGBoost: n_estimators={current_params.get('xgb_n_estimators', '?')}, max_depth={current_params.get('xgb_max_depth', '?')}, lr={current_params.get('xgb_learning_rate', '?')}, subsample={current_params.get('xgb_subsample', '?')}, colsample={current_params.get('xgb_colsample_bytree', '?')}, reg_alpha={current_params.get('xgb_reg_alpha', '?')}, reg_lambda={current_params.get('xgb_reg_lambda', '?')}
Ensemble: lstm_weight={current_params.get('ensemble_lstm_weight', '?')}, xgb_weight={current_params.get('ensemble_xgb_weight', '?')}
Data Split: train={current_params.get('data_train_ratio', '?')}, val={current_params.get('data_val_ratio', '?')}, test={current_params.get('data_test_ratio', '?')}
"""

        # Build previous comparison text
        history_text = ""
        
        # Priority 1: Use directly-provided previous_metrics (from auto-tune loop)
        if previous_metrics and isinstance(previous_metrics, dict):
            pm = previous_metrics.get("metrics", previous_metrics)
            pp = previous_metrics.get("params", {})
            prev_lstm = pm.get("lstm", {})
            prev_xgb = pm.get("xgboost", {})
            
            history_text = f"""
=== PREVIOUS Round Results (BEFORE this tuning) ===
LSTM prev: RMSE={prev_lstm.get('rmse', 'N/A')}, MAE={prev_lstm.get('mae', 'N/A')}, MAPE={prev_lstm.get('mape', 'N/A')}%, DirAcc={prev_lstm.get('directional_accuracy', 'N/A')}%, R²={prev_lstm.get('r_squared', 'N/A')}
XGBoost prev: RMSE={prev_xgb.get('rmse', 'N/A')}, MAE={prev_xgb.get('mae', 'N/A')}, MAPE={prev_xgb.get('mape', 'N/A')}%, DirAcc={prev_xgb.get('directional_accuracy', 'N/A')}%, R²={prev_xgb.get('r_squared', 'N/A')}
Previous Params: LSTM(epochs={pp.get('lstm_epochs', '?')}, lr={pp.get('lstm_learning_rate', '?')}, units=[{pp.get('lstm_units_1', '?')},{pp.get('lstm_units_2', '?')}], dropout={pp.get('lstm_dropout', '?')}, batch={pp.get('lstm_batch_size', '?')}), XGBoost(n_est={pp.get('xgb_n_estimators', '?')}, lr={pp.get('xgb_learning_rate', '?')}, depth={pp.get('xgb_max_depth', '?')}, subsample={pp.get('xgb_subsample', '?')}), Ensemble({pp.get('ensemble_lstm_weight', '?')}:{pp.get('ensemble_xgb_weight', '?')})
⚠️ IMPORTANT: Compare CURRENT vs PREVIOUS metrics above. State clearly if RMSE/MAPE improved or worsened.
"""
        else:
            # Priority 2: Read from database
            try:
                from config.settings import settings
                from data.storage import DatabaseManager
                db = DatabaseManager(settings.db_path)
                hist = db.get_training_history(symbol, limit=5)
                
                # hist is ordered DESC, so [0] is latest (current), [1] is previous
                if len(hist) > 1:
                    prev_run = hist[1]  # second most recent
                    pp = prev_run.get("params", {})
                    ts = prev_run.get("created_at", "?")
                    
                    history_text = f"""
=== PREVIOUS Training Run ({ts}) ===
LSTM prev: RMSE={prev_run.get('lstm_rmse', 'N/A')}, MAE={prev_run.get('lstm_mae', 'N/A')}, MAPE={prev_run.get('lstm_mape', 'N/A')}%, DirAcc={prev_run.get('lstm_directional_accuracy', 'N/A')}%, R²={prev_run.get('lstm_r_squared', 'N/A')}
XGBoost prev: RMSE={prev_run.get('xgb_rmse', 'N/A')}, MAE={prev_run.get('xgb_mae', 'N/A')}, MAPE={prev_run.get('xgb_mape', 'N/A')}%, DirAcc={prev_run.get('xgb_directional_accuracy', 'N/A')}%, R²={prev_run.get('xgb_r_squared', 'N/A')}
Previous Params: LSTM(epochs={pp.get('lstm_epochs', '?')}, lr={pp.get('lstm_learning_rate', '?')}, units=[{pp.get('lstm_units_1', '?')},{pp.get('lstm_units_2', '?')}], dropout={pp.get('lstm_dropout', '?')}, batch={pp.get('lstm_batch_size', '?')}), XGBoost(n_est={pp.get('xgb_n_estimators', '?')}, lr={pp.get('xgb_learning_rate', '?')}, depth={pp.get('xgb_max_depth', '?')}, subsample={pp.get('xgb_subsample', '?')}), Ensemble({pp.get('ensemble_lstm_weight', '?')}:{pp.get('ensemble_xgb_weight', '?')})
"""
                
                # Also include best run info
                best = db.get_best_training_run(symbol)
                if best:
                    bp = best.get("params", {})
                    history_text += f"""
=== BEST Run Ever (id={best.get('id')}) ===
Best LSTM MAPE: {best.get('lstm_mape', 'N/A')}%, Best XGBoost MAPE: {best.get('xgb_mape', 'N/A')}%
Best Params: LSTM(epochs={bp.get('lstm_epochs', '?')}, lr={bp.get('lstm_learning_rate', '?')}, dropout={bp.get('lstm_dropout', '?')}), XGBoost(lr={bp.get('xgb_learning_rate', '?')}, depth={bp.get('xgb_max_depth', '?')})
"""
            except Exception as e:
                logger.warning("Could not load DB history for explanation: %s", e)

        context = f"{lstm_text}\n{xgb_text}\n{history_text}\n{params_text}"

        analysis_prompt = f"""คุณเป็นผู้เชี่ยวชาญ Machine Learning ด้าน Hyperparameter Tuning สำหรับ Stock Prediction
วิเคราะห์ผลการ Train ของหุ้น {symbol} แล้วแนะนำการปรับค่า:
{context}

กรุณาวิเคราะห์:
1. 📊 สรุปผลการ Train (ปัจจุบัน) — ดีไหม? overfitting/underfitting หรือไม่?
2. 🔄 เปรียบเทียบกับครั้งก่อน — พัฒนาขึ้น หรือแย่ลง? (อธิบายสั้นๆ ว่าการปรับค่ารอบที่แล้วส่งผลอย่างไร)
3. 🔍 วิเคราะห์ Loss Curve — converge ดีหรือยัง?
4. 📈 เปรียบเทียบ LSTM vs XGBoost — ตัวไหนดีกว่า?
5. 🛠️ แนะนำค่า Hyperparameters ถัดไปที่ควรปรับ พร้อมค่าที่แนะนำ
6. ⚖️ แนะนำ Ensemble weights

ตอบเป็นภาษาไทย กระชับ ใช้ emoji ไม่เกิน 450 คำ"""

        json_prompt = f"""Based on these ML training results (including previous run comparison), recommend optimal future hyperparameters.
{context}

Return ONLY a valid JSON object with these fields:
1. "satisfaction_score": integer 1-10 rating of current model quality (10=excellent, no changes needed; 7+=good enough; <7=needs improvement)
2. "recommended_params": object with ONLY parameters you want to CHANGE. Use these key names:
   lstm_sequence_length, lstm_units_1, lstm_units_2, lstm_dropout, lstm_dense_units, lstm_learning_rate, lstm_epochs, lstm_batch_size, lstm_early_stopping, xgb_n_estimators, xgb_max_depth, xgb_learning_rate, xgb_subsample, xgb_colsample_bytree, xgb_reg_alpha, xgb_reg_lambda, ensemble_lstm_weight, ensemble_xgb_weight

Example: {{"satisfaction_score": 6, "recommended_params": {{"lstm_dropout": 0.3, "xgb_learning_rate": 0.05}}}}
If satisfaction_score >= 8, recommended_params should be empty {{}}.
Return ONLY the JSON, no markdown fences, no text."""

        explanation = ""
        recommended = {}
        score = 5

        try:
            # Get analysis text
            resp1 = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "คุณเป็น ML Engineer ผู้เชี่ยวชาญ Hyperparameter Tuning ตอบกระชับ actionable เป็นภาษาไทย"},
                    {"role": "user", "content": analysis_prompt},
                ],
                max_tokens=800,
                temperature=0.7,
            )
            explanation = resp1.choices[0].message.content

            # Get structured JSON recommendations with satisfaction score
            resp2 = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a ML expert. Return only valid JSON with no extra text."},
                    {"role": "user", "content": json_prompt},
                ],
                max_tokens=400,
                temperature=0.2,
            )
            json_text = resp2.choices[0].message.content.strip()
            # Handle markdown code blocks
            if json_text.startswith("```"):
                json_text = json_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            parsed = json.loads(json_text)
            
            # Extract score and params from response
            if isinstance(parsed, dict):
                score = parsed.get("satisfaction_score", 5)
                recommended = parsed.get("recommended_params", parsed)
                # If AI returned flat params without wrapper, remove non-param keys
                recommended.pop("satisfaction_score", None)

        except json.JSONDecodeError as e:
            logger.warning("Failed to parse AI recommended params: %s", e)
        except Exception as e:
            logger.error("OpenAI training analysis error: %s", e)
            if not explanation:
                explanation = f"⚠️ AI analysis error: {str(e)}"

        return {
            "explanation": explanation,
            "recommended_params": recommended,
            "satisfaction_score": score,
        }
