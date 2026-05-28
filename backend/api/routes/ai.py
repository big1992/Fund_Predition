"""
AI-powered explanation routes.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from api.main import app_state

router = APIRouter()


class PredictionExplainRequest(BaseModel):
    symbol: str
    current_price: float
    predictions: list[dict]
    signal: str
    signal_reason: str
    model_type: str = "ensemble"
    indicators: Optional[dict] = None


class PerformanceExplainRequest(BaseModel):
    models: list[dict]
    backtest: Optional[dict] = None


class ExplainResponse(BaseModel):
    explanation: str
    available: bool = True


@router.post("/explain-prediction", response_model=ExplainResponse)
async def explain_prediction(request: PredictionExplainRequest):
    """Get AI explanation for a prediction signal."""
    explainer = app_state.get("ai_explainer")
    if not explainer or not explainer.is_available():
        return ExplainResponse(
            explanation="⚠️ AI explanation unavailable — OPENAI_API_KEY not configured",
            available=False,
        )

    explanation = await explainer.explain_prediction(
        symbol=request.symbol,
        current_price=request.current_price,
        predictions=request.predictions,
        signal=request.signal,
        signal_reason=request.signal_reason,
        model_type=request.model_type,
        indicators=request.indicators,
    )
    return ExplainResponse(explanation=explanation)


@router.post("/explain-performance", response_model=ExplainResponse)
async def explain_performance(request: PerformanceExplainRequest):
    """Get AI explanation for model performance metrics."""
    explainer = app_state.get("ai_explainer")
    if not explainer or not explainer.is_available():
        return ExplainResponse(
            explanation="⚠️ AI explanation unavailable — OPENAI_API_KEY not configured",
            available=False,
        )

    explanation = await explainer.explain_performance(
        models=request.models,
        backtest=request.backtest,
    )
    return ExplainResponse(explanation=explanation)


class TrainingExplainRequest(BaseModel):
    symbol: str
    metrics: dict
    current_params: dict
    previous_metrics: Optional[dict] = None  # before/after comparison


class TrainingExplainResponse(BaseModel):
    explanation: str
    recommended_params: dict = {}
    satisfaction_score: int = 5
    available: bool = True


@router.post("/explain-training", response_model=TrainingExplainResponse)
async def explain_training(request: TrainingExplainRequest):
    """Get AI analysis of training results with hyperparameter recommendations."""
    explainer = app_state.get("ai_explainer")
    if not explainer or not explainer.is_available():
        return TrainingExplainResponse(
            explanation="⚠️ AI analysis unavailable — OPENAI_API_KEY not configured",
            available=False,
        )

    result = await explainer.explain_training(
        symbol=request.symbol,
        metrics=request.metrics,
        current_params=request.current_params,
        previous_metrics=request.previous_metrics,
    )
    return TrainingExplainResponse(
        explanation=result["explanation"],
        recommended_params=result.get("recommended_params", {}),
        satisfaction_score=result.get("satisfaction_score", 5),
    )
