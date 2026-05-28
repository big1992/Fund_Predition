import pytest
from pydantic import ValidationError

from schemas.api_schemas import TrainRequest, BacktestRequest, CollectRequest


def test_train_request_normalizes_symbols_to_uppercase():
    req = TrainRequest(symbols=["ptt.bk", "aapl"], model_type="all")
    assert req.symbols == ["PTT.BK", "AAPL"]


def test_train_request_rejects_invalid_symbol_format():
    with pytest.raises(ValidationError):
        TrainRequest(symbols=["BAD SYMBOL"], model_type="all")


def test_backtest_request_rejects_invalid_date_range():
    with pytest.raises(ValidationError):
        BacktestRequest(
            symbol="PTT.BK",
            start_date="2026-05-20",
            end_date="2026-05-01",
        )


def test_collect_request_period_is_restricted_literal():
    req = CollectRequest(period="5y")
    assert req.period == "5y"

    with pytest.raises(ValidationError):
        CollectRequest(period="10y")
