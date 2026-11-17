"""Live-API tests for the Dataloader-backed market-state loader.

These tests require DATALOADER_API_KEY and MASSIVE_API_KEY; they are skipped
automatically (see ``dataloader_credentials`` in tests/conftest.py) when the
keys are absent, so the default test run never depends on a live call.
"""
from __future__ import annotations

import pytest

from data import data_loader


@pytest.mark.market_data
def test_load_ohlcv_small_slice_has_expected_shape(dataloader_credentials):
    """A short SPY load returns clean OHLCV rows with no look-ahead ordering issues."""
    bars, meta = data_loader.load_ohlcv(
        ["SPY"], "2024-01-02", "2024-01-31", timeframe="1d", use_cache=False
    )
    assert not bars.empty
    assert {"timestamp", "date", "ticker", "open", "high", "low", "close", "volume"} <= set(
        bars.columns
    )
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1) + 1e-6).all()
    assert (bars[["open", "close"]].max(axis=1) <= bars["high"] + 1e-6).all()
    assert (bars["close"] > 0).all()
    assert bars["date"].is_monotonic_increasing
    assert meta["ticker_details"], "ticker_details must be populated for provenance logging"


@pytest.mark.market_data
def test_market_index_monthly_return_sign_matches_known_period(dataloader_credentials):
    """March 2020 is a documented sharp S&P 500 decline: the monthly return must be negative."""
    bars, _ = data_loader.load_ohlcv(
        ["SPY"], "2020-02-20", "2020-03-31", timeframe="1d", use_cache=False
    )
    bars = bars.sort_values("date")
    month_return = bars["close"].iloc[-1] / bars["close"].iloc[0] - 1.0
    assert month_return < 0
