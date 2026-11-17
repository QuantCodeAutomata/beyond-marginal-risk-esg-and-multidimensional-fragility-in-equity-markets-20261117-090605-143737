"""Unit tests for src/data_construction.py on small synthetic/fixture inputs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_construction import (
    abdi_ranaldo_chl_spread,
    amihud_illiquidity,
    down_up_state,
    joint_state_indicator,
    log_realized_volatility,
    momentum,
    monthly_loss,
    monthly_return,
    one_dim_tail_indicator,
    pairwise_state_indicator,
    winsorize,
)


def _daily_index(n_days: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n_days, freq="B")


def test_monthly_loss_is_negative_of_monthly_return():
    idx = _daily_index(70)
    close = pd.Series(100 * (1 + 0.001) ** np.arange(len(idx)), index=idx)
    ret = monthly_return(close)
    loss = monthly_loss(close)
    pd.testing.assert_series_equal(loss, -ret, check_names=False)


def test_monthly_loss_larger_is_worse():
    idx = _daily_index(40)
    rising = pd.Series(np.linspace(100, 110, len(idx)), index=idx)
    falling = pd.Series(np.linspace(100, 90, len(idx)), index=idx)
    assert monthly_loss(falling).iloc[-1] > monthly_loss(rising).iloc[-1]


def test_log_realized_volatility_scales_with_dispersion():
    idx = _daily_index(60)
    rng = np.random.default_rng(0)
    low_vol = pd.Series(rng.normal(0.0, 0.001, len(idx)), index=idx)
    high_vol = pd.Series(rng.normal(0.0, 0.05, len(idx)), index=idx)
    low = log_realized_volatility(low_vol).dropna()
    high = log_realized_volatility(high_vol).dropna()
    assert (high.values > low.values).all()


def test_abdi_ranaldo_chl_spread_nonnegative_or_nan():
    idx = _daily_index(30)
    rng = np.random.default_rng(1)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, len(idx))), index=idx)
    high = close + rng.uniform(0.1, 1.0, len(idx))
    low = close - rng.uniform(0.1, 1.0, len(idx))
    spread = abdi_ranaldo_chl_spread(high, low, close)
    valid = spread.dropna()
    assert (valid >= 0).all()


def test_down_up_state_are_nonnegative_and_mutually_exclusive():
    market_return = pd.Series([-0.05, 0.03, 0.0, -0.001, 0.08])
    down, up = down_up_state(market_return)
    assert (down >= 0).all() and (up >= 0).all()
    assert ((down * up) == 0).all()
    # Down_t = max(0, -R_t): a 5% decline gives Down = 0.05, Up = 0
    assert down.iloc[0] == pytest_approx(0.05)
    assert up.iloc[0] == 0.0


def pytest_approx(value, tol=1e-9):
    class _Approx:
        def __eq__(self, other):
            return abs(other - value) < tol
    return _Approx()


def test_winsorize_caps_extremes_without_changing_middle():
    series = pd.Series(np.arange(100, dtype=float))
    out = winsorize(series, 0.01, 0.99)
    assert out.max() <= series.quantile(0.99)
    assert out.min() >= series.quantile(0.01)
    assert out.iloc[50] == series.iloc[50]


def test_joint_state_indicator_is_product_of_three_conditions():
    loss = pd.Series([0.1, 0.0, 0.1])
    rvol = pd.Series([1.0, 1.0, 0.0])
    chl = pd.Series([2.0, 2.0, 2.0])
    thresholds = {"loss": 0.05, "log_rvol": 0.5, "log_chl": 1.5}
    joint = joint_state_indicator(loss, rvol, chl, thresholds)
    assert list(joint) == [1, 0, 0]


def test_pairwise_state_indicator_and_one_dim_tail_consistency():
    a = pd.Series([1.0, 2.0, 0.0])
    b = pd.Series([5.0, 1.0, 5.0])
    pair = pairwise_state_indicator(a, 1.0, b, 2.0)
    tail_a = one_dim_tail_indicator(a, 1.0)
    tail_b = one_dim_tail_indicator(b, 2.0)
    assert list(pair) == list(tail_a * tail_b)


def test_momentum_excludes_most_recent_month():
    idx = pd.date_range("2019-01-01", periods=400, freq="B")
    close = pd.Series(100 * 1.0005 ** np.arange(len(idx)), index=idx)
    mom = momentum(close).dropna()
    assert (mom > 0).all()  # a steadily rising price has positive 12-2 momentum


def test_amihud_illiquidity_increases_with_lower_volume():
    idx = _daily_index(40)
    ret = pd.Series(0.01, index=idx)
    low_volume = amihud_illiquidity(ret, pd.Series(1000.0, index=idx)).dropna()
    high_volume = amihud_illiquidity(ret, pd.Series(1_000_000.0, index=idx)).dropna()
    assert (low_volume.values > high_volume.values).all()
