"""Panel construction: outcomes, controls, exposures, market state, Joint states.

Implements paper Sec. 3.2-3.4 and 4.1 (Equations 1-4) from daily price/volume
bars and firm-level accounting data. All "as of month t" variables are built
from information strictly preceding month t to avoid look-ahead, per the
timing requirements in Sec. 3.1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def monthly_return(close: pd.Series) -> pd.Series:
    """Corporate-action-consistent monthly price return (paper Eq. 1 uses R_{i,t}).

    ``close`` must already be dividend- and split-adjusted (the Dataloader
    default), so a simple end-of-month-to-end-of-month percentage change is
    corporate-action consistent.
    """
    month_end = close.resample("ME").last()
    return month_end.pct_change()


def monthly_loss(close: pd.Series) -> pd.Series:
    """Loss_{i,t} = -R_{i,t} (paper Eq. 1); larger values are worse."""
    return -monthly_return(close)


def log_realized_volatility(daily_returns: pd.Series, annualization: int = 252) -> pd.Series:
    """log RVOL_{i,t} = log(sqrt(annualization) * sd(daily returns in month t)) (paper Eq. 2)."""
    monthly_sd = daily_returns.resample("ME").std(ddof=1)
    rvol = np.sqrt(annualization) * monthly_sd
    return np.log(rvol)


def abdi_ranaldo_chl_spread(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Daily close-high-low spread estimator of Abdi and Ranaldo [2017] (paper Sec. 3.3).

    Custom implementation (paper Eq. in Sec. 3.3; no equivalent estimator is
    provided by `ta`, `arch`, or `quantstats`). For each overlapping two-day
    window the estimator uses the two-day log mid-range
        eta_t = 0.5 * (log(high_t) + log(low_t))
    and the two-day average close
        c_bar_t = 0.5 * (log(close_{t-1}) + log(close_t)),
    giving
        S^2_t = 4 * (c_bar_t - eta_t) * (c_bar_t - eta_{t-1}),
    with the spread estimate sqrt(max(S^2_t, 0)). A day with S^2_t < 0 is not
    identified and is treated as missing, as in Abdi and Ranaldo [2017].
    """
    log_high = np.log(high)
    log_low = np.log(low)
    log_close = np.log(close)
    eta = 0.5 * (log_high + log_low)
    c_bar = 0.5 * (log_close + log_close.shift(1))
    s_squared = 4.0 * (c_bar - eta) * (c_bar - eta.shift(1))
    s_squared = s_squared.where(s_squared >= 0)
    return np.sqrt(s_squared)


def log_chl_spread_monthly(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """log of the monthly-aggregated CHL spread, averaged across valid daily estimates (Sec. 3.3)."""
    daily_spread = abdi_ranaldo_chl_spread(high, low, close)
    monthly_spread = daily_spread.resample("ME").mean()
    return np.log(monthly_spread)


def trailing_window_mean(monthly_series: pd.Series, window_months: int) -> pd.Series:
    """Trailing mean over ``window_months`` ending at t-1 (strictly preceding month t).

    Used for RVOL12 and CHL12 ex ante exposure blocks (Sec. 3.2): the value
    assigned to month t is computed from months (t-window) .. (t-1) only.
    """
    rolled = monthly_series.rolling(window_months, min_periods=window_months).mean()
    return rolled.shift(1)


def rolling_beta(asset_daily_returns: pd.Series, market_daily_returns: pd.Series,
                  window_months: int, as_of_month_end: pd.DatetimeIndex) -> pd.Series:
    """Daily-return market beta over a trailing ``window_months`` window ending before t.

    Paper Sec. 3.2: "the return model uses a 24-month daily market beta,
    estimated only from data preceding month t." For each month end in
    ``as_of_month_end`` the beta is OLS cov(r_i, r_m)/var(r_m) using daily
    returns strictly before that month-end's first day.
    """
    aligned = pd.concat(
        {"asset": asset_daily_returns, "market": market_daily_returns}, axis=1
    ).dropna()
    betas = pd.Series(index=as_of_month_end, dtype=float)
    for month_end in as_of_month_end:
        window_start = month_end - pd.DateOffset(months=window_months)
        window = aligned.loc[(aligned.index >= window_start) & (aligned.index < month_end.replace(day=1))]
        if len(window) < window_months * 10:  # require a reasonably full window
            betas[month_end] = np.nan
            continue
        cov = np.cov(window["asset"], window["market"])[0, 1]
        var = np.var(window["market"], ddof=1)
        betas[month_end] = cov / var if var > 0 else np.nan
    return betas


def downside_beta(asset_daily_returns: pd.Series, market_daily_returns: pd.Series,
                   window_months: int, as_of_month_end: pd.DatetimeIndex) -> pd.Series:
    """24-month downside beta: beta estimated only on days with market_return < 0 (Table A6 robustness)."""
    down_mask = market_daily_returns < 0
    return rolling_beta(asset_daily_returns[down_mask.reindex(asset_daily_returns.index, fill_value=False)],
                         market_daily_returns[down_mask], window_months, as_of_month_end)


def momentum(close: pd.Series) -> pd.Series:
    """Momentum_{i,t}: compounded return from month t-12 to t-2, excluding the most recent month
    (paper Sec. 3.2, Jegadeesh and Titman [1993])."""
    month_end_close = close.resample("ME").last()
    cumulative = (1.0 + month_end_close.pct_change()).rolling(11, min_periods=11).apply(np.prod, raw=True) - 1.0
    return cumulative.shift(2)


def winsorize(series: pd.Series, lower_pctl: float, upper_pctl: float) -> pd.Series:
    """Winsorize ``series`` at the given pooled percentile thresholds (Sec. 3.2)."""
    lower = series.quantile(lower_pctl)
    upper = series.quantile(upper_pctl)
    return series.clip(lower=lower, upper=upper)


def down_up_state(market_return: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Down_t = max(0, -R^M_t), Up_t = max(0, R^M_t) (paper Eq. 4)."""
    down = (-market_return).clip(lower=0.0)
    up = market_return.clip(lower=0.0)
    return down, up


def frozen_quantile_thresholds(pooled_values: dict[str, pd.Series], quantile: float) -> dict[str, float]:
    """Pooled unconditional percentile thresholds frozen before coefficients are inspected (Sec. 3.4)."""
    return {name: float(series.dropna().quantile(quantile)) for name, series in pooled_values.items()}


def joint_state_indicator(loss: pd.Series, log_rvol: pd.Series, log_chl: pd.Series,
                           thresholds: dict[str, float]) -> pd.Series:
    """J^q_{i,t} = 1{Loss>=q_L} * 1{logRVOL>=q_V} * 1{logCHL>=q_C} (paper Eq. 3)."""
    return (
        (loss >= thresholds["loss"]).astype(int)
        * (log_rvol >= thresholds["log_rvol"]).astype(int)
        * (log_chl >= thresholds["log_chl"]).astype(int)
    )


def pairwise_state_indicator(series_a: pd.Series, threshold_a: float,
                              series_b: pd.Series, threshold_b: float) -> pd.Series:
    """Two-dimensional pairwise tail intersection, e.g. Return+RVOL (exp_3, exp_5)."""
    return (series_a >= threshold_a).astype(int) * (series_b >= threshold_b).astype(int)


def one_dim_tail_indicator(series: pd.Series, threshold: float) -> pd.Series:
    """One-dimensional tail indicator 1{series >= threshold} (exp_5 1D ladder)."""
    return (series >= threshold).astype(int)


def amihud_illiquidity(daily_return: pd.Series, dollar_volume: pd.Series) -> pd.Series:
    """Amihud [2002] illiquidity: monthly mean of |daily return| / dollar volume (exp_3 validation)."""
    daily_measure = daily_return.abs() / dollar_volume.replace(0, np.nan)
    return daily_measure.resample("ME").mean()
