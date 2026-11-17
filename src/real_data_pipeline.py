"""Real-data assembly of the firm-month panel (Sec. 3.1-3.4) from permitted sources.

Builds the non-financial S&P 500 universe (Wikipedia constituent list), daily
OHLCV bars (QuantX Dataloader), and annual accounting statements (Massive
REST) into the same firm-month schema produced by
``data.generate_synthetic.generate_panel``, so ``src.pipeline`` can assemble a
panel whose market-based outcomes (Loss, log RVOL, log CHL), ex ante exposures
(Beta24, RVOL12, CHL12, momentum, Amihud) and accounting controls (log
assets, leverage, cash/assets, profitability, investment) come from real
market and accounting data. MSCI ESG scores (IAS, WAKIS, E/S/G) have no
permitted REST source (see ``data/data_loader.py``) and are attached from
``data.generate_synthetic`` keyed by these real firm identifiers.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from data.data_loader import DataloaderError, MassiveError, load_annual_fundamentals, load_ohlcv, load_sp500_constituents
from src import config
from src.data_construction import (
    abdi_ranaldo_chl_spread,
    log_chl_spread_monthly,
    log_realized_volatility,
    momentum as momentum_fn,
    monthly_loss,
    rolling_beta,
    trailing_window_mean,
    winsorize,
)

logger = logging.getLogger(__name__)


def select_liquid_universe(n_firms: int, ranking_start: str, ranking_end: str) -> pd.DataFrame:
    """Non-financial S&P 500 tickers ranked by mean dollar volume near the sample start.

    The paper's Wikipedia-sourced constituent list (Sec. 3.1) carries no
    liquidity ranking, so the reduced run universe is cut by mean dollar
    volume over the first months of the sample period, as recorded in
    `results/RESULTS.md`.
    """
    constituents = load_sp500_constituents()
    candidates = constituents[constituents["sector"] != "Financials"].copy()
    bars, _ = load_ohlcv(candidates["ticker"].tolist(), ranking_start, ranking_end, timeframe="1d")
    dollar_volume = (bars["close"] * bars["volume"]).groupby(bars["ticker"]).mean()
    ranked = dollar_volume.sort_values(ascending=False)
    selected = ranked.head(n_firms).index.tolist()
    return candidates[candidates["ticker"].isin(selected)].set_index("ticker").loc[selected].reset_index()


def _month_index(dates: pd.Series) -> pd.DatetimeIndex:
    periods = pd.DatetimeIndex(dates).to_period("M").to_timestamp()
    return pd.DatetimeIndex(sorted(periods.unique()))


def compute_outcomes_and_exposures(bars: pd.DataFrame, tickers: list[str], market_ticker: str) -> pd.DataFrame:
    """Per-firm monthly outcomes (Eq. 1-2, Sec. 3.3) and exposures (Sec. 3.2) from daily bars."""
    market_rows = bars[bars["ticker"] == market_ticker].set_index("date").sort_index()
    market_daily_return = market_rows["close"].pct_change()
    months = _month_index(bars["date"])
    rows = []
    for ticker in tickers:
        firm_rows = bars[bars["ticker"] == ticker].set_index("date").sort_index()
        if firm_rows.empty:
            continue
        close, high, low, volume = firm_rows["close"], firm_rows["high"], firm_rows["low"], firm_rows["volume"]
        daily_return = close.pct_change()
        loss = monthly_loss(close)
        log_rvol = log_realized_volatility(daily_return, config.RVOL_ANNUALIZATION_FACTOR)
        log_chl = log_chl_spread_monthly(high, low, close)
        rvol12 = trailing_window_mean(log_rvol, config.RVOL_EXPOSURE_WINDOW_MONTHS)
        chl12 = trailing_window_mean(log_chl, config.CHL_EXPOSURE_WINDOW_MONTHS)
        mom = momentum_fn(close)
        month_ends = loss.index
        beta24 = rolling_beta(daily_return, market_daily_return, config.BETA_WINDOW_MONTHS_PRIMARY, month_ends)
        daily_dollar_volume = close * volume
        amihud_daily = daily_return.abs() / daily_dollar_volume.replace(0, np.nan)
        amihud = amihud_daily.resample("ME").mean()
        amihud12 = trailing_window_mean(amihud, config.CHL_EXPOSURE_WINDOW_MONTHS)

        firm_frame = pd.DataFrame({
            "loss": loss, "log_rvol": log_rvol, "log_chl": log_chl,
            "rvol12": rvol12, "chl12": chl12, "momentum": mom,
            "beta24": beta24, "amihud12": amihud12,
        })
        firm_frame.index = firm_frame.index.to_period("M").to_timestamp()
        firm_frame = firm_frame.reindex(months)
        firm_frame.insert(0, "firm_id", ticker)
        firm_frame.insert(1, "month_id", firm_frame.index)
        rows.append(firm_frame.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True)


_ANNUAL_FIELDS = {
    "total_assets": ("balance_sheet", "total_assets"),
    "debt_current": ("balance_sheet", "debt_current"),
    "long_term_debt": ("balance_sheet", "long_term_debt_and_capital_lease_obligations"),
    "cash": ("balance_sheet", "cash_and_equivalents"),
    "net_income": ("income_statement", "net_income_loss_attributable_common_shareholders"),
    "capex": ("cash_flow", "purchase_of_property_plant_and_equipment"),
}


def load_accounting_controls(tickers: list[str], start_date: str, end_date: str, months: pd.DatetimeIndex) -> pd.DataFrame:
    """Annual Massive fundamentals mapped to monthly controls with a 6-month reporting lag (Sec. 3.2).

    leverage = (current + long-term debt) / total assets; cash_assets =
    cash & equivalents / total assets; profitability = net income
    attributable to common shareholders / total assets (Massive has no
    "income before extraordinary items" field; this is the closest available
    proxy, see `results/RESULTS.md`); investment = |capex| / total assets.
    """
    frames = []
    for ticker in tickers:
        try:
            raw = load_annual_fundamentals(ticker, start_date, end_date)
        except MassiveError as exc:
            logger.warning("Fundamentals unavailable for %s: %s", ticker, exc)
            continue
        by_kind = {kind: {row["period_end"]: row for row in rows} for kind, rows in raw.items() if isinstance(rows, list)}
        balance, income, cash_flow = by_kind.get("balance_sheet", {}), by_kind.get("income_statement", {}), by_kind.get("cash_flow", {})
        periods = sorted(set(balance) & set(income) & set(cash_flow))
        if not periods:
            continue
        records = []
        for period in periods:
            total_assets = balance[period].get("total_assets")
            if not total_assets:
                continue
            debt = (balance[period].get("debt_current") or 0.0) + (balance[period].get("long_term_debt_and_capital_lease_obligations") or 0.0)
            cash = balance[period].get("cash_and_equivalents") or 0.0
            net_income = income[period].get("net_income_loss_attributable_common_shareholders") or 0.0
            capex = cash_flow[period].get("purchase_of_property_plant_and_equipment") or 0.0
            effective_month = (pd.Timestamp(period) + pd.DateOffset(months=config.ACCOUNTING_LAG_MONTHS)).to_period("M").to_timestamp()
            records.append({
                "effective_month": effective_month,
                "log_assets": float(np.log(total_assets)),
                "leverage": debt / total_assets,
                "cash_assets": cash / total_assets,
                "profitability": net_income / total_assets,
                "investment": abs(capex) / total_assets,
            })
        annual = pd.DataFrame(records).sort_values("effective_month").set_index("effective_month")
        monthly = annual.reindex(months, method="ffill")
        monthly.insert(0, "firm_id", ticker)
        monthly.insert(1, "month_id", monthly.index)
        frames.append(monthly.reset_index(drop=True))
    if not frames:
        raise MassiveError("No ticker returned usable fundamentals")
    return pd.concat(frames, ignore_index=True)


def generate_esg_block(firm_ids: list[str], months: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    """Synthetic MSCI-style ESG scores keyed by real firm identifiers (see module docstring)."""
    rng = np.random.default_rng(seed)
    n_months = len(months)
    rows = []
    for firm in firm_ids:
        firm_fixed_ias = rng.normal(0.0, 2.0)
        ias = np.clip(firm_fixed_ias + np.cumsum(rng.normal(0.0, 0.3, n_months)), -8.0, 8.0)
        wakis_noise = rng.normal(0.0, 2.6, n_months)
        wakis = 0.2 * ias + np.sqrt(1 - 0.2 ** 2) * wakis_noise
        change_mask = rng.random(n_months) < config.SYNTHETIC_WAKIS_CHANGE_FREQ_PAPER
        wakis_held = np.copy(wakis)
        for t in range(1, n_months):
            if not change_mask[t]:
                wakis_held[t] = wakis_held[t - 1]
        e_pillar = ias + rng.normal(0.0, 1.5, n_months)
        s_pillar = ias + rng.normal(0.0, 1.5, n_months)
        g_pillar = ias + rng.normal(0.0, 1.5, n_months)
        rows.append(pd.DataFrame({
            "firm_id": firm, "month_id": months,
            "esg_ias": ias, "esg_wakis": wakis_held,
            "esg_e": e_pillar, "esg_s": s_pillar, "esg_g": g_pillar,
        }))
    return pd.concat(rows, ignore_index=True)


def build_real_firm_month_panel(
    n_firms: int, market_months: pd.DataFrame, seed: int = config.SYNTHETIC_SEED,
) -> tuple[pd.DataFrame, dict]:
    """Assemble the firm-month panel from real market and accounting data plus synthetic ESG.

    Raises ``DataloaderError``/``MassiveError`` if the market or accounting
    data genuinely cannot be loaded; the caller decides whether to fall back.
    """
    provenance: dict = {}
    ranking_start, ranking_end = config.SAMPLE_START, pd.Timestamp(config.SAMPLE_START) + pd.DateOffset(months=3)
    universe = select_liquid_universe(n_firms, ranking_start, ranking_end.strftime("%Y-%m-%d"))
    tickers = universe["ticker"].tolist()
    provenance["universe_cut_rule"] = (
        f"non-financial S&P 500 (Wikipedia), top {n_firms} by mean dollar volume "
        f"over {ranking_start}..{ranking_end.date()}"
    )
    provenance["universe_tickers_realized"] = len(tickers)

    bars, loader_meta = load_ohlcv(tickers + [config.MARKET_INDEX_TICKER], config.SAMPLE_START, config.SAMPLE_END, timeframe="1d")
    provenance["ohlcv_failed"] = loader_meta.get("failed")
    provenance["ohlcv_non_massive_source"] = loader_meta.get("non_massive_source")
    loaded_tickers = [t for t in tickers if t not in (loader_meta.get("failed") or {})]

    market_frame = compute_outcomes_and_exposures(bars, loaded_tickers, config.MARKET_INDEX_TICKER)
    months = pd.DatetimeIndex(sorted(market_frame["month_id"].unique()))

    accounting = load_accounting_controls(loaded_tickers, "2016-01-01", config.SAMPLE_END, months)
    provenance["tickers_with_fundamentals"] = accounting["firm_id"].nunique()

    esg = generate_esg_block(loaded_tickers, months, seed)

    panel = market_frame.merge(accounting, on=["firm_id", "month_id"], how="left")
    panel = panel.merge(esg, on=["firm_id", "month_id"], how="left")
    panel = panel.merge(market_months.rename(columns={"market_return": "market_return"}), on="month_id", how="inner")
    panel = panel.merge(universe.rename(columns={"ticker": "firm_id"})[["firm_id", "sector"]], on="firm_id", how="left")

    for column in ("log_assets", "leverage", "cash_assets", "profitability", "investment", "momentum"):
        panel[column] = winsorize(panel[column], config.WINSORIZE_LOWER_PCTL, config.WINSORIZE_UPPER_PCTL)

    essential = ["loss", "log_rvol", "log_chl", "beta24", "rvol12", "chl12", "log_assets", "leverage",
                 "cash_assets", "profitability", "investment", "momentum"]
    panel = panel.dropna(subset=essential).reset_index(drop=True)
    provenance["n_firm_months_real"] = int(len(panel))
    provenance["n_firms_real"] = int(panel["firm_id"].nunique())
    return panel, provenance
