"""Synthetic firm-month panel calibrated to the paper's descriptive statistics.

MSCI ESG scores (IAS, WAKIS, E/S/G pillars) have no public REST API (see
data/data_loader.py and results/RESULTS.md); this generator produces a panel
with the paper's reported sample dimensions, ESG summary statistics
(Sec. 3.2), and IAS/WAKIS co-movement (exp_4 data_requirements) so the
regression, Holm, wild-bootstrap and decomposition machinery in src/ can be
exercised end to end and validated against known injected effects. It is used
for methodology demonstration and for tests, never to stand in for the
paper's real-data headline numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def generate_market_months(n_months: int, seed: int) -> pd.DataFrame:
    """Monthly S&P 500 index returns with the paper's reported Down/Up month split
    (27 negative, 56 positive out of 83 months, Sec. 4.1 data_requirements)."""
    rng = np.random.default_rng(seed)
    months = pd.period_range(config.SAMPLE_START, periods=n_months, freq="M")
    share_down = config.N_DOWN_MONTHS_PAPER / config.N_MONTHS_PAPER
    is_down = rng.random(n_months) < share_down
    market_return = np.where(
        is_down, -rng.gamma(2.0, 0.02, n_months), rng.gamma(2.0, 0.015, n_months)
    )
    return pd.DataFrame({"month_id": months, "market_return": market_return})


def generate_panel(
    n_firms: int = config.SYNTHETIC_N_FIRMS_RUN,
    n_months: int = config.N_MONTHS_PAPER,
    seed: int = config.SYNTHETIC_SEED,
    esg_down_effect: dict[str, float] | None = None,
    market_months: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build an unbalanced firm-month panel with ESG, controls, exposures and outcomes.

    ``esg_down_effect`` injects a known theta_D per channel
    (``{"loss": ..., "log_rvol": ..., "log_chl": ...}``) so regression tests
    can check that `src.regression.fit_fe_panel` recovers the planted sign
    and order of magnitude (no look-ahead, Equation 5 structure).

    ``market_months`` optionally supplies a real ``month_id``/``market_return``
    frame (e.g. the S&P 500 index series loaded by
    `src.pipeline.load_real_market_months`) so the Down/Up split used to drive
    the synthetic firm-level outcomes matches the paper's actual sample
    months instead of a simulated market.
    """
    rng = np.random.default_rng(seed)
    esg_down_effect = esg_down_effect or {"loss": 0.0, "log_rvol": 0.0, "log_chl": 0.0}
    if market_months is not None:
        market = market_months.reset_index(drop=True)
        n_months = len(market)
    else:
        market = generate_market_months(n_months, seed)

    firm_ids = [f"F{i:04d}" for i in range(n_firms)]
    firm_fixed_ias = rng.normal(0.0, 2.0, n_firms)          # time-invariant firm component
    firm_fixed_level = rng.normal(0.0, 1.0, n_firms)        # firm fixed effect in outcomes

    rows = []
    for f_idx, firm in enumerate(firm_ids):
        ias = firm_fixed_ias[f_idx] + np.cumsum(rng.normal(0.0, 0.3, n_months))
        ias = np.clip(ias, -8.0, 8.0)
        # WAKIS: weak overall correlation with IAS level (paper: 0.196), refreshed more often.
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

        log_assets = rng.normal(8.0, 1.5) + np.cumsum(rng.normal(0.0, 0.02, n_months))
        leverage = np.clip(rng.normal(0.3, 0.15, n_months), 0.0, 1.5)
        cash_assets = np.clip(rng.normal(0.1, 0.08, n_months), 0.0, 1.0)
        profitability = rng.normal(0.08, 0.06, n_months)
        investment = np.clip(rng.normal(0.05, 0.03, n_months), -0.1, 0.5)
        momentum_series = rng.normal(0.05, 0.15, n_months)

        beta24 = np.clip(rng.normal(1.0, 0.35, n_months), -1.0, 3.0)
        rvol12 = np.clip(rng.normal(-1.3, 0.3, n_months), -3.0, 1.0)
        chl12 = np.clip(rng.normal(-5.0, 0.4, n_months), -8.0, -1.0)
        amihud12 = np.clip(rng.lognormal(-9.0, 1.0, n_months), 1e-9, 1e-2)

        down = (-market["market_return"]).clip(lower=0.0).to_numpy()
        up = market["market_return"].clip(lower=0.0).to_numpy()

        idio = rng.normal(0.0, 0.05, n_months)
        loss = (
            firm_fixed_level[f_idx] * 0.01
            + 0.4 * down - 0.1 * up
            + 0.002 * beta24 * down
            + esg_down_effect["loss"] * ias * down / config.ESG_POOLED_SD
            + idio
        )
        log_rvol = (
            -1.0 + 0.8 * down + 0.3 * rvol12 * 0
            + 0.05 * beta24 * down
            + esg_down_effect["log_rvol"] * ias * down / config.ESG_POOLED_SD
            + rng.normal(0.0, 0.2, n_months)
        )
        log_chl = (
            -5.0 + 0.6 * down
            + esg_down_effect["log_chl"] * ias * down / config.ESG_POOLED_SD
            + rng.normal(0.0, 0.25, n_months)
        )

        firm_df = pd.DataFrame({
            "firm_id": firm,
            "month_id": market["month_id"].to_numpy(),
            "market_return": market["market_return"].to_numpy(),
            "esg_ias": ias,
            "esg_wakis": wakis_held,
            "esg_e": e_pillar,
            "esg_s": s_pillar,
            "esg_g": g_pillar,
            "loss": loss,
            "log_rvol": log_rvol,
            "log_chl": log_chl,
            "log_assets": log_assets,
            "leverage": leverage,
            "cash_assets": cash_assets,
            "profitability": profitability,
            "investment": investment,
            "momentum": momentum_series,
            "beta24": beta24,
            "rvol12": rvol12,
            "chl12": chl12,
            "amihud12": amihud12,
            "sector": "Industrials" if f_idx % 5 else "Real Estate",
        })
        rows.append(firm_df)

    panel = pd.concat(rows, ignore_index=True)
    # linearmodels.PanelOLS requires a numeric or date-like time index; keep
    # month_id as a Timestamp (not a string) so it can be used directly.
    if isinstance(panel["month_id"].dtype, pd.PeriodDtype):
        panel["month_id"] = panel["month_id"].dt.to_timestamp()
    return panel
