"""Experiment 6: Entry/Persistence Dynamics and Compositional Decomposition Diagnostics.

Reproduces the paper's Table A9 / exp_6 methodology in three parts, all on the
primary Joint80 sample (Eq. 3, frozen pooled 80th-percentile thresholds).

Part A (entry/persistence dynamics): the Joint80 state transition is split
into an "Entry" subsample (observations whose Joint80 value in the preceding
*consecutive* calendar month was 0) and a "Persistence" subsample (preceding
month's Joint80 was 1). The fully risk-adjusted Equation 7 model is estimated
separately on each subsample, and the two ESG x Down p-values form a 2-test
Holm family. Only conventional two-way-clustered inference is used here: the
paper additionally reports a wild-cluster-bootstrap p-value of 0.109 for the
Entry ESG x Down coefficient, but that bootstrap (Cameron, Gelbach and Miller
[2008]; FNW11 null imposition) is time-expensive and is deliberately *not*
run in this script -- no wild p-value is fabricated or reported here.

Part B (pillar decomposition): for each of the three marginal channels
(return -> loss, volatility -> log_rvol, liquidity -> log_chl) the
channel-specific Equation 5 model is re-estimated three times, replacing the
ESG regressor with the Environmental, Social and Governance pillar scores in
turn (9 models total). The 9 ESG x Down p-values form a single Holm family.

Part C (within-firm decomposition): for each marginal channel, the IAS score
is decomposed into a firm-specific time-invariant mean and a within-firm
deviation from that mean, each entered with its own Down/Up interactions
(6 regressors replacing the single esg/esg_down/esg_up triple). The 3
firm-mean-Down and 3 within-deviation-Down p-values (6 tests total) form a
single Holm family.

Usage: ``python -m exp.exp6 [--full]``. ``--full`` uses the paper's panel
size (``config.SYNTHETIC_N_FIRMS_PAPER``) instead of the reduced run
configuration (``config.SYNTHETIC_N_FIRMS_RUN``; see results/RESULTS.md for
the runtime that motivated the reduction).
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import numpy as np
import pandas as pd

from src import config
from src.pipeline import build_experiment_panel
from src.panel_pipeline import (
    build_model_frame,
    regressors_equation5,
    regressors_equation7,
    joint_state,
    CONTROL_COLS,
    EXPOSURE_BLOCKS,
    OUTCOME_OF_CHANNEL,
)
from src.regression import fit_fe_panel, linear_contrast, PanelRegressionResult
from src.inference import holm_adjust
from src.reporting import save_table, save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHANNELS = ["return", "volatility", "liquidity"]
PILLARS = ["e", "s", "g"]

ENTRY_PERSISTENCE_COLUMNS = [
    "group", "nobs", "n_firms", "esg_down_estimate", "esg_down_se",
    "esg_down_raw_p", "esg_down_holm_p", "esg_up_estimate", "down_up_contrast_raw_p",
]
PILLAR_COLUMNS = [
    "channel", "pillar", "outcome", "nobs", "esg_down_estimate", "esg_down_se",
    "esg_down_raw_p", "esg_down_holm_p",
]
WITHIN_FIRM_COLUMNS = [
    "channel", "outcome", "nobs",
    "firm_mean_down_estimate", "firm_mean_down_se", "firm_mean_down_raw_p", "firm_mean_down_holm_p",
    "within_dev_down_estimate", "within_dev_down_se", "within_dev_down_raw_p", "within_dev_down_holm_p",
]


def _holm_with_nan(raw_p: list[float], alpha: float = config.HOLM_ALPHA) -> list[float]:
    """Holm step-down over the finite entries of ``raw_p``, leaving NaNs as NaN.

    A missing raw p-value (e.g. the Persistence fit could not be estimated)
    is excluded from the Holm family rather than being treated as a test, so
    the family size shrinks to the number of models that actually fit.
    """
    raw_p = np.asarray(raw_p, dtype=float)
    out = np.full(raw_p.shape, np.nan)
    finite = np.isfinite(raw_p)
    if finite.sum() > 0:
        out[finite] = holm_adjust(raw_p[finite], alpha=alpha)["holm_p"].to_numpy()
    return out.tolist()


def _fit_entry_persistence_group(df_group: pd.DataFrame, regressors: list[str], label: str) -> dict[str, Any]:
    """Fit Equation 7 (Joint80 outcome) on one Entry/Persistence subsample.

    Returns NaN contrasts with a diagnostic note instead of raising when the
    subsample is too small or degenerate for ``fit_fe_panel`` to estimate
    (singular design, too few clusters, etc.) -- Part A's instructions
    explicitly ask for this fallback on the Persistence group.
    """
    try:
        fit: PanelRegressionResult = fit_fe_panel(df_group, outcome="joint80", regressors=regressors)
    except Exception as exc:  # noqa: BLE001 - deliberately broad per task instructions
        logger.warning(
            "%s Equation 7 fit failed (%s); reporting NaN contrasts. Note: insufficient "
            "persistence-state observations in this run's synthetic/real-state sample.",
            label, exc,
        )
        return {
            "group": label,
            "nobs": int(len(df_group)),
            "n_firms": int(df_group["firm_id"].nunique()),
            "esg_down_estimate": np.nan,
            "esg_down_se": np.nan,
            "esg_down_raw_p": np.nan,
            "esg_up_estimate": np.nan,
            "down_up_contrast_raw_p": np.nan,
            "note": "insufficient persistence-state observations in this run's synthetic/real-state sample",
        }
    down_est, down_se, down_p = linear_contrast(fit, "esg_down")
    up_est, _, _ = linear_contrast(fit, "esg_up")
    _, _, contrast_p = linear_contrast(fit, "esg_down", "esg_up", weight_b=-1.0)
    logger.info("%s Equation 7 fit: nobs=%d, n_firms=%d", label, fit.nobs, fit.n_entities)
    return {
        "group": label,
        "nobs": fit.nobs,
        "n_firms": fit.n_entities,
        "esg_down_estimate": down_est,
        "esg_down_se": down_se,
        "esg_down_raw_p": down_p,
        "esg_up_estimate": up_est,
        "down_up_contrast_raw_p": contrast_p,
        "note": "",
    }


def build_entry_persistence_table(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Part A: Entry and Persistence Equation 7 fits, with a 2-test Holm family."""
    df_sorted = df.sort_values(["firm_id", "month_id"]).reset_index(drop=True)
    df_sorted["joint80_lag"] = df_sorted.groupby("firm_id")["joint80"].shift(1)
    df_sorted["month_gap_days"] = df_sorted.groupby("firm_id")["month_id"].diff().dt.days
    consecutive = df_sorted["month_gap_days"].between(25, 35)

    entry_df = df_sorted[consecutive & (df_sorted["joint80_lag"] == 0)].copy()
    persistence_df = df_sorted[consecutive & (df_sorted["joint80_lag"] == 1)].copy()
    logger.info(
        "Part A subsamples: entry nobs=%d (%d firms), persistence nobs=%d (%d firms)",
        len(entry_df), entry_df["firm_id"].nunique(), len(persistence_df), persistence_df["firm_id"].nunique(),
    )

    regs7 = regressors_equation7(include_blocks=None)
    entry_row = _fit_entry_persistence_group(entry_df, regs7, "entry")
    persistence_row = _fit_entry_persistence_group(persistence_df, regs7, "persistence")

    holm_p = _holm_with_nan([entry_row["esg_down_raw_p"], persistence_row["esg_down_raw_p"]])
    entry_row["esg_down_holm_p"] = holm_p[0]
    persistence_row["esg_down_holm_p"] = holm_p[1]

    table = pd.DataFrame([entry_row, persistence_row])[ENTRY_PERSISTENCE_COLUMNS]
    counts = {"n_entry_obs": int(len(entry_df)), "n_persistence_obs": int(len(persistence_df))}
    return table, counts


def build_pillar_decomposition_table(raw_panel: pd.DataFrame) -> pd.DataFrame:
    """Part B: channel x pillar Equation 5 fits, with a single 9-test Holm family."""
    rows: list[dict[str, Any]] = []
    for channel in CHANNELS:
        outcome = OUTCOME_OF_CHANNEL[channel]
        regressors = regressors_equation5(channel)
        for pillar in PILLARS:
            df_pillar = build_model_frame(raw_panel, esg_col=f"esg_{pillar}")
            fit = fit_fe_panel(df_pillar, outcome=outcome, regressors=regressors)
            est, se, p_value = linear_contrast(fit, "esg_down")
            logger.info(
                "Part B channel=%s pillar=%s outcome=%s: nobs=%d, n_firms=%d",
                channel, pillar, outcome, fit.nobs, fit.n_entities,
            )
            rows.append({
                "channel": channel,
                "pillar": pillar,
                "outcome": outcome,
                "nobs": fit.nobs,
                "esg_down_estimate": est,
                "esg_down_se": se,
                "esg_down_raw_p": p_value,
            })

    holm_p = _holm_with_nan([row["esg_down_raw_p"] for row in rows])
    for row, p_adj in zip(rows, holm_p):
        row["esg_down_holm_p"] = p_adj
    return pd.DataFrame(rows)[PILLAR_COLUMNS]


def build_within_firm_decomposition_table(raw_panel: pd.DataFrame) -> pd.DataFrame:
    """Part C: per-channel within-firm/between-firm IAS decomposition, 6-test Holm family.

    Holm order documented here and reused below: [firm_mean_return,
    within_dev_return, firm_mean_vol, within_dev_vol, firm_mean_liq,
    within_dev_liq] -- i.e. the two tests of one channel are adjacent, in
    the fixed channel order ``return, volatility, liquidity``.
    """
    raw_panel = raw_panel.copy()
    raw_panel["firm_mean_ias"] = raw_panel.groupby("firm_id")["esg_ias"].transform("mean")
    raw_panel["within_dev_ias"] = raw_panel["esg_ias"] - raw_panel["firm_mean_ias"]

    df_c = build_model_frame(raw_panel, esg_col="esg_ias")
    df_c["firm_mean_down"] = raw_panel["firm_mean_ias"] * df_c["down"]
    df_c["firm_mean_up"] = raw_panel["firm_mean_ias"] * df_c["up"]
    df_c["within_dev_down"] = raw_panel["within_dev_ias"] * df_c["down"]
    df_c["within_dev_up"] = raw_panel["within_dev_ias"] * df_c["up"]
    df_c["firm_mean"] = raw_panel["firm_mean_ias"]
    df_c["within_dev"] = raw_panel["within_dev_ias"]

    rows: list[dict[str, Any]] = []
    for channel in CHANNELS:
        outcome = OUTCOME_OF_CHANNEL[channel]
        block = EXPOSURE_BLOCKS[channel]
        regs_c = [
            "firm_mean", "firm_mean_down", "firm_mean_up",
            "within_dev", "within_dev_down", "within_dev_up",
            *CONTROL_COLS, block, f"{block}_down", f"{block}_up",
        ]
        # `firm_mean` is time-invariant per firm, so PanelOLS's entity (firm)
        # fixed effects fully absorb it (expected `AbsorbingEffectWarning`,
        # drop_absorbed=True silently removes it). The Down/Up-interacted
        # versions, `firm_mean_down`/`firm_mean_up`, vary over time through
        # the market state and are not absorbed, so the contrast of interest
        # (firm_mean_down) remains identified.
        fit = fit_fe_panel(df_c, outcome=outcome, regressors=regs_c)
        est_fm, se_fm, p_fm = linear_contrast(fit, "firm_mean_down")
        est_wd, se_wd, p_wd = linear_contrast(fit, "within_dev_down")
        logger.info(
            "Part C channel=%s outcome=%s: nobs=%d, n_firms=%d", channel, outcome, fit.nobs, fit.n_entities,
        )
        rows.append({
            "channel": channel,
            "outcome": outcome,
            "nobs": fit.nobs,
            "firm_mean_down_estimate": est_fm,
            "firm_mean_down_se": se_fm,
            "firm_mean_down_raw_p": p_fm,
            "within_dev_down_estimate": est_wd,
            "within_dev_down_se": se_wd,
            "within_dev_down_raw_p": p_wd,
        })

    # Single 6-test Holm family built from the 3 firm_mean_down and 3 within_dev_down
    # raw p-values together, in the documented order above.
    ordered_raw_p: list[float] = []
    for row in rows:
        ordered_raw_p.append(row["firm_mean_down_raw_p"])
        ordered_raw_p.append(row["within_dev_down_raw_p"])
    holm_p = _holm_with_nan(ordered_raw_p)
    for i, row in enumerate(rows):
        row["firm_mean_down_holm_p"] = holm_p[2 * i]
        row["within_dev_down_holm_p"] = holm_p[2 * i + 1]
    return pd.DataFrame(rows)[WITHIN_FIRM_COLUMNS]


def main(full: bool = False) -> dict:
    n_firms = config.SYNTHETIC_N_FIRMS_PAPER if full else config.SYNTHETIC_N_FIRMS_RUN

    raw_panel, provenance = build_experiment_panel(
        n_firms=n_firms,
        esg_down_effect=None,
        use_real_market_state=True,
    )
    logger.info(
        "Panel realized: %d firm-months, %d firms, %d months (%s)",
        len(raw_panel), raw_panel["firm_id"].nunique(), raw_panel["month_id"].nunique(),
        provenance.get("market_state_source"),
    )

    df = build_model_frame(raw_panel, esg_col="esg_ias")
    df["joint80"], thresholds80 = joint_state(df, quantile=config.JOINT_THRESHOLD_PRIMARY)

    entry_persistence_table, counts = build_entry_persistence_table(df)
    pillar_table = build_pillar_decomposition_table(raw_panel)
    within_firm_table = build_within_firm_decomposition_table(raw_panel)

    save_table(entry_persistence_table, "exp6_entry_persistence")
    save_table(pillar_table, "exp6_pillar_decomposition")
    save_table(within_firm_table, "exp6_within_firm_decomposition")
    metadata_path = save_json(
        {
            "provenance": provenance,
            "thresholds80": {k: float(v) for k, v in thresholds80.items()},
            "n_firms": n_firms,
            "full_run": full,
            "n_entry_obs": counts["n_entry_obs"],
            "n_persistence_obs": counts["n_persistence_obs"],
            "within_firm_holm_order": [
                "firm_mean_return", "within_dev_return",
                "firm_mean_vol", "within_dev_vol",
                "firm_mean_liq", "within_dev_liq",
            ],
            "wild_bootstrap_note": (
                "The Entry wild-cluster-bootstrap p-value reported in the paper (0.109) is NOT "
                "reproduced in this script; only conventional two-way-clustered inference is used "
                "here because the wild-cluster bootstrap is time-expensive (see src/inference.py "
                "wild_cluster_bootstrap_panel for the implementation used elsewhere in the repo)."
            ),
        },
        "exp6_run_metadata",
    )

    logger.info("exp6 Part A: entry/persistence dynamics:\n%s", entry_persistence_table.to_string())
    logger.info("exp6 Part B: pillar decomposition:\n%s", pillar_table.to_string())
    logger.info("exp6 Part C: within-firm decomposition:\n%s", within_firm_table.to_string())
    logger.info("exp6 run metadata written to %s", metadata_path)

    return {
        "entry_persistence": entry_persistence_table,
        "pillar_decomposition": pillar_table,
        "within_firm_decomposition": within_firm_table,
        "provenance": provenance,
        "thresholds80": thresholds80,
    }


if __name__ == "__main__":
    main(full="--full" in sys.argv)
