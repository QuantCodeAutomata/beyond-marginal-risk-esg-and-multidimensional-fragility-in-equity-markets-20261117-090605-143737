"""Experiment 5: Fragility Ladder and Threshold Sensitivity Diagnostics.

Reproduces the paper's "ladder" of multidimensional fragility (Table A7) and
the frozen-threshold sensitivity grid (Table A8), using the fully
risk-adjusted Equation 7 specification throughout (exp_5 methodology).

Part A (the ladder): the fully risk-adjusted Joint80-style model is
estimated for seven outcomes built from the same sample -- three
one-dimensional tail indicators (Return, RVOL, CHL tails at the pooled 80th
percentile), three two-dimensional pairwise intersections (Return+RVOL,
Return+CHL, RVOL+CHL), and the three-dimensional Joint80 indicator. Holm's
step-down adjustment (Sec. 4, methodology step 4/8) is applied within the
1D family (3 tests) and, separately, within the 2D family (3 tests); the 3D
Joint80 test is reported as a single test (Holm with a family of size 1 is a
no-op, so its raw and Holm-adjusted p-values coincide).

Part B (threshold grid): the ALL-THREE ("Joint") indicator is reconstructed
at pooled percentile thresholds q in {0.70, 0.75, 0.80, 0.85, 0.90} (the
paper's threshold-sensitivity grid, Sec. 3.4/Table A8) and the fully
adjusted Equation 7 model (all three exposure blocks) is re-estimated at
each threshold. The ESG x Down estimate/SE/raw p-value and the complete
fitted ESG slope at a -10% market decline are reported for every threshold
in the grid, with no threshold singled out as "preferred" and no
monotonicity imposed across the grid.

Usage: ``python -m exp.exp5 [--full]``. ``--full`` uses the paper's panel
size (``config.SYNTHETIC_N_FIRMS_PAPER``) instead of the reduced run
configuration (``config.SYNTHETIC_N_FIRMS_RUN``; see results/RESULTS.md for
the runtime that motivated the reduction).
"""
from __future__ import annotations

import sys
import logging
from typing import Sequence

import pandas as pd

from src import config
from src.pipeline import build_experiment_panel
from src.panel_pipeline import build_model_frame, regressors_equation7, pairwise_states, joint_state
from src.regression import fit_fe_panel, linear_contrast, complete_slope, PanelRegressionResult
from src.inference import holm_adjust
from src.reporting import save_table, save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LADDER_1D = ["tail_return", "tail_rvol", "tail_chl"]
LADDER_2D = ["pair_return_rvol", "pair_return_chl", "pair_rvol_chl"]
LADDER_3D = ["joint80"]

TABLE_COLUMNS = [
    "outcome", "prevalence", "nobs", "esg_down_estimate", "esg_down_se",
    "esg_down_raw_p", "esg_down_holm_p", "esg_up_estimate", "esg_up_se",
    "down_up_contrast_estimate", "down_up_contrast_raw_p",
]


def _fit_outcome(df: pd.DataFrame, outcome: str, regressors: list[str]) -> dict:
    """Fit Equation 7 for one ladder outcome and extract the three contrasts used throughout exp_5."""
    fit: PanelRegressionResult = fit_fe_panel(df, outcome=outcome, regressors=regressors)
    down_est, down_se, down_p = linear_contrast(fit, "esg_down")
    up_est, up_se, up_p = linear_contrast(fit, "esg_up")
    contrast_est, contrast_se, contrast_p = linear_contrast(fit, "esg_down", "esg_up", weight_b=-1.0)
    return {
        "outcome": outcome,
        "prevalence": float(df[outcome].mean()),
        "nobs": fit.nobs,
        "esg_down_estimate": down_est,
        "esg_down_se": down_se,
        "esg_down_raw_p": down_p,
        "esg_up_estimate": up_est,
        "esg_up_se": up_se,
        "esg_up_raw_p": up_p,
        "down_up_contrast_estimate": contrast_est,
        "down_up_contrast_raw_p": contrast_p,
    }


def build_ladder_family(df: pd.DataFrame, outcomes: Sequence[str], regressors: list[str]) -> pd.DataFrame:
    """Fit Equation 7 for every outcome in a ladder family and Holm-adjust the ESG x Down p-values."""
    rows = [_fit_outcome(df, outcome, regressors) for outcome in outcomes]
    holm = holm_adjust([row["esg_down_raw_p"] for row in rows], alpha=config.HOLM_ALPHA)
    for row, holm_p in zip(rows, holm["holm_p"].tolist()):
        row["esg_down_holm_p"] = holm_p
    return pd.DataFrame(rows)[TABLE_COLUMNS]


def build_threshold_grid(df: pd.DataFrame, regressors: list[str], grid: Sequence[float]) -> pd.DataFrame:
    """Re-estimate Equation 7 for the ALL-THREE indicator at each pooled-percentile threshold."""
    rows = []
    for q in grid:
        joint_q, _ = joint_state(df, quantile=q)
        df_q = df.copy()
        df_q["joint_q"] = joint_q
        fit_q = fit_fe_panel(df_q, outcome="joint_q", regressors=regressors)
        est, se, p_value = linear_contrast(fit_q, "esg_down")
        slope = complete_slope(fit_q, base_coef="esg", down_coef="esg_down", market_states=[-0.10])
        rows.append({
            "quantile": q,
            "n_events": int(df_q["joint_q"].sum()),
            "prevalence": float(df_q["joint_q"].mean()),
            "esg_down_estimate": est,
            "esg_down_se": se,
            "esg_down_raw_p": p_value,
            "complete_slope_minus10pct": float(slope.loc[slope["market_state"] == -0.10, "slope"].iloc[0]),
        })
    return pd.DataFrame(rows)


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

    tails = pairwise_states(df, quantile=config.JOINT_THRESHOLD_PRIMARY)
    for col in tails.columns:
        df[col] = tails[col]
    df["joint80"], thresholds80 = joint_state(df, quantile=config.JOINT_THRESHOLD_PRIMARY)

    regs = regressors_equation7(include_blocks=None)

    ladder_1d = build_ladder_family(df, LADDER_1D, regs)
    ladder_2d = build_ladder_family(df, LADDER_2D, regs)
    ladder_3d = build_ladder_family(df, LADDER_3D, regs)

    threshold_grid = build_threshold_grid(df, regs, config.JOINT_THRESHOLD_GRID)

    save_table(ladder_1d, "exp5_ladder_1d")
    save_table(ladder_2d, "exp5_ladder_2d")
    save_table(ladder_3d, "exp5_ladder_3d_joint80")
    save_table(threshold_grid, "exp5_threshold_grid")
    metadata_path = save_json(
        {"provenance": provenance, "thresholds80": {k: float(v) for k, v in thresholds80.items()}, "n_firms": n_firms, "full_run": full},
        "exp5_run_metadata",
    )

    logger.info("exp5 Part A: 1D tail ladder (nobs=%d):\n%s", int(ladder_1d["nobs"].iloc[0]), ladder_1d.to_string())
    logger.info("exp5 Part A: 2D pairwise ladder:\n%s", ladder_2d.to_string())
    logger.info("exp5 Part A: 3D Joint80:\n%s", ladder_3d.to_string())
    logger.info("exp5 Part B: threshold-sensitivity grid:\n%s", threshold_grid.to_string())
    logger.info("exp5 run metadata written to %s", metadata_path)

    return {
        "ladder_1d": ladder_1d,
        "ladder_2d": ladder_2d,
        "ladder_3d_joint80": ladder_3d,
        "threshold_grid": threshold_grid,
        "provenance": provenance,
        "thresholds80": thresholds80,
    }


if __name__ == "__main__":
    main(full="--full" in sys.argv)
