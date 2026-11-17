"""Experiment 3: Joint80 Model, Exact State Decomposition, and Construct Validations.

Reproduces the fully adjusted Joint80 linear probability model of Equation 7
(paper Table 3 / Table A3) and the construct-validity checks that accompany
it (paper Table A4-style decomposition and robustness panels). Runs:
  1. The primary Joint80 model: firm + month fixed effects, six lagged
     controls, and all three ex ante exposure blocks (beta24, rvol12, chl12)
     with Down/Up interactions, two-way clustered inference. Extracts
     esg_down, esg_up and the Down-minus-Up contrast.
  2. The complete fitted ESG slope beta_J + d*theta_{D,J} on the paper's
     market-state grid (Table A3), in raw units and scaled by the pooled
     ESG SD (probability percentage points per 1 SD of ESG).
  3. An exact OLS-linearity decomposition of the inclusive Return+RVOL state
     PairRV80 into the mutually exclusive RVOnly state and the Joint80 state
     (PairRV80 = RVOnly + Joint80 by construction), each refit under
     Equation 7, with an approximate numeric check that
     beta(PairRV80) ~= beta(RVOnly) + beta(Joint80) for esg_down.
  4. A stacked-system cross-equation contrast of theta_D^Joint80 vs
     theta_D^RVOnly.
  5. A sector x month fixed-effect validation of the Joint80 model.
  6. An Amihud-illiquidity alternative Joint80 construct validation.

Usage: ``python -m exp.exp3 [--full]``. ``--full`` uses the paper's panel
size (config.SYNTHETIC_N_FIRMS_PAPER) instead of the reduced run
configuration (config.SYNTHETIC_N_FIRMS_RUN); see results/RESULTS.md for the
runtime trade-off. No wild-cluster bootstrap is run here: the stacked system
alone already doubles the observation count relative to exp1/exp2, and a
Rademacher wild-cluster bootstrap on top of that would be time-prohibitive
for this run, so only conventional two-way-clustered inference is reported
for every table in this script.
"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from src import config
from src.decomposition import build_stacked_frame, cross_equation_contrast, fit_stacked_system
from src.panel_pipeline import build_model_frame, joint_state, pairwise_states, regressors_equation7
from src.pipeline import build_experiment_panel
from src.regression import PanelRegressionResult, complete_slope, fit_fe_panel, linear_contrast
from src.reporting import save_json, save_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_panel(n_firms: int) -> tuple[pd.DataFrame, dict, dict]:
    """Assemble the model frame and the three Joint/Pair/RVOnly state indicators."""
    raw_panel, provenance = build_experiment_panel(
        n_firms=n_firms, esg_down_effect=None, use_real_market_state=True,
    )
    df = build_model_frame(raw_panel, esg_col="esg_ias")

    tails = pairwise_states(df, quantile=config.JOINT_THRESHOLD_PRIMARY)
    df["joint80"], thresholds = joint_state(df, quantile=config.JOINT_THRESHOLD_PRIMARY)
    df["pair_rv80"] = tails["pair_return_rvol"]
    df["rvonly"] = tails["pair_return_rvol"] * (1 - tails["tail_chl"])

    # Exact state-decomposition identity (methodology step 3): the inclusive
    # Return+RVOL tail is the disjoint union of "Return+RVOL but not CHL"
    # (RVOnly) and "Return+RVOL+CHL" (Joint80).
    assert (df["pair_rv80"] == (df["rvonly"] + df["joint80"])).all(), (
        "PairRV80 must equal RVOnly + Joint80 for every observation"
    )

    df["tail_return"] = tails["tail_return"]
    df["tail_rvol"] = tails["tail_rvol"]
    df["tail_chl"] = tails["tail_chl"]

    thresholds_out = {k: float(v) for k, v in thresholds.items()}
    return df, provenance, thresholds_out


def run_primary_joint80(df: pd.DataFrame, regs: list[str]) -> tuple[PanelRegressionResult, pd.DataFrame]:
    """Step 1: the fully adjusted Joint80 model (Equation 7) and its primary table."""
    fit = fit_fe_panel(df, outcome="joint80", regressors=regs)
    down_est, down_se, down_p = linear_contrast(fit, "esg_down")
    up_est, up_se, up_p = linear_contrast(fit, "esg_up")
    contrast_est, contrast_se, contrast_p = linear_contrast(fit, "esg_down", "esg_up", weight_b=-1.0)
    table = pd.DataFrame([{
        "nobs": fit.nobs,
        "n_firms": fit.n_entities,
        "esg_down_estimate": down_est,
        "esg_down_se": down_se,
        "esg_down_raw_p": down_p,
        "esg_up_estimate": up_est,
        "esg_up_se": up_se,
        "esg_up_raw_p": up_p,
        "down_up_contrast_estimate": contrast_est,
        "down_up_contrast_raw_p": contrast_p,
    }])
    return fit, table


def run_complete_slopes(fit: PanelRegressionResult) -> pd.DataFrame:
    """Step 2: complete fitted ESG slope on the paper's market-state grid (Table A3)."""
    slopes = complete_slope(
        fit, base_coef="esg", down_coef="esg_down", market_states=list(config.MARKET_STATE_GRID),
    )
    slopes["slope_per_esg_sd"] = slopes["slope"] * config.ESG_POOLED_SD
    return slopes


def run_decomposition(df: pd.DataFrame, regs: list[str]) -> tuple[dict[str, PanelRegressionResult], pd.DataFrame]:
    """Step 3: refit Equation 7 on PairRV80, RVOnly and Joint80, report esg_down for all three."""
    fits = {
        "pair_rv80": fit_fe_panel(df, outcome="pair_rv80", regressors=regs),
        "rvonly": fit_fe_panel(df, outcome="rvonly", regressors=regs),
        "joint80": fit_fe_panel(df, outcome="joint80", regressors=regs),
    }
    labels = {
        "pair_rv80": "pair_return_rvol(inclusive)",
        "rvonly": "rvonly(exclusive)",
        "joint80": "joint80(exclusive)",
    }
    rows = []
    for key, fit in fits.items():
        est, se, p_value = linear_contrast(fit, "esg_down")
        rows.append({
            "state": labels[key],
            "outcome": key,
            "nobs": fit.nobs,
            "esg_down_estimate": est,
            "esg_down_se": se,
            "esg_down_raw_p": p_value,
        })
    table = pd.DataFrame(rows)

    sum_components = (
        table.loc[table["outcome"] == "rvonly", "esg_down_estimate"].iloc[0]
        + table.loc[table["outcome"] == "joint80", "esg_down_estimate"].iloc[0]
    )
    inclusive_estimate = table.loc[table["outcome"] == "pair_rv80", "esg_down_estimate"].iloc[0]
    logger.info(
        "Decomposition check (esg_down): PairRV80=%.6f vs RVOnly+Joint80=%.6f (approximate, FE differ across specs)",
        inclusive_estimate, sum_components,
    )
    return fits, table


def run_stacked_contrast(df: pd.DataFrame, regs: list[str]) -> dict[str, float]:
    """Step 4: stacked-system cross-equation contrast H0: theta_D^Joint80 - theta_D^RVOnly = 0."""
    stacked = build_stacked_frame(df, outcome_a="joint80", outcome_b="rvonly", regressors=regs,
                                   label_a="J80", label_b="RVOnly")
    stacked_fit = fit_stacked_system(stacked, regressors_a=regs, regressors_b=regs,
                                      label_a="J80", label_b="RVOnly")
    estimate, se, p_value = cross_equation_contrast(stacked_fit, "esg_down__J80", "esg_down__RVOnly")
    return {"estimate": estimate, "se": se, "p_value": p_value}


def run_sector_month_validation(df: pd.DataFrame, regs: list[str]) -> pd.DataFrame:
    """Step 5: Joint80 model with sector x month FE in place of month FE (firm FE kept).

    `fit_fe_panel` always clusters on `entity_col`/`time_col` (cluster_entity=True,
    cluster_time=True internally), so swapping `time_col` to the sector-month
    composite id also moves the time-cluster dimension from calendar month to
    sector-month for this one robustness fit. That is a documented
    simplification of the two-way clustering scheme for this validation only.
    """
    df = df.copy()
    df["sector_month"] = df["sector"].astype(str) + "_" + df["month_id"].astype(str)
    sector_month_codes, _ = pd.factorize(df["sector_month"], sort=True)
    df["sector_month_id"] = sector_month_codes
    fit = fit_fe_panel(df, outcome="joint80", regressors=regs, entity_col="firm_id", time_col="sector_month_id")
    est, se, p_value = linear_contrast(fit, "esg_down")
    return pd.DataFrame([{
        "nobs": fit.nobs,
        "esg_down_estimate": est,
        "esg_down_se": se,
        "esg_down_raw_p": p_value,
    }])


def run_amihud_validation(df: pd.DataFrame, regs: list[str]) -> pd.DataFrame:
    """Step 6: Amihud-illiquidity alternative Joint80 construct (Loss x RVOL x Amihud tail).

    The panel only carries a trailing `amihud12` exposure column, with no
    separate contemporaneous Amihud outcome series, so the pooled 80th
    percentile of `amihud12` itself is used as a practical proxy for an
    Amihud-based adverse-liquidity tail (a simplification, not the paper's
    contemporaneous CHL outcome construction).
    """
    df = df.copy()
    q_amihud = df["amihud12"].quantile(config.JOINT_THRESHOLD_PRIMARY)
    df["tail_amihud"] = (df["amihud12"] >= q_amihud).astype(int)
    df["joint80_amihud"] = df["tail_return"] * df["tail_rvol"] * df["tail_amihud"]

    fit = fit_fe_panel(df, outcome="joint80_amihud", regressors=regs)
    est, se, p_value = linear_contrast(fit, "esg_down")

    n_events = int(df["joint80_amihud"].sum())
    prevalence = float(n_events / len(df))
    overlap_share = float(
        ((df["joint80_amihud"] == 1) & (df["joint80"] == 1)).sum() / max(n_events, 1)
    )
    return pd.DataFrame([{
        "nobs": fit.nobs,
        "n_events": n_events,
        "prevalence": prevalence,
        "overlap_share_with_chl_joint80": overlap_share,
        "esg_down_estimate": est,
        "esg_down_se": se,
        "esg_down_raw_p": p_value,
    }])


def main(full: bool = False) -> dict:
    n_firms = config.SYNTHETIC_N_FIRMS_PAPER if full else config.SYNTHETIC_N_FIRMS_RUN

    df, provenance, thresholds = build_panel(n_firms)
    logger.info(
        "Panel realized: %d firm-months, %d firms, %d months (%s)",
        len(df), df["firm_id"].nunique(), df["month_id"].nunique(),
        provenance.get("market_state_source"),
    )

    regs = regressors_equation7(include_blocks=None)

    fit_joint80, primary_table = run_primary_joint80(df, regs)
    slopes_table = run_complete_slopes(fit_joint80)
    decomposition_fits, decomposition_table = run_decomposition(df, regs)
    stacked_contrast = run_stacked_contrast(df, regs)
    sector_month_table = run_sector_month_validation(df, regs)
    amihud_table = run_amihud_validation(df, regs)

    save_table(primary_table, "exp3_joint80_primary")
    save_table(slopes_table, "exp3_complete_slopes")
    save_table(decomposition_table, "exp3_decomposition")
    save_json(stacked_contrast, "exp3_stacked_contrast")
    save_table(sector_month_table, "exp3_sector_month_validation")
    save_table(amihud_table, "exp3_amihud_validation")
    save_json({
        "provenance": provenance,
        "thresholds": thresholds,
        "n_firms": n_firms,
        "full_run": full,
    }, "exp3_run_metadata")

    logger.info("exp3 Joint80 primary table:\n%s", primary_table.to_string())
    logger.info("exp3 complete fitted ESG slopes:\n%s", slopes_table.to_string())
    logger.info("exp3 decomposition table:\n%s", decomposition_table.to_string())
    logger.info("exp3 stacked cross-equation contrast: %s", stacked_contrast)
    logger.info("exp3 sector x month FE validation:\n%s", sector_month_table.to_string())
    logger.info("exp3 Amihud alternative validation:\n%s", amihud_table.to_string())

    return {
        "primary_table": primary_table,
        "slopes_table": slopes_table,
        "decomposition_table": decomposition_table,
        "stacked_contrast": stacked_contrast,
        "sector_month_table": sector_month_table,
        "amihud_table": amihud_table,
        "provenance": provenance,
        "thresholds": thresholds,
    }


if __name__ == "__main__":
    main(full="--full" in sys.argv)
