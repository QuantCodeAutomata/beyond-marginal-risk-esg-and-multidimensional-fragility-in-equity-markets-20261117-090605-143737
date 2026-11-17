"""Experiment 1: Primary Marginal ESG x Down Tests with Full Risk Adjustment.

Reproduces the three fully risk-adjusted marginal models of Equation 5
(return loss, log realized volatility, log CHL spread) and tests whether the
MSCI Industry Adjusted Score (IAS) has a negative, market-state-specific
association with adverse outcomes after controlling for conventional ex ante
exposures (Sec. 4, Eq. 5). Runs:
  1. The three channel regressions with firm and month fixed effects, six
     lagged controls and the channel-specific ex ante exposure block.
  2. The primary Holm family on the three ESG x Down coefficients, and the
     secondary Holm family on the three Down-minus-Up contrasts.
  3. Conventional two-way clustered inference and a Rademacher wild-cluster
     bootstrap (month cluster, null imposed on ESG x Down) for both families.
  4. A leave-one-month-out stability check of all ESG x Down estimates.

Usage: ``python -m exp.exp1 [--full]``. ``--full`` uses the paper's panel
size and 9,999 wild-bootstrap replications instead of the reduced run
configuration (see results/RESULTS.md for the runtime that motivated the
reduction).
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from src import config
from src.inference import holm_adjust, wild_cluster_bootstrap_panel
from src.panel_pipeline import OUTCOME_OF_CHANNEL, build_model_frame, regressors_equation5
from src.pipeline import build_experiment_panel
from src.regression import fit_fe_panel, linear_contrast
from src.reporting import save_json, save_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHANNELS = ["return", "volatility", "liquidity"]


def run_channel_models(model_frame: pd.DataFrame) -> dict[str, dict]:
    """Estimate Equation 5 for each of the three marginal channels."""
    results = {}
    for channel in CHANNELS:
        outcome = OUTCOME_OF_CHANNEL[channel]
        regressors = regressors_equation5(channel)
        fit = fit_fe_panel(model_frame, outcome, regressors)
        down_est, down_se, down_p = linear_contrast(fit, "esg_down")
        up_est, up_se, up_p = linear_contrast(fit, "esg_up")
        contrast_est, contrast_se, contrast_p = linear_contrast(fit, "esg_down", "esg_up", weight_b=-1.0)
        results[channel] = {
            "fit": fit,
            "outcome": outcome,
            "regressors": regressors,
            "esg_down": (down_est, down_se, down_p),
            "esg_up": (up_est, up_se, up_p),
            "down_up_contrast": (contrast_est, contrast_se, contrast_p),
            "nobs": fit.nobs,
            "n_entities": fit.n_entities,
        }
    return results


def run_wild_bootstrap(model_frame: pd.DataFrame, channel_results: dict, n_reps: int) -> dict:
    """Wild-cluster bootstrap p-values for ESG x Down and the Down-Up contrast (methodology step 6)."""
    wild = {}
    for channel in CHANNELS:
        outcome = channel_results[channel]["outcome"]
        regressors = channel_results[channel]["regressors"]
        down_boot = wild_cluster_bootstrap_panel(
            model_frame, outcome, regressors, "esg_down", "firm_id", "month_id",
            n_reps=n_reps, seed=config.WILD_BOOTSTRAP_SEED,
        )
        contrast_boot = wild_cluster_bootstrap_panel(
            model_frame, outcome, regressors, "esg_down", "firm_id", "month_id",
            n_reps=n_reps, seed=config.WILD_BOOTSTRAP_SEED, weight_b=-1.0, coef_b="esg_up",
        )
        wild[channel] = {"esg_down": down_boot, "down_up_contrast": contrast_boot}
    return wild


def leave_one_month_out(model_frame: pd.DataFrame) -> pd.DataFrame:
    """Re-estimate all 3 channels omitting each month once; record ESG x Down sign stability."""
    rows = []
    months = sorted(model_frame["month_id"].unique())
    for month in months:
        subset = model_frame[model_frame["month_id"] != month]
        for channel in CHANNELS:
            outcome = OUTCOME_OF_CHANNEL[channel]
            regressors = regressors_equation5(channel)
            try:
                fit = fit_fe_panel(subset, outcome, regressors)
                est, se, p_value = linear_contrast(fit, "esg_down")
            except Exception as exc:  # a singular design on one omitted month is reported, not hidden
                logger.warning("Leave-one-out failed for %s omitting %s: %s", channel, month, exc)
                est, se, p_value = float("nan"), float("nan"), float("nan")
            rows.append({"month_omitted": month, "channel": channel, "esg_down": est, "se": se, "p_value": p_value})
    return pd.DataFrame(rows)


def main(full: bool = False) -> None:
    n_firms = config.SYNTHETIC_N_FIRMS_PAPER if full else config.SYNTHETIC_N_FIRMS_RUN
    n_reps = config.WILD_BOOTSTRAP_REPLICATIONS if full else config.WILD_BOOTSTRAP_REPLICATIONS_RUN

    raw_panel, provenance = build_experiment_panel(n_firms=n_firms)
    logger.info(
        "Panel realized: %d firm-months, %d firms, %d months (%s)",
        len(raw_panel), raw_panel["firm_id"].nunique(), raw_panel["month_id"].nunique(),
        provenance.get("market_state_source"),
    )
    model_frame = build_model_frame(raw_panel, esg_col="esg_ias")

    channel_results = run_channel_models(model_frame)

    primary_table = pd.DataFrame([
        {
            "channel": channel,
            "outcome": channel_results[channel]["outcome"],
            "nobs": channel_results[channel]["nobs"],
            "n_firms": channel_results[channel]["n_entities"],
            "esg_down_estimate": channel_results[channel]["esg_down"][0],
            "esg_down_se": channel_results[channel]["esg_down"][1],
            "esg_down_raw_p": channel_results[channel]["esg_down"][2],
            "esg_up_estimate": channel_results[channel]["esg_up"][0],
            "esg_up_se": channel_results[channel]["esg_up"][1],
            "esg_up_raw_p": channel_results[channel]["esg_up"][2],
            "down_up_contrast": channel_results[channel]["down_up_contrast"][0],
            "down_up_contrast_raw_p": channel_results[channel]["down_up_contrast"][2],
        }
        for channel in CHANNELS
    ])

    holm_primary = holm_adjust([channel_results[c]["esg_down"][2] for c in CHANNELS], alpha=config.HOLM_ALPHA)
    holm_primary.insert(0, "channel", CHANNELS)
    holm_secondary = holm_adjust([channel_results[c]["down_up_contrast"][2] for c in CHANNELS], alpha=config.HOLM_ALPHA)
    holm_secondary.insert(0, "channel", CHANNELS)

    wild = run_wild_bootstrap(model_frame, channel_results, n_reps)
    wild_rows = [
        {
            "channel": channel,
            "esg_down_wild_p": wild[channel]["esg_down"]["wild_p_value"],
            "down_up_contrast_wild_p": wild[channel]["down_up_contrast"]["wild_p_value"],
            "n_reps_used": wild[channel]["esg_down"]["n_reps_used"],
        }
        for channel in CHANNELS
    ]
    wild_table = pd.DataFrame(wild_rows)
    holm_wild_primary = holm_adjust(wild_table["esg_down_wild_p"].tolist(), alpha=config.HOLM_ALPHA)
    holm_wild_primary.insert(0, "channel", CHANNELS)

    loo = leave_one_month_out(model_frame)
    loo_summary = pd.DataFrame([
        {
            "channel": channel,
            "n_estimates": int((loo["channel"] == channel).sum()),
            "n_negative": int(((loo["channel"] == channel) & (loo["esg_down"] < 0)).sum()),
            "all_negative": bool((loo.loc[loo["channel"] == channel, "esg_down"] < 0).all()),
        }
        for channel in CHANNELS
    ])

    save_table(primary_table, "exp1_primary_marginal_models")
    save_table(holm_primary, "exp1_holm_primary_esg_down")
    save_table(holm_secondary, "exp1_holm_secondary_down_up_contrast")
    save_table(wild_table, "exp1_wild_bootstrap")
    save_table(holm_wild_primary, "exp1_holm_wild_primary_esg_down")
    save_table(loo, "exp1_leave_one_month_out_full")
    save_table(loo_summary, "exp1_leave_one_month_out_summary")
    save_json({
        "provenance": provenance,
        "n_firms": n_firms,
        "n_wild_bootstrap_reps": n_reps,
        "full_run": full,
    }, "exp1_run_metadata")

    logger.info("exp1 primary table:\n%s", primary_table.to_string())
    logger.info("exp1 Holm primary family:\n%s", holm_primary.to_string())
    logger.info("exp1 leave-one-month-out summary:\n%s", loo_summary.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Use the paper's panel size and 9,999 bootstrap replications")
    args = parser.parse_args()
    main(full=args.full)
