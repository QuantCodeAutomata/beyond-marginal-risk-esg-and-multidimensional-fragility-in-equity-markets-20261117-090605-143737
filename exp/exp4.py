"""Experiment 4: Paired Standardized IAS vs WAKIS ESG Score Architecture Comparison.

Reproduces the paired-standardized architecture comparison between MSCI's
downstream Industry Adjusted Score (IAS) and the upstream Weighted Average
Key Issue Score (WAKIS) that accompanies the Joint80 model of Equation 7
(paper Table 4 / Tables A11-A13, exp_4 methodology). On the identical
firm-month sample used throughout the paper, both ESG constructs are
standardized separately to their own sample SD (z-score), the fully adjusted
Joint80 model is refit once per standardized construct, and a block-diagonal
stacked system (the same construction as exp_3's RVOnly-vs-Joint80 contrast,
here with the SAME outcome Joint80 shared by both equations and only the
ESG construct differing) is used to test whether the two score architectures
imply materially different fitted associations across the market-state grid.

`src.decomposition.build_stacked_frame` duplicates rows from a *single*
source dataframe with two outcome columns, renaming one shared regressor
list per equation label. That does not fit this experiment: the IAS and
WAKIS equations need their *own* physical "esg"/"esg_down"/"esg_up" values
(from two different model frames, `df_ias` and `df_wakis`) while sharing the
same outcome (Joint80) and the same control/exposure columns. So the stacked
system here is built directly in this module from the two source frames,
following the exact same duplicate-rows-plus-equation-specific-FE pattern as
`src/decomposition.py` (entity_eq = firm x equation, time_eq = month x
equation, PanelOLS with entity_effects=True, time_effects=True, two-way
clustering on the original firm_id/month_id). `cross_equation_contrast` from
`src.decomposition` is reused as-is since it only relies on the fitted
result's `.params`/`.cov`, which this custom stacked fit also exposes.

Steps:
  1. Standardize IAS and WAKIS on the common sample; refit Equation 7 for
     each standardized score; report beta, theta_D and the complete fitted
     slope at 0%/-5%/-10% market states for both.
  2. Build the stacked IAS-vs-WAKIS system (shared Joint80 outcome, firm +
     month FE per equation, two-way clustering on the original firm/month
     ids) and test four cross-equation contrasts: Delta_D (Down-interaction
     gap), Delta_0, Delta_-5, Delta_-10 (complete-slope gaps at market
     states 0%, -5%, -10%), with Holm adjustment over the {Delta_0,
     Delta_-5, Delta_-10} family and, separately, a conservative 4-test
     family adding Delta_D.
  3. Lighter-weight validations given the time budget: a leave-one-month-out
     refit of the stacked system (subsampled only if a single stacked refit
     proves too slow; see code comment at the call site), a sector x month
     fixed-effect refit, and a 2020-2021 exclusion refit.
  4. The paper's "observed-change timing synchronization" validation
     (a WAKIS proxy resynchronized to IAS's update timing) is NOT
     implemented in this run -- out of scope given the time budget; this is
     recorded in `exp4_run_metadata` so the parent integrator lists it under
     "Not reproduced".

Usage: ``python -m exp.exp4 [--full]``. ``--full`` uses the paper's panel
size (config.SYNTHETIC_N_FIRMS_PAPER) instead of the reduced run
configuration (config.SYNTHETIC_N_FIRMS_RUN); see results/RESULTS.md for the
runtime trade-off.
"""
from __future__ import annotations

import logging
import sys
import time

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from scipy import stats

from src import config
from src.decomposition import cross_equation_contrast
from src.inference import holm_adjust
from src.panel_pipeline import CONTROL_COLS, EXPOSURE_BLOCKS, build_model_frame, joint_state, regressors_equation7
from src.pipeline import build_experiment_panel
from src.regression import PanelRegressionResult, complete_slope, fit_fe_panel, linear_contrast
from src.reporting import save_json, save_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EXCLUDED_YEARS = (2020, 2021)
LABEL_IAS = "IAS"
LABEL_WAKIS = "WAKIS"


def _shared_regressors() -> list[str]:
    """Regressor names shared verbatim by `df_ias` and `df_wakis` (Equation 7, Sec. methodology)."""
    exposure_cols = [b for block in EXPOSURE_BLOCKS.values() for b in (block, f"{block}_down", f"{block}_up")]
    return ["esg", "esg_down", "esg_up"] + CONTROL_COLS + exposure_cols


def build_panels(n_firms: int) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Standardize IAS/WAKIS on the common sample and build the two Equation-7 model frames."""
    raw_panel, provenance = build_experiment_panel(
        n_firms=n_firms, esg_down_effect=None, use_real_market_state=True,
    )
    raw_panel["ias_std"] = (raw_panel["esg_ias"] - raw_panel["esg_ias"].mean()) / raw_panel["esg_ias"].std()
    raw_panel["wakis_std"] = (raw_panel["esg_wakis"] - raw_panel["esg_wakis"].mean()) / raw_panel["esg_wakis"].std()

    df_ias = build_model_frame(raw_panel, esg_col="ias_std")
    df_wakis = build_model_frame(raw_panel, esg_col="wakis_std")

    df_ias["joint80"], thresholds = joint_state(df_ias, quantile=config.JOINT_THRESHOLD_PRIMARY)
    df_wakis["joint80"] = df_ias["joint80"]  # identical outcome construction (Loss/RVOL/CHL only)

    thresholds_out = {k: float(v) for k, v in thresholds.items()}
    return df_ias, df_wakis, provenance, thresholds_out


def run_standardized_fits(
    df_ias: pd.DataFrame, df_wakis: pd.DataFrame, regs: list[str],
) -> tuple[PanelRegressionResult, PanelRegressionResult, pd.DataFrame]:
    """Step 1: refit Equation 7 separately for standardized IAS and standardized WAKIS."""
    fit_ias = fit_fe_panel(df_ias, outcome="joint80", regressors=regs)
    fit_wakis = fit_fe_panel(df_wakis, outcome="joint80", regressors=regs)

    rows = []
    for label, fit in ((LABEL_IAS, fit_ias), (LABEL_WAKIS, fit_wakis)):
        beta_est, beta_se, _ = linear_contrast(fit, "esg")
        thetaD_est, thetaD_se, _ = linear_contrast(fit, "esg_down")
        slopes = complete_slope(fit, base_coef="esg", down_coef="esg_down", market_states=[0.0, -0.05, -0.10])
        slope_at = {float(row["market_state"]): float(row["slope"]) for _, row in slopes.iterrows()}
        rows.append({
            "score": label,
            "beta_estimate": beta_est,
            "beta_se": beta_se,
            "thetaD_estimate": thetaD_est,
            "thetaD_se": thetaD_se,
            "slope_at_0pct": slope_at[0.0],
            "slope_at_minus5pct": slope_at[-0.05],
            "slope_at_minus10pct": slope_at[-0.10],
        })
    table = pd.DataFrame(rows)
    return fit_ias, fit_wakis, table


def build_stacked_frame_two(
    df_a: pd.DataFrame, df_b: pd.DataFrame, outcome: str, regressors: list[str], label_a: str, label_b: str,
) -> pd.DataFrame:
    """Duplicate rows from two source frames (same outcome, same regressor names, different values).

    Mirrors `src.decomposition.build_stacked_frame`'s row-duplication and
    equation-specific-FE construction, generalized to two distinct source
    dataframes instead of one dataframe with two outcome columns: this is
    what lets the IAS and WAKIS equations each carry their own physical
    "esg"/"esg_down"/"esg_up" values under the shared coefficient names
    "esg__IAS"/"esg_down__IAS" vs "esg__WAKIS"/"esg_down__WAKIS".
    """
    copies = []
    for df, label in ((df_a, label_a), (df_b, label_b)):
        copy = df[["firm_id", "month_id"] + regressors].copy()
        copy["outcome"] = df[outcome]
        copy["equation"] = label
        copy = copy.rename(columns={col: f"{col}__{label}" for col in regressors})
        copies.append(copy)
    stacked = pd.concat(copies, ignore_index=True, sort=False).fillna(0.0)
    stacked["firm_id"] = pd.concat([df_a["firm_id"], df_b["firm_id"]], ignore_index=True)
    stacked["month_id"] = pd.concat([df_a["month_id"], df_b["month_id"]], ignore_index=True)
    stacked["entity_eq"] = stacked["firm_id"].astype(str) + "_" + stacked["equation"]
    month_codes, month_levels = pd.factorize(stacked["month_id"], sort=True)
    equation_codes, _ = pd.factorize(stacked["equation"], sort=True)
    stacked["time_eq"] = month_codes + equation_codes * len(month_levels)
    return stacked


def fit_stacked_system_two(stacked: pd.DataFrame, regressors: list[str], label_a: str, label_b: str):
    """Fit the block-diagonal stacked system with equation-specific FE and two-way clustering.

    Same estimator call as `src.decomposition.fit_stacked_system`: PanelOLS
    with entity_effects=True, time_effects=True on the equation-specific
    entity_eq/time_eq index, clustered on the ORIGINAL firm_id/month_id so
    the covariance reflects correlation between the two equations' copies of
    the same firm-month.
    """
    cols_a = [f"{c}__{label_a}" for c in regressors]
    cols_b = [f"{c}__{label_b}" for c in regressors]
    panel = stacked.set_index(["entity_eq", "time_eq"])
    y = panel["outcome"]
    x = panel[cols_a + cols_b]
    model = PanelOLS(y, x, entity_effects=True, time_effects=True, drop_absorbed=True)
    clusters = pd.DataFrame(
        {"firm": stacked["firm_id"].to_numpy(), "month": stacked["month_id"].to_numpy()}, index=panel.index,
    )
    return model.fit(cov_type="clustered", clusters=clusters)


def fit_full_stacked_system(df_ias: pd.DataFrame, df_wakis: pd.DataFrame, regs: list[str]):
    """Build and fit the IAS-vs-WAKIS stacked system (shared Joint80 outcome) from two model frames."""
    stacked = build_stacked_frame_two(df_ias, df_wakis, "joint80", regs, LABEL_IAS, LABEL_WAKIS)
    fit = fit_stacked_system_two(stacked, regs, LABEL_IAS, LABEL_WAKIS)
    return stacked, fit


def delta_at_market_state(stacked_fit, weight: float) -> tuple[float, float, float]:
    """Delta_{-weight*100%} = (beta_IAS + weight*thetaD_IAS) - (beta_WAKIS + weight*thetaD_WAKIS).

    Computed directly from the stacked system's joint covariance matrix (the
    same algebra as `cross_equation_contrast`, generalized to a 4-coefficient
    linear combination since this contrast spans both the level and the
    Down-interaction coefficient of each equation).
    """
    names = ["esg__IAS", "esg_down__IAS", "esg__WAKIS", "esg_down__WAKIS"]
    w = np.array([1.0, weight, -1.0, -weight])
    params = stacked_fit.params[names].to_numpy()
    estimate = float(w @ params)
    sub_cov = stacked_fit.cov.loc[names, names].to_numpy()
    var = float(w @ sub_cov @ w)
    se = float(np.sqrt(var)) if var > 0 else np.nan
    z = estimate / se if se and se > 0 and np.isfinite(se) else np.nan
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    return estimate, se, p_value


def run_architecture_contrasts(stacked_fit) -> pd.DataFrame:
    """Step 2: the four cross-equation contrasts, with 3-test and 4-test Holm families."""
    delta_d = cross_equation_contrast(stacked_fit, "esg_down__IAS", "esg_down__WAKIS")
    delta_0 = cross_equation_contrast(stacked_fit, "esg__IAS", "esg__WAKIS")
    delta_m5 = delta_at_market_state(stacked_fit, 0.05)
    delta_m10 = delta_at_market_state(stacked_fit, 0.10)

    raw_p_3state = [delta_0[2], delta_m5[2], delta_m10[2]]
    holm_3 = holm_adjust(raw_p_3state, alpha=config.HOLM_ALPHA)
    raw_p_4test = [delta_d[2], delta_0[2], delta_m5[2], delta_m10[2]]
    holm_4 = holm_adjust(raw_p_4test, alpha=config.HOLM_ALPHA)

    names = ["Delta_D", "Delta_0", "Delta_-5", "Delta_-10"]
    contrasts = [delta_d, delta_0, delta_m5, delta_m10]
    holm_p_3state = [np.nan, *holm_3["holm_p"].to_numpy()]
    holm_p_4test = holm_4["holm_p"].to_numpy()

    rows = []
    for name, (estimate, se, raw_p), h3, h4 in zip(names, contrasts, holm_p_3state, holm_p_4test):
        rows.append({
            "contrast": name, "estimate": estimate, "se": se, "raw_p": raw_p,
            "holm_p_3state": h3, "holm_p_4test": h4,
        })
    return pd.DataFrame(rows)


def run_leave_one_month_out(df_ias: pd.DataFrame, df_wakis: pd.DataFrame, regs: list[str]) -> dict:
    """Validation (a): leave-one-month-out refit of the stacked system, recomputing Delta_-5.

    Time-driven simplification: a single stacked refit on this sample takes
    roughly 1-1.5 seconds. The first refit is timed; if it exceeds 1.5s the
    loop falls back to sampling every 3rd realized month instead of all ~82,
    to keep the total validation within the ~60-90 second budget stated in
    the experiment brief, rather than dropping the leave-one-out check
    entirely.
    """
    months = sorted(df_ias["month_id"].unique())
    sampling = "every month"
    deltas: list[float] = []
    t_start = time.time()
    for i, month in enumerate(months):
        keep = df_ias["month_id"] != month
        fit_iter_start = time.time()
        _, fit = fit_full_stacked_system(df_ias.loc[keep], df_wakis.loc[keep], regs)
        iter_time = time.time() - fit_iter_start
        if i == 0 and iter_time > 1.5:
            sampling = "every 3rd month"
        if sampling == "every 3rd month" and i % 3 != 0:
            continue
        est, _, _ = delta_at_market_state(fit, 0.05)
        deltas.append(est)
    total_time = time.time() - t_start

    deltas_arr = np.asarray(deltas, dtype=float)
    return {
        "leave_one_month_out_sampling": sampling,
        "n_months_realized": len(months),
        "n_sampled": int(deltas_arr.size),
        "n_negative": int((deltas_arr < 0).sum()),
        "min": float(deltas_arr.min()) if deltas_arr.size else np.nan,
        "max": float(deltas_arr.max()) if deltas_arr.size else np.nan,
        "mean": float(deltas_arr.mean()) if deltas_arr.size else np.nan,
        "total_seconds": total_time,
    }


def run_sector_month_validation(df_ias: pd.DataFrame, df_wakis: pd.DataFrame, regs: list[str]) -> dict:
    """Validation (b): sector x month FE in place of month FE for the stacked system.

    Same composite-id trick as exp_3's sector x month validation: factorize
    `sector + month_id` into a single numeric id and substitute it for
    `month_id` before building the stacked frame, so the time fixed effect
    (and, following the stacked system's clustering design, the time cluster
    dimension) becomes sector-month instead of calendar month.
    """
    df_ias_sm = df_ias.copy()
    df_wakis_sm = df_wakis.copy()
    sector_month = df_ias["sector"].astype(str) + "_" + df_ias["month_id"].astype(str)
    sector_month_codes, _ = pd.factorize(sector_month, sort=True)
    df_ias_sm["month_id"] = sector_month_codes
    df_wakis_sm["month_id"] = sector_month_codes

    _, fit = fit_full_stacked_system(df_ias_sm, df_wakis_sm, regs)
    estimate, se, p_value = delta_at_market_state(fit, 0.05)
    return {
        "estimate": estimate, "se": se, "raw_p": p_value,
        "n_sector_month_groups": int(len(np.unique(sector_month_codes))),
    }


def run_period_exclusion(df_ias: pd.DataFrame, df_wakis: pd.DataFrame, regs: list[str]) -> dict:
    """Validation (c): exclude 2020-2021 rows and recompute the IAS esg_down fit and Delta_-5."""
    keep = ~df_ias["month_id"].dt.year.isin(EXCLUDED_YEARS)
    df_ias_excl = df_ias.loc[keep]
    df_wakis_excl = df_wakis.loc[keep]

    fit_ias_excl = fit_fe_panel(df_ias_excl, outcome="joint80", regressors=regs)
    ias_down_est, ias_down_se, ias_down_p = linear_contrast(fit_ias_excl, "esg_down")

    _, fit_excl = fit_full_stacked_system(df_ias_excl, df_wakis_excl, regs)
    delta_m5_est, delta_m5_se, delta_m5_p = delta_at_market_state(fit_excl, 0.05)

    return {
        "excluded_years": list(EXCLUDED_YEARS),
        "n_obs_excluded": int(len(df_ias) - len(df_ias_excl)),
        "n_obs_remaining": int(len(df_ias_excl)),
        "ias_esg_down_estimate": ias_down_est,
        "ias_esg_down_se": ias_down_se,
        "ias_esg_down_raw_p": ias_down_p,
        "delta_minus5_estimate": delta_m5_est,
        "delta_minus5_se": delta_m5_se,
        "delta_minus5_raw_p": delta_m5_p,
    }


def main(full: bool = False) -> dict:
    n_firms = config.SYNTHETIC_N_FIRMS_PAPER if full else config.SYNTHETIC_N_FIRMS_RUN

    t0 = time.time()
    df_ias, df_wakis, provenance, thresholds = build_panels(n_firms)
    logger.info(
        "Panel realized: %d firm-months, %d firms, %d months (%s) [%.2fs]",
        len(df_ias), df_ias["firm_id"].nunique(), df_ias["month_id"].nunique(),
        provenance.get("market_state_source"), time.time() - t0,
    )

    regs_single = regressors_equation7(include_blocks=None)
    regs_stacked = _shared_regressors()
    assert regs_single == regs_stacked, "Equation 7 regressor list must match the shared stacked-system list"

    t0 = time.time()
    fit_ias, fit_wakis, standardized_table = run_standardized_fits(df_ias, df_wakis, regs_single)
    logger.info(
        "Standardized-score fits done: IAS nobs=%d, WAKIS nobs=%d [%.2fs]",
        fit_ias.nobs, fit_wakis.nobs, time.time() - t0,
    )

    t0 = time.time()
    _, stacked_fit = fit_full_stacked_system(df_ias, df_wakis, regs_stacked)
    logger.info("Stacked IAS-vs-WAKIS system fit: %d rows [%.2fs]", 2 * len(df_ias), time.time() - t0)

    contrasts_table = run_architecture_contrasts(stacked_fit)

    t0 = time.time()
    loo_summary = run_leave_one_month_out(df_ias, df_wakis, regs_stacked)
    logger.info(
        "Leave-one-month-out done: sampling=%s, n_sampled=%d, n_negative=%d [%.2fs]",
        loo_summary["leave_one_month_out_sampling"], loo_summary["n_sampled"], loo_summary["n_negative"],
        time.time() - t0,
    )

    t0 = time.time()
    sector_month_result = run_sector_month_validation(df_ias, df_wakis, regs_stacked)
    logger.info("Sector x month FE validation done [%.2fs]: %s", time.time() - t0, sector_month_result)

    t0 = time.time()
    period_exclusion_result = run_period_exclusion(df_ias, df_wakis, regs_single)
    logger.info("2020-2021 exclusion validation done [%.2fs]: %s", time.time() - t0, period_exclusion_result)

    save_table(standardized_table, "exp4_standardized_fits")
    save_table(contrasts_table, "exp4_architecture_contrasts")
    save_json(loo_summary, "exp4_leave_one_month_out")
    save_json(sector_month_result, "exp4_sector_month_validation")
    save_json(period_exclusion_result, "exp4_period_exclusion_2020_2021")
    save_json({
        "provenance": provenance,
        "thresholds": thresholds,
        "n_firms": n_firms,
        "full_run": full,
        "validations_run_at_reduced_scale": {
            "leave_one_month_out": loo_summary["leave_one_month_out_sampling"],
        },
        "not_reproduced": [
            "Observed-change timing synchronization validation (WAKIS proxy resynchronized to "
            "IAS's update timing) was not implemented in this run due to time constraints.",
        ],
    }, "exp4_run_metadata")

    logger.info("exp4 standardized fits:\n%s", standardized_table.to_string())
    logger.info("exp4 architecture contrasts:\n%s", contrasts_table.to_string())
    logger.info("exp4 leave-one-month-out summary: %s", loo_summary)
    logger.info("exp4 sector x month validation: %s", sector_month_result)
    logger.info("exp4 2020-2021 exclusion validation: %s", period_exclusion_result)

    return {
        "standardized_table": standardized_table,
        "contrasts_table": contrasts_table,
        "loo_summary": loo_summary,
        "sector_month_result": sector_month_result,
        "period_exclusion_result": period_exclusion_result,
        "provenance": provenance,
        "thresholds": thresholds,
    }


if __name__ == "__main__":
    main(full="--full" in sys.argv)
