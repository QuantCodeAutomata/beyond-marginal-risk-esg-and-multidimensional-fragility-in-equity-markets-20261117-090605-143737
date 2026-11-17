"""Experiment 2: Conventional Risk Attenuation of Joint80 ESG Association.

Reproduces the paper's Table A5 / exp_2 methodology: on one frozen Joint80
sample (Eq. 3, CHL-based primary 80th-percentile pooled thresholds on Loss,
log RVOL and log CHL), the ESG x Down / ESG x Up association with the joint
fragility state (Eq. 7) is re-estimated across five nested specifications
that progressively add ex ante exposure blocks -- (a) none, (b) beta only,
(c) volatility only, (d) liquidity only, (e) all three. Because the sample,
outcome and ESG construct are held fixed across all five columns, any change
in the ESG x Down coefficient as blocks are added is attributable to
conventional-risk confounding rather than to sample composition, which is the
paper's basis for arguing that part of the apparent ESG-resilience
association reflects omitted conventional risk exposures rather than ESG
information per se.

Usage: ``python -m exp.exp2 [--full]``. ``--full`` uses the paper's panel
size (``config.SYNTHETIC_N_FIRMS_PAPER``) instead of the reduced run
configuration (``config.SYNTHETIC_N_FIRMS_RUN``).
"""
from __future__ import annotations

import logging
import sys
import time

import pandas as pd

from src import config
from src.panel_pipeline import build_model_frame, joint_state, regressors_equation7
from src.pipeline import build_experiment_panel
from src.regression import PanelRegressionResult, fit_fe_panel, linear_contrast
from src.reporting import save_json, save_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# The five progressive-exposure specifications of Eq. 7 compared in Table A5.
# Each entry is (spec_name, include_blocks); include_blocks=None means "all three".
SPECIFICATIONS: list[tuple[str, list[str] | None]] = [
    ("none", []),
    ("beta", ["beta24"]),
    ("vol", ["rvol12"]),
    ("liq", ["chl12"]),
    ("full", None),
]


def run_specifications(df: pd.DataFrame) -> dict[str, dict]:
    """Fit Equation 7 on the identical Joint80 sample for each exposure specification."""
    results: dict[str, dict] = {}
    for spec_name, include_blocks in SPECIFICATIONS:
        regressors = regressors_equation7(include_blocks=include_blocks)
        fit: PanelRegressionResult = fit_fe_panel(df, outcome="joint80", regressors=regressors)
        down_est, down_se, down_p = linear_contrast(fit, "esg_down")
        up_est, up_se, up_p = linear_contrast(fit, "esg_up")
        contrast_est, contrast_se, contrast_p = linear_contrast(fit, "esg_down", "esg_up", weight_b=-1.0)
        results[spec_name] = {
            "included_blocks": include_blocks if include_blocks is not None else ["beta24", "rvol12", "chl12"],
            "nobs": fit.nobs,
            "n_firms": fit.n_entities,
            "esg_down": (down_est, down_se, down_p),
            "esg_up": (up_est, up_se, up_p),
            "down_up_contrast": (contrast_est, contrast_se, contrast_p),
        }
    return results


def main(full: bool = False) -> dict:
    start = time.monotonic()
    n_firms = config.SYNTHETIC_N_FIRMS_PAPER if full else config.SYNTHETIC_N_FIRMS_RUN

    raw_panel, provenance = build_experiment_panel(
        n_firms=n_firms,
        esg_down_effect=None,
        use_real_market_state=True,
    )
    df = build_model_frame(raw_panel, esg_col="esg_ias")

    # Frozen pooled Joint80 indicator (Eq. 3) computed once on this sample; it is held fixed
    # across all five specifications below so that only the right-hand side changes.
    df["joint80"], thresholds = joint_state(df, quantile=config.JOINT_THRESHOLD_PRIMARY)

    logger.info(
        "Panel realized: %d firm-months, %d firms, %d months, Joint80 events=%d (%.2f%%) (%s)",
        len(df), df["firm_id"].nunique(), df["month_id"].nunique(),
        int(df["joint80"].sum()), 100.0 * df["joint80"].mean(),
        provenance.get("market_state_source"),
    )

    spec_results = run_specifications(df)

    table = pd.DataFrame([
        {
            "spec_name": spec_name,
            "included_blocks": ",".join(spec_results[spec_name]["included_blocks"]) or "none",
            "nobs": spec_results[spec_name]["nobs"],
            "n_firms": spec_results[spec_name]["n_firms"],
            "esg_down_estimate": spec_results[spec_name]["esg_down"][0],
            "esg_down_se": spec_results[spec_name]["esg_down"][1],
            "esg_down_raw_p": spec_results[spec_name]["esg_down"][2],
            "esg_up_estimate": spec_results[spec_name]["esg_up"][0],
            "esg_up_se": spec_results[spec_name]["esg_up"][1],
            "esg_up_raw_p": spec_results[spec_name]["esg_up"][2],
            "down_up_contrast_estimate": spec_results[spec_name]["down_up_contrast"][0],
            "down_up_contrast_se": spec_results[spec_name]["down_up_contrast"][1],
            "down_up_contrast_raw_p": spec_results[spec_name]["down_up_contrast"][2],
        }
        for spec_name, _ in SPECIFICATIONS
    ])

    # Descriptive attenuation (paper Eq. 7 / Table A5): the fractional change in the ESG x Down
    # coefficient between the no-controls ("none") and fully risk-adjusted ("full") specifications.
    # A negative-magnitude-shrinking move towards zero (or a sign flip) indicates that conventional
    # ex ante risk exposures account for part of the raw ESG x Down association with Joint80.
    none_down_estimate = spec_results["none"]["esg_down"][0]
    full_down_estimate = spec_results["full"]["esg_down"][0]
    attenuation_fraction = (
        (full_down_estimate - none_down_estimate) / none_down_estimate
        if none_down_estimate != 0
        else float("nan")
    )

    beta_only_down_estimate = spec_results["beta"]["esg_down"][0]
    vol_only_down_estimate = spec_results["vol"]["esg_down"][0]
    liq_only_down_estimate = spec_results["liq"]["esg_down"][0]

    # Which single block, added alone, attenuates esg_down the most (largest move away from the
    # "none" estimate, i.e. the largest magnitude reduction towards/through zero)?
    single_block_down = {
        "beta24": beta_only_down_estimate,
        "rvol12": vol_only_down_estimate,
        "chl12": liq_only_down_estimate,
    }
    attenuation_by_block = {
        block: abs(none_down_estimate - est) for block, est in single_block_down.items()
    }
    largest_attenuation_block_down = max(attenuation_by_block, key=lambda b: attenuation_by_block[b])

    # Which single block, added alone, attenuates the Down-minus-Up contrast p-value the most
    # (i.e. moves the contrast p-value furthest from the "none" specification's p-value, towards
    # non-significance)?
    none_contrast_p = spec_results["none"]["down_up_contrast"][2]
    single_block_contrast_p = {
        "beta24": spec_results["beta"]["down_up_contrast"][2],
        "rvol12": spec_results["vol"]["down_up_contrast"][2],
        "chl12": spec_results["liq"]["down_up_contrast"][2],
    }
    contrast_p_shift_by_block = {
        block: p - none_contrast_p for block, p in single_block_contrast_p.items()
    }
    largest_attenuation_block_contrast = max(
        contrast_p_shift_by_block, key=lambda b: contrast_p_shift_by_block[b]
    )

    attenuation_summary = {
        "none_down_estimate": none_down_estimate,
        "full_down_estimate": full_down_estimate,
        "attenuation_fraction": attenuation_fraction,
        "beta_only_down_estimate": beta_only_down_estimate,
        "vol_only_down_estimate": vol_only_down_estimate,
        "liq_only_down_estimate": liq_only_down_estimate,
        "none_down_up_contrast_raw_p": none_contrast_p,
        "beta_only_down_up_contrast_raw_p": single_block_contrast_p["beta24"],
        "vol_only_down_up_contrast_raw_p": single_block_contrast_p["rvol12"],
        "liq_only_down_up_contrast_raw_p": single_block_contrast_p["chl12"],
        "largest_attenuation_block_esg_down": largest_attenuation_block_down,
        "largest_attenuation_block_down_up_contrast_p": largest_attenuation_block_contrast,
    }

    run_metadata = {
        "provenance": provenance,
        "joint80_thresholds": {k: float(v) for k, v in thresholds.items()},
        "n_firms": n_firms,
        "full_run": full,
        "runtime_seconds": time.monotonic() - start,
    }

    save_table(table, "exp2_joint80_exposure_specifications")
    save_json(attenuation_summary, "exp2_attenuation_summary")
    save_json(run_metadata, "exp2_run_metadata")

    logger.info(
        "exp2 realized sample: nobs=%d, n_firms=%d, runtime=%.2fs",
        int(table["nobs"].iloc[0]), int(table["n_firms"].iloc[0]), run_metadata["runtime_seconds"],
    )
    logger.info("exp2 Joint80 exposure specifications table:\n%s", table.to_string())
    logger.info("exp2 attenuation summary: %s", attenuation_summary)

    return {
        "table": table,
        "attenuation_summary": attenuation_summary,
        "run_metadata": run_metadata,
    }


if __name__ == "__main__":
    main(full="--full" in sys.argv)
