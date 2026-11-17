"""Stacked-system cross-equation contrasts (RVOnly vs Joint80; IAS vs WAKIS).

Custom implementation (paper methodology step 7 / exp_3, exp_4): no package
in this environment fits two score-specific equations with a shared firm/month
cluster structure and tests a cross-equation linear contrast, so the stacked
system is built by duplicating the panel (one copy per equation), making every
right-hand-side variable and fixed effect equation-specific, and estimating
the whole system by OLS with dummy-interacted regressors. Firm and month
cluster identifiers are preserved across both copies of an observation so the
two-way clustered covariance reflects the shared sampling variation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

from scipy import stats


def build_stacked_frame(
    df: pd.DataFrame,
    outcome_a: str,
    outcome_b: str,
    regressors: list[str],
    label_a: str = "A",
    label_b: str = "B",
) -> pd.DataFrame:
    """Duplicate ``df`` into two outcome-specific copies, stacked into one long frame.

    Every entry in ``regressors`` is renamed `{col}__{label}` for each copy, so
    no coefficient is shared between the two equations; firm_id/month_id are
    kept unchanged so both copies of a firm-month cluster together.
    """
    copies = []
    for outcome, label in ((outcome_a, label_a), (outcome_b, label_b)):
        copy = df[["firm_id", "month_id"] + regressors].copy()
        copy["outcome"] = df[outcome]
        copy["equation"] = label
        rename = {col: f"{col}__{label}" for col in regressors}
        copy = copy.rename(columns=rename)
        copies.append(copy)
    stacked = pd.concat(copies, ignore_index=True, sort=False).fillna(0.0)
    stacked["firm_id"] = pd.concat([df["firm_id"], df["firm_id"]], ignore_index=True)
    stacked["month_id"] = pd.concat([df["month_id"], df["month_id"]], ignore_index=True)
    # Equation-specific entity/time fixed effects: interact the firm/month id
    # with the equation label so each equation gets its own absorbed effects.
    # linearmodels requires the time dimension to be numeric or date-like, so
    # the equation is folded into a numeric offset rather than a string suffix:
    # time_eq = month_index + equation_offset * n_months stays unique and numeric
    # per (month, equation), giving equation-specific month fixed effects.
    stacked["entity_eq"] = stacked["firm_id"].astype(str) + "_" + stacked["equation"]
    month_codes, month_levels = pd.factorize(stacked["month_id"], sort=True)
    equation_codes, _ = pd.factorize(stacked["equation"], sort=True)
    stacked["time_eq"] = month_codes + equation_codes * len(month_levels)
    return stacked


def fit_stacked_system(stacked: pd.DataFrame, regressors_a: list[str], regressors_b: list[str],
                        label_a: str = "A", label_b: str = "B"):
    """Fit the block-diagonal stacked system with equation-specific FE and two-way clustering.

    Clustering uses the *original* firm_id/month_id (not the equation-specific
    entity/time ids), so the covariance accounts for correlation between the
    two equations' copies of the same firm-month.
    """
    cols_a = [f"{c}__{label_a}" for c in regressors_a]
    cols_b = [f"{c}__{label_b}" for c in regressors_b]
    exog_cols = cols_a + cols_b
    panel = stacked.set_index(["entity_eq", "time_eq"])
    y = panel["outcome"]
    x = panel[exog_cols]
    model = PanelOLS(y, x, entity_effects=True, time_effects=True, drop_absorbed=True)
    # Two-way clustering on the ORIGINAL firm/month ids requires passing them
    # as explicit cluster arrays (not the entity_eq/time_eq FE index).
    clusters = pd.DataFrame(
        {"firm": stacked["firm_id"].to_numpy(), "month": stacked["month_id"].to_numpy()},
        index=panel.index,
    )
    result = model.fit(cov_type="clustered", clusters=clusters)
    return result


def cross_equation_contrast(result, coef_a: str, coef_b: str) -> tuple[float, float, float]:
    """H0: coef_a - coef_b = 0 using the stacked system's joint covariance matrix."""
    params = result.params
    cov = result.cov
    estimate = params[coef_a] - params[coef_b]
    var = cov.loc[coef_a, coef_a] + cov.loc[coef_b, coef_b] - 2 * cov.loc[coef_a, coef_b]
    se = float(np.sqrt(var)) if var > 0 else np.nan
    z = estimate / se if se and se > 0 else np.nan
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    return float(estimate), se, float(p_value)
