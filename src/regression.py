"""Fixed-effects panel regression wrapper for Equations 5 and 7.

Uses `linearmodels.PanelOLS` (entity + time fixed effects, two-way clustered
covariance) rather than re-implementing within transformation and clustered
standard errors from scratch.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS


@dataclass
class PanelRegressionResult:
    """Container with the pieces every experiment script needs."""

    params: pd.Series
    std_errors: pd.Series
    pvalues: pd.Series
    cov: pd.DataFrame
    nobs: int
    n_entities: int
    raw_result: object = field(repr=False)


def fit_fe_panel(
    df: pd.DataFrame,
    outcome: str,
    regressors: list[str],
    entity_col: str = "firm_id",
    time_col: str = "month_id",
    entity_effects: bool = True,
    time_effects: bool = True,
) -> PanelRegressionResult:
    """Estimate Y = alpha_i + lambda_t + regressors'coef + eps with two-way clustering.

    ``df`` must contain ``entity_col``, ``time_col``, ``outcome`` and every
    name in ``regressors``. Returns coefficients, clustered SEs, p-values and
    the full covariance matrix for Wald tests on linear combinations.
    """
    panel = df.set_index([entity_col, time_col])
    y = panel[outcome]
    x = panel[regressors]
    model = PanelOLS(y, x, entity_effects=entity_effects, time_effects=time_effects, drop_absorbed=True)
    result = model.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
    cov = _psd_clip(result.cov)
    std_errors = pd.Series(np.sqrt(np.diag(cov)), index=cov.index, name="std_error")
    from scipy import stats as _stats

    tstats = result.params / std_errors
    pvalues = pd.Series(2.0 * (1.0 - _stats.norm.cdf(np.abs(tstats))), index=cov.index)
    return PanelRegressionResult(
        params=result.params,
        std_errors=std_errors,
        pvalues=pvalues,
        cov=cov,
        nobs=int(result.nobs),
        n_entities=int(panel.index.get_level_values(0).nunique()),
        raw_result=result,
    )


def _psd_clip(cov: pd.DataFrame) -> pd.DataFrame:
    """Clip negative eigenvalues of the two-way clustered covariance to zero.

    The Cameron-Gelbach-Miller (2011) two-way cluster sandwich
    V = V_firm + V_month - V_white is only guaranteed positive semi-definite
    asymptotically; with a small number of clusters (83 calendar months here)
    it can yield a slightly negative variance for individual coefficients,
    which CGM note can be corrected by setting negative eigenvalues to zero
    before taking the square root. This affects only the reported standard
    errors of the raw fixed-effects fit, not the point estimates.
    """
    values = cov.to_numpy()
    eigvals, eigvecs = np.linalg.eigh((values + values.T) / 2.0)
    eigvals_clipped = np.clip(eigvals, 0.0, None)
    fixed = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    return pd.DataFrame(fixed, index=cov.index, columns=cov.columns)


def linear_contrast(
    result: PanelRegressionResult, coef_a: str, coef_b: str | None = None, weight_b: float = -1.0,
) -> tuple[float, float, float]:
    """Wald test of H0: coef_a + weight_b * coef_b = 0 using the clustered covariance.

    If ``coef_b`` is None, tests H0: coef_a = 0 (returns the raw coefficient
    test). Returns (estimate, standard error, two-sided p-value).
    """
    from scipy import stats

    if coef_b is None:
        estimate = result.params[coef_a]
        se = result.std_errors[coef_a]
    else:
        estimate = result.params[coef_a] + weight_b * result.params[coef_b]
        var = (
            result.cov.loc[coef_a, coef_a]
            + (weight_b ** 2) * result.cov.loc[coef_b, coef_b]
            + 2 * weight_b * result.cov.loc[coef_a, coef_b]
        )
        se = float(np.sqrt(var))
    z = estimate / se if se > 0 else np.nan
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    return float(estimate), float(se), float(p_value)


def complete_slope(result: PanelRegressionResult, base_coef: str, down_coef: str,
                    market_states: list[float]) -> pd.DataFrame:
    """Complete fitted ESG slope beta + d*theta_D at market-return points d (Table A3).

    Negative ``d`` (a market decline) activates Down_t = -d; the paper's
    complete slope at a decline of magnitude |d| is beta + |d| * theta_D.
    """
    rows = []
    for d in market_states:
        if d <= 0:
            estimate, se, p_value = linear_contrast_sum(result, base_coef, down_coef, abs(d))
        else:
            up_coef = down_coef.replace("down", "up") if "down" in down_coef else None
            if up_coef is not None and up_coef in result.params.index:
                estimate, se, p_value = linear_contrast_sum(result, base_coef, up_coef, d)
            else:
                estimate, se, p_value = linear_contrast_sum(result, base_coef, down_coef, 0.0)
        rows.append({"market_state": d, "slope": estimate, "se": se, "p_value": p_value})
    return pd.DataFrame(rows)


def linear_contrast_sum(result: PanelRegressionResult, coef_a: str, coef_b: str, weight_b: float):
    """H0 evaluation of coef_a + weight_b*coef_b, with weight_b >= 0 (helper for complete_slope)."""
    return linear_contrast(result, coef_a, coef_b, weight_b=weight_b)
