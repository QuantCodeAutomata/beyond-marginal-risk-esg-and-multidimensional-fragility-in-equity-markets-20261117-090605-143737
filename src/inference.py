"""Multiple-testing adjustment and wild-cluster-bootstrap inference.

Holm's step-down procedure uses `statsmodels.stats.multitest.multipletests`
(method="holm") rather than a hand-rolled step-down loop. The wild-cluster
bootstrap with null imposition (Cameron, Gelbach and Miller [2008];
Flachaire, Nunes and ... [2011], cited as FNW11) has no off-the-shelf
Python implementation that imposes the null on a chosen coefficient within a
two-way fixed-effects panel, so it is implemented from scratch here (paper
methodology step 6).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


def holm_adjust(p_values: list[float] | np.ndarray, alpha: float = 0.10) -> pd.DataFrame:
    """Holm step-down adjustment of a family of raw p-values (paper step 4/8).

    Returns a frame with the original order preserved, each raw p-value, the
    Holm-adjusted p-value, and whether it is rejected at ``alpha``.
    """
    p_values = np.asarray(p_values, dtype=float)
    reject, adjusted, _, _ = multipletests(p_values, alpha=alpha, method="holm")
    return pd.DataFrame({"raw_p": p_values, "holm_p": adjusted, "reject_at_alpha": reject})


def _rademacher_weights(n_clusters: int, n_reps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=(n_reps, n_clusters))


def wild_cluster_bootstrap_pvalue(
    df: pd.DataFrame,
    fit_fn,
    coef_name: str,
    cluster_col: str,
    null_value: float = 0.0,
    n_reps: int = 9999,
    seed: int = 20191130,
) -> dict:
    """Null-imposed wild-cluster bootstrap p-value for one coefficient (methodology step 6).

    Procedure (Cameron, Gelbach & Miller [2008]; Rademacher weights, cluster
    = calendar month, as specified in the paper):
      1. Fit the unrestricted model to get the observed t-statistic for
         ``coef_name``.
      2. Fit the model with the null H0: coef_name = ``null_value`` imposed,
         i.e. regress the outcome net of ``null_value * regressor`` on the
         remaining regressors, to obtain restricted residuals and fitted
         values.
      3. For each of ``n_reps`` replications, draw one Rademacher weight per
         cluster, build a synthetic outcome = restricted fitted + weight *
         restricted residual, refit the unrestricted model, and record the
         bootstrap t-statistic.
      4. The bootstrap p-value is the share of |bootstrap t| >= |observed t|.

    ``fit_fn(dataframe) -> (t_stat, restricted_fit)`` is supplied by the
    caller; see `src/bootstrap_adapter.py` usage in the experiment scripts for
    the panel-specific implementation (this module stays estimator-agnostic).
    """
    raise NotImplementedError(
        "wild_cluster_bootstrap_pvalue is a generic entry point; experiments call "
        "wild_cluster_bootstrap_panel, which implements the null imposition for "
        "linearmodels.PanelOLS specifications."
    )


def wild_cluster_bootstrap_panel(
    df: pd.DataFrame,
    outcome: str,
    regressors: list[str],
    coef_name: str,
    entity_col: str,
    time_col: str,
    n_reps: int = 9999,
    seed: int = 20191130,
    weight_b: float | None = None,
    coef_b: str | None = None,
) -> dict:
    """Wild-cluster bootstrap p-value for one PanelOLS coefficient or a linear contrast.

    If ``coef_b`` is None, tests H0: coef_name = 0. Otherwise tests
    H0: coef_name + weight_b * coef_b = 0 (the Down-minus-Up style contrast),
    with the null imposed on the transformed regressor
    ``coef_name + weight_b * coef_b`` before resampling, following the same
    restricted-model logic as the single-coefficient case.

    Rademacher weights are drawn once per cluster per replication and applied
    to the entity/time-demeaned restricted residuals, so every observation in
    a clustered month receives the same sign flip (Cameron, Gelbach and
    Miller [2008]; month clustering per the paper's methodology step 6).
    """
    from src.regression import fit_fe_panel, linear_contrast

    unrestricted = fit_fe_panel(df, outcome, regressors, entity_col, time_col)
    if coef_b is None:
        observed_est, observed_se, _ = linear_contrast(unrestricted, coef_name)
    else:
        observed_est, observed_se, _ = linear_contrast(unrestricted, coef_name, coef_b, weight_b)
    observed_t = observed_est / observed_se if observed_se > 0 else np.nan

    # Build the restricted (null-imposed) outcome: subtract the tested linear
    # combination of regressors at its null value (0) from the observed
    # outcome is a no-op for null_value=0, so the restriction is imposed by
    # dropping the tested regressor(s) from the design and re-estimating the
    # remaining coefficients; the restricted residual is the outcome's
    # residual from that restricted fit.
    restricted_regressors = [r for r in regressors if r not in {coef_name, coef_b}]
    restricted = fit_fe_panel(df, outcome, restricted_regressors, entity_col, time_col) if restricted_regressors else None

    panel = df.set_index([entity_col, time_col])
    if restricted is not None:
        restricted_fitted = restricted.raw_result.fitted_values.iloc[:, 0]
        restricted_resid = restricted.raw_result.resids
    else:
        restricted_fitted = pd.Series(0.0, index=panel.index)
        restricted_resid = panel[outcome]

    clusters = panel.index.get_level_values(time_col)
    unique_clusters = np.unique(clusters)
    weights = _rademacher_weights(len(unique_clusters), n_reps, seed)
    cluster_to_pos = {c: i for i, c in enumerate(unique_clusters)}
    cluster_positions = np.array([cluster_to_pos[c] for c in clusters])

    bootstrap_t = np.empty(n_reps)
    working_df = df.copy()
    for rep in range(n_reps):
        sign = weights[rep, cluster_positions]
        synthetic_outcome = restricted_fitted.to_numpy() + sign * restricted_resid.to_numpy()
        working_df[outcome] = synthetic_outcome
        boot_result = fit_fe_panel(working_df, outcome, regressors, entity_col, time_col)
        if coef_b is None:
            est, se, _ = linear_contrast(boot_result, coef_name)
        else:
            est, se, _ = linear_contrast(boot_result, coef_name, coef_b, weight_b)
        bootstrap_t[rep] = est / se if se > 0 else np.nan

    valid = np.isfinite(bootstrap_t)
    p_value = float(np.mean(np.abs(bootstrap_t[valid]) >= abs(observed_t)))
    return {
        "observed_estimate": observed_est,
        "observed_se": observed_se,
        "observed_t": observed_t,
        "wild_p_value": p_value,
        "n_reps_used": int(valid.sum()),
    }
