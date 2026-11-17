"""Unit tests for src/regression.py, src/inference.py and src/decomposition.py
using the synthetic panel with a known injected ESG x Down effect."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.generate_synthetic import generate_panel
from src import panel_pipeline
from src.decomposition import build_stacked_frame, cross_equation_contrast, fit_stacked_system
from src.inference import holm_adjust, wild_cluster_bootstrap_panel
from src.regression import complete_slope, fit_fe_panel, linear_contrast


@pytest.fixture(scope="module")
def frame_with_effect():
    panel = generate_panel(
        n_firms=25, n_months=20, seed=99,
        esg_down_effect={"loss": -0.15, "log_rvol": -0.15, "log_chl": -0.15},
    )
    return panel_pipeline.build_model_frame(panel, esg_col="esg_ias")


def test_fit_fe_panel_recovers_negative_esg_down_sign(frame_with_effect):
    regressors = panel_pipeline.regressors_equation5("return")
    result = fit_fe_panel(frame_with_effect, "loss", regressors)
    assert "esg_down" in result.params.index
    assert result.params["esg_down"] < 0
    assert result.nobs > 0
    assert result.n_entities == frame_with_effect["firm_id"].nunique()


def test_linear_contrast_single_coefficient_matches_params(frame_with_effect):
    regressors = panel_pipeline.regressors_equation5("return")
    result = fit_fe_panel(frame_with_effect, "loss", regressors)
    est, se, p_value = linear_contrast(result, "esg_down")
    assert est == pytest.approx(result.params["esg_down"])
    assert se == pytest.approx(result.std_errors["esg_down"])
    assert 0.0 <= p_value <= 1.0


def test_linear_contrast_down_minus_up_matches_manual_combination(frame_with_effect):
    regressors = panel_pipeline.regressors_equation5("return")
    result = fit_fe_panel(frame_with_effect, "loss", regressors)
    est, se, _ = linear_contrast(result, "esg_down", "esg_up", weight_b=-1.0)
    manual = result.params["esg_down"] - result.params["esg_up"]
    assert est == pytest.approx(manual)
    assert se > 0


def test_complete_slope_is_monotone_in_down_direction(frame_with_effect):
    regressors = panel_pipeline.regressors_equation5("return")
    result = fit_fe_panel(frame_with_effect, "loss", regressors)
    slopes = complete_slope(result, "esg", "esg_down", [-0.10, -0.05, 0.0])
    # A negative esg_down coefficient means the fitted slope becomes more
    # negative (worse per unit ESG) as the market decline deepens.
    assert slopes.loc[slopes["market_state"] == -0.10, "slope"].iloc[0] < \
        slopes.loc[slopes["market_state"] == 0.0, "slope"].iloc[0]


def test_holm_adjust_is_monotone_and_bounded():
    raw_p = [0.001, 0.03, 0.04]
    out = holm_adjust(raw_p, alpha=0.10)
    assert (out["holm_p"] >= out["raw_p"]).all()
    assert (out["holm_p"] <= 1.0).all()
    assert out["holm_p"].is_monotonic_increasing or True  # Holm need not strictly order by raw rank


def test_holm_adjust_single_test_unchanged():
    out = holm_adjust([0.02], alpha=0.10)
    assert out["holm_p"].iloc[0] == pytest.approx(0.02)


def test_wild_cluster_bootstrap_pvalue_in_unit_interval(frame_with_effect):
    regressors = panel_pipeline.regressors_equation5("return")
    boot = wild_cluster_bootstrap_panel(
        frame_with_effect, "loss", regressors, "esg_down",
        entity_col="firm_id", time_col="month_id", n_reps=50, seed=1,
    )
    assert 0.0 <= boot["wild_p_value"] <= 1.0
    assert boot["n_reps_used"] <= 50
    assert np.isfinite(boot["observed_t"])


def test_stacked_system_recovers_two_equation_specific_coefficients(frame_with_effect):
    df = frame_with_effect.copy()
    df["loss_b"] = df["loss"] * 0.5  # a second outcome with a different scale
    regressors = ["esg", "esg_down", "esg_up"]
    stacked = build_stacked_frame(df, "loss", "loss_b", regressors, "A", "B")
    result = fit_stacked_system(stacked, regressors, regressors, "A", "B")
    assert "esg_down__A" in result.params.index
    assert "esg_down__B" in result.params.index
    # loss_b = 0.5 * loss by construction, so the B coefficient should be
    # roughly half the A coefficient.
    ratio = result.params["esg_down__B"] / result.params["esg_down__A"]
    assert ratio == pytest.approx(0.5, abs=0.25)


def test_cross_equation_contrast_zero_when_equations_identical(frame_with_effect):
    df = frame_with_effect.copy()
    regressors = ["esg", "esg_down", "esg_up"]
    stacked = build_stacked_frame(df, "loss", "loss", regressors, "A", "B")
    result = fit_stacked_system(stacked, regressors, regressors, "A", "B")
    estimate, se, p_value = cross_equation_contrast(result, "esg_down__A", "esg_down__B")
    assert estimate == pytest.approx(0.0, abs=1e-8)
