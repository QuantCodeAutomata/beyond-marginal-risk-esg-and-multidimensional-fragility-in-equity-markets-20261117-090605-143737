"""Shared pytest fixtures: synthetic panel data and the market_data marker."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.generate_synthetic import generate_panel
from src import config, panel_pipeline


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "market_data: test needs a live Dataloader/Massive API key"
    )


@pytest.fixture(scope="session")
def synthetic_raw_panel():
    """A small synthetic firm-month panel with a known negative ESG x Down effect."""
    return generate_panel(
        n_firms=30,
        n_months=24,
        seed=config.SYNTHETIC_SEED,
        esg_down_effect={"loss": -0.08, "log_rvol": -0.08, "log_chl": -0.08},
    )


@pytest.fixture(scope="session")
def synthetic_model_frame(synthetic_raw_panel):
    return panel_pipeline.build_model_frame(synthetic_raw_panel, esg_col="esg_ias")


@pytest.fixture(scope="session")
def dataloader_credentials():
    key = os.environ.get("DATALOADER_API_KEY")
    massive_key = os.environ.get("MASSIVE_API_KEY")
    if not key or not massive_key:
        pytest.skip("DATALOADER_API_KEY / MASSIVE_API_KEY not set")
    return {"dataloader": key, "massive": massive_key}
