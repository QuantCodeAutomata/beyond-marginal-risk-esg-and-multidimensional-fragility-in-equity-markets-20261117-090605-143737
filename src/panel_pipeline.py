"""Assembles the modeling frame and regressor lists for Equations 5 and 7.

Input contract (the "raw panel"): one row per (firm_id, month_id) with
outcome columns measured during month t, and every right-hand-side column
already point-in-time as of t-1 (ESG snapshot, accounting controls lagged per
Sec. 3.1, momentum, and ex ante exposures). This module only builds Down/Up,
interaction terms, winsorizes controls, and constructs Joint-state indicators;
it does not itself shift any series (data/data_loader.py and
data/generate_synthetic.py are responsible for timing).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.data_construction import down_up_state, winsorize

CONTROL_COLS = [
    "log_assets", "leverage", "cash_assets", "profitability", "investment", "momentum",
]

EXPOSURE_BLOCKS = {
    "return": "beta24",
    "volatility": "rvol12",
    "liquidity": "chl12",
}

OUTCOME_OF_CHANNEL = {
    "return": "loss",
    "volatility": "log_rvol",
    "liquidity": "log_chl",
}


def build_model_frame(raw_panel: pd.DataFrame, esg_col: str = "esg_ias") -> pd.DataFrame:
    """Add Down/Up, winsorized controls, and the primary ESG interactions to the raw panel.

    ``esg_col`` selects which ESG construct (IAS, WAKIS, E, S, G) feeds the
    `esg`, `esg_down`, `esg_up` regressors, so the same frame builder serves
    the primary models and the architecture/pillar robustness checks.
    """
    df = raw_panel.copy()
    down, up = down_up_state(df["market_return"])
    df["down"] = down
    df["up"] = up

    for col in CONTROL_COLS:
        if col in df.columns:
            df[col] = winsorize(df[col], config.WINSORIZE_LOWER_PCTL, config.WINSORIZE_UPPER_PCTL)

    df["esg"] = df[esg_col]
    df["esg_down"] = df["esg"] * df["down"]
    df["esg_up"] = df["esg"] * df["up"]

    for block in EXPOSURE_BLOCKS.values():
        if block in df.columns:
            df[f"{block}_down"] = df[block] * df["down"]
            df[f"{block}_up"] = df[block] * df["up"]
    return df


def regressors_equation5(channel: str) -> list[str]:
    """Right-hand side of Equation 5 for one marginal channel (firm/month FE absorb the rest)."""
    block = EXPOSURE_BLOCKS[channel]
    return ["esg", "esg_down", "esg_up", *CONTROL_COLS, block, f"{block}_down", f"{block}_up"]


def regressors_equation7(include_blocks: list[str] | None = None) -> list[str]:
    """Right-hand side of Equation 7 (Joint80 model); ``include_blocks`` restricts to a subset
    of {"beta24", "rvol12", "chl12"} for the progressive-exposure comparison in exp_2."""
    blocks = include_blocks if include_blocks is not None else list(EXPOSURE_BLOCKS.values())
    regressors = ["esg", "esg_down", "esg_up", *CONTROL_COLS]
    for block in blocks:
        regressors += [block, f"{block}_down", f"{block}_up"]
    return regressors


def joint_state(df: pd.DataFrame, quantile: float = config.JOINT_THRESHOLD_PRIMARY) -> pd.Series:
    """Frozen pooled-quantile Joint state indicator J^q (Eq. 3), computed on ``df`` directly."""
    thresholds = {
        "loss": df["loss"].quantile(quantile),
        "log_rvol": df["log_rvol"].quantile(quantile),
        "log_chl": df["log_chl"].quantile(quantile),
    }
    joint = (
        (df["loss"] >= thresholds["loss"]).astype(int)
        * (df["log_rvol"] >= thresholds["log_rvol"]).astype(int)
        * (df["log_chl"] >= thresholds["log_chl"]).astype(int)
    )
    return joint, thresholds


def pairwise_states(df: pd.DataFrame, quantile: float = config.JOINT_THRESHOLD_PRIMARY) -> pd.DataFrame:
    """The three 1D tails and three 2D pairwise intersections used by the fragility ladder (exp_5)."""
    q_loss = df["loss"].quantile(quantile)
    q_rvol = df["log_rvol"].quantile(quantile)
    q_chl = df["log_chl"].quantile(quantile)
    tail_loss = (df["loss"] >= q_loss).astype(int)
    tail_rvol = (df["log_rvol"] >= q_rvol).astype(int)
    tail_chl = (df["log_chl"] >= q_chl).astype(int)
    return pd.DataFrame({
        "tail_return": tail_loss,
        "tail_rvol": tail_rvol,
        "tail_chl": tail_chl,
        "pair_return_rvol": tail_loss * tail_rvol,
        "pair_return_chl": tail_loss * tail_chl,
        "pair_rvol_chl": tail_rvol * tail_chl,
    })
