"""Small helpers shared by every experiment script to save results/ artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def save_table(df: pd.DataFrame, name: str) -> Path:
    """Write ``df`` to ``results/{name}.csv`` and return the path."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def save_json(obj: dict, name: str) -> Path:
    """Write ``obj`` to ``results/{name}.json`` and return the path."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2, default=str))
    return path


def holm_family_table(tests: dict[str, tuple[float, float, float]], alpha: float = 0.10) -> pd.DataFrame:
    """Build a Holm-adjusted table from a dict of {label: (estimate, se, raw_p)}.

    Uses `src.inference.holm_adjust` (statsmodels Holm step-down).
    """
    from src.inference import holm_adjust

    labels = list(tests.keys())
    estimates = [tests[label][0] for label in labels]
    ses = [tests[label][1] for label in labels]
    raw_p = [tests[label][2] for label in labels]
    holm = holm_adjust(raw_p, alpha=alpha)
    out = pd.DataFrame({
        "test": labels,
        "estimate": estimates,
        "se": ses,
        "raw_p": raw_p,
        "holm_p": holm["holm_p"].to_numpy(),
        "reject_at_alpha": holm["reject_at_alpha"].to_numpy(),
    })
    return out
