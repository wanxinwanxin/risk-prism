"""Versioned model artifact serialization.

An artifact directory is the distributable unit of the model:
    exposures.parquet         asset x factor exposure matrix (final date)
    factor_covariance.parquet K x K annualized factor covariance
    specific_risk.parquet     per-asset annualized specific vol
    factor_returns.parquet    weekly factor return history
    meta.json                 model version, build date, config, coverage
"""

import json
from pathlib import Path

import pandas as pd

FILES = {
    "exposures": "exposures.parquet",
    "factor_covariance": "factor_covariance.parquet",
    "specific_risk": "specific_risk.parquet",
    "factor_returns": "factor_returns.parquet",
}
META_FILE = "meta.json"


def save_artifacts(
    path: str | Path,
    exposures: pd.DataFrame,
    factor_covariance: pd.DataFrame,
    specific_risk: pd.Series,
    factor_returns: pd.DataFrame,
    meta: dict,
) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    exposures.to_parquet(path / FILES["exposures"])
    factor_covariance.to_parquet(path / FILES["factor_covariance"])
    specific_risk.rename("specific_vol").to_frame().to_parquet(path / FILES["specific_risk"])
    factor_returns.to_parquet(path / FILES["factor_returns"])
    (path / META_FILE).write_text(json.dumps(meta, indent=2, default=str))
    return path


def load_artifacts(path: str | Path) -> dict:
    path = Path(path)
    if not (path / META_FILE).exists():
        raise FileNotFoundError(
            f"No model artifacts at {path} — run `riskprism-build` or download a "
            "published model release."
        )
    return {
        "exposures": pd.read_parquet(path / FILES["exposures"]),
        "factor_covariance": pd.read_parquet(path / FILES["factor_covariance"]),
        "specific_risk": pd.read_parquet(path / FILES["specific_risk"])["specific_vol"],
        "factor_returns": pd.read_parquet(path / FILES["factor_returns"]),
        "meta": json.loads((path / META_FILE).read_text()),
    }
