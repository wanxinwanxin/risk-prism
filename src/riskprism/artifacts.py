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
# Added in v0.2; loaders treat them as optional for older artifact dirs.
OPTIONAL_FILES = {
    "residuals": "residuals.parquet",             # capture-forward residual history
    "asset_meta": "asset_meta.parquet",           # per-asset estimation quality
    "fundamentals_store": "fundamentals_store.parquet",  # distilled EDGAR PIT data
}
META_FILE = "meta.json"


def save_artifacts(
    path: str | Path,
    exposures: pd.DataFrame,
    factor_covariance: pd.DataFrame,
    specific_risk: pd.Series,
    factor_returns: pd.DataFrame,
    meta: dict,
    residuals: pd.DataFrame | None = None,
    asset_meta: pd.DataFrame | None = None,
    fundamentals_store: pd.DataFrame | None = None,
) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    exposures.to_parquet(path / FILES["exposures"])
    factor_covariance.to_parquet(path / FILES["factor_covariance"])
    specific_risk.rename("specific_vol").to_frame().to_parquet(path / FILES["specific_risk"])
    factor_returns.to_parquet(path / FILES["factor_returns"])
    if residuals is not None:
        residuals.to_parquet(path / OPTIONAL_FILES["residuals"])
    if asset_meta is not None:
        asset_meta.to_parquet(path / OPTIONAL_FILES["asset_meta"])
    if fundamentals_store is not None:
        fundamentals_store.to_parquet(path / OPTIONAL_FILES["fundamentals_store"])
    (path / META_FILE).write_text(json.dumps(meta, indent=2, default=str))
    return path


def load_artifacts(path: str | Path) -> dict:
    path = Path(path)
    if not (path / META_FILE).exists():
        raise FileNotFoundError(
            f"No model artifacts at {path} — run `riskprism-build` or download a "
            "published model release."
        )
    out = {
        "exposures": pd.read_parquet(path / FILES["exposures"]),
        "factor_covariance": pd.read_parquet(path / FILES["factor_covariance"]),
        "specific_risk": pd.read_parquet(path / FILES["specific_risk"])["specific_vol"],
        "factor_returns": pd.read_parquet(path / FILES["factor_returns"]),
        "meta": json.loads((path / META_FILE).read_text()),
    }
    for key, fname in OPTIONAL_FILES.items():
        out[key] = pd.read_parquet(path / fname) if (path / fname).exists() else None
    return out
