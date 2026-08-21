"""Historical risk models, reconstructed from the artifacts.

The artifacts carry everything needed to stand the model up at any
historical week: `exposure_history.parquet` has the formation-date style
exposures and market caps; the factor-return and residual histories
replay the point-in-time EWMA state deterministically (the same state
the in-build validation used, so `model_asof` forecasts match the
shipped validation scores). Industry membership uses the final build's
classification — industry changes are rare enough to accept.

    from riskprism.model.asof import model_asof
    m = model_asof("artifacts", "2025-06-06")
    m.portfolio_risk({"AAPL": 0.5, "MSFT": 0.5})
"""

from pathlib import Path

import numpy as np
import pandas as pd

from riskprism.artifacts import load_artifacts
from riskprism.config import MARKET_FACTOR, ModelConfig
from riskprism.factors.industry import INDUSTRY_PREFIX, industry_dummies
from riskprism.model.validation import FULL_FACTORS, RunningRiskState
from riskprism.risk import RiskModel


def available_dates(artifacts: dict | str | Path) -> list[pd.Timestamp]:
    a = artifacts if isinstance(artifacts, dict) else load_artifacts(artifacts)
    eh = a.get("exposure_history")
    if eh is None or eh.empty:
        return []
    return sorted(pd.to_datetime(eh["date"].unique()))


def replay_state_through(a: dict, date: pd.Timestamp,
                         config: ModelConfig) -> RunningRiskState:
    """EWMA state using factor returns / residuals with index <= date."""
    state = RunningRiskState(config)
    fr, res = a["factor_returns"], a["residuals"]
    for d, row in fr.iterrows():
        if d > date:
            break
        e = res.loc[d].dropna() if res is not None and d in res.index else pd.Series(dtype=float)
        state.update(row, e)
    return state


def model_asof(artifacts: dict | str | Path, date, config: ModelConfig | None = None) -> RiskModel:
    """Reconstruct the risk model as it stood at a historical formation date.

    ``date`` must be one of `available_dates` (a weekly formation Friday).
    Covers the estimation universe as of that week. Not identical to the
    final batch estimator (recursive EWMA vs. windowed; specific risk is
    pure time-series here, no structural blend) — it is exactly the model
    the in-build validation was scored with.
    """
    config = config or ModelConfig()
    a = artifacts if isinstance(artifacts, dict) else load_artifacts(artifacts)
    date = pd.Timestamp(date)
    eh = a.get("exposure_history")
    if eh is None or eh.empty:
        raise ValueError("These artifacts carry no exposure history (pre-v0.2.1 build)")
    snap = eh[eh["date"] == date]
    if snap.empty:
        raise ValueError(f"{date.date()} is not a formation date; "
                         f"see riskprism.model.asof.available_dates()")
    snap = snap.set_index("ticker")

    am = a.get("asset_meta")
    industries = (am["industry"].reindex(snap.index).fillna("Other")
                  if am is not None else pd.Series("Other", index=snap.index))

    style_cols = [c for c in snap.columns if c not in ("date", "mktcap")]
    X = pd.concat(
        [pd.Series(1.0, index=snap.index, name=MARKET_FACTOR),
         snap[style_cols].astype(float),
         industry_dummies(industries)],
        axis=1,
    ).reindex(columns=FULL_FACTORS).fillna(0.0)

    state = replay_state_through(a, date, config)
    F = pd.DataFrame(state.factor_cov_weekly() * config.ann_factor,
                     index=FULL_FACTORS, columns=FULL_FACTORS)
    size = snap["size"].astype(float) if "size" in snap.columns else None
    spec = np.sqrt(state.specific_var_weekly(snap.index, size=size) * config.ann_factor)

    meta = dict(a["meta"])
    meta.update({"as_of": str(date.date()), "historical": True,
                 "n_assets": int(len(X)),
                 "note": "reconstructed point-in-time model (recursive EWMA, "
                         "time-series specific risk); industries as of final build"})
    return RiskModel(X, F, spec, meta, asset_meta=None)
