"""Specific (idiosyncratic) risk: time-series EWMA blended with a
cross-sectional structural model.

The structural model regresses ln(EWMA residual vol) on characteristics
(size, volatility, liquidity exposures + industry) across assets with
enough history, then predicts for everyone. Each asset's final estimate
blends its own time-series estimate with the structural prediction by
history length:

    sigma_i = w_i * ts_i + (1 - w_i) * structural_i,   w_i = T_i / (T_i + T0)

Assets with no residual history — recent IPOs, names outside the
estimation universe — get the pure structural prior, so coverage never
requires asset-level history.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from riskprism.config import ModelConfig
from riskprism.factors.industry import industry_dummies

_STRUCTURAL_STYLES = ["size", "volatility", "liquidity"]
_MIN_FIT_ASSETS = 30


@dataclass
class SpecificRiskResult:
    vol: pd.Series           # blended, annualized
    ts_vol: pd.Series        # time-series EWMA (NaN where insufficient history)
    structural: pd.Series    # structural prediction
    blend_weight: pd.Series  # weight on the time-series estimate
    n_obs: pd.Series         # residual observations per asset


def ewma_specific_vol(residuals: pd.DataFrame, config: ModelConfig
                      ) -> tuple[pd.Series, pd.Series]:
    """Annualized EWMA vol of regression residuals, and per-asset obs counts."""
    lam = 0.5 ** (1.0 / config.specific_half_life)
    vols, counts = {}, {}
    for ticker in residuals.columns:
        e = residuals[ticker].dropna()
        counts[ticker] = len(e)
        if len(e) < config.min_specific_obs:
            vols[ticker] = np.nan
            continue
        w = lam ** np.arange(len(e) - 1, -1, -1)
        w /= w.sum()
        vols[ticker] = float(np.sqrt((w * e.to_numpy() ** 2).sum() * config.ann_factor))
    return pd.Series(vols, dtype=float), pd.Series(counts, dtype=float)


def specific_risk(
    residuals: pd.DataFrame,
    exposures: pd.DataFrame,
    industries: pd.Series,
    config: ModelConfig,
) -> SpecificRiskResult:
    """Blended specific vol for every asset in ``exposures``."""
    idx = exposures.index
    ts_vol, n_obs = ewma_specific_vol(residuals, config)
    ts_vol = ts_vol.reindex(idx)
    n_obs = n_obs.reindex(idx).fillna(0.0)

    feats = exposures.reindex(columns=_STRUCTURAL_STYLES).fillna(0.0)
    dummies = industry_dummies(industries.reindex(idx).fillna("Other"))
    Xs = np.column_stack([dummies.to_numpy(), feats.to_numpy(dtype=float)])

    fit = ts_vol.notna() & (ts_vol > 0)
    if fit.sum() >= _MIN_FIT_ASSETS:
        fit_pos = idx.get_indexer(idx[fit])
        y = np.log(ts_vol[fit].to_numpy())
        beta, *_ = np.linalg.lstsq(Xs[fit_pos], y, rcond=None)
        pred_ln = Xs @ beta
        # Duan smearing: correct exp() retransformation bias on the fit set
        smear = float((ts_vol[fit].to_numpy() / np.exp(pred_ln[fit_pos])).mean())
        structural = pd.Series(np.exp(pred_ln) * smear, index=idx)
    else:
        structural = pd.Series(float(ts_vol[fit].mean()) if fit.any() else np.nan, index=idx)

    w = n_obs / (n_obs + config.structural_t0)
    w = w.where(ts_vol.notna(), 0.0)
    vol = w * ts_vol.fillna(0.0) + (1 - w) * structural
    vol = vol.fillna(structural)
    return SpecificRiskResult(vol=vol, ts_vol=ts_vol, structural=structural,
                              blend_weight=w, n_obs=n_obs)
