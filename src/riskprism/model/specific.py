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


def bayes_shrink_specific(vol: pd.Series, size: pd.Series,
                          config: ModelConfig) -> pd.Series:
    """Shrink specific vols toward their size-bucket mean (USE4 Bayesian
    shrinkage, q=0.1): v = q|s - s_bar| / (delta + q|s - s_bar|), pulling
    extreme forecasts toward the bucket mean with intensity growing in
    their distance from it. Flattens the classic low-vol-underforecast /
    high-vol-overforecast tilt across vol deciles (USE4: 1.08->0.92
    unshrunk becomes ~flat at 1.0)."""
    q, nb = config.specific_shrink_q, config.specific_shrink_buckets
    ok = vol.notna() & (vol > 0) & size.notna()
    if ok.sum() < 5 * nb or q <= 0:
        return vol
    out = vol.copy()
    ranks = size[ok].rank(pct=True, method="first")
    bucket = np.minimum((ranks * nb).astype(int), nb - 1)
    for _, members in vol[ok].groupby(bucket):
        mean, delta = float(members.mean()), float(members.std(ddof=0))
        if not np.isfinite(delta) or delta <= 0:
            continue
        dist = (members - mean).abs()
        v = q * dist / (delta + q * dist)
        out[members.index] = v * mean + (1 - v) * members
    return out


@dataclass
class SpecificRiskResult:
    vol: pd.Series           # blended, annualized
    ts_vol: pd.Series        # time-series EWMA (NaN where insufficient history)
    structural: pd.Series    # structural prediction
    blend_weight: pd.Series  # weight on the time-series estimate
    n_obs: pd.Series         # residual observations per asset


def ewma_specific_vol(residuals: pd.DataFrame, config: ModelConfig
                      ) -> tuple[pd.Series, pd.Series]:
    """Annualized EWMA vol of regression residuals (with a lag-1
    Newey-West adjustment for serial correlation), and per-asset obs
    counts."""
    lam = 0.5 ** (1.0 / config.specific_half_life)
    vols, counts = {}, {}
    for ticker in residuals.columns:
        e = residuals[ticker].dropna()
        counts[ticker] = len(e)
        if len(e) < config.min_specific_obs:
            vols[ticker] = np.nan
            continue
        ev = e.to_numpy()
        w = lam ** np.arange(len(ev) - 1, -1, -1)
        w /= w.sum()
        var = float((w * ev ** 2).sum())
        if config.nw_specific_lags >= 1 and len(ev) > 1 and var > 0:
            g1 = float((w[1:] * ev[1:] * ev[:-1]).sum())
            var *= float(np.clip((var + g1) / var,
                                 config.nw_ratio_min, config.nw_ratio_max))
        vols[ticker] = float(np.sqrt(var * config.ann_factor))
    return pd.Series(vols, dtype=float), pd.Series(counts, dtype=float)


def specific_risk(
    residuals: pd.DataFrame,
    exposures: pd.DataFrame,
    industries: pd.Series,
    config: ModelConfig,
    vra: float = 1.0,
) -> SpecificRiskResult:
    """Blended specific vol for every asset in ``exposures``: time-series
    EWMA (Newey-West adjusted) blended with the structural model by
    history length, Bayesian-shrunk toward size-bucket means, then scaled
    by the specific Volatility Regime Adjustment multiplier ``vra``."""
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
    if "size" in exposures.columns:
        vol = bayes_shrink_specific(vol, exposures["size"], config)
    vol = vol * vra
    return SpecificRiskResult(vol=vol, ts_vol=ts_vol, structural=structural,
                              blend_weight=w, n_obs=n_obs)
