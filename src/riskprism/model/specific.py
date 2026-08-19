"""Specific (idiosyncratic) risk estimation with cross-sectional shrinkage."""

import numpy as np
import pandas as pd

from riskprism.config import ModelConfig


def specific_risk(
    residuals: pd.DataFrame, mktcap: pd.Series, config: ModelConfig
) -> pd.Series:
    """Annualized specific volatility per asset.

    Time-series EWMA of squared regression residuals, shrunk toward the
    asset's size-bucket mean. Assets with too little history get the
    bucket mean outright — a crude structural fallback that keeps
    coverage complete.
    """
    lam = 0.5 ** (1.0 / config.specific_half_life)

    own = {}
    for ticker in residuals.columns:
        e = residuals[ticker].dropna()
        if len(e) < config.min_specific_obs:
            own[ticker] = np.nan
            continue
        w = lam ** np.arange(len(e) - 1, -1, -1)
        w /= w.sum()
        var = float((w * e.to_numpy() ** 2).sum())
        own[ticker] = np.sqrt(var * config.ann_factor)
    own = pd.Series(own)

    caps = mktcap.reindex(own.index)
    log_cap = np.log(caps.where(caps > 0))
    try:
        buckets = pd.qcut(log_cap, config.n_size_buckets, labels=False, duplicates="drop")
    except ValueError:
        buckets = pd.Series(0, index=own.index)
    bucket_mean = own.groupby(buckets).transform("mean")
    global_mean = own.mean()
    bucket_mean = bucket_mean.fillna(global_mean)

    gamma = config.specific_shrinkage
    shrunk = (1 - gamma) * own + gamma * bucket_mean
    shrunk = shrunk.fillna(bucket_mean).fillna(global_mean)
    return shrunk
