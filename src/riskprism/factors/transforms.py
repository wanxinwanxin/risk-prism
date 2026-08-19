"""Exposure winsorization and standardization.

All statistics (winsorization bounds, cap-weighted mean, std) can be fit
on a subset of names — the estimation universe — and applied to the full
coverage universe, so illiquid tails don't distort the scale that liquid
names are measured on.
"""

import numpy as np
import pandas as pd


def winsorize_z(s: pd.Series, z: float = 3.0, iterations: int = 2,
                fit: pd.Index | None = None) -> pd.Series:
    """Clip values beyond ``z`` standard deviations from the mean.

    Applied iteratively so extreme outliers do not inflate the first-pass
    standard deviation. Bounds are computed on ``fit`` rows (default: all)
    and applied to every row.
    """
    out = s.astype(float).copy()
    ref = out.loc[out.index.intersection(fit)] if fit is not None else out
    for _ in range(iterations):
        mu = ref.mean()
        sd = ref.std()
        if not np.isfinite(sd) or sd == 0:
            return out
        out = out.clip(mu - z * sd, mu + z * sd)
        ref = ref.clip(mu - z * sd, mu + z * sd)
    return out


def standardize(s: pd.Series, mktcap: pd.Series,
                fit: pd.Index | None = None) -> pd.Series:
    """Standardize to cap-weighted mean 0 and equal-weighted std 1.

    The Barra convention: the cap-weighted market portfolio has zero
    exposure to every style factor. Location and scale come from ``fit``
    rows (default: all) and are applied to every row.
    """
    mktcap = mktcap.reindex(s.index)
    ref_idx = s.index.intersection(fit) if fit is not None else s.index
    sr, cr = s.loc[ref_idx], mktcap.loc[ref_idx]
    valid = sr.notna() & cr.notna() & (cr > 0)
    if valid.sum() < 2:
        return s * np.nan
    w = cr[valid] / cr[valid].sum()
    cw_mean = float((sr[valid] * w).sum())
    sd = (sr[valid] - cw_mean).std()
    if not np.isfinite(sd) or sd == 0:
        return (s - cw_mean) * np.nan
    return (s - cw_mean) / sd


def process_exposure(raw: pd.Series, mktcap: pd.Series, z: float = 3.0,
                     fit: pd.Index | None = None) -> pd.Series:
    """Winsorize, standardize, and fill missing exposures with 0 (market average)."""
    return standardize(winsorize_z(raw, z=z, fit=fit), mktcap, fit=fit).fillna(0.0)
