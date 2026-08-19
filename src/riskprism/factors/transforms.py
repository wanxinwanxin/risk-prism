"""Exposure winsorization and standardization."""

import numpy as np
import pandas as pd


def winsorize_z(s: pd.Series, z: float = 3.0, iterations: int = 2) -> pd.Series:
    """Clip values beyond ``z`` standard deviations from the mean.

    Applied iteratively so extreme outliers do not inflate the first-pass
    standard deviation and let moderate outliers escape clipping.
    """
    out = s.astype(float).copy()
    for _ in range(iterations):
        mu = out.mean()
        sd = out.std()
        if not np.isfinite(sd) or sd == 0:
            return out
        out = out.clip(mu - z * sd, mu + z * sd)
    return out


def standardize(s: pd.Series, mktcap: pd.Series) -> pd.Series:
    """Standardize to cap-weighted mean 0 and equal-weighted std 1.

    The Barra convention: the cap-weighted market portfolio has zero
    exposure to every style factor, while the cross-sectional dispersion
    is unit-scaled so factor returns are comparable across styles.
    """
    mktcap = mktcap.reindex(s.index)
    valid = s.notna() & mktcap.notna() & (mktcap > 0)
    if valid.sum() < 2:
        return s * np.nan
    w = mktcap[valid] / mktcap[valid].sum()
    cw_mean = float((s[valid] * w).sum())
    centered = s - cw_mean
    sd = centered[valid].std()
    if not np.isfinite(sd) or sd == 0:
        return centered * np.nan
    return centered / sd


def process_exposure(raw: pd.Series, mktcap: pd.Series, z: float = 3.0) -> pd.Series:
    """Winsorize, standardize, and fill missing exposures with 0 (market average)."""
    return standardize(winsorize_z(raw, z=z), mktcap).fillna(0.0)
