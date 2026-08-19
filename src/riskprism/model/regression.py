"""Constrained cross-sectional factor return regression.

Each period:  r_i = f_mkt + sum_s X_is f_s + sum_j I_ij f_j + eps_i

The market factor is the intercept; industry factor returns are
constrained to be cap-weighted mean zero so the system is identified
(the classic Barra restriction). Estimated by WLS with sqrt(mktcap)
weights, which assumes specific variance inversely proportional to
sqrt of size.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from riskprism.config import MARKET_FACTOR
from riskprism.factors.industry import industry_dummies


@dataclass
class RegressionResult:
    factor_returns: pd.Series
    residuals: pd.Series
    r2: float
    n_assets: int


def _restriction_matrix(n_styles: int, ind_weights: np.ndarray) -> np.ndarray:
    """Map free parameters to the full factor vector under the constraint
    sum_j w_j f_j = 0 over industries.

    Eliminates the largest-weight industry (numerically safest divisor).
    Column order of the full vector: [market, styles..., industries...].
    """
    n_ind = len(ind_weights)
    k_full = 1 + n_styles + n_ind
    elim = int(np.argmax(ind_weights))
    free_cols = [i for i in range(k_full) if i != 1 + n_styles + elim]
    R = np.zeros((k_full, k_full - 1))
    for reduced_idx, full_idx in enumerate(free_cols):
        R[full_idx, reduced_idx] = 1.0
    elim_row = 1 + n_styles + elim
    for j in range(n_ind):
        if j == elim:
            continue
        full_idx = 1 + n_styles + j
        reduced_idx = free_cols.index(full_idx)
        R[elim_row, reduced_idx] = -ind_weights[j] / ind_weights[elim]
    return R


def cross_sectional_regression(
    asset_returns: pd.Series,
    style_exposures: pd.DataFrame,
    industries: pd.Series,
    mktcap: pd.Series,
    min_assets: int = 50,
) -> RegressionResult:
    idx = asset_returns.index
    valid = (
        asset_returns.notna()
        & mktcap.reindex(idx).notna()
        & (mktcap.reindex(idx) > 0)
        & style_exposures.reindex(idx).notna().all(axis=1)
        & industries.reindex(idx).notna()
    )
    idx = idx[valid]
    if len(idx) < min_assets:
        raise ValueError(f"Only {len(idx)} usable assets (< {min_assets})")

    y = asset_returns.loc[idx].to_numpy(dtype=float)
    styles = style_exposures.loc[idx]
    caps = mktcap.loc[idx].to_numpy(dtype=float)
    dummies = industry_dummies(industries.loc[idx])

    ind_caps = dummies.mul(caps, axis=0).sum(axis=0).to_numpy()
    ind_weights = ind_caps / ind_caps.sum()

    X = np.column_stack([np.ones(len(idx)), styles.to_numpy(dtype=float), dummies.to_numpy()])
    factor_names = [MARKET_FACTOR, *styles.columns, *dummies.columns]

    R = _restriction_matrix(styles.shape[1], ind_weights)
    w = np.sqrt(caps)
    w = w / w.mean()
    sw = np.sqrt(w)

    g, *_ = np.linalg.lstsq(X @ R * sw[:, None], y * sw, rcond=None)
    f = R @ g
    resid = y - X @ f

    y_wmean = float(np.average(y, weights=w))
    sst = float(np.sum(w * (y - y_wmean) ** 2))
    ssr = float(np.sum(w * resid**2))
    r2 = 1.0 - ssr / sst if sst > 0 else np.nan

    return RegressionResult(
        factor_returns=pd.Series(f, index=factor_names),
        residuals=pd.Series(resid, index=idx),
        r2=r2,
        n_assets=len(idx),
    )
