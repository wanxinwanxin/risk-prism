"""Factor covariance estimation: EWMA with separate vol/correlation half-lives."""

import numpy as np
import pandas as pd

from riskprism.config import ModelConfig


def _ewma_second_moment(F: np.ndarray, half_life: float) -> np.ndarray:
    """Zero-mean exponentially weighted second-moment matrix, newest-weighted."""
    T = F.shape[0]
    lam = 0.5 ** (1.0 / half_life)
    w = lam ** np.arange(T - 1, -1, -1)
    w /= w.sum()
    Fw = np.nan_to_num(F)
    return (Fw * w[:, None]).T @ Fw


def factor_covariance(factor_returns: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    """Annualized factor covariance.

    Correlations use a longer half-life than volatilities (correlations
    are noisier and more stable; vols move faster) — the standard
    responsive-vol / stable-corr decomposition. The result is repaired
    to positive semi-definite by eigenvalue flooring.
    """
    F = factor_returns.to_numpy(dtype=float)
    S_corr = _ewma_second_moment(F, config.corr_half_life)
    S_vol = _ewma_second_moment(F, config.vol_half_life)

    d = np.sqrt(np.diag(S_corr))
    d[d == 0] = np.nan
    C = S_corr / np.outer(d, d)
    C = np.nan_to_num(C)
    np.clip(C, -1.0, 1.0, out=C)
    np.fill_diagonal(C, 1.0)

    vols = np.sqrt(np.clip(np.diag(S_vol), 0, None) * config.ann_factor)
    cov = C * np.outer(vols, vols)

    cov = (cov + cov.T) / 2
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, config.eig_floor, None)
    cov = vecs @ np.diag(vals) @ vecs.T
    cov = (cov + cov.T) / 2

    return pd.DataFrame(cov, index=factor_returns.columns, columns=factor_returns.columns)
