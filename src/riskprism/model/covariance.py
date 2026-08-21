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


def _newey_west_variances(F: np.ndarray, half_life: float, lags: int,
                          ratio_min: float, ratio_max: float) -> np.ndarray:
    """Per-factor weekly variance with a Bartlett-weighted Newey-West
    adjustment for serial correlation: annualizing by x52 assumes iid
    weekly returns; autocorrelated factors (momentum especially) violate
    that. Applied to variances only, which keeps V·C·V trivially PSD;
    the ratio is clipped for robustness."""
    T = F.shape[0]
    lam = 0.5 ** (1.0 / half_life)
    w = lam ** np.arange(T - 1, -1, -1)
    w /= w.sum()
    Fw = np.nan_to_num(F)
    var = (w[:, None] * Fw ** 2).sum(axis=0)
    adj = var.copy()
    for l in range(1, lags + 1):
        if T <= l:
            break
        g = (w[l:, None] * Fw[l:] * Fw[:-l]).sum(axis=0)
        adj = adj + 2 * (1 - l / (lags + 1)) * g
    ratio = np.ones_like(var)
    pos = var > 0
    ratio[pos] = np.clip(adj[pos] / var[pos], ratio_min, ratio_max)
    return var * ratio


def factor_covariance(factor_returns: pd.DataFrame, config: ModelConfig,
                      vra: float = 1.0) -> pd.DataFrame:
    """Annualized factor covariance.

    Correlations use a longer half-life than volatilities (correlations
    are noisier and more stable; vols move faster) — the standard
    responsive-vol / stable-corr decomposition. Variances carry a
    Newey-West serial-correlation adjustment; the result is repaired to
    positive semi-definite by eigenvalue flooring, then scaled by the
    Volatility Regime Adjustment multiplier ``vra`` (all vols x vra, so
    the covariance scales by vra^2; correlations unchanged).
    """
    F = factor_returns.to_numpy(dtype=float)
    S_corr = _ewma_second_moment(F, config.corr_half_life)

    d = np.sqrt(np.diag(S_corr))
    d[d == 0] = np.nan
    C = S_corr / np.outer(d, d)
    C = np.nan_to_num(C)
    np.clip(C, -1.0, 1.0, out=C)
    np.fill_diagonal(C, 1.0)

    var_nw = _newey_west_variances(F, config.vol_half_life, config.nw_factor_lags,
                                   config.nw_ratio_min, config.nw_ratio_max)
    vols = np.sqrt(np.clip(var_nw, 0, None) * config.ann_factor) * vra
    cov = C * np.outer(vols, vols)

    cov = (cov + cov.T) / 2
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, config.eig_floor, None)
    cov = vecs @ np.diag(vals) @ vecs.T
    cov = (cov + cov.T) / 2

    return pd.DataFrame(cov, index=factor_returns.columns, columns=factor_returns.columns)
