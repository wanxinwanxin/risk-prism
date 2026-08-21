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


def pca_blend_corr(C: np.ndarray, config: ModelConfig) -> np.ndarray:
    """Bloomberg-style correlation blending (Menchero, MAC2/MAC3): blend
    the sample correlation with its own rank-J PCA reconstruction (plus an
    idiosyncratic diagonal that restores unit diagonals). Suppresses the
    noise in the correlation matrix's small directions that optimizers
    exploit, while the sample term keeps forecasts unbiased."""
    K = C.shape[0]
    J = max(1, int(np.ceil(config.blend_components_frac * K)))
    vals, vecs = np.linalg.eigh((C + C.T) / 2)
    top = vecs[:, -J:] * np.sqrt(np.clip(vals[-J:], 0, None))
    C_low = top @ top.T
    C_p = C_low + np.diag(np.clip(1.0 - np.diag(C_low), 0.0, None))
    Cb = config.blend_weight * C + (1 - config.blend_weight) * C_p
    d = np.sqrt(np.clip(np.diag(Cb), 1e-12, None))
    Cb = Cb / np.outer(d, d)
    np.clip(Cb, -1.0, 1.0, out=Cb)
    np.fill_diagonal(Cb, 1.0)
    return Cb


def estimate_weekly_cov(F: np.ndarray, config: ModelConfig,
                        blend: bool | None = None) -> np.ndarray:
    """The core weekly-covariance estimator (EWMA vol/corr split + NW),
    factored out so the eigenfactor Monte Carlo can debias exactly the
    estimator the model actually uses."""
    S_corr = _ewma_second_moment(F, config.corr_half_life)
    d = np.sqrt(np.diag(S_corr))
    d[d == 0] = np.nan
    C = S_corr / np.outer(d, d)
    C = np.nan_to_num(C)
    np.clip(C, -1.0, 1.0, out=C)
    np.fill_diagonal(C, 1.0)
    if blend is None:
        blend = config.factor_cov_adjust == "blend"
    if blend:
        C = pca_blend_corr(C, config)
    var_nw = _newey_west_variances(F, config.vol_half_life, config.nw_factor_lags,
                                   config.nw_ratio_min, config.nw_ratio_max)
    vols = np.sqrt(np.clip(var_nw, 0, None))
    return C * np.outer(vols, vols)


def eigen_bias_profile(cov_weekly: np.ndarray, T: int, config: ModelConfig,
                       seed: int = 0) -> np.ndarray:
    """Per-eigenfactor volatility bias v(k), by ascending-eigenvalue rank
    (Menchero, Wang & Orr 2011, Appendix A): treat the estimated
    covariance as truth, simulate T periods, re-estimate with the SAME
    estimator, and measure how much each simulated eigenfactor's true
    variance exceeds its estimated one. Small eigenfactors come out
    underestimated (v > 1) — the optimization bias. The empirical scaling
    a = 1.4 corrects for non-normality/non-stationarity the Gaussian MC
    can't see (their Eq. A6)."""
    K = cov_weekly.shape[0]
    vals, vecs = np.linalg.eigh((cov_weekly + cov_weekly.T) / 2)
    L = vecs * np.sqrt(np.clip(vals, 1e-14, None))
    rng = np.random.default_rng(seed)
    ratio = np.zeros(K)
    M = config.eigen_adjust_sims
    for _ in range(M):
        sim = rng.standard_normal((T, K)) @ L.T
        Fm = estimate_weekly_cov(sim, config, blend=False)
        vm, um = np.linalg.eigh((Fm + Fm.T) / 2)
        d_true = np.einsum("ji,jk,ki->i", um, cov_weekly, um)
        ratio += d_true / np.clip(vm, 1e-14, None)
    v = np.sqrt(ratio / M)
    v = config.eigen_adjust_a * (v - 1.0) + 1.0
    return np.clip(v, 0.5, 3.0)


def eigen_adjust(cov_weekly: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Scale the covariance's eigenvariances by the bias profile v(k)²
    (matched by ascending-eigenvalue rank) and rotate back."""
    vals, vecs = np.linalg.eigh((cov_weekly + cov_weekly.T) / 2)
    scaled = np.clip(vals, 0, None) * v ** 2
    return vecs @ np.diag(scaled) @ vecs.T


def factor_covariance(factor_returns: pd.DataFrame, config: ModelConfig,
                      vra: float = 1.0) -> pd.DataFrame:
    """Annualized factor covariance.

    Correlations use a longer half-life than volatilities (correlations
    are noisier and more stable; vols move faster) — the standard
    responsive-vol / stable-corr decomposition. Variances carry a
    Newey-West serial-correlation adjustment. Depending on
    ``config.factor_cov_adjust`` the matrix then gets the eigenfactor
    risk adjustment ("eigen") or correlation blending ("blend") to remove
    the optimization bias optimizers exploit. Finally: PSD repair by
    eigenvalue flooring and the Volatility Regime Adjustment multiplier
    ``vra`` (all vols x vra, so covariance x vra²; correlations
    unchanged).
    """
    F = factor_returns.to_numpy(dtype=float)
    cov_w = estimate_weekly_cov(F, config)
    if config.factor_cov_adjust == "eigen":
        prof = eigen_bias_profile(cov_w, len(F), config)
        cov_w = eigen_adjust(cov_w, prof)
    cov = cov_w * config.ann_factor * vra ** 2

    cov = (cov + cov.T) / 2
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, config.eig_floor, None)
    cov = vecs @ np.diag(vals) @ vecs.T
    cov = (cov + cov.T) / 2

    return pd.DataFrame(cov, index=factor_returns.columns, columns=factor_returns.columns)
