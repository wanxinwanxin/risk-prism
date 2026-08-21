"""Tests for the v0.3 risk machinery: Newey-West variance adjustment,
Volatility Regime Adjustment, Bayesian specific shrinkage, optimized test
portfolios, and the full-history validation replay."""

import numpy as np
import pandas as pd

from riskprism.config import MARKET_FACTOR, STYLE_FACTORS, ModelConfig
from riskprism.factors.industry import industry_dummies
from riskprism.model.covariance import factor_covariance
from riskprism.model.revalidate import revalidate_history
from riskprism.model.specific import bayes_shrink_specific
from riskprism.model.validation import (FULL_FACTORS, RunningRiskState,
                                        optimized_portfolios)

CFG = ModelConfig()


def _fr(t=200, seed=0, ar=0.0, sigma=0.01):
    """Factor-return frame, optionally AR(1) in every factor."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-06", periods=t, freq="W-FRI")
    eps = rng.normal(0, sigma, (t, len(FULL_FACTORS)))
    x = np.zeros_like(eps)
    for i in range(t):
        x[i] = ar * (x[i - 1] if i else 0) + eps[i]
    return pd.DataFrame(x, index=dates, columns=FULL_FACTORS)


# ---------------------------------------------------------------- Newey-West
def test_newey_west_raises_variance_for_positive_autocorr():
    iid = factor_covariance(_fr(ar=0.0), CFG)
    ar = factor_covariance(_fr(ar=0.4), CFG)
    # AR(1) with rho=0.4: true long-run variance is (1+rho)/(1-rho) ~ 2.3x
    # the innovation-driven sample variance; NW with 2 lags recovers a
    # meaningful part of that on average across factors
    ratio = np.diag(ar.to_numpy()).mean() / np.diag(iid.to_numpy()).mean()
    no_nw = ModelConfig(nw_factor_lags=0)
    ar_no = factor_covariance(_fr(ar=0.4), no_nw)
    ratio_no = np.diag(ar_no.to_numpy()).mean() / np.diag(iid.to_numpy()).mean()
    assert ratio > ratio_no * 1.15  # NW lifts autocorrelated variances


def test_newey_west_near_noop_for_iid():
    with_nw = factor_covariance(_fr(seed=3), CFG)
    without = factor_covariance(_fr(seed=3), ModelConfig(nw_factor_lags=0))
    d1, d0 = np.diag(with_nw.to_numpy()), np.diag(without.to_numpy())
    assert np.abs(d1 / d0 - 1).mean() < 0.25  # only noise-level changes


# ----------------------------------------------------------------------- VRA
def test_vra_multiplier_rises_in_high_vol_regime():
    rng = np.random.default_rng(1)
    state = RunningRiskState(CFG)
    calm = 0.01
    for i in range(60):
        f = pd.Series(rng.normal(0, calm, len(FULL_FACTORS)), index=FULL_FACTORS)
        state.update(f, pd.Series(dtype=float))
    lam_calm = state.vra_factor
    for i in range(8):  # vol doubles: forecasts lag, VRA should catch it
        f = pd.Series(rng.normal(0, 2.5 * calm, len(FULL_FACTORS)), index=FULL_FACTORS)
        state.update(f, pd.Series(dtype=float))
    assert state.vra_factor > lam_calm + 0.1
    assert state.vra_factor > 1.05


def test_vra_near_one_in_calibrated_world():
    rng = np.random.default_rng(2)
    state = RunningRiskState(CFG)
    for i in range(150):
        f = pd.Series(rng.normal(0, 0.01, len(FULL_FACTORS)), index=FULL_FACTORS)
        state.update(f, pd.Series(dtype=float))
    assert 0.85 < state.vra_factor < 1.15


# ---------------------------------------------------------------- shrinkage
def test_bayes_shrinkage_pulls_extremes_toward_bucket_mean():
    rng = np.random.default_rng(3)
    n = 400
    idx = pd.Index([f"T{i:03d}" for i in range(n)])
    size = pd.Series(rng.normal(0, 1, n), index=idx)
    vol = pd.Series(0.30 + rng.normal(0, 0.08, n), index=idx).clip(lower=0.02)
    shrunk = bayes_shrink_specific(vol, size, CFG)
    # dispersion falls, mean roughly preserved, order preserved within reason
    assert shrunk.std() < vol.std()
    assert abs(shrunk.mean() - vol.mean()) < 0.01
    # extremes move the most
    hi = vol.nlargest(20).index
    assert (vol[hi] - shrunk[hi]).min() > 0


def test_shrinkage_noop_when_disabled():
    rng = np.random.default_rng(4)
    idx = pd.Index([f"T{i}" for i in range(200)])
    vol = pd.Series(rng.uniform(0.1, 0.6, 200), index=idx)
    size = pd.Series(rng.normal(0, 1, 200), index=idx)
    out = bayes_shrink_specific(vol, size, ModelConfig(specific_shrink_q=0.0))
    assert out.equals(vol)


# ------------------------------------------------------ optimized portfolios
def _model_pieces(n=300, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.Index([f"T{i:03d}" for i in range(n)])
    X = pd.DataFrame(rng.normal(0, 1, (n, len(FULL_FACTORS))),
                     index=idx, columns=FULL_FACTORS)
    X[MARKET_FACTOR] = 1.0
    A = rng.normal(0, 0.01, (len(FULL_FACTORS), len(FULL_FACTORS)))
    F = A @ A.T + 1e-6 * np.eye(len(FULL_FACTORS))
    svar = pd.Series(rng.uniform(0.0005, 0.003, n), index=idx)
    caps = pd.Series(rng.lognormal(20, 1.5, n), index=idx)
    return X, F, svar, caps


def test_optimized_weights_match_dense_solution():
    X, F, svar, caps = _model_pieces(n=250)
    cfg = ModelConfig(opt_universe=250, opt_random_alphas=1)
    ports = optimized_portfolios(X, F, svar, caps, week_index=7, config=cfg)
    assert set(ports) == {"opt_minvar", "opt_alpha1"}
    # dense check: w ∝ Σ^{-1} 1
    Xn = X.to_numpy()
    Sigma = Xn @ F @ Xn.T + np.diag(svar.to_numpy())
    w_dense = np.linalg.solve(Sigma, np.ones(len(X)))
    w_dense /= np.abs(w_dense).sum()
    w = ports["opt_minvar"].reindex(X.index).to_numpy()
    assert np.allclose(w, w_dense, atol=1e-8)


def test_optimized_minvar_beats_market_variance_under_model():
    X, F, svar, caps = _model_pieces()
    cfg = ModelConfig(opt_universe=300)
    ports = optimized_portfolios(X, F, svar, caps, week_index=1, config=cfg)
    w = ports["opt_minvar"].reindex(X.index).fillna(0.0).to_numpy()
    wm = (caps / caps.sum()).to_numpy()
    Xn = X.to_numpy()

    def pvar(v):
        x = Xn.T @ v
        return x @ F @ x + (v ** 2 * svar.to_numpy()).sum()

    # scale both to unit gross for a fair comparison
    assert pvar(w / np.abs(w).sum()) < pvar(wm / np.abs(wm).sum())


# ------------------------------------------------------------- revalidation
def test_revalidate_reconstructs_and_scores():
    rng = np.random.default_rng(6)
    t, n = 80, 150
    dates = pd.date_range("2024-01-05", periods=t, freq="W-FRI")
    idx = pd.Index([f"T{i:03d}" for i in range(n)])
    fr = pd.DataFrame(rng.normal(0, 0.008, (t, len(FULL_FACTORS))),
                      index=dates, columns=FULL_FACTORS)
    fr[MARKET_FACTOR] = rng.normal(0, 0.02, t)
    res = pd.DataFrame(rng.normal(0, 0.03, (t, n)), index=dates, columns=idx)
    industries = pd.Series(rng.choice(["BusEq", "Hlth", "Money", "Shops"], n), index=idx)
    eh_rows = []
    for d in dates:
        snap = pd.DataFrame(rng.normal(0, 1, (n, len(STYLE_FACTORS))),
                            index=idx, columns=STYLE_FACTORS).astype("float32")
        snap.insert(0, "date", d - pd.Timedelta(days=7))
        snap.insert(1, "mktcap", rng.lognormal(20, 1, n).astype("float32"))
        eh_rows.append(snap.reset_index(names="ticker"))
    eh = pd.concat(eh_rows, ignore_index=True)

    validation, state = revalidate_history(fr, res, eh, industries, CFG)
    assert len(validation) > 0
    assert state.n_weeks == t
    groups = set(validation["group"])
    assert {"market", "equal", "opt"} <= groups
    # scored weeks start only after the warm-up
    assert validation["date"].nunique() <= t - 26
    # calibrated world: overall bias near 1
    bias = validation["z"].std()
    assert 0.75 < bias < 1.3


# ------------------------------------------- v0.4: optimization-bias fixes
def test_blend_preserves_unit_diag_and_psd():
    fr = _fr(t=150, seed=9, ar=0.1)
    from riskprism.model.covariance import estimate_weekly_cov, pca_blend_corr
    cfg = ModelConfig()
    cov = estimate_weekly_cov(fr.to_numpy(), cfg, blend=False)
    d = np.sqrt(np.diag(cov))
    Cb = pca_blend_corr(cov / np.outer(d, d), cfg)
    assert np.allclose(np.diag(Cb), 1.0)
    assert np.abs(Cb).max() <= 1.0 + 1e-12
    vals = np.linalg.eigvalsh((Cb + Cb.T) / 2)
    assert vals.min() > -1e-10  # convex blend of PSD matrices


def test_blend_weight_one_is_identity():
    from riskprism.model.covariance import pca_blend_corr
    rng = np.random.default_rng(2)
    A = rng.normal(0, 1, (12, 20))
    C = np.corrcoef(A.T)
    out = pca_blend_corr(C, ModelConfig(blend_weight=1.0))
    assert np.allclose(out, C, atol=1e-12)


def test_eigen_profile_shape_and_adjustment():
    from riskprism.model.covariance import (eigen_adjust, eigen_bias_profile,
                                            estimate_weekly_cov)
    cfg = ModelConfig(eigen_adjust_sims=100, eigen_adjust_a=1.0)
    fr = _fr(t=150, seed=4)
    cov = estimate_weekly_cov(fr.to_numpy(), cfg, blend=False)
    v = eigen_bias_profile(cov, 150, cfg, seed=1)
    # the USE4 signature: small eigenfactors substantially underestimated,
    # large ones nearly unbiased
    assert v[:4].mean() > 1.15
    assert v[:4].mean() > v[-4:].mean() + 0.1
    adj = eigen_adjust(cov, v)
    assert np.linalg.eigvalsh((adj + adj.T) / 2).min() > -1e-12
    # adjustment raises total variance (it only inflates eigenvariances)
    assert np.trace(adj) >= np.trace(cov)


def test_factor_covariance_modes_all_psd():
    fr = _fr(t=150, seed=5, ar=0.2)
    for mode in ["none", "blend", "eigen"]:
        cfg = ModelConfig(factor_cov_adjust=mode, eigen_adjust_sims=50)
        F = factor_covariance(fr, cfg).to_numpy()
        assert np.linalg.eigvalsh(F).min() >= 0
        assert np.isfinite(F).all()
