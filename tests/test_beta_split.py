"""v0.7 beta split: Market Sensitivity separated from Residual Volatility."""

import numpy as np
import pandas as pd
import pytest

from riskprism.config import STYLE_FACTORS, ModelConfig
from riskprism.factors.style import compute_style_exposures
from riskprism.model.build import FUND_FIELDS

N = 60
CFG = ModelConfig()


def _panel(seed=11, days=300):
    """Names with known market loadings: r_it = beta_i * f_t + eps_it."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=days)
    tickers = [f"T{i:02d}" for i in range(N)]
    true_beta = np.linspace(0.2, 2.2, N)
    f = rng.normal(0.0004, 0.012, days)
    eps = rng.normal(0, 0.008, (days, N))
    rets = true_beta[None, :] * f[:, None] + eps
    close = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)),
                         index=idx, columns=tickers)
    volume = pd.DataFrame(rng.uniform(1e5, 1e6, (days, N)), index=idx, columns=tickers)
    fund = pd.DataFrame(index=pd.Index(tickers), columns=FUND_FIELDS, dtype=float)
    # near-equal caps so the cap-weighted market ~ f and no single name
    # dominates its own benchmark
    fund["shares_out"] = rng.uniform(9e6, 1.1e7, N)
    fund["book_equity"] = rng.uniform(1e8, 1e10, N)
    fund["total_assets"] = fund["book_equity"] * rng.uniform(1.5, 4.0, N)
    fund["total_liabilities"] = fund["total_assets"] - fund["book_equity"]
    fund["net_income"] = fund["book_equity"] * rng.uniform(-0.05, 0.25, N)
    fund["op_cashflow"] = fund["net_income"] * rng.uniform(0.8, 1.5, N)
    fund["revenues"] = fund["total_assets"] * rng.uniform(0.3, 1.2, N)
    fund["cost_of_revenue"] = fund["revenues"] * rng.uniform(0.4, 0.9, N)
    fund["gross_profit"] = fund["revenues"] - fund["cost_of_revenue"]
    return close, volume, fund, idx[-1], true_beta


def test_beta_in_style_factors():
    assert "beta" in STYLE_FACTORS
    assert STYLE_FACTORS.index("beta") < STYLE_FACTORS.index("volatility")


def test_beta_exposure_recovers_true_loadings():
    close, volume, fund, as_of, true_beta = _panel()
    exp, _ = compute_style_exposures(close, volume, fund, as_of, CFG)
    r = pd.Series(exp["beta"].to_numpy()).rank().corr(
        pd.Series(true_beta).rank())
    assert r > 0.9


def test_residual_vol_is_not_total_vol():
    """Residual volatility ranks names by idiosyncratic noise, not by beta."""
    rng = np.random.default_rng(3)
    days = 300
    idx = pd.bdate_range("2024-01-02", periods=days)
    tickers = [f"T{i:02d}" for i in range(N)]
    # idio vols independent of beta (shuffled), so the beta-orthogonalized
    # exposure must still rank names by their own noise — raw total vol
    # would instead be dominated by the beta spread
    true_beta = np.linspace(2.2, 0.2, N)
    idio = rng.permutation(np.linspace(0.004, 0.02, N))
    f = rng.normal(0.0004, 0.012, days)
    rets = true_beta[None, :] * f[:, None] + rng.normal(0, 1, (days, N)) * idio[None, :]
    close = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=tickers)
    volume = pd.DataFrame(rng.uniform(1e5, 1e6, (days, N)), index=idx, columns=tickers)
    fund = pd.DataFrame(index=pd.Index(tickers), columns=FUND_FIELDS, dtype=float)
    fund["shares_out"] = rng.uniform(9e6, 1.1e7, N)
    fund["book_equity"] = rng.uniform(1e8, 1e10, N)
    fund["total_assets"] = fund["book_equity"] * 2.0
    fund["total_liabilities"] = fund["total_assets"] - fund["book_equity"]
    fund["net_income"] = fund["book_equity"] * 0.1
    fund["op_cashflow"] = fund["net_income"]
    fund["revenues"] = fund["total_assets"] * 0.5
    fund["cost_of_revenue"] = fund["revenues"] * 0.6
    fund["gross_profit"] = fund["revenues"] - fund["cost_of_revenue"]
    exp, _ = compute_style_exposures(close, volume, fund, idx[-1], CFG)
    r = pd.Series(exp["volatility"].to_numpy()).rank().corr(
        pd.Series(idio).rank())
    assert r > 0.8


def test_volatility_orthogonal_to_beta():
    close, volume, fund, as_of, _ = _panel()
    exp, _ = compute_style_exposures(close, volume, fund, as_of, CFG)
    assert abs(float(exp["beta"].corr(exp["volatility"]))) < 0.05


def test_short_history_name_falls_back_to_market_average():
    close, volume, fund, as_of, _ = _panel()
    cutoff = len(close) - CFG.volatility_window_days // 2 + 20
    close.iloc[:cutoff, 0] = np.nan  # T00 has too few return observations
    exp, _ = compute_style_exposures(close, volume, fund, as_of, CFG)
    # raw beta/vol are NaN for T00; process_exposure fills to 0, and the
    # orthogonalized vol re-standardizes, so both must simply be finite
    assert np.isfinite(exp.loc["T00", "beta"])
    assert np.isfinite(exp.loc["T00", "volatility"])
    assert exp.loc["T00", "beta"] == 0.0


def test_fit_universe_defines_the_market():
    """A name outside the fit universe cannot move the market it is measured against."""
    close, volume, fund, as_of, true_beta = _panel()
    fit = pd.Index([t for t in close.columns if t != "T59"])
    # blow up T59: without fit-scoping it would distort the market series
    close["T59"] = close["T59"] * np.exp(np.linspace(0, 3, len(close)))
    exp_fit, _ = compute_style_exposures(close, volume, fund, as_of, CFG, fit=fit)
    r = pd.Series(exp_fit["beta"].reindex(fit).to_numpy()).rank().corr(
        pd.Series(true_beta[:-1]).rank())
    assert r > 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
