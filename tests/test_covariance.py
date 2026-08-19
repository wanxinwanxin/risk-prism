import numpy as np
import pandas as pd

from riskprism.config import ModelConfig
from riskprism.model.covariance import factor_covariance
from riskprism.model.specific import specific_risk


def test_covariance_is_psd_and_recovers_scale():
    rng = np.random.default_rng(0)
    cfg = ModelConfig()
    weekly_vol = 0.02
    f = pd.DataFrame(rng.normal(0, weekly_vol, (300, 5)), columns=list("ABCDE"))
    cov = factor_covariance(f, cfg)

    vals = np.linalg.eigvalsh(cov.to_numpy())
    assert vals.min() >= 0

    ann_vol = np.sqrt(np.diag(cov))
    expected = weekly_vol * np.sqrt(cfg.ann_factor)
    assert np.all(np.abs(ann_vol / expected - 1) < 0.35)


def test_correlated_factors_show_up():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 0.02, 400)
    f = pd.DataFrame({"a": base, "b": base + rng.normal(0, 0.005, 400)})
    cov = factor_covariance(f, ModelConfig())
    corr = cov.loc["a", "b"] / np.sqrt(cov.loc["a", "a"] * cov.loc["b", "b"])
    assert corr > 0.9


def test_specific_risk_shrinks_and_fills():
    rng = np.random.default_rng(2)
    cfg = ModelConfig()
    resid = pd.DataFrame(rng.normal(0, 0.03, (100, 20)),
                         columns=[f"T{i}" for i in range(20)])
    resid.iloc[:-5, 0] = np.nan  # T0: only 5 obs, below min -> bucket fallback
    caps = pd.Series(rng.lognormal(10, 1, 20), index=resid.columns)
    spec = specific_risk(resid, caps, cfg)
    assert spec.notna().all()
    assert (spec > 0).all()
    # roughly annualized 3% weekly vol
    typical = 0.03 * np.sqrt(cfg.ann_factor)
    assert np.all((spec / typical > 0.4) & (spec / typical < 2.5))
