import numpy as np
import pandas as pd
import pytest

from riskprism.config import STYLE_FACTORS
from riskprism.factors.industry import INDUSTRY_PREFIX
from riskprism.model.regression import cross_sectional_regression


def _synthetic_cross_section(seed=0, n=600, noise=1e-4):
    rng = np.random.default_rng(seed)
    idx = [f"T{i:04d}" for i in range(n)]
    styles = pd.DataFrame(
        rng.normal(0, 1, (n, len(STYLE_FACTORS))), index=idx, columns=STYLE_FACTORS
    )
    inds = pd.Series(rng.choice(["BusEq", "Hlth", "Money", "Enrgy"], n), index=idx)
    caps = pd.Series(rng.lognormal(10, 1.5, n), index=idx)

    true_style = pd.Series(rng.normal(0, 0.01, len(STYLE_FACTORS)), index=STYLE_FACTORS)
    raw_ind = pd.Series(rng.normal(0, 0.01, 4), index=["BusEq", "Hlth", "Money", "Enrgy"])
    # Impose the identification constraint on the true industry returns
    ind_caps = caps.groupby(inds).sum()
    w = ind_caps / ind_caps.sum()
    raw_ind -= (raw_ind * w).sum() / w.sum() * 0 + (raw_ind * w).sum()  # cap-weighted demean
    true_mkt = 0.005

    y = (
        true_mkt
        + styles @ true_style
        + inds.map(raw_ind)
        + rng.normal(0, noise, n)
    )
    return y, styles, inds, caps, true_mkt, true_style, raw_ind


def test_recovers_true_factor_returns():
    y, styles, inds, caps, true_mkt, true_style, true_ind = _synthetic_cross_section()
    res = cross_sectional_regression(y, styles, inds, caps)
    assert abs(res.factor_returns["market"] - true_mkt) < 1e-3
    for f in STYLE_FACTORS:
        assert abs(res.factor_returns[f] - true_style[f]) < 1e-3
    for ind, val in true_ind.items():
        assert abs(res.factor_returns[f"{INDUSTRY_PREFIX}{ind}"] - val) < 1e-3
    assert res.r2 > 0.95


def test_industry_constraint_holds():
    y, styles, inds, caps, *_ = _synthetic_cross_section(seed=1)
    res = cross_sectional_regression(y, styles, inds, caps)
    ind_returns = res.factor_returns[res.factor_returns.index.str.startswith(INDUSTRY_PREFIX)]
    ind_caps = caps.groupby(inds).sum()
    w = (ind_caps / ind_caps.sum()).rename(lambda x: f"{INDUSTRY_PREFIX}{x}")
    assert abs((ind_returns * w.reindex(ind_returns.index)).sum()) < 1e-12


def test_infinite_returns_are_excluded_not_fatal():
    y, styles, inds, caps, true_mkt, *_ = _synthetic_cross_section(seed=2)
    y.iloc[0], y.iloc[1] = np.inf, -np.inf  # junk price data upstream
    res = cross_sectional_regression(y, styles, inds, caps)
    assert res.n_assets == len(y) - 2
    assert abs(res.factor_returns["market"] - true_mkt) < 1e-3


def test_raises_on_thin_cross_section():
    y, styles, inds, caps, *_ = _synthetic_cross_section(n=600)
    small = y.index[:10]
    with pytest.raises(ValueError):
        cross_sectional_regression(y[small], styles.loc[small], inds[small], caps[small])
