"""v0.8: growth and dividend-yield styles; leverage as a composite."""

import numpy as np
import pandas as pd
import pytest

from riskprism.config import STYLE_FACTORS, ModelConfig
from riskprism.data.edgar import Fundamentals
from riskprism.factors.style import compute_style_exposures
from riskprism.model.build import FUND_FIELDS, _sales_growth

N = 40
CFG = ModelConfig()


def _panel(seed=7, days=300):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=days)
    tickers = [f"T{i:02d}" for i in range(N)]
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.02, (days, N)), axis=0)),
        index=idx, columns=tickers,
    )
    volume = pd.DataFrame(rng.uniform(1e5, 1e6, (days, N)), index=idx, columns=tickers)
    fund = pd.DataFrame(index=pd.Index(tickers), columns=FUND_FIELDS, dtype=float)
    fund["shares_out"] = rng.uniform(1e6, 1e8, N)
    fund["book_equity"] = rng.uniform(1e8, 1e10, N)
    fund["total_assets"] = fund["book_equity"] * rng.uniform(1.5, 4.0, N)
    fund["total_liabilities"] = fund["total_assets"] - fund["book_equity"]
    fund["net_income"] = fund["book_equity"] * rng.uniform(-0.05, 0.25, N)
    fund["op_cashflow"] = fund["net_income"] * rng.uniform(0.8, 1.5, N)
    fund["revenues"] = fund["total_assets"] * rng.uniform(0.3, 1.2, N)
    fund["cost_of_revenue"] = fund["revenues"] * rng.uniform(0.4, 0.9, N)
    fund["gross_profit"] = fund["revenues"] - fund["cost_of_revenue"]
    fund["dividends_paid"] = 0.0
    fund["sales_growth"] = rng.normal(0.05, 0.1, N)
    return close, volume, fund, idx[-1]


def _fundamentals(revenues: list[tuple[str, str, float]]) -> Fundamentals:
    df = pd.DataFrame(revenues, columns=["end", "filed", "val"])
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    return Fundamentals({"revenues": df.sort_values(["filed", "end"])})


def test_new_styles_registered():
    assert "growth" in STYLE_FACTORS
    # dividend yield was measured and rejected (DECISIONS.md §13)
    assert "divyield" not in STYLE_FACTORS
    assert len(STYLE_FACTORS) == 9


def test_all_styles_finite():
    close, volume, fund, as_of = _panel()
    exp, _ = compute_style_exposures(close, volume, fund, as_of, CFG)
    assert list(exp.columns) == STYLE_FACTORS
    assert np.isfinite(exp.to_numpy()).all()


def test_leverage_composite_ranks_levered_names():
    close, volume, fund, as_of = _panel()
    # make T00 maximally levered on every descriptor, T01 minimally
    fund.loc["T00", "total_liabilities"] = fund.loc["T00", "total_assets"] * 0.95
    fund.loc["T00", "book_equity"] = fund.loc["T00", "total_assets"] * 0.05
    fund.loc["T01", "total_liabilities"] = fund.loc["T01", "total_assets"] * 0.05
    fund.loc["T01", "book_equity"] = fund.loc["T01", "total_assets"] * 0.95
    exp, _ = compute_style_exposures(close, volume, fund, as_of, CFG)
    assert exp.loc["T00", "leverage"] > exp.loc["T01", "leverage"]


def test_sales_growth_slope_sign_and_scale():
    grower = _fundamentals([
        ("2021-12-31", "2022-02-15", 100.0),
        ("2022-12-31", "2023-02-15", 120.0),
        ("2023-12-31", "2024-02-15", 140.0),
        ("2024-12-31", "2025-02-15", 160.0),
    ])
    shrinker = _fundamentals([
        ("2021-12-31", "2022-02-15", 160.0),
        ("2022-12-31", "2023-02-15", 140.0),
        ("2023-12-31", "2024-02-15", 120.0),
    ])
    d = pd.Timestamp("2025-06-01")
    g, s = _sales_growth(grower, d), _sales_growth(shrinker, d)
    assert g > 0 > s
    assert abs(g - 20.0 / 130.0) < 1e-9  # slope 20/yr over mean 130


def test_sales_growth_is_point_in_time():
    f = _fundamentals([
        ("2021-12-31", "2022-02-15", 100.0),
        ("2022-12-31", "2023-02-15", 120.0),
        ("2023-12-31", "2024-02-15", 140.0),
        # restatement of FY2023, filed later — invisible before its filed date
        ("2023-12-31", "2025-03-01", 400.0),
    ])
    before = _sales_growth(f, pd.Timestamp("2024-06-01"))
    after = _sales_growth(f, pd.Timestamp("2025-06-01"))
    assert abs(before - 20.0 / 120.0) < 1e-9
    assert after != before  # the restatement changes the visible history


def test_sales_growth_needs_three_periods():
    f = _fundamentals([
        ("2022-12-31", "2023-02-15", 100.0),
        ("2023-12-31", "2024-02-15", 120.0),
    ])
    assert np.isnan(_sales_growth(f, pd.Timestamp("2025-01-01")))


if __name__ == "__main__":
    pytest.main([__file__, "-q"])


def test_estimation_pool_caps_in_edgar_order():
    from riskprism.data.universe import estimation_pool
    from dataclasses import replace
    tickers = [f"T{i:02d}" for i in range(20)]
    cfg = replace(CFG, estimation_max_names=5)
    # EDGAR order (~market cap) is preserved: the FIRST five, not a re-rank
    assert estimation_pool(tickers, cfg) == tickers[:5]
    assert estimation_pool(tickers, replace(CFG, estimation_max_names=0)) == tickers


def test_structural_prior_clipped_to_fit_distribution():
    from riskprism.model.specific import specific_risk
    rng = np.random.default_rng(4)
    n_fit, n_ext, t = 120, 5, 200
    idx = pd.Index([f"F{i:03d}" for i in range(n_fit)] + [f"X{i}" for i in range(n_ext)])
    dates = pd.bdate_range("2024-01-02", periods=t)
    res = pd.DataFrame(rng.normal(0, 0.02, (t, n_fit)),
                       index=dates, columns=idx[:n_fit])
    X = pd.DataFrame(rng.normal(0, 1, (len(idx), 3)),
                     index=idx, columns=["size", "volatility", "liquidity"])
    # the extended names sit far outside the fit set on every feature
    X.iloc[n_fit:] = 8.0
    industries = pd.Series("BusEq", index=idx)
    out = specific_risk(res, X, industries, CFG)
    fit_hi = out.ts_vol.dropna().quantile(0.99) * 1.5
    assert (out.structural.iloc[n_fit:] <= fit_hi + 1e-9).all()
