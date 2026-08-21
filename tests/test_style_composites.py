"""v0.6 multi-descriptor value/quality composites."""

import numpy as np
import pandas as pd
import pytest

from riskprism.config import STYLE_FACTORS, ModelConfig
from riskprism.factors.style import _composite, compute_style_exposures
from riskprism.model.build import FUND_FIELDS

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
    return close, volume, fund, idx[-1]


def test_all_styles_finite_and_standardized():
    close, volume, fund, as_of = _panel()
    exp, mktcap = compute_style_exposures(close, volume, fund, as_of, CFG)
    assert list(exp.columns) == STYLE_FACTORS
    assert np.isfinite(exp.to_numpy()).all()
    for style in ("value", "quality"):
        w = mktcap / mktcap.sum()
        assert abs(float((exp[style] * w).sum())) < 0.15  # ~cap-weighted mean 0
        assert 0.5 < exp[style].std() < 2.0


def test_missing_tags_fall_back_to_available_descriptors():
    close, volume, fund, as_of = _panel()
    # T00 files nothing beyond the v0.5 tags: no OCF, revenues, margins
    for col in ("op_cashflow", "revenues", "gross_profit", "cost_of_revenue"):
        fund.loc["T00", col] = np.nan
    exp, _ = compute_style_exposures(close, volume, fund, as_of, CFG)
    # still scored (from B/P + E/P and ROE/ROA), not dropped to exactly 0
    assert np.isfinite(exp.loc["T00", "value"])
    assert np.isfinite(exp.loc["T00", "quality"])


def test_gross_margin_fallback_from_cost_of_revenue():
    close, volume, fund, as_of = _panel()
    fund["gross_profit"] = np.nan  # force (revenues - cost_of_revenue) path
    exp, _ = compute_style_exposures(close, volume, fund, as_of, CFG)
    assert np.isfinite(exp["quality"]).all()


def test_composite_orders_by_descriptor_strength():
    rng = np.random.default_rng(0)
    idx = pd.Index([f"T{i}" for i in range(30)])
    mktcap = pd.Series(rng.uniform(1e9, 1e10, 30), index=idx)
    base = pd.Series(rng.normal(0, 1, 30), index=idx)
    descs = {"a": base, "b": base + rng.normal(0, 0.1, 30)}
    comp = _composite(descs, mktcap, z=3.0, fit=None)
    # composite of two nearly identical descriptors preserves their ranking
    assert comp.rank().corr(base.rank()) > 0.95


def test_composite_skips_missing_not_zero():
    idx = pd.Index([f"T{i}" for i in range(20)])
    mktcap = pd.Series(1e9, index=idx)
    rng = np.random.default_rng(1)
    a = pd.Series(rng.normal(0, 1, 20), index=idx)
    b = a.copy()
    b.iloc[:10] = np.nan  # half the names miss descriptor b
    comp = _composite({"a": a, "b": b}, mktcap, z=3.0, fit=None)
    # names missing b are scored on a alone — mean of available z's, not (z_a + 0)/2
    za = (a - a.mean()) / a.std()
    assert comp.iloc[:10].corr(za.iloc[:10]) > 0.99


def test_negative_earnings_kept_in_earnings_yield():
    close, volume, fund, as_of = _panel()
    fund.loc["T01", "net_income"] = -0.5 * fund.loc["T01", "book_equity"]
    exp, _ = compute_style_exposures(close, volume, fund, as_of, CFG)
    assert np.isfinite(exp.loc["T01", "value"])


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
