import numpy as np
import pandas as pd
import pytest

from riskprism.artifacts import load_artifacts, save_artifacts
from riskprism.risk import RiskModel


@pytest.fixture
def model(tmp_path):
    rng = np.random.default_rng(0)
    tickers = ["AAPL", "MSFT", "XOM", "JPM"]
    factors = ["market", "size", "value", "ind_BusEq", "ind_Enrgy", "ind_Money"]
    X = pd.DataFrame(
        [
            [1.0, 1.5, -0.5, 1, 0, 0],
            [1.0, 1.4, -0.3, 1, 0, 0],
            [1.0, 0.8, 0.9, 0, 1, 0],
            [1.0, 0.9, 0.6, 0, 0, 1],
        ],
        index=tickers,
        columns=factors,
    )
    A = rng.normal(0, 0.05, (6, 6))
    F = pd.DataFrame(A @ A.T + np.eye(6) * 1e-4, index=factors, columns=factors)
    spec = pd.Series([0.20, 0.18, 0.25, 0.22], index=tickers)
    freturns = pd.DataFrame(rng.normal(0, 0.02, (52, 6)), columns=factors)
    meta = {"model_version": "test-0.1"}
    save_artifacts(tmp_path, X, F, spec, freturns, meta)
    return RiskModel.load(tmp_path)


def test_artifact_roundtrip(model, tmp_path):
    a = load_artifacts(tmp_path)
    assert a["meta"]["model_version"] == "test-0.1"
    assert list(a["exposures"].index) == ["AAPL", "MSFT", "XOM", "JPM"]


def test_decomposition_sums_to_total(model):
    r = model.portfolio_risk({"AAPL": 0.5, "XOM": 0.3, "JPM": 0.2})
    assert r["total_vol"] > 0
    assert r["factor_vol"] ** 2 + r["specific_vol"] ** 2 == pytest.approx(r["total_vol"] ** 2)
    # factor variance contributions sum to factor variance
    assert sum(r["factor_var_contributions"].values()) == pytest.approx(r["factor_vol"] ** 2)


def test_asset_contributions_sum_to_total_vol(model):
    weights = {"AAPL": 0.4, "MSFT": 0.3, "XOM": 0.3}
    r = model.portfolio_risk(weights)
    assert sum(r["top_asset_risk_contributions"].values()) == pytest.approx(r["total_vol"])


def test_uncovered_tickers_reported(model):
    r = model.portfolio_risk({"AAPL": 0.5, "ZZZZ": 0.5})
    assert r["uncovered_tickers"] == ["ZZZZ"]
    assert r["coverage_ratio"] == pytest.approx(0.5)


def test_stress_test_is_linear(model):
    w = {"AAPL": 0.6, "JPM": 0.4}
    one = model.stress_test(w, {"market": -0.10})
    two = model.stress_test(w, {"market": -0.20})
    assert two["pnl_estimate"] == pytest.approx(2 * one["pnl_estimate"])
    x = model.factor_exposures(w)
    assert one["pnl_estimate"] == pytest.approx(x["market"] * -0.10)


def test_stress_test_rejects_unknown_factor(model):
    with pytest.raises(ValueError):
        model.stress_test({"AAPL": 1.0}, {"not_a_factor": -0.1})


def test_short_and_leveraged_weights(model):
    r = model.portfolio_risk({"AAPL": 1.0, "XOM": -1.0})
    assert r["total_vol"] > 0
