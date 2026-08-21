import numpy as np
import pandas as pd
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from riskprism.api_server import create_app  # noqa: E402
from riskprism.risk import RiskModel  # noqa: E402


@pytest.fixture
def client(tmp_path):
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
    model = RiskModel(X, F, spec, meta={"model_version": "test-0.1"})
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("<title>riskprism</title>")
    app = create_app(model=model, site_dir=site)
    with TestClient(app) as c:
        yield c


def test_meta(client):
    r = client.get("/api/v1/meta")
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == "test-0.1"
    assert body["n_assets"] == 4
    assert "market" in body["factors"]


def test_factors_covariance_symmetric(client):
    body = client.get("/api/v1/factors").json()
    cov = body["covariance"]
    assert cov["market"]["size"] == pytest.approx(cov["size"]["market"])
    assert body["factor_vols"]["market"] == pytest.approx(
        cov["market"]["market"] ** 0.5
    )


def test_asset_found_and_missing(client):
    r = client.get("/api/v1/assets/aapl")
    assert r.status_code == 200
    assert r.json()["ticker"] == "AAPL"
    assert client.get("/api/v1/assets/ZZZZ").status_code == 404


def test_coverage(client):
    body = client.get("/api/v1/coverage", params={"tickers": "AAPL, zzzz"}).json()
    assert body == {"covered": ["AAPL"], "uncovered": ["ZZZZ"]}


def test_portfolio_risk_decomposition(client):
    r = client.post(
        "/api/v1/portfolio-risk",
        json={"weights": {"AAPL": 0.5, "XOM": 0.3, "JPM": 0.2}},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["factor_vol"] ** 2 + body["specific_vol"] ** 2 == pytest.approx(
        body["total_vol"] ** 2
    )
    assert body["coverage_ratio"] == 1.0


def test_portfolio_risk_rejects_empty_weights(client):
    r = client.post("/api/v1/portfolio-risk", json={"weights": {}})
    assert r.status_code == 422


def test_stress_test_bad_factor_is_400(client):
    r = client.post(
        "/api/v1/stress-test",
        json={"weights": {"AAPL": 1.0}, "factor_shocks": {"nope": -0.1}},
    )
    assert r.status_code == 400


def test_stress_test(client):
    r = client.post(
        "/api/v1/stress-test",
        json={"weights": {"AAPL": 1.0}, "factor_shocks": {"market": -0.10}},
    )
    assert r.status_code == 200
    assert r.json()["pnl_estimate"] == pytest.approx(-0.10)  # beta 1.0


def test_site_served_at_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "riskprism" in r.text


def test_health(client):
    assert client.get("/api/v1/health").json()["model_loaded"] is True
