import numpy as np
import pandas as pd
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from test_lookthrough import FakeClient, _fund  # noqa: E402

import riskprism.lookthrough as lookthrough  # noqa: E402
from riskprism.api_server import create_app  # noqa: E402
from riskprism.risk import RiskModel  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
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
    # look-through resolves against an in-memory client: no EDGAR traffic
    monkeypatch.setattr(lookthrough, "_client", FakeClient({
        "DEMOX": _fund("DEMOX", {"MSFT": 0.97}, cash=0.03),
        "BONDX": _fund("BONDX", {"MSFT": 0.05}, other=0.95),
    }))
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


def test_portfolio_risk_lookthrough_expands_funds(client):
    r = client.post(
        "/api/v1/portfolio-risk",
        json={"weights": {"AAPL": 0.5, "DEMOX": 0.5}},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["coverage_ratio"] == 1.0
    assert body["lookthrough"]["funds"]["DEMOX"]["holdings_coverage"] == 1.0


def test_portfolio_risk_lookthrough_disabled(client):
    r = client.post(
        "/api/v1/portfolio-risk",
        json={"weights": {"AAPL": 0.5, "DEMOX": 0.5}, "lookthrough": False},
    )
    body = r.json()
    assert "lookthrough" not in body
    assert body["uncovered_tickers"] == ["DEMOX"]


def test_fund_endpoint(client):
    r = client.get("/api/v1/funds/demox")
    assert r.status_code == 200
    body = r.json()
    assert body["ticker"] == "DEMOX"
    assert body["fund"]["holdings_coverage"] == 1.0
    assert body["total_vol"] > 0


def test_fund_endpoint_refuses_majorly_uncovered(client):
    r = client.get("/api/v1/funds/BONDX")
    assert r.status_code == 422
    assert "covers only" in r.json()["detail"]["message"]


def test_fund_endpoint_unknown_is_404(client):
    assert client.get("/api/v1/funds/ZZZZ").status_code == 404


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

# ---- hosted MCP endpoint (streamable HTTP, stateless) ----

MCP_HEADERS = {"content-type": "application/json",
               "accept": "application/json, text/event-stream"}


def _rpc(method, params=None, id=1):
    msg = {"jsonrpc": "2.0", "method": method, "id": id}
    if params is not None:
        msg["params"] = params
    return msg


def test_mcp_initialize(client):
    r = client.post("/mcp", json=_rpc("initialize", {
        "protocolVersion": "2025-06-18", "capabilities": {},
        "clientInfo": {"name": "pytest", "version": "0"}}), headers=MCP_HEADERS)
    assert r.status_code == 200
    assert r.json()["result"]["serverInfo"]["name"] == "riskprism"


def test_mcp_tools_list_and_call(client):
    r = client.post("/mcp", json=_rpc("tools/list", {}, id=2), headers=MCP_HEADERS)
    assert r.status_code == 200
    tools = {t["name"] for t in r.json()["result"]["tools"]}
    assert {"get_model_info", "get_portfolio_risk", "get_factor_exposures",
            "stress_test", "check_coverage"} <= tools
    r2 = client.post("/mcp", json=_rpc("tools/call", {
        "name": "get_model_info", "arguments": {}}, id=3), headers=MCP_HEADERS)
    assert r2.status_code == 200
    assert "test-0.1" in str(r2.json()["result"])

# ---- horizons & premium scaffold ----


def _tiny_model(version):
    tickers = ["AAPL", "XOM"]
    factors = ["market", "size"]
    X = pd.DataFrame([[1.0, 0.5], [1.0, -0.5]], index=tickers, columns=factors)
    F = pd.DataFrame(np.eye(2) * 0.02, index=factors, columns=factors)
    spec = pd.Series([0.2, 0.25], index=tickers)
    return RiskModel(X, F, spec, meta={"model_version": version})


@pytest.fixture
def client_two_horizons(tmp_path):
    app = create_app(model=_tiny_model("mh-test"), site_dir=tmp_path,
                     model_sh=_tiny_model("sh-test"))
    with TestClient(app) as c:
        yield c


def test_horizon_selects_model(client_two_horizons):
    c = client_two_horizons
    assert c.get("/api/v1/meta").json()["model_version"] == "mh-test"
    assert c.get("/api/v1/meta", params={"horizon": "short"}).json()[
        "model_version"] == "sh-test"
    assert c.get("/api/v1/meta", params={"horizon": "nope"}).status_code == 422


def test_short_horizon_missing_is_503(tmp_path, monkeypatch):
    # block the boot-time download so model_sh stays absent
    from riskprism import registry

    def boom(tag="latest", dest=None, horizon="medium"):
        raise RuntimeError("registry blocked in tests")

    monkeypatch.setattr(registry, "download_artifacts", boom)
    monkeypatch.setenv("RISKPRISM_ARTIFACTS_SH", str(tmp_path / "nope"))
    monkeypatch.setenv("RISKPRISM_ARTIFACTS_SH_URL", "http://127.0.0.1:1/x.tar.gz")
    app = create_app(model=_tiny_model("mh-test"), site_dir=tmp_path)
    with TestClient(app) as c:
        assert c.get("/api/v1/meta", params={"horizon": "short"}).status_code == 503
        assert c.get("/api/v1/health").json()["short_horizon_loaded"] is False


def test_premium_keys_gate_short_horizon_only(tmp_path, monkeypatch):
    monkeypatch.setenv("RISKPRISM_PREMIUM_KEYS", "sekret1, sekret2")
    app = create_app(model=_tiny_model("mh-test"), site_dir=tmp_path,
                     model_sh=_tiny_model("sh-test"))
    with TestClient(app) as c:
        # medium stays free
        assert c.get("/api/v1/meta").status_code == 200
        # short requires a key
        assert c.get("/api/v1/meta", params={"horizon": "short"}).status_code == 402
        r = c.get("/api/v1/meta", params={"horizon": "short"},
                  headers={"Authorization": "Bearer sekret2"})
        assert r.status_code == 200 and r.json()["model_version"] == "sh-test"
        # wrong key rejected
        assert c.get("/api/v1/meta", params={"horizon": "short"},
                     headers={"Authorization": "Bearer wrong"}).status_code == 402


def test_ensure_artifacts_falls_back_to_registry(tmp_path, monkeypatch):
    # the fixed releases/latest URL 404s when the newest release is a
    # package release; the registry must resolve the newest model build
    from riskprism import registry
    from riskprism.api_server import _ensure_artifacts

    calls = {}

    def fake_download(tag="latest", dest=None, horizon="medium"):
        calls["tag"], calls["horizon"] = tag, horizon
        (tmp_path / "meta.json").write_text("{}")

    monkeypatch.setattr(registry, "download_artifacts", fake_download)
    _ensure_artifacts(tmp_path, url="http://127.0.0.1:1/x.tar.gz",
                      horizon="short")
    assert calls == {"tag": "latest", "horizon": "short"}


def test_registry_endpoint(client, monkeypatch):
    from riskprism import registry
    builds = [
        {"tag": "model-2026-08-22b", "title": "Model build 2026-08-22b",
         "published_at": "2026-08-22T03:24:10Z",
         "model_version": "PRISM-US-MH-0.9", "prerelease": False,
         "horizons": ["medium", "short"], "assets": {}},
        {"tag": "model-2026-08-20-demo", "title": "demo",
         "published_at": "2026-08-20T15:47:08Z", "model_version": None,
         "prerelease": True, "horizons": ["medium"], "assets": {}},
    ]
    monkeypatch.setattr(registry, "list_models", lambda refresh=False: builds)
    body = client.get("/api/v1/registry").json()
    assert body["latest"] == "model-2026-08-22b"
    assert len(body["builds"]) == 2
    assert client.get("/api/v1/registry", params={"limit": 1}).json()["builds"] == builds[:1]


def test_registry_endpoint_unavailable(client, monkeypatch):
    from riskprism import registry

    def boom(refresh=False):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(registry, "list_models", boom)
    assert client.get("/api/v1/registry").status_code == 503
