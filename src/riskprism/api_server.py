"""HTTP API for the risk model, plus the rendered explorer site.

Run with `riskprism-api` (or `uvicorn riskprism.api_server:app`). Serves:
    /api/v1/*      JSON risk endpoints (same surface as the MCP server)
    /api/docs      interactive OpenAPI docs
    /              the rendered explorer site ($RISKPRISM_SITE, default ./site)

Artifacts load from $RISKPRISM_ARTIFACTS (default ./artifacts). If meta.json
is absent there, the latest release tarball is downloaded at boot from
$RISKPRISM_ARTIFACTS_URL, so a deployed image always serves the newest
published build without rebuilding. All volatilities are annualized decimals.
"""

import io
import os
import tarfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from riskprism.risk import RiskModel

DEFAULT_ARTIFACTS_URL = (
    "https://github.com/wanxinwanxin/risk-prism/releases/latest/download/"
    "riskprism-artifacts.tar.gz"
)


def _ensure_artifacts(path: Path) -> None:
    """Download and unpack the latest release tarball if `path` is empty."""
    if (path / "meta.json").exists():
        return
    import requests

    url = os.environ.get("RISKPRISM_ARTIFACTS_URL", DEFAULT_ARTIFACTS_URL)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    path.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name  # flatten; refuse traversal
            if not member.isfile() or name.startswith("."):
                continue
            src = tar.extractfile(member)
            if src is not None:
                (path / name).write_bytes(src.read())


class PortfolioRequest(BaseModel):
    weights: dict[str, float] = Field(
        ..., min_length=1,
        description="Ticker -> portfolio weight. Shorts negative; weights "
                    "need not sum to 1.",
        examples=[{"AAPL": 0.4, "MSFT": 0.4, "XOM": 0.2}],
    )
    optimized: bool = Field(
        False,
        description="Set true if the weights came from optimizing against "
                    "this model; reported vols then include the Shepard "
                    "second-order correction.",
    )


class StressRequest(BaseModel):
    weights: dict[str, float] = Field(..., min_length=1)
    factor_shocks: dict[str, float] = Field(
        ..., min_length=1,
        description="Factor -> shock in return units (-0.10 = -10%). "
                    "GET /api/v1/meta for valid factor names.",
        examples=[{"market": -0.10, "momentum": -0.05}],
    )


def create_app(model: RiskModel | None = None,
               site_dir: str | Path | None = None) -> FastAPI:
    state: dict = {"model": model, "error": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["model"] is None:
            artifacts = Path(os.environ.get("RISKPRISM_ARTIFACTS", "artifacts"))
            try:
                _ensure_artifacts(artifacts)
                state["model"] = RiskModel.load(artifacts)
            except Exception as exc:  # keep the site up even if artifacts fail
                state["error"] = f"{type(exc).__name__}: {exc}"
        yield

    app = FastAPI(
        title="riskprism API",
        summary="Open-source US equity factor risk model. Free, no key.",
        description=(
            "Barra-style factor risk model (market + 7 styles + 12 industries), "
            "rebuilt weekly from SEC EDGAR and market data, validated in public. "
            "Weights are portfolio weights; shorts are negative; they need not "
            "sum to 1. All volatilities are annualized decimals (0.20 = 20%/yr). "
            "Not investment advice."
        ),
        version=os.environ.get("RISKPRISM_VERSION", "v1"),
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"],
        allow_methods=["GET", "POST"], allow_headers=["*"],
    )

    def m() -> RiskModel:
        if state["model"] is None:
            raise HTTPException(503, detail=state["error"] or "model not loaded")
        return state["model"]

    @app.get("/api/v1/health", tags=["meta"])
    def health() -> dict:
        return {"status": "ok" if state["model"] is not None else "degraded",
                "model_loaded": state["model"] is not None,
                **({"error": state["error"]} if state["error"] else {})}

    @app.get("/api/v1/meta", tags=["meta"],
             summary="Model version, as-of date, factors, coverage")
    def meta() -> dict:
        model = m()
        return {**model.meta, "factors": model.factors,
                "n_assets": int(len(model.exposures))}

    @app.get("/api/v1/factors", tags=["meta"],
             summary="Factor list, annualized vols, and covariance matrix")
    def factors() -> dict:
        model = m()
        F = model.factor_covariance
        vols = {f: float(F.loc[f, f] ** 0.5) for f in model.factors}
        return {"factors": model.factors, "factor_vols": vols,
                "covariance": {f: {g: float(F.loc[f, g]) for g in model.factors}
                               for f in model.factors}}

    @app.get("/api/v1/assets/{ticker}", tags=["assets"],
             summary="Per-asset exposures and vol decomposition")
    def asset(ticker: str) -> dict:
        try:
            return m().asset_risk(ticker)
        except KeyError:
            raise HTTPException(404, detail=f"{ticker.upper()} not covered "
                                            "by this model build") from None

    @app.get("/api/v1/coverage", tags=["assets"],
             summary="Which tickers the current build covers")
    def coverage(tickers: str = Query(..., description="Comma-separated",
                                      examples=["AAPL,MSFT,BRK.B"])) -> dict:
        return m().coverage([t.strip().upper() for t in tickers.split(",")
                             if t.strip()])

    @app.post("/api/v1/portfolio-risk", tags=["portfolio"],
              summary="Full risk report: vol decomposition + contributions")
    def portfolio_risk(req: PortfolioRequest) -> dict:
        return m().portfolio_risk(req.weights, optimized=req.optimized)

    @app.post("/api/v1/stress-test", tags=["portfolio"],
              summary="Linear P&L estimate under factor shocks")
    def stress_test(req: StressRequest) -> dict:
        try:
            return m().stress_test(req.weights, req.factor_shocks)
        except ValueError as exc:
            raise HTTPException(400, detail=str(exc)) from None

    site = Path(site_dir if site_dir is not None
                else os.environ.get("RISKPRISM_SITE", "site"))
    if site.is_dir():
        app.mount("/", StaticFiles(directory=site, html=True), name="site")
    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))


if __name__ == "__main__":
    main()
