"""MCP server exposing the risk model to AI agents.

Run with `osrisk-mcp`. Artifacts are loaded from $OSRISK_ARTIFACTS
(default: ./artifacts). All volatilities are annualized decimals.
"""

import os

from mcp.server.fastmcp import FastMCP

from osrisk.risk import RiskModel

mcp = FastMCP(
    "osrisk",
    instructions=(
        "US equity factor risk model (Barra-style: 7 styles + 12 industries + market). "
        "Weights are portfolio weights; shorts are negative; they need not sum to 1. "
        "All volatilities are annualized decimals (0.20 = 20%/yr). "
        "Not investment advice."
    ),
)

_model: RiskModel | None = None


def _get_model() -> RiskModel:
    global _model
    if _model is None:
        _model = RiskModel.load(os.environ.get("OSRISK_ARTIFACTS", "artifacts"))
    return _model


@mcp.tool()
def get_model_info() -> dict:
    """Model version, as-of date, factor list, and asset coverage count."""
    m = _get_model()
    return {**m.meta, "factors": m.factors, "n_assets": int(len(m.exposures))}


@mcp.tool()
def get_portfolio_risk(weights: dict[str, float]) -> dict:
    """Full risk report for a portfolio: total/factor/specific vol, factor
    exposures, top factor variance contributions, and top asset risk
    contributions. `weights` maps ticker -> portfolio weight."""
    return _get_model().portfolio_risk(weights)


@mcp.tool()
def get_factor_exposures(tickers: list[str]) -> dict:
    """Per-asset factor exposures and total/factor/specific vol for each ticker."""
    m = _get_model()
    out, missing = {}, []
    for t in tickers:
        try:
            out[t.upper()] = m.asset_risk(t)
        except KeyError:
            missing.append(t.upper())
    return {"assets": out, "uncovered": missing}


@mcp.tool()
def stress_test(weights: dict[str, float], factor_shocks: dict[str, float]) -> dict:
    """Estimate portfolio P&L under factor shocks (return units: -0.10 = -10%).
    Example: {"market": -0.10, "momentum": -0.05}. Use get_model_info for
    valid factor names."""
    return _get_model().stress_test(weights, factor_shocks)


@mcp.tool()
def check_coverage(tickers: list[str]) -> dict:
    """Which of the given tickers the current model build covers."""
    return _get_model().coverage([t.upper() for t in tickers])


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
