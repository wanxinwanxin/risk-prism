"""MCP server exposing the risk model to AI agents.

Run with `riskprism-mcp`. Artifacts are loaded from $RISKPRISM_ARTIFACTS
(default: ./artifacts). All volatilities are annualized decimals.
"""

import os

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # mcp 2.x removed mcp.server.fastmcp
    raise SystemExit(
        "riskprism-mcp requires the MCP SDK 1.x: pip install 'mcp>=1.2,<2'"
    ) from None

from riskprism.risk import RiskModel

mcp = FastMCP(
    "riskprism",
    instructions=(
        "US equity factor risk model (Barra-style: 8 styles + 12 industries + market). "
        "Weights are portfolio weights; shorts are negative; they need not sum to 1. "
        "All volatilities are annualized decimals (0.20 = 20%/yr). "
        "Not investment advice."
    ),
)

_model: RiskModel | None = None
_model_sh: RiskModel | None = None


def _get_model(horizon: str = "medium") -> RiskModel:
    global _model, _model_sh
    if horizon == "short":
        if _model_sh is None:
            sh_dir = os.environ.get("RISKPRISM_ARTIFACTS_SH")
            if not sh_dir:
                raise ValueError(
                    "short-horizon model not loaded — derive it with "
                    "`riskprism-variant --artifacts artifacts --out artifacts_sh` "
                    "and set RISKPRISM_ARTIFACTS_SH, or use horizon='medium'")
            _model_sh = RiskModel.load(sh_dir)
        return _model_sh
    if _model is None:
        _model = RiskModel.load(os.environ.get("RISKPRISM_ARTIFACTS", "artifacts"))
    return _model


@mcp.tool()
def get_model_info(horizon: str = "medium") -> dict:
    """Model version, as-of date, factor list, and asset coverage count.
    `horizon`: "medium" (default) or "short" — the responsive variant with
    halved risk half-lives, when available."""
    m = _get_model(horizon)
    return {**m.meta, "factors": m.factors, "n_assets": int(len(m.exposures))}


@mcp.tool()
def get_portfolio_risk(weights: dict[str, float], optimized: bool = False,
                       horizon: str = "medium") -> dict:
    """Full risk report for a portfolio: total/factor/specific vol, factor
    exposures, top factor variance contributions, and top asset risk
    contributions. `weights` maps ticker -> portfolio weight. Set
    `optimized=true` if the weights came from optimizing against this
    model: reported vols then include the Shepard second-order correction
    (optimizers exploit covariance estimation noise, so raw forecasts
    understate an optimized portfolio's risk). `horizon`: "medium" or
    "short" (responsive variant, when available)."""
    return _get_model(horizon).portfolio_risk(weights, optimized=optimized)


@mcp.tool()
def get_factor_exposures(tickers: list[str], horizon: str = "medium") -> dict:
    """Per-asset factor exposures and total/factor/specific vol for each ticker."""
    m = _get_model(horizon)
    out, missing = {}, []
    for t in tickers:
        try:
            out[t.upper()] = m.asset_risk(t)
        except KeyError:
            missing.append(t.upper())
    return {"assets": out, "uncovered": missing}


@mcp.tool()
def stress_test(weights: dict[str, float], factor_shocks: dict[str, float],
                horizon: str = "medium") -> dict:
    """Estimate portfolio P&L under factor shocks (return units: -0.10 = -10%).
    Example: {"market": -0.10, "momentum": -0.05}. Use get_model_info for
    valid factor names."""
    return _get_model(horizon).stress_test(weights, factor_shocks)


@mcp.tool()
def check_coverage(tickers: list[str]) -> dict:
    """Which of the given tickers the current model build covers."""
    return _get_model().coverage([t.upper() for t in tickers])


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
