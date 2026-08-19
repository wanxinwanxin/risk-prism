# riskprism

**Decompose US equity portfolio risk into its factor spectrum.**

An open-source, Barra-style fundamental factor risk model built to be
**usable by AI agents out of the box**: a Python library, an MCP server, and
weekly-published model artifacts covering most liquid US common stocks.

- **7 style factors** (size, value, momentum, volatility, liquidity, quality,
  leverage) + **12 industries** (Fama-French scheme) + a market factor
- **Free, redistributable data chain**: fundamentals and SIC codes from SEC
  EDGAR (public domain), prices from pluggable providers
- **Hybrid distribution**: precomputed artifacts (exposures, factor
  covariance, specific risk) are published on a weekly schedule, *and* the
  full pipeline is open so anyone can reproduce or extend them

> **Disclaimer**: research software, provided as-is. Nothing here is
> investment advice.

## For AI agents (MCP)

```json
{
  "mcpServers": {
    "riskprism": {
      "command": "riskprism-mcp",
      "env": { "RISKPRISM_ARTIFACTS": "/path/to/artifacts" }
    }
  }
}
```

Tools exposed: `get_model_info`, `get_portfolio_risk`, `get_factor_exposures`,
`stress_test`, `check_coverage`. Weights are portfolio weights (shorts
negative); volatilities are annualized decimals.

## For humans (Python)

```python
from riskprism import RiskModel

model = RiskModel.load("artifacts")
report = model.portfolio_risk({"AAPL": 0.4, "MSFT": 0.3, "XOM": 0.3})
print(report["total_vol"], report["factor_var_contributions"])

model.stress_test({"AAPL": 1.0}, {"market": -0.10, "momentum": -0.05})
```

## Build the model yourself

```bash
pip install -e ".[dev]"
export RISKPRISM_EDGAR_UA="your-project (you@example.com)"   # SEC fair-access policy
riskprism-build --max-names 1500 --out artifacts             # yahoo prices, no key needed
riskprism-build --provider tiingo ...                        # licensed data, needs TIINGO_API_KEY
```

The weekly GitHub Action runs exactly this and publishes the artifact
directory; see `.github/workflows/build-model.yml`.

## Model summary

| Component | Choice |
|---|---|
| Horizon | Medium (weekly returns, annualized outputs) |
| Estimation | Cross-sectional WLS (√cap weights), cap-weighted industry constraint |
| Factor covariance | EWMA — vol half-life 13w, correlation half-life 26w, PSD-repaired |
| Specific risk | EWMA residual vol, shrunk 30% toward size-bucket mean |
| Universe | EDGAR-registered US common stocks, price ≥ $2, ADV ≥ $1M |

Full methodology in [docs/METHODOLOGY.md](docs/METHODOLOGY.md); design
decisions and their rationale in [docs/DECISIONS.md](docs/DECISIONS.md).

## License

MIT for code. Published model artifacts are derived data built from SEC EDGAR
(public domain) and third-party price providers — see docs/DECISIONS.md for
the data-licensing discussion.
