# riskprism

**Decompose US equity portfolio risk into its factor spectrum.**

**Explorer:** https://risk-prism-production.up.railway.app ·
**Agent model card:** [/model.md](https://risk-prism-production.up.railway.app/model.md)

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
riskprism-build --max-names 3000 --out artifacts             # yahoo prices, no key needed
riskprism-build --prior artifacts_prev --out artifacts       # append new weeks to a prior build
riskprism-build --provider tiingo ...                        # licensed data, needs TIINGO_API_KEY
```

The weekly GitHub Action runs exactly this and publishes the artifact
directory; see `.github/workflows/build-model.yml`.

## The explorer

A zero-backend static site (served on Railway, re-rendered by each weekly
build) for exploring the model: cumulative factor returns, factor vol and
correlations, a client-side portfolio risk sandbox with stress-test
sliders, per-stock factor profiles, and a visual methodology walkthrough.
All math runs in the browser on the embedded artifacts.

Agents get a plain-markdown mirror of every build at **`/model.md`**
(indexed by `/llms.txt`): model card, factor definitions, correlations,
and the full coverage list — no DOM parsing required.

Render everything locally:

```bash
riskprism-site --artifacts artifacts --out site   # index.html + model.md + llms.txt
```

## Model summary

| Component | Choice |
|---|---|
| Horizon | Medium — weekly formation, daily estimation (annualized outputs) |
| Estimation | Daily cross-sectional WLS (√cap weights) against Friday-formed exposures, cap-weighted industry constraint |
| Factor covariance | EWMA on daily factor returns — vol half-life 84d, correlation 252d (~730 effective observations) — with Newey-West variance adjustment, correlation regularization, PSD repair, and a Volatility Regime Adjustment multiplier |
| Specific risk | EWMA residual vol (NW-adjusted) blended with a structural (characteristic-based) prior by history length, Bayesian-shrunk toward size-decile means (q=0.1), with its own VRA multiplier |
| Universe | Estimation: price ≥ $2, ADV ≥ $1M, 26w+ history · Coverage: everything alive ≥ $1, priors fill the gaps |
| History | Capture-forward: weekly builds append to the prior release; delistings imputed, survivorship bias decays out |
| Validation | Recomputed from full history every build: bias statistics, Mincer–Zarnowitz, realized-vol ratios — on market/style/industry/random baskets, six real factor ETFs, and portfolios optimized against the model itself |

Full methodology in [docs/METHODOLOGY.md](docs/METHODOLOGY.md); design
decisions and their rationale in [docs/DECISIONS.md](docs/DECISIONS.md).

## License

MIT for code. Published model artifacts are derived data built from SEC EDGAR
(public domain) and third-party price providers — see docs/DECISIONS.md for
the data-licensing discussion.
