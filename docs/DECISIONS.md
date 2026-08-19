# Design decisions

Decisions made 2026-08-19 at project start. Revisit deliberately; each bump
of `ModelConfig.version` should reference the decision that changed.

## 1. Flagship methodology: fundamental factor model

Barra-style: style + industry factors, cross-sectional regressions, factor
covariance + specific risk. Chosen over statistical (PCA) because
interpretability is the product — agents need to *explain* risk, not just
quantify it. SEC EDGAR's XBRL API makes the fundamentals side free and
public domain, which is what makes an open-source version viable at all.
A statistical model may be added later as a validation baseline.

## 2. Data strategy: hybrid

We publish precomputed model artifacts (exposures, factor covariance,
specific risk — derived data, safe to redistribute) *and* ship the full
pipeline so anyone can rebuild with their own price source. Raw prices are
never redistributed. Fundamentals/SIC: EDGAR (public domain). Prices:
pluggable providers — Yahoo chart API is the keyless default (unofficial
endpoint, redistribution-restricted, hence artifacts-only distribution);
Tiingo with a key; stooq kept but currently behind a JS challenge. Users
must check their provider's terms for their own rebuilds.

## 3. Agent interface: Python package + MCP server

`pip install riskprism` for the math; `riskprism-mcp` for agents. A hosted REST
API was deliberately deferred (hosting cost/ops without proven demand).

## 4. Frontend: static explorer, zero backend (added 2026-08-19)

A single self-contained HTML page (`site/template.html`) with the model
data embedded as JSON by `riskprism-site`; all risk math re-implemented
client-side (~60 lines of JS). Chosen over a hosted app because the entire
model fits in ~100KB, which makes a server pure liability. Views: factor
returns/vol/correlation, portfolio sandbox with stress sliders, per-stock
profiles, methodology walkthrough, agent onboarding.

Hosting (2026-08-19): Railway, serving the `site/` directory via a Caddy
Dockerfile — the user's platform choice; the site is plain static files so
it deploys anywhere. The weekly Action re-renders and commits
`site/index.html` + `site/model.md` + `site/llms.txt`, then redeploys via
`railway up` when the RAILWAY_TOKEN secret is set.

Agent endpoint: every build also renders `/model.md` (llms.txt-style
markdown mirror — model card, factor definitions, correlations, coverage)
because a JS-rendered page is hostile to text-only agents.

## 5. Coverage via priors + capture-forward history (v0.2, 2026-08-19)

Split the estimation universe (liquid names that estimate factor returns)
from the coverage universe (everything alive at build date). Coverage
names get risk through the factor structure plus a structural specific-risk
model (ln vol regressed on size/volatility/liquidity/industry), blended
with their own EWMA by w = T/(T+26w) — so IPOs and illiquid names are
covered with explicit, inspectable priors (`asset_meta.parquet`).

Survivorship: rejected buying delisted-price data (CRSP/Sharadar/Norgate —
paid, redistribution-restricted; free sources only provide delisting
*lists*, not prices). Instead: capture-forward. Weekly builds append to the
prior release's factor-return/residual history; disappearing names get a
Shumway-style imputed delisting return (−30% under $5, else 0) and keep
their rows. With 13/26-week half-lives the biased cold start decays out
within ~18–24 months — the bias is documented and self-liquidating.

## 6. Refresh: weekly GitHub Actions

Cron builds the model weekly and publishes the artifact directory. Weekly
matches the medium-horizon design; daily refresh only matters for
short-horizon/trading use, which is out of scope for v1.

## Defaults adopted without much debate (revisit as needed)

- **Horizon**: medium (weekly returns). Short-horizon daily model = v2 idea.
- **Industries**: Fama-French 12 from SIC codes — public, stable, avoids GICS
  licensing. Coarser than commercial models; acceptable for v1.
- **Point-in-time**: fundamentals use EDGAR `filed` dates (no lookahead), but
  the price panel is a *current* snapshot — delisted names are missing, so
  historical factor returns have survivorship bias. Documented loudly in
  METHODOLOGY; fixing this properly (delisted-price archive) is the single
  biggest v2 data improvement.
- **Universe heuristics**: 1-5 letter tickers, one per CIK (shortest = primary
  class), price/ADV floors. Crude but transparent.
- **Stack**: Python 3.11+, numpy/pandas, parquet. pandas over polars for v1 —
  ubiquity beats speed at this data size; revisit if the pipeline slows.
- **License**: MIT (code). Artifacts: derived data, distributed with
  attribution of sources.
- **Naming** (decided 2026-08-19): `riskprism` — a prism decomposes light
  into its spectrum; the model decomposes portfolio risk into a factor
  spectrum. Free on PyPI, keeps "risk" searchable, and gives the frontend
  its visual identity. Runners-up: `beaufort`, `loadings`. Model version
  string: `PRISM-US-MH-x.y` (US, Medium Horizon).
