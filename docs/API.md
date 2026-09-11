# riskprism JSON API

Base URL: `https://risk-prism-production.up.railway.app` — free, no key, no
signup. The API serves the newest published weekly build; interactive docs
live at [`/api/docs`](https://risk-prism-production.up.railway.app/api/docs)
and the machine-readable spec at `/api/openapi.json`.

Conventions (same as the Python package and MCP server): weights are
portfolio weights — shorts negative, no need to sum to 1; all volatilities
are annualized decimals (`0.20` = 20%/yr). Not investment advice.

## Horizons

Every risk endpoint accepts `?horizon=medium|short` (default `medium`).
The short-horizon variant shares the medium model's daily regressions and
formation exposures but halves every risk-side half-life (vol 42d, corr
126d, specific 42d, VRA 21d) — the commercial S/L model-pair pattern. It
is derived from the same artifacts (`riskprism-variant`) and published as
`riskprism-artifacts-sh.tar.gz` with each release.

If the operator sets `RISKPRISM_PREMIUM_KEYS` (comma-separated tokens),
the short-horizon surface requires `Authorization: Bearer <key>`; unset —
the default — everything is free. The medium-horizon model and all
artifacts are free either way.

## Hosted MCP

The same seven tools as the local `riskprism-mcp` server are served over
streamable HTTP (stateless) at `/mcp`:

```json
{ "mcpServers": { "riskprism": {
    "type": "http",
    "url": "https://risk-prism-production.up.railway.app/mcp"
} } }
```

## Endpoints

### `GET /api/v1/meta`

Model version, as-of date, factor list, coverage counts, full build config,
and validation summary stats (mean R², VRA factors).

### `GET /api/v1/factors`

Factor list, per-factor annualized vols, and the full K×K annualized factor
covariance matrix as nested JSON.

### `GET /api/v1/assets/{ticker}`

Per-asset factor exposures and total/factor/specific vol decomposition, plus
estimation quality (residual history length, blend weight on own history).
`404` if the ticker isn't covered by the current build.

### `GET /api/v1/coverage?tickers=AAPL,MSFT,BRK.B`

Splits a comma-separated ticker list into `covered` / `uncovered`.

### `POST /api/v1/portfolio-risk`

```json
{ "weights": { "AAPL": 0.4, "VTI": 0.4, "XOM": 0.2 }, "optimized": false }
```

Returns total/factor/specific vol, factor exposures, top factor variance
contributions, top asset risk contributions, and coverage info. ETF and
mutual fund tickers expand into their filed N-PORT holdings before the
math runs (`"lookthrough": false` disables this); per-fund holdings
dates and coverage land in `lookthrough.funds`, and a fund the model
cannot estimate keeps its ticker as-is with the reason in
`lookthrough.notes`. Set `optimized: true` if the weights came from
optimizing against this model: reported vols then include the Shepard
second-order correction (optimizers exploit covariance estimation
noise, so raw forecasts understate an optimized portfolio's risk — see
the validation page's TEST 3).

### `GET /api/v1/funds/{ticker}`

Look-through risk report for one ETF or mutual fund: the latest SEC
N-PORT holdings resolve to model tickers (CUSIP map from the SEC
fails-to-deliver files, plus a name match against the EDGAR registry),
and the portfolio math runs on those weights. The `fund` block reports
the holdings date, the coverage ratio, and the renormalization applied.
Cash sleeves count as covered at zero risk. `404` when the ticker
locates no fund filing. `422` when the model covers less than half of
the holdings (bond and international funds) — by policy no estimate is
given for a fund that is majorly outside the model's US equity
universe. Methodology and measured accuracy: DECISIONS.md §17.

### `POST /api/v1/stress-test`

```json
{ "weights": { "AAPL": 1.0 }, "factor_shocks": { "market": -0.10, "momentum": -0.05 } }
```

First-order P&L estimate: exposure × shock per factor. `400` on unknown
factor names (valid names come from `/api/v1/meta`).

### `GET /api/v1/registry?limit=25`

The versioned model registry: every published model build, newest first —
release tag, model version, publish date, available horizons, and artifact
download URLs. Backed by the GitHub releases page (the source of truth),
cached server-side. `riskprism.registry.download_artifacts(tag)` fetches
any historical build from Python; artifact file formats are frozen and
documented in [ARTIFACTS.md](ARTIFACTS.md).

### `GET /api/v1/health`

Liveness + whether artifacts loaded.

## Self-hosting

```bash
pip install "riskprism[api]"
riskprism-api
```

On boot the server loads artifacts from `$RISKPRISM_ARTIFACTS` (default
`./artifacts`); if empty, it downloads the latest release tarball
(`$RISKPRISM_ARTIFACTS_URL` to override) — so a bare container serves the
newest build with zero setup. `$RISKPRISM_SITE` (default `./site`) is served
statically at `/` when present. The look-through endpoints fetch N-PORT
filings from SEC EDGAR, whose fair-access policy requires an identifying
User-Agent: set `RISKPRISM_EDGAR_UA` (for example `"my-project
you@example.com"`), or those endpoints answer `503`. Historical builds stay freely available as
GitHub release assets — that's a published promise, not a temporary state.

## Versioning

The path prefix `/api/v1` is the API contract; the model itself is versioned
independently (`model_version` in every meta payload, `PRISM-US-MH-x.y`).
Breaking response-shape changes would bump the path prefix; new fields may
appear without notice. Artifact files carry their own frozen schema version
(`artifact_schema_version` in `meta.json` — see
[ARTIFACTS.md](ARTIFACTS.md)), and `/api/v1/registry` catalogs every
published build.
