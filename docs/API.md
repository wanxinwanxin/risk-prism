# riskprism JSON API

Base URL: `https://risk-prism-production.up.railway.app` — free, no key, no
signup. The API serves the newest published weekly build; interactive docs
live at [`/api/docs`](https://risk-prism-production.up.railway.app/api/docs)
and the machine-readable spec at `/api/openapi.json`.

Conventions (same as the Python package and MCP server): weights are
portfolio weights — shorts negative, no need to sum to 1; all volatilities
are annualized decimals (`0.20` = 20%/yr). Not investment advice.

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
{ "weights": { "AAPL": 0.4, "MSFT": 0.4, "XOM": 0.2 }, "optimized": false }
```

Returns total/factor/specific vol, factor exposures, top factor variance
contributions, top asset risk contributions, and coverage info. Set
`optimized: true` if the weights came from optimizing against this model:
reported vols then include the Shepard second-order correction (optimizers
exploit covariance estimation noise, so raw forecasts understate an
optimized portfolio's risk — see the validation page's TEST 3).

### `POST /api/v1/stress-test`

```json
{ "weights": { "AAPL": 1.0 }, "factor_shocks": { "market": -0.10, "momentum": -0.05 } }
```

First-order P&L estimate: exposure × shock per factor. `400` on unknown
factor names (valid names come from `/api/v1/meta`).

### `GET /api/v1/health`

Liveness + whether artifacts loaded.

## Self-hosting

```bash
pip install "riskprism[api] @ git+https://github.com/wanxinwanxin/risk-prism"
riskprism-api
```

On boot the server loads artifacts from `$RISKPRISM_ARTIFACTS` (default
`./artifacts`); if empty, it downloads the latest release tarball
(`$RISKPRISM_ARTIFACTS_URL` to override) — so a bare container serves the
newest build with zero setup. `$RISKPRISM_SITE` (default `./site`) is served
statically at `/` when present. Historical builds stay freely available as
GitHub release assets — that's a published promise, not a temporary state.

## Versioning

The path prefix `/api/v1` is the API contract; the model itself is versioned
independently (`model_version` in every meta payload, `PRISM-US-MH-x.y`).
Breaking response-shape changes would bump the path prefix; new fields may
appear without notice.
