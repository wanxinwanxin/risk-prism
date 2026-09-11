# Roadmap

The plan is organized around one question: **what separates this model from
the commercial systems (Barra, Axioma, Bloomberg), and which of those gaps
can be closed with public, redistributable data?** Gaps that can't be closed
that way are listed as explicit non-goals rather than quietly ignored.

Status as of `PRISM-US-MH-0.9` (2026-08-22).

## Where we stand

| Dimension | riskprism today | Commercial typical | Gap |
|---|---|---|---|
| Estimation universe | ~2,800 liquid names | Axioma AXUS4 ~2,900 | at parity |
| Coverage universe | ~6,300 of ~8,000 EDGAR candidates | 8,000–9,000 US names | near parity — the rest is mostly untradeable |
| Factors | 40 (market + 9 styles + FF30) | 70–80 (a dozen styles + 60+ GICS industries) | near parity (public schemes go to FF48) |
| Estimation frequency | daily cross-sections | daily | at parity (since v0.5) |
| Live track record | ~3 years replayed, weeks live | 25–30 years | only time closes this |
| Horizons | short + medium (`?horizon=` on API/MCP) | short / medium / long variants | long variant closable — same engine, longer half-lives |
| Descriptor data | prices + SEC EDGAR | + analyst estimates, GICS, specialist feeds | partially a non-goal (see below) |
| Validation | public, reproducible, re-scored weekly | whitepaper snapshots | our advantage — keep extending it |

## Next — versioned model work

- **v0.7 — beta split (SHIPPED 2026-08-22).** Market Sensitivity (beta)
  separated from beta-orthogonalized Residual Volatility. Beta measured
  significant in 84% of daily cross-sections (second only to the market),
  mean R² 0.159 → 0.178, every scoreboard aggregate improved. Evidence:
  DECISIONS.md §12.
- **v0.8 — styles (SHIPPED 2026-08-22, inside the v0.9 build).** Growth
  ships (41.8% of cross-sections significant); leverage rebuilt as a
  3-descriptor composite (bias 1.50 → 1.11); dividend yield measured and
  rejected (0% significant — an honest negative). Evidence: DECISIONS §13.
- **v0.9 — FF30 industries (SHIPPED 2026-08-22).** K = 40 (market + 9
  styles + FF30 from Ken French's public SIC maps). Mean daily R² 0.212,
  overall bias 0.99. Evidence: DECISIONS §14. Coverage shipped the same
  day: 2,987 → 6,307 names with estimation pinned at the EDGAR-ordered
  top 3,000 (two failed attempts documented — §15). The eigenfactor A/B
  re-ran at K=40: blend stands (§14).
- **v1.0 — stability.** The infrastructure half shipped 2026-08-23
  (DECISIONS §16): artifact schema frozen at version 1 with a documented
  contract (ARTIFACTS.md), the package readied for PyPI with automated
  trusted publishing on `v*` tags (RELEASING.md), and a versioned model
  registry over the release history (`/api/v1/registry`, the
  `list_model_versions` MCP tool, `riskprism.registry.download_artifacts`).
  What remains is the part only time delivers: at least one year of
  uninterrupted live weekly out-of-sample record before v1.0 is declared.

## Later

- **ETF and fund look-through (SHIPPED 2026-09-11).** Fund tickers
  resolve to their latest N-PORT holdings (SEC, public domain) and the
  portfolio math runs on the covered constituents; below 50% model
  coverage no estimate is given. `GET /api/v1/funds/{ticker}`, the
  `get_etf_risk` MCP tool, and look-through by default in
  `portfolio-risk`. Evidence: DECISIONS.md §17.
- **Short-horizon variant (SHIPPED 2026-08-22).** Same daily engine,
  faster half-lives — derived from each weekly build via
  `riskprism-variant` and served at `?horizon=short` on the API and MCP.
- **Hosted API (SHIPPED 2026-08-21).** JSON API over the newest weekly
  build at `/api/v1`, plus a hosted MCP endpoint at `/mcp` (2026-08-22);
  historical builds stay free to download regardless.
- **Second validation family** — Fama-French portfolio panels alongside the
  ETF and optimized-portfolio panels.
- **Longer archive** — extend the price history capture so the replayed
  record grows beyond the provider lookback window.
- **Liquidity & crowding metrics** — days-to-liquidate from volume data;
  factor-crowding indicators from the model's own exposures.

## Non-goals

- **Analyst-estimate descriptors** (forward E/P, revisions): IBES-class
  data is proprietary. This is the one systematic sacrifice vs commercial
  value/growth factors, and we take it knowingly — a redistributable data
  chain is the point of the project.
- **GICS industries**: licensed. Fama-French schemes are public domain and
  auditable; we go deeper into FF granularity instead.
- **ESG factors**: no public, redistributable, point-in-time ESG data
  exists that meets the bar above.
- **Production SLA**: this is research software with a weekly public build,
  not a guaranteed risk system.
