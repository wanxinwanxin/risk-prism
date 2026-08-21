# Roadmap

The plan is organized around one question: **what separates this model from
the commercial systems (Barra, Axioma, Bloomberg), and which of those gaps
can be closed with public, redistributable data?** Gaps that can't be closed
that way are listed as explicit non-goals rather than quietly ignored.

Status as of `PRISM-US-MH-0.6` (2026-08-21).

## Where we stand

| Dimension | riskprism today | Commercial typical | Gap |
|---|---|---|---|
| Estimation universe | ~2,800 liquid names | Axioma AXUS4 ~2,900 | at parity |
| Coverage universe | ~3,000 (self-imposed cap) | 8,000–9,000 US names | closable — EDGAR has ~8,000 candidates |
| Factors | 20 (market + 7 styles + FF12) | 70–80 (a dozen styles + 60+ GICS industries) | partially closable (public schemes go to FF48) |
| Estimation frequency | daily cross-sections | daily | at parity (since v0.5) |
| Live track record | ~3 years replayed, weeks live | 25–30 years | only time closes this |
| Horizons | one (medium, weekly) | short / medium / long variants | closable — same engine, different half-lives |
| Descriptor data | prices + SEC EDGAR | + analyst estimates, GICS, specialist feeds | partially a non-goal (see below) |
| Validation | public, reproducible, re-scored weekly | whitepaper snapshots | our advantage — keep extending it |

## Next — versioned model work

- **v0.7 — beta split.** Separate Market Sensitivity (beta) from Residual
  Volatility, orthogonalized — the one structural style every commercial
  model has that we lack. Rework or demote **leverage**, now the weakest
  calibrated style (significant in 34% of cross-sections, style-portfolio
  bias 1.47).
- **v0.8 — growth & dividend yield.** The v0.6 fundamentals ingestion
  (revenues, OCF, dividends-adjacent tags) already carries what these need.
  Candidate midcap (size²) factor. Each addition gated on the same QC
  battery: %-significant, VIF, exposure stability, style-portfolio bias.
- **v0.9 — industries & coverage.** FF12 → FF30 industries (public Ken
  French SIC maps; K→~45), and raise the coverage universe toward the
  ~8,000 EDGAR candidates. Estimation universe stays liquidity-screened at
  ~2,800. Re-run the eigenfactor A/B at higher K (documented negative at
  K=20; the trade may flip).
- **v1.0 — stability.** Frozen artifact schema, PyPI package, versioned
  model registry, and at least one year of uninterrupted live weekly
  out-of-sample record.

## Later

- **Short-horizon variant** — same daily engine, faster half-lives, for
  users forecasting days rather than weeks.
- **Second validation family** — Fama-French portfolio panels alongside the
  ETF and optimized-portfolio panels.
- **Longer archive** — extend the price history capture so the replayed
  record grows beyond the provider lookback window.
- **Hosted API** — portfolio risk over HTTPS without downloading artifacts;
  historical builds stay free to download regardless.
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
