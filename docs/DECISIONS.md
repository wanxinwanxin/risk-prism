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

## 7. v0.3: the USE4-documented adjustments (decided 2026-08-20)

Literature review (Bloomberg MAC2/MAC3 decks, Barra USE4 Methodology
Notes, Axioma AXUS4 factsheet — links in METHODOLOGY) ranked our gaps by
the impact the vendors measured. Implemented, in that order:

1. **Volatility Regime Adjustment** (factor + specific): EWMA (8-week
   half-life) of the cross-sectional bias statistic, applied as a vol
   multiplier clipped to [0.5, 2]. Chosen first because USE4's published
   evidence is dramatic (rolling bias pinned near 1.0 through 2008–09 vs
   1.3→0.7 unadjusted) and it directly targets our measured
   Mincer–Zarnowitz slope of 0.70.
2. **Optimized portfolios in the validation panel**: min-variance +
   3 random-alpha min-risk portfolios (Woodbury Σ⁻¹, top 500 by cap),
   scored weekly like every other test portfolio. Measure before fixing:
   this is the documented 1.4–1.5-bias failure mode and no public model
   (commercial or open) publishes it continuously.
3. **Newey-West variance adjustment**: Bartlett, 2 lags on factor
   variances, 1 lag on specific — weekly ×52 annualization assumes iid;
   momentum's measured 1.40 realized/forecast daily-vol ratio says
   otherwise. Variance-only keeps V·C·V PSD without extra repair.
4. **Bayesian specific shrinkage**: USE4's q=0.1 distance-dependent
   shrinkage toward size-decile means (equal-weighted buckets on the size
   exposure — deviation from USE4's cap-weighting, chosen so the step is
   reproducible from shipped exposures alone).

Architectural consequence: **validation is now recomputed from history on
every build** (`model/revalidate.py`) rather than accrued across builds —
bias statistics always grade the shipped methodology, exactly (the weekly
regression identity r = Xf + ε makes replayed returns equal true returns
for regressed names, delisting imputations included). v0.2 regression
history carries forward across this version bump
(`compatible_prior_versions`): exposure/regression definitions are
unchanged, and discarding the prior would have re-introduced the
survivorship bias its capture-forward rows exist to prevent.

## 8. v0.4 plan: optimization-bias correction (written 2026-08-20)

The remaining big documented failure mode: optimizers seek out the
covariance matrix's underestimated directions (Shepard 2009: true vol of
an optimized portfolio ≈ predicted/(1−K/T)). Two published fixes; we now
measure the disease continuously (decision 7.2), so the cure is chosen on
our own evidence:

- **Phase 0 — measure (shipping now)**: watch the `opt` group's bias
  statistics for a few builds. With K=20 factors and T≈150 effective
  weeks, Shepard's formula predicts ≈ 1/(1−20/150) ≈ 1.15 before
  estimation noise in specific risk — expect roughly 1.1–1.3, milder than
  USE4's 1.4–1.6 (they have K=60+ industries and optimize harder).
- **Phase A — eigenfactor risk adjustment** (Menchero, Wang & Orr 2011,
  Appendix A): diagonalize F, Monte-Carlo the per-eigenfactor volatility
  bias (simulate T weeks from F ~1,000 times — trivial at K=20), de-bias
  eigenvariances with the paper's scaled variant (a=1.4), rotate back.
  Implement in `model/covariance.py` behind `config.eigen_adjust`;
  the replayed validation state gets the same treatment (adjust the
  weekly factor_cov_weekly output).
- **Phase B — correlation blending** (Bloomberg MAC2/MAC3, Menchero &
  Lazanas 2019): C ← w·C_sample + (1−w)·C_PCA(J), J = ⌈μK⌉ ≈ 5
  components + idiosyncratic diagonal, starting from Bloomberg's
  published (w=0.8, μ=0.25). Same config-flag treatment.
- **Decision rule**: one build per variant; compare (a) `opt`-group bias
  statistics (closer to 1 wins), (b) non-opt portfolios' bias unchanged
  within noise (the adjustment must not distort ordinary portfolios —
  USE4 checked style vols for this), (c) out-of-sample realized vol of
  the min-variance portfolio (lower wins; the De Nard–Ledoit–Wolf
  criterion). Ship the winner; keep the loser behind its flag with the
  comparison documented here.
- **Non-goals**: full Ledoit-Wolf nonlinear shrinkage (K=20 is small; the
  factor-covariance conditioning problem commercial models fight barely
  exists at our K) and daily-returns re-estimation (a separate, larger
  project — would change regression definitions and force a cold
  rebuild).

## 9. v0.4 result: correlation blending ships; eigenfactor adjustment is an honest negative (decided 2026-08-21)

Executed the §8 plan. Both cures implemented in `model/covariance.py`
(`config.factor_cov_adjust`: "blend" / "eigen" / "none"), A/B'd by full
validation replay over the same 123 scored weeks. Bias statistics by
group (1.0 = calibrated; minvar = the global min-variance portfolio
optimized against the model):

| variant | minvar | opt avg | style | market | random | equal | industry |
|---|---|---|---|---|---|---|---|
| none (v0.3) | 1.36 | 1.20 | 1.16 | 1.04 | 0.93 | 0.98 | 0.99 |
| eigen a=1.4 | 0.98 | 1.11 | 0.92 | 0.97 | **0.75** | **0.77** | 0.86 |
| eigen a=1.0 | 1.04 | 1.13 | 0.97 | 0.99 | **0.80** | **0.81** | 0.89 |
| eigen, median/trace/tapered variants | 1.06–1.29 | 1.12–1.19 | 0.99–1.10 | ~1.0 | 0.81–0.87 | 0.83–0.90 | 0.90–0.97 |
| **blend w=0.8, J=5 (shipped)** | **1.29** | **1.18** | 1.16 | 1.04 | 0.94 | 0.99 | 0.99 |

Findings:

- The eigenfactor Monte-Carlo reproduces USE4's headline exactly — our
  smallest eigenfactors are underestimated ~41% (they report ~40%) — and
  the adjustment does fix the optimized portfolios and even the style
  spreads. But at K=20 factors and an effective T≈37 weeks (13-week vol
  half-life), the profile inflates *mid-rank* eigenvalues ~20%, and broad
  long-only portfolios (equal-weight, random baskets) that were
  calibrated become badly over-forecast. Robustified profiles (median
  ratios, trace-preserving, rank-tapered) do not rescue it: the
  mid-rank variance eigen adds overlaps directions real portfolios hold.
  USE4's setting (K=65+, daily returns, effective T≈1000) does not
  transfer to a small-K weekly model. Eigen stays implemented behind the
  flag for anyone who wants it.
- Correlation blending at Bloomberg's *published* parameters (w=0.8,
  J=⌈K/4⌉=5 — deliberately not tuned on our own validation) improves the
  optimized portfolios modestly with zero measurable effect on any other
  group. Shipped as the v0.4 default.
- The residual minvar bias (~1.29) is consistent with Shepard's
  second-order risk arising substantially from *specific*-risk
  estimation noise (the optimizer selects the luckiest low-s² names),
  which no factor-covariance adjustment can reach. Candidate future
  work: stronger specific shrinkage, or reporting Shepard's analytic
  1/(1−K/T) correction alongside optimized-portfolio forecasts. The
  weekly `opt` validation rows keep measuring it either way.
- The out-of-sample realized vol of the min-variance portfolio was
  *lowest under the unadjusted matrix* (4.6% vs 4.8–4.9% adjusted) — at
  our scale the optimization bias manifests as understated forecasts,
  not as materially worse optimized portfolios.

Version bumped to PRISM-US-MH-0.4; 0.2/0.3 regression history carries
forward (`compatible_prior_versions`), validation rescored as always.
