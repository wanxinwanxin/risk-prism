# Backtesting riskprism against FactorsToday's risk model

*2026-08-24 · [riskprism](https://risk-prism-production.up.railway.app) · draft — analysis backing a comparison thread*

[FactorsToday](https://www.factorstoday.com) is a factor analytics platform
by Lukasz Tomicki (LRT Capital) with a free, keyless API that exposes risk
model primitives: factor covariance, per-stock loadings, and specific vol.
Its model family is different from ours — time-series/returns-based
(factors are purified ETF-spread portfolios; stock exposures are regression
betas) versus our cross-sectional fundamental model (characteristic
exposures; factor returns from daily WLS). This note runs both over the
same portfolios, weeks, and realized returns, point-in-time.

## Replicating their model

Their documentation and live API pin the methodology down:

- **Loadings**: rolling OLS/LASSO of ~756 trailing daily returns on their
  factor returns (`window_days: 756`, `l1_ratio: 1.0`, alpha ≈ 4e-6 ≈ OLS).
- **Factor covariance**: sample covariance of trailing 252 daily factor
  returns, annualized ×252.
- **Specific vol**: std of the last 252 daily in-sample residuals.
- Portfolio variance: `w'BΣB'w + w'Δw` (their docs state the formula; no
  endpoint assembles it).

We rebuilt exactly that from their bulk factor-return download (16 core
factors, "Base" tier), and validated the replica against their live API:
covariance matrix off-diagonal correlation **0.985** vs their
`/api/factor-covariance`; AAPL in-sample R² **0.474** vs their published
0.450; AAPL specific vol **22.4%** vs their 21.3%. Close enough that the
replica's portfolio vols are theirs.

riskprism's side is `model_asof()` — the exact point-in-time model the
shipped `validation.parquet` was scored with (EWMA cov + Newey-West +
correlation blending + VRA + shrunk specific).

Timing was matched to our validation convention: forecasts use data only
through formation Friday t, scored on the realized return over (t, t+1wk].
(First replica draft accidentally gave FT the realized week — a full week
of look-ahead — worth flagging because it moved bias stats by ~0.02–0.2.)

**Caveats that favor the replica**: their downloadable factor history is
today's reconstruction (weekly full recalibration, betas back-filled to
history start — documented look-ahead), so the replica sees cleaner factor
returns than their live model published at the time. And our replica uses
their Base tier; their All-Factors tier fits tighter in-sample but with 45
active regressors on 756 obs, likely gives back some of that out of sample.

## Results (2024-01 → 2026-08)

Bias statistic = std(realized weekly return / forecast weekly vol); 1.0 is
perfect, >1 under-forecasts risk. Tail = share of |z| > 1.96 (target 5%).

**Diversified ETF panel** (SPY, IWM, MTUM, VLUE, QUAL, USMV × 135 weeks,
same realized returns as our shipped validation):

|            | bias | tail |
|------------|------|------|
| riskprism  | 0.980 | 4.7% |
| FT replica | 1.003 | 5.6% |

A tie on bias; we're better calibrated in the tails. Per-ETF wins split
both ways. Honest reading: **for plain diversified long portfolios, a
textbook model (sample cov + regression betas) is nearly as well-calibrated
as the full commercial-recipe machinery.**

**Single names** (21 stocks across cap buckets, 69 biweekly formations,
1,449 scores):

|            | pooled bias | tail | large-cap bias | small/mid bias |
|------------|------|------|------|------|
| riskprism  | 1.097 | 7.5% | 1.016 | 1.167 |
| FT replica | 1.082 | 6.7% | 1.055 | 1.106 |

Another effective tie — both under-forecast single names (weekly stock
returns are fat-tailed; both models' >5% tails say the normal z benchmark
flatters no one). We win large caps; they win small/mids (our small-cap
specific risk under-forecasts — now on the punch list). They over-forecast
the vol-heavy names (NVDA 0.88, TSLA 0.88 — betas on vol-scaled factors
overshoot); we're near 1 there.

**Concentrated books**: equal-weight mega-tech 5: FT 0.990 vs RP 1.097
(theirs better). Dollar-neutral NVDA/AMD pair: RP 1.006 vs FT 0.912 (ours
better — cross-sectional exposures beat time-series betas once market beta
cancels).

**April 2025 vol shock**: both ramped ~+35–38% within 4 weeks (a −9% week
moves even an equal-weight window). Ours moved faster in the first two
weeks. The real difference is the decay: 3–6 months later ours had come
back down (2.39% weekly) while theirs stayed pinned (2.66%) — an
equal-weight 252d window holds a shock at full weight for exactly a year,
then drops it off a cliff. Both under-forecast the post-shock quarter
(1.65 vs 1.76): vol clustering beats everyone.

**Coverage** (stratified 50-name sample of our 6,311 covered names against
their loadings endpoint): mid/large caps 10/10 both. Micro/small they miss
1–2 of 10. Recent listings and coverage-tail names: **they miss 8 of 10 we
price** (some deliberately — OTC names are outside their stated universe).
This is what the structural specific-risk prior buys. On BMNR (listed
mid-2025, crypto-treasury, ~150% realized vol) both models fail in opposite
directions: our early forecasts graded 3.6 (under), the replica forecast
700%+ annualized vol and graded 0.29 (over) — nobody's covered in glory,
but we priced it 13 weeks before their history could.

**Specific-risk levels**: theirs run systematically lower (AAPL 21% vs our
28%, XOM 11% vs 19%) — in-sample residuals from up to 45 regressors absorb
variance. The z-scores say the truth is in between: they were fine on XOM
(1.08 vs our over-forecast 0.72), we were fine on JNJ (0.84 theirs 1.10).

## What we take from it

1. On unconditional calibration for long diversified portfolios over a
   mostly-calm 2.5 years, methodology sophistication barely shows. The
   USE4-style machinery pays at the edges: tails, shock decay, shorts and
   pairs, new/small names, and (untested for them, graded 1.14 for us)
   optimized portfolios — sample covariance + an optimizer is the classic
   bias amplifier, and none of their published surface tests it.
2. Their true live calibration is plausibly worse than the replica's:
   the replica inherits their reconstructed factor history and avoids
   their documented beta back-fill. Our numbers carry no look-ahead by
   construction and re-grade weekly in public.
3. Their genuine strengths stand: macro/country/industry breadth via ETF
   factors, applicability to any return stream (funds, ETFs, books without
   holdings), a genuinely open keyless API, and replicable factor
   equations. Different tool, honestly built, complementary aim.
4. Our small/mid-cap specific risk under-forecasts (bias 1.17) — the
   sharpest self-finding in this exercise.

One-off scripts (replica, backtests, coverage sample) live outside the
repo; the FT inputs are their public bulk download and live endpoints,
riskprism inputs are the published `model-2026-08-24` artifacts.
