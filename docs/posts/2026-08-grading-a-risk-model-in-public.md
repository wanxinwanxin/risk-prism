# Grading an open-source risk model in public

*2026-08-21 · [riskprism](https://risk-prism-production.up.railway.app) · draft*

Commercial factor risk models — Barra, Axioma, Bloomberg — are validated in
whitepapers: a snapshot of bias statistics on a sample the vendor chose,
published once, never re-run. The models may be excellent. You can't check.

riskprism is a Barra-style US equity factor risk model built entirely from
public data — daily prices and SEC EDGAR filings — that publishes its whole
validation history and re-grades itself every week. This post is about what
that grading found, including the parts that were embarrassing.

## The setup

The model covers ~3,000 US stocks with 20 factors (market + 7 styles + 12
Fama-French industries). Exposures form every Friday; a cross-sectional
regression runs every trading day against those frozen exposures; factor
and specific covariances are EWMA estimates with the standard commercial
adjustments (Newey-West, volatility regime adjustment, Bayesian shrinkage,
correlation blending). Everything — code, artifacts, methodology, decision
log — is public.

Every week, the model writes down a volatility forecast for a panel of test
portfolios: factor portfolios, real traded ETFs (SPY, IWM, MTUM, VLUE,
QUAL, USMV), random portfolios, and — this matters later — portfolios
*optimized against the model itself*. Then it gets graded:
`z = realized return / forecast vol`. If the forecasts are honest, z is
standard normal: its standard deviation (the **bias statistic**) should be
1.0, and |z| > 1.96 should happen about 5% of the time. Every build
re-scores the *entire* history under the current methodology, so there is
nowhere to hide a regression.

Current scoreboard, 121 weeks, 3,992 graded forecasts:

| statistic | value | ideal |
|---|---|---|
| overall bias statistic | 1.05 | 1.00 |
| weeks with \|z\| > 1.96 | ~6% | ~5% |
| Mincer–Zarnowitz slope (realized on forecast variance) | ~1.0–1.1 | 1.00 |
| traded-ETF bias range | 0.87–1.16 | ≈1 |

## The interesting failure: portfolios that fight back

The hardest test for any risk model is a portfolio optimized *against it*.
An optimizer hunts the covariance matrix's underestimated directions, so
optimized portfolios always run hotter than forecast — MSCI measured bias
statistics of 1.4–1.5 on its own pre-adjustment model, and Shepard (2009)
derived the size of the effect: true vol ≈ forecast ÷ (1 − K/N_eff), where
K is the number of factors and N_eff the covariance estimator's effective
sample size.

We test this on ourselves: every week we build a minimum-variance portfolio
and three optimized random-alpha portfolios against our own matrix and
grade them like everything else.

- **v0.3** (weekly estimation): min-variance bias **1.36**. Measured,
  published, ugly.
- **v0.4**: A/B-tested both published matrix-side cures. MSCI's
  eigenfactor adjustment reproduced their reported small-eigenvalue bias
  curve almost exactly — and still made broad portfolios *worse* at our
  sample size, so we shipped the alternative (Bloomberg-style correlation
  blending) and documented the eigenfactor result as a negative. Bias:
  1.36 → **1.29**. Barely worth it.
- **v0.5**: the actual cure was not an adjustment but *more data*. An
  EWMA's effective sample size is set by its half-life in observations, so
  switching estimation from weekly to daily multiplied N_eff by ~10 at the
  same calendar memory. Min-variance bias collapsed to **1.09** — which is
  exactly Shepard's 1/(1 − K/N_eff) at our parameters (1.090). When the
  measured bias lands on the theoretical floor, you stop tuning: the
  residual is now applied as an explicit reporting correction — pass
  `optimized=true` to the API and reported vols scale by that factor.

That arc — 1.36 → 1.29 → 1.09, ending on the theory line — is the most
useful thing the public grading produced. It found the bug, ranked the
cures, and told us when to stop.

## The embarrassing finding: two factors were dead

v0.5 also started publishing per-factor regression t-statistics (the same
relevance check Axioma publishes). Market: significant in 86% of daily
cross-sections. Volatility: 83%. Momentum: 75%. Then:

- **value: 4%**
- **quality: 1%**

Both were single-descriptor factors (book/price and ROE). The axes existed
in the model but carried almost no independent return variance — something
we would never have noticed without publishing the statistic that exposed
it.

v0.6 rebuilt both as multi-descriptor composites, the way commercial models
have always done it: value = book/earnings/cash-flow/sales yields; quality
= ROE, ROA, cash-flow-on-assets, gross margin — all from point-in-time
EDGAR XBRL (operating cash flow is filed by ~98% of companies; nobody
seems to use it). Result: **quality went from 1.6% significant to 49%**;
value tripled to 12%. And, in the spirit of grading in public: the new
value factor's own style portfolio is now *underforecast* (bias 1.24)
precisely because the axis finally carries variance the young EWMA history
is still learning. A dead factor is perfectly calibrated the way a stopped
clock is right twice a day. We shipped the live one.

## Could something simpler have done as well?

Fair question for any factor model. We replay every classic public
forecaster — RiskMetrics EWMA, trailing vol, realized-vol EWMA,
Fama-French 5-factor — point-in-time over the *same* portfolios and weeks,
graded with the same statistics. The factor model wins on RMSE and MZ R²
and holds its own everywhere else — but the honest point is different:
each univariate baseline needs that portfolio's own return history, while
the factor model prices *any* weight vector cold, including portfolios
that didn't exist last week. Matching the baselines on their home turf
while doing that is the actual bar.

And against the commercial models? Their licenses forbid published
benchmarks, so we run *ourselves* under *their* published test protocols
(USE4's rolling monthly bias statistics and MRAD; Axioma's benchmark-index
bias bands) and print the comparison, caveats and all.

## What's deliberately missing

No analyst estimates (IBES is proprietary — the one systematic sacrifice
vs commercial value/growth factors), no GICS (licensed; Fama-French
schemes are public domain), no ESG factors (no redistributable
point-in-time source exists). The full roadmap — including what's at
parity with commercial models today and what only time can close — is
published alongside the methodology.

## Links

- Explorer + full validation report card: https://risk-prism-production.up.railway.app
- Code (MIT): https://github.com/wanxinwanxin/risk-prism
- Weekly artifacts, free: https://github.com/wanxinwanxin/risk-prism/releases/latest
- For AI agents: `/model.md` and an MCP server
- Decision log with every measured A/B, including the failures:
  [docs/DECISIONS.md](https://github.com/wanxinwanxin/risk-prism/blob/main/docs/DECISIONS.md)
