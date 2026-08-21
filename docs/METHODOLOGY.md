# PRISM-US-MH methodology

Version `PRISM-US-MH-0.5`. Medium-horizon US equity fundamental factor
model: **weekly formation, daily estimation** — exposures form on
Fridays, cross-sectional regressions run on every trading day. All
parameters live in `riskprism.config.ModelConfig`.

v0.3 added four pieces of the commercial-model recipe (Newey-West
variance adjustment, Volatility Regime Adjustment, Bayesian specific-risk
shrinkage, optimized portfolios in the validation panel); v0.4 added
Bloomberg-style correlation blending after an A/B against the eigenfactor
risk adjustment (DECISIONS.md §9). **v0.5 switched estimation from weekly
to daily returns** (DECISIONS.md §10): an EWMA's effective sample size is
set by its half-life in observations, so daily sampling gives the
correlation matrix ~730 effective observations (252-day half-life) where
weekly gave ~75 (26-week half-life) — the mechanism behind commercial
models' conditioning advantage. The switch changed the regression data
unit, so v0.5 rebuilt history cold (`compatible_prior_versions` is empty;
the weekly capture-forward history was entirely cold-start at the time,
so nothing survivorship-free was lost). Validation is recomputed from
history on every build regardless (see Validation below).

## Two universes

Candidates are EDGAR-registered US issuers, filtered to 1–5 letter tickers
(single-letter class suffixes allowed), one per CIK (first-listed = primary
class).

- **Estimation universe** — participates in the factor-return regressions:
  last price ≥ $2, 21-day median dollar volume ≥ $1M, ≥ 26 weeks of
  history, each evaluated at the name's own last traded date (so a stock
  that was liquid before delisting still contributes its history).
- **Coverage universe** — gets exposures and risk: every name trading
  within 10 days of the build date at ≥ $1. No history requirement — a
  week-old IPO is covered. Risk comes through the factor structure
  (x'Fx uses the covariance estimated from liquid names) plus the
  structural specific-risk prior below. Exposure standardization
  statistics are fit on the estimation universe and applied to everyone,
  so illiquid tails can't distort the scale.

## Capture-forward history & delistings

Each build can append to a prior build's artifacts (`--prior`): only new
weeks are regressed, and the prior factor-return and residual history is
kept — including rows from names that have since delisted. A name whose
prices stop gets an imputed final-week return: −30% if its last price was
under $5 (performance delisting, per Shumway 1997), 0 otherwise (mergers —
the last traded price already reflects deal terms). History is capped at a
trailing 156 weeks.

Consequence: history recorded after launch is survivorship-free by
construction, and because the EWMA half-lives are 13/26 weeks, the biased
cold-start history decays out of the live model within ~18–24 months.
Weeks recorded before launch remain biased; factor-return *means* are
affected more than the covariances the model ships. A version bump that
changes exposure or regression definitions discards prior history (cold
rebuild); bumps that only change risk construction on top (v0.3, v0.4)
keep appending — the prior versions listed in
`config.compatible_prior_versions` — because validation is recomputed
from history under the current methodology every build anyway.

### Severity, quantified (audit 2026-08-20, `scripts/survivorship_audit.py`)

Measured on SEC bulk archives, which retain dead filers. Of ~11.0k XBRL
filers actively filing at the window start (2023-01), **3,115 ceased
filing during the window** (~28% over 3.6y, ~8%/yr — inflated relative to
the classic ~2%/yr listed-stock delisting rate by OTC registrants and the
2023–24 SPAC wind-down; the ADV/price-filtered investable universe sits
between). Departures skew small: median last-reported book equity $115M
vs $553M for survivors (28th percentile of the living), and the departed
sum to **~8.7% of filer book equity**. Implied upper bound on
cap-weighted return-mean bias: **~1.4bp/week** at a −30% delisting
return, ~2.5bp/week at −55% — under ~1.3%/yr, concentrated in
small-cap-heavy factor means. Covariance estimates are affected at
second order.

## Style factors

Raw descriptors, each winsorized at ±3σ (two passes), standardized to
cap-weighted mean 0 / equal-weighted std 1, missing → 0:

| Factor | Descriptor |
|---|---|
| size | ln(market cap) |
| value | book equity / market cap (book > 0 only) |
| momentum | 12-month return skipping the most recent month (252d window, 21d skip) |
| volatility | 252-day daily return std, annualized (≥126 obs) |
| liquidity | ln(63-day median dollar volume / market cap) |
| quality | ROE: annual (10-K FY) net income / book equity |
| leverage | total liabilities / total assets |

Fundamentals are point-in-time: values are used only after their EDGAR
`filed` date. Net income uses annual filings only (durations of 300–400
days) to avoid mixing quarterly and cumulative XBRL values.

## Industries

Fama-French 12 groups mapped from EDGAR SIC codes. One-hot exposures.

## Cross-sectional regression

**Weekly formation, daily estimation.** Exposures are computed on each
Friday exactly as before; then every trading day of the following week
regresses that day's returns on those frozen exposures:
`r_i = f_mkt + Σ_s X_is f_s + Σ_j I_ij f_j + ε_i`, estimated by WLS with
√(market cap) weights (normalized). Identification: industry factor
returns are constrained to cap-weighted zero — implemented via a
restriction matrix that eliminates the largest-cap industry (numerically
safest divisor). The market factor is therefore the cap-weighted market
return; styles and industries are return deltas relative to it. Five
cross-sections per week instead of one; days with fewer than 50 usable
assets are skipped. Each regression also produces WLS t-statistics per
factor, published as the factor-quality table in `/model.md`
(the Axioma-style %-of-periods-significant check).

## Factor covariance

EWMA on daily factor returns, zero-mean convention. Volatilities use an
84-trading-day half-life (responsive; matches USE4S); correlations use
252 days (stable). Daily sampling is the load-bearing choice: effective
sample size N_eff = (1+λ)/(1−λ) ≈ 730 observations for correlations vs
~75 under the old weekly 26-week half-life — same calendar memory, five
times the data, which is what conditions the K×K matrix optimizers lean
on.

**Newey-West adjustment (v0.3, re-parameterized for daily in v0.5)**:
annualizing daily variance by ×252 assumes iid daily returns;
autocorrelated factor returns violate that (and daily data adds
microstructure autocorrelation — the reason USE4 pairs daily estimation
with NW). Per-factor variances carry a Bartlett-weighted Newey-West
adjustment with 5 lags (`var_adj = var + 2·Σ_l (1−l/(L+1))·γ_l`), ratio
clipped to [0.5, 2]. Applied to variances only, which keeps V·C·V
trivially PSD (Menchero, Orr & Wang 2011, §4.1).

**Volatility Regime Adjustment (v0.3)**: each week the cross-sectional
factor bias statistic `B_t² = mean_k (f_kt / σ_kt)²` is computed against
the pre-update forecast vols; its EWMA (42-day half-life — USE4S's exact
parameter) gives a multiplier
`λ_F = √(EWMA[B²])`, clipped to [0.5, 2], which scales all factor vols —
covariance ×λ_F², correlations untouched. This is what lets the model
catch regime shifts that half-life-bound EWMA lags: USE4's version held
rolling bias statistics near 1.0 through 2008–09, where the unadjusted
model swung 1.3 → 0.7 (Menchero & Morozov, "Improving Risk Forecasts
Through Cross-Sectional Observations"). Directly targets our measured
Mincer–Zarnowitz slope of 0.70.

**Correlation blending (v0.4)**: the correlation matrix is blended with
its own rank-5 PCA reconstruction (plus an idiosyncratic diagonal
restoring unit diagonals) at Bloomberg's published parameters — w = 0.8
sample weight, J = ⌈K/4⌉ = 5 components (Menchero, Bloomberg MAC2/MAC3).
This suppresses the noise in the matrix's small directions that
optimizers exploit: min-variance-portfolio bias 1.36 → 1.29 with no
measurable effect on any other test portfolio. The eigenfactor risk
adjustment (Menchero, Wang & Orr 2011) is implemented behind
`config.factor_cov_adjust="eigen"` but off by default: our Monte-Carlo
reproduces their ~40% small-eigenfactor underestimation exactly, yet at
K=20 weekly factors the adjustment's mid-rank inflation over-forecasts
broad portfolios (equal-weight/random biases fall to ~0.8) — a negative
result documented in DECISIONS.md §9.

The combined matrix is annualized (×52) and repaired to PSD by eigenvalue
flooring at 1e-10.

## Specific risk

Two estimates, blended by history length:

1. **Time-series**: per-asset EWMA (84-day half-life) of squared daily
   regression residuals, with a lag-1 Newey-West adjustment (v0.3),
   annualized. Requires ≥ 63 observations.
2. **Structural**: each week, ln(time-series vol) is regressed
   cross-sectionally on characteristics — size, volatility, and liquidity
   exposures plus industry — over assets that have good history. The fit
   predicts specific vol for *every* asset (with a Duan smearing
   correction for the exp() retransformation).

Blend: `σᵢ = wᵢ·TSᵢ + (1−wᵢ)·structuralᵢ` with `wᵢ = Tᵢ/(Tᵢ + 126)` (days).
Assets with no residual history (IPOs, coverage-only names) get the pure
structural prior. `asset_meta.parquet` records each asset's blend weight
so consumers can distinguish measured from inferred.

**Bayesian shrinkage (v0.3)**: blended vols are shrunk toward their
size-decile mean with distance-dependent intensity
`v = q·|σ−σ̄| / (Δ + q·|σ−σ̄|)`, q = 0.1 — USE4's exact device, which
flattens the classic decile tilt (low-vol names underforecast at ~1.08,
high-vol overforecast at ~0.92, becoming ~flat at 1.0 in their tests).
Deviation from USE4: buckets key on the size exposure with equal-weighted
bucket means (USE4 cap-weights within cap deciles) so the shrinkage is
exactly reproducible from shipped exposures alone.

**Specific VRA (v0.3)**: a separate multiplier λ_S from the EWMA of the
cross-sectional specific bias statistic (equal-weighted across assets
with ≥13 residual observations; USE4 cap-weights), applied to all
specific vols. Both multipliers ship in `meta` (`vra_factor`,
`vra_specific`).

## Portfolio analytics

For weights `w`: exposures `x = Xᵀw`; factor variance `xᵀFx`; specific
variance `Σ wᵢ²sᵢ²`; total vol is the square root of the sum.
`portfolio_risk(..., optimized=True)` (also exposed on the MCP tool)
additionally applies the Shepard (2009) second-order correction
`1/(1 − K/N_eff)` to the reported vols: portfolios optimized against the
model exploit its estimation noise, so their raw forecasts understate
risk — the correction lives at the reporting layer so the matrix stays
unbiased for pre-specified portfolios (see the `opt` rows of the
published validation for the empirically measured counterpart). Factor
variance contributions `x_k (Fx)_k` sum to factor variance; asset
contributions `wᵢ · (Σw)ᵢ / σ_p` sum to total vol. Stress tests are
first-order: `ΔP&L ≈ Σ x_k Δf_k`.

## Validation

Continuous and out-of-sample by construction — and, since v0.3,
**recomputed from history on every build** (`model/revalidate.py`): a
fresh point-in-time risk state (recursive EWMA + Newey-West + blending +
VRA + shrinkage, warmed only on data through t) replays the entire
stored DAILY factor-return history and rescores every completed
formation week under the *current* methodology. Reconstruction is exact
for regressed names because each daily regression defines
`r_d,i = Xᵢ·f_d + ε_d,i`, so the true weekly return is recovered by
compounding `Π_d (1 + Xᵢ·f_d + ε_d,i) − 1` — including imputed delisting
returns. Forecasts scale from daily state variance to the one-week
horizon (×5; Newey-West has already absorbed the autocorrelation the
scaling assumes away). ETF exposures come from returns-based style
analysis on trailing DAILY returns (252-day window). Validation is
therefore a pure function of the shipped artifacts, never a mixture of
scores from different model versions.

The test panel: cap-weighted market, equal-weighted, top-minus-bottom
style spreads, cap-weighted industries, random 50-name baskets, six real
factor ETFs (returns-based point-in-time exposures), and — new in
v0.3 — **portfolios optimized against the model itself**: the global
minimum-variance portfolio and three random-alpha minimum-risk
portfolios (Σ⁻¹α via Woodbury over the top 500 names by cap, weekly).
Optimized portfolios are the documented worst case for risk models —
optimizers seek out the covariance matrix's underestimated directions
(Shepard 2009; Menchero, Wang & Orr 2011 measured bias statistics of
1.4–1.5 on such portfolios under sample covariance matrices). Publishing
this number is deliberately adversarial self-grading.

Headline metric: the **bias statistic** std(z) per portfolio (~1.0 =
calibrated; >1 = risk underforecast) with a ±2/√(2n) acceptance band,
plus the |z| > 1.96 exceedance rate (target ~5%), realized-vol ratios
from daily returns, and a Mincer–Zarnowitz regression. Current numbers
are on the explorer's Validation tab and in `/model.md` with every build.
