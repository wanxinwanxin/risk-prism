# PRISM-US-MH methodology

Version `PRISM-US-MH-0.2`. Weekly-frequency, medium-horizon US equity
fundamental factor model. All parameters live in `riskprism.config.ModelConfig`.

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
affected more than the covariances the model ships. A methodology version
bump discards prior history (cold rebuild) rather than appending across
incompatible definitions.

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

Each week: `r_i = f_mkt + Σ_s X_is f_s + Σ_j I_ij f_j + ε_i`, estimated by
WLS with √(market cap) weights (normalized). Identification: industry
factor returns are constrained to cap-weighted zero — implemented via a
restriction matrix that eliminates the largest-cap industry (numerically
safest divisor). The market factor is therefore the cap-weighted market
return; styles and industries are return deltas relative to it.

Weeks with fewer than 50 usable assets are skipped.

## Factor covariance

EWMA on weekly factor returns, zero-mean convention. Volatilities use a
13-week half-life (responsive); correlations use 26 weeks (stable). The
combined matrix is annualized (×52) and repaired to PSD by eigenvalue
flooring at 1e-10. Newey-West autocorrelation adjustment and eigenfactor
risk adjustment are v2 candidates.

## Specific risk

Two estimates, blended by history length:

1. **Time-series**: per-asset EWMA (13-week half-life) of squared
   regression residuals, annualized. Requires ≥ 13 observations.
2. **Structural**: each week, ln(time-series vol) is regressed
   cross-sectionally on characteristics — size, volatility, and liquidity
   exposures plus industry — over assets that have good history. The fit
   predicts specific vol for *every* asset (with a Duan smearing
   correction for the exp() retransformation).

Final estimate: `σᵢ = wᵢ·TSᵢ + (1−wᵢ)·structuralᵢ` with
`wᵢ = Tᵢ/(Tᵢ + 26)`. Assets with no residual history (IPOs, coverage-only
names) get the pure structural prior. `asset_meta.parquet` records each
asset's blend weight so consumers can distinguish measured from inferred.

## Portfolio analytics

For weights `w`: exposures `x = Xᵀw`; factor variance `xᵀFx`; specific
variance `Σ wᵢ²sᵢ²`; total vol is the square root of the sum. Factor
variance contributions `x_k (Fx)_k` sum to factor variance; asset
contributions `wᵢ · (Σw)ᵢ / σ_p` sum to total vol. Stress tests are
first-order: `ΔP&L ≈ Σ x_k Δf_k`.

## Validation (to build out)

Planned: weekly bias statistics (realized/forecast risk ratios) on random
and style-tilted test portfolios, published as a notebook with each release.
Until then, treat forecasts with appropriate skepticism.
