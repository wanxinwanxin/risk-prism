# OSRM-US-MH methodology

Version `OSRM-US-MH-0.1`. Weekly-frequency, medium-horizon US equity
fundamental factor model. All parameters live in `osrisk.config.ModelConfig`.

## Universe

EDGAR-registered US issuers, filtered to plain 1–5 letter tickers, one per
CIK (shortest ticker = primary share class heuristic). Liquidity filters:
last price ≥ $2, 21-day median dollar volume ≥ $1M, ≥ 26 weeks of history.

**Known limitation — survivorship bias**: the price panel is fetched as of
build time, so names delisted before the build are absent from the
regression history. Risk *forecasts* (the main product) are less affected
than historical factor-return studies. Treat `factor_returns.parquet` as
indicative, not research-grade, until a delisted-price source is added.

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

Per-asset EWMA (13-week half-life) of squared regression residuals,
annualized, then shrunk 30% toward the asset's size-quintile mean. Assets
with <13 residual observations receive the bucket mean outright.

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
