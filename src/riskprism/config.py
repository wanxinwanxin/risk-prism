"""Model configuration and constants for the PRISM-US medium-horizon model."""

from dataclasses import dataclass, asdict

MARKET_FACTOR = "market"

STYLE_FACTORS = [
    "size",
    "value",
    "momentum",
    "beta",
    "volatility",
    "liquidity",
    "quality",
    "leverage",
]


@dataclass(frozen=True)
class ModelConfig:
    """Parameters of the medium-horizon US model.

    Estimation is daily against weekly-formed exposures; half-lives are
    in trading days. The methodology behind each choice is documented in
    docs/METHODOLOGY.md; changing any of these constitutes a new model
    version.
    """

    version: str = "PRISM-US-MH-0.7"
    # Prior versions whose regression/exposure definitions match this one:
    # their factor-return history may be appended to (risk construction on
    # top differs, but validation is recomputed from history every build).
    # v0.7 split Market Sensitivity (beta) out of volatility and redefined
    # volatility as beta-orthogonalized residual volatility — exposure
    # definitions changed, so v0.6's daily factor returns are not
    # commensurable and the history rebuilds cold (still nearly free:
    # capture-forward history barely exceeds provider lookback).
    compatible_prior_versions: tuple = ()
    # Exposures form on Fridays; regressions run on every trading day
    # against the week's formation exposures ("weekly formation, daily
    # estimation"). Daily sampling is what buys effective observations:
    # N_eff = (1+lam)/(1-lam) ~ 730 at the 252d correlation half-life vs
    # ~75 at the old 26-week one, with better calendar responsiveness.
    frequency: str = "W-FRI"
    ann_factor: float = 252.0
    horizon_days: int = 5  # validation scores 1-week-ahead forecasts

    # Covariance estimation (half-lives in trading days; USE4S template:
    # 84d vol / 504d corr — we use 252d corr, warmable within a ~4y panel)
    corr_half_life: int = 252
    vol_half_life: int = 84
    specific_half_life: int = 84
    eig_floor: float = 1e-10

    # Optimization-bias correction on the factor covariance:
    # "blend" = Bloomberg-style correlation blending (shipped default,
    # their published w=0.8 / 25% of components), "eigen" = eigenfactor
    # risk adjustment (Menchero/Wang/Orr 2011 — implemented but off: at
    # K=20 weekly it overcorrects broad portfolios), "none" = off.
    # Full A/B evidence in docs/DECISIONS.md §9.
    factor_cov_adjust: str = "blend"
    eigen_adjust_sims: int = 200
    eigen_adjust_a: float = 1.4
    eigen_refresh_periods: int = 126  # state replay: recompute profile this often
    blend_weight: float = 0.8
    blend_components_frac: float = 0.25

    # Newey-West serial-correlation adjustment of variances (Bartlett
    # weights; ratio clipped for robustness). USE4's daily lag counts.
    nw_factor_lags: int = 5
    nw_specific_lags: int = 1  # implementation applies lag-1
    nw_ratio_min: float = 0.5
    nw_ratio_max: float = 2.0

    # Volatility Regime Adjustment: EWMA of the cross-sectional bias
    # statistic, applied as a multiplier to all factor (and, separately,
    # specific) vols. USE4S uses half the vol half-life (42d vs 84d).
    vra_half_life: int = 42
    vra_lambda_min: float = 0.5   # bounds on the multiplier itself
    vra_lambda_max: float = 2.0

    # Bayesian shrinkage of specific vol toward size-bucket means
    # (USE4: q=0.1, deciles). Buckets on the size exposure; equal-weighted
    # bucket means (USE4 cap-weights; we deviate for replayability).
    specific_shrink_q: float = 0.1
    specific_shrink_buckets: int = 10

    # Exposure construction
    winsor_z: float = 3.0
    momentum_skip_days: int = 21
    momentum_window_days: int = 252
    volatility_window_days: int = 252
    liquidity_window_days: int = 63

    # Specific risk: EWMA blended with a cross-sectional structural model;
    # blend weight w = T/(T + structural_t0) by residual history length
    # (both in trading days).
    min_specific_obs: int = 63
    structural_t0: int = 126

    # Observations before the running state's forecasts are scored
    min_warmup_obs: int = 126

    # Estimation universe (participates in factor regressions)
    min_price: float = 2.0
    min_dollar_adv: float = 1e6
    min_weekly_obs: int = 26

    # Coverage universe (gets exposures + risk, via priors where needed)
    coverage_min_price: float = 1.0
    coverage_max_stale_days: int = 10

    # Capture-forward history (weekly formation snapshots and daily
    # regression rows share the same calendar cap)
    history_cap_weeks: int = 156
    history_cap_days: int = 780
    delist_failure_price: float = 5.0
    delist_failure_return: float = -0.30

    # Shepard (2009) second-order correction, applied to reported
    # forecasts only when the caller declares the portfolio was optimized
    # against this model: true vol ~ predicted / (1 - K/N_eff).
    shepard_correction: bool = True

    # Regression
    min_assets_per_regression: int = 50

    # Optimized test portfolios in the validation panel (the documented
    # hard case for risk models: optimizers seek out underestimated
    # directions — Shepard 2009, Menchero/Wang/Orr 2011).
    opt_universe: int = 500      # top names by cap eligible for optimization
    opt_random_alphas: int = 3   # random-alpha min-risk portfolios per week

    def to_dict(self) -> dict:
        return asdict(self)
