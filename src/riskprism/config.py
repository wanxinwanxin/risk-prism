"""Model configuration and constants for the PRISM-US medium-horizon model."""

from dataclasses import dataclass, asdict

MARKET_FACTOR = "market"

STYLE_FACTORS = [
    "size",
    "value",
    "momentum",
    "volatility",
    "liquidity",
    "quality",
    "leverage",
]


@dataclass(frozen=True)
class ModelConfig:
    """Parameters of the medium-horizon (weekly) US model.

    Half-lives are in weeks. The methodology behind each choice is
    documented in docs/METHODOLOGY.md; changing any of these constitutes
    a new model version.
    """

    version: str = "PRISM-US-MH-0.3"
    # Prior versions whose regression/exposure definitions match this one:
    # their factor-return history may be appended to (risk construction on
    # top differs, but validation is recomputed from history every build).
    compatible_prior_versions: tuple = ("PRISM-US-MH-0.2",)
    frequency: str = "W-FRI"
    ann_factor: float = 52.0

    # Covariance estimation
    corr_half_life: int = 26
    vol_half_life: int = 13
    specific_half_life: int = 13
    eig_floor: float = 1e-10

    # Newey-West serial-correlation adjustment of variances (Bartlett
    # weights; ratio clipped for robustness). Weekly returns need fewer
    # lags than USE4's daily 5/2.
    nw_factor_lags: int = 2
    nw_specific_lags: int = 1
    nw_ratio_min: float = 0.5
    nw_ratio_max: float = 2.0

    # Volatility Regime Adjustment: EWMA of the cross-sectional bias
    # statistic, applied as a multiplier to all factor (and, separately,
    # specific) vols. USE4 uses half the vol half-life (42d vs 84d).
    vra_half_life: int = 8
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
    # blend weight w = T/(T + structural_t0) by residual history length.
    min_specific_obs: int = 13
    structural_t0: int = 26

    # Estimation universe (participates in factor regressions)
    min_price: float = 2.0
    min_dollar_adv: float = 1e6
    min_weekly_obs: int = 26

    # Coverage universe (gets exposures + risk, via priors where needed)
    coverage_min_price: float = 1.0
    coverage_max_stale_days: int = 10

    # Capture-forward history
    history_cap_weeks: int = 156
    delist_failure_price: float = 5.0
    delist_failure_return: float = -0.30

    # Regression
    min_assets_per_regression: int = 50

    # Optimized test portfolios in the validation panel (the documented
    # hard case for risk models: optimizers seek out underestimated
    # directions — Shepard 2009, Menchero/Wang/Orr 2011).
    opt_universe: int = 500      # top names by cap eligible for optimization
    opt_random_alphas: int = 3   # random-alpha min-risk portfolios per week

    def to_dict(self) -> dict:
        return asdict(self)
