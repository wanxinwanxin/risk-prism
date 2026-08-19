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

    version: str = "PRISM-US-MH-0.2"
    frequency: str = "W-FRI"
    ann_factor: float = 52.0

    # Covariance estimation
    corr_half_life: int = 26
    vol_half_life: int = 13
    specific_half_life: int = 13
    eig_floor: float = 1e-10

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

    def to_dict(self) -> dict:
        return asdict(self)
