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

    version: str = "PRISM-US-MH-0.1"
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

    # Specific risk
    specific_shrinkage: float = 0.3
    n_size_buckets: int = 5
    min_specific_obs: int = 13

    # Universe filters
    min_price: float = 2.0
    min_dollar_adv: float = 1e6
    min_weekly_obs: int = 26

    # Regression
    min_assets_per_regression: int = 50

    def to_dict(self) -> dict:
        return asdict(self)
