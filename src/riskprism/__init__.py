"""riskprism — open-source US equity factor risk model, built for AI agents."""

from importlib.metadata import PackageNotFoundError, version as _dist_version

from riskprism.config import MARKET_FACTOR, STYLE_FACTORS, ModelConfig
from riskprism.lookthrough import (
    FundCoverageError,
    fund_risk,
    portfolio_risk_lookthrough,
)
from riskprism.risk import RiskModel

try:
    __version__ = _dist_version("riskprism")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0.dev0"
__all__ = ["ModelConfig", "RiskModel", "STYLE_FACTORS", "MARKET_FACTOR",
           "FundCoverageError", "fund_risk", "portfolio_risk_lookthrough",
           "__version__"]
