"""riskprism — open-source US equity factor risk model, built for AI agents."""

from riskprism.config import MARKET_FACTOR, STYLE_FACTORS, ModelConfig
from riskprism.risk import RiskModel

__version__ = "0.1.0"
__all__ = ["ModelConfig", "RiskModel", "STYLE_FACTORS", "MARKET_FACTOR", "__version__"]
