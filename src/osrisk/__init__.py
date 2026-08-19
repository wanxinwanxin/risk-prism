"""osrisk — open-source US equity factor risk model, built for AI agents."""

from osrisk.config import MARKET_FACTOR, STYLE_FACTORS, ModelConfig
from osrisk.risk import RiskModel

__version__ = "0.1.0"
__all__ = ["ModelConfig", "RiskModel", "STYLE_FACTORS", "MARKET_FACTOR", "__version__"]
