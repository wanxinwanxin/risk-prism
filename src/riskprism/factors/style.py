"""Style factor exposure construction.

Seven styles, all computable from prices plus EDGAR fundamentals:
size, value, momentum, volatility, liquidity, quality, leverage.
Each raw descriptor is winsorized, standardized to cap-weighted mean 0 /
equal-weighted std 1, and missing values are set to 0 (market average).
"""

import numpy as np
import pandas as pd

from riskprism.config import STYLE_FACTORS, ModelConfig
from riskprism.factors.transforms import process_exposure


def compute_style_exposures(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    fund_df: pd.DataFrame,
    as_of: pd.Timestamp,
    config: ModelConfig,
) -> tuple[pd.DataFrame, pd.Series]:
    """Style exposures and market caps as of one date.

    ``close``/``volume`` are daily wide panels; ``fund_df`` is indexed by
    ticker with point-in-time columns [book_equity, total_assets,
    total_liabilities, net_income, shares_out] valid at ``as_of``.

    Returns (exposures indexed ticker x STYLE_FACTORS, mktcap Series).
    """
    px = close.loc[:as_of]
    if px.empty:
        raise ValueError(f"No price history on or before {as_of}")
    pxf = px.ffill()
    last_px = pxf.iloc[-1]

    tickers = close.columns
    fund_df = fund_df.reindex(tickers)
    mktcap = (last_px * fund_df["shares_out"]).where(lambda s: s > 0)

    raw = {}
    raw["size"] = np.log(mktcap)
    raw["value"] = (fund_df["book_equity"] / mktcap).where(fund_df["book_equity"] > 0)

    skip, window = config.momentum_skip_days, config.momentum_window_days
    if len(pxf) >= window + skip + 1:
        raw["momentum"] = pxf.iloc[-(skip + 1)] / pxf.iloc[-(window + skip + 1)] - 1.0
    else:
        raw["momentum"] = pd.Series(np.nan, index=tickers)

    rets = px.pct_change()
    recent = rets.iloc[-config.volatility_window_days:]
    vol = recent.std() * np.sqrt(252)
    raw["volatility"] = vol.where(recent.count() >= config.volatility_window_days // 2)

    dollar_vol = (close * volume).loc[:as_of].iloc[-config.liquidity_window_days:]
    turnover = dollar_vol.median() / mktcap
    raw["liquidity"] = np.log(turnover.where(turnover > 0))

    raw["quality"] = (fund_df["net_income"] / fund_df["book_equity"]).where(
        fund_df["book_equity"] > 0
    )
    raw["leverage"] = (fund_df["total_liabilities"] / fund_df["total_assets"]).where(
        fund_df["total_assets"] > 0
    )

    exposures = pd.DataFrame(
        {name: process_exposure(raw[name], mktcap, z=config.winsor_z) for name in STYLE_FACTORS},
        index=tickers,
    )
    return exposures, mktcap
