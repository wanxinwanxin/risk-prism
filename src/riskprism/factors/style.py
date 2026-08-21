"""Style factor exposure construction.

Seven styles, all computable from prices plus EDGAR fundamentals:
size, value, momentum, volatility, liquidity, quality, leverage.

Value and quality are multi-descriptor composites (the single-descriptor
v0.5 versions measured significant in only 4% and 1% of daily
cross-sections; USE4 and Axioma both build these factors from several
descriptors). Each descriptor is z-scored separately, the composite is
the mean of the z-scores a name actually has, and the composite is
re-standardized — so a filer missing one XBRL tag is scored on the rest
instead of dropping to the market average.

Each raw descriptor is winsorized and standardized with statistics fit on
the estimation universe (``fit``) and applied to the full cross-section.
Missing fundamental descriptors are imputed with the industry median
first (a bank with unfiled tags is more like other banks than like the
market); anything still missing standardizes to 0 (market average).
"""

import numpy as np
import pandas as pd

from riskprism.config import STYLE_FACTORS, ModelConfig
from riskprism.factors.transforms import process_exposure, standardize, winsorize_z

def _composite(descs: dict[str, pd.Series], mktcap: pd.Series, z: float,
               fit: pd.Index | None) -> pd.Series:
    """Mean of available descriptor z-scores, re-standardized.

    Descriptors are z-scored without the fill-to-zero step so a missing
    tag drops out of the mean instead of pulling the composite toward 0.
    """
    zs = pd.DataFrame({
        name: standardize(winsorize_z(raw, z=z, fit=fit), mktcap, fit=fit)
        for name, raw in descs.items()
    })
    return zs.mean(axis=1, skipna=True)


def compute_style_exposures(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    fund_df: pd.DataFrame,
    as_of: pd.Timestamp,
    config: ModelConfig,
    industries: pd.Series | None = None,
    fit: pd.Index | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Style exposures and market caps as of one date.

    ``close``/``volume`` are daily wide panels; ``fund_df`` is indexed by
    ticker with the point-in-time FUND_FIELDS columns (balance-sheet
    stocks, annual flows, shares outstanding) valid at ``as_of``.

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

    be = fund_df["book_equity"]
    ta = fund_df["total_assets"].where(fund_df["total_assets"] > 0)
    rev = fund_df["revenues"].where(fund_df["revenues"] > 0)
    gp = fund_df["gross_profit"].fillna(fund_df["revenues"] - fund_df["cost_of_revenue"])

    value_descs = {
        "book_to_price": (be / mktcap).where(be > 0),
        "earnings_to_price": fund_df["net_income"] / mktcap,
        "cashflow_to_price": fund_df["op_cashflow"] / mktcap,
        "sales_to_price": rev / mktcap,
    }
    quality_descs = {
        "roe": (fund_df["net_income"] / be).where(be > 0),
        "roa": fund_df["net_income"] / ta,
        "ocf_to_assets": fund_df["op_cashflow"] / ta,
        "gross_margin": gp / rev,
    }
    raw["leverage"] = fund_df["total_liabilities"] / ta

    if industries is not None:
        ind = industries.reindex(tickers)

        def impute(s: pd.Series) -> pd.Series:
            return s.fillna(s.groupby(ind).transform("median"))

        value_descs = {k: impute(v) for k, v in value_descs.items()}
        quality_descs = {k: impute(v) for k, v in quality_descs.items()}
        raw["leverage"] = impute(raw["leverage"])

    raw["value"] = _composite(value_descs, mktcap, config.winsor_z, fit)
    raw["quality"] = _composite(quality_descs, mktcap, config.winsor_z, fit)

    exposures = pd.DataFrame(
        {name: process_exposure(raw[name], mktcap, z=config.winsor_z, fit=fit)
         for name in STYLE_FACTORS},
        index=tickers,
    )
    return exposures, mktcap
