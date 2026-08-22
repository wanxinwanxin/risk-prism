"""Style factor exposure construction.

Nine styles, all computable from prices plus EDGAR fundamentals: size,
value, growth, momentum, beta, volatility, liquidity, quality, leverage.
(Dividend yield was measured and rejected in v0.8 — significant in 0% of
daily cross-sections and collinear with value; DECISIONS.md §13.)

Beta (Market Sensitivity) and volatility come from one time-series
regression per name: daily returns over the volatility window on the
cap-weighted market return. Beta is the slope; volatility is the
annualized std of the residuals, orthogonalized to beta cross-sectionally
(v0.6's raw total volatility conflated the two dimensions — USE4 and
Axioma both carry them as separate factors).

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

def _orthogonalize(y: pd.Series, x: pd.Series, mktcap: pd.Series,
                   fit: pd.Index | None) -> pd.Series:
    """Residual of a cross-sectional regression of y on x, re-standardized.

    The slope fits on ``fit`` rows (default: all) and applies everywhere.
    """
    ref = y.index.intersection(fit) if fit is not None else y.index
    yr, xr = y.loc[ref], x.loc[ref]
    ok = yr.notna() & xr.notna()
    if int(ok.sum()) < 2:
        return y
    xc = xr[ok] - xr[ok].mean()
    denom = float((xc ** 2).sum())
    if denom == 0:
        return y
    b = float(((yr[ok] - yr[ok].mean()) * xc).sum()) / denom
    a = float(yr[ok].mean()) - b * float(xr[ok].mean())
    return standardize(y - (a + b * x), mktcap, fit=fit).fillna(0.0)


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

    # beta and residual volatility, from one regression per name: daily
    # returns over the volatility window on the cap-weighted market return
    # (weights fixed at as-of caps; the market is defined on the fit
    # universe when one is given, matching the estimation-universe market
    # the factor regressions see).
    rets = px.pct_change()
    recent = rets.iloc[-config.volatility_window_days:]
    ref_cols = recent.columns.intersection(fit) if fit is not None else recent.columns
    w_ref = mktcap.reindex(ref_cols).where(lambda s: s > 0)
    ref = recent[ref_cols]
    w_present = ref.notna().mul(w_ref, axis=1).sum(axis=1)
    mkt = ref.mul(w_ref, axis=1).sum(axis=1, min_count=1) / w_present.where(w_present > 0)

    X = recent.to_numpy(dtype=float)
    m = mkt.to_numpy(dtype=float)
    mask = np.isfinite(X) & np.isfinite(m)[:, None]
    n_obs = mask.sum(axis=0)
    safe_n = np.maximum(n_obs, 1)
    X0 = np.where(mask, X, 0.0)
    M0 = np.where(mask, m[:, None], 0.0)
    xbar = X0.sum(axis=0) / safe_n
    mbar = M0.sum(axis=0) / safe_n
    xc = np.where(mask, X0 - xbar, 0.0)
    mc = np.where(mask, M0 - mbar, 0.0)
    cov = (xc * mc).sum(axis=0) / safe_n
    var = (mc * mc).sum(axis=0) / safe_n
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = cov / var
        resid = np.where(mask, xc - beta * mc, 0.0)
        rvol = np.sqrt((resid ** 2).sum(axis=0) / safe_n * 252.0)
    ok_obs = ((n_obs >= config.volatility_window_days // 2)
              & np.isfinite(beta) & (var > 0))
    raw["beta"] = pd.Series(np.where(ok_obs, beta, np.nan), index=tickers)
    raw["volatility"] = pd.Series(np.where(ok_obs, rvol, np.nan), index=tickers)

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
    # leverage composite (v0.8): book leverage, debt-to-equity and market
    # leverage — the single liabilities/assets descriptor measured
    # significant in only ~32% of cross-sections with style bias 1.50
    tl = fund_df["total_liabilities"]
    leverage_descs = {
        "book_leverage": tl / ta,
        "debt_to_equity": (tl / be).where(be > 0),
        "market_leverage": tl / (tl + mktcap),
    }
    # sales growth (v0.8): normalized 5y revenue slope, point-in-time
    raw["growth"] = fund_df["sales_growth"]

    if industries is not None:
        ind = industries.reindex(tickers)

        def impute(s: pd.Series) -> pd.Series:
            return s.fillna(s.groupby(ind).transform("median"))

        value_descs = {k: impute(v) for k, v in value_descs.items()}
        quality_descs = {k: impute(v) for k, v in quality_descs.items()}
        leverage_descs = {k: impute(v) for k, v in leverage_descs.items()}
        raw["growth"] = impute(raw["growth"])

    raw["value"] = _composite(value_descs, mktcap, config.winsor_z, fit)
    raw["quality"] = _composite(quality_descs, mktcap, config.winsor_z, fit)
    raw["leverage"] = _composite(leverage_descs, mktcap, config.winsor_z, fit)

    exposures = pd.DataFrame(
        {name: process_exposure(raw[name], mktcap, z=config.winsor_z, fit=fit)
         for name in STYLE_FACTORS},
        index=tickers,
    )
    # whatever beta/vol correlation the time-series regression left at the
    # exposure level is projected out, so the two styles stay separate axes
    exposures["volatility"] = _orthogonalize(
        exposures["volatility"], exposures["beta"], mktcap, fit)
    return exposures, mktcap
