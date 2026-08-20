"""External benchmark portfolios for validation: real, traded factor ETFs.

The model's own test portfolios share its construction assumptions; these
don't. Each ETF's factor exposures are estimated point-in-time by
regressing its trailing weekly returns on the model's factor returns
(returns-based style analysis — holdings aren't freely available
point-in-time), then its volatility is forecast from the same EWMA state
and scored against what the ETF actually did.
"""

import numpy as np
import pandas as pd

from riskprism.config import MARKET_FACTOR, STYLE_FACTORS

ETF_BENCHMARKS = {
    "SPY": "S&P 500",
    "IWM": "Russell 2000",
    "MTUM": "MSCI USA Momentum",
    "VLUE": "MSCI USA Value",
    "QUAL": "MSCI USA Quality",
    "USMV": "MSCI USA Min Vol",
}

# 52 weekly observations can't support 21 regressors; market + styles
# capture what diversified factor ETFs are — industries fold into residual.
REGRESSORS = [MARKET_FACTOR, *STYLE_FACTORS]
_MIN_WEEKS = 40
_WINDOW = 52


def estimate_exposures(trailing_fr: pd.DataFrame, trailing_ret: pd.Series
                       ) -> tuple[pd.Series, float] | None:
    """Regress trailing ETF returns on factor returns.

    Returns (exposures over REGRESSORS, weekly residual variance), or None
    with insufficient history.
    """
    df = pd.concat([trailing_ret.rename("_r"), trailing_fr[REGRESSORS]], axis=1).dropna()
    df = df.tail(_WINDOW)
    if len(df) < _MIN_WEEKS:
        return None
    y = df["_r"].to_numpy()
    X = np.column_stack([np.ones(len(df)), df[REGRESSORS].to_numpy()])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(df) - X.shape[1], 1)
    return pd.Series(beta[1:], index=REGRESSORS), float((resid ** 2).sum() / dof)


def score_etf_week(
    state,                      # RunningRiskState through week t
    trailing_fr: pd.DataFrame,  # factor returns with index <= t
    etf_weekly: pd.DataFrame,   # weekly ETF return panel
    etf_daily: pd.DataFrame,    # daily ETF return panel
    t: pd.Timestamp,
    t_next: pd.Timestamp,
) -> list[dict]:
    from riskprism.model.validation import FULL_FACTORS, _realized_vol_ann

    if not state.ready or trailing_fr.empty:
        return []
    F = state.factor_cov_weekly()
    pos = [FULL_FACTORS.index(f) for f in REGRESSORS]
    F_sub = F[np.ix_(pos, pos)]
    rows = []
    for ticker in etf_weekly.columns:
        if t_next not in etf_weekly.index:
            continue
        realized = etf_weekly.at[t_next, ticker]
        if not np.isfinite(realized):
            continue
        est = estimate_exposures(trailing_fr, etf_weekly[ticker].loc[:t])
        if est is None:
            continue
        b, resid_var = est
        var_w = float(b.to_numpy() @ F_sub @ b.to_numpy()) + resid_var
        if var_w <= 0:
            continue
        vol_w = np.sqrt(var_w)
        daily_slice = etf_daily[(etf_daily.index > t) & (etf_daily.index <= t_next)][[ticker]]
        rows.append({
            "date": t_next, "portfolio": ticker, "group": "etf",
            "forecast_vol_ann": vol_w * np.sqrt(52.0),
            "realized_ret": float(realized),
            "z": float(realized) / vol_w,
            "realized_vol_ann": _realized_vol_ann(
                daily_slice, pd.Series(1.0, index=[ticker])),
        })
    return rows
