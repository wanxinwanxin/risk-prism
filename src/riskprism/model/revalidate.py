"""Full-history validation replay: validation as a pure function of the
shipped artifacts.

Rather than accumulating scores across builds (which would leave old
weeks graded by old model versions), every build replays a fresh
point-in-time risk state over the entire merged factor-return history
and rescores every week under the CURRENT methodology. Reconstruction is
exact for regressed names: the weekly regression defines
``r_i = X_i·f + eps_i`` identically, and the artifacts store X (exposure
history), f (factor returns) and eps (residuals) — so replayed realized
returns equal the returns the regression actually saw, including imputed
delisting returns for names that have since disappeared.

Realized-vol columns use daily returns where fetchable (currently-alive
names); ETF scoring replays completely since ETF prices remain available.
"""

import bisect
from collections import deque

import numpy as np
import pandas as pd

from riskprism.config import MARKET_FACTOR, ModelConfig
from riskprism.factors.industry import industry_dummies
from riskprism.model.benchmarks import score_etf_week
from riskprism.model.validation import (FULL_FACTORS, RunningRiskState,
                                        score_portfolios)

_ETF_MIN_TRAILING = 40  # estimate_exposures needs >= _MIN_WEEKS anyway


def revalidate_history(
    factor_returns: pd.DataFrame,
    residuals: pd.DataFrame,
    exposure_history: pd.DataFrame,
    industries: pd.Series,
    config: ModelConfig,
    daily_returns: pd.DataFrame | None = None,
    etf_weekly: pd.DataFrame | None = None,
    etf_daily: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, RunningRiskState]:
    """Replay the whole history through a fresh risk state, scoring every
    week. Returns (validation rows, final warmed state) — the final state
    carries the VRA multipliers for the shipped batch model."""
    state = RunningRiskState(config)
    rows: list[dict] = []
    trailing: deque = deque(maxlen=52)

    have_eh = exposure_history is not None and len(exposure_history)
    if have_eh:
        eh = exposure_history.copy()
        eh["date"] = pd.to_datetime(eh["date"])
        eh_by_date = {d: g for d, g in eh.groupby("date")}
        eh_dates = sorted(eh_by_date)
    else:
        eh_by_date, eh_dates = {}, []

    for i, d_next in enumerate(factor_returns.index):
        f = factor_returns.loc[d_next]
        res_row = (residuals.loc[d_next] if residuals is not None
                   and d_next in residuals.index else pd.Series(dtype=float))
        # formation date = latest exposure snapshot strictly before d_next
        j = bisect.bisect_left(eh_dates, d_next) - 1
        t = eh_dates[j] if j >= 0 else None
        if t is not None and state.ready:
            snap = eh_by_date[t].set_index("ticker")
            style_cols = [c for c in snap.columns if c not in ("date", "mktcap")]
            ind = industries.reindex(snap.index).fillna("Other")
            X = pd.concat(
                [pd.Series(1.0, index=snap.index, name=MARKET_FACTOR),
                 snap[style_cols].astype(float),
                 industry_dummies(ind)],
                axis=1,
            ).reindex(columns=FULL_FACTORS).fillna(0.0)
            fv = f.reindex(FULL_FACTORS).fillna(0.0).to_numpy()
            y = pd.Series(X.to_numpy() @ fv, index=snap.index) \
                + res_row.reindex(snap.index)  # NaN eps -> NaN -> filtered
            mktcap = snap["mktcap"].astype(float)
            dr = None
            if daily_returns is not None:
                dr = daily_returns[(daily_returns.index > t)
                                   & (daily_returns.index <= d_next)]
            rows.extend(score_portfolios(
                state, X, ind, mktcap, y, d_next, i, daily_returns=dr))
            if etf_weekly is not None and len(trailing) >= _ETF_MIN_TRAILING:
                rows.extend(score_etf_week(
                    state, pd.DataFrame(dict(trailing)).T,
                    etf_weekly, etf_daily, t, d_next))
        state.update(f, res_row.dropna() if len(res_row) else pd.Series(dtype=float))
        trailing.append((d_next, f))

    validation = pd.DataFrame(rows)
    if len(validation):
        sort_cols = ["date", "portfolio"]
        validation = validation.sort_values(sort_cols).reset_index(drop=True)
    return validation, state
