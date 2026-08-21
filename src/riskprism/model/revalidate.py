"""Full-history validation replay: validation as a pure function of the
shipped artifacts.

Rather than accumulating scores across builds (which would leave old
weeks graded by old model versions), every build replays a fresh
point-in-time risk state over the entire stored DAILY factor-return
history and rescores every week under the CURRENT methodology.

Reconstruction is exact for regressed names: each daily regression
defines ``r_d,i = X_i·f_d + eps_d,i`` identically (X frozen at the week's
formation date), so the true weekly return is recovered by compounding
``prod(1 + X_i·f_d + eps_d,i) - 1`` over the week's trading days —
including imputed delisting returns for names that have since
disappeared. Realized-vol columns use daily asset returns where fetchable
(currently-alive names); ETF scoring replays completely since ETF prices
remain available.
"""

from collections import deque

import numpy as np
import pandas as pd

from riskprism.config import MARKET_FACTOR, ModelConfig
from riskprism.factors.industry import industry_dummies
from riskprism.model.benchmarks import score_etf_week
from riskprism.model.validation import (FULL_FACTORS, RunningRiskState,
                                        score_portfolios)

_ETF_TRAILING = 300  # daily factor-return rows kept for ETF exposure RBSA


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
    """Replay the whole daily history through a fresh risk state, scoring
    every completed formation week. Returns (validation rows, final
    warmed state) — the final state carries the VRA multipliers for the
    shipped batch model."""
    state = RunningRiskState(config)
    rows: list[dict] = []
    trailing: deque = deque(maxlen=_ETF_TRAILING)
    daily_dates = list(factor_returns.index)

    # ---- weekly scoring jobs, precomputed from stored data --------------
    have_eh = exposure_history is not None and len(exposure_history)
    jobs: list[dict] = []
    if have_eh and daily_dates:
        eh = exposure_history.copy()
        eh["date"] = pd.to_datetime(eh["date"])
        eh_by_date = {d: g for d, g in eh.groupby("date")}
        eh_dates = sorted(eh_by_date)
        didx = factor_returns.index
        for i, t in enumerate(eh_dates):
            t_next = eh_dates[i + 1] if i + 1 < len(eh_dates) else None
            if t_next is None:
                break
            days = didx[(didx > t) & (didx <= t_next)]
            if not len(days):
                continue
            snap = eh_by_date[t].set_index("ticker")
            style_cols = [c for c in snap.columns if c not in ("date", "mktcap")]
            ind = industries.reindex(snap.index).fillna("Other")
            X = pd.concat(
                [pd.Series(1.0, index=snap.index, name=MARKET_FACTOR),
                 snap[style_cols].astype(float),
                 industry_dummies(ind)],
                axis=1,
            ).reindex(columns=FULL_FACTORS).fillna(0.0)
            # exact weekly reconstruction: prod(1 + X f_d + eps_d) - 1
            Xn = X.to_numpy()
            growth = np.ones(len(snap))
            for d in days:
                fv = factor_returns.loc[d].reindex(FULL_FACTORS).fillna(0.0).to_numpy()
                eps = (residuals.loc[d].reindex(snap.index).to_numpy()
                       if d in residuals.index else np.full(len(snap), np.nan))
                growth = growth * (1.0 + Xn @ fv + eps)
            y = pd.Series(growth - 1.0, index=snap.index)
            jobs.append({
                "t": t, "t_next": t_next,
                "trigger": days[0],  # first daily date after t: state is PIT at t
                "X": X, "ind": ind,
                "mktcap": snap["mktcap"].astype(float),
                "y": y, "week_index": i,
            })

    # ---- daily replay with weekly scoring at formation boundaries -------
    ptr = 0
    for d in daily_dates:
        while ptr < len(jobs) and jobs[ptr]["trigger"] == d:
            j = jobs[ptr]
            if state.ready:
                dr = None
                if daily_returns is not None:
                    dr = daily_returns[(daily_returns.index > j["t"])
                                       & (daily_returns.index <= j["t_next"])]
                rows.extend(score_portfolios(
                    state, j["X"], j["ind"], j["mktcap"], j["y"],
                    j["t_next"], j["week_index"], daily_returns=dr))
                if etf_weekly is not None and len(trailing) >= 126:
                    rows.extend(score_etf_week(
                        state, pd.DataFrame(dict(trailing)).T,
                        etf_weekly, etf_daily, j["t"], j["t_next"]))
            ptr += 1
        f = factor_returns.loc[d]
        res_row = (residuals.loc[d] if residuals is not None
                   and d in residuals.index else pd.Series(dtype=float))
        state.update(f, res_row.dropna() if len(res_row) else pd.Series(dtype=float))
        trailing.append((d, f))

    validation = pd.DataFrame(rows)
    if len(validation):
        validation = validation.sort_values(["date", "portfolio"]).reset_index(drop=True)
    return validation, state
