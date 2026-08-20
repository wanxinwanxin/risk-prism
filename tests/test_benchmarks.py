import numpy as np
import pandas as pd

from riskprism.config import ModelConfig
from riskprism.model.benchmarks import REGRESSORS, estimate_exposures, score_etf_week
from riskprism.model.validation import FULL_FACTORS, RunningRiskState

CFG = ModelConfig()


def _factor_history(t=60, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-05", periods=t, freq="W-FRI")
    sig = pd.Series(0.006, index=FULL_FACTORS)
    sig["market"] = 0.02
    fr = pd.DataFrame(rng.normal(0, 1, (t, len(FULL_FACTORS))) * sig.to_numpy(),
                      index=dates, columns=FULL_FACTORS)
    return rng, dates, fr


def test_estimate_exposures_recovers_known_betas():
    rng, dates, fr = _factor_history()
    true_b = pd.Series(0.0, index=REGRESSORS)
    true_b["market"], true_b["momentum"] = 1.0, 0.5
    ret = fr[REGRESSORS] @ true_b + rng.normal(0, 0.002, len(fr))
    est = estimate_exposures(fr, pd.Series(ret, index=dates))
    assert est is not None
    b, resid_var = est
    assert abs(b["market"] - 1.0) < 0.1
    assert abs(b["momentum"] - 0.5) < 0.2
    assert resid_var < 0.001


def test_estimate_exposures_needs_history():
    rng, dates, fr = _factor_history(t=20)
    est = estimate_exposures(fr, pd.Series(rng.normal(0, 0.02, 20), index=dates))
    assert est is None


def test_score_etf_week_produces_calibratedish_forecast():
    rng, dates, fr = _factor_history(t=59)
    state = RunningRiskState(CFG)
    for d in dates[:-1]:
        state.update(fr.loc[d], pd.Series(dtype=float))
    t, t_next = dates[-2], dates[-1]
    # ETF = pure market tracker with small tracking error
    etf_weekly = pd.DataFrame(
        {"SPY": fr["market"] + rng.normal(0, 0.001, len(fr))}, index=dates)
    days = pd.date_range(t + pd.Timedelta(days=1), periods=5, freq="B")
    etf_daily = pd.DataFrame({"SPY": rng.normal(0, 0.009, 5)}, index=days)
    rows = score_etf_week(state, fr.loc[:t], etf_weekly, etf_daily, t, t_next)
    assert len(rows) == 1
    r = rows[0]
    # market weekly vol ~2% -> annualized ~14%; forecast should be in range
    assert 0.05 < r["forecast_vol_ann"] < 0.35
    assert np.isfinite(r["z"]) and np.isfinite(r["realized_vol_ann"])
    assert r["group"] == "etf" and r["portfolio"] == "SPY"
