import numpy as np
import pandas as pd

from riskprism.config import ModelConfig
from riskprism.model.benchmarks import REGRESSORS, estimate_exposures, score_etf_week
from riskprism.model.validation import FULL_FACTORS, RunningRiskState

CFG = ModelConfig(
    ann_factor=252.0, horizon_days=5, min_warmup_obs=126,
    corr_half_life=126, vol_half_life=42, specific_half_life=42,
    vra_half_life=21, nw_factor_lags=2, nw_specific_lags=1,
    min_specific_obs=21, structural_t0=63, eigen_refresh_periods=63,
    history_cap_days=400,
)


def _factor_history(t=300, seed=0):
    """Daily factor-return frame."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=t)
    sig = pd.Series(0.003, index=FULL_FACTORS)
    sig["market"] = 0.009
    fr = pd.DataFrame(rng.normal(0, 1, (t, len(FULL_FACTORS))) * sig.to_numpy(),
                      index=dates, columns=FULL_FACTORS)
    return rng, dates, fr


def test_estimate_exposures_recovers_known_betas():
    rng, dates, fr = _factor_history()
    true_b = pd.Series(0.0, index=REGRESSORS)
    true_b["market"], true_b["momentum"] = 1.0, 0.5
    ret = fr[REGRESSORS] @ true_b + rng.normal(0, 0.001, len(fr))
    est = estimate_exposures(fr, pd.Series(ret, index=dates))
    assert est is not None
    b, resid_var = est
    assert abs(b["market"] - 1.0) < 0.1
    assert abs(b["momentum"] - 0.5) < 0.2
    assert resid_var < 0.0001


def test_estimate_exposures_needs_history():
    rng, dates, fr = _factor_history(t=60)
    est = estimate_exposures(fr, pd.Series(rng.normal(0, 0.01, 60), index=dates))
    assert est is None


def test_score_etf_week_produces_calibratedish_forecast():
    rng, dates, fr = _factor_history(t=290)
    state = RunningRiskState(CFG)
    for d in dates[:-5]:
        state.update(fr.loc[d], pd.Series(dtype=float))
    t, t_next = dates[-6], dates[-1]
    # ETF = pure market tracker with small tracking error, daily
    etf_daily = pd.DataFrame(
        {"SPY": fr["market"] + rng.normal(0, 0.0005, len(fr))}, index=dates)
    week = etf_daily.loc[(etf_daily.index > t) & (etf_daily.index <= t_next), "SPY"]
    etf_weekly = pd.DataFrame({"SPY": [(1 + week).prod() - 1]}, index=[t_next])
    rows = score_etf_week(state, fr.loc[:t], etf_weekly, etf_daily, t, t_next)
    assert len(rows) == 1
    r = rows[0]
    # market daily vol ~0.9% -> annualized ~14%; forecast should be in range
    assert 0.05 < r["forecast_vol_ann"] < 0.35
    assert np.isfinite(r["z"]) and np.isfinite(r["realized_vol_ann"])
    assert r["group"] == "etf" and r["portfolio"] == "SPY"
