import numpy as np
import pandas as pd
import pytest

from riskprism.config import STYLE_FACTORS, ModelConfig
from riskprism.model.validation import (
    FULL_FACTORS, RunningRiskState, merge_validation, score_portfolios,
    validation_summary,
)

CFG = ModelConfig(
    ann_factor=52.0, horizon_days=1, min_warmup_obs=26,
    corr_half_life=26, vol_half_life=13, specific_half_life=13,
    vra_half_life=8, nw_factor_lags=2, nw_specific_lags=1,
    min_specific_obs=13, structural_t0=26, eigen_refresh_periods=26,
    history_cap_days=156,
)


def _world(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.Index([f"T{i:03d}" for i in range(n)])
    styles = pd.DataFrame(rng.normal(0, 1, (n, len(STYLE_FACTORS))),
                          index=idx, columns=STYLE_FACTORS)
    industries = pd.Series(rng.choice(["BusEq", "Hlth", "Money", "Enrgy"], n), index=idx)
    caps = pd.Series(rng.lognormal(10, 1.5, n), index=idx)
    dummies = pd.get_dummies(industries, prefix="ind", dtype=float)
    x_full = pd.concat([pd.Series(1.0, index=idx, name="market"), styles, dummies], axis=1)
    x_full = x_full.reindex(columns=FULL_FACTORS).fillna(0.0)
    # true weekly factor vols: market 2%, styles 60bp, industries 80bp
    sig = pd.Series(0.006, index=FULL_FACTORS)
    sig["market"] = 0.02
    sig[[f for f in FULL_FACTORS if f.startswith("ind_")]] = 0.008
    return rng, idx, styles, industries, caps, x_full, sig


def test_calibrated_world_gives_bias_stat_near_one():
    rng, idx, styles, industries, caps, x_full, sig = _world()
    state = RunningRiskState(CFG)
    rows = []
    sig_e = 0.04  # weekly specific vol
    dates = pd.date_range("2020-01-03", periods=250, freq="W-FRI")
    for wk, date in enumerate(dates):
        f = pd.Series(rng.normal(0, sig.to_numpy()), index=FULL_FACTORS)
        eps = pd.Series(rng.normal(0, sig_e, len(idx)), index=idx)
        y = pd.Series(x_full.to_numpy() @ f.to_numpy(), index=idx) + eps
        rows.extend(score_portfolios(state, x_full, industries, caps, y, date, wk))
        state.update(f, eps)
    summary = validation_summary(pd.DataFrame(rows))
    assert len(summary) >= 10
    # a correctly specified model should be calibrated across every portfolio
    assert summary["bias_stat"].between(0.7, 1.35).all(), summary.to_string()
    med = summary["bias_stat"].median()
    assert 0.85 < med < 1.15


def test_realized_vol_recovers_true_vol():
    from riskprism.model.validation import _realized_vol_ann
    rng = np.random.default_rng(3)
    idx = pd.Index([f"T{i}" for i in range(50)])
    w = pd.Series(1.0 / 50, index=idx)
    sig_d = 0.02
    rvs = []
    for _ in range(300):
        daily = pd.DataFrame(rng.normal(0, sig_d, (5, 50)), columns=idx)
        rvs.append(_realized_vol_ann(daily, w))
    true_ann = sig_d / np.sqrt(50) * np.sqrt(5 * 52)  # eq-weight diversification
    assert np.sqrt(np.mean(np.array(rvs) ** 2)) == pytest.approx(true_ann, rel=0.08)


def test_scores_carry_realized_vol_only_with_daily_data():
    rng, idx, styles, industries, caps, x_full, sig = _world(seed=4)
    state = RunningRiskState(CFG)
    for wk in range(30):
        f = pd.Series(rng.normal(0, sig.to_numpy()), index=FULL_FACTORS)
        state.update(f, pd.Series(rng.normal(0, 0.03, len(idx)), index=idx))
    y = pd.Series(rng.normal(0, 0.03, len(idx)), index=idx)
    daily = pd.DataFrame(rng.normal(0, 0.012, (5, len(idx))), columns=idx)
    with_rv = score_portfolios(state, x_full, industries, caps, y,
                               pd.Timestamp("2021-01-08"), 30, daily_returns=daily)
    without = score_portfolios(state, x_full, industries, caps, y,
                               pd.Timestamp("2021-01-08"), 30)
    assert all(np.isfinite(r["realized_vol_ann"]) for r in with_rv)
    assert all(np.isnan(r["realized_vol_ann"]) for r in without)


def test_no_scores_before_warmup():
    rng, idx, styles, industries, caps, x_full, sig = _world(seed=1)
    state = RunningRiskState(CFG)
    y = pd.Series(rng.normal(0, 0.03, len(idx)), index=idx)
    assert score_portfolios(state, x_full, industries, caps, y,
                            pd.Timestamp("2020-01-10"), 0) == []


def test_merge_validation_handles_frames_without_portfolio_column():
    d = pd.date_range("2024-01-05", periods=4, freq="W-FRI")
    prior = pd.DataFrame({"date": d.repeat(2), "ticker": ["A", "B"] * 4, "size": 1.0})
    new = pd.DataFrame({"date": [d[-1]], "ticker": ["A"], "size": [2.0]})
    merged = merge_validation(prior, new, cap_weeks=100)
    assert merged[merged["date"] == d[-1]]["size"].tolist() == [2.0]


def test_merge_validation_dedups_and_caps():
    d1 = pd.date_range("2024-01-05", periods=10, freq="W-FRI")
    prior = pd.DataFrame({"date": d1.repeat(2), "portfolio": ["market", "equal_weight"] * 10,
                          "group": "x", "z": 1.0})
    new = pd.DataFrame({"date": [d1[-1]], "portfolio": ["market"], "group": ["x"], "z": [9.0]})
    merged = merge_validation(prior, new, cap_weeks=100)
    last = merged[merged["date"] == d1[-1]]
    assert len(last) == 1 and last["z"].iloc[0] == 9.0  # new week supersedes entirely
    assert merge_validation(prior, new, cap_weeks=3)["date"].nunique() == 3
