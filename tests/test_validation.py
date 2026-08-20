import numpy as np
import pandas as pd

from riskprism.config import STYLE_FACTORS, ModelConfig
from riskprism.model.validation import (
    FULL_FACTORS, RunningRiskState, merge_validation, score_portfolios,
    validation_summary,
)

CFG = ModelConfig()


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


def test_no_scores_before_warmup():
    rng, idx, styles, industries, caps, x_full, sig = _world(seed=1)
    state = RunningRiskState(CFG)
    y = pd.Series(rng.normal(0, 0.03, len(idx)), index=idx)
    assert score_portfolios(state, x_full, industries, caps, y,
                            pd.Timestamp("2020-01-10"), 0) == []


def test_merge_validation_dedups_and_caps():
    d1 = pd.date_range("2024-01-05", periods=10, freq="W-FRI")
    prior = pd.DataFrame({"date": d1.repeat(2), "portfolio": ["market", "equal_weight"] * 10,
                          "group": "x", "z": 1.0})
    new = pd.DataFrame({"date": [d1[-1]], "portfolio": ["market"], "group": ["x"], "z": [9.0]})
    merged = merge_validation(prior, new, cap_weeks=100)
    last = merged[merged["date"] == d1[-1]]
    assert len(last) == 1 and last["z"].iloc[0] == 9.0  # new week supersedes entirely
    assert merge_validation(prior, new, cap_weeks=3)["date"].nunique() == 3
