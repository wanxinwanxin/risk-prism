import numpy as np
import pandas as pd

from riskprism.config import ModelConfig
from riskprism.factors.transforms import process_exposure
from riskprism.model.history import delisting_return, merge_history

CFG = ModelConfig()


def test_delisting_return_heuristic():
    assert delisting_return(2.50, CFG) == CFG.delist_failure_return
    assert delisting_return(45.0, CFG) == 0.0


def test_merge_history_appends_and_caps():
    dates1 = pd.date_range("2024-01-05", periods=10, freq="W-FRI")
    dates2 = pd.date_range(dates1[-1], periods=4, freq="W-FRI")  # overlaps last row
    prior = pd.DataFrame({"a": 1.0, "b": 2.0}, index=dates1)
    new = pd.DataFrame({"a": 9.0, "c": 3.0}, index=dates2)
    merged = merge_history(prior, new, cap_weeks=200)
    assert len(merged) == 13  # 10 + 4 - 1 overlap
    assert merged.loc[dates1[-1], "a"] == 9.0  # new supersedes prior on overlap
    assert set(merged.columns) == {"a", "b", "c"}  # columns unioned
    assert merged.loc[dates1[0], "a"] == 1.0  # old history intact


def test_merge_history_trims_to_cap_and_handles_no_prior():
    dates = pd.date_range("2024-01-05", periods=10, freq="W-FRI")
    new = pd.DataFrame({"a": np.arange(10.0)}, index=dates)
    assert len(merge_history(None, new, cap_weeks=6)) == 6
    assert merge_history(None, new, cap_weeks=6).iloc[-1, 0] == 9.0


def test_exposure_fit_subset_sets_scale():
    # stats fit on the estimation subset; wild outsiders can't distort them
    rng = np.random.default_rng(0)
    idx = pd.Index([f"T{i}" for i in range(120)])
    s = pd.Series(rng.normal(0, 1, 120), index=idx)
    s.iloc[100:] = 50.0  # coverage-only names with absurd raw values
    cap = pd.Series(rng.lognormal(10, 1, 120), index=idx)
    fit = idx[:100]
    out = process_exposure(s, cap, z=3.0, fit=fit)
    w = cap[fit] / cap[fit].sum()
    assert abs((out[fit] * w).sum()) < 1e-9      # cap-weighted mean 0 on fit set
    assert out.iloc[100:].max() < 5.0            # outsiders clipped to fit bounds
