import numpy as np
import pandas as pd

from osrisk.factors.transforms import process_exposure, standardize, winsorize_z


def test_winsorize_clips_outliers():
    s = pd.Series([0.0] * 50 + [1000.0])
    out = winsorize_z(s, z=3.0)
    assert out.max() < 1000.0


def test_standardize_cap_weighted_mean_zero_unit_std():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(5, 2, 500))
    cap = pd.Series(rng.lognormal(10, 1, 500))
    out = standardize(s, cap)
    w = cap / cap.sum()
    assert abs((out * w).sum()) < 1e-10
    assert abs(out.std() - 1.0) < 1e-10


def test_process_exposure_fills_missing_with_zero():
    s = pd.Series([1.0, 2.0, np.nan, 4.0])
    cap = pd.Series([1.0, 1.0, 1.0, 1.0])
    out = process_exposure(s, cap)
    assert out.iloc[2] == 0.0
    assert out.notna().all()
