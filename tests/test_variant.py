"""Short-horizon variant derived from artifacts, no rebuild."""

import numpy as np
import pandas as pd
import pytest

from riskprism.config import STYLE_FACTORS, ModelConfig
from riskprism.model.validation import FULL_FACTORS
from riskprism.model.variant import derive_variant, short_horizon_config

CFG = ModelConfig()


def _artifacts(seed=0, t=220, n=150):
    # replay scoring needs the full panel machinery: >=126 rows of warm-up
    # (default min_warmup_obs) and enough names for 50-name random baskets
    rng = np.random.default_rng(seed)
    idx = pd.Index([f"T{i:03d}" for i in range(n)])
    dates = pd.date_range("2022-01-07", periods=t, freq="W-FRI")
    fr = pd.DataFrame(rng.normal(0, 0.01, (t, len(FULL_FACTORS))),
                      index=dates, columns=FULL_FACTORS)
    res = pd.DataFrame(rng.normal(0, 0.03, (t, n)), index=dates, columns=idx)
    rows = []
    for d in dates:
        df = pd.DataFrame(rng.normal(0, 1, (n, len(STYLE_FACTORS))),
                          index=idx, columns=STYLE_FACTORS)
        df.insert(0, "date", d - pd.Timedelta(days=7))
        df.insert(1, "mktcap", rng.lognormal(20, 1, n))
        rows.append(df.reset_index(names="ticker"))
    eh = pd.concat(rows, ignore_index=True)
    am = pd.DataFrame({"industry": rng.choice(["BusEq", "Hlth"], n),
                       "in_estimation": True}, index=idx)
    X = pd.DataFrame(rng.normal(0, 1, (n, len(FULL_FACTORS))),
                     index=idx, columns=FULL_FACTORS)
    X["market"] = 1.0
    return {"factor_returns": fr, "residuals": res, "exposure_history": eh,
            "asset_meta": am, "exposures": X,
            "meta": {"model_version": CFG.version, "as_of": "2026-01-02",
                     "n_assets": n, "config": CFG.to_dict()}}


def test_short_config_halves_risk_half_lives_only():
    sh = short_horizon_config()
    assert sh.version == CFG.version.replace("-MH-", "-SH-")
    assert sh.vol_half_life == 42 and sh.corr_half_life == 126
    assert sh.specific_half_life == 42 and sh.vra_half_life == 21
    # exposure construction unchanged — the variants share formation exposures
    assert sh.volatility_window_days == CFG.volatility_window_days
    assert sh.momentum_window_days == CFG.momentum_window_days


def test_derive_variant_recomputes_risk_not_exposures():
    a = _artifacts()
    out = derive_variant(a, short_horizon_config())
    assert out["meta"]["model_version"].startswith("PRISM-US-SH-")
    assert out["meta"]["derived_from"] == CFG.version
    # exposures pass through identically
    pd.testing.assert_frame_equal(out["exposures"], a["exposures"])
    F = out["factor_covariance"]
    vals = np.linalg.eigvalsh(F.to_numpy())
    assert vals.min() >= -1e-12
    assert (out["specific_risk"] > 0).all()
    assert out["meta"]["n_validation_weeks"] > 0


def test_short_horizon_reacts_faster():
    """Late-history shocks weigh more under the faster half-life."""
    a = _artifacts()
    # a volatility burst in the last 10 rows of the market factor
    a["factor_returns"].iloc[-10:, 0] *= 6
    from riskprism.model.covariance import factor_covariance
    F_mh = factor_covariance(a["factor_returns"], CFG)
    F_sh = factor_covariance(a["factor_returns"], short_horizon_config())
    assert F_sh.iloc[0, 0] > F_mh.iloc[0, 0]


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
