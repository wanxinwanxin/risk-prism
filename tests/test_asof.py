import numpy as np
import pandas as pd
import pytest

from riskprism.config import STYLE_FACTORS, ModelConfig
from riskprism.model.asof import available_dates, model_asof

CFG = ModelConfig()


def _artifacts(seed=0, t=60, n=40):
    rng = np.random.default_rng(seed)
    idx = pd.Index([f"T{i:02d}" for i in range(n)])
    dates = pd.date_range("2024-01-05", periods=t, freq="W-FRI")
    from riskprism.model.validation import FULL_FACTORS
    fr = pd.DataFrame(rng.normal(0, 0.01, (t, len(FULL_FACTORS))),
                      index=dates, columns=FULL_FACTORS)
    res = pd.DataFrame(rng.normal(0, 0.03, (t, n)), index=dates, columns=idx)
    rows = []
    for d in dates:
        df = pd.DataFrame(rng.normal(0, 1, (n, len(STYLE_FACTORS))),
                          index=idx, columns=STYLE_FACTORS)
        df.insert(0, "date", d)
        df.insert(1, "mktcap", rng.lognormal(10, 1, n))
        rows.append(df.reset_index(names="ticker"))
    eh = pd.concat(rows, ignore_index=True)
    am = pd.DataFrame({"industry": rng.choice(["BusEq", "Hlth"], n)}, index=idx)
    return {"factor_returns": fr, "residuals": res, "exposure_history": eh,
            "asset_meta": am, "meta": {"model_version": CFG.version}}


def test_model_asof_reconstructs_and_prices():
    a = _artifacts()
    dates = available_dates(a)
    assert len(dates) == 60
    m = model_asof(a, dates[40], config=CFG)
    assert m.meta["historical"] and m.meta["as_of"] == str(dates[40].date())
    r = m.portfolio_risk({"T00": 0.5, "T01": 0.5})
    assert 0 < r["total_vol"] < 5
    assert r["factor_vol"] ** 2 + r["specific_vol"] ** 2 == pytest.approx(r["total_vol"] ** 2)
    vals = np.linalg.eigvalsh(m.factor_covariance.to_numpy())
    assert vals.min() >= -1e-12


def test_no_lookahead_in_replay():
    a = _artifacts(seed=1)
    dates = available_dates(a)
    cut = dates[30]
    # inject a catastrophic factor shock AFTER the reconstruction date
    a["factor_returns"].loc[dates[45], "market"] = 5.0
    m_before = model_asof(_artifacts(seed=1), cut, config=CFG)
    m_after_shock_added = model_asof(a, cut, config=CFG)
    v1 = m_before.portfolio_risk({"T00": 1.0})["total_vol"]
    v2 = m_after_shock_added.portfolio_risk({"T00": 1.0})["total_vol"]
    assert v1 == pytest.approx(v2)  # the future must be invisible


def test_bad_date_raises():
    a = _artifacts()
    with pytest.raises(ValueError, match="formation date"):
        model_asof(a, "2019-01-01", config=CFG)
