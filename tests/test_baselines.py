import numpy as np
import pandas as pd

from riskprism.model.baselines import (
    axioma_protocol, baseline_forecasts, comparison_payload,
    harness_comparison, use4_protocol,
)

_ANN = 52.0


def _panel(n_weeks=140, vol_w=0.02, seed=0, portfolios=("market", "SPY")):
    """Synthetic validation history: iid weekly returns at a known vol."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-06", periods=n_weeks, freq="W-FRI")
    rows = []
    for name in portfolios:
        r = rng.normal(0, vol_w, n_weeks)
        # within-week RV is a noisy draw around the true weekly variance
        rv_w = vol_w * np.sqrt(rng.chisquare(5, n_weeks) / 5)
        for d, ret, rv in zip(dates, r, rv_w):
            fv = vol_w * np.sqrt(_ANN)
            rows.append({"date": d, "portfolio": name, "group": "market",
                         "forecast_vol_ann": fv, "realized_ret": ret,
                         "z": ret / vol_w, "realized_vol_ann": rv * np.sqrt(_ANN)})
    return pd.DataFrame(rows)


def _ff_weekly(n_weeks=400, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-04", periods=n_weeks, freq="W-FRI")
    cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    return pd.DataFrame(rng.normal(0, 0.015, (n_weeks, len(cols))),
                        index=dates, columns=cols)


def test_baseline_forecasts_are_point_in_time():
    val = _panel()
    fc = baseline_forecasts(val, ff_weekly=_ff_weekly())
    g = fc[fc["portfolio"] == "market"].sort_values("date")
    # nothing before the burn-in, forecasts after it
    for col in ("fc_ewma", "fc_trail", "fc_rv_ewma", "fc_ff5"):
        assert g[col].iloc[:26].isna().all(), col
        assert g[col].iloc[60:].notna().all(), col
    # with iid returns at 2% weekly, every baseline should land near the
    # true annualized vol
    true_ann = 0.02 * np.sqrt(_ANN)
    for col in ("fc_ewma", "fc_trail", "fc_rv_ewma"):
        assert abs(g[col].iloc[-1] - true_ann) / true_ann < 0.35, col


def test_harness_comparison_shares_one_sample():
    val = _panel()
    out = harness_comparison(val, ff_weekly=_ff_weekly(), min_obs=30)
    assert out is not None
    keys = [m["key"] for m in out["models"]]
    assert keys[0] == "riskprism" and "ff5" in keys
    # perfect forecasts on calibrated data: bias near 1 for the model
    model = out["models"][0]
    assert 0.8 < model["bias"] < 1.2
    assert out["n_obs"] <= len(val)
    for m in out["models"]:
        for stat in ("bias", "exc", "vratio", "mz_slope", "qlike", "rmse"):
            assert np.isfinite(m[stat]), (m["key"], stat)


def test_harness_comparison_without_ff():
    out = harness_comparison(_panel(), ff_weekly=None, min_obs=30)
    assert "ff5" not in [m["key"] for m in out["models"]]


def test_use4_protocol_calibrated_near_one():
    out = use4_protocol(_panel(n_weeks=200))
    assert out is not None
    assert out["n_windows"] > 10
    # perfect forecasts: rolling monthly bias stats near 1, MRAD near its
    # sampling floor (~0.17 for normal returns, 12-month windows)
    assert 0.75 < out["mean_bias"] < 1.25
    assert out["mrad"] < 0.35


def test_use4_protocol_needs_history():
    assert use4_protocol(_panel(n_weeks=40)) is None


def test_axioma_protocol_bands():
    rows = axioma_protocol(_panel())
    names = {r["name"] for r in rows}
    assert any("SPY" in n for n in names)
    for r in rows:
        assert r["lo"] < 1 < r["hi"]
        half = 1.96 / np.sqrt(2 * r["n"])
        assert abs((r["hi"] - r["lo"]) / 2 - half) < 1e-3


def test_comparison_payload_shape():
    p = comparison_payload(_panel(n_weeks=200), ff_weekly=_ff_weekly(), fetch_ff=False)
    assert set(p) == {"harness", "vendor"}
    assert "msci" in p["vendor"] and "axioma" in p["vendor"]
    assert comparison_payload(None) is None
    assert comparison_payload(pd.DataFrame()) is None
