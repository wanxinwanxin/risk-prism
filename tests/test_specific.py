import numpy as np
import pandas as pd

from riskprism.config import STYLE_FACTORS, ModelConfig
from riskprism.model.specific import specific_risk

CFG = ModelConfig()


def _setup(n=120, t=100, seed=0):
    """Assets whose true residual vol is driven by their size exposure."""
    rng = np.random.default_rng(seed)
    idx = pd.Index([f"T{i:03d}" for i in range(n)])
    exposures = pd.DataFrame(rng.normal(0, 1, (n, len(STYLE_FACTORS))),
                             index=idx, columns=STYLE_FACTORS)
    industries = pd.Series(rng.choice(["BusEq", "Hlth", "Money"], n), index=idx)
    true_vol_weekly = 0.03 * np.exp(-0.4 * exposures["size"])  # small caps riskier
    resid = pd.DataFrame(
        rng.normal(0, 1, (t, n)) * true_vol_weekly.to_numpy(),
        columns=idx,
    )
    return exposures, industries, resid, true_vol_weekly * np.sqrt(CFG.ann_factor)


def test_structural_model_recovers_characteristic_link():
    exposures, industries, resid, true_ann = _setup()
    res = specific_risk(resid, exposures, industries, CFG)
    corr = np.corrcoef(np.log(res.structural), np.log(true_ann))[0, 1]
    assert corr > 0.8  # structural prediction tracks the true size-vol link


def test_no_history_assets_get_structural_prior():
    exposures, industries, resid, _ = _setup()
    resid.iloc[:, :10] = np.nan  # first 10 assets: zero residual history
    # blend arithmetic in isolation: shrinkage off
    cfg = ModelConfig(specific_shrink_q=0.0)
    res = specific_risk(resid, exposures, industries, cfg)
    head = res.vol.iloc[:10]
    assert head.notna().all() and (head > 0).all()
    assert (res.blend_weight.iloc[:10] == 0).all()
    assert np.allclose(head, res.structural.iloc[:10])


def test_blend_weight_grows_with_history():
    exposures, industries, resid, _ = _setup()
    resid.iloc[:-20, 0] = np.nan  # T000: 20 obs vs 100 for the rest
    res = specific_risk(resid, exposures, industries, CFG)
    assert 0 < res.blend_weight.iloc[0] < res.blend_weight.iloc[1]
    expected = 20 / (20 + CFG.structural_t0)
    assert abs(res.blend_weight.iloc[0] - expected) < 1e-9


def test_estimates_are_right_scale():
    exposures, industries, resid, true_ann = _setup()
    res = specific_risk(resid, exposures, industries, CFG)
    ratio = res.vol / true_ann
    assert ratio.median() > 0.7 and ratio.median() < 1.4
