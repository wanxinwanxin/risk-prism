"""Derive horizon variants from shipped artifacts.

The daily regressions are horizon-agnostic: factor returns and residuals
are facts about each day. Everything horizon-specific happens on top —
EWMA half-lives, Newey-West, VRA, shrinkage — so a variant re-runs risk
construction and the full validation replay under a different config,
straight from a downloaded artifact directory: no data fetch, no new
regressions, a couple of minutes instead of a build.

`riskprism-variant --artifacts mh_dir --horizon short --out sh_dir`
"""

import argparse
from dataclasses import replace

from riskprism.artifacts import load_artifacts, save_artifacts
from riskprism.config import ModelConfig
from riskprism.model.covariance import factor_covariance
from riskprism.model.revalidate import revalidate_history
from riskprism.model.specific import specific_risk


def short_horizon_config(base: ModelConfig | None = None) -> ModelConfig:
    """Responsive variant: same daily regressions, half the memory.

    The medium-horizon model follows the USE4S template (84d vol / 252d
    corr half-lives); the short-horizon variant halves every risk-side
    half-life, the way commercial S/L model pairs differ. Exposure
    construction parameters are untouched — the two variants share
    formation exposures exactly.
    """
    base = base or ModelConfig()
    return replace(
        base,
        version=base.version.replace("-MH-", "-SH-"),
        vol_half_life=42,
        corr_half_life=126,
        specific_half_life=42,
        vra_half_life=21,
    )


HORIZONS = {"short": short_horizon_config}


def derive_variant(a: dict, config: ModelConfig) -> dict:
    """Recompute risk construction + validation replay under ``config``.

    Returns the kwargs for :func:`save_artifacts`. Exposures and the
    daily regression history pass through untouched. ETF scoring and
    realized-vol columns need price data and are skipped here — the
    derived validation grades the reconstructable panel only.
    """
    fr, res, eh = a["factor_returns"], a["residuals"], a["exposure_history"]
    am, X = a["asset_meta"], a["exposures"]
    industries = am["industry"]
    validation, state = revalidate_history(fr, res, eh, industries, config)
    F = factor_covariance(fr, config, vra=state.vra_factor)
    spec = specific_risk(res, X, industries, config, vra=state.vra_specific)
    src = a["meta"]
    asset_meta = am.copy()
    asset_meta["history_weeks"] = spec.n_obs.astype(int)
    asset_meta["specific_blend_weight"] = spec.blend_weight.round(3)
    asset_meta["specific_ts"] = spec.ts_vol.round(4)
    asset_meta["specific_structural"] = spec.structural.round(4)
    meta = {
        **{k: src.get(k) for k in ("as_of", "n_assets", "n_estimation",
                                   "n_periods", "n_weeks", "mean_r2",
                                   "price_provider")},
        "model_version": config.version,
        "derived_from": src.get("model_version"),
        "variant_note": ("derived from the medium-horizon build's daily "
                         "regressions; validation replayed without ETF "
                         "and realized-vol columns"),
        "n_validation_weeks": int(validation["date"].nunique()) if len(validation) else 0,
        "vra_factor": round(state.vra_factor, 4),
        "vra_specific": round(state.vra_specific, 4),
        "config": config.to_dict(),
    }
    return {"exposures": X, "factor_covariance": F,
            "specific_risk": spec.vol.reindex(X.index),
            "factor_returns": fr, "meta": meta, "residuals": res,
            "asset_meta": asset_meta, "validation": validation,
            "exposure_history": eh, "factor_tstats": a.get("factor_tstats")}


def main() -> None:
    p = argparse.ArgumentParser(
        prog="riskprism-variant",
        description="Derive a horizon variant from existing model artifacts.")
    p.add_argument("--artifacts", default="artifacts",
                   help="Source (medium-horizon) artifact directory")
    p.add_argument("--horizon", choices=sorted(HORIZONS), default="short")
    p.add_argument("--out", required=True, help="Output artifact directory")
    args = p.parse_args()
    a = load_artifacts(args.artifacts)
    config = HORIZONS[args.horizon](
        ModelConfig(**{**ModelConfig().to_dict(),
                       **{k: v for k, v in (a["meta"].get("config") or {}).items()
                          if k in ModelConfig.__dataclass_fields__}}))
    pieces = derive_variant(a, config)
    path = save_artifacts(args.out, **pieces)
    m = pieces["meta"]
    print(f"[riskprism] {m['model_version']} derived from "
          f"{m['derived_from']} → {path} "
          f"({m['n_validation_weeks']} validation weeks replayed)")


if __name__ == "__main__":
    main()
