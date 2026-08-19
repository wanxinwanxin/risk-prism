"""Export model artifacts as the static explorer site.

Reads an artifact directory, compacts it to JSON, and injects it into
``site/template.html`` to produce a fully self-contained ``index.html`` —
no server, no external requests. The weekly GitHub Action publishes the
result to GitHub Pages; the same file can be opened locally.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from riskprism.artifacts import load_artifacts
from riskprism.config import MARKET_FACTOR, STYLE_FACTORS
from riskprism.factors.industry import INDUSTRY_PREFIX

PLACEHOLDER = "__RISKPRISM_DATA__"


def _round(values, nd=5):
    return [round(float(v), nd) for v in values]


def build_site_data(artifacts_dir: str | Path) -> dict:
    a = load_artifacts(artifacts_dir)
    X = a["exposures"]
    F = a["factor_covariance"]
    spec = a["specific_risk"].reindex(X.index).fillna(0.0)
    freturns = a["factor_returns"]

    factors = list(F.columns)
    ind_cols = [f for f in factors if f.startswith(INDUSTRY_PREFIX)]

    # Per-asset industry name from the one-hot columns
    ind_block = X[ind_cols]
    industry = ind_block.idxmax(axis=1).str.replace(INDUSTRY_PREFIX, "", regex=False)
    industry[ind_block.sum(axis=1) == 0] = "Other"

    # Per-asset total vol
    Xv = X.to_numpy()
    Fv = F.to_numpy()
    factor_var = np.einsum("ik,kl,il->i", Xv, Fv, Xv)
    total_vol = np.sqrt(np.clip(factor_var, 0, None) + spec.to_numpy() ** 2)

    cum = (1 + freturns).cumprod() - 1
    show = [MARKET_FACTOR, *STYLE_FACTORS]

    return {
        "meta": a["meta"],
        "factors": factors,
        "styles": STYLE_FACTORS,
        "market": MARKET_FACTOR,
        "cov": [_round(row, 8) for row in Fv],
        "factor_vol": {f: round(float(np.sqrt(F.loc[f, f])), 5) for f in factors},
        "assets": {
            "tickers": list(X.index),
            "exposures": [_round(row, 4) for row in Xv],
            "specific": _round(spec, 4),
            "total_vol": _round(total_vol, 4),
            "industry": list(industry),
        },
        "factor_returns": {
            "dates": [d.strftime("%Y-%m-%d") for d in cum.index],
            "cumulative": {f: _round(cum[f], 4) for f in show if f in cum},
        },
    }


def export_site(artifacts_dir: str | Path, template: str | Path, out: str | Path) -> Path:
    data = build_site_data(artifacts_dir)
    html = Path(template).read_text()
    if PLACEHOLDER not in html:
        raise ValueError(f"Template {template} is missing the {PLACEHOLDER} placeholder")
    payload = json.dumps(data, separators=(",", ":"))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html.replace(PLACEHOLDER, payload))
    return out


def main() -> None:
    p = argparse.ArgumentParser(prog="riskprism-site",
                                description="Render the static explorer from model artifacts.")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--template", default="site/template.html")
    p.add_argument("--out", default="site/index.html")
    args = p.parse_args()
    out = export_site(args.artifacts, args.template, args.out)
    print(f"[riskprism] site written to {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
