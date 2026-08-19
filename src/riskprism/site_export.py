"""Export model artifacts as the static explorer site.

Renders three files into the output directory:

    index.html   the interactive explorer (data embedded, no server calls)
    model.md     agent-readable model card — plain markdown mirror of the build
    llms.txt     llms.txt-convention pointer to model.md

The weekly GitHub Action publishes the directory; Railway (or any static
file server) serves it as-is.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from riskprism.artifacts import load_artifacts
from riskprism.config import MARKET_FACTOR, STYLE_FACTORS
from riskprism.factors.industry import INDUSTRY_PREFIX

PLACEHOLDER = "__RISKPRISM_DATA__"
REPO_URL = "https://github.com/wanxinwanxin/risk-prism"

STYLE_DEFS = {
    "size": "ln(market cap)",
    "value": "book equity / market cap",
    "momentum": "12-month return, skipping the most recent month",
    "volatility": "252-day daily return std, annualized",
    "liquidity": "ln(63-day median dollar volume / market cap)",
    "quality": "ROE: annual net income / book equity",
    "leverage": "total liabilities / total assets",
}


def _round(values, nd=5):
    return [round(float(v), nd) for v in values]


def _load(artifacts_dir):
    a = load_artifacts(artifacts_dir)
    X = a["exposures"]
    F = a["factor_covariance"]
    spec = a["specific_risk"].reindex(X.index).fillna(0.0)

    ind_cols = [f for f in F.columns if f.startswith(INDUSTRY_PREFIX)]
    ind_block = X[ind_cols]
    industry = ind_block.idxmax(axis=1).str.replace(INDUSTRY_PREFIX, "", regex=False)
    industry[ind_block.sum(axis=1) == 0] = "Other"

    Xv, Fv = X.to_numpy(), F.to_numpy()
    factor_var = np.einsum("ik,kl,il->i", Xv, Fv, Xv)
    total_vol = np.sqrt(np.clip(factor_var, 0, None) + spec.to_numpy() ** 2)

    am = a.get("asset_meta")
    if am is None:
        am = pd.DataFrame({"in_estimation": True, "history_weeks": 0,
                           "specific_blend_weight": 1.0}, index=X.index)
    am = am.reindex(X.index)
    return a, X, F, spec, industry, total_vol, am


def build_site_data(artifacts_dir: str | Path) -> dict:
    a, X, F, spec, industry, total_vol, am = _load(artifacts_dir)
    freturns = a["factor_returns"]
    cum = (1 + freturns).cumprod() - 1
    show = [MARKET_FACTOR, *STYLE_FACTORS]
    return {
        "meta": a["meta"],
        "factors": list(F.columns),
        "styles": STYLE_FACTORS,
        "market": MARKET_FACTOR,
        "repo": REPO_URL,
        "cov": [_round(row, 8) for row in F.to_numpy()],
        "factor_vol": {f: round(float(np.sqrt(F.loc[f, f])), 5) for f in F.columns},
        "assets": {
            "tickers": list(X.index),
            "exposures": [_round(row, 4) for row in X.to_numpy()],
            "specific": _round(spec, 4),
            "total_vol": _round(total_vol, 4),
            "industry": list(industry),
            "estu": [int(v) for v in am["in_estimation"].fillna(False)],
            "weeks": [int(v) for v in am["history_weeks"].fillna(0)],
            "blend": _round(am["specific_blend_weight"].fillna(0.0), 3),
        },
        "factor_returns": {
            "dates": [d.strftime("%Y-%m-%d") for d in cum.index],
            "cumulative": {f: _round(cum[f], 4) for f in show if f in cum},
        },
    }


def build_model_md(artifacts_dir: str | Path) -> str:
    a, X, F, spec, industry, total_vol, am = _load(artifacts_dir)
    m = a["meta"]
    cfg = m.get("config", {})
    vol = lambda f: float(np.sqrt(F.loc[f, f]))
    n_estu = int(am["in_estimation"].fillna(False).sum())

    lines = [
        "# riskprism — model card",
        "",
        "> Open-source Barra-style US equity factor risk model, built for AI",
        "> agents. This file is the agent-readable mirror of the current weekly",
        "> build. Research software; not investment advice.",
        "",
        "## Current build",
        "",
        "| field | value |",
        "|---|---|",
        f"| model version | {m.get('model_version')} |",
        f"| as of | {m.get('as_of')} |",
        f"| assets covered | {m.get('n_assets')} |",
        f"| estimation universe | {n_estu} |",
        f"| regression weeks | {m.get('n_periods')} |",
        f"| mean weekly R² | {(m.get('mean_r2') or 0):.3f} |",
        f"| price provider | {m.get('price_provider')} |",
        f"| frequency | {cfg.get('frequency', 'W-FRI')} (annualized outputs) |",
        "",
        "## How to use this model",
        "",
        "### As an AI agent (MCP server)",
        "",
        "```json",
        json.dumps({"mcpServers": {"riskprism": {
            "command": "riskprism-mcp",
            "env": {"RISKPRISM_ARTIFACTS": "/path/to/artifacts"}}}}, indent=2),
        "```",
        "",
        "Tools: `get_model_info`, `get_portfolio_risk`, `get_factor_exposures`,",
        "`stress_test`, `check_coverage`. Weights are portfolio weights (shorts",
        "negative, any gross); volatilities are annualized decimals.",
        "",
        "### In Python",
        "",
        "```python",
        f"# pip install git+{REPO_URL}",
        "from riskprism import RiskModel",
        'model = RiskModel.load("artifacts")',
        'model.portfolio_risk({"AAPL": 0.4, "MSFT": 0.3, "XOM": 0.3})',
        "```",
        "",
        f"### Raw artifacts (parquet): [{REPO_URL}/releases]({REPO_URL}/releases)",
        "",
        "`exposures.parquet` (asset × factor), `factor_covariance.parquet`",
        "(K × K annualized), `specific_risk.parquet`, `factor_returns.parquet`,",
        "`meta.json`.",
        "",
        "## Factors",
        "",
        "| factor | descriptor | current ann. vol |",
        "|---|---|---|",
        f"| market | intercept — cap-weighted market return | {vol(MARKET_FACTOR):.1%} |",
    ]
    for f in STYLE_FACTORS:
        lines.append(f"| {f} | {STYLE_DEFS[f]} | {vol(f):.1%} |")
    lines += [
        "",
        "Style exposures are winsorized at ±3σ and standardized to cap-weighted",
        "mean 0 / equal-weighted std 1 each week. Industries: Fama-French 12",
        "from SEC EDGAR SIC codes, one-hot, cap-weighted returns constrained",
        "to zero for identification.",
        "",
        "### Industry coverage in this build",
        "",
        "| industry | assets |",
        "|---|---|",
    ]
    for name, count in industry.value_counts().items():
        lines.append(f"| {name} | {count} |")

    fs = [MARKET_FACTOR, *STYLE_FACTORS]
    lines += ["", "## Factor correlation (market + styles)", "",
              "| | " + " | ".join(fs) + " |",
              "|---|" + "---|" * len(fs)]
    for fr in fs:
        row = [f"{F.loc[fr, fc] / np.sqrt(F.loc[fr, fr] * F.loc[fc, fc]):.2f}" for fc in fs]
        lines.append(f"| **{fr}** | " + " | ".join(row) + " |")

    lines += [
        "",
        f"## Coverage ({len(X)} tickers)",
        "",
        ", ".join(X.index),
        "",
        "## Methodology in brief",
        "",
        "1. Universe: EDGAR-registered US common stocks; price ≥ $2, 21-day",
        "   median dollar volume ≥ $1M, ≥ 26 weeks of history.",
        "2. Point-in-time fundamentals from EDGAR XBRL (values used only after",
        "   their `filed` date); daily adjusted prices from a pluggable provider.",
        "3. Two universes: liquid names (price ≥ $2, ADV ≥ $1M, ≥ 26w history)",
        "   estimate the factor returns; every name alive at the build date is",
        "   covered — risk comes through the factor structure plus a structural",
        "   specific-risk prior, so no asset-level history is required.",
        "4. Weekly cross-sectional WLS regression of returns on exposures,",
        "   √(market cap) weights, industry returns cap-weighted to zero.",
        "5. Factor covariance: EWMA on weekly factor returns — vol half-life",
        f"   {cfg.get('vol_half_life', 13)}w, correlation half-life {cfg.get('corr_half_life', 26)}w — annualized ×{int(cfg.get('ann_factor', 52))}, repaired to PSD.",
        "6. Specific risk: each asset's EWMA residual vol blended with a",
        "   cross-sectional structural prediction (from size, volatility,",
        "   liquidity, industry) by history length: w = T/(T + 26w). Assets",
        "   without history get the pure structural prior.",
        "7. Capture-forward history: each weekly build appends to the prior",
        "   build's factor returns and residuals; names that stop trading get",
        "   an imputed delisting return in their final week and keep their",
        "   historical rows, so post-launch history is survivorship-free.",
        "8. Portfolio risk: Σ = X F Xᵀ + diag(s²).",
        "",
        "### Per-asset estimation quality",
        "",
        "`get_factor_exposures` and the artifacts' `asset_meta.parquet` report,",
        "per asset: `in_estimation` (participates in factor regressions),",
        "`history_weeks` (residual observations), and `specific_blend_weight`",
        "(how much of the specific-risk estimate is the asset's own history vs",
        "the structural prior). Low-weight names are prior-driven — treat their",
        "numbers as informed estimates, not measurements.",
        "",
        "## Known limitations",
        "",
        "- Survivorship bias in the cold-start history: weeks recorded before",
        "  this project launched exclude names that had already delisted.",
        "  Capture-forward appending plus the 13/26-week EWMA half-lives make",
        "  this bias decay away — the effective window is largely bias-free",
        "  ~18-24 months after launch. Factor-return means are affected more",
        "  than the covariances this model actually ships.",
        "- Delisting classification is a price heuristic (merger vs failure),",
        "  not filing-verified.",
        "- Universe heuristics are crude (ticker-pattern filters; some ADRs",
        "  leak through).",
        "- Stress tests are first-order (exposure × shock).",
        "",
        "## Links",
        "",
        "- Interactive explorer: [/](/)",
        f"- Source, docs, full methodology: [{REPO_URL}]({REPO_URL})",
        "",
    ]
    return "\n".join(lines)


LLMS_TXT = f"""# riskprism

> Open-source Barra-style US equity factor risk model for AI agents:
> factor exposures, factor covariance, and specific risk for liquid US
> stocks, rebuilt weekly from public data. Not investment advice.

- [Model card, current build, and usage (markdown)](/model.md)
- [Interactive explorer](/)
- [Source and methodology]({REPO_URL})
"""


def export_site(artifacts_dir: str | Path, template: str | Path, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    html = Path(template).read_text()
    if PLACEHOLDER not in html:
        raise ValueError(f"Template {template} is missing the {PLACEHOLDER} placeholder")
    payload = json.dumps(build_site_data(artifacts_dir), separators=(",", ":"))
    (out_dir / "index.html").write_text(html.replace(PLACEHOLDER, payload))
    (out_dir / "model.md").write_text(build_model_md(artifacts_dir))
    (out_dir / "llms.txt").write_text(LLMS_TXT)
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(prog="riskprism-site",
                                description="Render the static explorer from model artifacts.")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--template", default="site/template.html")
    p.add_argument("--out", default="site", help="Output directory")
    args = p.parse_args()
    out = export_site(args.artifacts, args.template, args.out)
    size = sum(f.stat().st_size for f in out.glob("*") if f.name != "template.html")
    print(f"[riskprism] site rendered to {out}/ (index.html + model.md + llms.txt, {size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
