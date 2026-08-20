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
from riskprism.config import STYLE_FACTORS as _STYLES, ModelConfig
from riskprism.model.baselines import comparison_payload
from riskprism.model.validation import FULL_FACTORS, RunningRiskState, validation_summary

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
        "validation": _validation_payload(a.get("validation")),
        "sample_week": {
            "date": freturns.index[-1].strftime("%Y-%m-%d"),
            "f": {f: round(float(freturns.iloc[-1].get(f, 0.0)), 5)
                  for f in [MARKET_FACTOR, *STYLE_FACTORS]},
        },
    }


def _validation_payload(val: pd.DataFrame | None) -> dict | None:
    if val is None or val.empty:
        return None
    summary = validation_summary(val)
    if summary.empty:
        return None
    z = val["z"].to_numpy()
    z = z[np.isfinite(z)]
    edges = np.linspace(-4, 4, 33)
    hist, _ = np.histogram(np.clip(z, -3.99, 3.99), bins=edges, density=True)

    def _num(v, nd=3):
        return None if v is None or not np.isfinite(v) else round(float(v), nd)

    payload = {
        "n_weeks": int(val["date"].nunique()),
        "n_scores": int(len(z)),
        "overall_bias": round(float(np.std(z, ddof=1)), 3),
        "exceed_95": round(float((np.abs(z) > 1.96).mean()), 4),
        "portfolios": [
            {"name": r["portfolio"], "group": r["group"], "n": int(r["n"]),
             "bias": round(r["bias_stat"], 3), "exc": round(r["exceed_95"], 3),
             "vol": round(r["mean_forecast_vol"], 4),
             "rvol": _num(r.get("mean_realized_vol"), 4),
             "vratio": _num(r.get("vol_ratio"))}
            for _, r in summary.iterrows()
        ],
        "z_hist": {"edges": _round(edges, 3), "density": _round(hist, 4)},
    }
    dates = sorted(val["date"].unique())
    date_pos = {d: i for i, d in enumerate(dates)}
    series = {}
    for name, g in val.groupby("portfolio"):
        if len(g) < 30:
            continue
        fc = [None] * len(dates)
        rv = [None] * len(dates)
        for _, r in g.iterrows():
            i = date_pos[r["date"]]
            fc[i] = _num(r["forecast_vol_ann"], 4)
            rv[i] = _num(r.get("realized_vol_ann"), 4)
        series[name] = {"fc": fc, "rv": rv, "group": g["group"].iloc[0]}
    payload["series"] = {
        "dates": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates],
        "portfolios": series,
    }
    if "realized_vol_ann" in val.columns:
        ok = val.dropna(subset=["realized_vol_ann", "forecast_vol_ann"])
        if len(ok) >= 30:
            fv2 = (ok["forecast_vol_ann"] ** 2).to_numpy()
            rv2 = (ok["realized_vol_ann"] ** 2).to_numpy()
            slope, intercept = np.polyfit(fv2, rv2, 1)
            r2 = float(np.corrcoef(fv2, rv2)[0, 1] ** 2)
            sample = ok.sample(min(1200, len(ok)), random_state=0)
            payload["rv"] = {
                "mz_slope": round(float(slope), 3),
                "mz_intercept": round(float(intercept), 5),
                "r2": round(r2, 3),
                "vol_ratio": round(float(np.sqrt(rv2.mean() / fv2.mean())), 3),
                "scatter": [[round(float(a), 3), round(float(b), 3), g]
                            for a, b, g in zip(sample["forecast_vol_ann"],
                                               sample["realized_vol_ann"],
                                               sample["group"])],
            }
    # baseline / vendor-protocol comparison; the render must survive the
    # FF download (or anything else here) failing
    try:
        comparison = comparison_payload(val)
    except Exception as exc:
        print(f"[riskprism] comparison block skipped: {exc}")
        comparison = None
    if comparison:
        payload["comparison"] = comparison
    return payload


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
        f"| fundamentals | {m.get('fundamentals_live', 'n/a')} live from EDGAR"
        + (f", {m['fundamentals_from_prior']} from prior release"
           if m.get("fundamentals_from_prior") else "") + " |",
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
        "`residuals.parquet`, `exposure_history.parquet`, `validation.parquet`,",
        "`asset_meta.parquet`, `fundamentals_store.parquet`, `meta.json`.",
        "",
        "### Historical (point-in-time) models",
        "",
        "Every weekly formation date is reconstructible from the same artifact",
        "download — `riskprism.model.asof.model_asof(artifacts, date)` returns a",
        "full RiskModel with no lookahead. The explorer serves the same",
        "snapshots as static JSON under [/history/index.json](/history/index.json).",
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
        *_validation_md(a.get("validation")),
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
        "  Audited severity (SEC bulk archives, 2026-08): departed filers are",
        "  ~8.7% of filer book equity, bounding cap-weighted return-mean bias",
        "  at ~1.4-2.5bp/week; covariances (what this model ships) are affected",
        "  at second order. Capture-forward appending plus the 13/26-week EWMA",
        "  half-lives make the bias decay away within ~18-24 months of launch.",
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


def _validation_md(val: pd.DataFrame | None) -> list[str]:
    if val is None or val.empty:
        return []
    summary = validation_summary(val)
    if summary.empty:
        return []
    z = val["z"].to_numpy()
    z = z[np.isfinite(z)]
    lines = [
        "## Forecast validation (out-of-sample, in-build)",
        "",
        "Every historical week, the model forecast next-week volatility for a",
        "panel of test portfolios using only data available at the time; z =",
        "realized return / forecast vol. A calibrated model gives std(z) — the",
        "**bias statistic** — of ~1.0 (>1 underforecasts risk, <1 overforecasts)",
        "and |z| > 1.96 about 5% of the time.",
        "",
        f"Overall: **bias statistic {np.std(z, ddof=1):.2f}**, "
        f"|z|>1.96 rate {(np.abs(z) > 1.96).mean():.1%}, "
        f"{val['date'].nunique()} weeks × {summary['n'].sum()} portfolio-scores.",
        "",
        "| portfolio | bias stat | \\|z\\|>1.96 | mean forecast vol | mean realized vol | vol ratio |",
        "|---|---|---|---|---|---|",
    ]
    def _fmt(v, f):
        return f.format(v) if v is not None and np.isfinite(v) else "—"
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['portfolio']} | {r['bias_stat']:.2f} | {r['exceed_95']:.1%} "
            f"| {r['mean_forecast_vol']:.1%} "
            f"| {_fmt(r.get('mean_realized_vol'), '{:.1%}')} "
            f"| {_fmt(r.get('vol_ratio'), '{:.2f}')} |")
    lines += [
        "",
        "The vol ratio compares average realized variance (from daily returns",
        "within each week) to average forecast variance, in vol units — an",
        "RV-based check with far more statistical power than z-scores alone.",
        "",
    ]
    return lines


LLMS_TXT = f"""# riskprism

> Open-source Barra-style US equity factor risk model for AI agents:
> factor exposures, factor covariance, and specific risk for liquid US
> stocks, rebuilt weekly from public data. Not investment advice.

- [Model card, current build, and usage (markdown)](/model.md)
- [Interactive explorer](/)
- [Source and methodology]({REPO_URL})
"""


def export_history(artifacts_dir: str | Path, out_dir: str | Path,
                   config: ModelConfig | None = None) -> int:
    """Per-week point-in-time model snapshots as static JSON.

    Each file carries the formation-date exposures, the replayed EWMA
    factor covariance and specific risk (annualized), and the *next*
    week's factor returns and residuals — everything the explorer needs
    for as-of portfolio risk and forecast-vs-realized backtests, served
    as plain static files.
    """
    config = config or ModelConfig()
    a = load_artifacts(artifacts_dir)
    eh, fr, res = a.get("exposure_history"), a["factor_returns"], a["residuals"]
    am = a.get("asset_meta")
    if eh is None or eh.empty or res is None:
        return 0
    hist_dir = Path(out_dir) / "history"
    hist_dir.mkdir(parents=True, exist_ok=True)

    ind_of = (am["industry"].to_dict() if am is not None else {})
    fr_dates = list(fr.index)
    eh_dates = sorted(pd.to_datetime(eh["date"].unique()))
    state = RunningRiskState(config)
    k = 0
    written = []
    for t in eh_dates:
        while k < len(fr_dates) and fr_dates[k] <= t:
            d = fr_dates[k]
            e = res.loc[d].dropna() if d in res.index else pd.Series(dtype=float)
            state.update(fr.loc[d], e)
            k += 1
        if not state.ready or k >= len(fr_dates):
            continue  # warm-up, or last formation week (nothing realized yet)
        snap = eh[eh["date"] == t].set_index("ticker")
        tickers = list(snap.index)
        t_next = fr_dates[k]
        resid_next = res.loc[t_next].reindex(tickers) if t_next in res.index else pd.Series(np.nan, index=tickers)
        spec_ann = np.sqrt(state.specific_var_weekly(pd.Index(tickers)) * config.ann_factor)
        payload = {
            "date": t.strftime("%Y-%m-%d"),
            "tickers": tickers,
            "styles": snap[_STYLES].astype(float).round(3).to_numpy().tolist(),
            "industry": [ind_of.get(tk, "Other") for tk in tickers],
            "spec": _round(spec_ann, 4),
            "fcov": [_round(row, 8) for row in state.factor_cov_weekly() * config.ann_factor],
            "next": {
                "date": t_next.strftime("%Y-%m-%d"),
                "f": _round(fr.loc[t_next].reindex(FULL_FACTORS).fillna(0.0), 6),
                "resid": [None if not np.isfinite(v) else round(float(v), 5)
                          for v in resid_next],
            },
        }
        name = f"{payload['date']}.json"
        (hist_dir / name).write_text(json.dumps(payload, separators=(",", ":")))
        written.append(payload["date"])
    (hist_dir / "index.json").write_text(json.dumps(
        {"dates": written, "factors": FULL_FACTORS, "style_factors": _STYLES},
        separators=(",", ":")))
    return len(written)


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
    n_hist = export_history(args.artifacts, args.out)
    size = sum(f.stat().st_size for f in out.glob("*") if f.name != "template.html")
    print(f"[riskprism] site rendered to {out}/ (index.html + model.md + llms.txt, "
          f"{size / 1024:.0f} KB · {n_hist} weekly history snapshots)")


if __name__ == "__main__":
    main()
