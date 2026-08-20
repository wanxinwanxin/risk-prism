"""End-to-end model build: universe -> data -> exposures -> regressions -> risk.

Two universes:
  * estimation universe — liquid names whose returns estimate the factor
    returns (clean cross-sections);
  * coverage universe — every name alive at the build date; risk comes
    through the factor structure (x'Fx) plus a structural specific-risk
    prior, so no asset-level history is required.

Capture-forward: pass ``prior_artifacts`` (the previous build) and only
the new weeks are regressed, appended to the prior factor-return and
residual history. Names that stop trading get an imputed delisting
return in their final week and keep their historical rows — so history
accrued after launch is survivorship-free by construction.
"""

import numpy as np
import pandas as pd

from riskprism.artifacts import load_artifacts, save_artifacts
from riskprism.config import MARKET_FACTOR, ModelConfig
from riskprism.data.edgar import EdgarClient, Fundamentals, store_from_frame, store_to_frame
from riskprism.data.prices import get_provider, load_price_panel
from riskprism.data.universe import apply_liquidity_filters, candidate_tickers, coverage_universe
from riskprism.factors.industry import industry_dummies, sic_to_industry
from riskprism.factors.style import compute_style_exposures
from riskprism.model.covariance import factor_covariance
from riskprism.model.history import delisting_return, merge_history
from riskprism.model.regression import cross_sectional_regression
from riskprism.model.specific import specific_risk

FUND_FIELDS = ["book_equity", "total_assets", "total_liabilities", "net_income", "shares_out"]

# Daily lookback needed before the first regression date (momentum window
# plus skip, with slack for holidays).
_BURN_IN_DAYS = 280


def build_model(
    tickers: list[str] | None = None,
    provider: str = "yahoo",
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    max_names: int | None = None,
    artifacts_dir: str = "artifacts",
    config: ModelConfig | None = None,
    edgar: EdgarClient | None = None,
    prior_artifacts: str | None = None,
    verbose: bool = True,
) -> dict:
    config = config or ModelConfig()
    edgar = edgar or EdgarClient()
    end = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    start = pd.Timestamp(start) if start else end - pd.DateOffset(years=4)

    def log(msg: str):
        if verbose:
            print(f"[riskprism] {msg}")

    # ---- prior build (capture-forward) --------------------------------
    prior_fr = prior_res = None
    prior_fund: dict[str, Fundamentals] = {}
    prior_industry: dict[str, str] = {}
    if prior_artifacts:
        prior = load_artifacts(prior_artifacts)
        prior_version = prior["meta"].get("model_version")
        # fundamentals/industry fallbacks survive version bumps — they're
        # data, not methodology
        if prior.get("fundamentals_store") is not None:
            prior_fund = store_from_frame(prior["fundamentals_store"])
        if prior.get("asset_meta") is not None:
            prior_industry = prior["asset_meta"]["industry"].to_dict()
        if prior_version != config.version:
            log(f"prior build is {prior_version}, config is {config.version}: "
                "methodology changed, rebuilding history cold")
        else:
            prior_fr, prior_res = prior["factor_returns"], prior["residuals"]
            log(f"prior build loaded: {len(prior_fr)} weeks through {prior_fr.index[-1].date()}")

    # ---- universe ------------------------------------------------------
    ticker_meta = candidate_tickers(edgar, max_names=max_names)
    if tickers is not None:
        wanted = {t.upper() for t in tickers}
        ticker_meta = ticker_meta[ticker_meta["ticker"].isin(wanted)]
    cik_by_ticker = dict(zip(ticker_meta["ticker"], ticker_meta["cik"]))
    log(f"universe candidates: {len(cik_by_ticker)}")

    # ---- prices --------------------------------------------------------
    price_provider = get_provider(provider)
    close, volume = load_price_panel(
        list(cik_by_ticker), price_provider, start, end, progress=verbose
    )
    close = close.where(close > 0)  # zero/negative closes are data errors, not prices
    volume = volume.where(volume >= 0)
    estimation = apply_liquidity_filters(close, volume, config)
    coverage = coverage_universe(close, config)
    active = sorted(set(estimation) | set(coverage))
    close, volume = close[active], volume[active]
    log(f"estimation universe: {len(estimation)} · coverage universe: {len(coverage)}")
    if len(estimation) < config.min_assets_per_regression:
        raise ValueError(
            f"Estimation universe too small ({len(estimation)}); "
            f"need >= {config.min_assets_per_regression}"
        )

    # ---- fundamentals & industries -------------------------------------
    # Large universes come from SEC's nightly bulk zips (one request);
    # per-company API calls are only a small-scale top-up. When SEC is
    # unreachable entirely, the prior release's distilled store fills in —
    # fundamentals move quarterly, so a weeks-old snapshot barely changes
    # exposures.
    try:
        edgar.bulk_prefetch([cik_by_ticker[t] for t in active], verbose=verbose)
    except Exception as exc:
        log(f"bulk prefetch unavailable ({exc}); using per-company cache/API")
    fundamentals: dict[str, Fundamentals] = {}
    industries = {}
    n_live = n_fallback = 0
    for i, ticker in enumerate(active):
        cik = cik_by_ticker[ticker]
        facts = edgar.company_facts(cik)
        if facts:
            fundamentals[ticker] = Fundamentals.from_facts(facts)
            n_live += 1
        elif ticker in prior_fund:
            fundamentals[ticker] = prior_fund[ticker]
            n_fallback += 1
        else:
            fundamentals[ticker] = Fundamentals({})
        sic = edgar.sic_code(cik)
        industries[ticker] = (sic_to_industry(sic) if sic is not None
                              else prior_industry.get(ticker, "Other"))
        if verbose and (i + 1) % 100 == 0:
            log(f"fundamentals: {i + 1}/{len(active)}")
    industries = pd.Series(industries)
    if n_fallback:
        log(f"fundamentals: {n_live} live from EDGAR, {n_fallback} from prior release "
            "(EDGAR unreachable from this IP)")

    # ---- weekly rebalance loop (estimation universe only) ---------------
    estu_idx = pd.Index(estimation)
    weekly_close = close[estimation].resample(config.frequency).last()
    weekly_returns = weekly_close.pct_change(fill_method=None)
    last_traded_week = weekly_close.apply(lambda s: s.last_valid_index())

    first_regression = close.index[0] + pd.Timedelta(days=_BURN_IN_DAYS)
    if prior_fr is not None and len(prior_fr):
        first_regression = max(first_regression, prior_fr.index[-1])
    rebal_dates = [d for d in weekly_close.index if d >= first_regression]

    def fund_asof(names, date):
        return pd.DataFrame(
            {tk: fundamentals[tk].asof(date) for tk in names}
        ).T.reindex(columns=FUND_FIELDS)

    factor_return_rows: dict[pd.Timestamp, pd.Series] = {}
    residual_rows: dict[pd.Timestamp, pd.Series] = {}
    r2s = []
    for t, t_next in zip(rebal_dates[:-1], rebal_dates[1:]):
        exposures, mktcap = compute_style_exposures(
            close[estimation], volume[estimation], fund_asof(estimation, t), t,
            config, industries=industries, fit=estu_idx,
        )
        y = weekly_returns.loc[t_next].copy()
        # capture-forward: a name whose last trade was week t gets an
        # imputed delisting return in week t_next instead of dropping out
        for tk in estimation:
            if last_traded_week[tk] == t:
                y[tk] = delisting_return(float(weekly_close.at[t, tk]), config)
        try:
            res = cross_sectional_regression(
                y, exposures, industries, mktcap,
                min_assets=config.min_assets_per_regression,
            )
        except ValueError:
            continue
        factor_return_rows[t_next] = res.factor_returns
        residual_rows[t_next] = res.residuals
        r2s.append(res.r2)

    new_fr = pd.DataFrame(factor_return_rows).T.sort_index()
    new_res = pd.DataFrame(residual_rows).T.sort_index()
    factor_returns = merge_history(prior_fr, new_fr, config.history_cap_weeks).fillna(0.0)
    residuals = merge_history(prior_res, new_res, config.history_cap_weeks)
    if len(factor_returns) < config.vol_half_life:
        raise ValueError(
            f"Only {len(factor_returns)} usable regression periods; "
            "extend the date range or loosen filters"
        )
    log(f"regressions: {len(new_fr)} new weeks, {len(factor_returns)} total"
        + (f", mean new R2 = {np.mean(r2s):.3f}" if r2s else ""))

    # ---- assemble risk model (coverage universe) ------------------------
    F = factor_covariance(factor_returns, config)
    as_of = weekly_close.index[-1]

    cov_exposures, cov_mktcap = compute_style_exposures(
        close[coverage], volume[coverage], fund_asof(coverage, as_of), as_of,
        config, industries=industries,
        fit=estu_idx.intersection(coverage),
    )
    final_dummies = industry_dummies(industries.reindex(cov_exposures.index).fillna("Other"))
    X_final = pd.concat(
        [pd.Series(1.0, index=cov_exposures.index, name=MARKET_FACTOR),
         cov_exposures, final_dummies],
        axis=1,
    ).reindex(columns=F.columns).fillna(0.0)

    spec = specific_risk(residuals, X_final, industries, config)

    asset_meta = pd.DataFrame({
        "in_estimation": X_final.index.isin(estimation),
        "history_weeks": spec.n_obs.astype(int),
        "specific_blend_weight": spec.blend_weight.round(3),
        "specific_ts": spec.ts_vol.round(4),
        "specific_structural": spec.structural.round(4),
        "industry": industries.reindex(X_final.index).fillna("Other"),
    }, index=X_final.index)

    meta = {
        "model_version": config.version,
        "as_of": str(as_of.date()),
        "n_assets": int(len(X_final)),
        "n_estimation": int(len(estimation)),
        "n_periods": int(len(factor_returns)),
        "n_new_periods": int(len(new_fr)),
        "mean_r2": float(np.mean(r2s)) if r2s else None,
        "price_provider": provider,
        "incremental": bool(prior_artifacts),
        "fundamentals_live": n_live,
        "fundamentals_from_prior": n_fallback,
        "config": config.to_dict(),
    }
    path = save_artifacts(
        artifacts_dir, X_final, F, spec.vol.reindex(X_final.index), factor_returns, meta,
        residuals=residuals, asset_meta=asset_meta,
        fundamentals_store=store_to_frame(fundamentals),
    )
    log(f"artifacts written to {path} ({meta['n_assets']} covered, "
        f"{meta['n_estimation']} estimation, as of {meta['as_of']})")
    return meta
