"""End-to-end model build: universe -> data -> exposures -> regressions -> risk.

This is the pipeline the weekly GitHub Action runs, and what anyone can
run locally to reproduce a published model (hybrid open-data strategy:
artifacts are distributed, the pipeline that made them is open).
"""

import numpy as np
import pandas as pd

from osrisk.artifacts import save_artifacts
from osrisk.config import MARKET_FACTOR, ModelConfig
from osrisk.data.edgar import EdgarClient, Fundamentals
from osrisk.data.prices import get_provider, load_price_panel
from osrisk.data.universe import apply_liquidity_filters, candidate_tickers
from osrisk.factors.industry import industry_dummies, sic_to_industry
from osrisk.factors.style import compute_style_exposures
from osrisk.model.covariance import factor_covariance
from osrisk.model.regression import cross_sectional_regression
from osrisk.model.specific import specific_risk

FUND_FIELDS = ["book_equity", "total_assets", "total_liabilities", "net_income", "shares_out"]

# Daily lookback needed before the first regression date (momentum window
# plus skip, with slack for holidays).
_BURN_IN_DAYS = 280


def build_model(
    tickers: list[str] | None = None,
    provider: str = "stooq",
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    max_names: int | None = None,
    artifacts_dir: str = "artifacts",
    config: ModelConfig | None = None,
    edgar: EdgarClient | None = None,
    verbose: bool = True,
) -> dict:
    config = config or ModelConfig()
    edgar = edgar or EdgarClient()
    end = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    start = pd.Timestamp(start) if start else end - pd.DateOffset(years=4)

    def log(msg: str):
        if verbose:
            print(f"[osrisk] {msg}")

    # ---- universe ----------------------------------------------------
    ticker_meta = candidate_tickers(edgar, max_names=max_names)
    if tickers is not None:
        wanted = {t.upper() for t in tickers}
        ticker_meta = ticker_meta[ticker_meta["ticker"].isin(wanted)]
    cik_by_ticker = dict(zip(ticker_meta["ticker"], ticker_meta["cik"]))
    log(f"universe candidates: {len(cik_by_ticker)}")

    # ---- prices ------------------------------------------------------
    price_provider = get_provider(provider)
    close, volume = load_price_panel(
        list(cik_by_ticker), price_provider, start, end, progress=verbose
    )
    kept = apply_liquidity_filters(close, volume, config)
    close, volume = close[kept], volume[kept]
    log(f"after liquidity filters: {len(kept)} names")
    if len(kept) < config.min_assets_per_regression:
        raise ValueError(
            f"Universe too small after filters ({len(kept)}); "
            f"need >= {config.min_assets_per_regression}"
        )

    # ---- fundamentals & industries ------------------------------------
    fundamentals: dict[str, Fundamentals] = {}
    industries = {}
    for i, ticker in enumerate(kept):
        cik = cik_by_ticker[ticker]
        facts = edgar.company_facts(cik)
        fundamentals[ticker] = Fundamentals.from_facts(facts) if facts else Fundamentals({})
        industries[ticker] = sic_to_industry(edgar.sic_code(cik))
        if verbose and (i + 1) % 100 == 0:
            log(f"fundamentals: {i + 1}/{len(kept)}")
    industries = pd.Series(industries)

    # ---- weekly rebalance loop ----------------------------------------
    weekly_close = close.resample(config.frequency).last()
    weekly_returns = weekly_close.ffill().pct_change()
    first_regression = close.index[0] + pd.Timedelta(days=_BURN_IN_DAYS)
    rebal_dates = [d for d in weekly_close.index if d >= first_regression]

    factor_return_rows: dict[pd.Timestamp, pd.Series] = {}
    residual_rows: dict[pd.Timestamp, pd.Series] = {}
    r2s = []
    last_exposures, last_mktcap = None, None

    for t, t_next in zip(rebal_dates[:-1], rebal_dates[1:]):
        fund_df = pd.DataFrame(
            {tk: fundamentals[tk].asof(t) for tk in kept}
        ).T.reindex(columns=FUND_FIELDS)
        exposures, mktcap = compute_style_exposures(close, volume, fund_df, t, config)
        try:
            res = cross_sectional_regression(
                weekly_returns.loc[t_next],
                exposures,
                industries,
                mktcap,
                min_assets=config.min_assets_per_regression,
            )
        except ValueError:
            continue
        factor_return_rows[t_next] = res.factor_returns
        residual_rows[t_next] = res.residuals
        r2s.append(res.r2)
        last_exposures, last_mktcap = exposures, mktcap

    if len(factor_return_rows) < config.vol_half_life:
        raise ValueError(
            f"Only {len(factor_return_rows)} usable regression periods; "
            "extend the date range or loosen filters"
        )
    log(f"regressions: {len(factor_return_rows)} weeks, mean R2 = {np.mean(r2s):.3f}")

    # ---- assemble risk model -------------------------------------------
    factor_returns = pd.DataFrame(factor_return_rows).T.sort_index().fillna(0.0)
    residuals = pd.DataFrame(residual_rows).T.sort_index()
    F = factor_covariance(factor_returns, config)
    spec = specific_risk(residuals, last_mktcap, config)

    final_dummies = industry_dummies(industries.reindex(last_exposures.index))
    X_final = pd.concat(
        [pd.Series(1.0, index=last_exposures.index, name=MARKET_FACTOR), last_exposures, final_dummies],
        axis=1,
    ).reindex(columns=F.columns).fillna(0.0)

    as_of = rebal_dates[-1]
    meta = {
        "model_version": config.version,
        "as_of": str(as_of.date()),
        "n_assets": int(len(X_final)),
        "n_periods": int(len(factor_returns)),
        "mean_r2": float(np.mean(r2s)),
        "price_provider": provider,
        "config": config.to_dict(),
    }
    path = save_artifacts(artifacts_dir, X_final, F, spec.reindex(X_final.index), factor_returns, meta)
    log(f"artifacts written to {path} ({meta['n_assets']} assets as of {meta['as_of']})")
    return meta
