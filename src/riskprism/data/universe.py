"""Universe construction from EDGAR's ticker registry plus liquidity filters."""

import re

import pandas as pd

from riskprism.config import ModelConfig
from riskprism.data.edgar import EdgarClient

# Plain 1-5 letter tickers, optionally a single-letter class suffix
# (BRK-B, BF-B). Drops units, warrants, and preferreds, whose suffixes
# are two letters or longer (-UN, -WT, -PA).
_COMMON_TICKER = re.compile(r"^[A-Z]{1,5}(-[A-Z])?$")


def candidate_tickers(edgar: EdgarClient, max_names: int | None = None) -> pd.DataFrame:
    """Candidate US common stocks: columns [ticker, cik, title].

    One ticker per CIK, keeping the first-listed: EDGAR's ordering tracks
    the primary listing (GOOGL before GOOG, BRK-B before BRK-A) and
    roughly tracks market cap, so ``max_names`` keeps the largest
    companies first.
    """
    df = edgar.ticker_map()
    df = df[df["ticker"].str.match(_COMMON_TICKER)]
    df = df.drop_duplicates("cik").reset_index(drop=True)
    if max_names:
        df = df.head(max_names)
    return df


def apply_liquidity_filters(
    close: pd.DataFrame, volume: pd.DataFrame, config: ModelConfig
) -> list[str]:
    """Estimation universe: price, dollar ADV, and history floors.

    Each name is evaluated at its own last traded date, so a stock that
    was liquid before delisting mid-window still qualifies — its history
    belongs in the factor regressions.
    """
    keep = []
    dollar_vol = close * volume
    for ticker in close.columns:
        px = close[ticker].dropna()
        if px.empty or px.iloc[-1] < config.min_price:
            continue
        if len(px) < config.min_weekly_obs * 5:
            continue
        adv = dollar_vol[ticker].dropna().tail(21)
        if adv.empty or adv.median() < config.min_dollar_adv:
            continue
        keep.append(ticker)
    return keep


def estimation_pool(tickers: list[str], config: ModelConfig) -> list[str]:
    """Candidate pool the estimation screens run on: the first
    ``estimation_max_names`` tickers in EDGAR order (≈ market cap).

    Decouples estimation from coverage: widening the candidate universe
    to ~8,000 EDGAR names must not change which names estimate the
    factors — measured on the 2026-08-22 coverage raise, letting the
    wider pool into estimation blew style-portfolio calibration up
    (ADV-rank capping: style bias 1.12 → 1.57; no cap: overall bias
    0.99 → 1.22 — DECISIONS.md §15). Coverage names get risk through
    the factor structure and the structural prior instead.
    """
    cap = getattr(config, "estimation_max_names", 0) or 0
    if cap and len(tickers) > cap:
        return list(tickers)[:cap]
    return list(tickers)


def coverage_universe(close: pd.DataFrame, config: ModelConfig) -> list[str]:
    """Coverage universe: every name alive and sane at the panel end.

    Much looser than the estimation universe — no history or ADV floor —
    because coverage names get risk through the factor structure and the
    structural specific-risk prior, not through their own history.
    """
    if close.empty:
        return []
    end = close.index[-1]
    keep = []
    for ticker in close.columns:
        px = close[ticker].dropna()
        if px.empty or px.iloc[-1] < config.coverage_min_price:
            continue
        if (end - px.index[-1]).days > config.coverage_max_stale_days:
            continue  # stopped trading: historical name, not coverable today
        keep.append(ticker)
    return keep
