"""Universe construction from EDGAR's ticker registry plus liquidity filters."""

import re

import pandas as pd

from osrisk.config import ModelConfig
from osrisk.data.edgar import EdgarClient

# Plain 1-5 letter tickers only: drops units, warrants, preferreds, and
# most non-common share classes registered on EDGAR.
_COMMON_TICKER = re.compile(r"^[A-Z]{1,5}$")


def candidate_tickers(edgar: EdgarClient, max_names: int | None = None) -> pd.DataFrame:
    """Candidate US common stocks: columns [ticker, cik, title].

    One ticker per CIK (the shortest, a heuristic for the primary share
    class). EDGAR ordering roughly tracks market cap, so ``max_names``
    keeps the largest companies first.
    """
    df = edgar.ticker_map()
    df = df[df["ticker"].str.match(_COMMON_TICKER)]
    df = df.sort_values("ticker", key=lambda s: s.str.len()).drop_duplicates("cik")
    df = df.sort_index().reset_index(drop=True)
    if max_names:
        df = df.head(max_names)
    return df


def apply_liquidity_filters(
    close: pd.DataFrame, volume: pd.DataFrame, config: ModelConfig
) -> list[str]:
    """Filter to investable names: price, dollar ADV, and history floors."""
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
