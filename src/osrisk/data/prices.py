"""Pluggable daily price providers.

The repo distributes derived model artifacts, not raw prices. To rebuild
the model yourself, bring a provider: Stooq works without a key; Tiingo
needs TIINGO_API_KEY. Check each provider's terms for your use case.
"""

import io
import os
from pathlib import Path
from typing import Protocol

import pandas as pd
import requests

from osrisk.data.edgar import default_cache_dir


class PriceProvider(Protocol):
    name: str

    def get_daily(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Daily bars indexed by date with columns [close, volume].

        ``close`` must be split-adjusted (dividend adjustment preferred
        when available). Empty DataFrame when the ticker is unknown.
        """
        ...


class StooqProvider:
    """Free, keyless daily data from stooq.com (US tickers as ``xxx.us``)."""

    name = "stooq"
    BASE = "https://stooq.com/q/d/l/"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "osrisk/0.1"

    def get_daily(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        params = {
            "s": f"{ticker.lower().replace('.', '-')}.us",
            "i": "d",
            "d1": start.strftime("%Y%m%d"),
            "d2": end.strftime("%Y%m%d"),
        }
        resp = self.session.get(self.BASE, params=params, timeout=30)
        resp.raise_for_status()
        text = resp.text
        if not text.lstrip().startswith("Date"):
            return pd.DataFrame(columns=["close", "volume"])
        df = pd.read_csv(io.StringIO(text), parse_dates=["Date"])
        if df.empty or "Close" not in df:
            return pd.DataFrame(columns=["close", "volume"])
        out = df.rename(columns={"Date": "date", "Close": "close", "Volume": "volume"})
        if "volume" not in out:
            out["volume"] = float("nan")
        return out.set_index("date")[["close", "volume"]].sort_index()


class TiingoProvider:
    """Tiingo daily prices (adjusted); requires TIINGO_API_KEY."""

    name = "tiingo"
    BASE = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("TIINGO_API_KEY")
        if not self.api_key:
            raise ValueError("TiingoProvider requires TIINGO_API_KEY")
        self.session = requests.Session()

    def get_daily(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        resp = self.session.get(
            self.BASE.format(ticker=ticker.lower()),
            params={
                "startDate": start.strftime("%Y-%m-%d"),
                "endDate": end.strftime("%Y-%m-%d"),
                "token": self.api_key,
            },
            timeout=30,
        )
        if resp.status_code == 404:
            return pd.DataFrame(columns=["close", "volume"])
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            return pd.DataFrame(columns=["close", "volume"])
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df.rename(columns={"adjClose": "close", "adjVolume": "volume"})
        return df.set_index("date")[["close", "volume"]].sort_index()


_PROVIDERS = {"stooq": StooqProvider, "tiingo": TiingoProvider}


def get_provider(name: str = "stooq") -> PriceProvider:
    if name not in _PROVIDERS:
        raise ValueError(f"Unknown provider {name!r}; choose from {sorted(_PROVIDERS)}")
    return _PROVIDERS[name]()


def load_price_panel(
    tickers: list[str],
    provider: PriceProvider,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path | None = None,
    cache_max_age_days: float = 3.0,
    progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch (close, volume) wide panels for a ticker list, with a parquet cache."""
    import time

    cache = Path(cache_dir) if cache_dir else default_cache_dir() / "prices" / provider.name
    cache.mkdir(parents=True, exist_ok=True)
    closes, volumes = {}, {}
    for i, ticker in enumerate(tickers):
        path = cache / f"{ticker}.parquet"
        df = None
        if path.exists() and (time.time() - path.stat().st_mtime) / 86400 < cache_max_age_days:
            df = pd.read_parquet(path)
        if df is None:
            try:
                df = provider.get_daily(ticker, start, end)
            except requests.RequestException:
                df = pd.DataFrame(columns=["close", "volume"])
            df.to_parquet(path)
        df = df.loc[(df.index >= start) & (df.index <= end)]
        if not df.empty:
            closes[ticker] = df["close"]
            volumes[ticker] = df["volume"]
        if progress and (i + 1) % 100 == 0:
            print(f"prices: {i + 1}/{len(tickers)}")
    close = pd.DataFrame(closes).sort_index()
    volume = pd.DataFrame(volumes).reindex(close.index)
    return close, volume
