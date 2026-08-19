"""SEC EDGAR data client — fundamentals, tickers, and SIC codes.

EDGAR XBRL data is public domain, which is what makes an open-source
fundamental risk model with broad US coverage legally distributable.

SEC fair-access policy: max 10 requests/second and a descriptive
User-Agent header are required. Set RISKPRISM_EDGAR_UA to identify yourself.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

_UA_HELP = (
    "SEC EDGAR requires a User-Agent identifying you with a contact email "
    '(fair-access policy). Set RISKPRISM_EDGAR_UA, e.g. '
    '"my-project you@example.com", or pass user_agent= to EdgarClient.'
)

# Concept fallbacks, tried in order. Chosen for near-universal coverage
# across filers rather than accounting precision — see docs/METHODOLOGY.md.
CONCEPTS = {
    "book_equity": [
        ("us-gaap", "StockholdersEquity"),
        ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    ],
    "total_assets": [("us-gaap", "Assets")],
    "total_liabilities": [("us-gaap", "Liabilities")],
    "net_income": [("us-gaap", "NetIncomeLoss"), ("us-gaap", "ProfitLoss")],
    "shares_out": [
        ("dei", "EntityCommonStockSharesOutstanding"),
        ("us-gaap", "CommonStockSharesOutstanding"),
    ],
}

_DURATION_CONCEPTS = {"net_income"}


def default_cache_dir() -> Path:
    root = os.environ.get("RISKPRISM_CACHE")
    if root:
        return Path(root)
    return Path.home() / ".cache" / "riskprism"


class EdgarClient:
    """Throttled, disk-cached EDGAR JSON client."""

    def __init__(self, user_agent: str | None = None, cache_dir: Path | None = None,
                 min_interval: float = 0.12, cache_max_age_days: float = 7.0):
        ua = user_agent or os.environ.get("RISKPRISM_EDGAR_UA")
        if not ua:
            raise ValueError(_UA_HELP)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = ua
        self.cache_dir = Path(cache_dir) if cache_dir else default_cache_dir() / "edgar"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self.cache_max_age_days = cache_max_age_days
        self._last_request = 0.0

    def _get_json(self, url: str, cache_key: str) -> dict:
        path = self.cache_dir / f"{cache_key}.json"
        if path.exists():
            age_days = (time.time() - path.stat().st_mtime) / 86400
            if age_days < self.cache_max_age_days:
                return json.loads(path.read_text())
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        resp = self.session.get(url, timeout=30)
        self._last_request = time.monotonic()
        if resp.status_code == 403:
            raise requests.HTTPError(f"EDGAR rejected the request (403). {_UA_HELP}",
                                     response=resp)
        resp.raise_for_status()
        path.write_text(resp.text)
        return resp.json()

    def ticker_map(self) -> pd.DataFrame:
        """All EDGAR-registered tickers: columns [ticker, cik, title]."""
        raw = self._get_json(TICKER_URL, "company_tickers")
        rows = [
            {"ticker": v["ticker"].upper(), "cik": int(v["cik_str"]), "title": v["title"]}
            for v in raw.values()
        ]
        return pd.DataFrame(rows)

    def company_facts(self, cik: int) -> dict | None:
        try:
            return self._get_json(FACTS_URL.format(cik=cik), f"facts_{cik:010d}")
        except requests.HTTPError:
            return None

    def sic_code(self, cik: int) -> int | None:
        try:
            meta = self._get_json(SUBMISSIONS_URL.format(cik=cik), f"subs_{cik:010d}")
        except requests.HTTPError:
            return None
        sic = meta.get("sic")
        try:
            return int(sic) if sic else None
        except (TypeError, ValueError):
            return None


def concept_series(facts: dict, taxonomy: str, tag: str, annual_only: bool = False) -> pd.DataFrame:
    """Point-in-time series for one XBRL concept.

    Returns columns [end, filed, val] sorted by filed date. ``filed`` is
    the date the value became publicly known — the as-of lookups use it,
    not ``end``, to avoid lookahead bias.
    """
    node = facts.get("facts", {}).get(taxonomy, {}).get(tag)
    if not node:
        return pd.DataFrame(columns=["end", "filed", "val"])
    rows = []
    for unit_entries in node.get("units", {}).values():
        for e in unit_entries:
            if "val" not in e or "end" not in e or "filed" not in e:
                continue
            if annual_only:
                if e.get("fp") != "FY":
                    continue
                start = e.get("start")
                if start:
                    span = (pd.Timestamp(e["end"]) - pd.Timestamp(start)).days
                    if not (300 <= span <= 400):
                        continue
            rows.append({"end": pd.Timestamp(e["end"]),
                         "filed": pd.Timestamp(e["filed"]),
                         "val": float(e["val"])})
    if not rows:
        return pd.DataFrame(columns=["end", "filed", "val"])
    df = pd.DataFrame(rows).drop_duplicates(subset=["end", "filed"])
    return df.sort_values(["filed", "end"]).reset_index(drop=True)


def latest_asof(series: pd.DataFrame, as_of: pd.Timestamp) -> float:
    """Most recent period-end value filed on or before ``as_of``."""
    if series.empty:
        return np.nan
    known = series[series["filed"] <= as_of]
    if known.empty:
        return np.nan
    latest_end = known["end"].max()
    return float(known[known["end"] == latest_end].iloc[-1]["val"])


class Fundamentals:
    """Point-in-time fundamental store for a single company."""

    def __init__(self, series: dict[str, pd.DataFrame]):
        self.series = series

    @classmethod
    def from_facts(cls, facts: dict) -> "Fundamentals":
        series = {}
        for field, candidates in CONCEPTS.items():
            annual_only = field in _DURATION_CONCEPTS
            df = pd.DataFrame(columns=["end", "filed", "val"])
            for taxonomy, tag in candidates:
                df = concept_series(facts, taxonomy, tag, annual_only=annual_only)
                if not df.empty:
                    break
            series[field] = df
        return cls(series)

    def asof(self, date: pd.Timestamp) -> dict[str, float]:
        return {field: latest_asof(df, date) for field, df in self.series.items()}
