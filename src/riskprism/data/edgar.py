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
# Nightly bulk archives: one request for every company. SEC's WAF blocks
# sustained per-company crawling (thousands of API calls), so any build
# with a large universe MUST come through these instead.
BULK_URLS = {
    "facts": "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip",
    "subs": "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip",
}
_BULK_CACHE_PREFIX = {"facts": "facts", "subs": "subs"}

_UA_HELP = (
    "SEC EDGAR requires a User-Agent identifying you with a contact email "
    '(fair-access policy). Set RISKPRISM_EDGAR_UA, e.g. '
    '"my-project you@example.com", or pass user_agent= to EdgarClient.'
)

# Concept fallbacks, tried in order. Chosen for near-universal coverage
# across filers rather than accounting precision — see docs/METHODOLOGY.md.
# ifrs-full entries pick up foreign private issuers (20-F filers).
CONCEPTS = {
    "book_equity": [
        ("us-gaap", "StockholdersEquity"),
        ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
        ("ifrs-full", "Equity"),
    ],
    "total_assets": [("us-gaap", "Assets"), ("ifrs-full", "Assets")],
    "total_liabilities": [("us-gaap", "Liabilities"), ("ifrs-full", "Liabilities")],
    "net_income": [("us-gaap", "NetIncomeLoss"), ("us-gaap", "ProfitLoss"),
                   ("ifrs-full", "ProfitLoss")],
    "shares_out": [
        ("dei", "EntityCommonStockSharesOutstanding"),
        ("us-gaap", "CommonStockSharesOutstanding"),
        ("ifrs-full", "NumberOfSharesOutstanding"),
    ],
    "op_cashflow": [
        ("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
        ("us-gaap", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
        ("ifrs-full", "CashFlowsFromUsedInOperatingActivities"),
    ],
    "revenues": [
        ("us-gaap", "Revenues"),
        ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
        ("us-gaap", "RevenueFromContractWithCustomerIncludingAssessedTax"),
        ("us-gaap", "SalesRevenueNet"),
        ("ifrs-full", "Revenue"),
    ],
    "gross_profit": [("us-gaap", "GrossProfit"), ("ifrs-full", "GrossProfit")],
    "cost_of_revenue": [
        ("us-gaap", "CostOfRevenue"),
        ("us-gaap", "CostOfGoodsAndServicesSold"),
        ("us-gaap", "CostOfGoodsSold"),
        ("ifrs-full", "CostOfSales"),
    ],
}

# Flow (duration) concepts: only full-fiscal-year frames are kept, so
# point-in-time lookups never mix quarterly and annual magnitudes.
_DURATION_CONCEPTS = {"net_income", "op_cashflow", "revenues",
                      "gross_profit", "cost_of_revenue"}


def default_cache_dir() -> Path:
    root = os.environ.get("RISKPRISM_CACHE")
    if root:
        return Path(root)
    return Path.home() / ".cache" / "riskprism"


class EdgarClient:
    """Throttled, disk-cached EDGAR JSON client."""

    def __init__(self, user_agent: str | None = None, cache_dir: Path | None = None,
                 min_interval: float = 0.25, cache_max_age_days: float = 7.0):
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
        self._consecutive_blocks = 0

    def _get_json(self, url: str, cache_key: str) -> dict:
        path = self.cache_dir / f"{cache_key}.json"
        if path.exists():
            age_days = (time.time() - path.stat().st_mtime) / 86400
            if age_days < self.cache_max_age_days:
                return json.loads(path.read_text())
        if self._consecutive_blocks >= 8:
            # WAF-blocked: cached data — even stale — always beats no data
            if path.exists():
                return json.loads(path.read_text())
            raise RuntimeError(
                "EDGAR has rejected 8 requests in a row — this IP appears to be "
                "blocked by SEC's WAF. Aborting instead of burning hours on retries."
            )
        last_exc: Exception | None = None
        for attempt in range(3):
            wait = self.min_interval - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            try:
                resp = self.session.get(url, timeout=30)
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(2 * (attempt + 1))
                continue
            self._last_request = time.monotonic()
            if resp.status_code in (403, 429):
                # SEC's WAF intermittently blocks datacenter IPs (CI runners);
                # back off and retry before giving up.
                last_exc = requests.HTTPError(
                    f"EDGAR rejected the request ({resp.status_code}) for {url}. "
                    f"{_UA_HELP} Datacenter/CI IPs are sometimes blocked outright.",
                    response=resp,
                )
                time.sleep(3 * (attempt + 1))
                continue
            resp.raise_for_status()
            self._consecutive_blocks = 0
            path.write_text(resp.text)
            return resp.json()
        self._consecutive_blocks += 1
        if path.exists():  # stale cache beats no data
            return json.loads(path.read_text())
        raise last_exc

    def ticker_map(self) -> pd.DataFrame:
        """All EDGAR-registered tickers: columns [ticker, cik, title].

        The live file sits on www.sec.gov, whose WAF blocks many CI/cloud
        IPs. On failure, fall back to the snapshot bundled with the
        package (public domain; refreshed with releases) so automated
        builds always have a universe to start from.
        """
        try:
            raw = self._get_json(TICKER_URL, "company_tickers")
        except (requests.RequestException, RuntimeError) as exc:
            import gzip
            from importlib import resources

            print(f"[riskprism] live ticker file unavailable ({exc}); "
                  "using bundled snapshot")
            blob = (resources.files("riskprism.data")
                    / "company_tickers_snapshot.json.gz").read_bytes()
            raw = json.loads(gzip.decompress(blob))
        rows = [
            {"ticker": v["ticker"].upper(), "cik": int(v["cik_str"]), "title": v["title"]}
            for v in raw.values()
        ]
        return pd.DataFrame(rows)

    @property
    def is_blocked(self) -> bool:
        """True once the circuit breaker has tripped (WAF rejecting this IP)."""
        return self._consecutive_blocks >= 8

    def company_facts(self, cik: int) -> dict | None:
        try:
            return self._get_json(FACTS_URL.format(cik=cik), f"facts_{cik:010d}")
        except (requests.HTTPError, RuntimeError):
            return None

    def sic_code(self, cik: int) -> int | None:
        try:
            meta = self._get_json(SUBMISSIONS_URL.format(cik=cik), f"subs_{cik:010d}")
        except (requests.HTTPError, RuntimeError):
            return None
        sic = meta.get("sic")
        try:
            return int(sic) if sic else None
        except (TypeError, ValueError):
            return None


    # ---- bulk archives ---------------------------------------------------

    def _extract_bulk(self, zip_path, ciks: list[int], kind: str) -> int:
        """Extract the CIK members we need from a bulk zip into the cache."""
        import zipfile

        prefix = _BULK_CACHE_PREFIX[kind]
        extracted = 0
        with zipfile.ZipFile(zip_path) as z:
            members = set(z.namelist())
            for cik in ciks:
                member = f"CIK{cik:010d}.json"
                if member not in members:
                    continue
                (self.cache_dir / f"{prefix}_{cik:010d}.json").write_bytes(z.read(member))
                extracted += 1
        return extracted

    def bulk_prefetch(self, ciks: list[int], min_missing: int = 200,
                      verbose: bool = True) -> None:
        """Populate the cache for many CIKs from SEC's nightly bulk zips.

        One HTTP request per archive instead of one per company — the
        access pattern SEC's fair-access policy is designed around.
        Per-company crawling at universe scale reliably trips their WAF
        and gets the IP blocked. No-op when the cache is mostly warm.
        """
        for kind, url in BULK_URLS.items():
            prefix = _BULK_CACHE_PREFIX[kind]
            missing = [
                cik for cik in ciks
                if not (self.cache_dir / f"{prefix}_{cik:010d}.json").exists()
            ]
            if len(missing) < min_missing:
                continue
            zip_path = self.cache_dir / f"bulk_{kind}.zip"
            age_ok = (zip_path.exists()
                      and (time.time() - zip_path.stat().st_mtime) / 86400
                      < self.cache_max_age_days)
            if not age_ok:
                if verbose:
                    print(f"[riskprism] downloading SEC bulk {kind} archive "
                          f"({len(missing)} companies missing from cache)…")
                with self.session.get(url, stream=True, timeout=120) as resp:
                    if resp.status_code in (403, 429):
                        raise requests.HTTPError(
                            f"SEC bulk archive rejected ({resp.status_code}). {_UA_HELP}",
                            response=resp)
                    resp.raise_for_status()
                    tmp = zip_path.with_suffix(".part")
                    with open(tmp, "wb") as fh:
                        for chunk in resp.iter_content(chunk_size=1 << 20):
                            fh.write(chunk)
                    tmp.rename(zip_path)
            n = self._extract_bulk(zip_path, missing, kind)
            if verbose:
                print(f"[riskprism] bulk {kind}: extracted {n}/{len(missing)} missing companies")


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

    def to_frame(self, ticker: str) -> pd.DataFrame:
        frames = []
        for field, df in self.series.items():
            if df.empty:
                continue
            f = df.copy()
            f["ticker"], f["field"] = ticker, field
            frames.append(f)
        if not frames:
            return pd.DataFrame(columns=["ticker", "field", "end", "filed", "val"])
        return pd.concat(frames)[["ticker", "field", "end", "filed", "val"]]


def store_to_frame(store: dict[str, "Fundamentals"]) -> pd.DataFrame:
    """Serialize a {ticker: Fundamentals} store into one long DataFrame.

    Distilled from EDGAR XBRL (public domain), so this ships inside the
    model artifacts — which lets CI builds run even when SEC's WAF blocks
    the runner's IP: fundamentals fall back to the prior release's store.
    """
    frames = [f.to_frame(t) for t, f in store.items()]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["ticker", "field", "end", "filed", "val"])
    return pd.concat(frames, ignore_index=True)


def store_from_frame(df: pd.DataFrame) -> dict[str, "Fundamentals"]:
    empty = pd.DataFrame(columns=["end", "filed", "val"])
    store: dict[str, Fundamentals] = {}
    for ticker, tdf in df.groupby("ticker"):
        series = {
            field: fdf[["end", "filed", "val"]].sort_values(["filed", "end"]).reset_index(drop=True)
            for field, fdf in tdf.groupby("field")
        }
        for field in CONCEPTS:  # fields with no filings round-trip as empty
            series.setdefault(field, empty)
        store[str(ticker)] = Fundamentals(series)
    return store
