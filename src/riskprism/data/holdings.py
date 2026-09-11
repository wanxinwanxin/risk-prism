"""ETF and fund holdings from SEC N-PORT filings (public domain).

The look-through risk path resolves a fund ticker to its latest filed
portfolio. N-PORT gives each holding's name, CUSIP, weight (pctVal),
and asset category. CUSIPs map to tickers through the SEC
fails-to-deliver files, with a normalized-name match against the EDGAR
ticker registry as the fallback. Measured on IVV (2026-06-30 filing):
the CUSIP map resolves 97.0% of NAV, the name match adds 2.6%, and
0.2% stays unresolved.

Funds file N-PORT within 60 days of each fiscal quarter end, so
holdings lag the market by one to five months. That staleness is
acceptable for broad funds with low turnover, and every result carries
its ``as_of`` date so the caller can judge.
"""

import io
import json
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
import requests

from riskprism.data.edgar import _UA_HELP, TICKER_URL, default_cache_dir
import os

MF_TICKER_URL = "https://www.sec.gov/files/company_tickers_mf.json"
# count=10, first entry taken: browse-edgar drops the newest filing when
# count=1 (measured on CIK 884394, 2026-09-11)
BROWSE_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
    "&CIK={entity}&type=NPORT-P&dateb=&owner=include&count=10&output=atom"
)
FTD_URL = "https://www.sec.gov/files/data/fails-deliver-data/cnsfails{period}.zip"

# N-PORT asset categories treated as cash-like (zero-risk): money-market
# vehicles and repurchase agreements.
CASH_ASSET_CATS = {"STIV", "RA"}
EQUITY_ASSET_CATS = {"EC", "EP"}

_FTD_LOOKBACK_MONTHS = 8  # half-month files probed for the CUSIP map

# Corporate suffix tokens dropped by the name normalizer.
_NAME_NOISE = re.compile(
    r"\b(INC|INCORPORATED|CORP|CORPORATION|CO|COMPANY|PLC|LTD|LIMITED|"
    r"HOLDINGS|HOLDING|GROUP|SA|NV|SE|AG|THE|CL|CLASS|A|B|C|NEW|COM|TRUST)\b"
)


def normalize_name(name: str) -> str:
    """Issuer name reduced to its distinctive tokens, for fuzzy-free matching."""
    s = re.sub(r"[^A-Z0-9 ]", " ", name.upper())
    return " ".join(_NAME_NOISE.sub(" ", s).split())


@dataclass
class FundHoldings:
    """One fund's latest filed portfolio, resolved to tickers.

    ``weights`` are signed decimals of NAV (0.07 = 7%). ``cash_weight``
    is the cash-like sleeve (zero-risk). ``other_weight`` is the gross
    weight of non-equity, non-cash positions (bonds, derivatives).
    ``unresolved_weight`` is the gross equity weight with no ticker.
    """

    ticker: str
    name: str
    as_of: str
    weights: dict[str, float]
    cash_weight: float = 0.0
    other_weight: float = 0.0
    unresolved_weight: float = 0.0
    unresolved_names: list[str] = field(default_factory=list)
    source: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FundHoldings":
        return cls(**d)


class HoldingsClient:
    """Throttled, disk-cached client for N-PORT holdings resolution."""

    def __init__(self, user_agent: str | None = None, cache_dir: Path | None = None,
                 min_interval: float = 0.25, cache_max_age_days: float = 1.0):
        ua = user_agent or os.environ.get("RISKPRISM_EDGAR_UA")
        if not ua:
            raise ValueError(_UA_HELP)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = ua
        self.cache_dir = Path(cache_dir) if cache_dir else default_cache_dir() / "holdings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_max_age_days = cache_max_age_days
        self.min_interval = min_interval
        self._last_request = 0.0
        self._series_map: dict[str, str] | None = None
        self._cik_map: dict[str, str] | None = None
        self._cusip_map: dict[str, str] | None = None
        self._name_map: dict[str, str] | None = None

    # ---- transport -------------------------------------------------------

    def _get_bytes(self, url: str, cache_key: str, max_age_days: float,
                   ok_status: tuple = (200,)) -> bytes | None:
        """Fetch with a disk cache. Returns None on a non-OK status."""
        path = self.cache_dir / cache_key
        if path.exists():
            age = (time.time() - path.stat().st_mtime) / 86400
            if age < max_age_days:
                data = path.read_bytes()
                return None if data == b"" else data  # b"" caches a miss
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        resp = self.session.get(url, timeout=60)
        self._last_request = time.monotonic()
        if resp.status_code == 404 or resp.status_code not in ok_status:
            if resp.status_code not in (403, 429):
                path.write_bytes(b"")  # cache the miss, but never a block
            return None
        path.write_bytes(resp.content)
        return resp.content

    # ---- fund location ---------------------------------------------------

    def _load_locators(self) -> None:
        if self._series_map is not None:
            return
        raw = self._get_bytes(MF_TICKER_URL, "company_tickers_mf.json", 7.0)
        series: dict[str, str] = {}
        if raw:
            mf = json.loads(raw)
            idx = {f: i for i, f in enumerate(mf["fields"])}
            for row in mf["data"]:
                sym = str(row[idx["symbol"]] or "").upper()
                sid = str(row[idx["seriesId"]] or "")
                if sym and sid.startswith("S") and sym not in series:
                    series[sym] = sid
        ciks: dict[str, str] = {}
        cik_tickers: dict[str, list[str]] = {}
        raw = self._get_bytes(TICKER_URL, "company_tickers.json", 7.0)
        titles: dict[str, str] = {}
        fundlike: set[str] = set()
        if raw:
            for v in json.loads(raw).values():
                t = v["ticker"].upper()
                cik = f"{int(v['cik_str']):010d}"
                ciks.setdefault(t, cik)
                cik_tickers.setdefault(cik, []).append(t)
                title = v["title"].upper()
                if "ETF" in title or " FUND" in title or "TRUST" in title:
                    fundlike.add(t)
                n = normalize_name(v["title"])
                if n and n not in titles:
                    titles[n] = t
        self._series_map, self._cik_map, self._name_map = series, ciks, titles
        self._cik_tickers = cik_tickers
        self._fundlike = fundlike

    def alias_for(self, ticker: str, universe) -> str | None:
        """Another share class of the same issuer that ``universe`` holds
        (GOOG -> GOOGL, FOX -> FOXA). The model keeps one ticker per CIK,
        so this recovers holdings filed under the sibling class."""
        self._load_locators()
        cik = self._cik_map.get(ticker.upper())
        if cik is None:
            return None
        for t in self._cik_tickers.get(cik, []):
            if t != ticker.upper() and t in universe:
                return t
        return None

    def fund_locator(self, ticker: str) -> str | None:
        """EDGAR entity for a fund ticker: a series id, or a trust CIK for
        single-fund trusts (SPY, QQQ, DIA). None when the ticker is not a
        registered fund share class."""
        self._load_locators()
        t = ticker.upper()
        loc = self._series_map.get(t)
        if loc:
            return loc
        # Single-fund trusts register their ticker directly; require a
        # fund-like title so common stocks never route here.
        if t in self._fundlike:
            return self._cik_map.get(t)
        return None

    # ---- N-PORT ----------------------------------------------------------

    def _latest_nport(self, entity: str) -> tuple[bytes, str] | None:
        """Latest NPORT-P primary document for a series id or CIK."""
        atom = self._get_bytes(BROWSE_URL.format(entity=entity),
                               f"atom_{entity}.xml", self.cache_max_age_days)
        if not atom:
            return None
        text = atom.decode("utf-8", "replace")
        m = re.search(r"<filing-href>([^<]+)-index.htm", text)
        if not m:
            return None
        base = m.group(1)
        folder = base.rsplit("/", 1)[0]
        accession = base.rsplit("/", 1)[1]
        url = f"{folder}/primary_doc.xml"
        doc = self._get_bytes(url, f"nport_{accession}.xml", 365.0)
        if doc is None:
            # Some filers name the primary document differently; ask the
            # accession folder's index for the first XML document.
            idx = self._get_bytes(f"{folder}/index.json",
                                  f"nportidx_{accession}.json", 365.0)
            if not idx:
                return None
            items = json.loads(idx).get("directory", {}).get("item", [])
            names = [i["name"] for i in items
                     if i["name"].endswith(".xml") and "primary" in i["name"]]
            if not names:
                return None
            url = f"{folder}/{names[0]}"
            doc = self._get_bytes(url, f"nport_{accession}.xml", 365.0)
        return (doc, url) if doc else None

    # ---- CUSIP and name maps ---------------------------------------------

    def _ftd_periods(self) -> list[str]:
        end = pd.Timestamp.today()
        months = pd.period_range(end=end.to_period("M"),
                                 periods=_FTD_LOOKBACK_MONTHS, freq="M")
        return [f"{m.strftime('%Y%m')}{half}" for m in months for half in "ab"]

    def cusip_map(self) -> dict[str, str]:
        """CUSIP -> EDGAR ticker, aggregated from recent fails-to-deliver
        files (newer files win). Symbols are normalized to EDGAR's dashed
        class-share form (BRKB -> BRK-B)."""
        if self._cusip_map is not None:
            return self._cusip_map
        combined = self.cache_dir / "cusip_map.json"
        if combined.exists() and (time.time() - combined.stat().st_mtime) / 86400 < 7.0:
            self._cusip_map = json.loads(combined.read_text())
            return self._cusip_map
        pairs: dict[str, str] = {}
        for period in self._ftd_periods():  # oldest first: newer files win
            raw = self._get_bytes(FTD_URL.format(period=period),
                                  f"ftd_{period}.zip", 365.0)
            if not raw:
                continue
            try:
                z = zipfile.ZipFile(io.BytesIO(raw))
                text = z.read(z.namelist()[0]).decode("latin-1")
            except (zipfile.BadZipFile, IndexError):
                continue
            for line in text.splitlines()[1:]:
                p = line.split("|")
                if len(p) >= 3 and p[1] and p[2] and p[2] != ".":
                    pairs[p[1]] = p[2].upper()
        self._load_locators()
        # Symbols the EDGAR registries know: companies and fund share
        # classes. The fails files carry placeholder symbols for some
        # corporate-action CUSIPs (XOMXXXX); keep only known symbols, and
        # dropped entries fall through to the name match.
        known = set(self._cik_map or {}) | set(self._series_map or {})
        clean: dict[str, str] = {}
        for cusip, sym in pairs.items():
            if sym in known:
                clean[cusip] = sym
            elif len(sym) > 1 and f"{sym[:-1]}-{sym[-1]}" in known:
                clean[cusip] = f"{sym[:-1]}-{sym[-1]}"
        combined.write_text(json.dumps(clean))
        self._cusip_map = clean
        return clean

    def name_map(self) -> dict[str, str]:
        """Normalized issuer name -> EDGAR ticker (first listing wins)."""
        self._load_locators()
        return self._name_map or {}

    # ---- public entry point ----------------------------------------------

    def holdings(self, ticker: str) -> FundHoldings:
        """Latest filed holdings for a fund ticker, resolved to tickers.

        Raises KeyError when the ticker does not locate a fund or the
        fund has no N-PORT filing.
        """
        t = ticker.upper()
        cached = self.cache_dir / f"resolved_{t}.json"
        if cached.exists():
            age = (time.time() - cached.stat().st_mtime) / 86400
            if age < self.cache_max_age_days:
                return FundHoldings.from_dict(json.loads(cached.read_text()))
        entity = self.fund_locator(t)
        if entity is None:
            raise KeyError(f"{t} is not a registered fund share class on EDGAR")
        doc = self._latest_nport(entity)
        if doc is None:
            raise KeyError(f"{t}: no N-PORT filing found for EDGAR entity {entity}")
        xml, url = doc
        result = _resolve(t, xml, url, self.cusip_map(), self.name_map())
        cached.write_text(json.dumps(result.to_dict()))
        return result


def parse_nport(xml: bytes) -> tuple[dict, list[dict]]:
    """N-PORT primary document -> (fund info, holdings rows)."""
    root = ET.fromstring(xml)

    def txt(el: ET.Element | None) -> str:
        return (el.text or "").strip() if el is not None else ""

    series_name = txt(root.find(".//{*}seriesName"))
    if series_name in ("", "N/A"):  # unit investment trusts file no series
        series_name = txt(root.find(".//{*}regName"))
    info = {
        "series_name": series_name,
        "as_of": txt(root.find(".//{*}repPdDate")),
    }
    rows = []
    for h in root.iter():
        if not h.tag.endswith("invstOrSec"):
            continue
        d = {re.sub(r"\{.*\}", "", c.tag): (c.text or "").strip() for c in h}
        try:
            pct = float(d.get("pctVal") or 0.0)
        except ValueError:
            continue
        rows.append({
            "name": d.get("name", ""),
            "cusip": d.get("cusip", ""),
            "weight": pct / 100.0,
            "asset_cat": d.get("assetCat", ""),
            "issuer_cat": d.get("issuerCat", ""),
        })
    return info, rows


def _resolve(ticker: str, xml: bytes, source: str,
             cusips: dict[str, str], names: dict[str, str]) -> FundHoldings:
    info, rows = parse_nport(xml)
    weights: dict[str, float] = {}
    cash = other = unresolved = 0.0
    unresolved_names: list[tuple[float, str]] = []
    for r in rows:
        w = r["weight"]
        if w == 0.0:
            continue
        cat = r["asset_cat"]
        # Fund-of-fund constituents file with an empty assetCat and
        # issuerCat RF; resolve them so look-through can recurse.
        if cat in CASH_ASSET_CATS:
            cash += w
        elif cat in EQUITY_ASSET_CATS or (not cat and r["issuer_cat"] == "RF"):
            sym = cusips.get(r["cusip"]) or names.get(normalize_name(r["name"]))
            if sym:
                weights[sym] = weights.get(sym, 0.0) + w
            else:
                unresolved += abs(w)
                unresolved_names.append((abs(w), r["name"]))
        else:
            other += abs(w)
    unresolved_names.sort(reverse=True)
    return FundHoldings(
        ticker=ticker,
        name=info["series_name"],
        as_of=info["as_of"],
        weights=weights,
        cash_weight=cash,
        other_weight=other,
        unresolved_weight=unresolved,
        unresolved_names=[n for _, n in unresolved_names[:10]],
        source=source,
    )
