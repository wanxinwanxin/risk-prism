import numpy as np
import pandas as pd
import pytest

from riskprism.data.holdings import FundHoldings, normalize_name, parse_nport
from riskprism.lookthrough import (
    FundCoverageError,
    expand_fund,
    fund_risk,
    portfolio_risk_lookthrough,
)
from riskprism.risk import RiskModel


@pytest.fixture
def model():
    tickers = ["AAPL", "MSFT", "XOM", "JPM"]
    factors = ["market", "size", "value"]
    X = pd.DataFrame(
        [[1.0, 1.5, -0.5], [1.0, 1.4, -0.3], [1.0, 0.8, 0.9], [1.0, 0.9, 0.6]],
        index=tickers, columns=factors,
    )
    F = pd.DataFrame(np.eye(3) * 0.02, index=factors, columns=factors)
    spec = pd.Series([0.20, 0.18, 0.25, 0.22], index=tickers)
    return RiskModel(X, F, spec, meta={"model_version": "test-0.1"})


class FakeClient:
    """In-memory stand-in for HoldingsClient: no network, no cache."""

    def __init__(self, funds: dict[str, FundHoldings] | None = None,
                 aliases: dict[str, str] | None = None):
        self.funds = funds or {}
        self.aliases = aliases or {}

    def fund_locator(self, ticker):
        return "S000000001" if ticker.upper() in self.funds else None

    def holdings(self, ticker):
        t = ticker.upper()
        if t not in self.funds:
            raise KeyError(f"{t} is not a registered fund share class on EDGAR")
        return self.funds[t]

    def alias_for(self, ticker, universe):
        alias = self.aliases.get(ticker.upper())
        return alias if alias in universe else None


def _fund(ticker, weights, cash=0.0, other=0.0, unresolved=0.0):
    return FundHoldings(ticker=ticker, name=f"{ticker} fund", as_of="2026-06-30",
                        weights=weights, cash_weight=cash, other_weight=other,
                        unresolved_weight=unresolved)


# ---- expand_fund -----------------------------------------------------------


def test_expand_renormalizes_covered_sleeve(model):
    h = _fund("ETF", {"AAPL": 0.60, "XOM": 0.30, "ZZZZ": 0.05},
              cash=0.02, unresolved=0.03)
    client = FakeClient({"ETF": h})
    weights, diag = expand_fund(model, h, client)
    # covered 0.90 of gross 1.00; cash counts as covered
    assert diag["holdings_coverage"] == pytest.approx(0.92)
    # covered sleeve scales to the whole risky sleeve (1 - cash)
    scale = (1.0 - 0.02) / 0.90
    assert diag["renormalization"] == pytest.approx(scale)
    assert weights["AAPL"] == pytest.approx(0.60 * scale)
    assert weights["XOM"] == pytest.approx(0.30 * scale)
    assert "ZZZZ" not in weights
    assert diag["uncovered_examples"] == ["ZZZZ"]


def test_expand_refuses_majorly_uncovered(model):
    h = _fund("BOND", {"AAPL": 0.10}, other=0.90)
    with pytest.raises(FundCoverageError) as exc:
        expand_fund(model, h, FakeClient({"BOND": h}))
    assert exc.value.coverage == pytest.approx(0.10)
    assert exc.value.detail["non_equity_weight"] == pytest.approx(0.90)


def test_expand_respects_min_coverage_param(model):
    h = _fund("ETF", {"AAPL": 0.60}, other=0.40)
    client = FakeClient({"ETF": h})
    weights, _ = expand_fund(model, h, client, min_coverage=0.5)
    assert weights["AAPL"] == pytest.approx(1.0)  # 0.60 scaled to the full sleeve
    with pytest.raises(FundCoverageError):
        expand_fund(model, h, client, min_coverage=0.7)


def test_expand_recurses_into_sub_funds(model):
    child = _fund("SUB", {"MSFT": 0.99}, cash=0.01)
    parent = _fund("TOP", {"AAPL": 0.50, "SUB": 0.50})
    client = FakeClient({"SUB": child, "TOP": parent})
    weights, diag = expand_fund(model, parent, client)
    assert diag["holdings_coverage"] == pytest.approx(1.0)
    assert diag["sub_funds"]["SUB"] == pytest.approx(1.0)
    assert weights["AAPL"] == pytest.approx(0.50)
    # 0.50 x (child's 0.99 scaled to its 0.99 risky sleeve) = 0.495
    assert weights["MSFT"] == pytest.approx(0.50 * 0.99)


def test_expand_survives_self_referencing_fund(model):
    h = _fund("LOOP", {"AAPL": 0.50, "LOOP": 0.50})
    weights, diag = expand_fund(model, h, FakeClient({"LOOP": h}))
    # the self-position counts as uncovered, never as infinite recursion
    assert diag["holdings_coverage"] == pytest.approx(0.5)
    assert weights == {"AAPL": pytest.approx(1.0)}


def test_expand_uses_share_class_alias(model):
    h = _fund("ETF", {"AAPL2": 1.0})
    client = FakeClient({"ETF": h}, aliases={"AAPL2": "AAPL"})
    weights, diag = expand_fund(model, h, client)
    assert weights == {"AAPL": pytest.approx(1.0)}
    assert diag["holdings_coverage"] == pytest.approx(1.0)


def test_cash_only_fund_has_zero_risk(model):
    h = _fund("MMKT", {}, cash=1.0)
    weights, diag = expand_fund(model, h, FakeClient({"MMKT": h}))
    assert weights == {}
    assert diag["holdings_coverage"] == pytest.approx(1.0)
    assert model.portfolio_risk(weights)["total_vol"] == 0.0


# ---- fund_risk -------------------------------------------------------------


def test_fund_risk_report_shape(model):
    h = _fund("ETF", {"AAPL": 0.50, "MSFT": 0.48}, cash=0.02)
    r = fund_risk(model, "etf", client=FakeClient({"ETF": h}))
    assert r["ticker"] == "ETF"
    assert r["fund"]["as_of"] == "2026-06-30"
    assert r["total_vol"] == pytest.approx(
        (r["factor_vol"] ** 2 + r["specific_vol"] ** 2) ** 0.5)
    # cash drag: the weight vector sums below 1
    assert sum(r["factor_exposures"].values()) < 3.0


def test_fund_risk_unknown_ticker_raises(model):
    with pytest.raises(KeyError):
        fund_risk(model, "ZZZZ", client=FakeClient())


# ---- portfolio_risk_lookthrough --------------------------------------------


def test_portfolio_expands_funds_and_keeps_stocks(model):
    h = _fund("ETF", {"MSFT": 1.0})
    client = FakeClient({"ETF": h})
    r = portfolio_risk_lookthrough(model, {"AAPL": 0.4, "ETF": 0.6}, client=client)
    direct = model.portfolio_risk({"AAPL": 0.4, "MSFT": 0.6})
    assert r["total_vol"] == pytest.approx(direct["total_vol"])
    assert r["lookthrough"]["funds"]["ETF"]["holdings_coverage"] == pytest.approx(1.0)
    assert r["coverage_ratio"] == pytest.approx(1.0)


def test_portfolio_refused_fund_stays_uncovered_with_note(model):
    h = _fund("BOND", {"AAPL": 0.05}, other=0.95)
    client = FakeClient({"BOND": h})
    r = portfolio_risk_lookthrough(model, {"AAPL": 0.8, "BOND": 0.2}, client=client)
    assert "covers only" in r["lookthrough"]["notes"]["BOND"]
    assert r["uncovered_tickers"] == ["BOND"]
    assert r["coverage_ratio"] == pytest.approx(0.8)


def test_portfolio_short_fund_position(model):
    h = _fund("ETF", {"MSFT": 1.0})
    r = portfolio_risk_lookthrough(model, {"AAPL": 1.0, "ETF": -0.5},
                                   client=FakeClient({"ETF": h}))
    direct = model.portfolio_risk({"AAPL": 1.0, "MSFT": -0.5})
    assert r["total_vol"] == pytest.approx(direct["total_vol"])


def test_portfolio_plain_when_no_funds(model):
    r = portfolio_risk_lookthrough(model, {"AAPL": 1.0}, client=FakeClient())
    assert "lookthrough" not in r
    assert r["total_vol"] == pytest.approx(
        model.portfolio_risk({"AAPL": 1.0})["total_vol"])


# ---- N-PORT parsing --------------------------------------------------------

_NPORT_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<edgarSubmission xmlns="http://www.sec.gov/edgar/nport">
  <formData>
    <genInfo>
      <seriesName>N/A</seriesName>
      <regName>Demo Trust</regName>
      <repPdDate>2026-06-30</repPdDate>
    </genInfo>
    <invstOrSecs>
      <invstOrSec>
        <name>Apple, Inc.</name>
        <cusip>037833100</cusip>
        <pctVal>60.0</pctVal>
        <assetCat>EC</assetCat>
        <issuerCat>CORP</issuerCat>
      </invstOrSec>
      <invstOrSec>
        <name>Cash Fund</name>
        <cusip>000000000</cusip>
        <pctVal>2.0</pctVal>
        <assetCat>STIV</assetCat>
        <issuerCat>RF</issuerCat>
      </invstOrSec>
      <invstOrSec>
        <name>Some Sub Fund ETF</name>
        <cusip>111111111</cusip>
        <pctVal>38.0</pctVal>
        <issuerCat>RF</issuerCat>
      </invstOrSec>
    </invstOrSecs>
  </formData>
</edgarSubmission>"""


def test_parse_nport_fixture():
    info, rows = parse_nport(_NPORT_XML)
    assert info["series_name"] == "Demo Trust"  # N/A falls back to regName
    assert info["as_of"] == "2026-06-30"
    assert len(rows) == 3
    assert rows[0]["weight"] == pytest.approx(0.60)
    assert rows[1]["asset_cat"] == "STIV"
    assert rows[2]["asset_cat"] == "" and rows[2]["issuer_cat"] == "RF"


def test_resolve_classifies_and_maps():
    from riskprism.data.holdings import _resolve

    cusips = {"037833100": "AAPL"}
    names = {normalize_name("Some Sub Fund ETF"): "SUB"}
    h = _resolve("DEMO", _NPORT_XML, "http://example/doc.xml", cusips, names)
    assert h.weights == {"AAPL": pytest.approx(0.60), "SUB": pytest.approx(0.38)}
    assert h.cash_weight == pytest.approx(0.02)
    assert h.unresolved_weight == 0.0
    assert h.name == "Demo Trust"


def test_resolve_unmapped_equity_counts_as_unresolved():
    from riskprism.data.holdings import _resolve

    h = _resolve("DEMO", _NPORT_XML, "", {}, {})
    assert h.weights == {}
    assert h.unresolved_weight == pytest.approx(0.98)
    assert "Apple, Inc." in h.unresolved_names


def test_normalize_name():
    assert normalize_name("Exxon Mobil Corp.") == normalize_name("EXXON MOBIL CORP")
    assert normalize_name("Alphabet, Inc. Class A") == "ALPHABET"


def test_fund_holdings_roundtrip():
    h = _fund("ETF", {"AAPL": 0.5}, cash=0.1)
    assert FundHoldings.from_dict(h.to_dict()) == h
