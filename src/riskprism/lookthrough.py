"""Look-through risk for ETFs and funds from filed holdings.

A fund's risk is computed from its constituents: the latest N-PORT
holdings (data/holdings.py) resolve to model tickers, and the model's
portfolio math runs on those weights. Coverage policy:

- Cash-like sleeves count as covered with zero risk.
- Constituents the model covers carry the risk. Their weights are
  scaled up so the covered sleeve represents the whole risky sleeve
  (small uncovered residuals are assumed to behave like the covered
  part, pro rata).
- When the model covers less than ``min_coverage`` of the fund
  (default 50%), no estimate is given — the fund is majorly outside
  the model's universe (bond funds, international funds).

Every result reports the holdings date, the coverage ratio, and the
renormalization applied, so the caller can judge the estimate.
"""

from collections.abc import Mapping

from riskprism.data.holdings import FundHoldings, HoldingsClient
from riskprism.risk import RiskModel

MIN_FUND_COVERAGE = 0.5
MAX_FUND_DEPTH = 2  # funds of funds: AOR -> equity ETFs -> stocks

_client: HoldingsClient | None = None


def _default_client() -> HoldingsClient:
    global _client
    if _client is None:
        _client = HoldingsClient()
    return _client


class FundCoverageError(ValueError):
    """The model covers too little of the fund to give an estimate."""

    def __init__(self, ticker: str, coverage: float, min_coverage: float,
                 detail: dict | None = None):
        self.ticker = ticker
        self.coverage = coverage
        self.detail = detail or {}
        super().__init__(
            f"{ticker}: the model covers only {coverage:.0%} of the fund's "
            f"holdings (minimum {min_coverage:.0%}) — no estimate is given "
            "for a fund that is majorly outside the model's US equity universe"
        )


def expand_fund(model: RiskModel, holdings: FundHoldings,
                client: HoldingsClient | None = None,
                min_coverage: float = MIN_FUND_COVERAGE,
                depth: int = MAX_FUND_DEPTH,
                _seen: frozenset = frozenset()) -> tuple[dict, dict]:
    """Fund holdings -> model-ticker weights per 1.0 of fund NAV.

    Returns (weights, diagnostics). Constituents that are themselves
    funds expand recursively down to ``depth`` levels. Raises
    FundCoverageError below the coverage floor.
    """
    h = holdings
    seen = _seen | {h.ticker}
    covered: dict[str, float] = {}
    covered_gross = 0.0
    uncovered_gross = 0.0
    uncovered: list[tuple[float, str]] = []
    sub_funds: dict[str, float] = {}
    for t, w in h.weights.items():
        if t not in model.exposures.index and client is not None:
            # Sibling share class the model holds (GOOG -> GOOGL)
            alias = client.alias_for(t, model.exposures.index)
            if alias is not None:
                t = alias
        if t in model.exposures.index:
            covered[t] = covered.get(t, 0.0) + w
            covered_gross += abs(w)
            continue
        if client is not None and depth > 0 and t not in seen and client.fund_locator(t):
            try:
                child = client.holdings(t)
                cw, cdiag = expand_fund(model, child, client, min_coverage,
                                        depth - 1, seen)
            except Exception:  # not expandable: counts as uncovered below
                pass
            else:
                for st, sw in cw.items():
                    covered[st] = covered.get(st, 0.0) + w * sw
                covered_gross += abs(w)
                sub_funds[t] = cdiag["holdings_coverage"]
                continue
        uncovered_gross += abs(w)
        uncovered.append((abs(w), t))
    cash = abs(h.cash_weight)
    gross = (covered_gross + uncovered_gross + cash
             + h.other_weight + h.unresolved_weight)
    coverage = (covered_gross + cash) / gross if gross else 0.0
    uncovered.sort(reverse=True)
    diag = {
        "name": h.name,
        "as_of": h.as_of,
        "source": h.source,
        "holdings_coverage": coverage,
        "n_constituents_covered": len(covered),
        "cash_weight": h.cash_weight,
        "non_equity_weight": h.other_weight,
        "unresolved_weight": h.unresolved_weight,
        "uncovered_model_weight": uncovered_gross,
        "uncovered_examples": [t for _, t in uncovered[:10]],
        **({"sub_funds": sub_funds} if sub_funds else {}),
    }
    if coverage < min_coverage:
        raise FundCoverageError(h.ticker, coverage, min_coverage, diag)
    scale = (gross - cash) / covered_gross if covered_gross else 0.0
    diag["renormalization"] = scale
    return {t: w * scale for t, w in covered.items()}, diag


def fund_risk(model: RiskModel, ticker: str,
              client: HoldingsClient | None = None,
              min_coverage: float = MIN_FUND_COVERAGE,
              top_n: int = 10) -> dict:
    """Full look-through risk report for one ETF or fund ticker.

    Raises KeyError when the ticker locates no N-PORT filer, and
    FundCoverageError when the model covers too little of the fund.
    """
    client = client or _default_client()
    h = client.holdings(ticker)
    weights, diag = expand_fund(model, h, client, min_coverage)
    report = model.portfolio_risk(weights, top_n=top_n)
    return {"ticker": ticker.upper(), "fund": diag, **report}


def portfolio_risk_lookthrough(model: RiskModel, weights: Mapping[str, float],
                               client: HoldingsClient | None = None,
                               optimized: bool = False,
                               min_coverage: float = MIN_FUND_COVERAGE,
                               top_n: int = 10) -> dict:
    """Portfolio risk with fund positions expanded to their holdings.

    Each fund ticker in ``weights`` expands into its constituents before
    the portfolio math runs. A fund that fails look-through (no filing,
    coverage below the floor, network failure) keeps its ticker as-is —
    the model's direct estimate when covered, otherwise uncovered — and
    the reason lands in ``lookthrough.notes``. This function degrades to
    plain ``portfolio_risk`` when EDGAR access is not configured.
    """
    weights = {str(k).upper(): float(v) for k, v in weights.items()}
    notes: dict[str, str] = {}
    if client is None:
        try:
            client = _default_client()
        except ValueError as exc:  # no EDGAR User-Agent configured
            report = model.portfolio_risk(weights, top_n=top_n, optimized=optimized)
            report["lookthrough"] = {"funds": {}, "notes": {"_config": str(exc)}}
            return report
    expanded: dict[str, float] = {}
    funds: dict[str, dict] = {}

    def add(t: str, w: float) -> None:
        expanded[t] = expanded.get(t, 0.0) + w

    for t, w in weights.items():
        try:
            is_fund = client.fund_locator(t) is not None
        except Exception as exc:
            notes[t] = f"fund lookup unavailable: {type(exc).__name__}: {exc}"
            is_fund = False
        if not is_fund:
            add(t, w)
            continue
        try:
            h = client.holdings(t)
            fw, diag = expand_fund(model, h, client, min_coverage)
        except FundCoverageError as exc:
            notes[t] = str(exc)
            add(t, w)
            continue
        except Exception as exc:
            notes[t] = f"look-through unavailable: {type(exc).__name__}: {exc}"
            add(t, w)
            continue
        funds[t] = diag
        for st, sw in fw.items():
            add(st, w * sw)
    report = model.portfolio_risk(expanded, top_n=top_n, optimized=optimized)
    if funds or notes:
        report["lookthrough"] = {"funds": funds,
                                 **({"notes": notes} if notes else {})}
    return report
