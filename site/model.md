# riskprism — model card

> Open-source Barra-style US equity factor risk model, built for AI
> agents. This file is the agent-readable mirror of the current weekly
> build. Research software; not investment advice.

## Current build

| field | value |
|---|---|
| model version | PRISM-US-MH-0.1 |
| as of | 2026-08-21 |
| assets covered | 294 |
| regression weeks | 149 |
| mean weekly R² | 0.299 |
| price provider | yahoo |
| frequency | W-FRI (annualized outputs) |

## How to use this model

### As an AI agent (MCP server)

```json
{
  "mcpServers": {
    "riskprism": {
      "command": "riskprism-mcp",
      "env": {
        "RISKPRISM_ARTIFACTS": "/path/to/artifacts"
      }
    }
  }
}
```

Tools: `get_model_info`, `get_portfolio_risk`, `get_factor_exposures`,
`stress_test`, `check_coverage`. Weights are portfolio weights (shorts
negative, any gross); volatilities are annualized decimals.

### In Python

```python
# pip install git+https://github.com/wanxinwanxin/risk-prism
from riskprism import RiskModel
model = RiskModel.load("artifacts")
model.portfolio_risk({"AAPL": 0.4, "MSFT": 0.3, "XOM": 0.3})
```

### Raw artifacts (parquet): [https://github.com/wanxinwanxin/risk-prism/releases](https://github.com/wanxinwanxin/risk-prism/releases)

`exposures.parquet` (asset × factor), `factor_covariance.parquet`
(K × K annualized), `specific_risk.parquet`, `factor_returns.parquet`,
`meta.json`.

## Factors

| factor | descriptor | current ann. vol |
|---|---|---|
| market | intercept — cap-weighted market return | 13.6% |
| size | ln(market cap) | 3.8% |
| value | book equity / market cap | 3.2% |
| momentum | 12-month return, skipping the most recent month | 12.2% |
| volatility | 252-day daily return std, annualized | 11.5% |
| liquidity | ln(63-day median dollar volume / market cap) | 4.5% |
| quality | ROE: annual net income / book equity | 3.0% |
| leverage | total liabilities / total assets | 3.7% |

Style exposures are winsorized at ±3σ and standardized to cap-weighted
mean 0 / equal-weighted std 1 each week. Industries: Fama-French 12
from SEC EDGAR SIC codes, one-hot, cap-weighted returns constrained
to zero for identification.

### Industry coverage in this build

| industry | assets |
|---|---|
| Money | 66 |
| BusEq | 62 |
| Other | 44 |
| Manuf | 24 |
| Hlth | 23 |
| Shops | 17 |
| Enrgy | 17 |
| Utils | 12 |
| NoDur | 9 |
| Telcm | 8 |
| Durbl | 6 |
| Chems | 6 |

## Factor correlation (market + styles)

| | market | size | value | momentum | volatility | liquidity | quality | leverage |
|---|---|---|---|---|---|---|---|---|
| **market** | 1.00 | 0.21 | 0.13 | 0.07 | 0.72 | -0.12 | -0.05 | -0.10 |
| **size** | 0.21 | 1.00 | 0.10 | -0.13 | 0.24 | 0.32 | 0.06 | -0.22 |
| **value** | 0.13 | 0.10 | 1.00 | 0.17 | 0.01 | 0.13 | 0.28 | -0.21 |
| **momentum** | 0.07 | -0.13 | 0.17 | 1.00 | -0.06 | 0.03 | 0.24 | 0.11 |
| **volatility** | 0.72 | 0.24 | 0.01 | -0.06 | 1.00 | -0.09 | -0.24 | -0.22 |
| **liquidity** | -0.12 | 0.32 | 0.13 | 0.03 | -0.09 | 1.00 | 0.21 | -0.14 |
| **quality** | -0.05 | 0.06 | 0.28 | 0.24 | -0.24 | 0.21 | 1.00 | -0.15 |
| **leverage** | -0.10 | -0.22 | -0.21 | 0.11 | -0.22 | -0.14 | -0.15 | 1.00 |

## Coverage (294 tickers)

NVDA, AAPL, GOOGL, MSFT, AMZN, AVGO, META, TSLA, MU, BRK-B, LLY, JPM, WMT, AMD, ASML, V, XOM, JNJ, INTC, MA, BAC, CSCO, ABBV, LRCX, AMAT, ORCL, COST, PLTR, CAT, CVX, GE, KO, HSBC, UNH, MS, HD, MRK, PG, NFLX, DELL, PANW, GS, RY, RTX, BABA, ARM, GEV, NVS, PM, KLAC, WFC, MUFG, SNDK, TXN, ANET, SHEL, AZN, SAP, C, AXP, STX, AMGN, TM, BHP, CRWD, LIN, TMO, IBM, KXIAY, APH, MRVL, SAN, TD, VZ, TTE, NVO, SHOP, TMUS, SCHW, ADI, ABT, PEP, MCD, BLK, WDC, DIS, BA, NEE, UNP, ETN, UBS, GILD, ATEYY, QCOM, WELL, T, BX, TJX, SMFG, DE, SCCO, SMERY, IBKR, DTEGY, BBVA, RIO, HTHIY, CRM, UBER, BKNG, COP, PFE, BUD, GLW, DHR, ISRG, SONY, LMT, COF, PLD, PH, UL, CB, MFG, BMY, VRTX, BMO, SYK, NEM, PDD, SBUX, NOW, SPGI, LOW, CVS, BTI, HDB, PGR, PBR, MDT, SNOW, HWM, FTNT, CM, VRT, BNY, BNS, ENB, BP, NET, EQIX, PWR, ABNB, MO, CVNA, SO, SNY, ADP, GD, IBN, TT, ACN, APP, ASX, PNC, SPOT, ING, MPC, CNQ, USB, ADBE, GSK, MCK, VLO, EQNR, KKR, CEG, CME, DUK, PSX, FCX, BN, AEM, CSX, BCS, IFNNY, DASH, MMM, MAR, JCI, INTU, CMCSA, WMB, EMR, MELI, DDOG, CDNS, WM, TKOMY, LYG, MNST, CMI, BAESY, MRSH, HCA, UPS, ICE, LITE, HOOD, SHW, ELV, BAM, SPG, EPD, REGN, MCO, CP, ITW, NGG, NOC, ITUB, RCL, AMT, E, NTES, APO, SLB, MDLZ, CTAS, SNPS, FDX, SU, ECL, HPE, CNI, NSC, GM, TRV, EOG, ROST, NWG, MSI, DLR, BSX, AON, NBIS, HLT, SE, MFC, RACE, ORLY, CI, URI, HON, KMI, CL, ET, NU, DB, AMX, MPWR, B, WBD, COHR, TER, PCAR, TGT, TDG, BE, APD, IMO, TRP, FIX, RSG, ALL, TFC, BKR, CIEN, CRH, AJG, TEL, WPM, GWW, KEYS, MET, NUE, ARGX, AFL, NOK, PSA, D, CRWV, OKE, MPLX

## Methodology in brief

1. Universe: EDGAR-registered US common stocks; price ≥ $2, 21-day
   median dollar volume ≥ $1M, ≥ 26 weeks of history.
2. Point-in-time fundamentals from EDGAR XBRL (values used only after
   their `filed` date); daily adjusted prices from a pluggable provider.
3. Weekly cross-sectional WLS regression of returns on exposures,
   √(market cap) weights, industry returns cap-weighted to zero.
4. Factor covariance: EWMA on weekly factor returns — vol half-life
   13w, correlation half-life 26w — annualized ×52, repaired to PSD.
5. Specific risk: EWMA of squared residuals, shrunk
   30% toward size-quintile means.
6. Portfolio risk: Σ = X F Xᵀ + diag(s²).

## Known limitations

- Survivorship bias: the price panel is fetched at build time, so
  delisted names are absent from the regression history. Risk
  forecasts are less affected than historical factor-return studies.
- Universe heuristics are crude (ticker-pattern filters; some ADRs
  leak through).
- Stress tests are first-order (exposure × shock).

## Links

- Interactive explorer: [/](/)
- Source, docs, full methodology: [https://github.com/wanxinwanxin/risk-prism](https://github.com/wanxinwanxin/risk-prism)
