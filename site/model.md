# riskprism — model card

> Open-source Barra-style US equity factor risk model, built for AI
> agents. This file is the agent-readable mirror of the current weekly
> build. Research software; not investment advice.

## Current build

| field | value |
|---|---|
| model version | PRISM-US-MH-0.2 |
| as of | 2026-08-14 |
| assets covered | 300 |
| estimation universe | 294 |
| regression weeks | 148 |
| mean weekly R² | 0.345 |
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
| market | intercept — cap-weighted market return | 13.8% |
| size | ln(market cap) | 3.8% |
| value | book equity / market cap | 2.9% |
| momentum | 12-month return, skipping the most recent month | 12.4% |
| volatility | 252-day daily return std, annualized | 11.3% |
| liquidity | ln(63-day median dollar volume / market cap) | 4.3% |
| quality | ROE: annual net income / book equity | 3.1% |
| leverage | total liabilities / total assets | 3.6% |

Style exposures are winsorized at ±3σ and standardized to cap-weighted
mean 0 / equal-weighted std 1 each week. Industries: Fama-French 12
from SEC EDGAR SIC codes, one-hot, cap-weighted returns constrained
to zero for identification.

### Industry coverage in this build

| industry | assets |
|---|---|
| Money | 66 |
| BusEq | 63 |
| Other | 49 |
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
| **market** | 1.00 | 0.25 | 0.10 | 0.06 | 0.72 | -0.10 | -0.19 | 0.08 |
| **size** | 0.25 | 1.00 | -0.00 | -0.17 | 0.33 | 0.29 | -0.08 | -0.03 |
| **value** | 0.10 | -0.00 | 1.00 | 0.21 | -0.07 | 0.08 | 0.32 | -0.17 |
| **momentum** | 0.06 | -0.17 | 0.21 | 1.00 | -0.07 | -0.01 | 0.37 | -0.12 |
| **volatility** | 0.72 | 0.33 | -0.07 | -0.07 | 1.00 | -0.03 | -0.40 | -0.02 |
| **liquidity** | -0.10 | 0.29 | 0.08 | -0.01 | -0.03 | 1.00 | 0.04 | 0.10 |
| **quality** | -0.19 | -0.08 | 0.32 | 0.37 | -0.40 | 0.04 | 1.00 | -0.23 |
| **leverage** | 0.08 | -0.03 | -0.17 | -0.12 | -0.02 | 0.10 | -0.23 | 1.00 |

## Coverage (300 tickers)

NVDA, AAPL, GOOGL, MSFT, AMZN, AVGO, META, TSLA, MU, BRK-B, LLY, JPM, WMT, AMD, ASML, V, XOM, JNJ, INTC, MA, BAC, CSCO, ABBV, LRCX, AMAT, ORCL, COST, PLTR, CAT, CVX, GE, CYATY, KO, HSBC, UNH, MS, HD, MRK, PG, NFLX, DELL, PANW, GS, RY, RTX, BABA, ARM, GEV, NVS, PM, KLAC, WFC, MUFG, SNDK, TXN, ANET, SHEL, AZN, SAP, C, AXP, STX, AMGN, TM, BHP, CRWD, LIN, TMO, IBM, KXIAY, APH, MRVL, RTNTF, SAN, TD, VZ, TTE, NVO, SHOP, TMUS, SCHW, ADI, ABT, PEP, MCD, BLK, WDC, DIS, BA, NEE, UNP, ETN, UBS, GILD, ATEYY, QCOM, WELL, T, BX, TJX, SMFG, DE, SCCO, SMERY, IBKR, DTEGY, BBVA, RIO, HTHIY, CRM, UBER, BKNG, COP, PFE, BUD, GLW, DHR, ISRG, SONY, LMT, COF, PLD, PH, UL, CB, MFG, BMY, VRTX, BMO, SYK, NEM, PDD, SBUX, NOW, SPGI, LOW, CVS, BTI, HDB, PGR, PBR, MDT, SNOW, HWM, FTNT, CM, VRT, BNY, BNS, ENB, BP, NET, EQIX, PWR, ABNB, MO, CVNA, SO, SNY, ADP, GD, IBN, TT, ACN, APP, ASX, PNC, SPOT, ING, MPC, CNQ, USB, ADBE, GSK, MCK, VLO, EQNR, KKR, CEG, CME, DUK, PSX, FCX, BN, AEM, CSX, BCS, IFNNY, DASH, MMM, MGCLY, MAR, JCI, INTU, CMCSA, WMB, EMR, MELI, DDOG, CDNS, WM, TKOMY, LYG, MNST, CMI, BAESY, MRSH, HCA, SNHIY, UPS, ICE, LITE, HOOD, SHW, ELV, BAM, SPG, EPD, REGN, MCO, CP, ITW, NGG, NOC, ITUB, RCL, AMT, E, NTES, APO, SLB, MDLZ, CTAS, SNPS, FDX, SU, ECL, HPE, CNI, NSC, GM, TRV, EOG, ROST, NWG, MSI, DLR, BSX, AON, NBIS, HLT, SE, MFC, RACE, ORLY, CI, URI, HON, KMI, CL, ET, NU, DB, AMX, MPWR, B, WBD, COHR, TER, PCAR, TGT, TDG, BE, APD, IMO, TRP, FIX, RSG, ALL, JHPCY, TFC, BKR, CIEN, CRH, AJG, TEL, WPM, GWW, KEYS, MET, NUE, ARGX, AFL, CBRS, NOK, PSA, D, CRWV, OKE, MPLX

## Methodology in brief

1. Universe: EDGAR-registered US common stocks; price ≥ $2, 21-day
   median dollar volume ≥ $1M, ≥ 26 weeks of history.
2. Point-in-time fundamentals from EDGAR XBRL (values used only after
   their `filed` date); daily adjusted prices from a pluggable provider.
3. Two universes: liquid names (price ≥ $2, ADV ≥ $1M, ≥ 26w history)
   estimate the factor returns; every name alive at the build date is
   covered — risk comes through the factor structure plus a structural
   specific-risk prior, so no asset-level history is required.
4. Weekly cross-sectional WLS regression of returns on exposures,
   √(market cap) weights, industry returns cap-weighted to zero.
5. Factor covariance: EWMA on weekly factor returns — vol half-life
   13w, correlation half-life 26w — annualized ×52, repaired to PSD.
6. Specific risk: each asset's EWMA residual vol blended with a
   cross-sectional structural prediction (from size, volatility,
   liquidity, industry) by history length: w = T/(T + 26w). Assets
   without history get the pure structural prior.
7. Capture-forward history: each weekly build appends to the prior
   build's factor returns and residuals; names that stop trading get
   an imputed delisting return in their final week and keep their
   historical rows, so post-launch history is survivorship-free.
8. Portfolio risk: Σ = X F Xᵀ + diag(s²).

### Per-asset estimation quality

`get_factor_exposures` and the artifacts' `asset_meta.parquet` report,
per asset: `in_estimation` (participates in factor regressions),
`history_weeks` (residual observations), and `specific_blend_weight`
(how much of the specific-risk estimate is the asset's own history vs
the structural prior). Low-weight names are prior-driven — treat their
numbers as informed estimates, not measurements.

## Known limitations

- Survivorship bias in the cold-start history: weeks recorded before
  this project launched exclude names that had already delisted.
  Capture-forward appending plus the 13/26-week EWMA half-lives make
  this bias decay away — the effective window is largely bias-free
  ~18-24 months after launch. Factor-return means are affected more
  than the covariances this model actually ships.
- Delisting classification is a price heuristic (merger vs failure),
  not filing-verified.
- Universe heuristics are crude (ticker-pattern filters; some ADRs
  leak through).
- Stress tests are first-order (exposure × shock).

## Links

- Interactive explorer: [/](/)
- Source, docs, full methodology: [https://github.com/wanxinwanxin/risk-prism](https://github.com/wanxinwanxin/risk-prism)
