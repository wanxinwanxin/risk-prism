# riskprism — model card

> Open-source Barra-style US equity factor risk model, built for AI
> agents. This file is the agent-readable mirror of the current weekly
> build. Research software; not investment advice.

## Current build

| field | value |
|---|---|
| model version | PRISM-US-MH-0.2 |
| as of | 2026-08-21 |
| assets covered | 2987 |
| estimation universe | 2774 |
| regression weeks | 149 |
| mean weekly R² | 0.143 |
| price provider | yahoo |
| fundamentals | 2837 live from EDGAR |
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
`residuals.parquet`, `exposure_history.parquet`, `validation.parquet`,
`asset_meta.parquet`, `fundamentals_store.parquet`, `meta.json`.

### Historical (point-in-time) models

Every weekly formation date is reconstructible from the same artifact
download — `riskprism.model.asof.model_asof(artifacts, date)` returns a
full RiskModel with no lookahead. The explorer serves the same
snapshots as static JSON under [/history/index.json](/history/index.json).

## Factors

| factor | descriptor | current ann. vol |
|---|---|---|
| market | intercept — cap-weighted market return | 16.0% |
| size | ln(market cap) | 3.9% |
| value | book equity / market cap | 33.1% |
| momentum | 12-month return, skipping the most recent month | 8.8% |
| volatility | 252-day daily return std, annualized | 12.5% |
| liquidity | ln(63-day median dollar volume / market cap) | 3.8% |
| quality | ROE: annual net income / book equity | 0.5% |
| leverage | total liabilities / total assets | 1.7% |

Style exposures are winsorized at ±3σ and standardized to cap-weighted
mean 0 / equal-weighted std 1 each week. Industries: Fama-French 12
from SEC EDGAR SIC codes, one-hot, cap-weighted returns constrained
to zero for identification.

### Industry coverage in this build

| industry | assets |
|---|---|
| Other | 659 |
| Money | 604 |
| BusEq | 438 |
| Hlth | 373 |
| Manuf | 241 |
| Shops | 175 |
| NoDur | 111 |
| Utils | 104 |
| Enrgy | 99 |
| Chems | 64 |
| Durbl | 60 |
| Telcm | 59 |

## Factor correlation (market + styles)

| | market | size | value | momentum | volatility | liquidity | quality | leverage |
|---|---|---|---|---|---|---|---|---|
| **market** | 1.00 | 0.20 | 0.29 | 0.11 | 0.68 | -0.32 | -0.10 | 0.24 |
| **size** | 0.20 | 1.00 | 0.37 | 0.04 | 0.59 | 0.33 | 0.01 | -0.13 |
| **value** | 0.29 | 0.37 | 1.00 | 0.13 | 0.32 | -0.08 | 0.02 | -0.27 |
| **momentum** | 0.11 | 0.04 | 0.13 | 1.00 | -0.09 | 0.03 | -0.12 | 0.03 |
| **volatility** | 0.68 | 0.59 | 0.32 | -0.09 | 1.00 | 0.05 | -0.00 | 0.11 |
| **liquidity** | -0.32 | 0.33 | -0.08 | 0.03 | 0.05 | 1.00 | 0.06 | -0.27 |
| **quality** | -0.10 | 0.01 | 0.02 | -0.12 | -0.00 | 0.06 | 1.00 | -0.07 |
| **leverage** | 0.24 | -0.13 | -0.27 | 0.03 | 0.11 | -0.27 | -0.07 | 1.00 |

## Coverage (2987 tickers)

NVDA, AAPL, GOOGL, MSFT, AMZN, AVGO, META, TSLA, MU, BRK-B, LLY, JPM, WMT, AMD, ASML, V, XOM, JNJ, INTC, MA, BAC, CSCO, ABBV, LRCX, AMAT, ORCL, COST, PLTR, CAT, CVX, GE, CYATY, KO, HSBC, UNH, MS, HD, MRK, PG, NFLX, DELL, PANW, GS, RY, RTX, BABA, ARM, GEV, NVS, PM, KLAC, WFC, MUFG, SNDK, TXN, ANET, SHEL, AZN, SAP, C, AXP, STX, AMGN, TM, BHP, CRWD, LIN, TMO, IBM, KXIAY, APH, MRVL, RTNTF, SAN, TD, VZ, TTE, NVO, SHOP, TMUS, SCHW, ADI, ABT, PEP, MCD, BLK, WDC, DIS, BA, NEE, UNP, ETN, UBS, GILD, ATEYY, QCOM, WELL, T, BX, TJX, SMFG, DE, SCCO, SMERY, IBKR, DTEGY, BBVA, RIO, HTHIY, CRM, UBER, BKNG, COP, PFE, BUD, GLW, DHR, ISRG, SONY, LMT, COF, PLD, PH, UL, CB, MFG, BMY, VRTX, BMO, SYK, NEM, PDD, SBUX, NOW, SPGI, LOW, CVS, BTI, HDB, PGR, PBR, MDT, SNOW, HWM, FTNT, CM, VRT, BNY, BNS, ENB, BP, NET, EQIX, PWR, ABNB, MO, CVNA, SO, SNY, ADP, GD, IBN, TT, ACN, APP, ASX, PNC, SPOT, ING, MPC, CNQ, USB, ADBE, GSK, MCK, VLO, EQNR, KKR, CEG, CME, DUK, PSX, FCX, BN, AEM, CSX, BCS, IFNNY, DASH, MMM, MGCLY, MAR, JCI, INTU, CMCSA, WMB, EMR, MELI, DDOG, CDNS, WM, TKOMY, LYG, MNST, CMI, BAESY, MRSH, HCA, SNHIY, UPS, ICE, LITE, HOOD, SHW, ELV, BAM, SPG, EPD, REGN, MCO, CP, ITW, NGG, NOC, ITUB, RCL, AMT, E, NTES, APO, SLB, MDLZ, CTAS, SNPS, FDX, SU, ECL, HPE, CNI, NSC, GM, TRV, EOG, ROST, NWG, MSI, DLR, BSX, AON, NBIS, HLT, SE, MFC, RACE, ORLY, CI, URI, HON, KMI, CL, ET, NU, DB, AMX, MPWR, B, WBD, COHR, TER, PCAR, TGT, TDG, BE, APD, IMO, TRP, FIX, RSG, ALL, JHPCY, TFC, BKR, CIEN, CRH, AJG, TEL, WPM, GWW, KEYS, MET, NUE, ARGX, AFL, CBRS, NOK, PSA, D, CRWV, OKE, MPLX, GWLIF, GRMN, COR, TRGP, NXPI, CVE, FAST, OXY, O, RELX, AME, VALE, NKE, DAL, FANG, SRE, ALAB, F, MT, TAK, LNG, LSEGY, CAH, NDAQ, STT, CRDO, EONGY, RKLB, FITB, EW, DVN, LHX, HEI, PYPL, ADSK, CARR, HONA, DEO, WAB, CTVA, ETR, AU, STM, AMP, BDX, AZO, ROK, VST, XEL, AXON, UMC, GALDY, XYZ, SMTOY, FLEX, FERG, WDAY, INFY, EXC, MDLN, VTR, CCEP, RVMD, ARES, HUM, FNV, FER, CSLLY, EBAY, TTWO, NTRA, WDS, SLF, VIK, ODFL, MCHP, CLS, IX, RBGLY, CCJ, CBRE, LYV, PRU, TRI, BSBR, IDXX, ABEV, CMG, TEVA, PAYX, HLN, KB, A, WCN, DHI, HMC, ONC, KDP, SCMWY, RKT, NTAP, MSCI, ED, BMWKY, TEAM, WAT, UAL, FMX, COIN, IQV, AIG, YUM, ADM, VEEV, SYY, P, JBL, ROP, PCG, EXPE, JD, MSTR, IRM, EME, CCL, PEG, HIG, VOD, GFI, STLD, ESLT, TKO, MTB, EC, HSY, HBAN, KVUE, VMC, WEC, KMB, SHG, TWLO, MDB, DLMAY, QSR, NTRS, PUK, ALC, BIDU, UI, RSHGY, RJF, SUNB, VG, KR, RPRX, FRFHF, BBD, ACGL, ERIC, DXCM, KGC, CHT, CKHGF, EQT, CQP, NTR, GEHC, CCI, ON, RDDT, EXR, MLM, CASY, CNC, TECK, RMD, IR, TDY, FTI, UNVGY, CFG, ATI, ZM, BIIB, CBOE, FSNUY, KBGGY, EL, WTW, AEE, TSEM, ALNY, ZS, BAP, ZTS, RYAAY, GFS, Q, LVS, CPRT, KHC, DTE, ILMN, FOXA, LPLA, HAL, FTS, ATO, NMR, VICI, OWL, NVT, MTD, PBA, WSM, TCOM, CPNG, FISV, INSM, DOV, ASTS, EIX, XYL, FE, ECHO, PPL, ES, RF, OTIS, CPAY, HPQ, RBLX, TS, CRS, CNP, HUBB, AWK, JBHT, CHLSY, DG, LH, CINF, ROIV, SYF, PHG, DGX, WRB, NRG, MRNA, TPR, SN, CW, GELHY, DRI, SYM, CTSH, FCNCA, SW, OKTA, ENTG, PPG, INCY, AFRM, AMRZ, MTSI, EQR, TPL, KEY, DLTR, XPO, SMCI, VRSN, PHM, WST, EXPD, MTZ, PFG, BSP, ARXS, GPN, VLTO, AER, SOFI, IHG, TROW, OMC, FTAI, FSLR, USFD, ROKU, FWONA, CHD, TRMOY, FICO, BNTX, L, BRO, CIB, VRSK, HSAI, DOW, KOF, IOT, TW, RL, PKG, VNOM, STE, CMS, FFIV, SITM, MKL, STZ, HKHHY, BG, WWD, FDXF, MTNOY, BCE, RS, UTHR, MKSI, GH, EDPFY, EXE, FN, ONTO, THC, RIVN, SQM, IP, ULTA, FIS, LUV, BURL, LYB, INIO, AMCR, SNX, IFF, RBRK, CNGKY, EFX, SNA, LEN, BCH, YPF, TSN, CRCL, GIS, NI, PAAS, GMAB, WES, U, CDE, RGLD, TPG, TOST, ITT, CHTR, RCI, SBAC, ESS, DD, EVRG, BR, APG, TELNY, STLA, MGA, LSCC, IONQ, WIT, SSNC, AS, KSPI, ZBH, BEKE, VTRS, EWBC, PR, STRL, FTV, VIV, GPC, DIDIY, LNT, TSCO, BBY, WCC, CF, NTNX, PKX, LDOS, AKAM, IEX, INVH, JHX, RBC, WTKWY, OVV, NLY, ZBRA, DKS, BEN, WF, ZTO, WY, ROL, NDSN, CLH, TLN, CG, QNT, JLL, ARNNY, CDW, H, CHRW, DINO, NVR, J, FLUT, BEP, PAA, CNH, MEDP, GEN, PFGC, ARMK, IREN, BSAC, ASND, KIM, PNFP, WPC, RYAN, BALL, YUMC, RGA, DOCN, SBS, JAZZ, CX, PHYS, PS, PTC, IESC, ALB, BAWAY, HST, DSEEY, TIGO, BWXT, EMA, NBIX, MAA, SCRNY, BBIO, LECO, NWSA, ULS, LAMR, SUI, FUTU, AMKR, RBA, NXT, LTM, TU, SKM, KEP, TRU, TXT, SWK, QXO, GIB, GFL, CRBG, TTMI, PUGBY, SOLV, MAIR, OHI, MLI, GRAB, COO, TLK, UNM, DOC, LOGI, MKC, APA, TME, MAS, CACI, UHAL, IVZ, GWRE, PAG, CSL, RPM, BWA, LII, LYTHF, SMTC, HTHT, JBS, AGI, REG, W, SUN, MOG-A, ARCC, EQH, EG, LNTCY, AEG, GL, DT, AEIS, AIZ, ALLE, AMH, WMG, EQX, BLMOY, CRL, DTM, EMBJ, UDR, SGI, TOL, PSLV, AA, CNA, AUR, LGN, HAS, TXRH, BAX, AVY, NVMI, FIG, AAOI, ALLY, FIVE, RNR, HRL, GGG, TYL, AIT, FNF, TRMB, LULU, PINS, CHKP, EXEL, GNRC, ELS, ICLR, RVTY, GLXY, AGNC, ERIE, DY, WTS, PEN, CLX, BMRN, WSO, WSE, WBS, SEIC, SF, CCK, TMTNY, DKNG, HMY, HL, CR, GLPI, CSGP, CORT, SJM, BF-B, HII, FMS, ARWR, VARRY, SNN, COKE, FHN, DECK, ALGN, OBICY, LI, PAC, PNW, HBM, AFG, DRS, OC, AKZOY, BXP, BZLFY, EHC, BJ, TFII, FPS, BTSG, LFUS, CHYM, MDGL, BNT, KTOS, RTO, GDDY, KNX, ELAN, HALO, ENLT, VICR, MICC, XPEV, MOD, EVR, SANM, FROG, UMBF, RRX, NIO, PSKY, MANH, DOCU, GMED, VSAT, CART, BMNR, YARIY, BPOP, SCI, DVA, WTRG, VIAV, AXSM, TECH, CGNX, AHR, IT, DLAKY, MGM, DPZ, AR, ARW, WMS, JEF, DCI, HUBS, RMBS, SPXC, GKOS, FORM, WTFC, CPT, IAG, EGP, HUT, AYI, SSB, GSAT, MUSA, DAR, JKHY, SAIL, TX, MOH, ABVX, WYNN, ENSG, EGO, PRAX, AES, ALSN, BSY, AM, EAT, NYT, CFR, ZION, CYTK, ONON, GIL, ONB, MP, APTV, FLS, LKNCY, ORI, FRT, NCLTY, SAIA, PNR, PTGX, LINE, SWKS, APGE, SMMT, R, PAYP, RDY, LTH, UHS, HSIC, KYMR, WLK, KRYS, SOLS, SNROY, MSGS, FDS, AMG, SIRI, VMI, OGE, PSO, IONS, AG, PAYC, OSCR, FRHC, PODD, SUZ, ESI, MINBY, RUSHA, PRI, RGEN, AAL, BIO, OSK, LLYVA, PNDRY, FRO, GGB, FCFS, TTC, COMP, TEM, TKR, CUBE, ADC, YMM, SPIIY, AVAV, CTRE, TTEK, MHK, SIMO, SYRE, AVTR, COLB, RRC, SARO, CHWY, BRX, VOYA, BAH, DDS, KT, APLD, IMVT, JXN, CRNX, PB, IDCC, GWKSY, KNTK, PBF, PCVX, WAL, ALV, BOKF, SFD, EDU, PCOR, SNAP, QGEN, BROS, CNM, WULF, PL, HLI, PRMB, NNN, NEU, BRKR, CAE, MTCH, LEVI, ESTC, BVN, LNC, CBSH, HQY, FR, TTAN, QRVO, EMN, ALGM, ATR, STN, KNSL, TIMB, CAVA, CWEN, AOS, ZWS, KMX, SM, AGX, NCLH, ALKS, IBRX, CEF, LEA, PATH, GME, VLY, KRMN, LPGCY, SNEX, IDA, REXR, LAD, HESM, OKLO, CAMT, XP, PCTY, AVT, CMC, ESE, SSD, RAL, ACM, VIST, ARE, RYTM, SBSW, POWL, FIGR, ELPC, MBLY, ASR, JOBY, RHP, HXL, MXL, VBREY, MORN, QBTS, CBC, CHRD, THG, TEX, TWST, AXTA, FSS, NFG, ZG, CIFR, NOV, SOBO, TXG, TAP, TGTX, S, AWI, SSL, SHMXY, VNO, SFM, BLDR, UGI, CAG, OMF, FAF, ORKA, DPC, KEX, MAAS, CELH, ECG, KLAR, TRNO, GGAL, NPO, FLR, AXS, MSA, DBX, RIOT, PLXS, SLAB, ENS, VFS, LW, NAVN, GAP, VCTR, PJT, BJGPY, ETSY, MAC, LOAR, BILI, NE, NXE, STAG, AAON, HNGE, GTES, ZETA, PDI, ACA, GDS, ICL, PACS, AGCO, ORA, FNMA, MSM, BZ, CLF, APPF, UGP, BCHMY, FNB, LQDA, POOL, GTLB, GLBE, BTG, ROAD, MTDR, BIRK, AN, PPC, ACT, ESYJY, MRCY, VSEC, CHE, CGON, HLNE, CPB, VSXY, MOS, LUMN, VIPS, EPRT, ST, HIMS, UBSI, KGS, WFRD, HR, SSRM, SGHC, SWX, LOTMY, LYFT, LNTH, AUGO, OGC, MATX, SEI, URBN, COGT, INGR, BEPC, ENVA, JAN, TFPM, BLTE, CORZ, LSTR, RIG, AVAL, MGY, GBCI, IBP, TAL, SFGYY, RYN, HRB, ERAS, WEX, RRR, INGM, DNTH, LKQ, DSGX, MTG, GATX, AMRX, SNDR, HWC, AXTI, SRRK, MBSHY, MBGL, QLYS, MWH, EXP, TTD, NOVT, ENIC, SLFPY, OR, CAAP, RGTI, HOMB, CROX, STEP, ESNT, CHDN, RSI, LB, NTSK, STWD, KNSA, MIRM, JBTM, AIR, CAI, FSV, DOX, PTCT, SPHR, AUB, M, ASB, CPA, AROC, BLCO, ERELY, BETA, TVTX, CZR, XENE, CRUS, DUOL, IFS, BTDPY, VAL, DFTX, BYD, YOU, PAGP, MTRN, ABCB, RDNT, CVLT, FND, ACMR, POR, CACC, TXNM, BBUC, MCY, ACI, LGND, WHD, TBBB, STVN, ARSTY, OTEX, FLG, XE, RKGRY, NICE, FTDR, DLB, RLI, HRI, MDA, RITM, NXST, MKTX, OBDC, TEO, CWST, SXT, MAUTF, BCPC, G, VFC, AX, MMYT, BXSL, TFX, SON, PECO, CSW, PSMT, GVA, OZK, WPP, RELY, BKH, NJR, UEC, FBIN, SIGI, CIG, CAKE, MAIN, REYN, VSNT, VSH, GXO, ELF, WH, ANDG, MMED, KRG, RNG, HASI, BDC, EBC, GTX, GLNG, OII, WFG, SHC, MMSI, BC, PRM, EQPT, KLIC, IRDM, VIRT, PIPR, MYRG, ADT, GOLF, FRVO, GRFS, SAIC, CIGI, ACIW, BGC, PI, KVYO, PEGA, EXLS, OUT, MIDD, CRSP, CNX, LAUR, MRX, JMKE, UNF, HCC, MTN, ONDS, SBRA, PAY, ESAB, TKMTY, EPAM, SLM, TPC, SKY, ENPH, RMVEY, XMTR, BMA, OTF, WTM, NIQ, BB, CNO, CMBT, CAR, GNTX, DIOD, IEP, OGS, FFIN, VRNS, GHC, MRP, LPX, MC, MUR, ACAD, TREX, GEF, ALH, SFBS, AMTM, CE, OMAB, LBRDA, GBTG, CRGY, RDN, ACHR, ALK, BOOT, PSN, TFSL, INSW, DYN, ATAT, CECO, SR, BIPC, UFPI, OCTV, BILL, FHI, LFST, CUZ, CHH, CNR, IQMX, KBR, EROC, EWTX, WAY, CRC, HPHTY, MTH, TKC, GFF, FULT, TDW, DOO, AEHR, ANF, ALM, NWSGY, SIM, CENX, USAR, CVCO, FELE, INDV, EPR, BCO, AZZ, PBLS, MTLMY, ICUI, SKT, FSLY, PTEN, ATRO, TNL, IBOC, OLLI, FLY, KTB, MANE, CHEF, FBP, CELC, RLAY, PRIM, CVSA, BFH, CON, ACLS, CBT, BOX, AQN, NNI, BNL, NVST, LIVN, BLLN, MEOH, HP, BRC, SLGN, AMVOY, ITRI, HGTY, UCB, TCBI, NSIT, NGXXF, DOCS, VNT, SYNA, RHI, CNK, FLIDY, MDU, BULL, NWE, VVV, CATY, CNS, SLG, LAZ, OPLN, ELVN, CGAU, TNGX, KFY, SEB, DAVE, NP, ARX, TGS, CDP, PAM, ITGR, UWMC, CLMT, WSC, MIAX, ATMU, KRC, WSFS, SITE, AVNT, MBRFY, MAT, PARR, FCN, VAC, DNP, COLD, LIF, SKYW, DK, GEO, DLO, EE, SEZL, PHI, RUM, CVBF, WSBC, PTRN, INDB, HAE, ADOOY, QTWO, THO, TAC, CSTM, MANU, STNG, IRT, MSGE, SKE, DORM, CCC, HHH, ADPT, LEGN, TDS, OLED, TLX, PFSI, RNST, CRK, MWA, VEON, SXI, HAFN, VKTX, TENB, UCTT, MIR, FRMI, PII, SUNC, KAI, DNLI, LMND, OTTR, SRAD, USAC, BANF, AYA, PLNT, JOE, NVTS, IRTC, GPCR, MHO, RXO, NEXHY, WK, GNW, AKO-A, BLSH, COCO, OSIS, CALM, BBWI, BMI, SMR, KNF, APLE, NIC, QS, ARIS, ABG, BFAM, IPGP, UTG, JOYY, FIBK, MCHB, RCUS, PRK, LOPE, NMRK, CVI, UUUU, MARA, GPGI, LEU, ERO, AWATY, HTFL, NESR, BELFA, BATRA, LBRT, OGN, ZGN, POST, FFBC, NHC, LXP, FMCC, NHI, LBTYA, IPAR, LION, PVH, IDYA, LPL, POWI, MNDY, BWLP, OPCH, FRPT, ABCL, MGNI, WT, HGV, SA, OUST, SBCF, AWR, FA, HNI, PONY, FOUR, FG, ZIM, AMBA, SFNC, FUL, CSQR, LRN, AAP, SMG, CURB, NATL, CBU, NMIH, HIW, KC, FRSH, GBDC, NG, XNDU, SHOO, ASH, BKU, VCYT, RDW, KBH, OPEN, DAN, NEA, USLM, KEN, SIG, AB, KN, AGO, CRVL, SBLK, CSQ, EROK, FHB, LIFE, VGNT, LFTO, ARLP, RH, CWK, FSK, ALMS, CXW, VECO, MBX, AAMI, CLDX, PLMR, PENG, ADEA, BHF, GPK, QURE, ADX, CHRN, PLSE, LUNR, AVA, ARCB, NN, ARQT, CPK, DX, IOVA, KOZAY, MNSO, CARG, DGII, PFS, WRBY, FSM, BTU, CLSK, SHAK, BTE, REZI, TGB, GRAL, ATKR, EXTR, AGYS, BRZE, MZTI, EXPO, YETI, WING, AAUC, BOH, PLUG, DXPE, BMRRY, NAMS, TRMD, WIX, HAYW, SOUN, BBAR, SII, VTMX, LASR, TNET, HTGC, DCO, AMBP, IMNM, BANC, TARS, TMDX, GPI, WDFC, BSM, TNK, IRON, AD, FBK, GSHD, EXG, MGEE, PPTA, CWT, GRBK, TR, KALU, DHT, UTF, KYIV, PK, WTTR, AKR, BHE, INFQ, EXK, HYPMY, UE, INTA, IOND, GPOR, COHU, CALY, CBZ, APAM, SDRL, DBRG, PPLI, MGRC, DNOW, KWR, BKD, COLM, BWIN, FIZZ, ZLAB, UNFI, TTAM, CXT, BKV, VC, DNN, JPC, UPST, FLNC, IMAX, DKL, ASO, BCC, MMS, NTCT, IHS, AVPT, HOG, WOR, NRIX, TRMK, TBBK, SUPN, WAFD, BEAM, ABM, CUBI, NBTB, KD, ACHC, EFXT, ALHC, GRND, PRVA, HWKN, SPSC, PATK, NAD, AEO, SHAZ, FCPT, FXCNY, PHIN, FBNC, STUB, AVAH, FRME, TALO, SPNT, HLIO, ICHR, BBT, CENT, HIMX, OSW, RGC, ALRM, NOG, ATAI, GDV, NVG, VISN, DAC, EBOSY, SLS, MAN, TSAT, SVM, KOD, CLBT, EEFT, HTO, VOYG, SYBT, AXGN, TRVI, VRDN, ATHM, NGVT, LCLN, VVX, NWL, IVT, RARE, ROG, BGSI, SIND, WHR, WLY, BUSE, SAH, NEOG, LCII, BCRX, FLOC, BTDR, SLDE, TDC, DRH, NKTR, APPN, MBIN, CURLF, AGM, KYN, BANR, UFPT, RNW, PENN, PTY, ECO, PTXKY, CSAN, NTB, CALX, PHVS, COAG, TSCFY, HYMC, LCID, KLRA, NZF, HUBG, MPT, BVC, PLBL, CTRI, BHC, CDNA, PAYO, MH, KMT, LIME, AMLX, BXMT, UNIT, DBD, URGN, CTOS, FIGS, GRDN, TRN, EFSC, BFLY, PBH, PGEN, FIVN, COTY, PAGS, NMM, HURN, CC, KEEL, BOBS, SKWD, PTON, AEGXF, WERN, PLPC, RAPP, OCUL, WGS, ARR, RUN, HG, RVT, ETY, INTR, DEI, SPGNY, SMSGY, IOSP, ATRC, GOF, SXGCF, ALGT, HAWK, NUVB, NWBI, RAMP, MLYS, NGL, ANDE, UAA, SSMR, VERA, ESTA, HCI, PLUS, MTX, RLX, HTH, OFG, STNE, CDLR, VNET, PVLA, ALMR, DJT, XRAY, VCEL, AERO, NCNO, CLOV, TDTH, ETOR, PDFS, CMPR, BKE, NBTX, FLYW, MCRI, PBI, WU, TILE, HRMY, ECPG, HCM, NMZ, VIA, HAPN, EVT, MESO, BSTZ, AMC, BHVN, INVX, PRLB, CCU, FCF, LTC, OLN, LIND, ALKT, KSS, ADUS, GENI, DRD, STDN, AMR, STGW, MD, GRC, SRCE, NWN, INOD, TMC, MLCO, MNR, BRSL, ADMA, DRVN, LPG, NEO, MRVI, XPRO, STOK, HMN, TIC, BXDC, NTST, STC, SEPN, PIII, CLM, CHCO, NVCR, QUBT, GHRS, PEB, UTZ, DV, GTY, BLX, HE, GNL, DHC, MFP, ASAN, SPB, PRKS, PSNY, AGIO, FSUN, PLGO, VERX, SHO, TSHA, AUPH, WRD, BLBD, LOB, GENB, CCS, ZBIO, JBLU, PRDO, BLKB, ALG, DAO, NLST, CNXN, IMOS, ATEN, PLAB, CEPU, PGNY, ASTH, BRVE, NSP, KARO, GEL, CHA, WS, SRPT, NBHC, KRP, ATS, TY, ALNT, GABC, NEXA, DMRA, EPAC, CMPS, OCFC, RBCAA, SEDG, PAEXY, PRCH, LMAT, XHR, SMA, NUV, KARD, ZD, FOIL, USA, SAM, WB, SGRY, SNDA, ETG, NEXT, NAC, TFIN, BVENY, PAX, TCBK, EVTC, HOPE, SONO, WKC, MBC, TRLV, CMRE, CTS, HUN, NYAX, PDO, AIN, EZPW, DCOM, VET, EVCM, LKFT, GCT, STBA, ZYME, WOLF, TGLS, GAB, STRA, CASH, IMCR, BY, ATEX, EFC, CNOB, TH, STEW, PRGO, AVVSY, ANAB, SNDX, VSTS, USAS, ETON, ETV, TSLX, PRGS, WMK, OPRA, AAT, ANIP, GT, CBL, WBI, CYD, BDJ, XXI, PGY, FCEL, BST, BFC, XNCR, HLMN, MDCOY, QCRH, GLP, BLFS, BL, BAND, FTRE, TPB, NTLA, AVLN, AMPL, RMIX, AMPX, AAPG, FORTY, PRG, IE, CPRI, RXRX, RLJ, RQI, FLNG, AEVA, LILA, LZB, OBK, PWP, BCAT, DXC, IDT, SFL, AMBQ, EMAT, ARCO, RVLV, IMKTA, OMCL, INSP, MQ, FBRX, KMTS, CET, WEN, NIMU, GAM, DSGR, AGL, DCH, JJSF, POET, BCAX, YSWY, TRIN, APC, IIPR, WWW, COUR, SVV, NOMD, GTBIF, UROY, VIR, RCAT, AESI, MLKN, ADIG, FUN, TRAX, ADNT, CXM, AGBK, CSWC, ITG, ABSI, GLOB, PBT, TYRA, GSL, AMSC, ICFI, XERS, MAZE, HLX, PUMP, KOS, FLO, LKFN, ECAT, BBAI, ENR, TNDM, ARDT, BXBL, AMAL, SBH, KMPR, ESQ, SAFT, LBRX, SLBT, TV, INVA, JBIO, HPP, AI, UMAC, RYZ, PSNL, AMRC, QQQX, BW, LNXSY, FOR, HROW, APPS, EYE, GIC, HLIT, REX, ARLO, NFJ, HCSG, SBCLY, PEBO, SCL, AZUL, CNMD, SLVM, ULCC, AZTA, ATEC, QFIN, BHRB, TMP, NAT, TE, YSS, RES, AVBP, PURR, NBR, PCT, OMDA, BTT, DMC, ESRT, IAUX, INNV, CDRE, EBCRY, CTBI, BRUN, ENOV, TWFG, TOP, BJRI, GBX, CNXC, GIII, EOSE, UTI, SBET, PAHC, MLTX, ROCK, TRS, XZO, WD, AIP, AORT, BORR, XPEL, CRI, FMBH, VTOL, DMLP, UFCS, IART, WABC, WLTH, NRP, UMH, SHLS, CNL, AKTS, CCEC, ORIC, REPL, SFDMY, PNTG, LGIH, FRNM, DFH, JCAP, ALX, GHM, NSSC, ELE, VOR, HYT, MGRT, INBX, LQDT, NMAX, REAL, HQH, ARHS, HBT, ANRO, SLSR, NVAX, PICS, THRM, ORC, AGRO, AMN, APMD, LSPD, ODC, CBRL, NVGS, GLUE, OSBC, ACVA, PRSU, SDGR, TRUP, MGTX, DRTS, UAN, SPTX, CRSR, GOLD, NUTX, SENEA, TBN, LEG, MSDL, EPC, IQ, BKSY, ALVO, WLDN, CGEM, TEN, HBNB, VZLA, PFBC, SGML, EFOR, FMC, AFYA, IMTX, YELP, AIIR, ASM, OPY, GDRX, ESBA, LADR, LU, OMER, ELVR, BH-A, CRVS, MNKD, WINA, NBXG, RIOFF, TTI, MRTN, MYE, IRS, DSL, JFR, NXRT, HLF, DOLE, NEWP, BLZE, CPPKF, WLFC, RDWR, CERT, NPKI, CLBK, HFWA, CLVT, MNSKY, WYFI, STAA, DEA, TDOC, UVE, BFS, UVSP, PRCT, BRBR, CCO, MMI, TRIP, DFIN, LWLG, EOS, QNST, SPH, CRCT, LTGO, CRON, PDM, LNN, BVS, USPH, MCB, GNK, HZO, ASIC, BRAI, FSBC, LMRI, TNC, PDS, AVEX, LAC, ECVT, OCSL, AVO, AMTB, AOD, GO, UPBD, NWPX, TK, RHLD, TYG, VRTS, SIFY, ASA, WBTN, ASST, MUX, GPRE, IRMD, IGIC, VMET, PSEC, RBTK, GSBD, IVA, DLX, BOW, CMP, SVRA, GTM, XIFR, SANA, RDVT, UVV, MSEX, TUYA, EDN, BTX, CHY, LOMA, FPF, EMO, LAR, CRF, CFFN, MALG, BV, CHI, HSLV, AVTX, OBX, ETW, VYX, EVV, MBWM, SAFE, SBR, EQBK, TALMF, LUXE, SCSC, ARDX, HBNC, GLIBA, TBLA, BFST, SVC, LXRX, TDAY, CRAI, PTA, NNE, RZLV, MCBS, ITRN, FSCO, NPK, CCNE, SBGI, DXYZ, FTK, KFRC, OI, INDI, FINV, RAJAF, CPAC, OPK, RXT, LTGHY, ODTX, HRTG, UPWK, ABR, FMCB, MUC, RERE, WVE, CSIQ, CII, NBBK, BMEZ, CMRF, KOPN, VPG, QMCO, ASTE, QDEL, KURA, JANX, ACEL, HTLD, CPF, EVLV, BBDC, GERN, DEC, HEPS, CAC, JBSS, HPK, RNP, UTL, CIM, FET, SBSI, PXED, BCX, CLYM, BBN, PDX, CCXI, TRST, THFF, NXDR, CSR, SLDB, MPB, PLOW, KODK, DQ, SMPL, AEBI, DAKT, BTZ, FMNB, CRML, FRBT, HAFC, KRT, ACDC, NX, REF, CADL, KRO, AOSL, MTA, STK, RPC, BUR, RJET, OLMA, LPTH, FCBC, HQ, ABUS, LUCK, LZ, ALOY, DSGN, CEVA, CAPL, SYNSY, AIO, AVR, DBVT, HMH, BLMN, MNTN, MCS, COLL, MFA, CSTL, PCRX, OXLC, EVEX, TROX, CBLL, MTUS, LINC, NRK, CODI, PSIX, IMC, ADAM, SATL, LYTS, PD, HTD, CRTO, NAK, NXP, CVLG, ADSE, REEMF, KOP, CCBG, SMBK, ANL, WRLD, AMPH, WGO, ABX, FTH, KBDC, IPX, SABR, GLAS, TBPH, TRNS, VLRS, GSBC, JBGS, APOG, ARI, PCN, ANGX, AWF, FOXF, BZH, PLTK, FSSL, SMP, TALK, EGBN, EIG, JKS, GILT, ACRS, INMD, TIGR, HIPO, ABTC, ENRD, MAKO, FEIM, SGP, PSTL, SPFI, NAVI, HIVE, MOMO, OGG, PHK, ORRF, PHAR, KRNGY, ANNX, SWIM, SMBC, DSP, GGN, SWRD, RSKD, RPD, CAZGF, MITK, SIBN, PMT, ALRS, DJCO, EVER, GOOS, HBIA, UAMY, HDL, KDK, EOI, PGC, GCMG, IBTA, GRPN, HSTM, SB, SECZ, EVC, DC, FISI, CTO, BTO, YALA, MQY, PKE, SCZM, THQ, APEI, IMMX, WEST, NBRG, CDNL, REPX, HTB, MFIC, GFR, SXC, BAK, MEGI, GSM, MAMA, RIGL, SHBI, IBCP, MOV, AURA, ENVX, BBNX, SKYH, BOT, PEO, JMIA, SCHL, RMT, VREX, PUBM, RVI, FFC

## Forecast validation (out-of-sample, in-build)

Every historical week, the model forecast next-week volatility for a
panel of test portfolios using only data available at the time; z =
realized return / forecast vol. A calibrated model gives std(z) — the
**bias statistic** — of ~1.0 (>1 underforecasts risk, <1 overforecasts)
and |z| > 1.96 about 5% of the time.

Overall: **bias statistic 1.07**, |z|>1.96 rate 6.9%, 123 weeks × 3483 portfolio-scores.

| portfolio | bias stat | \|z\|>1.96 | mean forecast vol | mean realized vol | vol ratio |
|---|---|---|---|---|---|
| equal_weight | 1.02 | 5.7% | 17.7% | 18.5% | 1.04 |
| IWM | 0.98 | 5.5% | 21.1% | 21.6% | 1.02 |
| MTUM | 1.06 | 11.0% | 21.2% | 24.8% | 1.16 |
| QUAL | 0.99 | 4.6% | 14.9% | 15.5% | 1.03 |
| SPY | 1.06 | 6.4% | 15.3% | 16.8% | 1.09 |
| USMV | 1.03 | 4.6% | 10.6% | 11.0% | 1.03 |
| VLUE | 1.13 | 9.2% | 16.9% | 19.4% | 1.14 |
| industry_BusEq | 1.05 | 6.5% | 22.5% | 23.6% | 1.04 |
| industry_Chems | 0.80 | 2.4% | 18.3% | 17.4% | 0.93 |
| industry_Durbl | 0.99 | 6.5% | 26.5% | 27.7% | 1.04 |
| industry_Enrgy | 1.04 | 4.9% | 22.6% | 22.9% | 1.00 |
| industry_Hlth | 1.06 | 6.5% | 16.5% | 16.5% | 0.99 |
| industry_Manuf | 1.04 | 3.3% | 20.0% | 22.1% | 1.10 |
| industry_Money | 1.11 | 7.3% | 15.1% | 17.3% | 1.14 |
| industry_NoDur | 1.01 | 5.7% | 15.3% | 16.2% | 1.06 |
| industry_Shops | 1.01 | 7.3% | 16.7% | 17.3% | 1.03 |
| industry_Telcm | 0.95 | 5.7% | 19.4% | 17.7% | 0.91 |
| industry_Utils | 0.99 | 4.9% | 14.9% | 15.4% | 1.03 |
| market | 1.07 | 4.9% | 17.8% | 21.2% | 1.15 |
| random_1 | 0.97 | 4.9% | 20.5% | 19.7% | 0.92 |
| random_2 | 1.03 | 6.5% | 20.0% | 20.3% | 0.99 |
| random_3 | 1.00 | 4.1% | 20.1% | 18.9% | 0.91 |
| style_leverage | 1.38 | 16.3% | 10.0% | 13.3% | 1.33 |
| style_liquidity | 1.01 | 8.1% | 14.8% | 16.8% | 1.10 |
| style_momentum | 1.05 | 5.7% | 15.3% | 21.9% | 1.40 |
| style_quality | 1.27 | 15.4% | 13.7% | 16.6% | 1.21 |
| style_size | 1.18 | 11.4% | 13.3% | 14.1% | 1.06 |
| style_value | 1.13 | 8.1% | 10.4% | 11.6% | 1.08 |
| style_volatility | 1.05 | 8.1% | 26.5% | 28.8% | 1.08 |

The vol ratio compares average realized variance (from daily returns
within each week) to average forecast variance, in vol units — an
RV-based check with far more statistical power than z-scores alone.

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
  Audited severity (SEC bulk archives, 2026-08): departed filers are
  ~8.7% of filer book equity, bounding cap-weighted return-mean bias
  at ~1.4-2.5bp/week; covariances (what this model ships) are affected
  at second order. Capture-forward appending plus the 13/26-week EWMA
  half-lives make the bias decay away within ~18-24 months of launch.
- Delisting classification is a price heuristic (merger vs failure),
  not filing-verified.
- Universe heuristics are crude (ticker-pattern filters; some ADRs
  leak through).
- Stress tests are first-order (exposure × shock).

## Links

- Interactive explorer: [/](/)
- Source, docs, full methodology: [https://github.com/wanxinwanxin/risk-prism](https://github.com/wanxinwanxin/risk-prism)
