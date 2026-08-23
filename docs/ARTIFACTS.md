# Artifact schema (frozen — schema version 1)

An artifact directory is the distributable unit of the model: everything a
consumer needs to price portfolio risk, replay validation, or extend the
build, with no access to the pipeline that produced it. This document is
the **frozen contract** for schema version 1, introduced on the v1.0
stability track (2026-08-23).

## The freeze promise

- `meta.json` carries `artifact_schema_version` (integer). Directories
  written before the freeze carry no stamp and are read as schema 1 —
  every published build back to the first release is schema-1 shaped.
- Within schema 1, the **required** files, their names, their index
  conventions, and their existing columns never change or disappear.
- New **optional** files and new columns on existing files may still be
  added; consumers must ignore columns they don't recognize.
- A breaking change bumps the schema version. `load_artifacts` refuses a
  directory stamped newer than the schema it understands, with an error
  telling the user to upgrade the package.

Units, everywhere: volatilities are **annualized decimals** (0.20 =
20%/yr), covariances are annualized variance units, returns are plain
decimals per period. Asset identifiers are exchange tickers (the joint
key across all files); factor names match `meta.json`'s `config` and the
`/api/v1/meta` factor list (`market`, nine styles, `ind_*` industries).

## Required files

| file | shape convention | content |
|---|---|---|
| `exposures.parquet` | assets × factors, ticker index | final-date exposure matrix: `market` ≡ 1, styles cross-sectionally standardized, `ind_*` 0/1 dummies |
| `factor_covariance.parquet` | K × K, factor index/columns | annualized factor covariance, after Newey-West, eigenvalue floor, correlation blending, and VRA |
| `specific_risk.parquet` | assets × 1, ticker index | column `specific_vol`: annualized specific (idiosyncratic) volatility per asset |
| `factor_returns.parquet` | dates × factors, DatetimeIndex | daily cross-sectional WLS factor returns (per estimation period; weekly rows before v0.5 history was recut) |
| `meta.json` | JSON object | build provenance — see below |

`meta.json` required keys: `artifact_schema_version`, `model_version`,
`as_of`, `n_assets`, `n_estimation`, `n_periods`, `mean_r2`, `config`
(the full `ModelConfig.to_dict()`, including every half-life, gate, and
adjustment switch needed to interpret or replay the build). Additional
keys (`n_weeks`, `vra_factor`, `vra_specific`, coverage counters, variant
provenance like `derived_from`) are informational and may grow.

## Optional files

Optional files may be absent from older or reduced artifact directories;
`load_artifacts` returns `None` for a missing one. They are what makes a
build replayable and auditable, and the hosted validation panel runs off
them.

| file | shape convention | content |
|---|---|---|
| `residuals.parquet` | dates × tickers, DatetimeIndex | daily WLS regression residuals for estimation-universe names (capture-forward history) |
| `asset_meta.parquet` | assets × 6, ticker index | `in_estimation` (bool), `history_weeks`, `specific_blend_weight`, `specific_ts`, `specific_structural`, `industry` |
| `fundamentals_store.parquet` | long format | distilled point-in-time EDGAR XBRL: `ticker`, `field`, `end`, `filed`, `val` |
| `validation.parquet` | long format | in-loop forecast scores: `date`, `portfolio`, `group`, `forecast_vol_ann`, `realized_ret`, `z`, `realized_vol_ann` |
| `exposure_history.parquet` | long format | weekly formation exposures: `ticker`, `date`, `mktcap`, then one column per style |
| `factor_tstats.parquet` | dates × factors, DatetimeIndex | daily WLS t-statistics per factor (the relevance panel) |

## Reading and writing

`riskprism.artifacts` is the only reader/writer:

```python
from riskprism.artifacts import load_artifacts, save_artifacts, SCHEMA_VERSION

a = load_artifacts("artifacts")          # dict of DataFrames + meta
a["meta"]["artifact_schema_version"]     # 1
```

`save_artifacts` stamps the current `SCHEMA_VERSION` on every write —
including derived variants (`riskprism-variant` routes through it).
Distribution is a gzipped tarball of the directory (flat, no leading
folder): `riskprism-artifacts.tar.gz` (medium horizon) and
`riskprism-artifacts-sh.tar.gz` (short horizon) on each `model-*` GitHub
release. See the model registry (`/api/v1/registry`,
`riskprism.registry`) for the published catalog.
