"""Public-methodology baselines and vendor-protocol replications.

Two comparison blocks for the validation page, both computed from the
validation history itself (no extra build state):

**Same-harness baselines.** Every classic public vol-forecasting method
is replayed point-in-time over the *same* test portfolios and weeks the
model was scored on, then graded with the identical statistics — a true
apples-to-apples table. Commercial models can't appear here (their
forecasts are proprietary and licenses forbid published benchmarks), so
the columns are the reproducible classics:

  * ewma     — RiskMetrics EWMA (J.P. Morgan 1996), λ = 0.94, on the
               portfolio's weekly returns
  * trail    — 26-week trailing zero-mean sample vol
  * rv_ewma  — λ = 0.94 EWMA on weekly realized variance (from daily
               returns within each week)
  * ff5      — Fama-French five-factor model, returns-based: rolling
               52-week betas on the public FF5 weekly factor returns,
               EWMA factor covariance, plus residual variance

Each baseline's forecast for week t uses only data through week t-1.
Stats are computed on the portfolio-weeks where *every* method has a
forecast, so all columns share one sample.

**Vendor protocols.** The commercial vendors publish out-of-sample
results under their own test protocols. We can't run their models, but
we can run ours under their protocols:

  * MSCI Barra USE4 empirical notes (Menchero-Orr-Wang 2011): monthly
    standardized returns b = R/σ, rolling 12-month windows, mean bias
    statistic and MRAD (mean |B-1|) across a portfolio collection.
    Ideal: bias 1.0; MRAD floor ≈0.17 normal, ≈0.19 at kurtosis 3.5-4.
  * Axioma AXUS4 factsheet (2016): single-window total-risk bias
    statistics on benchmark index portfolios with a 95% confidence
    interval 1 ± 1.96/√(2T).
"""

import io
import urllib.request
import zipfile

import numpy as np
import pandas as pd

_RM_LAMBDA = 0.94        # RiskMetrics (1996) decay
_TRAIL_WEEKS = 26
_MIN_OBS = 26            # weeks a baseline must see before forecasting
_FF_WINDOW = 52          # rolling beta window for the FF5 baseline
_ANN = 52.0

FF5_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")
FF5_FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

BASELINE_LABELS = {
    "riskprism": "riskprism",
    "ewma": "RiskMetrics EWMA",
    "trail": "26w trailing vol",
    "rv_ewma": "Realized-vol EWMA",
    "ff5": "Fama-French 5F",
}

# Axioma's published check runs on benchmark index portfolios; these are
# the panel's direct analogs.
_AXIOMA_BENCHMARKS = {"SPY": "S&P 500 (SPY)", "IWM": "Russell 2000 (IWM)",
                      "market": "Cap-weighted US market"}


def fetch_ff5_weekly(url: str = FF5_URL, timeout: int = 60) -> pd.DataFrame:
    """Ken French's daily FF5 factors, compounded to W-FRI weekly returns."""
    req = urllib.request.Request(url, headers={"User-Agent": "riskprism-validation"})
    raw = urllib.request.urlopen(req, timeout=timeout).read()
    zf = zipfile.ZipFile(io.BytesIO(raw))
    txt = zf.read(zf.namelist()[0]).decode("latin-1")
    lines = txt.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip().startswith(","))
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])), index_col=0)
    df = df[pd.to_numeric(df.index, errors="coerce").notna()]
    df.index = pd.to_datetime(df.index.astype(int).astype(str))
    df = df[FF5_FACTORS].astype(float) / 100.0
    return (1 + df).resample("W-FRI").prod() - 1


# ---------------------------------------------------------------------------
# baseline forecast engines — one portfolio's ordered weekly history in,
# annualized vol forecasts out; forecast[i] uses rows < i only
# ---------------------------------------------------------------------------

def _ewma_forecasts(x2: np.ndarray) -> np.ndarray:
    """EWMA of a squared series; out[i] is the forecast made before obs i."""
    out = np.full(len(x2), np.nan)
    v, seen = np.nan, 0
    for i in range(len(x2)):
        if seen >= _MIN_OBS and np.isfinite(v):
            out[i] = v
        xi = x2[i]
        if np.isfinite(xi):
            v = xi if not np.isfinite(v) else _RM_LAMBDA * v + (1 - _RM_LAMBDA) * xi
            seen += 1
    return out


def _trailing_forecasts(r: np.ndarray) -> np.ndarray:
    out = np.full(len(r), np.nan)
    for i in range(len(r)):
        past = r[max(0, i - _TRAIL_WEEKS):i]
        past = past[np.isfinite(past)]
        if len(past) >= _MIN_OBS:
            out[i] = float((past ** 2).mean())
    return out


def _ff5_forecasts(dates: pd.DatetimeIndex, r: np.ndarray,
                   ff_weekly: pd.DataFrame) -> np.ndarray:
    """Rolling FF5 regression + EWMA factor covariance, per USE-style x'Fx."""
    ff = ff_weekly.reindex(dates)
    # point-in-time EWMA covariance of the FF factors (zero-mean),
    # warmed on the full published history before the panel starts
    warm = ff_weekly.loc[:dates[0]].iloc[:-1].tail(int(3 * _ANN))
    S = np.zeros((len(FF5_FACTORS), len(FF5_FACTORS)))
    have_S = False
    for row in warm.to_numpy():
        if np.isfinite(row).all():
            S = _RM_LAMBDA * S + (1 - _RM_LAMBDA) * np.outer(row, row)
            have_S = True
    out = np.full(len(r), np.nan)
    F = ff.to_numpy()
    for i in range(len(r)):
        lo = max(0, i - _FF_WINDOW)
        y, X = r[lo:i], F[lo:i]
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        y, X = y[ok], X[ok]
        if len(y) >= _MIN_OBS and have_S and np.isfinite(F[i]).all():
            Xc = np.column_stack([np.ones(len(y)), X])
            beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
            resid = y - Xc @ beta
            rvar = float((resid ** 2).sum() / max(len(y) - Xc.shape[1], 1))
            b = beta[1:]
            out[i] = float(b @ S @ b) + rvar
        if np.isfinite(F[i]).all():
            S = _RM_LAMBDA * S + (1 - _RM_LAMBDA) * np.outer(F[i], F[i])
            have_S = True
    return out


def baseline_forecasts(validation: pd.DataFrame,
                       ff_weekly: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per portfolio-week: annualized vol forecasts for every method.

    Returns the validation frame plus one ``fc_<model>`` column per
    baseline (annualized vol, NaN before the method's burn-in).
    """
    frames = []
    for name, g in validation.groupby("portfolio"):
        g = g.sort_values("date").reset_index(drop=True)
        r = g["realized_ret"].to_numpy(dtype=float)
        g["fc_riskprism"] = g["forecast_vol_ann"]
        g["fc_ewma"] = np.sqrt(_ewma_forecasts(r ** 2) * _ANN)
        g["fc_trail"] = np.sqrt(_trailing_forecasts(r) * _ANN)
        rv_w2 = (g["realized_vol_ann"].to_numpy(dtype=float) ** 2) / _ANN
        g["fc_rv_ewma"] = np.sqrt(_ewma_forecasts(rv_w2) * _ANN)
        if ff_weekly is not None:
            dates = pd.DatetimeIndex(g["date"])
            g["fc_ff5"] = np.sqrt(_ff5_forecasts(dates, r, ff_weekly) * _ANN)
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# same-harness comparison stats
# ---------------------------------------------------------------------------

def _model_stats(fc_ann: np.ndarray, ret: np.ndarray, rv_ann: np.ndarray) -> dict:
    fv_w = fc_ann / np.sqrt(_ANN)
    z = ret / fv_w
    fv2, rv2 = fc_ann ** 2, rv_ann ** 2
    slope, _ = np.polyfit(fv2, rv2, 1)
    r2 = float(np.corrcoef(fv2, rv2)[0, 1] ** 2)
    return {
        "bias": round(float(np.std(z, ddof=1)), 3),
        "exc": round(float((np.abs(z) > 1.96).mean()), 4),
        "vratio": round(float(np.sqrt(rv2.mean() / fv2.mean())), 3),
        "mz_slope": round(float(slope), 3),
        "mz_r2": round(r2, 3),
        "qlike": round(float((np.log(fv2) + rv2 / fv2).mean()), 3),
        "rmse": round(float(np.sqrt(((fc_ann - rv_ann) ** 2).mean())), 4),
    }


def harness_comparison(validation: pd.DataFrame,
                       ff_weekly: pd.DataFrame | None = None,
                       min_obs: int = 100) -> dict | None:
    """Identical stats for the model and every baseline on one common sample."""
    fc = baseline_forecasts(validation, ff_weekly)
    models = [m for m in BASELINE_LABELS if f"fc_{m}" in fc.columns]
    cols = [f"fc_{m}" for m in models]
    ok = fc.dropna(subset=[*cols, "realized_ret", "realized_vol_ann"])
    ok = ok[(ok[cols] > 0).all(axis=1)]
    if len(ok) < min_obs:
        return None
    ret = ok["realized_ret"].to_numpy(dtype=float)
    rv = ok["realized_vol_ann"].to_numpy(dtype=float)
    return {
        "n_obs": int(len(ok)),
        "n_portfolios": int(ok["portfolio"].nunique()),
        "start": pd.Timestamp(ok["date"].min()).strftime("%Y-%m-%d"),
        "end": pd.Timestamp(ok["date"].max()).strftime("%Y-%m-%d"),
        "models": [
            {"key": m, "label": BASELINE_LABELS[m],
             **_model_stats(ok[f"fc_{m}"].to_numpy(dtype=float), ret, rv)}
            for m in models
        ],
    }


# ---------------------------------------------------------------------------
# vendor-protocol replications (run on the model's own forecasts)
# ---------------------------------------------------------------------------

def use4_protocol(validation: pd.DataFrame, min_windows: int = 5) -> dict | None:
    """USE4 empirical-notes protocol on our forecasts.

    Monthly standardized returns b = R_month / σ_month, where σ is the
    forecast standing at the start of the month; per-portfolio rolling
    12-month bias statistics B (eq. C2); mean bias and MRAD = mean|B-1|
    pooled across windows and portfolios.
    """
    biases = []
    n_months = set()
    for _, g in validation.groupby("portfolio"):
        g = g.sort_values("date")
        per = pd.PeriodIndex(pd.DatetimeIndex(g["date"]), freq="M")
        m = g.groupby(per).agg(
            ret=("realized_ret", lambda r: float(np.prod(1 + r) - 1)),
            fv_w=("forecast_vol_ann", lambda v: float(v.iloc[0]) / np.sqrt(_ANN)),
            n_weeks=("realized_ret", "size"),
        )
        m = m.iloc[1:-1]  # first/last calendar months may be partial weeks
        if len(m) < 13:
            continue
        b = (m["ret"] / (m["fv_w"] * np.sqrt(m["n_weeks"]))).to_numpy()
        months = m.index
        n_months.update(months)
        for i in range(12, len(b) + 1):
            window = months[i - 12:i]
            if (window[-1] - window[0]).n != 11:  # calendar gap: not rolling
                continue
            biases.append(float(np.std(b[i - 12:i], ddof=1)))
    if len(biases) < min_windows:
        return None
    biases = np.array(biases)
    return {
        "n_months": int(len(n_months)),
        "n_windows": int(len(biases)),
        "mean_bias": round(float(biases.mean()), 3),
        "mrad": round(float(np.abs(biases - 1).mean()), 3),
    }


def axioma_protocol(validation: pd.DataFrame, min_obs: int = 30) -> list[dict]:
    """Axioma factsheet protocol: full-sample benchmark bias stats ± 95% CI."""
    rows = []
    for key, label in _AXIOMA_BENCHMARKS.items():
        g = validation[validation["portfolio"] == key]
        z = g["z"].to_numpy(dtype=float)
        z = z[np.isfinite(z)]
        if len(z) < min_obs:
            continue
        half = 1.96 / np.sqrt(2 * len(z))
        rows.append({"name": label, "n": int(len(z)),
                     "bias": round(float(np.std(z, ddof=1)), 3),
                     "lo": round(1 - half, 3), "hi": round(1 + half, 3)})
    return rows


def comparison_payload(validation: pd.DataFrame | None,
                       ff_weekly: pd.DataFrame | None = None,
                       fetch_ff: bool = True) -> dict | None:
    """The validation page's model-comparison block."""
    if validation is None or validation.empty:
        return None
    if ff_weekly is None and fetch_ff:
        try:
            ff_weekly = fetch_ff5_weekly()
        except Exception:
            ff_weekly = None  # FF column simply drops out
    harness = harness_comparison(validation, ff_weekly)
    if harness is None:
        return None
    out = {"harness": harness}
    vendor = {}
    if (use4 := use4_protocol(validation)) is not None:
        vendor["msci"] = use4
    if axioma := axioma_protocol(validation):
        vendor["axioma"] = axioma
    if vendor:
        out["vendor"] = vendor
    return out
