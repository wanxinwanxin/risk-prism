"""In-loop forecast validation: the model grades itself as it builds.

At each historical week t the weekly loop already has that week's
exposures. This module maintains a point-in-time risk state (EWMA factor
covariance + EWMA specific risk, warmed only on data through t), issues
volatility forecasts for a panel of test portfolios, and scores them
against week t+1's realized returns:

    z_t = realized_return / forecast_weekly_vol

For a calibrated model z is standard normal: the **bias statistic**
std(z) should be ~1 (>1 = risk underforecast, <1 = overforecast), and
|z| > 1.96 should happen ~5% of the time. Results ship with every
release as validation.parquet.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from riskprism.config import MARKET_FACTOR, STYLE_FACTORS, ModelConfig
from riskprism.factors.industry import INDUSTRIES, INDUSTRY_PREFIX

FULL_FACTORS = [MARKET_FACTOR, *STYLE_FACTORS,
                *[f"{INDUSTRY_PREFIX}{i}" for i in INDUSTRIES]]

_MIN_WARMUP_WEEKS = 26
_RANDOM_PORTFOLIOS = 3
_RANDOM_NAMES = 50


@dataclass
class RunningRiskState:
    """Incrementally updated factor covariance and specific risk.

    Same estimator family as the batch model (zero-mean EWMA, separate
    vol/correlation half-lives) in recursive form, so a forecast at week
    t uses exactly the information available at week t.
    """

    config: ModelConfig
    n_weeks: int = 0
    _s_vol: np.ndarray = field(default=None, repr=False)
    _s_corr: np.ndarray = field(default=None, repr=False)
    _spec_var: dict = field(default_factory=dict, repr=False)
    _spec_obs: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        k = len(FULL_FACTORS)
        if self._s_vol is None:
            self._s_vol = np.zeros((k, k))
            self._s_corr = np.zeros((k, k))
        self._lam_vol = 0.5 ** (1.0 / self.config.vol_half_life)
        self._lam_corr = 0.5 ** (1.0 / self.config.corr_half_life)
        self._lam_spec = 0.5 ** (1.0 / self.config.specific_half_life)

    def update(self, factor_returns: pd.Series, residuals: pd.Series) -> None:
        f = factor_returns.reindex(FULL_FACTORS).fillna(0.0).to_numpy()
        ff = np.outer(f, f)
        self._s_vol = self._lam_vol * self._s_vol + (1 - self._lam_vol) * ff
        self._s_corr = self._lam_corr * self._s_corr + (1 - self._lam_corr) * ff
        for ticker, e in residuals.items():
            if not np.isfinite(e):
                continue
            prev = self._spec_var.get(ticker, e * e)
            self._spec_var[ticker] = self._lam_spec * prev + (1 - self._lam_spec) * e * e
            self._spec_obs[ticker] = self._spec_obs.get(ticker, 0) + 1
        self.n_weeks += 1

    def warm_up(self, factor_returns: pd.DataFrame, residuals: pd.DataFrame | None) -> None:
        """Replay a prior build's history (incremental builds)."""
        for date, row in factor_returns.iterrows():
            res = residuals.loc[date].dropna() if residuals is not None and date in residuals.index \
                else pd.Series(dtype=float)
            self.update(row, res)

    @property
    def ready(self) -> bool:
        return self.n_weeks >= max(_MIN_WARMUP_WEEKS, self.config.vol_half_life)

    def factor_cov_weekly(self) -> np.ndarray:
        d = np.sqrt(np.clip(np.diag(self._s_corr), 1e-18, None))
        corr = self._s_corr / np.outer(d, d)
        np.clip(corr, -1.0, 1.0, out=corr)
        np.fill_diagonal(corr, 1.0)
        vols = np.sqrt(np.clip(np.diag(self._s_vol), 0, None))
        cov = corr * np.outer(vols, vols)
        cov = (cov + cov.T) / 2
        vals, vecs = np.linalg.eigh(cov)
        vals = np.clip(vals, 0, None)
        return vecs @ np.diag(vals) @ vecs.T

    def specific_var_weekly(self, tickers: pd.Index) -> pd.Series:
        min_obs = self.config.min_specific_obs
        own = pd.Series(
            [self._spec_var.get(t, np.nan)
             if self._spec_obs.get(t, 0) >= min_obs else np.nan for t in tickers],
            index=tickers,
        )
        return own.fillna(own.median())


def test_portfolios(exposures: pd.DataFrame, industries: pd.Series,
                    mktcap: pd.Series, week_index: int) -> dict[str, pd.Series]:
    """Weight vectors for the validation panel, from week-t information only."""
    idx = exposures.index
    caps = mktcap.reindex(idx)
    valid = caps.notna() & (caps > 0)
    idx = idx[valid]
    caps = caps[idx]
    if len(idx) < 100:
        return {}
    ports: dict[str, pd.Series] = {}
    ports["market"] = caps / caps.sum()
    ports["equal_weight"] = pd.Series(1.0 / len(idx), index=idx)
    for f in STYLE_FACTORS:
        e = exposures.loc[idx, f]
        q1, q4 = e.quantile(0.2), e.quantile(0.8)
        top, bot = idx[e >= q4], idx[e <= q1]
        if len(top) < 10 or len(bot) < 10:
            continue
        w = pd.Series(0.0, index=idx)
        w[top], w[bot] = 1.0 / len(top), -1.0 / len(bot)
        ports[f"style_{f}"] = w
    inds = industries.reindex(idx)
    for name, members in idx.groupby(inds).items():
        if name == "Other" or len(members) < 15:
            continue
        w = pd.Series(0.0, index=idx)
        w[members] = caps[members] / caps[members].sum()
        ports[f"industry_{name}"] = w
    for k in range(_RANDOM_PORTFOLIOS):
        rng = np.random.default_rng(10_000 * (k + 1) + week_index)
        picks = rng.choice(len(idx), size=min(_RANDOM_NAMES, len(idx)), replace=False)
        w = pd.Series(0.0, index=idx)
        w.iloc[picks] = 1.0 / len(picks)
        ports[f"random_{k + 1}"] = w
    return ports


def _group(name: str) -> str:
    return name.split("_")[0] if "_" in name else name


def score_portfolios(
    state: RunningRiskState,
    exposures_full: pd.DataFrame,  # incl. market + industry columns, FULL_FACTORS order
    industries: pd.Series,
    mktcap: pd.Series,
    realized: pd.Series,           # week t+1 returns
    date: pd.Timestamp,
    week_index: int,
) -> list[dict]:
    """Forecast each test portfolio's vol at t and score against t+1."""
    if not state.ready:
        return []
    X = exposures_full.reindex(columns=FULL_FACTORS).fillna(0.0)
    tradable = realized.reindex(X.index)
    ok = np.isfinite(tradable.astype(float))
    X, tradable = X.loc[ok], tradable[ok]
    F = state.factor_cov_weekly()
    svar = state.specific_var_weekly(X.index)
    rows = []
    for name, w in test_portfolios(X, industries, mktcap, week_index).items():
        w = w.reindex(X.index).fillna(0.0)
        x = X.to_numpy().T @ w.to_numpy()
        var_w = float(x @ F @ x) + float((w.to_numpy() ** 2 * svar.to_numpy()).sum())
        if var_w <= 0:
            continue
        vol_w = np.sqrt(var_w)
        r = float((w * tradable).sum())
        rows.append({
            "date": date, "portfolio": name, "group": _group(name),
            "forecast_vol_ann": vol_w * np.sqrt(52.0),
            "realized_ret": r,
            "z": r / vol_w,
        })
    return rows


def merge_validation(prior: pd.DataFrame | None, new: pd.DataFrame,
                     cap_weeks: int) -> pd.DataFrame:
    if prior is None or prior.empty:
        merged = new
    elif new.empty:
        merged = prior
    else:
        keep = prior[~prior["date"].isin(set(new["date"]))]
        merged = pd.concat([keep, new], ignore_index=True)
    if merged.empty:
        return merged
    dates = sorted(merged["date"].unique())[-cap_weeks:]
    sort_cols = ["date"] + [c for c in ("portfolio", "ticker") if c in merged.columns]
    return merged[merged["date"].isin(dates)].sort_values(sort_cols).reset_index(drop=True)


def validation_summary(validation: pd.DataFrame, min_obs: int = 30) -> pd.DataFrame:
    """Per-portfolio calibration: bias statistic and tail coverage."""
    if validation is None or validation.empty:
        return pd.DataFrame(columns=["portfolio", "group", "n", "bias_stat", "exceed_95"])
    out = []
    for name, g in validation.groupby("portfolio"):
        if len(g) < min_obs:
            continue
        z = g["z"].to_numpy()
        out.append({
            "portfolio": name,
            "group": g["group"].iloc[0],
            "n": len(g),
            "bias_stat": float(np.std(z, ddof=1)),
            "exceed_95": float((np.abs(z) > 1.96).mean()),
            "mean_forecast_vol": float(g["forecast_vol_ann"].mean()),
        })
    return pd.DataFrame(out).sort_values(["group", "portfolio"]).reset_index(drop=True)
