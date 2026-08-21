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
from riskprism.model.covariance import (eigen_adjust, eigen_bias_profile,
                                        pca_blend_corr)
from riskprism.model.specific import bayes_shrink_specific

FULL_FACTORS = [MARKET_FACTOR, *STYLE_FACTORS,
                *[f"{INDUSTRY_PREFIX}{i}" for i in INDUSTRIES]]

_MIN_WARMUP_WEEKS = 26
_RANDOM_PORTFOLIOS = 3
_RANDOM_NAMES = 50


@dataclass
class RunningRiskState:
    """Incrementally updated factor covariance and specific risk.

    Same estimator family as the batch model (zero-mean EWMA, separate
    vol/correlation half-lives, Newey-West variance adjustment, Bayesian
    specific shrinkage, Volatility Regime Adjustment) in recursive form,
    so a forecast at week t uses exactly the information available at
    week t.
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
        self._lam_vra = 0.5 ** (1.0 / self.config.vra_half_life)
        # Newey-West: per-factor lagged autocovariances (vol half-life)
        self._nw_g = np.zeros((self.config.nw_factor_lags, k))
        self._f_hist: list[np.ndarray] = []
        # specific lag-1 autocovariance and previous residual, per ticker
        self._spec_g1: dict = {}
        self._spec_prev: dict = {}
        # Volatility Regime Adjustment: EWMA of squared cross-sectional
        # bias statistics; multiplier lambda = sqrt of these
        self._vra_f2 = 1.0
        self._vra_s2 = 1.0
        # eigenfactor bias profile cache (recomputed periodically)
        self._eigen_v = None
        self._eigen_at = -10_000

    # ---- volatility regime multipliers ---------------------------------
    @property
    def vra_factor(self) -> float:
        c = self.config
        return float(np.clip(np.sqrt(self._vra_f2), c.vra_lambda_min, c.vra_lambda_max))

    @property
    def vra_specific(self) -> float:
        c = self.config
        return float(np.clip(np.sqrt(self._vra_s2), c.vra_lambda_min, c.vra_lambda_max))

    def _nw_factor_var(self) -> np.ndarray:
        """Per-factor weekly variance with Newey-West (Bartlett) adjustment."""
        var = np.clip(np.diag(self._s_vol), 0, None)
        L = self.config.nw_factor_lags
        adj = var + 2 * sum((1 - l / (L + 1)) * self._nw_g[l - 1] for l in range(1, L + 1))
        ratio = np.ones_like(var)
        pos = var > 0
        ratio[pos] = np.clip(adj[pos] / var[pos],
                             self.config.nw_ratio_min, self.config.nw_ratio_max)
        return var * ratio

    def _spec_var_raw(self, ticker) -> float:
        """Weekly specific variance with NW lag-1 adjustment, no shrink/VRA."""
        v = self._spec_var.get(ticker, np.nan)
        if not np.isfinite(v) or v <= 0:
            return np.nan
        if self.config.nw_specific_lags >= 1:
            g1 = self._spec_g1.get(ticker, 0.0)
            ratio = np.clip((v + g1) / v, self.config.nw_ratio_min, self.config.nw_ratio_max)
            v = v * ratio
        return v

    def update(self, factor_returns: pd.Series, residuals: pd.Series) -> None:
        f = factor_returns.reindex(FULL_FACTORS).fillna(0.0).to_numpy()
        # --- VRA: score this week's returns against the PRE-update
        # forecast (pre-VRA, so the multiplier doesn't feed back).
        # Per-element z² is winsorized at 2.5σ before averaging: the raw
        # mean is hijacked by fat tails (single-name blowups, near-zero
        # variance forecasts on quiet factors) and would bias the
        # multiplier persistently above 1; the winsorized mean still
        # targets the variance ratio (bias ~0.5% under normality) while
        # bounding any one observation's influence. USE4 needs no such
        # guard on daily data with cap-weighting; weekly + equal-weight
        # does. -----------------------------------------------------------
        z2cap = 6.25
        if self.n_weeks >= self.config.vol_half_life:
            var = self._nw_factor_var()
            ok = var > 0
            if ok.any():
                b2 = float(np.clip(f[ok] ** 2 / var[ok], 0.0, z2cap).mean())
                self._vra_f2 = self._lam_vra * self._vra_f2 + (1 - self._lam_vra) * b2
            zs = []
            for tk, e in residuals.items():
                if not np.isfinite(e) or self._spec_obs.get(tk, 0) < self.config.min_specific_obs:
                    continue
                v = self._spec_var_raw(tk)
                if np.isfinite(v) and v > 0:
                    zs.append(min(e * e / v, z2cap))
            if len(zs) >= 30:
                self._vra_s2 = (self._lam_vra * self._vra_s2
                                + (1 - self._lam_vra) * float(np.mean(zs)))
        # --- second moments ----------------------------------------------
        ff = np.outer(f, f)
        self._s_vol = self._lam_vol * self._s_vol + (1 - self._lam_vol) * ff
        self._s_corr = self._lam_corr * self._s_corr + (1 - self._lam_corr) * ff
        for l in range(1, self.config.nw_factor_lags + 1):
            if len(self._f_hist) >= l:
                self._nw_g[l - 1] = (self._lam_vol * self._nw_g[l - 1]
                                     + (1 - self._lam_vol) * f * self._f_hist[-l])
        self._f_hist.append(f)
        if len(self._f_hist) > self.config.nw_factor_lags:
            self._f_hist.pop(0)
        for ticker, e in residuals.items():
            if not np.isfinite(e):
                continue
            prev = self._spec_var.get(ticker, e * e)
            self._spec_var[ticker] = self._lam_spec * prev + (1 - self._lam_spec) * e * e
            e_prev = self._spec_prev.get(ticker)
            if e_prev is not None and np.isfinite(e_prev):
                g = self._spec_g1.get(ticker, 0.0)
                self._spec_g1[ticker] = self._lam_spec * g + (1 - self._lam_spec) * e * e_prev
            self._spec_prev[ticker] = e
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
        if self.config.factor_cov_adjust == "blend":
            corr = pca_blend_corr(corr, self.config)
        vols = np.sqrt(self._nw_factor_var())
        cov = corr * np.outer(vols, vols)
        cov = (cov + cov.T) / 2
        if self.config.factor_cov_adjust == "eigen":
            # the bias profile drifts slowly with T; refresh periodically
            # (deterministically seeded so replays are reproducible)
            if (self._eigen_v is None
                    or self.n_weeks - self._eigen_at >= self.config.eigen_refresh_weeks):
                T = min(self.n_weeks, self.config.history_cap_weeks)
                self._eigen_v = eigen_bias_profile(cov, T, self.config, seed=self.n_weeks)
                self._eigen_at = self.n_weeks
            cov = eigen_adjust(cov, self._eigen_v)
        vals, vecs = np.linalg.eigh(cov)
        vals = np.clip(vals, 0, None)
        return (vecs @ np.diag(vals) @ vecs.T) * self.vra_factor ** 2

    def specific_var_weekly(self, tickers: pd.Index,
                            size: pd.Series | None = None) -> pd.Series:
        min_obs = self.config.min_specific_obs
        own = pd.Series(
            [self._spec_var_raw(t)
             if self._spec_obs.get(t, 0) >= min_obs else np.nan for t in tickers],
            index=tickers,
        )
        var = own.fillna(own.median())
        if size is not None:
            vol = np.sqrt(var.clip(lower=0.0))
            vol = bayes_shrink_specific(vol, size.reindex(tickers), self.config)
            var = vol ** 2
        return var * self.vra_specific ** 2


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


def optimized_portfolios(X: pd.DataFrame, F: np.ndarray, svar: pd.Series,
                         mktcap: pd.Series, week_index: int,
                         config: ModelConfig) -> dict[str, pd.Series]:
    """Minimum-risk portfolios optimized AGAINST the current model — the
    documented hard case (Shepard 2009; Menchero/Wang/Orr 2011: bias
    statistics 1.4-1.5 on such portfolios under sample covariances).

    w ∝ Σ⁻¹α via Woodbury on Σ = XFXᵀ + D: the fully-invested global
    minimum-variance portfolio (α = 1) plus min-risk portfolios for
    random unit-normal alphas, over the top names by cap. Weights are
    rescaled to unit gross exposure (z-scores are scale-invariant)."""
    caps = mktcap.reindex(X.index)
    ok = (svar.reindex(X.index) > 0) & caps.notna() & (caps > 0)
    names = caps[ok].nlargest(config.opt_universe).index
    if len(names) < 100:
        return {}
    Xn = X.loc[names].to_numpy(dtype=float)
    dinv = 1.0 / svar.reindex(names).to_numpy(dtype=float)
    K = Xn.shape[1]
    try:
        Finv = np.linalg.inv(F + 1e-12 * np.eye(K))
        M = Finv + (Xn * dinv[:, None]).T @ Xn
        M_chol = np.linalg.cholesky((M + M.T) / 2)
    except np.linalg.LinAlgError:
        return {}

    def sigma_inv(alpha: np.ndarray) -> np.ndarray:
        da = dinv * alpha
        u = np.linalg.solve(M_chol.T, np.linalg.solve(M_chol, Xn.T @ da))
        return da - dinv * (Xn @ u)

    ports: dict[str, pd.Series] = {}
    alphas = {"opt_minvar": np.ones(len(names))}
    for k in range(config.opt_random_alphas):
        rng = np.random.default_rng(77_000 * (k + 1) + week_index)
        alphas[f"opt_alpha{k + 1}"] = rng.standard_normal(len(names))
    for name, alpha in alphas.items():
        w = sigma_inv(alpha)
        gross = np.abs(w).sum()
        if gross > 0 and np.isfinite(gross):
            ports[name] = pd.Series(w / gross, index=names)
    return ports


def _group(name: str) -> str:
    return name.split("_")[0] if "_" in name else name


def _realized_vol_ann(daily_returns: pd.DataFrame | None, w: pd.Series) -> float:
    """Within-week realized vol of the portfolio from daily returns, annualized.

    RV = sum of squared daily portfolio returns over the week — a direct
    (chi-squared-noisy, ~5 obs) observation of that week's variance,
    giving vol-forecast tests far more power than one return draw.
    """
    if daily_returns is None or daily_returns.empty:
        return float("nan")
    rets = daily_returns.reindex(columns=w.index).fillna(0.0)
    pr = rets.to_numpy() @ w.to_numpy()
    if len(pr) < 3:
        return float("nan")
    return float(np.sqrt((pr ** 2).sum() * 52.0))


def score_portfolios(
    state: RunningRiskState,
    exposures_full: pd.DataFrame,  # incl. market + industry columns, FULL_FACTORS order
    industries: pd.Series,
    mktcap: pd.Series,
    realized: pd.Series,           # week t+1 returns
    date: pd.Timestamp,
    week_index: int,
    daily_returns: pd.DataFrame | None = None,  # week t+1 daily returns
) -> list[dict]:
    """Forecast each test portfolio's vol at t and score against t+1."""
    if not state.ready:
        return []
    X = exposures_full.reindex(columns=FULL_FACTORS).fillna(0.0)
    tradable = realized.reindex(X.index)
    ok = np.isfinite(tradable.astype(float))
    X, tradable = X.loc[ok], tradable[ok]
    F = state.factor_cov_weekly()
    svar = state.specific_var_weekly(X.index, size=X.get("size"))
    ports = test_portfolios(X, industries, mktcap, week_index)
    ports.update(optimized_portfolios(X, F, svar, mktcap, week_index, state.config))
    rows = []
    for name, w in ports.items():
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
            "realized_vol_ann": _realized_vol_ann(daily_returns, w),
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
    has_rv = "realized_vol_ann" in validation.columns
    out = []
    for name, g in validation.groupby("portfolio"):
        if len(g) < min_obs:
            continue
        z = g["z"].to_numpy()
        row = {
            "portfolio": name,
            "group": g["group"].iloc[0],
            "n": len(g),
            "bias_stat": float(np.std(z, ddof=1)),
            "exceed_95": float((np.abs(z) > 1.96).mean()),
            "mean_forecast_vol": float(g["forecast_vol_ann"].mean()),
        }
        if has_rv:
            rv2 = g["realized_vol_ann"].dropna() ** 2
            fc2 = g.loc[rv2.index, "forecast_vol_ann"] ** 2
            # ratio of average realized to average forecast variance, in vol
            # units: >1 means the model underforecast this portfolio's vol
            row["vol_ratio"] = (float(np.sqrt(rv2.mean() / fc2.mean()))
                                if len(rv2) >= min_obs and fc2.mean() > 0 else np.nan)
            row["mean_realized_vol"] = float(np.sqrt(rv2.mean())) if len(rv2) else np.nan
        out.append(row)
    return pd.DataFrame(out).sort_values(["group", "portfolio"]).reset_index(drop=True)
