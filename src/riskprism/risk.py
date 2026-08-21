"""Portfolio risk analytics on top of a built factor model.

All volatilities are annualized decimals (0.20 = 20% a year). Weights are
portfolio weights (long positive, short negative); they need not sum to 1
— net-short and leveraged books are handled naturally.
"""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from riskprism.artifacts import load_artifacts


class RiskModel:
    def __init__(
        self,
        exposures: pd.DataFrame,
        factor_covariance: pd.DataFrame,
        specific_vol: pd.Series,
        meta: dict | None = None,
        asset_meta: pd.DataFrame | None = None,
    ):
        factors = list(factor_covariance.columns)
        self.exposures = exposures.reindex(columns=factors).fillna(0.0)
        self.factor_covariance = factor_covariance
        self.specific_vol = specific_vol.reindex(exposures.index)
        self.meta = meta or {}
        self.asset_meta = asset_meta
        self.factors = factors

    @classmethod
    def load(cls, path: str | Path) -> "RiskModel":
        a = load_artifacts(path)
        return cls(a["exposures"], a["factor_covariance"], a["specific_risk"],
                   a["meta"], a.get("asset_meta"))

    def estimation_quality(self, ticker: str) -> dict | None:
        """How much of this asset's risk is measured vs. inferred from priors."""
        if self.asset_meta is None or ticker not in self.asset_meta.index:
            return None
        row = self.asset_meta.loc[ticker]
        return {
            "in_estimation_universe": bool(row["in_estimation"]),
            "residual_history_weeks": int(row["history_weeks"]),
            "specific_risk_weight_on_own_history": float(row["specific_blend_weight"]),
        }

    # ------------------------------------------------------------------

    def coverage(self, tickers: list[str]) -> dict:
        known = [t for t in tickers if t in self.exposures.index]
        return {"covered": known, "uncovered": [t for t in tickers if t not in known]}

    def _weight_vector(self, weights: Mapping[str, float]) -> tuple[pd.Series, dict]:
        weights = {str(k).upper(): float(v) for k, v in weights.items()}
        cov = self.coverage(list(weights))
        w = pd.Series(0.0, index=self.exposures.index)
        for t in cov["covered"]:
            w[t] = weights[t]
        uncovered_weight = float(sum(abs(weights[t]) for t in cov["uncovered"]))
        gross = float(sum(abs(v) for v in weights.values()))
        info = {
            "uncovered_tickers": cov["uncovered"],
            "uncovered_gross_weight": uncovered_weight,
            "coverage_ratio": 1.0 - (uncovered_weight / gross if gross else 0.0),
        }
        return w, info

    def factor_exposures(self, weights: Mapping[str, float]) -> pd.Series:
        w, _ = self._weight_vector(weights)
        return self.exposures.T @ w

    def _shepard_multiplier(self) -> float:
        """Second-order correction for portfolios optimized against this
        model (Shepard 2009): optimizers seek out the covariance matrix's
        underestimated directions, so true vol ~ predicted / (1 - K/N_eff),
        with N_eff the EWMA's effective sample size. Applied to reported
        forecasts only when the caller declares optimization — never baked
        into the matrix, which stays unbiased for pre-specified portfolios."""
        cfg = self.meta.get("config", {}) or {}
        # N_eff from the VOL half-life — the binding noise source: with it,
        # the analytic multiplier (1.09 at 84d/K=20) matches the min-var
        # bias measured in the published validation almost exactly
        hl = float(cfg.get("vol_half_life", 84))
        lam = 0.5 ** (1.0 / hl)
        n_eff = (1 + lam) / (1 - lam)
        k = len(self.factors)
        if k >= n_eff:
            return 2.0
        return float(min(1.0 / (1.0 - k / n_eff), 2.0))

    def portfolio_risk(self, weights: Mapping[str, float], top_n: int = 10,
                       optimized: bool = False) -> dict:
        """Total risk with factor/specific decomposition and contributions.

        Pass ``optimized=True`` if the weights were produced by optimizing
        against this model: the reported vols are scaled by the Shepard
        second-order correction (see the `opt` rows of the published
        validation for the empirically measured counterpart).
        """
        w, info = self._weight_vector(weights)
        X = self.exposures.to_numpy()
        F = self.factor_covariance.to_numpy()
        s2 = np.nan_to_num(self.specific_vol.to_numpy()) ** 2
        wv = w.to_numpy()

        x = X.T @ wv
        Fx = F @ x
        factor_var = float(x @ Fx)
        specific_var = float((wv**2 * s2).sum())
        total_var = factor_var + specific_var
        total_vol = float(np.sqrt(total_var))

        factor_contrib = pd.Series(x * Fx, index=self.factors)
        if total_var > 0:
            sigma_w = X @ Fx + s2 * wv
            mctr = pd.Series(sigma_w / total_vol, index=self.exposures.index)
            ctr = (w * mctr)[w != 0].sort_values(key=np.abs, ascending=False)
        else:
            ctr = pd.Series(dtype=float)

        shep = 1.0
        if optimized and (self.meta.get("config", {}) or {}).get("shepard_correction", True):
            shep = self._shepard_multiplier()
        return {
            "model_version": self.meta.get("model_version"),
            "total_vol": total_vol * shep,
            **({"total_vol_unadjusted": total_vol,
                "optimized_correction": shep} if shep != 1.0 else {}),
            "factor_vol": float(np.sqrt(factor_var)) * shep,
            "specific_vol": float(np.sqrt(specific_var)) * shep,
            "factor_var_share": factor_var / total_var if total_var else np.nan,
            "factor_exposures": {k: float(v) for k, v in zip(self.factors, x)},
            "factor_var_contributions": {
                k: float(v) for k, v in factor_contrib.sort_values(key=np.abs, ascending=False).head(top_n).items()
            },
            "top_asset_risk_contributions": {k: float(v) for k, v in ctr.head(top_n).items()},
            **info,
        }

    def stress_test(
        self, weights: Mapping[str, float], factor_shocks: Mapping[str, float]
    ) -> dict:
        """Linear P&L estimate for factor shocks (in return units, e.g. -0.10)."""
        x = self.factor_exposures(weights)
        unknown = [f for f in factor_shocks if f not in x.index]
        if unknown:
            raise ValueError(f"Unknown factors {unknown}; model factors: {list(x.index)}")
        contributions = {f: float(x[f] * shock) for f, shock in factor_shocks.items()}
        return {
            "pnl_estimate": float(sum(contributions.values())),
            "per_factor": contributions,
            "note": "First-order estimate: exposure x shock, ignoring specific returns.",
        }

    def asset_risk(self, ticker: str) -> dict:
        ticker = ticker.upper()
        if ticker not in self.exposures.index:
            raise KeyError(f"{ticker} not covered by this model build")
        x = self.exposures.loc[ticker].to_numpy()
        factor_var = float(x @ self.factor_covariance.to_numpy() @ x)
        spec = float(self.specific_vol.get(ticker, np.nan))
        out = {
            "ticker": ticker,
            "total_vol": float(np.sqrt(factor_var + spec**2)),
            "factor_vol": float(np.sqrt(factor_var)),
            "specific_vol": spec,
            "exposures": {k: float(v) for k, v in self.exposures.loc[ticker].items()},
        }
        quality = self.estimation_quality(ticker)
        if quality is not None:
            out["estimation_quality"] = quality
        return out
