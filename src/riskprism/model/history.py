"""Capture-forward history: incremental appends and delisting handling.

Survivorship bias only contaminates history recorded before you started
recording. Each weekly build appends its new factor returns and residuals
to the previous build's, so a stock that was alive in June contributes to
June's cross-section forever — even after it delists in September. With
13/26-week EWMA half-lives, the biased cold-start history decays out of
the model automatically within ~18-24 months.
"""

import pandas as pd

from riskprism.config import ModelConfig


def delisting_return(last_price: float, config: ModelConfig) -> float:
    """Imputed final-week return for a name that stops trading.

    No free source distinguishes mergers (holders get paid) from failures
    (holders get wiped), so we use the price heuristic behind Shumway
    (1997): cheap stocks that vanish are performance delistings (~-30%);
    expensive stocks that vanish are overwhelmingly acquisitions (~0
    surprise beyond the last traded price, which already reflects terms).
    """
    if last_price < config.delist_failure_price:
        return config.delist_failure_return
    return 0.0


def merge_history(
    prior: pd.DataFrame | None,
    new: pd.DataFrame,
    cap_weeks: int,
) -> pd.DataFrame:
    """Append new weekly rows to a prior panel.

    Rows present in both keep the NEW values (a re-run of the same week
    supersedes the old one); columns are unioned (new assets/factors
    appear, dead ones keep their historical rows); the result is trimmed
    to the trailing ``cap_weeks`` rows.
    """
    if prior is None or prior.empty:
        merged = new
    else:
        keep_prior = prior.loc[~prior.index.isin(new.index)]
        merged = pd.concat([keep_prior, new], axis=0).sort_index()
    return merged.tail(cap_weeks)
