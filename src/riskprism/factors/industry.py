"""Industry classification from SIC codes.

Uses the Fama-French 12-industry scheme (Ken French data library), which is
public, stable, and avoids licensed classifications such as GICS. SIC codes
come from SEC EDGAR company submissions, so the whole chain is free to
redistribute.
"""

import math

import pandas as pd

INDUSTRY_PREFIX = "ind_"

# Fama-French 12 industries: (name, list of inclusive SIC ranges)
_FF12_RANGES: dict[str, list[tuple[int, int]]] = {
    "NoDur": [(100, 999), (2000, 2399), (2700, 2749), (2770, 2799), (3100, 3199), (3940, 3989)],
    "Durbl": [(2500, 2519), (2590, 2599), (3630, 3659), (3710, 3711), (3714, 3714),
              (3716, 3716), (3750, 3751), (3792, 3792), (3900, 3939), (3990, 3999)],
    "Manuf": [(2520, 2589), (2600, 2699), (2750, 2769), (3000, 3099), (3200, 3569),
              (3580, 3629), (3700, 3709), (3712, 3713), (3715, 3715), (3717, 3749),
              (3752, 3791), (3793, 3799), (3830, 3839), (3860, 3899)],
    "Enrgy": [(1200, 1399), (2900, 2999)],
    "Chems": [(2800, 2829), (2840, 2899)],
    "BusEq": [(3570, 3579), (3660, 3692), (3694, 3699), (3810, 3829), (7370, 7379)],
    "Telcm": [(4800, 4899)],
    "Utils": [(4900, 4949)],
    "Shops": [(5000, 5999), (7200, 7299), (7600, 7699)],
    "Hlth": [(2830, 2839), (3693, 3693), (3840, 3859), (8000, 8099)],
    "Money": [(6000, 6999)],
}

INDUSTRIES = list(_FF12_RANGES.keys()) + ["Other"]


def sic_to_industry(sic) -> str:
    """Map a SIC code to a Fama-French 12 industry name."""
    if sic is None or (isinstance(sic, float) and math.isnan(sic)):
        return "Other"
    code = int(sic)
    for name, ranges in _FF12_RANGES.items():
        for lo, hi in ranges:
            if lo <= code <= hi:
                return name
    return "Other"


def industry_dummies(industries: pd.Series) -> pd.DataFrame:
    """One-hot industry exposures with float dtype, columns prefixed ``ind_``."""
    dummies = pd.get_dummies(industries, prefix=INDUSTRY_PREFIX.rstrip("_"), dtype=float)
    return dummies
