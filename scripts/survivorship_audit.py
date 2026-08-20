"""Quantify survivorship bias in the model window from SEC bulk archives.

Population: all XBRL filers in companyfacts.zip (operating companies,
including recently dead ones — EDGAR archives don't forget). For each,
the submissions archive gives the last filing date; a company whose
filings stop mid-window and stay stopped is a departure our price panel
cannot see (no delisted-price source). Their last reported book equity
sizes the missing mass against the live universe.

Usage: .venv/bin/python3 scripts/survivorship_audit.py
Reads  ~/.cache/riskprism/edgar/bulk_facts.zip and bulk_subs.zip.
"""

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path.home() / ".cache" / "riskprism" / "edgar"
WINDOW_START = pd.Timestamp("2023-01-01")
TODAY = pd.Timestamp.today().normalize()
GRACE_DAYS = 180  # no filing for this long => treated as ceased


def last_equity(facts: dict) -> float:
    for tax, tag in [("us-gaap", "StockholdersEquity"), ("ifrs-full", "Equity")]:
        node = facts.get("facts", {}).get(tax, {}).get(tag)
        if not node:
            continue
        vals = [e for u in node.get("units", {}).values() for e in u if "val" in e]
        if vals:
            return float(max(vals, key=lambda e: e.get("end", ""))["val"])
    return np.nan


def main() -> None:
    zf = zipfile.ZipFile(CACHE / "bulk_facts.zip")
    zs = zipfile.ZipFile(CACHE / "bulk_subs.zip")
    sub_members = set(zs.namelist())

    rows = []
    members = [m for m in zf.namelist() if m.startswith("CIK")]
    print(f"XBRL filer population: {len(members)}")
    for i, member in enumerate(members):
        if member not in sub_members:
            continue
        try:
            sub = json.loads(zs.read(member))
        except Exception:
            continue
        dates = sub.get("filings", {}).get("recent", {}).get("filingDate", [])
        if not dates:
            continue
        last = pd.Timestamp(max(dates))
        first = pd.Timestamp(min(dates))
        rows.append({
            "cik": member[3:13],
            "tickers": bool(sub.get("tickers")),
            "sic": sub.get("sic") or None,
            "first_filed": first,
            "last_filed": last,
            "member": member,
        })
        if (i + 1) % 4000 == 0:
            print(f"  scanned {i + 1}/{len(members)}")

    df = pd.DataFrame(rows)
    cutoff = TODAY - pd.Timedelta(days=GRACE_DAYS)
    df["alive_at_start"] = df["first_filed"] < WINDOW_START
    df["ceased_in_window"] = (df["last_filed"] >= WINDOW_START) & (df["last_filed"] < cutoff)
    df["active_now"] = df["last_filed"] >= cutoff

    base = df[df["alive_at_start"]]
    ceased = base[base["ceased_in_window"]]
    print("\n=== Survivorship audit ===")
    print(f"window: {WINDOW_START.date()} .. {TODAY.date()} (grace {GRACE_DAYS}d)")
    print(f"XBRL filers alive at window start: {len(base)}")
    print(f"  still filing: {int(base['active_now'].sum())}")
    print(f"  ceased during window: {len(ceased)} "
          f"({len(ceased) / len(base):.1%} of start population, "
          f"~{len(ceased) / len(base) / ((TODAY - WINDOW_START).days / 365.25):.1%}/yr)")

    # size the missing mass by last reported book equity
    eq_ceased, eq_alive = [], []
    alive_sample = base[base["active_now"]].sample(
        min(1500, int(base["active_now"].sum())), random_state=0)
    for member_list, out in [(ceased["member"], eq_ceased), (alive_sample["member"], eq_alive)]:
        for m in member_list:
            try:
                out.append(last_equity(json.loads(zf.read(m))))
            except Exception:
                out.append(np.nan)
    eq_c = pd.Series(eq_ceased).dropna()
    eq_a = pd.Series(eq_alive).dropna()
    eq_c, eq_a = eq_c[eq_c > 0], eq_a[eq_a > 0]
    print(f"\nlast reported book equity — ceased median: ${eq_c.median() / 1e6:,.0f}M, "
          f"alive median: ${eq_a.median() / 1e6:,.0f}M")
    total_c, est_total_a = eq_c.sum(), eq_a.sum() * len(base[base['active_now']]) / max(len(eq_a), 1)
    share = total_c / (total_c + est_total_a)
    print(f"ceased filers' share of total book equity: ~{share:.2%}")
    print(f"median ceased name sits at the "
          f"{(eq_a < eq_c.median()).mean():.0%}th percentile of the living")

    # order-of-magnitude bias bound on weekly factor-return means:
    # (weight share of missing names) x (delisting abnormal return)
    for dl in (-0.30, -0.55):
        print(f"upper-bound cap-weighted return-mean bias @ {dl:+.0%} delisting return: "
              f"~{share * abs(dl) / ((TODAY - WINDOW_START).days / 7):.4%}/week")
    print("\n(Covariances, which the model actually ships, are affected at second "
          "order; return means are the first-order casualty.)")


if __name__ == "__main__":
    main()
