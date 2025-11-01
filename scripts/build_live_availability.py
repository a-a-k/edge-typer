#!/usr/bin/env python3
"""
Build a live availability table (entrypoint × p_fail) from Locust CSV exports.

Inputs
------
--stats:    locust_stats.csv (required)
--failures: locust_failures.csv (optional)
--typed:    availability_typed.csv (optional; used to infer the p_fail grid)
--replica:  replicate-* name to stamp into the output (optional)
--p-grid:   comma-separated list like "0.1,0.3,0.5,0.7,0.9" (fallback)
--out:      path to write live_availability.csv

Output schema
-------------
replica, entrypoint, p_fail, R_live
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import csv

import math
import pandas as pd


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _infer_p_grid(typed_path: Optional[Path], p_grid_arg: Optional[str]) -> List[float]:
    # Prefer the exact p_fail grid present in availability_typed.csv (so joins match).
    if typed_path and typed_path.exists():
        try:
            t = pd.read_csv(typed_path)
            if "p_fail" in t.columns and not t["p_fail"].dropna().empty:
                vals = sorted(float(x) for x in sorted(t["p_fail"].unique()))
                return vals
        except Exception:
            pass
    # Then fallback to explicit argument if provided
    if p_grid_arg:
        vals = []
        for tok in p_grid_arg.split(","):
            tok = tok.strip()
            if tok:
                try:
                    vals.append(float(tok))
                except Exception:
                    pass
        if vals:
            return sorted(vals)
    # Finally, the default grid used by the model/CLI.
    return [0.1, 0.3, 0.5, 0.7, 0.9]


def _compute_failures_by_name(failures_df: Optional[pd.DataFrame]) -> Dict[str, int]:
    """
    Locust failures.csv typically has columns like: Method, Name/Request, Type, Error, Occurrences/Count.
    We aggregate by the Request/Name column into a {entrypoint -> failures} map.
    """
    if failures_df is None or failures_df.empty:
        return {}
    name_col = _pick_col(failures_df, ["Name", "Request", "name", "request"])
    cnt_col  = _pick_col(failures_df, ["Occurrences", "Count", "count", "occurrences"])
    if not name_col or not cnt_col:
        return {}
    grp = failures_df.groupby(name_col)[cnt_col].sum()
    out: Dict[str, int] = {}
    for k, v in grp.items():
        try:
            out[str(k)] = int(v)
        except Exception:
            continue
    return out


def _extract_live_by_entrypoint(stats_df: pd.DataFrame,
                                failures_map: Dict[str, int]) -> Dict[str, float]:
    """
    Compute R_live per entrypoint from stats_df (+ optional failures_map).
    We try to read Requests and Failures columns from stats_df;
    if failures column is missing, we use failures_map (from failures.csv).
    """
    name_col = _pick_col(stats_df, ["Name", "name", "Request"])
    req_col  = _pick_col(stats_df, ["Requests", "requests", "# requests", "Request Count", "Count"])
    fail_col = _pick_col(stats_df, ["Failures", "failures", "# failures"])

    if not name_col or not req_col:
        return {}

    # Drop the "Total" summary row if present
    df = stats_df.copy()
    df[name_col] = df[name_col].astype(str)
    df = df[df[name_col].str.lower() != "total"]

    live: Dict[str, float] = {}
    for _, row in df.iterrows():
        name = str(row[name_col])
        try:
            total = float(row[req_col])
        except Exception:
            continue
        if not math.isfinite(total) or total <= 0:
            continue
        if fail_col and fail_col in row:
            try:
                bad = float(row[fail_col])
            except Exception:
                bad = None
        else:
            bad = None
        if bad is None:
            bad = float(failures_map.get(name, 0))
        bad = max(0.0, bad)
        good = max(0.0, total - bad)
        R = good / total if total > 0 else 0.0
        # Bound to [0,1]
        R = min(1.0, max(0.0, R))
        live[name] = R
    return live


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats",    required=True, type=Path, help="locust_stats.csv")
    ap.add_argument("--failures", required=False, type=Path, help="locust_failures.csv (optional)")
    ap.add_argument("--typed",    required=False, type=Path, help="availability_typed.csv to infer p_fail grid")
    ap.add_argument("--replica",  required=False, type=str,   help="replicate-* name to stamp into 'replica' column")
    ap.add_argument("--p-grid",   required=False, type=str,   help="explicit grid, e.g. '0.1,0.3,0.5,0.7,0.9'")
    ap.add_argument("--out",      required=True,  type=Path,  help="output live_availability.csv")
    args = ap.parse_args()

    stats_df = _read_csv(args.stats)
    if stats_df is None or stats_df.empty:
        # Be quiet and exit 0 to keep the workflow optional
        return
    failures_df = _read_csv(args.failures) if args.failures else None
    failures_map = _compute_failures_by_name(failures_df)

    live_by_ep = _extract_live_by_entrypoint(stats_df, failures_map)
    if not live_by_ep:
        return

    grid = _infer_p_grid(args.typed, args.p_grid)
    replica = args.replica or ""

    out_rows = []
    for ep, rlive in live_by_ep.items():
        for p in grid:
            out_rows.append({
                "replica": replica,
                "entrypoint": ep,
                "p_fail": float(p),
                "R_live": float(rlive),
            })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["replica", "entrypoint", "p_fail", "R_live"])
        w.writeheader()
        w.writerows(out_rows)


if __name__ == "__main__":
    main()
