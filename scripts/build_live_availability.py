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


def _extract_live_counts(stats_df: pd.DataFrame,
                           failures_map: Dict[str, int]) -> Dict[str, tuple[float, float]]:
    """
    Return per-request totals and bad counts: {request_name -> (total, bad)}.
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

    out: Dict[str, tuple[float, float]] = {}
    for _, row in df.iterrows():
        name = str(row[name_col])
        try:
            total = float(pd.to_numeric(row[req_col], errors="coerce") or 0.0)
        except Exception:
            continue
        if not math.isfinite(total) or total <= 0:
            continue
        if fail_col and fail_col in row:
            try:
                bad = float(pd.to_numeric(row[fail_col], errors="coerce") or 0.0)
            except Exception:
                bad = 0.0
        else:
            bad = float(failures_map.get(name, 0) or 0.0)
        bad = max(0.0, min(bad, total))
        out[name] = (total, bad)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats",    required=True, type=Path, help="locust_stats.csv")
    ap.add_argument("--failures", required=False, type=Path, help="locust_failures.csv (optional)")
    ap.add_argument("--typed",    required=False, type=Path, help="availability_typed.csv to infer p_fail grid")
    ap.add_argument("--replica",  required=False, type=str,   help="replicate-* name to stamp into 'replica' column")
    ap.add_argument("--p-grid",   required=False, type=str,   help="explicit grid, e.g. '0.1,0.3,0.5,0.7,0.9'")
    ap.add_argument("--targets",  required=False, type=Path,  help="optional targets.yaml with entrypoint mapping (regex)")
    ap.add_argument("--entrypoints", required=False, type=Path, help="optional entrypoints (CSV with column 'entrypoint' or newline-delimited .txt) to filter")
    ap.add_argument("--out",      required=True,  type=Path,  help="output live_availability.csv")
    # Strict mode: fail fast on missing/empty inputs
    ap.add_argument("--strict", dest="strict", action="store_true", help="fail if inputs are missing/empty", default=True)
    ap.add_argument("--no-strict", dest="strict", action="store_false", help="silently skip on missing/empty inputs")
    args = ap.parse_args()

    def _bail(msg: str, code: int = 1):
        import sys
        print(f"[availability-live] ERROR: {msg}", file=sys.stderr)
        if args.strict:
            raise SystemExit(code)
        return

    stats_df = _read_csv(args.stats)
    if stats_df is None or stats_df.empty:
        return _bail(f"stats CSV is missing or empty: {args.stats}", code=2)
    failures_df = _read_csv(args.failures) if args.failures else None
    failures_map = _compute_failures_by_name(failures_df)

    # Extract per-request totals and bad counts
    live_counts_by_name = _extract_live_counts(stats_df, failures_map)
    if not live_counts_by_name:
        return _bail("no valid per-request totals could be extracted from stats/failures CSVs", code=3)

    # Map request names to entrypoints; default: everything → 'load-generator'
    name_to_ep: Dict[str, str] = {}
    eps_filter: Optional[set[str]] = None
    if args.entrypoints and args.entrypoints.exists():
        try:
            df_eps = pd.read_csv(args.entrypoints)
            col = "entrypoint" if "entrypoint" in df_eps.columns else df_eps.columns[0]
            eps_filter = set(str(x) for x in df_eps[col].dropna().astype(str).tolist())
        except Exception:
            # fallback to newline-delimited .txt
            try:
                eps_filter = set(ln.strip() for ln in args.entrypoints.read_text(encoding="utf-8").splitlines() if ln.strip())
            except Exception:
                eps_filter = None
    if args.targets and args.targets.exists():
        import yaml
        try:
            cfg = yaml.safe_load(args.targets.read_text()) or {}
            rules: list[tuple[str, str]] = []
            for ep, plist in (cfg.get("entrypoints") or {}).items():
                for rule in (plist or []):
                    pat = str(rule.get("name_regex") or rule.get("re") or ".*")
                    rules.append((str(ep), pat))
            import re
            for name in live_counts_by_name.keys():
                ep_hit: Optional[str] = None
                for ep, pat in rules:
                    try:
                        if re.search(pat, name):
                            ep_hit = ep; break
                    except Exception:
                        try:
                            if re.search(pat, name, re.I):
                                ep_hit = ep; break
                        except Exception:
                            continue
                name_to_ep[name] = ep_hit or "load-generator"
        except Exception:
            name_to_ep = {name: "load-generator" for name in live_counts_by_name.keys()}
    else:
        name_to_ep = {name: "load-generator" for name in live_counts_by_name.keys()}

    # Aggregate by mapped entrypoint and optionally filter using WEIGHTED totals
    ep_totals: Dict[str, float] = {}
    ep_bad: Dict[str, float] = {}
    for name, (total, bad) in live_counts_by_name.items():
        ep = name_to_ep.get(name, "load-generator")
        if eps_filter is not None and ep not in eps_filter:
            continue
        ep_totals[ep] = ep_totals.get(ep, 0.0) + float(total)
        ep_bad[ep]    = ep_bad.get(ep, 0.0) + float(bad)

    live_by_ep: Dict[str, float] = {}
    for ep, tot in ep_totals.items():
        bad = ep_bad.get(ep, 0.0)
        R = 0.0 if tot <= 0 else max(0.0, min(1.0, 1.0 - (bad / tot)))
        live_by_ep[ep] = R

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
    if not out_rows:
        return _bail("no entrypoints produced live availability rows (check targets/entrypoints filters)", code=4)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["replica", "entrypoint", "p_fail", "R_live"])
        w.writeheader()
        w.writerows(out_rows)


if __name__ == "__main__":
    main()
