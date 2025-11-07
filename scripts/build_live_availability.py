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
from typing import List, Dict, Optional, Tuple
from collections import Counter
import csv
import re

import math
import pandas as pd

_ENTRYPOINT_PREFIX = re.compile(r"entry:([^:]+):")


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


def _normalize_name(value: str) -> str:
    """Match the normalization used in entrypoints-from-locust (strip whitespace)."""
    return str(value).strip()


def _clean_numeric(val) -> float:
    if isinstance(val, str):
        val = val.replace(",", "").strip()
    try:
        num = pd.to_numeric(val, errors="coerce")
    except Exception:
        return 0.0
    try:
        f = float(num)
    except Exception:
        return 0.0
    if not math.isfinite(f):
        return 0.0
    return f


def _compute_failures_by_name(
    failures_df: Optional[pd.DataFrame],
    method_hint: Optional[str],
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Aggregate failures.csv by (method, name) and bucket errors (5xx/timeout/socket/other).
    Returns {(method,name) -> bucket_counts}.
    """
    if failures_df is None or failures_df.empty:
        return {}
    name_col = _pick_col(failures_df, ["Name", "Request", "name", "request"])
    cnt_col  = _pick_col(failures_df, ["Occurrences", "Count", "count", "occurrences"])
    err_col  = _pick_col(failures_df, ["Error", "error", "Message"])
    meth_col = method_hint if method_hint and method_hint in failures_df.columns else _pick_col(failures_df, ["Method", "method"])
    if not name_col or not cnt_col:
        return {}

    def _bucket(msg: str) -> str:
        s = str(msg)
        if re.search(r"\b5\d\d\b", s) or re.search(r"status\s*code\s*5\d\d", s, re.I):
            return "n_5xx"
        if re.search(r"timeout|timed\s*out|readtimeout|connecttimeout", s, re.I):
            return "n_timeout"
        if re.search(r"connection|refused|reset|broken\s*pipe|socket|dns|ssl|remote|protocol", s, re.I):
            return "n_socket"
        return "n_other"

    df = failures_df.copy()
    df[name_col] = df[name_col].astype(str).map(_normalize_name)
    if meth_col:
        df[meth_col] = df[meth_col].astype(str).map(lambda s: s.upper().strip())
    df["_bucket"] = df[err_col].map(_bucket) if err_col else "n_other"
    df["_count"] = pd.to_numeric(df[cnt_col], errors="coerce").fillna(0.0)

    agg: Dict[Tuple[str, str], Dict[str, float]] = {}
    for _, row in df.iterrows():
        name = _normalize_name(row[name_col])
        if not name:
            continue
        method = str(row[meth_col]).upper().strip() if meth_col else ""
        bucket = row["_bucket"]
        entry = agg.setdefault((method, name), {"n_5xx": 0.0, "n_timeout": 0.0, "n_socket": 0.0, "n_other": 0.0})
        entry[bucket] = entry.get(bucket, 0.0) + float(row["_count"])
    return agg


def _extract_live_counts(
    stats_df: pd.DataFrame,
    failures_map: Dict[Tuple[str, str], Dict[str, float]],
) -> Tuple[Dict[str, tuple[float, Dict[str, float], str]], Optional[str]]:
    """
    Return ({request_name -> (total, bucket_counts, method)}, method_column_name_if_present)
    """
    name_col = _pick_col(stats_df, ["Name", "name", "Request"])
    req_col  = _pick_col(stats_df, ["Requests", "requests", "# requests", "Request Count", "Count"])
    fail_col = _pick_col(stats_df, ["Failures", "failures", "# failures"])
    method_col = _pick_col(stats_df, ["Method", "method"])

    if not name_col or not req_col:
        return {}, method_col

    df = stats_df.copy()
    df[name_col] = df[name_col].astype(str)
    df = df[~df[name_col].str.contains("Aggregated|Total", case=False, na=False)]

    out: Dict[str, tuple[float, Dict[str, float], str]] = {}
    for _, row in df.iterrows():
        raw_name = str(row[name_col])
        name = _normalize_name(raw_name)
        if not name:
            continue
        total = _clean_numeric(row[req_col])
        if total <= 0:
            continue
        method = ""
        if method_col and method_col in row:
            method = str(row[method_col]).upper().strip()

        if fail_col and fail_col in row:
            bad = _clean_numeric(row[fail_col])
            bad = max(0.0, min(bad, total))
            buckets = {
                "n_5xx": bad,
                "n_timeout": 0.0,
                "n_socket": 0.0,
                "n_other": 0.0,
            }
        else:
            buckets = failures_map.get((method, name)) or failures_map.get(("", name)) or {"n_5xx": 0.0, "n_timeout": 0.0, "n_socket": 0.0, "n_other": 0.0}
            total_fail = buckets.get("n_5xx", 0.0) + buckets.get("n_timeout", 0.0) + buckets.get("n_socket", 0.0)
            if total_fail > total:
                scale = total / total_fail if total_fail > 0 else 0.0
                for key in ("n_5xx", "n_timeout", "n_socket"):
                    buckets[key] *= scale

        out[name] = (total, buckets, method)
    return out, method_col


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
    stats_method_col = _pick_col(stats_df, ["Method", "method"])
    failures_map = _compute_failures_by_name(failures_df, stats_method_col)

    # Extract per-request totals and bad counts
    live_counts_by_name, _ = _extract_live_counts(stats_df, failures_map)
    if not live_counts_by_name:
        return _bail("no valid per-request totals could be extracted from stats/failures CSVs", code=3)

    eps_filter: Optional[set[str]] = None
    if args.entrypoints and args.entrypoints.exists():
        try:
            df_eps = pd.read_csv(args.entrypoints)
            col = "entrypoint" if "entrypoint" in df_eps.columns else df_eps.columns[0]
            eps_filter = set(str(x).strip() for x in df_eps[col].dropna().astype(str).tolist() if str(x).strip())
        except Exception:
            try:
                eps_filter = set(ln.strip() for ln in args.entrypoints.read_text(encoding="utf-8").splitlines() if ln.strip())
            except Exception:
                eps_filter = None
        if eps_filter is not None and not eps_filter:
            eps_filter = None

    rules: list[tuple[str, re.Pattern[str], Optional[str]]] = []
    if args.targets and args.targets.exists():
        import yaml
        try:
            cfg = yaml.safe_load(args.targets.read_text()) or {}
            for ep, plist in (cfg.get("entrypoints") or {}).items():
                for rule in (plist or []):
                    pat = str(rule.get("name_regex") or rule.get("re") or ".*")
                    meth = rule.get("method")
                    try:
                        cre = re.compile(pat)
                    except Exception:
                        cre = re.compile(pat, re.I)
                    rules.append((str(ep), cre, (str(meth).upper().strip() if meth else None)))
        except Exception:
            rules = []

    def _map_entrypoint(name: str, method: str) -> Optional[str]:
        prefix = _ENTRYPOINT_PREFIX.match(name)
        if prefix:
            return prefix.group(1).strip()
        for ep, cre, meth in rules:
            if meth and method and meth != method:
                continue
            if cre.search(name):
                return ep
        return None

    # Aggregate by mapped entrypoint and optionally filter using WEIGHTED totals
    ep_totals: Dict[str, float] = {}
    ep_bad: Dict[str, float] = {}
    matched_eps: set[str] = set()
    unmatched = Counter()
    for name, (total, buckets, method) in live_counts_by_name.items():
        ep = _map_entrypoint(name, method or "")
        if ep is None:
            unmatched[name] += max(1, int(total))
            continue
        if eps_filter is not None and ep not in eps_filter:
            continue
        matched_eps.add(ep)
        ep_totals[ep] = ep_totals.get(ep, 0.0) + float(total)
        bad = float(buckets.get("n_5xx", 0.0) + buckets.get("n_timeout", 0.0) + buckets.get("n_socket", 0.0))
        ep_bad[ep]    = ep_bad.get(ep, 0.0) + bad

    live_by_ep: Dict[str, float] = {}
    for ep, tot in ep_totals.items():
        bad = ep_bad.get(ep, 0.0)
        R = 0.0 if tot <= 0 else max(0.0, min(1.0, 1.0 - (bad / tot)))
        live_by_ep[ep] = R

    missing_eps: list[str] = []
    if eps_filter:
        missing_eps = sorted(eps_filter - matched_eps)
        if missing_eps:
            preview = ", ".join(missing_eps[:5])
            msg = (
                f"live availability missing {len(missing_eps)} entrypoints from filter "
                f"(examples: {preview}). Did Locust emit any requests for them or do the regex rules need adjusting?"
            )
            if args.strict:
                return _bail(msg, code=5)
            print(f"[availability-live] WARNING: {msg}")

    if unmatched:
        top = ", ".join(f"{name} ({cnt})" for name, cnt in unmatched.most_common(5))
        print(f"[availability-live] WARNING: ignored {sum(unmatched.values())} requests with no entrypoint mapping (e.g., {top})")

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
