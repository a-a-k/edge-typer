#!/usr/bin/env python3
"""
Compute live availability (entrypoint × p_fail) from Locust CSV exports using a
strict mapping between Locust Name and canonical entrypoints.

Inputs
------
--stats:       Locust *_stats.csv (required)
--failures:    Locust *_failures.csv (optional; improves failure counts)
--entrypoints: CSV/TXT listing canonical entrypoint names (one per row)
--targets:     YAML mapping entrypoint -> [{name_regex, method}] rules (mandatory)
--p-grid:      comma-separated list of p_fail labels (default: 0.1,...,0.9)
--replica:     replicate-* label to stamp into output
--strict:      fail if any entrypoint lacks live data or if Locust names lack mapping

Output schema
-------------
replica, entrypoint, p_fail, R_live
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import yaml


def _read_locust_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"[availability-live] stats CSV not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit(f"[availability-live] stats CSV {path} has no rows.")
    return df


def _pick_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    norm = {c.lower().replace(" ", "").replace("_", "").replace("#", ""): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "").replace("_", "").replace("#", "")
        if key in norm:
            return norm[key]
    return None


def _load_entrypoints(path: Path) -> List[str]:
    if not path.exists():
        raise SystemExit(f"[availability-live] entrypoints file missing: {path}")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        col = "entrypoint" if "entrypoint" in df.columns else df.columns[0]
        vals = [str(x).strip() for x in df[col].dropna().tolist() if str(x).strip()]
    else:
        vals = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not vals:
        raise SystemExit(f"[availability-live] entrypoints list is empty: {path}")
    return vals


def _load_targets(path: Path, entrypoints: List[str]) -> Dict[str, List[Dict[str, Optional[str]]]]:
    if not path.exists():
        raise SystemExit(f"[availability-live] targets YAML not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    mapping = {}
    for ep, rules in (data.get("entrypoints") or {}).items():
        specs: List[Dict[str, Optional[str]]] = []
        for rule in rules or []:
            regex = rule.get("name_regex") or rule.get("re")
            if not regex:
                continue
            method = rule.get("method")
            try:
                compiled = re.compile(regex)
            except Exception:
                compiled = re.compile(regex, re.I)
            specs.append({"regex": compiled, "method": str(method).upper() if method else None, "raw": regex})
        if specs:
            mapping[str(ep)] = specs
    missing = sorted(set(entrypoints) - set(mapping.keys()))
    if missing:
        raise SystemExit(
            "[availability-live] targets YAML lacks mapping rules for entrypoints: "
            + ", ".join(missing)
        )
    return mapping


def _infer_p_grid(p_grid_arg: Optional[str]) -> List[float]:
    if p_grid_arg:
        vals: List[float] = []
        for tok in p_grid_arg.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                vals.append(float(tok))
            except Exception:
                continue
        if vals:
            return sorted(vals)
    return [0.1, 0.3, 0.5, 0.7, 0.9]


def _clean_num(val) -> float:
    if isinstance(val, str):
        val = val.replace(",", "").strip()
    try:
        num = float(pd.to_numeric(val, errors="coerce") or 0.0)
    except Exception:
        return 0.0
    if not math.isfinite(num):
        return 0.0
    return num


def _match_entrypoint(name: str, method: str, mapping: Dict[str, List[Dict[str, Optional[str]]]]) -> Optional[str]:
    prefix = re.match(r"entry:([^:]+):", name)
    if prefix:
        token = prefix.group(1).strip()
        return token if token in mapping else None
    for ep, rules in mapping.items():
        for rule in rules:
            if rule["method"] and rule["method"] != method:
                continue
            if rule["regex"].search(name):
                return ep
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True, type=Path, help="locust_stats.csv")
    ap.add_argument("--failures", type=Path, help="locust_failures.csv")
    ap.add_argument("--entrypoints", required=True, type=Path, help="entrypoints.csv/txt from traces")
    ap.add_argument("--targets", required=True, type=Path, help="live_targets.yaml mapping Locust Name → entrypoint")
    ap.add_argument("--replica", type=str, default="", help="replicate-* label")
    ap.add_argument("--p-grid", type=str, help="comma-separated list of p_fail labels")
    ap.add_argument("--strict", action="store_true", default=True, help="fail if coverage incomplete")
    ap.add_argument("--no-strict", dest="strict", action="store_false")
    ap.add_argument("--out", required=True, type=Path, help="live_availability.csv")
    ap.add_argument("--append", action="store_true", help="append rows instead of overwriting", default=False)
    args = ap.parse_args()

    stats = _read_locust_csv(args.stats)
    failures = pd.read_csv(args.failures) if args.failures and args.failures.exists() else pd.DataFrame()

    name_col = _pick_col(stats, ["Name", "Request"])
    total_col = _pick_col(stats, ["Requests", "Request Count", "# Requests", "Count"])
    fail_col = _pick_col(stats, ["Failures", "Failure Count", "# Failures"])
    method_col = _pick_col(stats, ["Method"])
    if not name_col or not total_col or not fail_col:
        raise SystemExit("[availability-live] stats CSV missing Name/Requests/Failures columns.")

    entrypoints = _load_entrypoints(args.entrypoints)
    mapping = _load_targets(args.targets, entrypoints)

    fail_counts: Dict[str, float] = defaultdict(float)
    if not failures.empty:
        fname = _pick_col(failures, ["Name", "Request"])
        fcnt = _pick_col(failures, ["Occurrences", "Count"])
        if fname and fcnt:
            grp = failures.groupby(fname)[fcnt].sum()
            fail_counts = {str(k): float(v) for k, v in grp.items()}

    stats[name_col] = stats[name_col].astype(str)
    stats_filtered = stats[~stats[name_col].str.contains("Aggregated|Total", case=False, na=False)].copy()
    stats_filtered[method_col] = stats_filtered[method_col].astype(str).str.upper() if method_col else ""

    totals: Dict[str, Dict[str, float]] = {ep: {"total": 0.0, "bad": 0.0} for ep in entrypoints}
    unmatched_rows: list[str] = []

    for _, row in stats_filtered.iterrows():
        name = str(row[name_col]).strip()
        method = str(row[method_col]).upper() if method_col else ""
        ep = _match_entrypoint(name, method, mapping)
        if ep is None:
            unmatched_rows.append(name)
            continue
        total = _clean_num(row[total_col])
        bad = _clean_num(row[fail_col])
        if name in fail_counts:
            bad = fail_counts[name]
        bad = max(0.0, min(bad, total))
        slot = totals[ep]
        slot["total"] += total
        slot["bad"] += bad

    missing_eps = [ep for ep, agg in totals.items() if agg["total"] <= 0]
    if unmatched_rows:
        msg = "[availability-live] Locust Names missing mapping rules: " + ", ".join(sorted(set(unmatched_rows)))
        if args.strict:
            raise SystemExit(msg)
        print(msg)
    if missing_eps:
        msg = "[availability-live] entrypoints with zero matched requests: " + ", ".join(missing_eps)
        if args.strict:
            raise SystemExit(msg)
        print(msg)

    grid = _infer_p_grid(args.p_grid)
    replica = args.replica or ""

    rows = []
    for ep, agg in totals.items():
        total = agg["total"]
        bad = agg["bad"]
        if total <= 0:
            continue
        R = max(0.0, min(1.0, 1.0 - (bad / total)))
        for p in grid:
            rows.append({
                "replica": replica,
                "entrypoint": ep,
                "p_fail": float(p),
                "R_live": float(R),
            })

    if not rows:
        raise SystemExit("[availability-live] no entrypoints produced live availability rows; check stats/mapping.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append and args.out.exists() else "w"
    with args.out.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["replica", "entrypoint", "p_fail", "R_live"])
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)
    print(f"[availability-live] wrote {len(rows)} rows → {args.out}")


if __name__ == "__main__":
    main()
