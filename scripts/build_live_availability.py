#!/usr/bin/env python3
"""
Derive entrypoints + live availability from Locust CSV exports.

Outputs:
  * live_availability.csv  (replica, entrypoint, p_fail, R_live)
  * entrypoints.csv / entrypoints.txt
  * live_targets.yaml      (regex mapping for availability-live CLI)
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    norm = {c.lower().replace(" ", "").replace("_", "").replace("#", ""): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "").replace("_", "").replace("#", "")
        if key in norm:
            return norm[key]
    return None


def _slugify(label: str, taken: set[str]) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    if not slug:
        slug = "entrypoint"
    base = slug
    i = 2
    while slug in taken:
        slug = f"{base}-{i}"
        i += 1
    taken.add(slug)
    return slug


def _infer_p_grid(typed_path: Optional[Path], p_grid_arg: Optional[str]) -> List[float]:
    if typed_path and typed_path.exists():
        try:
            df = pd.read_csv(typed_path)
            if "p_fail" in df.columns and not df["p_fail"].dropna().empty:
                vals = sorted(float(v) for v in df["p_fail"].unique())
                if vals:
                    return vals
        except Exception:
            pass
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
            return sorted(set(vals))
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
    return float(num)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True, type=Path)
    ap.add_argument("--failures", required=False, type=Path)
    ap.add_argument("--typed", required=False, type=Path, help="Use p_fail grid from availability_typed.csv")
    ap.add_argument("--p-grid", required=False, type=str, help="Fallback grid (comma separated)")
    ap.add_argument("--replica", required=False, type=str, help="replicate-* label")
    ap.add_argument("--min-requests", type=float, default=25.0, help="minimum Locust requests per Name to keep")
    ap.add_argument("--out", required=True, type=Path, help="live_availability.csv")
    ap.add_argument("--out-entrypoints", required=True, type=Path, help="entrypoints.csv")
    ap.add_argument("--out-entrypoints-txt", required=True, type=Path, help="entrypoints.txt")
    ap.add_argument("--out-targets", required=True, type=Path, help="live_targets.yaml")
    args = ap.parse_args()

    def _fail(msg: str, code: int = 1):
        raise SystemExit(f"[build-live] {msg} (code={code})")

    if not args.stats.exists():
        _fail(f"stats CSV missing: {args.stats}", code=2)
    stats = pd.read_csv(args.stats)
    if stats.empty:
        _fail(f"stats CSV is empty: {args.stats}", code=3)

    name_col = _pick_col(stats, ["Name", "Request"])
    req_col = _pick_col(stats, ["Requests", "Request Count", "# Requests", "Count"])
    fail_col = _pick_col(stats, ["Failures", "Failure Count"])
    method_col = _pick_col(stats, ["Method"])
    if not name_col or not req_col or not fail_col:
        _fail("Could not locate Name/Requests/Failures columns in stats CSV", code=4)

    df = stats.copy()
    df[name_col] = df[name_col].astype(str)
    df = df[~df[name_col].str.contains("Aggregated|Total", case=False, na=False)]
    if df.empty:
        _fail("No per-endpoint rows found in stats CSV after filtering totals", code=5)

    # Basic failure buckets (if failures.csv present we use Occurrences sums)
    fail_counts: Dict[str, float] = defaultdict(float)
    if args.failures and args.failures.exists():
        fails_df = pd.read_csv(args.failures)
        f_name = _pick_col(fails_df, ["Name", "Request"])
        f_cnt = _pick_col(fails_df, ["Occurrences", "Count"])
        if f_name and f_cnt:
            fails_df[f_name] = fails_df[f_name].astype(str)
            grp = fails_df.groupby(f_name)[f_cnt].sum()
            for name, val in grp.items():
                fail_counts[str(name)] += float(val)

    keep_rows = []
    taken_ids: set[str] = set()
    name_to_ep: Dict[str, str] = {}
    for _, row in df.iterrows():
        total = _clean_num(row[req_col])
        if total < args.min_requests:
            continue
        raw_name = str(row[name_col]).strip()
        if not raw_name:
            continue
        ep = name_to_ep.get(raw_name)
        if not ep:
            ep = _slugify(raw_name, taken_ids)
            name_to_ep[raw_name] = ep
        fail_val = _clean_num(row[fail_col])
        if raw_name in fail_counts:
            fail_val = fail_counts[raw_name]
        fail_val = max(0.0, min(fail_val, total))
        keep_rows.append((ep, raw_name, float(total), float(fail_val)))

    if not keep_rows:
        _fail(f"No Locust endpoints exceeded min-requests ({args.min_requests}).", code=6)

    entrypoints = [ep for ep, *_ in keep_rows]
    entrypoints_csv = pd.DataFrame({"entrypoint": entrypoints})
    args.out_entrypoints.parent.mkdir(parents=True, exist_ok=True)
    entrypoints_csv.to_csv(args.out_entrypoints, index=False)
    args.out_entrypoints_txt.parent.mkdir(parents=True, exist_ok=True)
    args.out_entrypoints_txt.write_text("\n".join(entrypoints) + "\n", encoding="utf-8")

    targets_payload = {
        "entrypoints": {
            ep: [{"name_regex": f"^{re.escape(raw)}$"}]
            for ep, raw, *_ in keep_rows
        }
    }
    args.out_targets.parent.mkdir(parents=True, exist_ok=True)
    args.out_targets.write_text(yaml.safe_dump(targets_payload, sort_keys=False), encoding="utf-8")

    grid = _infer_p_grid(args.typed, args.p_grid)
    replica = args.replica or ""

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["replica", "entrypoint", "p_fail", "R_live"])
        writer.writeheader()
        for ep, _raw, total, fail_val in keep_rows:
            R = 0.0 if total <= 0 else max(0.0, min(1.0, 1.0 - (fail_val / total)))
            for p in grid:
                writer.writerow({
                    "replica": replica,
                    "entrypoint": ep,
                    "p_fail": float(p),
                    "R_live": float(R),
                })
    print(f"[build-live] wrote {len(keep_rows)} entrypoints; live grid rows={len(keep_rows)*len(grid)}")


if __name__ == "__main__":
    main()
