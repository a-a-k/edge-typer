#!/usr/bin/env python3
"""
Build a live availability table (entrypoint × p_fail) from Locust CSV exports.

This script reuses the same implementation as `edgetyper availability-live`
to ensure the Monte-Carlo model and live measurements stay on the exact
entrypoint grid. It computes one live success rate per entrypoint from the
Locust stats/ failures CSVs, then expands it across the requested p_fail grid.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional

import pandas as pd

from edgetyper.live import compute_live_availability


def _infer_p_grid(typed_path: Optional[Path], p_grid_arg: Optional[str]) -> List[float]:
    if typed_path and typed_path.exists():
        try:
            df = pd.read_csv(typed_path)
            if "p_fail" in df.columns and not df["p_fail"].dropna().empty:
                vals = sorted(float(x) for x in sorted(df["p_fail"].unique()))
                if vals:
                    return vals
        except Exception:
            pass
    if p_grid_arg:
        vals: list[float] = []
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats",    required=True, type=Path, help="locust_stats.csv")
    ap.add_argument("--failures", required=False, type=Path, help="locust_failures.csv (optional)")
    ap.add_argument("--typed",    required=False, type=Path, help="availability_typed.csv to infer p_fail grid")
    ap.add_argument("--replica",  required=False, type=str,   help="replicate-* name to stamp into 'replica' column")
    ap.add_argument("--p-grid",   required=False, type=str,   help="explicit grid, e.g. '0.1,0.3,0.5,0.7,0.9'")
    ap.add_argument("--targets",  required=False, type=Path,  help="targets.yaml with entrypoint mapping (regex)")
    ap.add_argument("--entrypoints", required=False, type=Path,
                   help="entrypoints CSV/TXT (must match data passed to edgetyper resilience)")
    ap.add_argument("--out",      required=True,  type=Path,  help="output live_availability.csv")
    ap.add_argument("--strict", dest="strict", action="store_true", default=True,
                    help="fail if inputs are missing/empty")
    ap.add_argument("--no-strict", dest="strict", action="store_false",
                    help="silently skip on missing/empty inputs")
    args = ap.parse_args()

    def _bail(msg: str, code: int = 1):
        import sys
        print(f"[availability-live] ERROR: {msg}", file=sys.stderr)
        if args.strict:
            raise SystemExit(code)
        return

    if not args.stats.exists():
        return _bail(f"stats CSV is missing: {args.stats}", code=2)

    failures = args.failures if args.failures and args.failures.exists() else None
    df_rows, missing_eps, unmatched = compute_live_availability(
        args.stats,
        failures,
        args.targets,
        args.entrypoints,
    )

    if missing_eps:
        preview = ", ".join(missing_eps[:5])
        msg = (
            f"{len(missing_eps)} entrypoints listed in --entrypoints have zero Locust requests "
            f"(examples: {preview})."
        )
        if args.strict:
            return _bail(msg, code=5)
        print(f"[availability-live] WARNING: {msg}")

    if df_rows.empty:
        sample = ", ".join(name for name, _cnt in unmatched.most_common(5))
        hint = f" Example unmatched names: {sample}" if sample else ""
        return _bail(
            "No requests matched entrypoint mapping — ensure live_targets.yaml aligns with Locust Names."
            + hint,
            code=6,
        )

    if unmatched:
        total = sum(unmatched.values())
        preview = ", ".join(f"{name} ({cnt})" for name, cnt in unmatched.most_common(3))
        print(
            f"[availability-live] WARNING: ignored {total} requests with no entrypoint mapping "
            f"(e.g., {preview})"
        )

    grid = _infer_p_grid(args.typed, args.p_grid)
    replica = args.replica or ""

    out_rows = []
    for row in df_rows.itertuples(index=False):
        for p in grid:
            out_rows.append({
                "replica": replica,
                "entrypoint": row.entrypoint,
                "p_fail": float(p),
                "R_live": float(row.R_live),
            })

    if not out_rows:
        return _bail("no entrypoints produced live availability rows", code=4)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["replica", "entrypoint", "p_fail", "R_live"])
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"[availability-live] wrote {len(out_rows)} rows → {args.out}")


if __name__ == "__main__":
    main()
