#!/usr/bin/env python3
"""
Derive canonical entrypoint services directly from spans.parquet.

Definition: an entrypoint is any service that owns at least one SERVER span
whose parent span is either absent (root span) or belongs to an unknown service.
This captures the concrete set of services that receive traffic from outside
the trace graph, eliminating guesswork during resilience estimation.

Outputs:
  --out-csv  entrypoints.csv  (column: entrypoint)
  --out-txt  entrypoints.txt  (newline-delimited)
  --out-targets (optional) stub live_targets.yaml to help map Locust Names → entrypoints
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd
import yaml


def _write_list_csv(path: Path, values: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"entrypoint": list(values)}).to_csv(path, index=False)


def _write_list_txt(path: Path, values: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def _write_stub_targets(path: Path, entrypoints: Sequence[str]) -> None:
    """
    Emit a YAML skeleton that the user can edit to map Locust Names (regex) to entrypoints.
    By default each entrypoint gets an empty rule list as a reminder.
    """
    payload = {
        "entrypoints": {
            ep: [
                {"name_regex": "^REPLACE_WITH_LOCUST_NAME_REGEX$"}
            ]
            for ep in entrypoints
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Fill in Locust Name regexes for each entrypoint before running availability-live.\n"
        + yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract entrypoint services from spans.parquet")
    ap.add_argument("--spans", type=Path, required=True, help="Path to spans.parquet")
    ap.add_argument("--edges", type=Path, help="Optional edges.parquet to ensure entrypoints exist in the graph")
    ap.add_argument("--out-csv", type=Path, required=True, help="entrypoints.csv destination")
    ap.add_argument("--out-txt", type=Path, required=True, help="entrypoints.txt destination")
    ap.add_argument("--out-targets", type=Path, help="Optional stub live_targets.yaml destination")
    args = ap.parse_args()

    if not args.spans.exists():
        raise SystemExit(f"[entrypoints-from-traces] spans file not found: {args.spans}")

    spans = pd.read_parquet(args.spans)
    needed = {"span_id", "parent_span_id", "service_name", "span_kind"}
    missing = sorted(needed - set(spans.columns))
    if missing:
        raise SystemExit(f"[entrypoints-from-traces] spans parquet missing columns: {missing}")

    # Normalize
    spans["span_kind"] = spans["span_kind"].astype(str).str.upper()
    spans["service_name"] = spans["service_name"].astype(str)

    servers = spans.loc[spans["span_kind"] == "SERVER", ["span_id", "parent_span_id", "service_name"]]
    if servers.empty:
        raise SystemExit("[entrypoints-from-traces] no SERVER spans observed; cannot derive entrypoints.")

    span_to_service = spans.set_index("span_id")["service_name"]
    parent_service = servers["parent_span_id"].map(span_to_service)
    external_mask = servers["parent_span_id"].isna() | parent_service.isna()
    entry_services = sorted(set(servers.loc[external_mask, "service_name"]))
    graph_services: set[str] | None = None
    if args.edges and args.edges.exists():
        edges = pd.read_parquet(args.edges)
        cols = [c for c in ("src_service", "dst_service") if c in edges.columns]
        if cols:
            graph_services = set(pd.unique(edges[cols].values.ravel("K")))
            entry_services = [svc for svc in entry_services if svc in graph_services]
    if not entry_services:
        raise SystemExit("[entrypoints-from-traces] no root SERVER spans detected; entrypoints set would be empty.")

    _write_list_csv(args.out_csv, entry_services)
    _write_list_txt(args.out_txt, entry_services)
    if args.out_targets:
        _write_stub_targets(args.out_targets, entry_services)

    print(f"[entrypoints-from-traces] derived {len(entry_services)} entrypoints from {args.spans}")


if __name__ == "__main__":
    main()
