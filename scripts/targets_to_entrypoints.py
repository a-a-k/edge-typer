#!/usr/bin/env python3
"""Derive entrypoint lists from src/edgetyper/targets.yaml."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import yaml


def _load_entrypoints(targets: Path) -> list[str]:
    data = yaml.safe_load(targets.read_text(encoding="utf-8")) or {}
    eps: list[str] = []
    for ep, rules in (data.get("entrypoints") or {}).items():
        if not ep:
            continue
        eps.append(str(ep))
    return sorted(set(eps))


def _write_csv(path: Path, entrypoints: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "entrypoint\n" + "\n".join(str(ep) for ep in entrypoints) + ("\n" if entrypoints else ""),
        encoding="utf-8",
    )


def _write_txt(path: Path, entrypoints: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(ep) for ep in entrypoints) + ("\n" if entrypoints else ""), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Render entrypoint list(s) from targets.yaml")
    ap.add_argument("--targets", type=Path, required=True, help="YAML file (src/edgetyper/targets.yaml)")
    ap.add_argument("--out", type=Path, required=True, help="Path to write entrypoints.csv")
    ap.add_argument("--txt", type=Path, help="Optional path to also write newline-delimited entrypoints")
    ap.add_argument(
        "--fail-empty",
        action="store_true",
        help="Exit with error if no entrypoints are discovered",
    )
    args = ap.parse_args()

    if not args.targets.exists():
        raise SystemExit(f"targets file not found: {args.targets}")
    entrypoints = _load_entrypoints(args.targets)
    if args.fail_empty and not entrypoints:
        raise SystemExit("No entrypoints discovered in targets.yaml")

    _write_csv(args.out, entrypoints)
    if args.txt:
        _write_txt(args.txt, entrypoints)

    print(f"[targets_to_entrypoints] wrote {len(entrypoints)} entrypoints -> {args.out}")
    if args.txt:
        print(f"[targets_to_entrypoints] wrote newline list -> {args.txt}")


if __name__ == "__main__":
    main()
