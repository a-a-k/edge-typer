"""
Command-line interface for the EdgeTyper pipeline.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import click
import pandas as pd

from edgetyper.io.otlp_json import read_otlp_json


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """EdgeTyper CLI (OpenTelemetry Demo → traces → analysis)."""
    pass


@main.command("extract")
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to OTLP-JSON traces file produced by the Collector file exporter.",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Output Parquet file for the normalized spans table.",
)
@click.option(
    "--service-attr",
    default="service.name",
    show_default=True,
    help="Resource attribute key that holds the service name.",
)
@click.option(
    "--min-spans",
    default=100,
    show_default=True,
    type=int,
    help="Fail fast if fewer than this many spans are parsed (helps catch empty captures).",
)
def extract_cmd(input_path: Path, out_path: Path, service_attr: str, min_spans: int) -> None:
    """
    Parse an OTLP-JSON traces file and write a normalized spans table to Parquet.
    """
    try:
        df = read_otlp_json(input_path, service_attr_key=service_attr)
    except Exception as exc:
        click.echo(f"[extract] ERROR: failed to parse {input_path}: {exc}", err=True)
        sys.exit(2)

    if df.empty or len(df) < min_spans:
        click.echo(
            f"[extract] ERROR: parsed {len(df)} spans (<{min_spans}). "
            "The capture may be incomplete; increase soak time or check Collector config.",
            err=True,
        )
        sys.exit(3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    click.echo(f"[extract] wrote {len(df)} spans → {out_path} ({size_mb:.2f} MiB)")


if __name__ == "__main__":
    main()
