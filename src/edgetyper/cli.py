"""
Command-line interface for the EdgeTyper pipeline.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import click
import pandas as pd
import re
import yaml 

from edgetyper.io.otlp_json import read_otlp_json
from edgetyper.graph.build import build as build_graph
from edgetyper.features.semconv import features_semconv
from edgetyper.features.timing import features_timing
from edgetyper.classify.rules import baseline_semconv, baseline_timing, rule_labels
from edgetyper.classify.model import label_with_fallback

@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """EdgeTyper CLI (OpenTelemetry Demo → traces → analysis)."""
    pass

# ---------------- extract ----------------
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


# ---------------- graph ----------------
@main.command("graph")
@click.option("--spans", "spans_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Input Parquet with normalized spans (from 'extract').")
@click.option("--out-events", "out_events", type=click.Path(dir_okay=False, path_type=Path), required=True,
              help="Output Parquet with per-interaction events.")
@click.option("--out-edges", "out_edges", type=click.Path(dir_okay=False, path_type=Path), required=True,
              help="Output Parquet with aggregated service→service edges.")
def graph_cmd(spans_path: Path, out_events: Path, out_edges: Path) -> None:
    """Construct RPC and messaging interactions and aggregate to service→service edges."""
    spans = pd.read_parquet(spans_path)
    results = build_graph(spans)
    out_events.parent.mkdir(parents=True, exist_ok=True)
    out_edges.parent.mkdir(parents=True, exist_ok=True)
    results.events.to_parquet(out_events, index=False)
    results.edges.to_parquet(out_edges, index=False)
    click.echo(
        f"[graph] events={len(results.events)} edges={len(results.edges)} "
        f"→ {out_events.name}, {out_edges.name}"
    )



# ---------------- featurize ----------------
@main.command("featurize")
@click.option("--events", "events_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--edges", "edges_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--out", "out_path", type=click.Path(dir_okay=False, path_type=Path), required=True)
def featurize_cmd(events_path: Path, edges_path: Path, out_path: Path) -> None:
    events = pd.read_parquet(events_path)
    edges = pd.read_parquet(edges_path)
    f_sem = features_semconv(events, edges)
    f_tim = features_timing(events)
    feats = f_sem.merge(f_tim, on=["src_service", "dst_service"], how="left").fillna(
        {"median_lag_ns": 0, "p_overlap": 0.0, "p_nonneg_lag": 0.0}
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out_path, index=False)
    click.echo(f"[featurize] wrote {len(feats)} edges → {out_path}")


# ---------------- baseline ----------------
@main.command("baseline")
@click.option("--features", "features_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--mode", type=click.Choice(["semconv", "timing"], case_sensitive=False), required=True)
@click.option("--out", "out_path", type=click.Path(dir_okay=False, path_type=Path), required=True)
def baseline_cmd(features_path: Path, mode: str, out_path: Path) -> None:
    feats = pd.read_parquet(features_path)
    if mode.lower() == "semconv":
        pred = baseline_semconv(feats)
    else:
        pred = baseline_timing(feats)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out_path, index=False)
    click.echo(f"[baseline:{mode}] wrote {len(pred)} labels → {out_path}")


# ---------------- label (rules + ML) ----------------
@main.command("label")
@click.option("--features", "features_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--out", "out_path", type=click.Path(dir_okay=False, path_type=Path), required=True)
def label_cmd(features_path: Path, out_path: Path) -> None:
    feats = pd.read_parquet(features_path)
    rules_df = rule_labels(feats, feats)  # timing columns are already merged in feats
    pred = label_with_fallback(rules_df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out_path, index=False)
    click.echo(f"[label] wrote {len(pred)} labels → {out_path}")


# ---------------- eval ----------------
def _normalize_service_name(name: str) -> str:
    # Case-insensitive, strip non-alnum, drop trailing "-service" or "service"
    s = re.sub(r"[^a-z0-9]", "", name.lower())
    s = re.sub(r"(?:service)$", "", s)  # drop suffix
    return s

def _load_ground_truth(gt_path: Path) -> pd.DataFrame:
    """
    Supports:
      - YAML: {links: [{source, target, type: ASYNC|BLOCKING}, ...]}
      - CSV:  src_pattern,dst_pattern,label  (backward compat)
    Returns a dataframe with columns: src_norm, dst_norm, gt_label
    """
    if gt_path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(gt_path.read_text())
        links = data.get("links", [])
        rows = []
        for link in links:
            src = str(link.get("source", "")).strip()
            dst = str(link.get("target", "")).strip()
            typ = str(link.get("type", "")).strip().upper()
            if not src or not dst or typ not in {"ASYNC", "BLOCKING"}:
                continue
            rows.append({
                "src_norm": _normalize_service_name(src),
                "dst_norm": _normalize_service_name(dst),
                "gt_label": "async" if typ == "ASYNC" else "sync",
            })
        return pd.DataFrame(rows)
    else:
        # Legacy CSV with regex patterns → keep behavior
        gt = pd.read_csv(gt_path)
        required = {"src_pattern", "dst_pattern", "label"}
        missing = required - set(gt.columns)
        if missing:
            raise click.ClickException(f"ground truth CSV missing columns: {sorted(missing)}")
        # Compile regex once
        gt["src_re"] = gt["src_pattern"].apply(lambda p: re.compile(p, re.IGNORECASE))
        gt["dst_re"] = gt["dst_pattern"].apply(lambda p: re.compile(p, re.IGNORECASE))
        return gt  # handled by matcher below

def _match_ground_truth(edges_df: pd.DataFrame, gt_df: pd.DataFrame, is_yaml: bool) -> pd.DataFrame:
    if is_yaml:
        # Normalize discovered edges and join on normalized names
        e = edges_df.copy()
        e["src_norm"] = e["src_service"].astype(str).map(_normalize_service_name)
        e["dst_norm"] = e["dst_service"].astype(str).map(_normalize_service_name)
        return e.merge(gt_df, on=["src_norm", "dst_norm"], how="inner")[["src_service","dst_service","gt_label"]]
    else:
        # Regex CSV mode
        rows = []
        for _, e in edges_df[["src_service","dst_service"]].drop_duplicates().iterrows():
            for _, g in gt_df.iterrows():
                if g["src_re"].search(str(e["src_service"])) and g["dst_re"].search(str(e["dst_service"])):
                    rows.append({"src_service": e["src_service"], "dst_service": e["dst_service"], "gt_label": g["label"]})
                    break
        return pd.DataFrame(rows)

@main.command("eval")
@click.option("--pred", "pred_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--features", "features_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--gt", "gt_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--out", "out_path", type=click.Path(dir_okay=False, path_type=Path), required=True)
def eval_cmd(pred_path: Path, features_path: Path, gt_path: Path, out_path: Path) -> None:
    pred = pd.read_csv(pred_path)
    feats = pd.read_parquet(features_path)
    gt_df = _load_ground_truth(gt_path)
    is_yaml = gt_path.suffix.lower() in {".yaml", ".yml"}

    gt_pairs = _match_ground_truth(
        feats[["src_service","dst_service"]].drop_duplicates(), gt_df, is_yaml=is_yaml
    )
    merged = gt_pairs.merge(pred, on=["src_service","dst_service"], how="left").dropna(subset=["pred_label"])
    if merged.empty:
        raise click.ClickException("No edges matched ground truth; check service names/tag and YAML.")

    # Metrics
    y_true = merged["gt_label"].astype(str)
    y_pred = merged["pred_label"].astype(str)

    from sklearn.metrics import classification_report, confusion_matrix

    labels = ["async", "sync"]
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    metrics: Dict[str, object] = {
        "labels": labels,
        "classification_report": report,
        "confusion_matrix": cm,
        "n_eval_edges": int(len(merged)),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    click.echo(f"[eval] evaluated {len(merged)} edges → {out_path}")


# ---------------- report (minimal static HTML) ----------------
@main.command("report")
@click.option("--metrics-dir", "metrics_dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--outdir", "outdir", type=click.Path(file_okay=False, path_type=Path), required=True)
def report_cmd(metrics_dir: Path, outdir: Path) -> None:
    """
    Aggregate any *_metrics_*.json files under metrics_dir and render a minimal HTML report.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    items = []
    for p in sorted(metrics_dir.glob("**/metrics_*.json")):
        try:
            items.append(json.loads(p.read_text()))
        except Exception:
            continue
    html = ["<html><head><meta charset='utf-8'><title>EdgeTyper Report</title></head><body>"]
    html.append("<h1>EdgeTyper — Results</h1>")
    if not items:
        html.append("<p>No metrics found.</p>")
    else:
        for m in items:
            html.append("<h2>Run</h2>")
            html.append(f"<p>Evaluated edges: {m.get('n_eval_edges', 0)}</p>")
            rep = m.get("classification_report", {})
            for cls in ("async", "sync", "macro avg"):
                if cls in rep:
                    pr = rep[cls]
                    html.append(
                        f"<h3>{cls}</h3><ul>"
                        f"<li>precision: {pr.get('precision', 0):.3f}</li>"
                        f"<li>recall: {pr.get('recall', 0):.3f}</li>"
                        f"<li>f1-score: {pr.get('f1-score', 0):.3f}</li>"
                        "</ul>"
                    )
    html.append("</body></html>")
    (outdir / "index.html").write_text("\n".join(html))
    click.echo(f"[report] wrote {outdir / 'index.html'}")
