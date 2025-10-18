"""
EdgeTyper CLI
Stages:
  - extract:   read OTLP-JSON traces → spans.parquet
  - graph:     spans.parquet → events.parquet + edges.parquet
  - featurize: events+edges → features.parquet
  - baseline:  features → predictions (semconv|timing)
  - label:     features → predictions (rules + ML fallback)
  - eval:      predictions + ground_truth.(yaml|csv) → metrics.json
  - report:    aggregate metrics (directory) → site/ (static HTML)
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict

import click
import pandas as pd
import yaml  # PyYAML

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
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--out", "out_path", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--service-attr", default="service.name", show_default=True)
@click.option("--min-spans", default=100, show_default=True, type=int)
def extract_cmd(input_path: Path, out_path: Path, service_attr: str, min_spans: int) -> None:
    df = read_otlp_json(input_path, service_attr_key=service_attr)
    if df.empty or len(df) < min_spans:
        raise click.ClickException(f"[extract] Parsed {len(df)} spans (<{min_spans}). Increase soak or fix config.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    click.echo(f"[extract] wrote {len(df)} spans → {out_path}")


# ---------------- graph ----------------
@main.command("graph")
@click.option("--spans", "spans_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--out-events", "out_events", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--out-edges", "out_edges", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--with-broker-edges/--no-broker-edges", default=True, show_default=True,
              help="Emit producer→broker and broker→consumer edges in addition to producer→consumer.")
@click.option("--broker-service", default="kafka", show_default=True, help="Name to use for the broker node.")
def graph_cmd(spans_path: Path, out_events: Path, out_edges: Path, with_broker_edges: bool, broker_service: str) -> None:
    spans = pd.read_parquet(spans_path)
    results = build_graph(spans, emit_broker_edges=with_broker_edges, broker_service_name=broker_service)
    out_events.parent.mkdir(parents=True, exist_ok=True)
    out_edges.parent.mkdir(parents=True, exist_ok=True)
    results.events.to_parquet(out_events, index=False)
    results.edges.to_parquet(out_edges, index=False)
    click.echo(f"[graph] events={len(results.events)} edges={len(results.edges)}")


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

    feats = f_sem.merge(f_tim, on=["src_service", "dst_service"], how="left")

    # ---- Guarantee timing features exist (esp. p_nonneg_lag) ----
    if "median_lag_ns" not in feats.columns:
        feats["median_lag_ns"] = 0
    if "p_overlap" not in feats.columns:
        feats["p_overlap"] = 0.0
    if "p_nonneg_lag" not in feats.columns:
        # Conservative fallback: treat non-negative lag as present when median lag >= 0
        feats["p_nonneg_lag"] = (feats["median_lag_ns"] >= 0).astype(float)

    # Fill any remaining NaNs from join
    feats = feats.fillna({"median_lag_ns": 0, "p_overlap": 0.0, "p_nonneg_lag": 0.0})

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
    pred = baseline_semconv(feats) if mode.lower() == "semconv" else baseline_timing(feats)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out_path, index=False)
    click.echo(f"[baseline:{mode}] wrote {len(pred)} labels → {out_path}")


# ---------------- label (rules + ML) ----------------
@main.command("label")
@click.option("--features", "features_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--out", "out_path", type=click.Path(dir_okay=False, path_type=Path), required=True)
def label_cmd(features_path: Path, out_path: Path) -> None:
    feats = pd.read_parquet(features_path)
    rules_df = rule_labels(feats, feats)  # timing already merged in feats
    pred = label_with_fallback(rules_df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out_path, index=False)
    click.echo(f"[label] wrote {len(pred)} labels → {out_path}")


# ---------------- eval ----------------
def _normalize_service_name(name: str) -> str:
    # Case-insensitive, strip non-alnum, drop trailing 'service'
    s = re.sub(r"[^a-z0-9]", "", str(name).lower())
    s = re.sub(r"(?:service)$", "", s)
    return s


def _load_ground_truth(gt_path: Path) -> tuple[pd.DataFrame, bool]:
    """
    Returns (df, is_yaml). YAML schema:
      links:
        - source: <service name>
          target: <service name>
          type:   ASYNC|BLOCKING
    CSV schema (legacy):
      src_pattern,dst_pattern,label
    """
    if gt_path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(gt_path.read_text())
        links = data.get("links", []) if isinstance(data, dict) else []
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
        return pd.DataFrame(rows), True
    else:
        gt = pd.read_csv(gt_path)
        required = {"src_pattern", "dst_pattern", "label"}
        missing = required - set(gt.columns)
        if missing:
            raise click.ClickException(f"ground truth CSV missing columns: {sorted(missing)}")
        gt["src_re"] = gt["src_pattern"].apply(lambda p: re.compile(p, re.IGNORECASE))
        gt["dst_re"] = gt["dst_pattern"].apply(lambda p: re.compile(p, re.IGNORECASE))
        return gt, False


def _match_ground_truth(edges_df: pd.DataFrame, gt_df: pd.DataFrame, is_yaml: bool) -> pd.DataFrame:
    if is_yaml:
        e = edges_df[["src_service", "dst_service"]].drop_duplicates().copy()
        e["src_norm"] = e["src_service"].map(_normalize_service_name)
        e["dst_norm"] = e["dst_service"].map(_normalize_service_name)
        m = e.merge(gt_df, on=["src_norm", "dst_norm"], how="inner")
        return m[["src_service", "dst_service", "gt_label"]]
    else:
        rows = []
        for _, e in edges_df[["src_service", "dst_service"]].drop_duplicates().iterrows():
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
    gt_df, is_yaml = _load_ground_truth(gt_path)
    gt_pairs = _match_ground_truth(feats, gt_df, is_yaml=is_yaml)

    merged = gt_pairs.merge(pred, on=["src_service", "dst_service"], how="left").dropna(subset=["pred_label"])
    if merged.empty:
        raise click.ClickException("No edges matched ground truth; check service names/tag and ground_truth file.")

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


# ---------------- report ----------------
@main.command("report")
@click.option("--metrics-dir", "metrics_dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--outdir", "outdir", type=click.Path(file_okay=False, path_type=Path), required=True)
def report_cmd(metrics_dir: Path, outdir: Path) -> None:
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


if __name__ == "__main__":
    main()
