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
from edgetyper.classify.rules import baseline_semconv, baseline_timing, rule_labels_from_features
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
@click.option(
    "--mask-semconv",
    is_flag=True, default=False, show_default=True,
    help="Simulate missing messaging semantics: set p_messaging=0, n_messaging=0, any_messaging_semconv=False.",
)
def featurize_cmd(events_path: Path, edges_path: Path, out_path: Path, mask_semconv: bool) -> None:
    events = pd.read_parquet(events_path)
    edges = pd.read_parquet(edges_path)
    f_sem = features_semconv(events, edges)
    f_tim = features_timing(events)
    feats = f_sem.merge(f_tim, on=["src_service", "dst_service"], how="left")

    # Guarantee timing columns exist
    if "median_lag_ns" not in feats.columns: feats["median_lag_ns"] = 0
    if "p_overlap" not in feats.columns:     feats["p_overlap"] = 0.0
    if "p_nonneg_lag" not in feats.columns:  feats["p_nonneg_lag"] = (feats["median_lag_ns"] >= 0).astype(float)

    # ---- robustness knob: hide SemConv features ----
    if mask_semconv:
        feats["n_messaging"] = 0
        feats["p_messaging"] = 0.0
        feats["any_messaging_semconv"] = False

    feats = feats.fillna({"median_lag_ns": 0, "p_overlap": 0.0, "p_nonneg_lag": 0.0})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out_path, index=False)
    click.echo(f"[featurize{' (masked) ' if mask_semconv else ' '}] wrote {len(feats)} edges → {out_path}")


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
    rules_df = rule_labels_from_features(feats)
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
@click.option("--name", "run_name", type=str, default=None,
              help="Human-readable label for this run (e.g., 'Baseline — SemConv').")
def eval_cmd(pred_path: Path, features_path: Path, gt_path: Path, out_path: Path, run_name: str | None) -> None:
    pred = pd.read_csv(pred_path)
    feats = pd.read_parquet(features_path)
    gt_df, is_yaml = _load_ground_truth(gt_path)
    gt_pairs = _match_ground_truth(feats, gt_df, is_yaml=is_yaml)

    merged = gt_pairs.merge(pred, on=["src_service", "dst_service"], how="left").dropna(subset=["pred_label"])
    if merged.empty:
        raise click.ClickException("No edges matched ground truth; check service names/tag and ground_truth file.")

    # Metrics
    from sklearn.metrics import classification_report, confusion_matrix
    labels = ["async", "sync"]
    y_true = merged["gt_label"].astype(str)
    y_pred = merged["pred_label"].astype(str)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    # Class counts
    n_async = int((y_true == "async").sum())
    n_sync = int((y_true == "sync").sum())

    # Name fallback from filename if --name not given
    if not run_name:
        stem = Path(pred_path).stem  # e.g., pred_ours
        run_name = stem.replace("pred_", "").replace("_", " ").title()

    metrics: Dict[str, object] = {
        "run_name": run_name,
        "labels": labels,
        "n_eval_edges": int(len(merged)),
        "n_async": n_async,
        "n_sync": n_sync,
        "classification_report": report,
        "confusion_matrix": cm,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    click.echo(f"[eval] ({run_name}) evaluated {len(merged)} edges → {out_path}")


# ---------------- report ----------------
@main.command("report")
@click.option("--metrics-dir", "metrics_dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--outdir", "outdir", type=click.Path(file_okay=False, path_type=Path), required=True)
def report_cmd(metrics_dir: Path, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    items = []
    for p in sorted(metrics_dir.glob("**/metrics_*.json")):
        try:
            m = json.loads(p.read_text())
            # If run_name missing (old files), infer from filename
            rn = m.get("run_name") or p.stem.replace("metrics_", "").replace("_", " ").title()
            m["run_name"] = rn
            items.append(m)
        except Exception:
            continue

    # Sort in a friendly order if we recognize names
    order = {
      "Baseline — Timing": 0, "Baseline — SemConv": 1, "EdgeTyper (ours)": 2,
      "Baseline — Timing (dropped)": 3, "Baseline — SemConv (dropped)": 4, "EdgeTyper (ours) — SemConv dropped": 5,
    }
    items.sort(key=lambda x: order.get(x["run_name"], 99))


    def row(m):
        rep = m.get("classification_report", {})
        def s(cls, k):
            return f'{rep.get(cls, {}).get(k, 0):.3f}'
        # Confusion matrix unpack (labels are ["async", "sync"])
        cm = m.get("confusion_matrix", [[0,0],[0,0]])
        tn = cm[1][1]  # sync→sync
        tp = cm[0][0]  # async→async
        fp = cm[1][0]  # sync→async
        fn = cm[0][1]  # async→sync
        return f"""
        <tr>
          <td><b>{m["run_name"]}</b></td>
          <td>{m.get("n_eval_edges", 0)}</td>
          <td>{m.get("n_async", 0)}</td>
          <td>{m.get("n_sync", 0)}</td>
          <td>{s("async","precision")}/{s("async","recall")}/{s("async","f1-score")}</td>
          <td>{s("sync","precision")}/{s("sync","recall")}/{s("sync","f1-score")}</td>
          <td>{s("macro avg","precision")}/{s("macro avg","recall")}/{s("macro avg","f1-score")}</td>
          <td>TP={tp}, FP={fp}, FN={fn}, TN={tn}</td>
        </tr>
        """

    html = [
        "<html><head><meta charset='utf-8'><title>EdgeTyper — Results</title>",
        "<style>body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:24px} table{border-collapse:collapse;width:100%} th,td{border:1px solid #ddd;padding:8px;text-align:center} th{background:#f5f5f5}</style>",
        "</head><body>",
        "<h1>EdgeTyper — Typed Service Edges</h1>",
        "<p>Metrics are edge-level. Per-class cells show <em>precision/recall/F1</em>. Confusion matrix is for labels [async, sync].</p>",
    ]
    if not items:
        html.append("<p><b>No metrics found.</b></p>")
    else:
        html.append("""
        <table>
          <thead>
            <tr>
              <th>Run</th><th>Eval edges</th><th>#Async (GT)</th><th>#Sync (GT)</th>
              <th>Async P/R/F1</th><th>Sync P/R/F1</th><th>Macro P/R/F1</th><th>Confusion (TP/FP/FN/TN)</th>
            </tr>
          </thead><tbody>
        """)
        for m in items:
            html.append(row(m))
        html.append("</tbody></table>")

        # Short interpretation block
        html.append("<h2>Interpretation</h2>")
        html.append("<ul>")
        html.append("<li><b>Baseline — SemConv</b> uses OpenTelemetry messaging semantics only (PRODUCER/CONSUMER & messaging.*). When the demo is fully instrumented, it should be near-perfect.</li>")
        html.append("<li><b>Baseline — Timing</b> uses only timing/overlap. It can over-label edges as async if median lag ≥ 0 and overlap is low.</li>")
        html.append("<li><b>EdgeTyper (ours)</b> combines SemConv with timing and span links; when semantics are present it matches SemConv, and when they are missing it should degrade gracefully.</li>")
        html.append("</ul>")

    html.append("</body></html>")
    (outdir / "index.html").write_text("\n".join(html))
    click.echo(f"[report] wrote {outdir / 'index.html'}")


# ---------------- debug ----------------
@main.command("debug")
@click.option("--features", "features_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--gt", "gt_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--pred", "pred_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=False)
@click.option("--out", "out_csv", type=click.Path(dir_okay=False, path_type=Path), required=True)
def debug_cmd(features_path: Path, gt_path: Path, pred_path: Path | None, out_csv: Path) -> None:
    feats = pd.read_parquet(features_path)
    gt_df, is_yaml = _load_ground_truth(gt_path)
    matched = _match_ground_truth(feats, gt_df, is_yaml=is_yaml)
    # Join features for matched edges
    cols = ["src_service","dst_service","p_messaging","link_ratio","median_lag_ns","p_overlap","p_nonneg_lag"]
    df = matched.merge(feats[cols], on=["src_service","dst_service"], how="left")
    # Add rule-only labels for visibility
    from edgetyper.classify.rules import rule_labels_from_features
    rules = rule_labels_from_features(feats)[["src_service","dst_service","rule_label","rule_conf"]]
    df = df.merge(rules, on=["src_service","dst_service"], how="left")
    # Optionally join predictions
    if pred_path:
        pred = pd.read_csv(pred_path)
        df = df.merge(pred, on=["src_service","dst_service"], how="left")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    click.echo(f"[debug] wrote {len(df)} matched edges with features → {out_csv}")


# --------------------------------
if __name__ == "__main__":
    main()
