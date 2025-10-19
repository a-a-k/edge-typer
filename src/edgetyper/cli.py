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

@click.option("--count-mode",
              type=click.Choice(["pred", "gt", "both"], case_sensitive=False),
              default="pred", show_default=True,
              help="Which counts to show in the metrics table: predicted by each run, ground truth, or both.")

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
@click.option(
    "--mask-timing",
    is_flag=True, default=False, show_default=True,
    help="Simulate missing timing: set median_lag_ns=0, p_overlap=0.5, p_nonneg_lag=0.5.",
)
def featurize_cmd(events_path: Path, edges_path: Path, out_path: Path, mask_semconv: bool, mask_timing: bool) -> None:
    events = pd.read_parquet(events_path)
    edges = pd.read_parquet(edges_path)
    f_sem = features_semconv(events, edges)
    f_tim = features_timing(events)
    feats = f_sem.merge(f_tim, on=["src_service", "dst_service"], how="left")

    # Ensure timing columns exist
    if "median_lag_ns" not in feats.columns: feats["median_lag_ns"] = 0
    if "p_overlap" not in feats.columns:     feats["p_overlap"] = 0.0
    if "p_nonneg_lag" not in feats.columns:  feats["p_nonneg_lag"] = (feats["median_lag_ns"] >= 0).astype(float)

    # Drop SemConv (robustness)
    if mask_semconv:
        feats["n_messaging"] = 0
        feats["p_messaging"] = 0.0
        feats["any_messaging_semconv"] = False

    # Drop Timing (robustness)
    if mask_timing:
        feats["median_lag_ns"] = 0
        feats["p_overlap"]     = 0.5
        feats["p_nonneg_lag"]  = 0.5

    feats = feats.fillna({"median_lag_ns": 0, "p_overlap": 0.0, "p_nonneg_lag": 0.0})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out_path, index=False)
    tag = " (masked semconv)" if mask_semconv else (" (masked timing)" if mask_timing else "")
    click.echo(f"[featurize{tag}] wrote {len(feats)} edges → {out_path}")


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
@click.option("--spans", "spans_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=False)
@click.option("--events", "events_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=False)
@click.option("--edges", "edges_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=False)
@click.option("--features", "features_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=False,
              help="Optional features.parquet to compute coverage and snapshot.")
@click.option("--gt", "gt_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=False,
              help="Optional ground_truth.(yaml|csv) to compute coverage.")
@click.option("--coverage-top", type=int, default=30, show_default=True,
              help="How many unmatched edges to list in the coverage table.")
@click.option("--provenance", "prov_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=False,
              help="Optional provenance.json describing the capture run (commit/ref/soak, etc.).")
@click.option("--assets-dir", "assets_dir", type=click.Path(file_okay=False, path_type=Path), required=False,
              help="Optional directory under outdir to place downloadable CSVs (default: outdir/data).")
@click.option("--count-mode",
              type=click.Choice(["pred", "gt", "both"], case_sensitive=False),
              default="pred", show_default=True,
              help="Which counts to show in the metrics table: predicted by each run, ground truth, or both.")
def report_cmd(metrics_dir: Path, outdir: Path, spans_path: Path | None, events_path: Path | None,
               edges_path: Path | None, features_path: Path | None, gt_path: Path | None, coverage_top: int,
               prov_path: Path | None, assets_dir: Path | None, count_mode: str) -> None:
    import re, shutil

    outdir.mkdir(parents=True, exist_ok=True)
    assets = assets_dir or (outdir / "data")
    assets.mkdir(parents=True, exist_ok=True)

    # ---------- load metrics ----------
    items = []
    for p in sorted(metrics_dir.glob("**/metrics_*.json")):
        try:
            m = json.loads(p.read_text())
            rn = m.get("run_name") or p.stem.replace("metrics_", "").replace("_", " ").title()
            m["run_name"] = rn
            items.append(m)
        except Exception:
            continue

    # Canonicalize names and group by scenario → method
    def canon(name: str) -> str:
        name = re.sub(r"\s+", " ", name).strip()
        name = name.replace("(dropped)", "(SemConv dropped)")  # normalize older names
        return name

    def scenario_of(name: str) -> str:
        if "SemConv dropped" in name: return "semconv"
        if "Timing dropped"  in name: return "timing"
        return "full"

    def method_of(name: str) -> str:
        if "EdgeTyper (ours)" in name: return "ours"
        if "SemConv" in name:          return "semconv"
        return "timing"

    dedup = {}
    for m in items:
        m["run_name"] = canon(m["run_name"])
        key = (scenario_of(m["run_name"]), method_of(m["run_name"]))
        dedup[key] = m  # keep last if duplicates
    items = list(dedup.values())

    scenario_order = {"full": 0, "semconv": 1, "timing": 2}
    method_order   = {"timing": 0, "semconv": 1, "ours": 2}
    items.sort(key=lambda m: (scenario_order[scenario_of(m["run_name"])],
                              method_order[method_of(m["run_name"])]))

    # ---------- snapshot (counts & tops) ----------
    spans_df  = pd.read_parquet(spans_path)  if spans_path  else None
    events_df = pd.read_parquet(events_path) if events_path else None
    edges_df  = pd.read_parquet(edges_path)  if edges_path  else None
    feats_df  = pd.read_parquet(features_path) if features_path else None

    n_spans = len(spans_df) if spans_df is not None else None
    n_services = int(spans_df["service_name"].nunique()) if spans_df is not None and "service_name" in spans_df.columns else None
    n_events = len(events_df) if events_df is not None else None
    n_edges  = len(edges_df)  if edges_df  is not None else None
    n_edges_rpc = int((edges_df["n_rpc"] > 0).sum()) if edges_df is not None else None
    n_edges_msg = int((edges_df["n_messaging"] > 0).sum()) if edges_df is not None else None

    top_services_html = ""
    if spans_df is not None:
        top_svc = (
            spans_df.groupby("service_name", as_index=False).size().sort_values("size", ascending=False).head(10)
        )
        (assets / "top_services.csv").write_text(top_svc.to_csv(index=False))
        rows = "".join(f"<tr><td>{i+1}</td><td>{r.service_name}</td><td>{int(r.size)}</td></tr>"
                       for i, r in enumerate(top_svc.itertuples(index=False)))
        top_services_html = (
            "<h3>Top services by span count</h3>"
            "<table><thead><tr><th>#</th><th>Service</th><th>spans</th></tr></thead><tbody>"
            f"{rows}</tbody></table>"
            "<p><a href='data/top_services.csv' download>Download CSV</a></p>"
        )

    top_edges_html = ""
    if edges_df is not None:
        top_edges = (
            edges_df.sort_values("n_events", ascending=False)
                    .loc[:, ["src_service","dst_service","n_events","n_rpc","n_messaging"]]
                    .head(10)
        )
        (assets / "top_edges.csv").write_text(top_edges.to_csv(index=False))
        rows = "".join(
            f"<tr><td>{i+1}</td><td>{r.src_service} → {r.dst_service}</td>"
            f"<td>{int(r.n_events)}</td><td>{int(r.n_rpc)}</td><td>{int(r.n_messaging)}</td></tr>"
            for i, r in enumerate(top_edges.itertuples(index=False))
        )
        top_edges_html = (
            "<h3>Top edges by events</h3>"
            "<table><thead><tr><th>#</th><th>Edge</th><th>events</th><th>rpc</th><th>messaging</th></tr></thead><tbody>"
            f"{rows}</tbody></table>"
            "<p><a href='data/top_edges.csv' download>Download CSV</a></p>"
        )

    # ---------- coverage computation ----------
    coverage_html = ""
    if feats_df is not None and gt_path:
        gt_df, is_yaml = _load_ground_truth(gt_path)

        def _is_infra(name: str) -> bool:
            n = _normalize_service_name(name)
            tokens = {"otelcollector","otelcol","otel","jaeger","opensearch",
                      "grafana","prometheus","loki","tempo","zookeeper",
                      "kafka","kafkaui","ui","loadgenerator","locust","frontendproxy"}
            return any(t in n for t in tokens)

        cand = feats_df[["src_service","dst_service","n_events","n_rpc","n_messaging"]].drop_duplicates()
        cand = cand[~cand["src_service"].map(_is_infra) & ~cand["dst_service"].map(_is_infra)]
        matched = _match_ground_truth(cand, gt_df, is_yaml=is_yaml)
        n_total = int(cand.shape[0]); n_matched = int(matched.shape[0]); pct = (n_matched/n_total*100.0) if n_total else 0.0

        unmatched = (
            cand.merge(matched.assign(_m=1), on=["src_service","dst_service"], how="left")
                .query("_m.isna()").drop(columns=["_m"])
                .sort_values("n_events", ascending=False).head(coverage_top)
        )
        (assets / "coverage_unmatched_top.csv").write_text(unmatched.to_csv(index=False))

        rows = "".join(
            f"<tr><td>{i+1}</td><td>{r.src_service} → {r.dst_service}</td>"
            f"<td>{int(r.n_events)}</td><td>{int(r.n_rpc)}</td><td>{int(r.n_messaging)}</td></tr>"
            for i, r in enumerate(unmatched.itertuples(index=False))
        )
        coverage_html = (
            "<h2>Ground‑truth coverage</h2>"
            f"<p>Matched <b>{n_matched}</b> of <b>{n_total}</b> discovered <i>app‑level</i> edges "
            f"(<b>{pct:.1f}%</b>) after ignoring infra/broker edges.</p>"
            "<p>Top unmatched edges by event volume:</p>"
            "<table><thead><tr><th>#</th><th>Edge</th><th>events</th><th>rpc</th><th>messaging</th></tr></thead><tbody>"
            f"{rows}</tbody></table>"
            "<p><a href='data/coverage_unmatched_top.csv' download>Download CSV</a></p>"
        )

    # ---------- Downloads: copy debug CSV if present ----------
    debug_src = next(metrics_dir.glob("**/debug_matched_edges.csv"), None)
    debug_link_html = ""
    if debug_src and debug_src.is_file():
        dst = assets / "debug_matched_edges.csv"
        try:
            shutil.copyfile(debug_src, dst)
            debug_link_html = "<li><a href='data/debug_matched_edges.csv' download>debug_matched_edges.csv</a> (matched edges + features + predictions)</li>"
        except Exception:
            pass

    # ---------- Provenance ----------
    prov_html = ""
    if prov_path and prov_path.exists():
        try:
            prov = json.loads(prov_path.read_text())
            def g(k, d="—"): return prov.get(k, d)
            rows = "".join([
                f"<tr><td>Target</td><td>{g('target')}</td></tr>",
                f"<tr><td>Demo ref</td><td>{g('demo_ref')}</td></tr>",
                f"<tr><td>Demo commit</td><td><code>{g('demo_commit')}</code></td></tr>",
                f"<tr><td>Soak seconds</td><td>{g('soak_seconds')}</td></tr>",
                f"<tr><td>Trace file size</td><td>{g('traces_size_mb','—')} MB</td></tr>",
                f"<tr><td>Docker</td><td>{g('docker_version')}</td></tr>",
                f"<tr><td>Compose</td><td>{g('compose_version')}</td></tr>",
                f"<tr><td>Python</td><td>{g('python_version')}</td></tr>",
                f"<tr><td>EdgeTyper</td><td>{g('edgetyper_version')}</td></tr>",
            ])
            prov_html = (
                "<h2>Provenance</h2>"
                "<table><thead><tr><th>Key</th><th>Value</th></tr></thead><tbody>"
                f"{rows}</tbody></table>"
            )
        except Exception:
            pass

    # ---------- render (title + metrics) ----------
    html = [
        "<html><head><meta charset='utf-8'><title>EdgeTyper — Results</title>",
        "<style>body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:24px}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #ddd;padding:8px;text-align:center}"
        "th{background:#f5f5f5}</style>",
        "</head><body>",
        "<h1>EdgeTyper — Typed Service Edges</h1>",
        "<p>Metrics are edge‑level. Per‑class cells show <em>precision/recall/F1</em>. "
        "Confusion matrix is for labels [async, sync]. "
        f"Counts shown below are <b>{count_mode.upper()}</b> per run.</p>",
    ]

    if not items:
        html.append("<p><b>No metrics found.</b></p>")
    else:
        # dynamic count header
        if count_mode.lower() == "pred":
            count_hdr = "<th>#Pred Async</th><th>#Pred Sync</th>"
        elif count_mode.lower() == "gt":
            count_hdr = "<th>#Async (GT)</th><th>#Sync (GT)</th>"
        else:
            count_hdr = "<th>#Async (GT)</th><th>#Sync (GT)</th><th>#Pred Async</th><th>#Pred Sync</th>"

        html.append(f"""
        <table>
          <thead>
            <tr>
              <th>Run</th><th>Eval edges</th>{count_hdr}
              <th>Async P/R/F1</th><th>Sync P/R/F1</th><th>Macro P/R/F1</th><th>Confusion (TP/FP/FN/TN)</th>
            </tr>
          </thead><tbody>""")

        def s(m, cls, k):
            rep = m.get("classification_report", {})
            return f'{rep.get(cls, {}).get(k, 0):.3f}'

        for m in items:
            cm = m.get("confusion_matrix", [[0,0],[0,0]])
            # labels: [async, sync]
            tp, fp, fn, tn = cm[0][0], cm[1][0], cm[0][1], cm[1][1]
            gt_async  = tp + fn
            gt_sync   = tn + fp
            pred_async = tp + fp
            pred_sync  = tn + fn

            if count_mode.lower() == "pred":
                counts_html = f"<td>{pred_async}</td><td>{pred_sync}</td>"
            elif count_mode.lower() == "gt":
                counts_html = f"<td>{gt_async}</td><td>{gt_sync}</td>"
            else:
                counts_html = f"<td>{gt_async}</td><td>{gt_sync}</td><td>{pred_async}</td><td>{pred_sync}</td>"

            html.append(
                "<tr>"
                f"<td><b>{m['run_name']}</b></td>"
                f"<td>{m.get('n_eval_edges', gt_async + gt_sync)}</td>"
                f"{counts_html}"
                f"<td>{s(m,'async','precision')}/{s(m,'async','recall')}/{s(m,'async','f1-score')}</td>"
                f"<td>{s(m,'sync','precision')}/{s(m,'sync','recall')}/{s(m,'sync','f1-score')}</td>"
                f"<td>{s(m,'macro avg','precision')}/{s(m,'macro avg','recall')}/{s(m,'macro avg','f1-score')}</td>"
                f"<td>TP={tp}, FP={fp}, FN={fn}, TN={tn}</td>"
                "</tr>"
            )
        html.append("</tbody></table>")

    # ---------- snapshot ----------
    snapshot_rows = []
    if n_spans is not None:     snapshot_rows.append(f"<tr><td>Spans parsed</td><td>{n_spans}</td></tr>")
    if n_services is not None:  snapshot_rows.append(f"<tr><td>Services discovered</td><td>{n_services}</td></tr>")
    if n_events is not None:    snapshot_rows.append(f"<tr><td>Interactions (events)</td><td>{n_events}</td></tr>")
    if n_edges is not None:     snapshot_rows.append(f"<tr><td>Edges discovered</td><td>{n_edges}</td></tr>")
    if n_edges_rpc is not None and n_edges_msg is not None:
        snapshot_rows.append(f"<tr><td>Edges by type</td><td>rpc={n_edges_rpc}, messaging={n_edges_msg}</td></tr>")

    if snapshot_rows:
        html.append("<h2>Dataset snapshot</h2>"
                    "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>"
                    + "".join(snapshot_rows) + "</tbody></table>")
    if top_services_html: html.append(top_services_html)
    if top_edges_html:    html.append(top_edges_html)

    # ---------- coverage, provenance, downloads ----------
    if coverage_html: html.append(coverage_html)
    if prov_html:     html.append(prov_html)
    dl = ["<h2>Downloads</h2><ul>"]
    if (assets / "coverage_unmatched_top.csv").exists():
        dl.append("<li><a href='data/coverage_unmatched_top.csv' download>coverage_unmatched_top.csv</a></li>")
    if (assets / "top_services.csv").exists():
        dl.append("<li><a href='data/top_services.csv' download>top_services.csv</a></li>")
    if (assets / "top_edges.csv").exists():
        dl.append("<li><a href='data/top_edges.csv' download>top_edges.csv</a></li>")
    if debug_link_html: dl.append(debug_link_html)
    dl.append("</ul>")
    html.extend(dl)

    # ---------- interpretation ----------
    html.append("<h2>Interpretation</h2><ul>"
                "<li><b>Baseline — SemConv</b> uses OpenTelemetry messaging semantics only (PRODUCER/CONSUMER & messaging.*).</li>"
                "<li><b>Baseline — Timing</b> uses only timing/overlap.</li>"
                "<li><b>EdgeTyper (ours)</b> combines SemConv with timing and span links; robust when one signal is missing.</li>"
                "</ul>")

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
