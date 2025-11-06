from __future__ import annotations

import json
import re
from collections import Counter
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
from edgetyper.resilience import (
    SimConfig, blocking_adjacency_from_edges, guess_entrypoints, estimate_availability
)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """EdgeTyper CLI (OpenTelemetry Demo → traces → analysis)."""
    pass


# ---------------- extract ----------------
@main.command("extract")
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--out",   "out_path",   type=click.Path(dir_okay=False,        path_type=Path), required=True)
def extract_cmd(input_path: Path, out_path: Path, **_ignore):
    """Parse OTLP‑JSON into spans.parquet."""
    df = read_otlp_json(Path(input_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    click.echo(f"[extract] wrote {len(df)} spans → {out_path}")


# ---------------- graph ----------------
@main.command("graph")
@click.option("--spans", "spans_path", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--out-events", "out_events", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--out-edges", "out_edges", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--with-broker-edges/--no-broker-edges", default=True, show_default=True,
              help="Emit producer→broker and broker→consumer edges in addition to producer→consumer.")
@click.option("--broker-service", default="kafka", show_default=True, help="Name to use for the broker node.")
def graph_cmd(spans_path: Path, out_events: Path, out_edges: Path, with_broker_edges: bool, broker_service: str) -> None:
    def _maybe_restore_spans() -> pd.DataFrame:
        if spans_path.exists():
            return pd.read_parquet(spans_path)

        json_fallback = spans_path.with_suffix(".json")
        if json_fallback.exists():
            click.echo(
                f"[graph] spans parquet missing; rebuilding from {json_fallback.name}",
                err=True,
            )
            df = read_otlp_json(json_fallback)
            try:
                spans_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(spans_path, index=False)
                click.echo(f"[graph] wrote reconstructed spans → {spans_path}")
            except Exception as exc:
                raise click.ClickException(
                    f"Failed to materialize spans parquet at {spans_path}: {exc}"
                )
            return df

        raise click.ClickException(
            f"Spans parquet not found: {spans_path}. Run 'edgetyper extract' first or provide the JSON next to it."
        )

    spans = _maybe_restore_spans()
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
@click.option(
    "--uncertain-threshold",
    type=click.FloatRange(0.0, 1.0),
    default=None,
    show_default=False,
    help="If set, mark rule-confidence below the threshold as 'uncertain' (skip ML fallback).",
)
def label_cmd(features_path: Path, out_path: Path, uncertain_threshold: float | None) -> None:
    feats = pd.read_parquet(features_path)
    rules_df = rule_labels_from_features(feats)
    if uncertain_threshold is not None:
        def _conf_to_float(val: object) -> float:
            if isinstance(val, str):
                v = val.lower()
                if v == "high":
                    return 1.0
                if v == "low":
                    return 0.0
                try:
                    return float(val)
                except Exception:
                    return 0.0
            try:
                return float(val)  # type: ignore[arg-type]
            except Exception:
                return 0.0

        if "rule_conf" in rules_df.columns:
            conf_raw = rules_df["rule_conf"]
        else:
            conf_raw = pd.Series(["low"] * len(rules_df), index=rules_df.index)
        conf = conf_raw.map(_conf_to_float)
        pred = rules_df[["src_service", "dst_service", "rule_label"]].copy()
        pred["pred_label"] = "uncertain"
        pred["pred_score"] = 0.5
        confident = conf >= uncertain_threshold
        confident_idx = confident[confident].index
        if not confident_idx.empty:
            confident_labels = rules_df.loc[confident_idx, "rule_label"].replace({"unknown": "uncertain"})
            pred.loc[confident_idx, "pred_label"] = confident_labels
            pred.loc[confident_idx, "pred_score"] = pred.loc[confident_idx, "pred_label"].map(
                {"async": 1.0, "sync": 0.0}
            ).fillna(0.5)
        pred = pred[["src_service", "dst_service", "pred_label", "pred_score"]]
    else:
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
                if g["src_re"].search(str(e["src_service"])) and g["dst_re"].search(str(e["dst_service"])):  # noqa: E501
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
    merged["pred_label"] = merged["pred_label"].astype(str)
    n_uncertain_pred = int((merged["pred_label"] == "uncertain").sum())
    evaluated = merged[merged["pred_label"] != "uncertain"].copy()

    if evaluated.empty:
        report = {
            "async": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0},
            "sync": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0},
            "accuracy": 0.0,
            "macro avg": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0},
            "weighted avg": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0},
        }
        cm = [[0, 0], [0, 0]]
        n_async = 0
        n_sync = 0
    else:
        y_true = evaluated["gt_label"].astype(str)
        y_pred = evaluated["pred_label"].astype(str)
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
        "n_eval_edges": int(len(evaluated)),
        "n_async": n_async,
        "n_sync": n_sync,
        "classification_report": report,
        "confusion_matrix": cm,
        "n_uncertain_pred": n_uncertain_pred,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    click.echo(f"[eval] ({run_name}) evaluated {len(merged)} edges → {out_path}")


# ---------------- report ----------------
@main.command("report")
@click.option("--metrics-dir", "--metrics", "metrics_dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--outdir", "--out", "outdir", type=click.Path(file_okay=False, path_type=Path), required=True)
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
@click.option("--plan", "plan_csv", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=False,
              help="Optional chaos plan CSV to embed.")
@click.option("--plan-blocking", "plan_blocking_csv", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=False,
              help="Optional chaos plan CSV computed with the all-blocking assumption.")
@click.option("--live", "live_json", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=False,
              help="Optional live observations JSON to embed.")
@click.option("--include-brokers", is_flag=True, default=True,
              help="Include broker edges (e.g., kafka) in coverage.")
@click.option("--availability-typed", "availability_typed_csv", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=False,
              help="Optional availability CSV from `edgetyper resilience` on the typed graph.")
@click.option("--availability-block", "availability_block_csv", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=False,
              help="Optional availability CSV from `edgetyper resilience --assume-all-blocking`.")
def report_cmd(metrics_dir: Path, outdir: Path, spans_path: Path | None, events_path: Path | None,
               edges_path: Path | None, features_path: Path | None, gt_path: Path | None, coverage_top: int,
               prov_path: Path | None, assets_dir: Path | None, count_mode: str,
               plan_csv: Path | None, plan_blocking_csv: Path | None,
               live_json: Path | None, include_brokers: bool,
               availability_typed_csv: Path | None, availability_block_csv: Path | None) -> None:
    import re, shutil

    outdir.mkdir(parents=True, exist_ok=True)
    assets = assets_dir or (outdir / "data")
    assets.mkdir(parents=True, exist_ok=True)

    # --- Auto-detect optional inputs in metrics_dir when flags are omitted ---
    # This makes the workflow call resilient: we can invoke 'report' with only --metrics/--out.
    if plan_csv is None:
        _c = metrics_dir / "plan_physical.csv"
        if _c.exists():
            plan_csv = _c
    if plan_blocking_csv is None:
        _c = metrics_dir / "plan_all_blocking.csv"
        if _c.exists():
            plan_blocking_csv = _c
    if availability_typed_csv is None:
        _c = metrics_dir / "availability_typed.csv"
        if _c.exists():
            availability_typed_csv = _c
    if availability_block_csv is None:
        _c = metrics_dir / "availability_block.csv"
        if _c.exists():
            availability_block_csv = _c

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
        s = name.lower()
        if "semconv dropped" in s:
            return "semconv_missing"
        if "timing dropped" in s:
            return "timing_missing"
        if "timing" in s:
            return "timing"
        if "semconv" in s:
            return "semconv"
        return "full"

    def method_of(name: str) -> str:
        s = name.lower()
        if any(tok in s for tok in ("ours", "edgetyper", "typed")):
            return "ours"
        if "baseline" in s and "semconv" in s:
            return "semconv"
        if "baseline" in s and "timing" in s:
            return "timing"
        if "semconv" in s:
            return "semconv"
        if "timing" in s:
            return "timing"
        return "ours"  # safe fallback so we don’t lose rows

    dedup = {}
    for m in items:
        m["run_name"] = canon(m["run_name"])
        key = (scenario_of(m["run_name"]), method_of(m["run_name"]))
        dedup[key] = m  # keep last if duplicates
    items = list(dedup.values())

    scenario_order = {
        "timing": 0,
        "semconv": 1,
        "full": 2,
        "semconv_missing": 3,
        "timing_missing": 4,
    }
    method_order   = {"timing": 0, "semconv": 1, "ours": 2}
    items.sort(
        key=lambda m: (
            scenario_order.get(scenario_of(m["run_name"]), len(scenario_order)),
            method_order.get(method_of(m["run_name"]), len(method_order)),
            m["run_name"],
        )
    )

    # ---------- snapshot (counts & tops) ----------
    spans_df  = pd.read_parquet(spans_path)  if spans_path  else None
    events_df = pd.read_parquet(events_path) if events_path else None
    edges_df  = pd.read_parquet(edges_path)  if edges_path  else None
    feats_df  = pd.read_parquet(features_path) if features_path else None
    typed_plan_df = None
    blocking_plan_df = None
    live_data: dict | None = None

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
                      "kafkaui","ui","loadgenerator","locust","frontendproxy",
                      "flagd","email"}
            if not include_brokers:
                tokens.update({"kafka","zookeeper"})
            return any(t in n for t in tokens)

        cand = feats_df[["src_service","dst_service","n_events","n_rpc","n_messaging"]].drop_duplicates()
        cand = cand[~cand["src_service"].map(_is_infra) & ~cand["dst_service"].map(_is_infra)]
        is_rpc = cand["n_rpc"] > 0
        if include_brokers:
            # physical messaging only: must touch the broker
            is_msg_phys = (cand["n_messaging"] > 0) & (
                cand["src_service"].str.contains("kafka", case=False) |
                cand["dst_service"].str.contains("kafka", case=False)
            )
        else:
            # brokers excluded → no messaging edges remain for evaluation
            is_msg_phys = pd.Series(False, index=cand.index)
        
        cand = cand[is_rpc | is_msg_phys]
        
        matched = _match_ground_truth(cand, gt_df, is_yaml=is_yaml)
        n_total = int(cand.shape[0]); n_matched = int(matched.shape[0]); pct = (n_matched/n_total*100.0) if n_total else 0.0

        unmatched = (
            cand.merge(matched.assign(_m=1), on=["src_service","dst_service"], how="left")
                .query("_m.isna()").drop(columns=["_m"])
                .sort_values("n_events", ascending=False).head(coverage_top)
        )
        (assets / "coverage_unmatched_top.csv").write_text(unmatched.to_csv(index=False))

        ignored_text = "infra edges" if include_brokers else "infra/broker edges"
        rows = "".join(
            f"<tr><td>{i+1}</td><td>{r.src_service} → {r.dst_service}</td>"
            f"<td>{int(r.n_events)}</td><td>{int(r.n_rpc)}</td><td>{int(r.n_messaging)}</td></tr>"
            for i, r in enumerate(unmatched.itertuples(index=False))
        )
        coverage_html = (
            "<h2>Ground‑truth coverage</h2>"
            f"<p>Matched <b>{n_matched}</b> of <b>{n_total}</b> discovered <i>app‑level</i> edges "
            f"(<b>{pct:.1f}%</b>) after ignoring {ignored_text}.</p>"
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
        cmode = count_mode.lower()
        if cmode == "pred":
            count_hdr = "<th>#Pred Async</th><th>#Pred Sync</th><th>#Pred Uncertain</th>"
        elif cmode == "gt":
            count_hdr = "<th>#Async (GT)</th><th>#Sync (GT)</th>"
        else:
            count_hdr = (
                "<th>#Async (GT)</th><th>#Sync (GT)</th>"
                "<th>#Pred Async</th><th>#Pred Sync</th><th>#Pred Uncertain</th>"
            )

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

            pred_uncertain = int(m.get("n_uncertain_pred", 0) or 0)

            if cmode == "pred":
                counts_html = f"<td>{pred_async}</td><td>{pred_sync}</td><td>{pred_uncertain}</td>"
            elif cmode == "gt":
                counts_html = f"<td>{gt_async}</td><td>{gt_sync}</td>"
            else:
                counts_html = (
                    f"<td>{gt_async}</td><td>{gt_sync}</td>"
                    f"<td>{pred_async}</td><td>{pred_sync}</td><td>{pred_uncertain}</td>"
                )

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
    # Include JSON artifacts if present
    try:
        for _name in ("graph.json", "availability.json"):
            _p = metrics_dir / _name
            if _p.exists():
                shutil.copyfile(_p, assets / _name)
                dl.append(f"<li><a href='data/{_name}' download>{_name}</a></li>")
    except Exception:
        pass

    if (assets / "coverage_unmatched_top.csv").exists():
        dl.append("<li><a href='data/coverage_unmatched_top.csv' download>coverage_unmatched_top.csv</a></li>")
    if (assets / "top_services.csv").exists():
        dl.append("<li><a href='data/top_services.csv' download>top_services.csv</a></li>")
    if (assets / "top_edges.csv").exists():
        dl.append("<li><a href='data/top_edges.csv' download>top_edges.csv</a></li>")
    if (assets / "plan_all_blocking.csv").exists():
        dl.append("<li><a href='data/plan_all_blocking.csv' download>plan_all_blocking.csv</a></li>")
    if (assets / "resilience_compare.csv").exists():
        dl.append("<li><a href='data/resilience_compare.csv' download>resilience_compare.csv</a></li>")
    if availability_typed_csv:
        try:
            shutil.copyfile(availability_typed_csv, assets / "availability_typed.csv")
            dl.append("<li><a href='data/availability_typed.csv'>Availability (typed)</a></li>")
        except Exception:
            pass
    if availability_block_csv:
        try:
            shutil.copyfile(availability_block_csv, assets / "availability_block.csv")
            dl.append("<li><a href='data/availability_block.csv'>Availability (all‑blocking)</a></li>")
        except Exception:
            pass
    if debug_link_html: dl.append(debug_link_html)
    dl.append("</ul>")
    html.extend(dl)

    # ---------- Availability (table) ----------
    try:
        av_t, av_b, av_l = None, None, None
        if availability_typed_csv:
            av_t = pd.read_csv(availability_typed_csv).rename(columns={"R_model":"R_model_typed"})
        if availability_block_csv:
            av_b = pd.read_csv(availability_block_csv).rename(columns={"R_model":"R_model_block"})
        live_csv_path = metrics_dir / "live_availability.csv"
        if live_csv_path.exists():
            av_l = pd.read_csv(live_csv_path)
        mix = None
        def _num(x):
            try:
                return float(x)
            except Exception:
                return None
        if av_t is not None:
            av_t["p_fail"] = av_t["p_fail"].apply(_num)
            mix = av_t
        if av_b is not None:
            av_b["p_fail"] = av_b["p_fail"].apply(_num)
            mix = mix.merge(av_b, on=["entrypoint","p_fail"], how="outer") if mix is not None else av_b
        if av_l is not None:
            av_l["p_fail"] = av_l["p_fail"].apply(_num)
            mix = mix.merge(av_l, on=["entrypoint","p_fail"], how="left") if mix is not None else av_l
            if "R_live" in mix.columns:
                if "R_model_typed" in mix.columns:
                    mix["MAE_typed"] = (mix["R_model_typed"] - mix["R_live"]).abs()
                if "R_model_block" in mix.columns:
                    mix["MAE_block"] = (mix["R_model_block"] - mix["R_live"]).abs()
        if mix is not None and not mix.empty:
            mix = mix.sort_values(["entrypoint","p_fail"]).reset_index(drop=True)
            def _fmt3(val):
                try:
                    return f"{float(val):.3f}"
                except Exception:
                    return "—"
            rows = []
            for r in mix.itertuples(index=False):
                rows.append(
                    "<tr>"
                    f"<td>{getattr(r,'entrypoint','')}</td>"
                    f"<td>{getattr(r,'p_fail','')}</td>"
                    f"<td>{_fmt3(getattr(r,'R_model_typed', float('nan')))}</td>"
                    f"<td>{_fmt3(getattr(r,'R_model_block', float('nan')))}</td>"
                    f"<td>{_fmt3(getattr(r,'R_live', float('nan')))}</td>"
                    f"<td>{_fmt3(getattr(r,'MAE_typed', float('nan')))}</td>"
                    f"<td>{_fmt3(getattr(r,'MAE_block', float('nan')))}</td>"
                    "</tr>"
                )
            html.append("<h2>Availability</h2>")
            html.append(
                "<table><thead><tr>"
                "<th>Entrypoint</th><th>p_fail</th>"
                "<th>Typed (R_model)</th><th>All‑blocking (R_model)</th>"
                "<th>Live (R_live)</th><th>MAE typed</th><th>MAE all‑block</th>"
                "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
            )
    except Exception:
        pass


    # ---------- interpretation ----------
    html.append("<h2>Interpretation</h2><ul>"
                "<li><b>Baseline — SemConv</b> uses OpenTelemetry messaging semantics only (PRODUCER/CONSUMER & messaging.*).</li>"
                "<li><b>Baseline — Timing</b> uses only timing/overlap.</li>"
                "<li><b>EdgeTyper (ours)</b> combines SemConv with timing and span links; robust when one signal is missing.</li>"
                "</ul>")
                 
    # ---- Chaos plan (if provided)
    if plan_csv:
        try:
            typed_plan_df = pd.read_csv(plan_csv)
            for col in ["IBS", "DBS"]:
                if col in typed_plan_df.columns:
                    typed_plan_df[col] = pd.to_numeric(typed_plan_df[col], errors="coerce").fillna(0.0)
            typed_plan_df["typed_score"] = typed_plan_df.get("IBS", 0.0) + typed_plan_df.get("DBS", 0.0)
            typed_plan_df = typed_plan_df.sort_values(["IBS", "DBS"], ascending=False)
            (assets / "plan_physical.csv").write_text(typed_plan_df.to_csv(index=False))
            rows = ""
            for i, r in enumerate(typed_plan_df.head(10).itertuples(index=False), 1):
                rows += (f"<tr><td>{i}</td><td>{r.target}</td><td>{r.kind}</td>"
                         f"<td>{(r.IBS):.1f}</td><td>{(r.DBS):.1f}</td>"
                         f"<td>{(r.ib_edges_top or '')}</td><td>{(r.db_edges_top or '')}</td></tr>")
            html.append("<h2>Chaos plan (draft, physical graph)</h2>"
                        "<table><thead><tr><th>#</th><th>Target</th><th>Kind</th>"
                        "<th>IBS</th><th>DBS</th><th>Top blocking contributors</th><th>Top async contributors</th>"
                        "</tr></thead><tbody>"+rows+"</tbody></table>"
                        "<p><a href='data/plan_physical.csv' download>Download plan_physical.csv</a></p>")
        except Exception:
            typed_plan_df = None

    if plan_blocking_csv:
        try:
            blocking_plan_df = pd.read_csv(plan_blocking_csv)
            for col in ["IBS", "DBS"]:
                if col in blocking_plan_df.columns:
                    blocking_plan_df[col] = pd.to_numeric(blocking_plan_df[col], errors="coerce").fillna(0.0)
            blocking_plan_df["blocking_score"] = (
                blocking_plan_df.get("IBS", 0.0) + blocking_plan_df.get("DBS", 0.0)
            )
            blocking_plan_df = blocking_plan_df.sort_values(["IBS", "DBS"], ascending=False)
            (assets / "plan_all_blocking.csv").write_text(blocking_plan_df.to_csv(index=False))
        except Exception:
            blocking_plan_df = None

    if typed_plan_df is not None and (assets / "plan_all_blocking.csv").exists():
        html.append(
            "<p><a href='data/plan_all_blocking.csv' download>Download plan_all_blocking.csv</a> "
            "(all-blocking baseline)</p>"
        )
    
    # ---- Live sanity (if provided)
    if live_json:
        live_data = json.loads(Path(live_json).read_text())
        if live_data.get("ok"):
            html.append("<h2>Live sanity (micro‑faults)</h2>")
            html.append("<ul>")
            for seg in live_data.get("segments", []):
                html.append(f"<li><b>{seg['name']}</b>: total spans={seg['total_spans']}, kinds={seg.get('by_kind',{})}</li>")
            html.append("</ul>")
            dst = assets / "observations.json"
            try:
                import shutil
                shutil.copyfile(live_json, dst)
                html.append("<p><a href='data/observations.json' download>observations.json</a></p>")
            except Exception:
                pass

    if (
        typed_plan_df is not None
        and blocking_plan_df is not None
        and live_data is not None
        and live_data.get("ok")
    ):
        try:
            typed_cols = typed_plan_df.copy()
            block_cols = blocking_plan_df.copy()

            keep_cols_typed = ["target", "kind", "IBS", "DBS", "typed_score", "ib_edges_top", "db_edges_top"]
            keep_cols_block = ["target", "kind", "IBS", "DBS", "blocking_score"]
            typed_cols = typed_cols[[c for c in keep_cols_typed if c in typed_cols.columns]]
            block_cols = block_cols[[c for c in keep_cols_block if c in block_cols.columns]]

            typed_cols = typed_cols.rename(
                columns={
                    "kind": "kind_typed",
                    "IBS": "IBS_typed",
                    "DBS": "DBS_typed",
                    "ib_edges_top": "ib_edges_top_typed",
                    "db_edges_top": "db_edges_top_typed",
                }
            )
            block_cols = block_cols.rename(
                columns={
                    "kind": "kind_blocking",
                    "IBS": "IBS_blocking",
                    "DBS": "DBS_blocking",
                }
            )

            comp = typed_cols.merge(block_cols, on="target", how="outer")
            comp["kind"] = comp.get("kind_typed").fillna(comp.get("kind_blocking")).fillna("app")
            for col in [
                "IBS_typed",
                "DBS_typed",
                "typed_score",
                "IBS_blocking",
                "DBS_blocking",
                "blocking_score",
            ]:
                if col in comp.columns:
                    comp[col] = pd.to_numeric(comp[col], errors="coerce").fillna(0.0)

            typed_score_map = {
                str(row.target): float(row.typed_score)
                for row in comp.itertuples(index=False)
                if getattr(row, "typed_score", None) is not None
            }
            block_score_map = {
                str(row.target): float(row.blocking_score)
                for row in comp.itertuples(index=False)
                if getattr(row, "blocking_score", None) is not None
            }

            def _counts_per_sec(seg: dict[str, object] | None) -> tuple[dict[str, float], dict[str, float], float]:
                if not seg:
                    return {}, {}, 0.0
                counts_raw = seg.get("by_service") or seg.get("by_service_top") or {}
                counts = {str(k): float(v) for k, v in counts_raw.items()}
                duration_s = float(seg.get("duration_s") or 0.0)
                if duration_s <= 0:
                    duration_ns = seg.get("duration_ns")
                    try:
                        duration_s = float(duration_ns) / 1_000_000_000 if duration_ns else 0.0
                    except Exception:
                        duration_s = 0.0
                if duration_s <= 0:
                    s = seg.get("start_ns")
                    e = seg.get("end_ns")
                    if isinstance(s, (int, float)) and isinstance(e, (int, float)) and e > s:
                        duration_s = (float(e) - float(s)) / 1_000_000_000
                if duration_s <= 0:
                    duration_s = 1.0
                per_sec = {svc: cnt / duration_s for svc, cnt in counts.items()}
                return counts, per_sec, duration_s

            def _sym_delta(base: float, fault: float) -> float:
                denom = (abs(base) + abs(fault)) / 2.0
                if denom <= 0:
                    return 0.0
                return abs(fault - base) / denom

            segments = live_data.get("segments", [])
            baseline_seg = None
            for seg in segments:
                if str(seg.get("name", "")).lower().startswith("baseline"):
                    baseline_seg = seg
                    break
            if baseline_seg is None and segments:
                baseline_seg = segments[0]

            base_counts, base_per_sec, _ = _counts_per_sec(baseline_seg)

            def _resolve_target(seg: dict[str, object], deltas: dict[str, float]) -> str | None:
                explicit = seg.get("target_service")
                if explicit:
                    return str(explicit)
                name = str(seg.get("name", ""))
                if ":" in name:
                    return name.split(":", 1)[1]
                if deltas:
                    return max(deltas.items(), key=lambda kv: kv[1])[0]
                return None

            def _corr(scores: dict[str, float], impacts: dict[str, float], method: str) -> float | None:
                common = sorted(set(scores) & set(impacts))
                if len(common) < 2:
                    return None
                s_scores = pd.Series([scores[k] for k in common])
                s_imp = pd.Series([impacts[k] for k in common])
                val = s_scores.corr(s_imp, method=method)
                if pd.isna(val):
                    return None
                return float(val)

            def _fmt(val: float | None) -> str:
                return "n/a" if val is None else f"{val:.3f}"

            def _precision_at_k(scores: dict[str, float], target: str | None, k: int) -> float | None:
                if not target or target not in scores:
                    return None
                ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
                top = [svc for svc, _ in ordered[:k]]
                return 1.0 if target in top else 0.0

            fault_imp_rows: list[dict[str, object]] = []
            fault_metric_rows: list[dict[str, object]] = []
            pooled_lists: dict[str, list[float]] = {}
            max_impacts: dict[str, float] = {}
            p3_typed_vals: list[float] = []
            p3_block_vals: list[float] = []
            p5_typed_vals: list[float] = []
            p5_block_vals: list[float] = []

            for seg in segments:
                if seg is baseline_seg:
                    continue
                counts_fault, per_sec_fault, duration_fault = _counts_per_sec(seg)
                deltas: dict[str, float] = {}
                union = set(base_per_sec) | set(per_sec_fault)
                for svc in union:
                    deltas[svc] = _sym_delta(base_per_sec.get(svc, 0.0), per_sec_fault.get(svc, 0.0))
                    pooled_lists.setdefault(svc, []).append(deltas[svc])
                    max_impacts[svc] = max(max_impacts.get(svc, 0.0), deltas[svc])

                target = _resolve_target(seg, deltas)

                spearman_t = _corr(typed_score_map, deltas, "spearman")
                spearman_b = _corr(block_score_map, deltas, "spearman")
                kendall_t = _corr(typed_score_map, deltas, "kendall")
                kendall_b = _corr(block_score_map, deltas, "kendall")
                p3_t = _precision_at_k(typed_score_map, target, 3)
                p3_b = _precision_at_k(block_score_map, target, 3)
                p5_t = _precision_at_k(typed_score_map, target, 5)
                p5_b = _precision_at_k(block_score_map, target, 5)

                if p3_t is not None:
                    p3_typed_vals.append(p3_t)
                if p3_b is not None:
                    p3_block_vals.append(p3_b)
                if p5_t is not None:
                    p5_typed_vals.append(p5_t)
                if p5_b is not None:
                    p5_block_vals.append(p5_b)

                fault_metric_rows.append(
                    {
                        "fault": seg.get("name"),
                        "target_service": target,
                        "duration_s": duration_fault,
                        "spearman_typed": spearman_t,
                        "spearman_blocking": spearman_b,
                        "kendall_typed": kendall_t,
                        "kendall_blocking": kendall_b,
                        "p_at_3_typed": p3_t,
                        "p_at_3_blocking": p3_b,
                        "p_at_5_typed": p5_t,
                        "p_at_5_blocking": p5_b,
                    }
                )

                for svc in sorted(union):
                    fault_imp_rows.append(
                        {
                            "fault": seg.get("name"),
                            "target_service": target,
                            "service": svc,
                            "baseline_per_sec": base_per_sec.get(svc, 0.0),
                            "fault_per_sec": per_sec_fault.get(svc, 0.0),
                            "sym_relative_delta": deltas.get(svc, 0.0),
                        }
                    )

            pooled_impacts = {svc: sum(vals) / len(vals) for svc, vals in pooled_lists.items() if vals}
            comp["max_live_delta"] = comp["target"].map(lambda s: max_impacts.get(str(s), 0.0))
            comp["mean_live_delta"] = comp["target"].map(lambda s: pooled_impacts.get(str(s), 0.0))

            comp_sorted = comp.sort_values(["typed_score", "blocking_score"], ascending=False).reset_index(drop=True)
            (assets / "resilience_compare.csv").write_text(comp_sorted.to_csv(index=False))

            if fault_imp_rows:
                (assets / "per_fault_impacts.csv").write_text(pd.DataFrame(fault_imp_rows).to_csv(index=False))
            if fault_metric_rows:
                (assets / "per_fault_metrics.csv").write_text(pd.DataFrame(fault_metric_rows).to_csv(index=False))
            if pooled_impacts:
                pooled_df = pd.DataFrame(
                    sorted(pooled_impacts.items(), key=lambda kv: kv[1], reverse=True),
                    columns=["service", "mean_sym_relative_delta"],
                )
                (assets / "pooled_live_impacts.csv").write_text(pooled_df.to_csv(index=False))

            pooled_spearman_t = _corr(typed_score_map, pooled_impacts, "spearman")
            pooled_spearman_b = _corr(block_score_map, pooled_impacts, "spearman")
            pooled_kendall_t = _corr(typed_score_map, pooled_impacts, "kendall")
            pooled_kendall_b = _corr(block_score_map, pooled_impacts, "kendall")

            mean_p3_t = sum(p3_typed_vals) / len(p3_typed_vals) if p3_typed_vals else None
            mean_p3_b = sum(p3_block_vals) / len(p3_block_vals) if p3_block_vals else None
            mean_p5_t = sum(p5_typed_vals) / len(p5_typed_vals) if p5_typed_vals else None
            mean_p5_b = sum(p5_block_vals) / len(p5_block_vals) if p5_block_vals else None

            html.append("<h2>Resilience prediction vs live</h2>")
            html.append(
                "<p>Pooled symmetric per-second deltas (averaged over faults): "
                f"typed Spearman ρ={_fmt(pooled_spearman_t)}, "
                f"all-blocking Spearman ρ={_fmt(pooled_spearman_b)}, "
                f"typed Kendall τ={_fmt(pooled_kendall_t)}, "
                f"all-blocking Kendall τ={_fmt(pooled_kendall_b)}, "
                f"P@3 typed={_fmt(mean_p3_t)}, P@3 all-blocking={_fmt(mean_p3_b)}, "
                f"P@5 typed={_fmt(mean_p5_t)}, P@5 all-blocking={_fmt(mean_p5_b)}.</p>"
            )

            rows = ""
            for i, r in enumerate(comp_sorted.head(15).itertuples(index=False), 1):
                rows += (
                    f"<tr><td>{i}</td><td>{r.target}</td><td>{r.kind}</td>"
                    f"<td>{getattr(r, 'IBS_typed', 0.0):.1f}</td><td>{getattr(r, 'DBS_typed', 0.0):.1f}</td><td>{getattr(r, 'typed_score', 0.0):.1f}</td>"
                    f"<td>{getattr(r, 'IBS_blocking', 0.0):.1f}</td><td>{getattr(r, 'DBS_blocking', 0.0):.1f}</td><td>{getattr(r, 'blocking_score', 0.0):.1f}</td>"
                    f"<td>{getattr(r, 'max_live_delta', 0.0):.3f}</td><td>{getattr(r, 'mean_live_delta', 0.0):.3f}</td></tr>"
                )

            html.append(
                "<table><thead><tr><th>#</th><th>Target</th><th>Kind</th>"
                "<th>Typed IBS</th><th>Typed DBS</th><th>Typed IBS+DBS</th>"
                "<th>All-blocking IBS</th><th>All-blocking DBS</th><th>All-blocking IBS+DBS</th>"
                "<th>Max live Δ</th><th>Mean live Δ</th></tr></thead><tbody>" + rows + "</tbody></table>"
            )

            if fault_metric_rows:
                fault_table_rows = ""
                for r in fault_metric_rows:
                    fault_table_rows += (
                        "<tr>"
                        f"<td>{r['fault']}</td>"
                        f"<td>{r.get('target_service') or '—'}</td>"
                        f"<td>{_fmt(r.get('spearman_typed'))}</td>"
                        f"<td>{_fmt(r.get('spearman_blocking'))}</td>"
                        f"<td>{_fmt(r.get('kendall_typed'))}</td>"
                        f"<td>{_fmt(r.get('kendall_blocking'))}</td>"
                        f"<td>{_fmt(r.get('p_at_3_typed'))}</td>"
                        f"<td>{_fmt(r.get('p_at_3_blocking'))}</td>"
                        f"<td>{_fmt(r.get('p_at_5_typed'))}</td>"
                        f"<td>{_fmt(r.get('p_at_5_blocking'))}</td>"
                        "</tr>"
                    )
                html.append(
                    "<h3>Per-fault rank quality</h3>"
                    "<table><thead><tr><th>Fault</th><th>Target</th>"
                    "<th>ρ typed</th><th>ρ all-block</th><th>τ typed</th><th>τ all-block</th>"
                    "<th>P@3 typed</th><th>P@3 all-block</th><th>P@5 typed</th><th>P@5 all-block</th>"
                    "</tr></thead><tbody>" + fault_table_rows + "</tbody></table>"
                )

            html.append(
                "<p><a href='data/resilience_compare.csv' download>Download resilience_compare.csv</a> | "
                "<a href='data/per_fault_metrics.csv' download>per_fault_metrics.csv</a> | "
                "<a href='data/per_fault_impacts.csv' download>per_fault_impacts.csv</a> | "
                "<a href='data/pooled_live_impacts.csv' download>pooled_live_impacts.csv</a></p>"
            )
        except Exception:
            pass

    html.append("</body></html>")
    (outdir / "index.html").write_text("\n".join(html))
    click.echo(f"[report] wrote {outdir / 'index.html'}")


# ---------------- live-from-locust ----------------
@main.command("live-from-locust")
@click.option("--stats", "stats_csv",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              required=True,
              help="Locust stats CSV (aggregated row or per-endpoint stats).")
@click.option("--failures", "fail_csv",
              type=click.Path(exists=False, dir_okay=False, path_type=Path),
              required=False,
              help="Optional Locust failures CSV to refine failure count.")
@click.option("--entrypoint", "entrypoint", type=str, required=True,
              help="Model entrypoint name (must match 'entrypoint' in availability_*.csv, e.g., 'frontend').")
@click.option("--p-fail", "p_fail", type=float, required=True,
              help="Failure fraction label to align with model grid (e.g., 0.3).")
@click.option("--out", "out_csv",
              type=click.Path(dir_okay=False, path_type=Path),
              required=True,
              help="Output CSV path (usually RUN_DIR/live_availability.csv).")
def live_from_locust_cmd(stats_csv: Path, fail_csv: Path | None,
                         entrypoint: str, p_fail: float, out_csv: Path) -> None:
    """
    Convert Locust CSVs into the minimal live availability grid consumed by the aggregator.
    Output schema: entrypoint,p_fail,R_live
    Robust to different Locust column names: 'Requests'/'Request Count', 'Failures'/'Failure Count'.
    """
    import pandas as _pd

    def _col(df: _pd.DataFrame, candidates: list[str]) -> str | None:
        # case-insensitive resolution
        lut = {c.lower(): c for c in df.columns}
        for c in candidates:
            if c.lower() in lut:
                return lut[c.lower()]
        return None

    df = _pd.read_csv(stats_csv)
    # Prefer 'Aggregated' row if present; otherwise use totals
    if "Name" in df.columns and (df["Name"].astype(str).str.lower() == "aggregated").any():
        row = df[df["Name"].astype(str).str.lower() == "aggregated"].iloc[0:1]
    else:
        row = df

    req_col  = _col(df, ["Requests", "Request Count", "Total Request Count"])
    fail_col = _col(df, ["Failures", "Failure Count", "Total Failure Count"])
    if req_col is None or fail_col is None:
        raise click.ClickException(f"Cannot find request/failure columns in {stats_csv}.")

    total  = float(row[req_col].astype(float).sum())
    fails  = float(row[fail_col].astype(float).sum())

    # If failures CSV is present and has 'Count', prefer it
    if fail_csv and Path(fail_csv).exists():
        try:
            fdf = _pd.read_csv(fail_csv)
            if "Count" in fdf.columns:
                fails = float(fdf["Count"].astype(float).sum())
        except Exception:
            pass

    R = 0.0 if total <= 0 else max(0.0, total - fails) / total
    out = _pd.DataFrame([{"entrypoint": str(entrypoint), "p_fail": float(p_fail), "R_live": float(R)}])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    click.echo(f"[live-from-locust] wrote {out_csv} (R_live={R:.3f}, entrypoint={entrypoint}, p_fail={p_fail})")

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


# ---------------- plan ----------------
@main.command("plan")
@click.option("--edges", "edges_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--pred",  "pred_path",  type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--out",   "out_path",   type=click.Path(dir_okay=False,  path_type=Path),           required=True)
@click.option("--weight", type=click.Choice(["events","rpc","messaging"], case_sensitive=False), default="events",
              show_default=True, help="Edge weight to aggregate.")
@click.option("--alpha-ack", type=float, default=1.0, show_default=True,
              help="Immediate producer impact fraction if the target is a broker (0..1).")
@click.option("--broker-tokens", type=str, default="kafka,zookeeper", show_default=True,
              help="Comma-separated substrings to identify broker nodes.")
@click.option(
    "--assume-all-blocking",
    is_flag=True,
    default=False,
    show_default=True,
    help="Treat every edge as blocking (ignore predicted types).",
)
def plan_cmd(
    edges_path: Path,
    pred_path: Path,
    out_path: Path,
    weight: str,
    alpha_ack: float,
    broker_tokens: str,
    assume_all_blocking: bool,
) -> None:
    from collections import defaultdict, deque

    tok = {t.strip().lower() for t in broker_tokens.split(",") if t.strip()}

    def _fmt_top(df: pd.DataFrame, k: int = 5) -> str:
        if df is None or df.empty:
            return ""
        cols = ["src_service", "dst_service", "w"]
        head = df.loc[:, cols].sort_values("w", ascending=False).head(k)
        return "; ".join(f"{s}→{d} ({int(w)})" for s, d, w in head.itertuples(index=False, name=None))

    def _norm(s: str) -> str:
        return "".join(ch for ch in s.lower() if ch.isalnum())

    def _is_broker(name: str) -> bool:
        n = _norm(str(name))
        return any(t in n for t in tok)

    edges = pd.read_parquet(edges_path)
    preds = pd.read_csv(pred_path)

    # Join predicted labels onto edges
    g = edges.merge(preds[["src_service", "dst_service", "pred_label"]],
                    on=["src_service", "dst_service"], how="left")
    if assume_all_blocking:
        g["etype"] = "BLOCKING"
    else:
        g["etype"] = g["pred_label"].str.lower().map({"async": "ASYNC", "sync": "BLOCKING", "uncertain": "BLOCKING"})
    g = g.dropna(subset=["etype"]).copy()

    wcol = {"events":"n_events","rpc":"n_rpc","messaging":"n_messaging"}[weight.lower()]
    if wcol not in g.columns:
        raise click.ClickException(f"Weight column '{wcol}' not present in {edges_path}")
    g["w"] = g[wcol].fillna(0).astype(float)

    # Split by type
    gb = g[g["etype"]=="BLOCKING"][["src_service","dst_service","w"]].copy()
    ga = g[g["etype"]=="ASYNC"][["src_service","dst_service","w"]].copy()

    # Reverse adjacency for BLOCKING edges (for upstream closure)
    pre = defaultdict(set)
    for s, d, _ in gb.itertuples(index=False, name=None):
        pre[d].add(s)

    nodes = sorted(set(g["src_service"]).union(g["dst_service"]))
    gb_dst_groups = gb.groupby("dst_service")
    ga_dst_groups = ga.groupby("dst_service")

    rows = []
    for v in nodes:
        # Upstream blocking closure U(v)
        U = set()
        q = deque([v])
        while q:
            y = q.popleft()
            for u in pre.get(y, ()):
                if u not in U and u != v:
                    U.add(u)
                    q.append(u)

        # ---- IBS: blocking edges entering U(v) + broker ack term ----
        gbU = gb.iloc[0:0]       # empty frame with schema
        ibs_block = 0.0
        if U:
            keys = [x for x in U if x in gb_dst_groups.groups]
            if keys:
                gbU = pd.concat([gb_dst_groups.get_group(k) for k in keys], ignore_index=True)
                ibs_block = float(gbU["w"].sum())

        ibs_ack = 0.0
        if _is_broker(v) and v in ga_dst_groups.groups:
            ibs_ack = float(ga_dst_groups.get_group(v)["w"].sum()) * max(0.0, min(1.0, alpha_ack))

        IBS = ibs_block + ibs_ack

        # ---- DBS: async edges entering I = U ∪ {v} ----
        I = set(U)
        I.add(v)
        gaI = ga.iloc[0:0]       # empty frame with schema
        DBS = 0.0
        if I:
            keysI = [x for x in I if x in ga_dst_groups.groups]
            if keysI:
                gaI = pd.concat([ga_dst_groups.get_group(k) for k in keysI], ignore_index=True)
                DBS = float(gaI["w"].sum())

        rows.append({
            "target": v,
            "kind": "broker" if _is_broker(v) else "app",
            "IBS": round(IBS, 3),
            "DBS": round(DBS, 3),
            "n_upstream_blocking": len(U),
            "ib_edges_top": _fmt_top(gbU),
            "db_edges_top": _fmt_top(gaI),
        })

    out = pd.DataFrame(rows).sort_values(["IBS", "DBS"], ascending=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    click.echo(f"[plan] wrote plan for {len(out)} nodes → {out_path}")


# ---------------- observe ----------------
@main.command("observe")
@click.option("--spans",    "spans_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--segments", "segments_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="JSON with segments: [{name,start_ns,end_ns}, ...]")
@click.option("--out",      "out_path", type=click.Path(dir_okay=False, path_type=Path), required=True)
def observe_cmd(spans_path: Path, segments_path: Path, out_path: Path) -> None:
    spans = pd.read_parquet(spans_path)
    segs  = json.loads(Path(segments_path).read_text())

    # Flexible field detection
    ts_candidates = [c for c in ["start_time_unix_nano","start_unix_nano","start_ns","time_unix_nano"] if c in spans.columns]
    kind_candidates = [c for c in ["span_kind","kind","kind_name"] if c in spans.columns]
    svc = "service_name" if "service_name" in spans.columns else None
    if not ts_candidates or not svc:
        out_path.write_text(json.dumps({"ok": False, "reason": "timestamps_or_service_missing"}, indent=2))
        click.echo("[observe] timestamps or service_name missing; wrote stub JSON.")
        return

    ts = ts_candidates[0]
    spans["_ts"] = pd.to_numeric(spans[ts], errors="coerce")
    spans = spans.dropna(subset=["_ts"]).copy()

    if kind_candidates:
        kind_col = kind_candidates[0]
        spans["_kind"] = spans[kind_col].astype(str).str.upper()
    else:
        spans["_kind"] = "UNKNOWN"

    result = {"ok": True, "ts_field": ts, "segments": []}
    for seg in segs.get("segments", []):
        name = seg.get("name")
        s, e = int(seg.get("start_ns", 0)), int(seg.get("end_ns", 0))
        df = spans[(spans["_ts"] >= s) & (spans["_ts"] <= e)]
        total = int(len(df))

        duration_ns = max(0, e - s)
        duration_s = duration_ns / 1_000_000_000 if duration_ns > 0 else 0.0

        svc_counts = df.groupby(svc).size().sort_values(ascending=False) if total else pd.Series(dtype="int64")
        by_service_all = {str(k): int(v) for k, v in svc_counts.items()}
        by_service_top = {str(k): int(v) for k, v in svc_counts.head(12).items()}
        if duration_s > 0:
            by_service_per_sec = {k: float(v) / duration_s for k, v in by_service_all.items()}
        else:
            by_service_per_sec = {k: float(v) for k, v in by_service_all.items()}

        by_kind = (
            df.groupby("_kind").size().sort_values(ascending=False).to_dict()
            if total
            else {}
        )
        result["segments"].append({
            "name": name, "start_ns": s, "end_ns": e,
            "duration_ns": duration_ns,
            "duration_s": duration_s,
            "total_spans": total,
            "by_service": by_service_all,
            "by_service_top": by_service_top,
            "by_service_per_sec": by_service_per_sec,
            "by_kind": by_kind,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    click.echo(f"[observe] wrote {out_path}")


# ---------------- availability-live (build live_availability.csv from Locust CSV) ----------------
@main.command("availability-live")
@click.option(
    "--locust-prefix", "locust_prefix",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True
)
@click.option(
    "--targets", "targets_yaml",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=False
)
@click.option(
    "--entrypoints", "eps_path",
    type=click.Path(exists=True, path_type=Path),
    required=False
)
@click.option("--p-fail", "p_fail", type=float, required=True)
@click.option("--out", "out_csv", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--append/--no-append", default=True, show_default=True)
def availability_live_cmd(
    locust_prefix: Path,
    targets_yaml: Path | None,
    eps_path: Path | None,
    p_fail: float,
    out_csv: Path,
    append: bool,
) -> None:
    base = locust_prefix
    stats_path = base.with_name(base.name + "_stats.csv")
    fails_path = base.with_name(base.name + "_failures.csv")
    if not stats_path.exists():
        raise click.ClickException(f"Locust stats file not found: {stats_path}")

    stats = pd.read_csv(stats_path)
    cols = {c.lower().strip().replace(" ", "").replace("_", ""): c for c in stats.columns}
    name_col = cols.get("name", "Name")
    method_col = cols.get("method", None)
    req_col = cols.get("requests", None) or cols.get("requestcount", None) or cols.get("#requests", None)
    if req_col is None:
        num_cols = [c for c in stats.columns if pd.api.types.is_numeric_dtype(stats[c])]
        req_col = num_cols[0] if num_cols else "Requests"
    df_stats = stats.copy()
    df_stats[name_col] = df_stats[name_col].astype(str)
    df_stats = df_stats[~df_stats[name_col].str.contains("Aggregated|Total", case=False, na=False)]

    # ---- failures.csv
    if fails_path.exists():
        fails = pd.read_csv(fails_path)
        fcols = {c.lower().strip().replace(" ", "").replace("_", ""): c for c in fails.columns}
        f_name = fcols.get("name", "Name")
        f_method = fcols.get("method", None)
        f_err = fcols.get("error", "Error")
        f_occ = fcols.get("occurrences", None) or fcols.get("count", None) or "Occurrences"

        def _cat(msg: object) -> str:
            s = str(msg)
            if re.search(r"\b5\d\d\b", s) or re.search(r"status\s*code\s*5\d\d", s, re.I):
                return "5xx"
            if re.search(r"timeout|timed\s*out|readtimeout|connecttimeout", s, re.I):
                return "timeout"
            if re.search(r"connection|refused|reset|broken\s*pipe|socket|dns|ssl|remote|protocol", s, re.I):
                return "socket"
            return "other"

        fails["_cat"] = fails[f_err].map(_cat)
        idx = [f_name]
        if f_method and f_method in fails.columns:
            idx.insert(0, f_method)
        pivot = fails.pivot_table(index=idx, columns="_cat", values=f_occ, aggfunc="sum", fill_value=0).reset_index()
        pivot = pivot.rename(columns={"5xx": "n_5xx", "timeout": "n_timeout", "socket": "n_socket"})
    else:
        pivot = pd.DataFrame(columns=[name_col, "n_5xx", "n_timeout", "n_socket"])
        if method_col:
            pivot[method_col] = ""

    # ---- Name→entrypoint
    rules: list[tuple[str, re.Pattern[str], str | None]] = []
    if targets_yaml:
        cfg = yaml.safe_load(Path(targets_yaml).read_text()) or {}
        for ep, plist in (cfg.get("entrypoints") or {}).items():
            for rule in (plist or []):
                pat = rule.get("name_regex") or rule.get("re")
                if not pat:
                    continue
                meth = rule.get("method")
                try:
                    cre = re.compile(str(pat))
                except Exception:
                    cre = re.compile(str(pat), re.I)
                rules.append((str(ep), cre, (str(meth).upper() if meth else None)))

    eps_filter: set[str] | None = None
    if eps_path:
        try:
            eps_df = pd.read_csv(eps_path)
            col = "entrypoint" if "entrypoint" in eps_df.columns else eps_df.columns[0]
            eps_filter = set(str(x) for x in eps_df[col].dropna().tolist())
        except Exception:
            eps_filter = set(ln.strip() for ln in Path(eps_path).read_text().splitlines() if ln.strip())

    # ---- join stats + failures
    merge_keys = [name_col]
    if method_col and method_col in df_stats.columns and method_col in pivot.columns:
        merge_keys.insert(0, method_col)
    dfm = df_stats.merge(pivot, on=merge_keys, how="left").fillna({"n_5xx": 0, "n_timeout": 0, "n_socket": 0})

    # ---- aggregation by entrypoint
    acc: dict[str, dict[str, float]] = {}
    unmatched = Counter()
    matched_eps: set[str] = set()
    for _, r in dfm.iterrows():
        name = str(r[name_col])
        method = str(r.get(method_col, "")).upper() if method_col else ""
        m = re.match(r"entry:([^:]+):", name)
        ep = m.group(1) if m else None
        if ep is None and rules:
            for ep_name, cre, meth in rules:
                if meth and meth != method:
                    continue
                if cre.search(name):
                    ep = ep_name
                    break
        if eps_filter is not None and ep and ep not in eps_filter:
            continue
        if ep is None:
            total_requests = float(pd.to_numeric(r.get(req_col), errors="coerce") or 0.0)
            if total_requests > 0:
                count = int(total_requests)
                if count <= 0:
                    count = 1
                unmatched[name] += count
            else:
                unmatched[name] += 1
            continue
        n_total = float(pd.to_numeric(r[req_col], errors="coerce") or 0.0)
        if n_total <= 0:
            continue
        n_5xx = float(r.get("n_5xx", 0.0) or 0.0)
        n_to  = float(r.get("n_timeout", 0.0) or 0.0)
        n_so  = float(r.get("n_socket", 0.0) or 0.0)
        s = acc.setdefault(ep, {"n_total": 0.0, "n_5xx": 0.0, "n_timeout": 0.0, "n_socket": 0.0})
        s["n_total"]  += n_total
        s["n_5xx"]    += n_5xx
        s["n_timeout"]+= n_to
        s["n_socket"] += n_so
        matched_eps.add(ep)

    rows = []
    for ep, s in acc.items():
        bad = s["n_5xx"] + s["n_timeout"] + s["n_socket"]
        R = 0.0 if s["n_total"] <= 0 else max(0.0, min(1.0, 1.0 - (bad / s["n_total"])))
        rows.append({
            "entrypoint": ep,
            "p_fail": float(p_fail),
            "R_live": float(R),
            "n_total": int(s["n_total"]),
            "n_5xx":   int(s["n_5xx"]),
            "n_timeout": int(s["n_timeout"]),
            "n_socket":  int(s["n_socket"]),
        })

    missing_eps: list[str] = []
    if eps_filter:
        missing_eps = sorted(eps_filter - matched_eps)
        if missing_eps:
            preview = ", ".join(missing_eps[:5])
            click.echo(
                f"[availability-live] warning: {len(missing_eps)} entrypoints from --entrypoints had no matching requests ({preview})",
                err=True,
            )

    if not rows:
        if eps_filter:
            preview = ", ".join(missing_eps[:5]) if missing_eps else ""
            raise click.ClickException(
                "No requests matched the provided --entrypoints filter. "
                "Ensure the file lists entrypoints emitted by live targets and that mapping rules cover the Locust names."
                + (f" Missing entrypoints: {preview}" if preview else "")
            )
        sample = ", ".join(name for name, _count in unmatched.most_common(5))
        hint = f" Example unmatched names: {sample}" if sample else ""
        raise click.ClickException(
            "No requests matched entrypoint mapping — provide --targets live_targets.yaml with name_regex rules or rely on 'entry:<name>:' prefixes."
            + hint
        )

    if unmatched:
        total_ignored = sum(unmatched.values())
        preview = ", ".join(f"{name} ({count})" for name, count in unmatched.most_common(3))
        click.echo(
            f"[availability-live] warning: ignored {total_ignored} requests with no entrypoint mapping (e.g., {preview})",
            err=True,
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(rows)
    if append and out_csv.exists():
        df_out.to_csv(out_csv, mode="a", header=False, index=False)
    else:
        df_out.to_csv(out_csv, index=False)
    click.echo(f"[availability-live] wrote {len(df_out)} rows → {out_csv}")


# ---------------- resilience (Monte‑Carlo availability) ----------------
@main.command("resilience")
@click.option("--edges", "edges_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--pred",  "pred_path",  type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--replicas", "replicas_path", type=click.Path(exists=False, dir_okay=False, path_type=Path), required=False,
              help="CSV with columns: service, replicas (defaults to 1 if missing).")
@click.option("--entrypoints", "eps_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=False,
              help="Optional text/CSV with entrypoint service names (one per line or column 'entrypoint').")
@click.option("--p-fail", "p_fail", multiple=True, type=float, default=[0.1,0.3,0.5,0.7,0.9], show_default=True)
@click.option("--samples", type=int, default=900000, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--assume-all-blocking", is_flag=True, default=False, show_default=True,
              help="Treat every edge as blocking (ignore predicted types).")
@click.option("--out", "out_csv", type=click.Path(dir_okay=False, path_type=Path), required=True)
def resilience_cmd(edges_path: Path, pred_path: Path, replicas_path: Path | None,
                   eps_path: Path | None, p_fail: tuple[float, ...],
                   samples: int, seed: int, assume_all_blocking: bool, out_csv: Path) -> None:
    """Monte‑Carlo availability estimator (Algorithm 1)."""
    edges = pd.read_parquet(edges_path)
    preds = pd.read_csv(pred_path)
    adj = blocking_adjacency_from_edges(edges, preds, assume_all_blocking=assume_all_blocking)

    # replicas (optional)
    replicas: Dict[str, int] = {}
    if replicas_path:
        rp = Path(replicas_path)
        if rp.exists():
            r = pd.read_csv(rp)
            key = "service" if "service" in r.columns else ("target" if "target" in r.columns else None)
            if key and "replicas" in r.columns:
                replicas = {str(row[key]): int(row["replicas"]) for _, row in r.iterrows()}
        else:
            click.echo(f"[resilience] replicas file not found: {rp} — defaulting to 1 per service", err=True)

    # entrypoints (filter to services present in graph; fallback if none)
    if eps_path:
        try:
            eps_df = pd.read_csv(eps_path)
            col = "entrypoint" if "entrypoint" in eps_df.columns else eps_df.columns[0]
            entrypoints = [str(x) for x in eps_df[col].dropna().astype(str).tolist()]
        except Exception:
            entrypoints = [ln.strip() for ln in Path(eps_path).read_text().splitlines() if ln.strip()]
    else:
        entrypoints = guess_entrypoints(adj)

    # Filter to nodes present in the blocking graph
    services = set(adj.keys()) | {v for vs in adj.values() for v in vs}
    eps_in = [e for e in entrypoints if e in services]
    if not eps_in:
        click.echo("[resilience] Provided entrypoints do not match the graph — guessing from graph indegree-0/top-degree", err=True)
        entrypoints = guess_entrypoints(adj)
        eps_in = [e for e in entrypoints if e in services]
    else:
        entrypoints = eps_in

    # Visibility: log final entrypoints and first services from edges
    try:
        all_services = pd.unique(pd.concat([edges['src_service'].astype(str), edges['dst_service'].astype(str)])).tolist()
        click.echo(f"[resilience] edges services (first 10): {all_services[:10]}")
    except Exception:
        pass
    click.echo(f"[resilience] using entrypoints: {entrypoints}")

    cfg = SimConfig(p_fail=list(p_fail), samples=int(samples), seed=int(seed))
    out = estimate_availability(adj, replicas, entrypoints, cfg)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    click.echo(f"[resilience] wrote {len(out)} rows → {out_csv}")


# --------------------------------
if __name__ == "__main__":
    main()
