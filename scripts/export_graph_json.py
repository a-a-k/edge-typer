#!/usr/bin/env python3
# scripts/export_graph_json.py
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", required=True, type=Path)
    ap.add_argument("--features", type=Path)           # optional, enrich edges
    ap.add_argument("--pred", type=Path)               # optional, pred_ours.csv
    ap.add_argument("--out", required=True, type=Path) # e.g., out/graph.json
    args = ap.parse_args()

    edges = pd.read_parquet(args.edges)
    # reduce to canonical columns if present
    keep_base = [c for c in ["src_service","dst_service","n_events","n_rpc","n_messaging","n_links","n_errors"] if c in edges.columns]
    edges = edges.loc[:, list(dict.fromkeys(keep_base + [c for c in edges.columns if c not in {"src_service","dst_service"}]))]

    # Optional enrichments
    if args.features and args.features.exists():
        feats = pd.read_parquet(args.features)
        keep = [c for c in ["src_service","dst_service","p_messaging","link_ratio","median_lag_ns","p_overlap","p_nonneg_lag"] if c in feats.columns]
        edges = edges.merge(feats[keep], on=["src_service","dst_service"], how="left")

    if args.pred and args.pred.exists():
        pred = pd.read_csv(args.pred)
        # tolerate column variations
        if "label" in pred.columns and "pred_label" not in pred.columns:
            pred["pred_label"] = pred["label"]
        if "score" in pred.columns and "pred_score" not in pred.columns:
            pred["pred_score"] = pred["score"]
        keep = [c for c in ["src_service","dst_service","pred_label","pred_score"] if c in pred.columns]
        edges = edges.merge(pred[keep], on=["src_service","dst_service"], how="left")

    # Nodes
    services = pd.Index(pd.unique(pd.concat([edges["src_service"], edges["dst_service"]]))).astype(str)
    indeg = edges.groupby("dst_service").size().rename("in_degree")
    outdeg = edges.groupby("src_service").size().rename("out_degree")
    deg = pd.concat([indeg, outdeg], axis=1).fillna(0).astype(int).reset_index().rename(columns={"index":"id"})
    deg = deg.rename(columns={"dst_service":"id","src_service":"id"}).set_index("id", drop=False)
    for s in services:
        if s not in deg.index:
            deg.loc[s, ["id","in_degree","out_degree"]] = [s, 0, 0]
    nodes = [{"id": s, "label": s, "in_degree": int(deg.loc[s,'in_degree']), "out_degree": int(deg.loc[s,'out_degree'])} for s in services.tolist()]

    # Edges payload
    ecols = set(edges.columns)
    out_edges = []
    for _, r in edges.iterrows():
        d = {"source": r["src_service"], "target": r["dst_service"]}
        counts = {}
        for k in ("n_events","n_rpc","n_messaging","n_links","n_errors"):
            if k in ecols and pd.notna(r.get(k)):
                try:
                    counts[k] = int(r[k])
                except Exception:
                    try: counts[k] = float(r[k])
                    except Exception: pass
        if counts:
            d["counts"] = counts
        feats = {}
        for k in ("p_messaging","link_ratio","median_lag_ns","p_overlap","p_nonneg_lag"):
            if k in ecols and pd.notna(r.get(k)):
                feats[k] = float(r[k])
        if feats:
            d["features"] = feats
        if "pred_label" in ecols or "pred_score" in ecols:
            pred = {}
            if "pred_label" in ecols and pd.notna(r.get("pred_label")):
                pred["label"] = str(r["pred_label"])
            if "pred_score" in ecols and pd.notna(r.get("pred_score")):
                pred["score"] = float(r["pred_score"])
            if pred:
                d["prediction"] = pred
        out_edges.append(d)

    graph = {
        "schema": "edgetyper-graph@v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "nodes": nodes,
        "edges": out_edges,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(graph, indent=2))
    print(f"[export_graph_json] wrote {args.out}  (nodes={len(nodes)}, edges={len(out_edges)})")

if __name__ == "__main__":
    main()
