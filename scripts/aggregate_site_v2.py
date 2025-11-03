#!/usr/bin/env python3
# Aggregate replicas: build Availability section + aggregate artifacts
# Safe if live data is missing; reads per‑replica availability_* and availability.json
from __future__ import annotations
import argparse, json, os, re, time, glob, shutil
from pathlib import Path
import pandas as pd

def _fmt3(x):
    try:
        return f"{float(x):.3f}"
    except Exception:
        return "—"

def _read_csv(p: Path, expect: list[str] | None = None) -> pd.DataFrame | None:
    if not p or not p.exists() or p.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(p)
    except Exception:
        return None
    if expect:
        for c in expect:
            if c not in df.columns:
                df[c] = None
        df = df.loc[:, expect]
    return df

def discover_replicas(root_candidates: list[Path]) -> list[Path]:
    out = []
    for root in root_candidates:
        if not root.exists():
            continue
        # Common layouts from actions/download-artifact
        out.extend(sorted([p for p in root.glob("replicate-*") if p.is_dir()]))
        # Fallback recursive search (depth <= 3)
        out.extend(sorted([p for p in root.glob("**/replicate-*") if p.is_dir()]))
    # Deduplicate while preserving order
    seen = set(); uniq = []
    for p in out:
        if p.resolve() not in seen:
            uniq.append(p)
            seen.add(p.resolve())
    return uniq

def build_aggregate(replicas: list[Path], site: Path) -> None:
    site.mkdir(parents=True, exist_ok=True)
    data_dir = site / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    typed_all = []
    block_all = []
    live_all  = []

    for rdir in replicas:
        rid = rdir.name
        t = _read_csv(rdir / "availability_typed.csv",    ["entrypoint","p_fail","R_model"])
        b = _read_csv(rdir / "availability_block.csv",    ["entrypoint","p_fail","R_model"])
        l = _read_csv(rdir / "live_availability.csv",     ["entrypoint","p_fail","R_live"])
        if t is not None:
            t = t.copy()
            t["replica"] = rid
            typed_all.append(t)
        if b is not None:
            b = b.copy()
            b["replica"] = rid
            block_all.append(b)
        if l is not None:
            l = l.copy()
            l["replica"] = rid
            live_all.append(l)

    typed_all = pd.concat(typed_all, ignore_index=True) if typed_all else pd.DataFrame(columns=["entrypoint","p_fail","R_model","replica"])
    block_all = pd.concat(block_all, ignore_index=True) if block_all else pd.DataFrame(columns=["entrypoint","p_fail","R_model","replica"])
    live_all  = pd.concat(live_all,  ignore_index=True) if live_all  else pd.DataFrame(columns=["entrypoint","p_fail","R_live","replica"])

    # Canonicalize types
    for df, col in ((typed_all, "R_model"), (block_all, "R_model")):
        if not df.empty:
            df["p_fail"]   = pd.to_numeric(df["p_fail"], errors="coerce")
            df[col]        = pd.to_numeric(df[col],      errors="coerce")
            df["entrypoint"] = df["entrypoint"].astype(str)
    if not live_all.empty:
        live_all["p_fail"] = pd.to_numeric(live_all["p_fail"], errors="coerce")
        live_all["R_live"] = pd.to_numeric(live_all["R_live"], errors="coerce")
        live_all["entrypoint"] = live_all["entrypoint"].astype(str)

    # Save per‑replica long tables
    if not typed_all.empty:
        typed_all.to_csv(data_dir / "availability_typed_all.csv", index=False)
    if not block_all.empty:
        block_all.to_csv(data_dir / "availability_block_all.csv", index=False)
    if not live_all.empty:
        live_all.to_csv(data_dir / "availability_live_all.csv", index=False)

    # Aggregate by entrypoint × p_fail
    def agg_mean_std(df: pd.DataFrame, value_col: str, prefix: str) -> pd.DataFrame:
        if df.empty: 
            return pd.DataFrame(columns=["entrypoint","p_fail", f"{prefix}_mean", f"{prefix}_std", f"n_{prefix}"])
        g = (df.groupby(["entrypoint","p_fail"], as_index=False)
                .agg(**{f"{prefix}_mean": (value_col, "mean"),
                        f"{prefix}_std":  (value_col, "std"),
                        f"n_{prefix}":    (value_col, "count")}))
        return g

    agg_t = agg_mean_std(typed_all, "R_model", "typed")
    agg_b = agg_mean_std(block_all, "R_model", "block")
    agg_l = None
    if not live_all.empty:
        agg_l = (live_all.groupby(["entrypoint","p_fail"], as_index=False)
                   .agg(R_live_mean=("R_live","mean"),
                        n_live=("R_live","count")))

    # MAE vs live (computed only where both exist for same replica/entrypoint/p_fail)
    mae_t = pd.DataFrame(columns=["entrypoint","p_fail","MAE_typed_mean"])
    mae_b = pd.DataFrame(columns=["entrypoint","p_fail","MAE_block_mean"])
    if not live_all.empty and not typed_all.empty:
        jt = (typed_all.merge(live_all, on=["replica","entrypoint","p_fail"], how="inner")
                       .assign(err=lambda d: (d["R_model"] - d["R_live"]).abs()))
        if not jt.empty:
            mae_t = jt.groupby(["entrypoint","p_fail"], as_index=False).agg(MAE_typed_mean=("err","mean"))
    if not live_all.empty and not block_all.empty:
        jb = (block_all.merge(live_all, on=["replica","entrypoint","p_fail"], how="inner")
                       .assign(err=lambda d: (d["R_model"] - d["R_live"]).abs()))
        if not jb.empty:
            mae_b = jb.groupby(["entrypoint","p_fail"], as_index=False).agg(MAE_block_mean=("err","mean"))

    # Join all aggregates
    agg = None
    for piece in (agg_t, agg_b, agg_l, mae_t, mae_b):
        if piece is None or piece.empty: 
            continue
        agg = piece if agg is None else agg.merge(piece, on=["entrypoint","p_fail"], how="outer")
    if agg is None:
        agg = pd.DataFrame(columns=[
            "entrypoint","p_fail","typed_mean","typed_std","n_typed",
            "block_mean","block_std","n_block","R_live_mean","n_live",
            "MAE_typed_mean","MAE_block_mean"
        ])

    # Persist aggregate CSV + JSON
    agg_out_csv = data_dir / "availability_aggregate.csv"
    agg.sort_values(["entrypoint","p_fail"], inplace=True, ignore_index=True)
    agg.to_csv(agg_out_csv, index=False)

    agg_json = {
        "schema": "edgetyper-availability-aggregate@v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "replicas": [p.name for p in replicas],
        "by_entrypoint": agg.to_dict(orient="records"),
    }
    (data_dir / "availability_aggregate.json").write_text(json.dumps(agg_json, indent=2))

    # Also move existing aggregate outputs (if any) into data/ for linking
    for fname in ("aggregate_summary.json", "replicas_summary.csv"):
        src = site / fname
        if src.exists():
            shutil.copyfile(src, data_dir / fname)

    # Build/overwrite index.html with an Availability section
    rows = []
    for r in agg.itertuples(index=False):
        rows.append(
            "<tr>"
            f"<td>{r.entrypoint}</td>"
            f"<td>{_fmt3(r.p_fail)}</td>"
            f"<td>{_fmt3(getattr(r,'typed_mean',float('nan')))}"
            f" ± { _fmt3(getattr(r,'typed_std',float('nan')))}"
            f" <small>(n={int(getattr(r,'n_typed',0) or 0)})</small></td>"
            f"<td>{_fmt3(getattr(r,'block_mean',float('nan')))}"
            f" ± { _fmt3(getattr(r,'block_std',float('nan')))}"
            f" <small>(n={int(getattr(r,'n_block',0) or 0)})</small></td>"
            f"<td>{_fmt3(getattr(r,'R_live_mean',float('nan')))}"
            f" <small>(n={int(getattr(r,'n_live',0) or 0)})</small></td>"
            f"<td>{_fmt3(getattr(r,'MAE_typed_mean',float('nan')))}</td>"
            f"<td>{_fmt3(getattr(r,'MAE_block_mean',float('nan')))}</td>"
            "</tr>"
        )
    availability_table = (
        "<h2>Availability (aggregate)</h2>"
        "<p>Means and standard deviations across replicas. Live/MAE columns appear when Locust data is present.</p>"
        "<table><thead><tr>"
        "<th>Entrypoint</th><th>p_fail</th>"
        "<th>Typed R_model (mean ± std)</th>"
        "<th>All‑blocking R_model (mean ± std)</th>"
        "<th>Live R_live (mean)</th>"
        "<th>MAE typed</th><th>MAE all‑block</th>"
        "</tr></thead><tbody>"
        + "".join(rows) +
        "</tbody></table>"
    )

    # Downloads
    dl_items = []
    for fname in ("aggregate_summary.json","replicas_summary.csv",
                  "availability_aggregate.json","availability_aggregate.csv",
                  "availability_typed_all.csv","availability_block_all.csv","availability_live_all.csv"):
        p = data_dir / fname
        if p.exists():
            dl_items.append(f"<li><a href='data/{fname}' download>{fname}</a></li>")
    downloads_html = "<h2>Downloads</h2><ul>" + "".join(dl_items) + "</ul>"

    html = (
        "<!doctype html><html><head><meta charset='utf-8'/>"
        "<title>EdgeTyper — Aggregate</title>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'/>"
        "<style>body{font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;line-height:1.4;padding:24px}"
        "table{border-collapse:collapse;width:100%;max-width:100%}th,td{border:1px solid #ddd;padding:6px 8px}"
        "th{background:#fafafa;text-align:left}</style>"
        "</head><body>"
        "<h1>EdgeTyper — Aggregate</h1>"
        + downloads_html +
        availability_table +
        "</body></html>"
    )
    (site / "index.html").write_text(html)
    print(f"[aggregate_site_v2] wrote {site/'index.html'} and {agg_out_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicas-root", default="replicas", help="Directory containing subfolders replicate-*")
    ap.add_argument("--site", default="site", help="Output site directory (overwrites index.html)")
    args = ap.parse_args()

    candidates = [Path(args.replicas_root), Path("replicas"), Path("artifacts"), Path("runs")]
    replicas = discover_replicas(candidates)
    if not replicas:
        print("::warning ::No replicas discovered; aggregate page will have no availability content.")
    build_aggregate(replicas, Path(args.site))

if __name__ == "__main__":
    main()
