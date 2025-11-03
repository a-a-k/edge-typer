#!/usr/bin/env python3
# Aggregate replicas: build Availability section + aggregate artifacts.
# Self-healing: if a replica's availability CSVs are missing or header-only,
# recompute them from edges.parquet + pred_ours.csv with an explicit entrypoint
# set and a fixed p_fail grid—so the aggregate page is never empty.
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd

# ---------------------------- helpers ---------------------------------

DEFAULT_PFAIL_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]

def _fmt3(x):
    try:
        return f"{float(x):.3f}"
    except Exception:
        return "—"

def _read_csv(p: Path, expect: list[str] | None = None) -> Optional[pd.DataFrame]:
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

def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ln in lines:
            f.write(f"{ln}\n")

def _ensure_pkg_installed() -> None:
    """Install the package (editable) if the edgetyper CLI is missing."""
    try:
        subprocess.run(["edgetyper", "--help"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)

def discover_replicas(root_candidates: list[Path]) -> list[Path]:
    out = []
    for root in root_candidates:
        if not root.exists():
            continue
        out.extend(sorted([p for p in root.glob("replicate-*") if p.is_dir()]))
        out.extend(sorted([p for p in root.glob("**/replicate-*") if p.is_dir()]))
    seen = set(); uniq = []
    for p in out:
        rp = p.resolve()
        if rp not in seen:
            uniq.append(p)
            seen.add(rp)
    return uniq

# --------------------- entrypoint discovery ----------------------------

def _guess_entrypoints_from_edges(edges_parquet: Path) -> list[str]:
    df = pd.read_parquet(edges_parquet)
    if not {"src_service", "dst_service"}.issubset(df.columns):
        return []
    src = df["src_service"].astype(str)
    dst = df["dst_service"].astype(str)
    indeg = dst.value_counts().rename("in_degree")
    services = pd.Index(pd.unique(pd.concat([src, dst])))
    indeg = indeg.reindex(services).fillna(0).astype(int)
    candidates = indeg[indeg == 0].index.tolist()
    if not candidates:
        # fallback heuristics
        hl = [s for s in services if any(k in s.lower() for k in ("front", "api", "edge", "web", "gateway"))]
        candidates = hl[:3] if hl else []
    if not candidates:
        outdeg = src.value_counts()
        candidates = outdeg.sort_values(ascending=False).index.tolist()[:3]
    return sorted(set(map(str, candidates)))

def _load_entrypoints_text(ep_txt: Path) -> list[str]:
    if ep_txt.exists() and ep_txt.stat().st_size > 0:
        return [ln.strip() for ln in ep_txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return []

def _entrypoints_from_csv(ep_csv: Path) -> list[str]:
    try:
        if ep_csv.exists():
            df = pd.read_csv(ep_csv)
            if "entrypoint" in df.columns:
                return [str(x) for x in df["entrypoint"].dropna().astype(str).tolist()]
    except Exception:
        pass
    return []

def ensure_entrypoints(replica_dir: Path) -> Path:
    """
    Ensure entrypoints.txt exists in replica_dir; attempt, in order:
    1) entrypoints.txt
    2) entrypoints.csv (column 'entrypoint')
    3) derive from edges.parquet
    """
    ep_txt = replica_dir / "entrypoints.txt"
    existing = _load_entrypoints_text(ep_txt)
    if existing:
        return ep_txt
    ep_csv = replica_dir / "entrypoints.csv"
    csv_eps = _entrypoints_from_csv(ep_csv)
    if csv_eps:
        _write_lines(ep_txt, csv_eps)
        return ep_txt
    edges = replica_dir / "edges.parquet"
    if edges.exists():
        guessed = _guess_entrypoints_from_edges(edges)
        if guessed:
            _write_lines(ep_txt, guessed)
            return ep_txt
    # last resort: write a placeholder to avoid CLI failure
    _write_lines(ep_txt, ["frontend"])
    return ep_txt

# ------------------ resilience self-healing ----------------------------

def _header_only_or_empty(p: Path) -> bool:
    if not p.exists() or p.stat().st_size == 0:
        return True
    try:
        with p.open("r", encoding="utf-8") as f:
            lines = [ln for ln in f.readlines() if ln.strip()]
        # header only if <=1 useful lines
        return len(lines) <= 1
    except Exception:
        return True

def ensure_availability(replica_dir: Path, p_fail_grid: Optional[list[float]] = None) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Ensure availability_typed.csv and availability_block.csv exist and are non-empty.
    Recompute them in-place if needed using the edgetyper CLI.
    """
    p_fail_grid = p_fail_grid or DEFAULT_PFAIL_GRID

    typed_p  = replica_dir / "availability_typed.csv"
    block_p  = replica_dir / "availability_block.csv"
    edges_p  = replica_dir / "edges.parquet"
    pred_p   = replica_dir / "pred_ours.csv"

    needs_fix = _header_only_or_empty(typed_p) or _header_only_or_empty(block_p)
    if not needs_fix:
        return typed_p if typed_p.exists() else None, block_p if block_p.exists() else None

    if not edges_p.exists() or not pred_p.exists():
        print(f"::warning ::[{replica_dir.name}] Missing edges.parquet or pred_ours.csv; cannot recompute availability.")
        return typed_p if typed_p.exists() else None, block_p if block_p.exists() else None

    _ensure_pkg_installed()
    ep_txt = ensure_entrypoints(replica_dir)

    # Build p-fail args
    pf_args = []
    for p in p_fail_grid:
        try:
            pf_args += ["--p-fail", f"{float(p)}"]
        except Exception:
            continue

    # Recompute typed
    cmd_typed = ["edgetyper", "resilience",
                 "--edges", str(edges_p),
                 "--pred",  str(pred_p),
                 "--entrypoints", str(ep_txt),
                 "--out",   str(typed_p)] + pf_args
    # Recompute all-blocking
    cmd_block = ["edgetyper", "resilience",
                 "--edges", str(edges_p),
                 "--pred",  str(pred_p),
                 "--entrypoints", str(ep_txt),
                 "--assume-all-blocking",
                 "--out",   str(block_p)] + pf_args

    try:
        print(f"[{replica_dir.name}] Recomputing typed availability…")
        subprocess.run(cmd_typed, check=True)
        print(f"[{replica_dir.name}] Recomputing all-blocking availability…")
        subprocess.run(cmd_block, check=True)
    except subprocess.CalledProcessError as e:
        print(f"::warning ::[{replica_dir.name}] edgetyper resilience failed: {e}")

    return typed_p if typed_p.exists() else None, block_p if block_p.exists() else None

# ----------------------- aggregation / HTML ----------------------------

def build_aggregate(replicas: list[Path], site: Path) -> None:
    site.mkdir(parents=True, exist_ok=True)
    data_dir = site / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    typed_all = []
    block_all = []
    live_all  = []

    # Ensure each replica has non-empty availability, then ingest
    grid_env = os.environ.get("P_FAIL_GRID", "")
    pgrid = [float(x) for x in grid_env.split() if x] if grid_env else DEFAULT_PFAIL_GRID

    for rdir in replicas:
        rid = rdir.name
        ensure_availability(rdir, pgrid)  # self-heal if needed

        t = _read_csv(rdir / "availability_typed.csv",    ["entrypoint","p_fail","R_model"])
        b = _read_csv(rdir / "availability_block.csv",    ["entrypoint","p_fail","R_model"])
        l = _read_csv(rdir / "live_availability.csv",     ["entrypoint","p_fail","R_live"])
        if t is not None and not t.empty:
            t = t.copy(); t["replica"] = rid; typed_all.append(t)
        if b is not None and not b.empty:
            b = b.copy(); b["replica"] = rid; block_all.append(b)
        if l is not None and not l.empty:
            l = l.copy(); l["replica"] = rid; live_all.append(l)

    typed_all = pd.concat(typed_all, ignore_index=True) if typed_all else pd.DataFrame(columns=["entrypoint","p_fail","R_model","replica"])
    block_all = pd.concat(block_all, ignore_index=True) if block_all else pd.DataFrame(columns=["entrypoint","p_fail","R_model","replica"])
    live_all  = pd.concat(live_all,  ignore_index=True) if live_all  else pd.DataFrame(columns=["entrypoint","p_fail","R_live","replica"])

    # Canonicalize types
    for df, col in ((typed_all, "R_model"), (block_all, "R_model")):
        if not df.empty:
            df["p_fail"] = pd.to_numeric(df["p_fail"], errors="coerce")
            df[col]      = pd.to_numeric(df[col],      errors="coerce")
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

    # MAE vs live (only where both exist for same replica/entrypoint/p_fail)
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