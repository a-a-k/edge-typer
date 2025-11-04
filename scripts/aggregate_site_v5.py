#!/usr/bin/env python3
"""
EdgeTyper — Aggregate (v5, robust & safe links)
 - Discovers replica artifacts even when nested (e.g. runs/<id>/…)
 - If availability CSVs are missing or header-only, recomputes them from:
     (a) edges.parquet + pred_ours.csv, else
     (b) graph.json → rebuild edges/preds, then recompute
 - Derives entrypoints.txt (txt → csv → edges → graph.json → fallback ['frontend'])
 - Writes aggregate CSV/JSON and overwrites site/index.html (Availability table)
 - **Links only existing files** (no 404s)
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path
from typing import Iterable, Optional, Tuple, List
import pandas as pd

DEFAULT_PFAIL_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]

# ------------------- small utils -------------------
def _fmt3(x): 
    try: return f"{float(x):.3f}"
    except Exception: return "—"

def _read_csv(p: Path, expect: list[str] | None = None) -> Optional[pd.DataFrame]:
    if not p or not p.exists() or p.stat().st_size == 0: return None
    try: df = pd.read_csv(p)
    except Exception: return None
    if expect:
        for c in expect:
            if c not in df.columns: df[c] = None
        df = df.loc[:, expect]
    return df

def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ln in lines: f.write(f"{ln}\n")

def _ensure_pkg() -> None:
    """Ensure edgetyper CLI & pyarrow are present."""
    try:
        subprocess.run(["edgetyper","--help"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        subprocess.run([sys.executable,"-m","pip","install","-e","."], check=True)

# ------------------- discovery helpers -------------------
def _find_first(root: Path, name: str) -> Optional[Path]:
    """Find first file `name` within depth<=3; prefer shallower paths."""
    for depth in range(0, 4):
        for p in root.rglob(name):
            rel_depth = len(p.relative_to(root).parts) - 1
            if p.is_file() and rel_depth <= depth:
                return p
    return None

def _copy_if_nested(replica_dir: Path, filename: str) -> Optional[Path]:
    """If only nested, copy up to replica root."""
    root = replica_dir / filename
    if root.exists(): return root
    found = _find_first(replica_dir, filename)
    if found and found != root:
        root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(found, root)
        return root
    return None

def discover_replicas(candidates: list[Path]) -> list[Path]:
    out = []
    for root in candidates:
        if not root.exists(): continue
        out.extend(sorted([p for p in root.glob("replicate-*") if p.is_dir()]))
        out.extend(sorted([p for p in root.glob("**/replicate-*") if p.is_dir()]))
    seen, uniq = set(), []
    for p in out:
        rp = p.resolve()
        if rp not in seen: uniq.append(p); seen.add(rp)
    return uniq

# ------------------- rebuild from graph.json -------------------
def _nodes_from_graph(obj: dict) -> list[str]:
    nodes = []
    for n in obj.get("nodes", []) or []:
        nid = str(n.get("id","")).strip()
        if nid: nodes.append(nid)
    if not nodes:
        # fallback: from edges endpoints
        nodes = sorted(set([str(e.get("source","")).strip() for e in obj.get("edges",[])] +
                           [str(e.get("target","")).strip() for e in obj.get("edges",[])]))
    return [n for n in nodes if n]

def _rebuild_from_graph_json(replica_dir: Path) -> bool:
    g = (replica_dir / "graph.json")
    if not g.exists(): g = _find_first(replica_dir, "graph.json")
    if not g or not g.exists(): return False
    try: obj = json.loads(g.read_text(encoding="utf-8"))
    except Exception: return False
    edges, preds = [], []
    for e in obj.get("edges", []):
        s, t = str(e.get("source","") or ""), str(e.get("target","") or "")
        if not s or not t: continue
        counts = (e.get("counts") or {})
        edges.append({
            "src_service": s, "dst_service": t,
            "n_events": counts.get("n_events", 0),
            "n_rpc": counts.get("n_rpc", 0),
            "n_messaging": counts.get("n_messaging", 0),
            "n_links": counts.get("n_links", 0),
            "n_errors": counts.get("n_errors", 0),
        })
        pred = (e.get("prediction") or {})
        if "label" in pred:
            preds.append({
                "src_service": s, "dst_service": t,
                "pred_label": pred.get("label"),
                "pred_score": pred.get("score", None),
            })
    if not edges: return False
    _ensure_pkg()
    pd.DataFrame(edges).to_parquet(replica_dir / "edges.parquet", index=False)
    if preds:
        pd.DataFrame(preds).to_csv(replica_dir / "pred_ours.csv", index=False)
    elif not (replica_dir / "pred_ours.csv").exists():
        pd.DataFrame([{"src_service":r["src_service"],"dst_service":r["dst_service"],"pred_label":"sync"} for r in edges]) \
          .to_csv(replica_dir / "pred_ours.csv", index=False)
    # also drop a nodes.txt to help entrypoint guessing later
    nodes = _nodes_from_graph(obj)
    if nodes:
        _write_lines(replica_dir / "nodes.txt", nodes)
    return True

# ------------------- entrypoints logic -------------------
def _guess_entrypoints_from_edges(edges_parquet: Path) -> list[str]:
    _ensure_pkg()
    try: df = pd.read_parquet(edges_parquet)
    except Exception: return []
    if not {"src_service","dst_service"}.issubset(df.columns): return []
    src, dst = df["src_service"].astype(str), df["dst_service"].astype(str)
    indeg = dst.value_counts().rename("in_degree")
    services = pd.Index(pd.unique(pd.concat([src, dst])))
    indeg = indeg.reindex(services).fillna(0).astype(int)
    eps = indeg[indeg==0].index.tolist()
    if not eps:
        hl = [s for s in services if any(k in s.lower() for k in ("front","api","edge","web","gateway"))]
        eps = hl[:3] if hl else []
    if not eps:
        eps = src.value_counts().sort_values(ascending=False).index.tolist()[:3]
    return sorted(set(map(str, eps)))

def _load_entrypoints_text(p: Path) -> list[str]:
    if p.exists() and p.stat().st_size>0:
        return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return []

def _entrypoints_from_csv(p: Path) -> list[str]:
    try:
        if p.exists():
            df = pd.read_csv(p)
            col = "entrypoint" if "entrypoint" in df.columns else df.columns[0]
            return [str(x) for x in df[col].dropna().astype(str).tolist()]
    except Exception:
        pass
    return []

def _entrypoints_from_nodes_txt(p: Path) -> list[str]:
    return _load_entrypoints_text(p)

def ensure_entrypoints(replica_dir: Path) -> Path:
    txt = replica_dir / "entrypoints.txt"
    if not txt.exists():
        src = _find_first(replica_dir, "entrypoints.txt")
        if src: shutil.copyfile(src, txt)
    if _load_entrypoints_text(txt): return txt
    csv = replica_dir / "entrypoints.csv"
    if not csv.exists(): _copy_if_nested(replica_dir, "entrypoints.csv")
    eps = _entrypoints_from_csv(csv)
    if eps:
        _write_lines(txt, eps); return txt
    # from edges
    edges = replica_dir / "edges.parquet"
    if not edges.exists(): _copy_if_nested(replica_dir, "edges.parquet")
    if edges.exists():
        eps = _guess_entrypoints_from_edges(edges)
        if eps: _write_lines(txt, eps); return txt
    # from graph nodes (if we rebuilt)
    nodes_txt = replica_dir / "nodes.txt"
    if not nodes_txt.exists(): _copy_if_nested(replica_dir, "nodes.txt")
    eps = _entrypoints_from_nodes_txt(nodes_txt)
    if eps:
        _write_lines(txt, eps[:3]); return txt
    # fallback
    _write_lines(txt, ["frontend"]); return txt

def _header_only_or_empty(p: Path) -> bool:
    if not p.exists() or p.stat().st_size==0: return True
    try:
        lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return len(lines) <= 1
    except Exception:
        return True

# ------------------- availability (self-heal) -------------------
def ensure_availability(replica_dir: Path, pgrid: list[float]) -> Tuple[Optional[Path], Optional[Path]]:
    typed_p = replica_dir / "availability_typed.csv"
    block_p = replica_dir / "availability_block.csv"
    _copy_if_nested(replica_dir, "availability_typed.csv")
    _copy_if_nested(replica_dir, "availability_block.csv")
    if (typed_p.exists() and not _header_only_or_empty(typed_p)) and \
       (block_p.exists() and not _header_only_or_empty(block_p)):
        return typed_p, block_p
    edges = replica_dir / "edges.parquet"
    pred  = replica_dir / "pred_ours.csv"
    if not edges.exists(): _copy_if_nested(replica_dir, "edges.parquet")
    if not pred.exists():  _copy_if_nested(replica_dir, "pred_ours.csv")
    if (not edges.exists()) or (not pred.exists()):
        if not _rebuild_from_graph_json(replica_dir):
            print(f"::warning ::[{replica_dir.name}] missing edges/pred; cannot recompute")
            return typed_p if typed_p.exists() else None, block_p if block_p.exists() else None
    _ensure_pkg()
    ep_txt = ensure_entrypoints(replica_dir)
    pf_args: list[str] = []
    for p in pgrid:
        try: pf_args += ["--p-fail", f"{float(p)}"]
        except Exception: pass
    cmd_t = ["edgetyper","resilience","--edges",str(edges),"--pred",str(pred),
             "--entrypoints",str(ep_txt),"--out",str(typed_p)] + pf_args
    cmd_b = ["edgetyper","resilience","--edges",str(edges),"--pred",str(pred),
             "--entrypoints",str(ep_txt),"--assume-all-blocking","--out",str(block_p)] + pf_args
    try:
        print(f"[{replica_dir.name}] recomputing typed…"); subprocess.run(cmd_t, check=True)
        print(f"[{replica_dir.name}] recomputing all-blocking…"); subprocess.run(cmd_b, check=True)
    except subprocess.CalledProcessError as e:
        print(f"::warning ::[{replica_dir.name}] resilience failed: {e}")
    # final guard: if still empty, try a last-ditch entrypoint = first node in graph.json (if any)
    if _header_only_or_empty(typed_p) or _header_only_or_empty(block_p):
        g = _find_first(replica_dir, "graph.json")
        if g and g.exists():
            try:
                obj = json.loads(g.read_text(encoding="utf-8"))
                nodes = _nodes_from_graph(obj)
                if nodes:
                    _write_lines(ep_txt, [nodes[0]])
                    print(f"[{replica_dir.name}] retrying with entrypoint={nodes[0]}")
                    subprocess.run(cmd_t, check=True)
                    subprocess.run(cmd_b, check=True)
            except Exception:
                pass
    return typed_p if typed_p.exists() else None, block_p if block_p.exists() else None

# ------------------- aggregate & HTML -------------------
def build_aggregate(replicas: list[Path], site: Path) -> None:
    site.mkdir(parents=True, exist_ok=True)
    data = site / "data"; data.mkdir(parents=True, exist_ok=True)
    typed_all, block_all, live_all = [], [], []
    grid_env = os.environ.get("P_FAIL_GRID","")
    pgrid = [float(x) for x in grid_env.split()] if grid_env else DEFAULT_PFAIL_GRID
    for rdir in replicas:
        rid = rdir.name
        ensure_availability(rdir, pgrid)
        t = _read_csv(rdir / "availability_typed.csv", ["entrypoint","p_fail","R_model"])
        b = _read_csv(rdir / "availability_block.csv", ["entrypoint","p_fail","R_model"])
        l = _read_csv(rdir / "live_availability.csv",  ["entrypoint","p_fail","R_live"])
        if t is not None and not t.empty: typed_all.append(t.assign(replica=rid))
        if b is not None and not b.empty: block_all.append(b.assign(replica=rid))
        if l is not None and not l.empty: live_all.append(l.assign(replica=rid))
    T = pd.concat(typed_all, ignore_index=True) if typed_all else pd.DataFrame(columns=["entrypoint","p_fail","R_model","replica"])
    B = pd.concat(block_all, ignore_index=True) if block_all else pd.DataFrame(columns=["entrypoint","p_fail","R_model","replica"])
    L = pd.concat(live_all,  ignore_index=True) if live_all  else pd.DataFrame(columns=["entrypoint","p_fail","R_live","replica"])
    for df, col in ((T,"R_model"), (B,"R_model")):
        if not df.empty:
            df["entrypoint"]=df["entrypoint"].astype(str)
            df["p_fail"]=pd.to_numeric(df["p_fail"], errors="coerce")
            df[col]=pd.to_numeric(df[col], errors="coerce")
    if not L.empty:
        L["entrypoint"]=L["entrypoint"].astype(str)
        L["p_fail"]=pd.to_numeric(L["p_fail"], errors="coerce")
        L["R_live"]=pd.to_numeric(L["R_live"], errors="coerce")
    # persist long tables (only if non-empty)
    if not T.empty: T.to_csv(data/"availability_typed_all.csv", index=False)
    if not B.empty: B.to_csv(data/"availability_block_all.csv", index=False)
    if not L.empty: L.to_csv(data/"availability_live_all.csv", index=False)
    # aggregates
    def agg(df: pd.DataFrame, val: str, prefix: str) -> pd.DataFrame:
        if df.empty: 
            return pd.DataFrame(columns=["entrypoint","p_fail",f"{prefix}_mean",f"{prefix}_std",f"n_{prefix}"])
        return (df.groupby(["entrypoint","p_fail"], as_index=False)
                 .agg(**{f"{prefix}_mean":(val,"mean"), f"{prefix}_std":(val,"std"), f"n_{prefix}":(val,"count")}))
    agg_t = agg(T, "R_model", "typed")
    agg_b = agg(B, "R_model", "block")
    agg_l = (L.groupby(["entrypoint","p_fail"], as_index=False)
               .agg(R_live_mean=("R_live","mean"), n_live=("R_live","count"))) if not L.empty else None
    mae_t = pd.DataFrame(columns=["entrypoint","p_fail","MAE_typed_mean"])
    mae_b = pd.DataFrame(columns=["entrypoint","p_fail","MAE_block_mean"])
    if not L.empty and not T.empty:
        jt = (T.merge(L, on=["replica","entrypoint","p_fail"], how="inner")
                .assign(err=lambda d:(d["R_model"]-d["R_live"]).abs()))
        if not jt.empty: mae_t = jt.groupby(["entrypoint","p_fail"], as_index=False).agg(MAE_typed_mean=("err","mean"))
    if not L.empty and not B.empty:
        jb = (B.merge(L, on=["replica","entrypoint","p_fail"], how="inner")
                .assign(err=lambda d:(d["R_model"]-d["R_live"]).abs()))
        if not jb.empty: mae_b = jb.groupby(["entrypoint","p_fail"], as_index=False).agg(MAE_block_mean=("err","mean"))
    agg_all = None
    for piece in (agg_t, agg_b, agg_l, mae_t, mae_b):
        if piece is None or piece.empty: continue
        agg_all = piece if agg_all is None else agg_all.merge(piece, on=["entrypoint","p_fail"], how="outer")
    if agg_all is None:
        agg_all = pd.DataFrame(columns=["entrypoint","p_fail","typed_mean","typed_std","n_typed","block_mean","block_std","n_block","R_live_mean","n_live","MAE_typed_mean","MAE_block_mean"])
    agg_all.sort_values(["entrypoint","p_fail"], inplace=True, ignore_index=True)
    # write aggregate artifacts
    (data/"availability_aggregate.csv").write_text(agg_all.to_csv(index=False), encoding="utf-8")
    (data/"availability_aggregate.json").write_text(json.dumps({
        "schema":"edgetyper-availability-aggregate@v1",
        "generated_at":time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "replicas":[p.name for p in replicas],
        "by_entrypoint":agg_all.to_dict(orient="records"),
    }, indent=2), encoding="utf-8")
    # carry over existing top files into data/ if present
    for fname in ("aggregate_summary.json","replicas_summary.csv"):
        src = site/fname
        if src.exists(): shutil.copyfile(src, data/fname)
    # downloads (link only if file exists)
    def _li(name: str) -> str:
        p = data/name
        return f"<li><a href='data/{name}'>{name}</a></li>" if p.exists() else ""
    downloads = (
        "<h2>Downloads</h2><ul>"
        + _li("aggregate_summary.json")
        + _li("replicas_summary.csv")
        + _li("availability_aggregate.json")
        + _li("availability_aggregate.csv")
        + _li("availability_typed_all.csv")
        + _li("availability_block_all.csv")
        + _li("availability_live_all.csv")
        + "</ul>"
    )
    # table
    rows = []
    for r in agg_all.itertuples(index=False):
        rows.append(
            "<tr>"
            f"<td>{r.entrypoint}</td>"
            f"<td>{_fmt3(r.p_fail)}</td>"
            f"<td>{_fmt3(getattr(r,'typed_mean',float('nan')))} ± {_fmt3(getattr(r,'typed_std',float('nan')))} (n={int(getattr(r,'n_typed',0) or 0)})</td>"
            f"<td>{_fmt3(getattr(r,'block_mean',float('nan')))} ± {_fmt3(getattr(r,'block_std',float('nan')))} (n={int(getattr(r,'n_block',0) or 0)})</td>"
            f"<td>{_fmt3(getattr(r,'R_live_mean',float('nan')))} (n={int(getattr(r,'n_live',0) or 0)})</td>"
            f"<td>{_fmt3(getattr(r,'MAE_typed_mean',float('nan')))}</td>"
            f"<td>{_fmt3(getattr(r,'MAE_block_mean',float('nan')))}</td>"
            "</tr>"
        )
    html = (
        "<!doctype html><meta charset='utf-8'>"
        "<title>EdgeTyper — Aggregate</title>"
        "<h1>EdgeTyper — Aggregate</h1>"
        + downloads
        + "<h2>Availability (aggregate)</h2>"
        + "<p>Means and standard deviations across replicas. Live/MAE columns appear when Locust data is present.</p>"
        + "<table border='1' cellspacing='0' cellpadding='4'><thead><tr>"
          "<th>Entrypoint</th><th>p_fail</th>"
          "<th>Typed R_model (mean ± std)</th>"
          "<th>All-blocking R_model (mean ± std)</th>"
          "<th>Live R_live (mean)</th>"
          "<th>MAE typed</th><th>MAE all-block</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
    (site/"index.html").write_text(html, encoding="utf-8")
    print(f"[aggregate_site_v5] wrote {site/'index.html'} and {data/'availability_aggregate.csv'}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicas-root", default="replicas")
    ap.add_argument("--site", default="site")
    args = ap.parse_args()
    reps = discover_replicas([Path(args.replicas-root) if hasattr(args,'replicas-root') else Path(args.replicas_root),
                              Path("replicas"), Path("artifacts"), Path("runs")])
    if not reps: print("::warning ::no replicas discovered; page will be empty")
    build_aggregate(reps, Path(args.site))

if __name__ == "__main__":
    main()
