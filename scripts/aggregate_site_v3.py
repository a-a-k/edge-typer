#!/usr/bin/env python3
"""
EdgeTyper — Aggregate (v3, self‑healing)

Build the Availability section and aggregate artifacts from downloaded replica
artifacts produced by the matrix workflow. Compared to v2, this version:
  • Discovers replica contents even when files are nested (e.g. runs/<id>/…).
  • Recomputes availability CSVs in‑place if missing or header‑only, using the
    edgetyper CLI and a derived entrypoint list.
  • Reconstructs edges/predictions from graph.json if parquet/CSV are absent.
  • Persists CSV+JSON aggregate artifacts and overwrites site/index.html.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple, List

import pandas as pd

# ---------------------------- constants / utils ----------------------------

DEFAULT_PFAIL_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]

def _fmt3(x) -> str:
    try:
        return f"{float(x):.3f}"
    except Exception:
        return "—"

def _read_csv(p: Path, expect: list[str] | None = None) -> Optional[pd.DataFrame]:
    """Read CSV if present; coerce to expected columns (adds missing as None)."""
    if not p or not p.exists() or p.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(p)
    except Exception:
        return None
    if expect:
        for col in expect:
            if col not in df.columns:
                df[col] = None
        df = df.loc[:, expect]
    return df

def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ln in lines:
            f.write(f"{ln}\n")

def _ensure_pkg_installed() -> None:
    """
    Ensure the edgetyper package (and its deps, incl. pyarrow) are available.
    This is needed for pd.read_parquet and the 'edgetyper' CLI.
    """
    try:
        subprocess.run(
            ["edgetyper", "--help"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)

# ------------------------------- discovery --------------------------------

def discover_replicas(root_candidates: list[Path]) -> list[Path]:
    """Return unique paths that look like replica roots: */replicate-*."""
    out: list[Path] = []
    for root in root_candidates:
        if not root.exists():
            continue
        out.extend(sorted([p for p in root.glob("replicate-*") if p.is_dir()]))
        out.extend(sorted([p for p in root.glob("**/replicate-*") if p.is_dir()]))
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in out:
        rp = p.resolve()
        if rp not in seen:
            uniq.append(p)
            seen.add(rp)
    return uniq

def _find_first(root: Path, name: str) -> Optional[Path]:
    """Find the first file named `name` at depth ≤ 3 under root."""
    # breadth-ish limited depth search
    maxdepth = 3
    stack = [(root, 0)]
    while stack:
        base, depth = stack.pop(0)
        cand = base / name
        if cand.exists():
            return cand
        if depth < maxdepth:
            for child in sorted(base.iterdir()):
                if child.is_dir():
                    stack.append((child, depth + 1))
    return None

def _copy_if_nested(replica_dir: Path, filename: str) -> Optional[Path]:
    """If `filename` exists only nested under replica_dir, copy it to the root."""
    root_path = replica_dir / filename
    if root_path.exists():
        return root_path
    found = _find_first(replica_dir, filename)
    if found and found != root_path:
        try:
            root_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(found, root_path)
            return root_path
        except Exception as e:
            print(f"::warning ::[{replica_dir.name}] cannot copy {filename} from {found}: {e}")
    return None

# --------------------------- reconstruction path --------------------------

def _rebuild_from_graph_json(replica_dir: Path) -> bool:
    """
    Rebuild edges.parquet and pred_ours.csv from graph.json (searched nested).
    """
    gpath = (replica_dir / "graph.json")
    if not gpath.exists():
        alt = _find_first(replica_dir, "graph.json")
        if alt:
            gpath = alt
    if not gpath or not gpath.exists():
        return False
    try:
        obj = json.loads(gpath.read_text(encoding="utf-8"))
    except Exception:
        return False

    edges, preds = [], []
    for e in obj.get("edges", []):
        s = str(e.get("source", "")); t = str(e.get("target", ""))
        if not s or not t:
            continue
        counts = e.get("counts", {}) or {}
        edges.append({
            "src_service": s, "dst_service": t,
            "n_events": counts.get("n_events", 0),
            "n_rpc": counts.get("n_rpc", 0),
            "n_messaging": counts.get("n_messaging", 0),
            "n_links": counts.get("n_links", 0),
            "n_errors": counts.get("n_errors", 0),
        })
        pred = e.get("prediction", {}) or {}
        if "label" in pred:
            preds.append({
                "src_service": s,
                "dst_service": t,
                "pred_label": pred.get("label"),
                "pred_score": pred.get("score", None),
            })
    if not edges:
        return False

    _ensure_pkg_installed()  # ensure pyarrow for parquet
    try:
        pd.DataFrame(edges).to_parquet(replica_dir / "edges.parquet", index=False)
        if preds:
            pd.DataFrame(preds).to_csv(replica_dir / "pred_ours.csv", index=False)
        elif not (replica_dir / "pred_ours.csv").exists():
            # conservative default if predictions missing in graph.json
            pd.DataFrame(
                [{"src_service": r["src_service"], "dst_service": r["dst_service"], "pred_label": "sync"}
                 for r in edges]
            ).to_csv(replica_dir / "pred_ours.csv", index=False)
        return True
    except Exception as e:
        print(f"::warning ::[{replica_dir.name}] Failed to rebuild from graph.json: {e}")
        return False

# ---------------------------- entrypoint logic ----------------------------

def _guess_entrypoints_from_edges(edges_parquet: Path) -> list[str]:
    """Heuristics: services with in-degree 0; otherwise 'front|api|edge|web|gateway' or top out-degree."""
    _ensure_pkg_installed()  # for pd.read_parquet
    try:
        df = pd.read_parquet(edges_parquet)
    except Exception:
        return []
    need = {"src_service", "dst_service"}
    if not need.issubset(df.columns):
        return []
    src = df["src_service"].astype(str)
    dst = df["dst_service"].astype(str)
    indeg = dst.value_counts().rename("in_degree")
    services = pd.Index(pd.unique(pd.concat([src, dst])))
    indeg = indeg.reindex(services).fillna(0).astype(int)
    candidates = indeg[indeg == 0].index.tolist()
    if not candidates:
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
    Ensure entrypoints.txt exists at replica root.
    Order of attempts:
      1) existing entrypoints.txt (root or nested → copied up)
      2) entrypoints.csv (column 'entrypoint') (root or nested → copied up)
      3) derive from edges.parquet using heuristics
      4) final fallback: ['frontend']
    """
    # 1) entrypoints.txt
    txt_root = replica_dir / "entrypoints.txt"
    if not txt_root.exists():
        src = _find_first(replica_dir, "entrypoints.txt")
        if src:
            try:
                shutil.copyfile(src, txt_root)
            except Exception:
                pass
    if _load_entrypoints_text(txt_root):
        return txt_root

    # 2) entrypoints.csv → entrypoints.txt
    csv_root = replica_dir / "entrypoints.csv"
    if not csv_root.exists():
        _copy_if_nested(replica_dir, "entrypoints.csv")
    csv_eps = _entrypoints_from_csv(csv_root)
    if csv_eps:
        _write_lines(txt_root, csv_eps)
        return txt_root

    # 3) derive from edges.parquet
    edges_root = replica_dir / "edges.parquet"
    if not edges_root.exists():
        _copy_if_nested(replica_dir, "edges.parquet")
    if edges_root.exists():
        guessed = _guess_entrypoints_from_edges(edges_root)
        if guessed:
            _write_lines(txt_root, guessed)
            return txt_root

    # 4) fallback
    _write_lines(txt_root, ["frontend"])
    return txt_root

# --------------------------- availability recompute -----------------------

def _header_only_or_empty(p: Path) -> bool:
    if not p.exists() or p.stat().st_size == 0:
        return True
    try:
        with p.open("r", encoding="utf-8") as f:
            lines = [ln for ln in f.readlines() if ln.strip()]
        return len(lines) <= 1
    except Exception:
        return True

def ensure_availability(replica_dir: Path,
                        p_fail_grid: Optional[list[float]] = None
                        ) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Ensure `availability_typed.csv` and `availability_block.csv` exist and are
    non‑empty at the replica root. Recompute in‑place if needed.
    """
    p_fail_grid = p_fail_grid or DEFAULT_PFAIL_GRID
    typed_p = replica_dir / "availability_typed.csv"
    block_p = replica_dir / "availability_block.csv"

    # Surface nested files if they already exist deeper
    _copy_if_nested(replica_dir, "availability_typed.csv")
    _copy_if_nested(replica_dir, "availability_block.csv")

    needs_fix = _header_only_or_empty(typed_p) or _header_only_or_empty(block_p)
    if not needs_fix:
        return (typed_p if typed_p.exists() else None,
                block_p if block_p.exists() else None)

    # Ensure edges/pred are available (copy from nested or rebuild from JSON)
    edges_p = replica_dir / "edges.parquet"
    pred_p = replica_dir / "pred_ours.csv"
    if not edges_p.exists():
        _copy_if_nested(replica_dir, "edges.parquet")
    if not pred_p.exists():
        _copy_if_nested(replica_dir, "pred_ours.csv")
    if (not edges_p.exists()) or (not pred_p.exists()):
        if not _rebuild_from_graph_json(replica_dir):
            print(f"::warning ::[{replica_dir.name}] Missing edges.parquet or pred_ours.csv; cannot recompute availability.")
            return (typed_p if typed_p.exists() else None,
                    block_p if block_p.exists() else None)

    _ensure_pkg_installed()
    ep_txt = ensure_entrypoints(replica_dir)

    # Build p-fail args
    pf_args: list[str] = []
    for p in p_fail_grid:
        try:
            pf_args += ["--p-fail", f"{float(p)}"]
        except Exception:
            continue

    # Recompute typed / all-blocking
    cmd_typed = [
        "edgetyper", "resilience",
        "--edges", str(edges_p),
        "--pred", str(pred_p),
        "--entrypoints", str(ep_txt),
        "--out", str(typed_p),
    ] + pf_args

    cmd_block = [
        "edgetyper", "resilience",
        "--edges", str(edges_p),
        "--pred", str(pred_p),
        "--entrypoints", str(ep_txt),
        "--assume-all-blocking",
        "--out", str(block_p),
    ] + pf_args

    try:
        print(f"[{replica_dir.name}] Recomputing typed availability…")
        subprocess.run(cmd_typed, check=True)
        print(f"[{replica_dir.name}] Recomputing all-blocking availability…")
        subprocess.run(cmd_block, check=True)
    except subprocess.CalledProcessError as e:
        print(f"::warning ::[{replica_dir.name}] edgetyper resilience failed: {e}")

    return (typed_p if typed_p.exists() else None,
            block_p if block_p.exists() else None)

# ------------------------------ aggregation -------------------------------

def build_aggregate(replicas: list[Path], site: Path) -> None:
    site.mkdir(parents=True, exist_ok=True)
    data_dir = site / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    typed_all: list[pd.DataFrame] = []
    block_all: list[pd.DataFrame] = []
    live_all: list[pd.DataFrame] = []

    # Self‑heal each replica, then ingest
    grid_env = os.environ.get("P_FAIL_GRID", "")
    pgrid = [float(x) for x in grid_env.split()] if grid_env else DEFAULT_PFAIL_GRID

    for rdir in replicas:
        rid = rdir.name
        ensure_availability(rdir, pgrid)
        t = _read_csv(rdir / "availability_typed.csv", ["entrypoint", "p_fail", "R_model"])
        b = _read_csv(rdir / "availability_block.csv", ["entrypoint", "p_fail", "R_model"])
        l = _read_csv(rdir / "live_availability.csv", ["entrypoint", "p_fail", "R_live"])
        if t is not None and not t.empty:
            typed_all.append(t.assign(replica=rid))
        if b is not None and not b.empty:
            block_all.append(b.assign(replica=rid))
        if l is not None and not l.empty:
            live_all.append(l.assign(replica=rid))

    typed_all_df = pd.concat(typed_all, ignore_index=True) if typed_all else pd.DataFrame(
        columns=["entrypoint", "p_fail", "R_model", "replica"]
    )
    block_all_df = pd.concat(block_all, ignore_index=True) if block_all else pd.DataFrame(
        columns=["entrypoint", "p_fail", "R_model", "replica"]
    )
    live_all_df = pd.concat(live_all, ignore_index=True) if live_all else pd.DataFrame(
        columns=["entrypoint", "p_fail", "R_live", "replica"]
    )

    for df, col in ((typed_all_df, "R_model"), (block_all_df, "R_model")):
        if not df.empty:
            df["entrypoint"] = df["entrypoint"].astype(str)
            df["p_fail"] = pd.to_numeric(df["p_fail"], errors="coerce")
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if not live_all_df.empty:
        live_all_df["entrypoint"] = live_all_df["entrypoint"].astype(str)
        live_all_df["p_fail"] = pd.to_numeric(live_all_df["p_fail"], errors="coerce")
        live_all_df["R_live"] = pd.to_numeric(live_all_df["R_live"], errors="coerce")

    # Persist long tables
    if not typed_all_df.empty:
        typed_all_df.to_csv(data_dir / "availability_typed_all.csv", index=False)
    if not block_all_df.empty:
        block_all_df.to_csv(data_dir / "availability_block_all.csv", index=False)
    if not live_all_df.empty:
        live_all_df.to_csv(data_dir / "availability_live_all.csv", index=False)

    def agg_mean_std(df: pd.DataFrame, value_col: str, prefix: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["entrypoint", "p_fail",
                                         f"{prefix}_mean", f"{prefix}_std", f"n_{prefix}"])
        g = (df.groupby(["entrypoint", "p_fail"], as_index=False)
               .agg(**{
                   f"{prefix}_mean": (value_col, "mean"),
                   f"{prefix}_std":  (value_col, "std"),
                   f"n_{prefix}":    (value_col, "count"),
               }))
        return g

    agg_t = agg_mean_std(typed_all_df, "R_model", "typed")
    agg_b = agg_mean_std(block_all_df, "R_model", "block")
    agg_l = None
    if not live_all_df.empty:
        agg_l = (live_all_df.groupby(["entrypoint", "p_fail"], as_index=False)
