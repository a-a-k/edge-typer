from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple
import random

import pandas as pd

@dataclass
class SimConfig:
    p_fail: List[float]
    samples: int = 900_000
    seed: int = 0

def _tarjan_scc(adj: Dict[str, Set[str]]) -> List[Set[str]]:
    idx, low = {}, {}
    stack: List[str] = []
    on = set()
    out: List[Set[str]] = []
    i = 0

    def dfs(v: str) -> None:
        nonlocal i
        idx[v] = i
        low[v] = i
        i += 1
        stack.append(v); on.add(v)
        for w in adj.get(v, ()):
            if w not in idx:
                dfs(w)
                low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], idx[w])
        if low[v] == idx[v]:
            comp: Set[str] = set()
            while True:
                w = stack.pop()
                on.discard(w)
                comp.add(w)
                if w == v: break
            out.append(comp)

    # ensure isolated vertices are included
    allv = set(adj.keys()) | {w for vs in adj.values() for w in vs}
    for v in sorted(allv):
        if v not in idx:
            dfs(v)
    return out

def _contract_to_dag(adj: Dict[str, Set[str]]) -> Tuple[Dict[int, Set[int]], Dict[str, int], Dict[int, Set[str]]]:
    sccs = _tarjan_scc(adj)
    comp_of: Dict[str, int] = {}
    for i, comp in enumerate(sccs):
        for v in comp:
            comp_of[v] = i
    dag: Dict[int, Set[int]] = {i: set() for i in range(len(sccs))}
    for u, nbrs in adj.items():
        cu = comp_of[u]
        for v in nbrs:
            cv = comp_of[v]
            if cu != cv:
                dag[cu].add(cv)
    members: Dict[int, Set[str]] = {}
    for v, c in comp_of.items():
        members.setdefault(c, set()).add(v)
    return dag, comp_of, members

# ---------- Helpers to build BLOCKING adjacency from edges + labels ----------
def blocking_adjacency_from_edges(edges: pd.DataFrame,
                                  preds: pd.DataFrame,
                                  assume_all_blocking: bool = False) -> Dict[str, Set[str]]:
    """
    edges: columns include [src_service, dst_service, ...]
    preds: columns include [src_service, dst_service, pred_label in {'async','sync'}]
    """
    g = edges.merge(preds[["src_service", "dst_service", "pred_label"]],
                    on=["src_service", "dst_service"], how="left")
    if assume_all_blocking:
        g["etype"] = "BLOCKING"
    else:
        g["etype"] = g["pred_label"].str.lower().map({"async": "ASYNC", "sync": "BLOCKING"})
    g = g.dropna(subset=["etype"]).copy()

    adj: Dict[str, Set[str]] = {}
    for s, d, et in g[["src_service","dst_service","etype"]].itertuples(index=False, name=None):
        if et != "BLOCKING":
            continue
        adj.setdefault(s, set()).add(d)
        adj.setdefault(d, adj.get(d, set()))
    # ensure isolated nodes present
    for s, d in g[["src_service","dst_service"]].itertuples(index=False, name=None):
        adj.setdefault(s, adj.get(s, set()))
        adj.setdefault(d, adj.get(d, set()))
    return adj

def guess_entrypoints(adj: Dict[str, Set[str]]) -> List[str]:
    indeg: Dict[str, int] = {v: 0 for v in adj}
    for u, nbrs in adj.items():
        for v in nbrs:
            indeg[v] = indeg.get(v, 0) + 1
    eps = [v for v, d in indeg.items() if d == 0]
    return eps or sorted(adj.keys())

# ---------- Monte‑Carlo reachability on alive BLOCKING graph ----------
def estimate_availability(adj_blocking: Dict[str, Set[str]],
                          replicas: Dict[str, int],
                          entrypoints: Iterable[str],
                          cfg: SimConfig) -> pd.DataFrame:

    dag, comp_of, members = _contract_to_dag(adj_blocking)
    leaves: Set[int] = {c for c, nbrs in dag.items() if not nbrs}

    eps = [e for e in entrypoints if e in comp_of]
    if not eps:
        return pd.DataFrame(columns=["entrypoint","p_fail","samples","R_model"])
    eps_comp = {e: comp_of[e] for e in eps}

    rng = random.Random(cfg.seed)

    def alive_service(v: str, p: float) -> bool:
        r = int(replicas.get(v, 1))
        if r <= 0:
            return False
        # any replica survives
        for _ in range(r):
            if rng.random() > p:
                return True
        return False

    rows = []
    for p in cfg.p_fail:
        success = {e: 0 for e in eps}
        for _ in range(cfg.samples):
            alive_v = {v: alive_service(v, p) for v in comp_of.keys()}
            # component alive iff all members alive (conservative for blocking within SCC)
            alive_c = {c: all(alive_v[v] for v in members[c]) for c in members}

            # BFS on alive condensation DAG
            for e, ce in eps_comp.items():
                if not alive_c.get(ce, False):
                    continue
                q = [ce]
                seen = {ce}
                ok = False
                while q and not ok:
                    x = q.pop(0)
                    if x in leaves:
                        ok = True
                        break
                    for y in dag.get(x, ()):
                        if alive_c.get(y, False) and y not in seen:
                            seen.add(y); q.append(y)
                if ok:
                    success[e] += 1

        for e in eps:
            rows.append({
                "entrypoint": e,
                "p_fail": float(p),
                "samples": int(cfg.samples),
                "R_model": float(success[e] / cfg.samples),
            })
    return pd.DataFrame(rows)
