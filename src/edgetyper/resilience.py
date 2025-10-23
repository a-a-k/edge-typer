import argparse, csv, json, random
from collections import defaultdict
import networkx as nx
import numpy as np

def load_edges(path):
    typed = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            typed.append((row["src"], row["dst"], row["label"]))
    return typed

def build_graph(typed_edges):
    G_block = nx.DiGraph()
    G_all = nx.DiGraph()
    for s,d,lab in typed_edges:
        if lab == "blocking":
            G_block.add_edge(s,d)
        # all edges blocking in baseline:
        G_all.add_edge(s,d)
    # collect nodes
    for s,d,_ in typed_edges:
        if s not in G_block: G_block.add_node(s)
        if d not in G_block: G_block.add_node(d)
        if s not in G_all: G_all.add_node(s)
        if d not in G_all: G_all.add_node(d)
    return G_block, G_all

def alive_blocking(G: nx.DiGraph, failed: set[str]):
    # Node alive if not failed and none of its blocking predecessors are dead recursively
    # We approximate via iterative fixed-point
    alive = {n for n in G.nodes if n not in failed}
    changed = True
    while changed:
        changed = False
        for n in list(alive):
            for p in G.predecessors(n):
                if p not in alive:
                    alive.remove(n); changed = True; break
    return alive

def simulate(G_block: nx.DiGraph, G_all: nx.DiGraph, p=0.3, samples=1000):
    nodes = sorted(set(G_all.nodes))
    n = len(nodes)
    R_typed = []; R_all = []
    for _ in range(samples):
        failed = {nodes[i] for i in range(n) if random.random() < p}
        alive_t = alive_blocking(G_block, failed)
        alive_a = alive_blocking(G_all, failed)
        R_typed.append(len(alive_t)/n)
        R_all.append(len(alive_a)/n)
    return {
        "R_avg_typed": float(np.mean(R_typed)),
        "R_avg_allblk": float(np.mean(R_all)),
        "R_samples_typed": R_typed[:100],  # preview
        "R_samples_allblk": R_all[:100]
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", required=True)
    ap.add_argument("--p", type=float, default=0.30)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--out-typed", required=True)
    ap.add_argument("--out-allblk", required=True)
    args = ap.parse_args()

    typed_edges = load_edges(args.edges)
    G_block, G_all = build_graph(typed_edges)
    res = simulate(G_block, G_all, p=args.p, samples=args.samples)

    with open(args.out_typed, "w") as f:
        json.dump({"R_model": res["R_avg_typed"], "detail": res["R_samples_typed"]}, f, indent=2)
    with open(args.out_allblk, "w") as f:
        json.dump({"R_model": res["R_avg_allblk"], "detail": res["R_samples_allblk"]}, f, indent=2)

if __name__ == "__main__":
    main()
