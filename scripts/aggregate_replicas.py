#!/usr/bin/env python3
import argparse, json, os, random
from pathlib import Path
import pandas as pd

def read_json(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None

def bootstrap_ci(values, iters=2000, alpha=0.05, seed=0):
    if not values: return (None, None)
    rng = random.Random(seed)
    n = len(values)
    boots = []
    for _ in range(iters):
        samp = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(pd.Series(samp).mean())
    boots.sort()
    lo = boots[int((alpha/2)*iters)]
    hi = boots[int((1-alpha/2)*iters)]
    return (lo, hi)

def load_metrics(rep_dir):
    m = {}
    for key in ["ours_physical","semconv_physical","timing_physical","ours_semconv_drop","ours_timing_drop"]:
        path = next(Path(rep_dir).glob(f"metrics_{key}.json"), None)
        if path: m[key] = read_json(path)
    return m

def summary_from_metrics(mj):
    out = {}
    for k,v in mj.items():
        rep = v.get("classification_report",{}) if v else {}
        s_typed = rep.get("macro avg",{}).get("f1-score", None)
        out[f"macroF1_{k}"] = s_typed
        out[f"n_eval_{k}"] = v.get("n_eval_edges") if v else None
        out[f"n_uncertain_{k}"] = v.get("n_uncertain_pred") if v else 0
    return out

def load_plans(rep_dir):
    typed = Path(rep_dir)/"plan_physical.csv"
    block = Path(rep_dir)/"plan_all_blocking.csv"
    pt = pd.read_csv(typed) if typed.exists() else None
    pb = pd.read_csv(block) if block.exists() else None
    def score(df):
        if df is None: return {}
        if not {"target","IBS","DBS"}.issubset(df.columns): return {}
        s = df.groupby("target", as_index=False).agg(score=("IBS", "sum"))
        s2 = df.groupby("target", as_index=False).agg(DBS=("DBS","sum"))
        s = s.merge(s2, on="target", how="left"); s["score"]=s["score"]+s["DBS"].fillna(0)
        return dict(zip(s["target"], s["score"]))
    return score(pt), score(pb)

def load_live(rep_dir):
    obs = read_json(Path(rep_dir)/"observations.json") or {}
    segs = {s.get("name"): s for s in obs.get("segments",[])}
    base = segs.get("baseline",{})
    base_counts = base.get("by_service") or base.get("by_service_top") or {}
    impacts = {}
    for name, seg in segs.items():
        if not name.startswith("fault:"): continue
        fcounts = seg.get("by_service") or seg.get("by_service_top") or {}
        # relative change on counts (per-second normalization is upstream in observe if needed)
        all_svc = set(base_counts) | set(fcounts)
        for svc in all_svc:
            b = float(base_counts.get(svc, 0))
            f = float(fcounts.get(svc, 0))
            val = abs(f-b) / (b if b>0 else 1.0)
            impacts.setdefault(name, {})[svc] = val
    return impacts  # dict fault -> {service -> impact}

def rank_spearman(a, b):
    # simple spearman without scipy
    if not a or not b: return None
    common = list(set(a) & set(b))
    if len(common) < 3: return None
    sa = pd.Series([a[t] for t in common]).rank()
    sb = pd.Series([b[t] for t in common]).rank()
    return float(sa.corr(sb, method='pearson'))

def precision_at_k(scores, target, k):
    if not scores or target not in scores: return None
    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return 1.0 if any(t==target for t,_ in top) else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicas-dir", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    outdir = Path(args.outdir); data = outdir/"data"; data.mkdir(parents=True, exist_ok=True)

    rows = []
    spearman_typed, spearman_block = [], []
    p3_typed, p3_block, p5_typed, p5_block = [], [], [], []

    for rep_dir in sorted(Path(args.replicas_dir).glob("replicate-*")):
        mets = load_metrics(rep_dir)
        row = {"replica": rep_dir.name}
        row.update(summary_from_metrics(mets))
        # plans + live
        s_typed, s_block = load_plans(rep_dir)
        live = load_live(rep_dir)  # fault -> {svc -> impact}
        # global spearman on union over faults (average per svc impact)
        if live:
            imp_all = {}
            for d in live.values():
                for svc,val in d.items():
                    imp_all.setdefault(svc, []).append(val)
            imp_avg = {svc: sum(v)/len(v) for svc,v in imp_all.items()}
            sp_t = rank_spearman(s_typed, imp_avg)
            sp_b = rank_spearman(s_block, imp_avg)
            row["spearman_typed"] = sp_t
            row["spearman_block"] = sp_b
            if sp_t is not None: spearman_typed.append(sp_t)
            if sp_b is not None: spearman_block.append(sp_b)
            # per-fault p@k (k=3,5), averaged over faults
            p3_t = []; p3_b = []; p5_t = []; p5_b = []
            for fname, imp in live.items():
                # fault target from name if encoded as "fault:service"
                target = fname.split(":",1)[1] if ":" in fname else None
                if target:
                    pt3 = precision_at_k(s_typed, target, 3)
                    pb3 = precision_at_k(s_block, target, 3)
                    pt5 = precision_at_k(s_typed, target, 5)
                    pb5 = precision_at_k(s_block, target, 5)
                    if pt3 is not None: p3_t.append(pt3)
                    if pb3 is not None: p3_b.append(pb3)
                    if pt5 is not None: p5_t.append(pt5)
                    if pb5 is not None: p5_b.append(pb5)
            row["p_at_3_typed"] = sum(p3_t)/len(p3_t) if p3_t else None
            row["p_at_3_block"] = sum(p3_b)/len(p3_b) if p3_b else None
            row["p_at_5_typed"] = sum(p5_t)/len(p5_t) if p5_t else None
            row["p_at_5_block"] = sum(p5_b)/len(p5_b) if p5_b else None
            if row["p_at_3_typed"] is not None: p3_typed.append(row["p_at_3_typed"])
            if row["p_at_3_block"] is not None: p3_block.append(row["p_at_3_block"])
            if row["p_at_5_typed"] is not None: p5_typed.append(row["p_at_5_typed"])
            if row["p_at_5_block"] is not None: p5_block.append(row["p_at_5_block"])
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(data/"replicas_summary.csv", index=False)

    # aggregates
    agg = {
      "n_replicas": int(len(df)),
      "macroF1_means": {c: float(df[c].dropna().mean()) for c in df.columns if c.startswith("macroF1_")},
      "uncertain_means": {c: float(df[c].dropna().mean()) for c in df.columns if c.startswith("n_uncertain_")},
      "spearman_mean_typed": float(pd.Series(spearman_typed).mean()) if spearman_typed else None,
      "spearman_mean_block": float(pd.Series(spearman_block).mean()) if spearman_block else None,
      "spearman_winrate": float(sum(1 for t,b in zip(spearman_typed, spearman_block) if t is not None and b is not None and t>b) / max(1,len(spearman_typed))) if spearman_typed and spearman_block else None,
      "p_at_3_mean_typed": float(pd.Series(p3_typed).mean()) if p3_typed else None,
      "p_at_3_mean_block": float(pd.Series(p3_block).mean()) if p3_block else None,
      "p_at_5_mean_typed": float(pd.Series(p5_typed).mean()) if p5_typed else None,
      "p_at_5_mean_block": float(pd.Series(p5_block).mean()) if p5_block else None
    }
    # bootstrap CIs
    for key, vec in [("spearman_typed", spearman_typed), ("spearman_block", spearman_block),
                     ("p_at_3_typed", p3_typed), ("p_at_3_block", p3_block),
                     ("p_at_5_typed", p5_typed), ("p_at_5_block", p5_block)]:
        lo, hi = bootstrap_ci(list(vec))
        agg[f"{key}_ci95"] = [lo, hi] if lo is not None else None

    (data/"aggregate_summary.json").write_text(json.dumps(agg, indent=2))

    # HTML (compact)
    html = f"""<!doctype html><meta charset="utf-8">
<title>EdgeTyper — Aggregate (256×, soak 1800s)</title>
<style>body{{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:24px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ddd;padding:6px 10px}}</style>
<h1>EdgeTyper — Aggregate (n={agg['n_replicas']}, soak=1800s)</h1>
<h2>Typing quality</h2>
<p>Macro-F1 means (higher is better):</p>
<ul>
<li>Ours — physical: {agg['macroF1_means'].get('macroF1_ours_physical')}</li>
<li>SemConv baseline: {agg['macroF1_means'].get('macroF1_semconv_physical')}</li>
<li>Timing baseline: {agg['macroF1_means'].get('macroF1_timing_physical')}</li>
<li>Ours — SemConv dropped: {agg['macroF1_means'].get('macroF1_ours_semconv_drop')}</li>
<li>Ours — Timing dropped: {agg['macroF1_means'].get('macroF1_ours_timing_drop')}</li>
</ul>
<h2>Prediction vs live (rank, pooled)</h2>
<table><tr><th>Metric</th><th>Typed</th><th>All-blocking</th><th>Win rate (typed&gt;block)</th></tr>
<tr><td>Spearman ρ (mean)</td><td>{agg['spearman_mean_typed']}</td><td>{agg['spearman_mean_block']}</td><td>{agg['spearman_winrate']}</td></tr>
</table>
<p>95% CIs (bootstrap): ρ_typed={agg.get('spearman_typed_ci95')}, ρ_block={agg.get('spearman_block_ci95')}</p>
<h2>Top-k hit rate</h2>
<table><tr><th></th><th>Typed</th><th>All-blocking</th></tr>
<tr><td>P@3 (mean)</td><td>{agg['p_at_3_mean_typed']}</td><td>{agg['p_at_3_mean_block']}</td></tr>
<tr><td>P@5 (mean)</td><td>{agg['p_at_5_mean_typed']}</td><td>{agg['p_at_5_mean_block']}</td></tr>
</table>
<p>95% CIs: P@3_typed={agg.get('p_at_3_typed_ci95')}, P@3_block={agg.get('p_at_3_block_ci95')}<br>
P@5_typed={agg.get('p_at_5_typed_ci95')}, P@5_block={agg.get('p_at_5_block_ci95')}</p>
<h2>Downloads</h2>
<ul>
<li><a href="data/aggregate_summary.json" download>aggregate_summary.json</a></li>
<li><a href="data/replicas_summary.csv" download>replicas_summary.csv</a></li>
</ul>
"""
    (outdir/"index.html").write_text(html)

if __name__ == "__main__":
    main()
