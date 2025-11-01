#!/usr/bin/env python3
"""Aggregate 256-replica EdgeTyper runs into a single summary site."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import pandas as pd


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def bootstrap_ci(values: Iterable[float], iters: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float | None, float | None]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None, None
    rng = random.Random(seed)
    n = len(vals)
    boots = []
    for _ in range(iters):
        samp = [vals[rng.randrange(n)] for _ in range(n)]
        boots.append(mean(samp))
    boots.sort()
    lo_idx = max(0, int((alpha / 2) * iters))
    hi_idx = min(iters - 1, int((1 - alpha / 2) * iters))
    return boots[lo_idx], boots[hi_idx]


def load_metrics(rep_dir: Path) -> dict[str, dict | None]:
    keys = {
        "ours_physical": "metrics_ours_physical.json",
        "semconv_physical": "metrics_semconv_physical.json",
        "timing_physical": "metrics_timing_physical.json",
        "ours_semconv_drop": "metrics_ours_semconv_drop.json",
        "ours_timing_drop": "metrics_ours_timing_drop.json",
    }
    out: dict[str, dict | None] = {}
    for key, fname in keys.items():
        path = rep_dir / fname
        out[key] = read_json(path) if path.exists() else None
    return out


def summary_from_metrics(metrics: dict[str, dict | None]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key, data in metrics.items():
        if not data:
            out[f"macroF1_{key}"] = None
            out[f"n_eval_{key}"] = None
            out[f"n_uncertain_{key}"] = None
            continue
        report = data.get("classification_report", {})
        macro = report.get("macro avg", {})
        out[f"macroF1_{key}"] = float(macro.get("f1-score")) if macro.get("f1-score") is not None else None
        out[f"n_eval_{key}"] = data.get("n_eval_edges")
        out[f"n_uncertain_{key}"] = data.get("n_uncertain_pred")
    return out


def _counts_per_sec(segment: dict[str, Any] | None) -> tuple[dict[str, float], float]:
    if not segment:
        return {}, 0.0
    counts_raw = segment.get("by_service") or segment.get("by_service_top") or {}
    counts = {str(k): float(v) for k, v in counts_raw.items()}
    duration_s = float(segment.get("duration_s") or 0.0)
    if duration_s <= 0:
        duration_ns = segment.get("duration_ns")
        if duration_ns:
            try:
                duration_s = float(duration_ns) / 1_000_000_000
            except Exception:
                duration_s = 0.0
    if duration_s <= 0:
        s = segment.get("start_ns")
        e = segment.get("end_ns")
        if isinstance(s, (int, float)) and isinstance(e, (int, float)) and e > s:
            duration_s = (float(e) - float(s)) / 1_000_000_000
    if duration_s <= 0:
        duration_s = 1.0
    per_sec = {svc: cnt / duration_s for svc, cnt in counts.items()}
    return per_sec, duration_s


def _sym_delta(base: float, fault: float) -> float:
    denom = (abs(base) + abs(fault)) / 2.0
    if denom <= 0:
        return 0.0
    return abs(fault - base) / denom


def _resolve_target(seg: dict[str, Any], deltas: dict[str, float]) -> str | None:
    if seg.get("target_service"):
        return str(seg["target_service"])
    name = str(seg.get("name", ""))
    if ":" in name:
        return name.split(":", 1)[1]
    if deltas:
        return max(deltas.items(), key=lambda kv: kv[1])[0]
    return None


def load_live(rep_dir: Path) -> dict[str, Any] | None:
    obs = read_json(rep_dir / "observations.json") or {}
    segments = obs.get("segments", [])
    if not segments:
        return None
    baseline = None
    for seg in segments:
        if str(seg.get("name", "")).lower().startswith("baseline"):
            baseline = seg
            break
    if baseline is None:
        baseline = segments[0]
    base_per_sec, _ = _counts_per_sec(baseline)

    faults: list[dict[str, Any]] = []
    for seg in segments:
        if seg is baseline:
            continue
        per_sec_fault, duration_s = _counts_per_sec(seg)
        deltas = {
            svc: _sym_delta(base_per_sec.get(svc, 0.0), per_sec_fault.get(svc, 0.0))
            for svc in set(base_per_sec) | set(per_sec_fault)
        }
        faults.append(
            {
                "name": seg.get("name"),
                "target": _resolve_target(seg, deltas),
                "per_sec": per_sec_fault,
                "delta": deltas,
                "duration_s": duration_s,
            }
        )
    return {"baseline_per_sec": base_per_sec, "faults": faults}


def load_plans(rep_dir: Path) -> tuple[dict[str, float], dict[str, float]]:
    typed_path = rep_dir / "plan_physical.csv"
    block_path = rep_dir / "plan_all_blocking.csv"

    def _scores(path: Path, score_col: str) -> dict[str, float]:
        if not path.exists():
            return {}
        df = pd.read_csv(path)
        if "target" not in df.columns:
            return {}
        for col in ["IBS", "DBS"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        if score_col not in df.columns:
            df[score_col] = df.get("IBS", 0.0) + df.get("DBS", 0.0)
        df["target"] = df["target"].astype(str)
        return dict(zip(df["target"], df[score_col].astype(float)))

    return _scores(typed_path, "typed_score"), _scores(block_path, "blocking_score")


def rank_corr(scores: dict[str, float], impacts: dict[str, float], method: str) -> float | None:
    common = sorted(set(scores) & set(impacts))
    if len(common) < 2:
        return None
    s_scores = pd.Series([scores[k] for k in common])
    s_imp = pd.Series([impacts[k] for k in common])
    val = s_scores.corr(s_imp, method=method)
    if pd.isna(val):
        return None
    return float(val)


def precision_at_k(scores: dict[str, float], target: str | None, k: int) -> float | None:
    if not target or target not in scores:
        return None
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top = [svc for svc, _ in ordered[:k]]
    return 1.0 if target in top else 0.0


def win_rate(pairs: Iterable[tuple[float | None, float | None]]) -> float | None:
    wins = 0
    total = 0
    for typed, block in pairs:
        if typed is None or block is None:
            continue
        total += 1
        if typed > block:
            wins += 1
    if total == 0:
        return None
    return wins / total


def load_availability(rep_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (typed_df, block_df) with columns [entrypoint, p_fail, R_model] or empty frames."""
    def _load(name: str) -> pd.DataFrame:
        path = rep_dir / name
        if not path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
        need = {"entrypoint", "p_fail", "R_model"}
        if not need.issubset(set(df.columns)):
            return pd.DataFrame()
        return df.loc[:, ["entrypoint", "p_fail", "R_model"]].copy()
    return _load("availability_typed.csv"), _load("availability_block.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicas-dir", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    args = ap.parse_args()

    outdir: Path = args.outdir
    data_dir = outdir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    spearman_typed_vals: list[float] = []
    spearman_block_vals: list[float] = []
    kendall_typed_vals: list[float] = []
    kendall_block_vals: list[float] = []
    spearman_pairs: list[tuple[float | None, float | None]] = []
    kendall_pairs: list[tuple[float | None, float | None]] = []
    p3_typed_vals: list[float] = []
    p3_block_vals: list[float] = []
    p5_typed_vals: list[float] = []
    p5_block_vals: list[float] = []
    avail_typed_frames: list[pd.DataFrame] = []
    avail_block_frames: list[pd.DataFrame] = []

    replicas_dir: Path = args.replicas_dir
    for rep_dir in sorted(replicas_dir.glob("replicate-*")):
        metrics = load_metrics(rep_dir)
        row: dict[str, Any] = {"replica": rep_dir.name}
        row.update(summary_from_metrics(metrics))

        typed_scores, block_scores = load_plans(rep_dir)
        av_t, av_b = load_availability(rep_dir)
        if not av_t.empty:
            avail_typed_frames.append(av_t.assign(replica=rep_dir.name))
        if not av_b.empty:
            avail_block_frames.append(av_b.assign(replica=rep_dir.name))
        live = load_live(rep_dir)

        pooled_impacts: dict[str, float] = {}
        fault_metrics: list[dict[str, float | None]] = []
        if live and live.get("faults"):
            accum: dict[str, list[float]] = {}
            for fault in live["faults"]:
                deltas: dict[str, float] = fault.get("delta", {})
                for svc, val in deltas.items():
                    accum.setdefault(svc, []).append(val)

                spearman_fault_t = rank_corr(typed_scores, deltas, "spearman")
                spearman_fault_b = rank_corr(block_scores, deltas, "spearman")
                kendall_fault_t = rank_corr(typed_scores, deltas, "kendall")
                kendall_fault_b = rank_corr(block_scores, deltas, "kendall")
                p3_t = precision_at_k(typed_scores, fault.get("target"), 3)
                p3_b = precision_at_k(block_scores, fault.get("target"), 3)
                p5_t = precision_at_k(typed_scores, fault.get("target"), 5)
                p5_b = precision_at_k(block_scores, fault.get("target"), 5)

                fault_metrics.append(
                    {
                        "spearman_typed": spearman_fault_t,
                        "spearman_block": spearman_fault_b,
                        "kendall_typed": kendall_fault_t,
                        "kendall_block": kendall_fault_b,
                        "p_at_3_typed": p3_t,
                        "p_at_3_block": p3_b,
                        "p_at_5_typed": p5_t,
                        "p_at_5_block": p5_b,
                    }
                )

                for val_list, val in ((p3_typed_vals, p3_t), (p3_block_vals, p3_b), (p5_typed_vals, p5_t), (p5_block_vals, p5_b)):
                    if val is not None:
                        val_list.append(val)

            pooled_impacts = {svc: mean(vals) for svc, vals in accum.items() if vals}

            row["spearman_typed"] = rank_corr(typed_scores, pooled_impacts, "spearman")
            row["spearman_block"] = rank_corr(block_scores, pooled_impacts, "spearman")
            row["kendall_typed"] = rank_corr(typed_scores, pooled_impacts, "kendall")
            row["kendall_block"] = rank_corr(block_scores, pooled_impacts, "kendall")

            if fault_metrics:
                row["p_at_3_typed"] = mean([fm["p_at_3_typed"] for fm in fault_metrics if fm["p_at_3_typed"] is not None]) if any(fm["p_at_3_typed"] is not None for fm in fault_metrics) else None
                row["p_at_3_block"] = mean([fm["p_at_3_block"] for fm in fault_metrics if fm["p_at_3_block"] is not None]) if any(fm["p_at_3_block"] is not None for fm in fault_metrics) else None
                row["p_at_5_typed"] = mean([fm["p_at_5_typed"] for fm in fault_metrics if fm["p_at_5_typed"] is not None]) if any(fm["p_at_5_typed"] is not None for fm in fault_metrics) else None
                row["p_at_5_block"] = mean([fm["p_at_5_block"] for fm in fault_metrics if fm["p_at_5_block"] is not None]) if any(fm["p_at_5_block"] is not None for fm in fault_metrics) else None

        spearman_pairs.append((row.get("spearman_typed"), row.get("spearman_block")))
        kendall_pairs.append((row.get("kendall_typed"), row.get("kendall_block")))

        if row.get("spearman_typed") is not None:
            spearman_typed_vals.append(row["spearman_typed"])
        if row.get("spearman_block") is not None:
            spearman_block_vals.append(row["spearman_block"])
        if row.get("kendall_typed") is not None:
            kendall_typed_vals.append(row["kendall_typed"])
        if row.get("kendall_block") is not None:
            kendall_block_vals.append(row["kendall_block"])

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(data_dir / "replicas_summary.csv", index=False)

    # Pooled availability (optional)
    avail_table_html = ""
    try:
        if avail_typed_frames and avail_block_frames:
            av_t_all = pd.concat(avail_typed_frames, ignore_index=True)
            av_b_all = pd.concat(avail_block_frames, ignore_index=True)
            t_mean = (av_t_all.groupby("p_fail")["R_model"].mean().reset_index(name="R_model_typed"))
            b_mean = (av_b_all.groupby("p_fail")["R_model"].mean().reset_index(name="R_model_block"))
            mix = t_mean.merge(b_mean, on="p_fail", how="outer").sort_values("p_fail")
            (data_dir / "availability_typed_pooled.csv").write_text(t_mean.to_csv(index=False))
            (data_dir / "availability_block_pooled.csv").write_text(b_mean.to_csv(index=False))
            rows_av = "".join(
                f"<tr><td>{float(r.p_fail):.1f}</td>"
                f"<td>{float(getattr(r,'R_model_typed',0.0)):.3f}</td>"
                f"<td>{float(getattr(r,'R_model_block',0.0)):.3f}</td></tr>"
                for r in mix.itertuples(index=False)
            )
            avail_table_html = (
                "<h2>Monte-Carlo availability (pooled)</h2>"
                "<table><thead><tr><th>p_fail</th><th>Typed</th><th>All-blocking</th></tr></thead>"
                f"<tbody>{rows_av}</tbody></table>"
            )
    except Exception:

        agg: dict[str, Any] = {
            "n_replicas": int(len(df)),
            "macroF1_means": {c: float(df[c].dropna().mean()) for c in df.columns if c.startswith("macroF1_") and not df[c].dropna().empty},
            "uncertain_means": {c: float(df[c].dropna().mean()) for c in df.columns if c.startswith("n_uncertain_") and not df[c].dropna().empty},
            "spearman_mean_typed": float(pd.Series(spearman_typed_vals).mean()) if spearman_typed_vals else None,
            "spearman_mean_block": float(pd.Series(spearman_block_vals).mean()) if spearman_block_vals else None,
            "kendall_mean_typed": float(pd.Series(kendall_typed_vals).mean()) if kendall_typed_vals else None,
            "kendall_mean_block": float(pd.Series(kendall_block_vals).mean()) if kendall_block_vals else None,
            "spearman_winrate": win_rate(spearman_pairs),
            "kendall_winrate": win_rate(kendall_pairs),
            "p_at_3_mean_typed": float(pd.Series(p3_typed_vals).mean()) if p3_typed_vals else None,
            "p_at_3_mean_block": float(pd.Series(p3_block_vals).mean()) if p3_block_vals else None,
            "p_at_5_mean_typed": float(pd.Series(p5_typed_vals).mean()) if p5_typed_vals else None,
            "p_at_5_mean_block": float(pd.Series(p5_block_vals).mean()) if p5_block_vals else None,
        }

    # Ensure 'agg' exists even if an earlier step failed before its construction.
    # This prevents UnboundLocalError when writing *_ci95 fields.
    if 'agg' not in locals():
        agg = {"n_replicas": int(len(df)) if 'df' in locals() else 0}
    for key, vec in [
        ("spearman_typed", spearman_typed_vals),
        ("spearman_block", spearman_block_vals),
        ("kendall_typed", kendall_typed_vals),
        ("kendall_block", kendall_block_vals),
        ("p_at_3_typed", p3_typed_vals),
        ("p_at_3_block", p3_block_vals),
        ("p_at_5_typed", p5_typed_vals),
        ("p_at_5_block", p5_block_vals),
    ]:
        lo, hi = bootstrap_ci(vec)
        agg[f"{key}_ci95"] = [lo, hi] if lo is not None and hi is not None else None

    (data_dir / "aggregate_summary.json").write_text(json.dumps(agg, indent=2))

    def fmt(val: float | None) -> str:
        return "n/a" if val is None else f"{val:.3f}"

    agg.setdefault("macroF1_means", {})
    html = f"""<!doctype html><meta charset="utf-8">
<title>EdgeTyper — Aggregate (256×, soak 1800s)</title>
<style>body{{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:24px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ddd;padding:6px 10px}}</style>
<h1>EdgeTyper — Aggregate (n={agg['n_replicas']}, soak=1800s)</h1>
<h2>Typing quality</h2>
<ul>
  <li>Ours — physical: {fmt(agg['macroF1_means'].get('macroF1_ours_physical'))}</li>
  <li>SemConv baseline: {fmt(agg['macroF1_means'].get('macroF1_semconv_physical'))}</li>
  <li>Timing baseline: {fmt(agg['macroF1_means'].get('macroF1_timing_physical'))}</li>
  <li>Ours — SemConv dropped: {fmt(agg['macroF1_means'].get('macroF1_ours_semconv_drop'))}</li>
  <li>Ours — Timing dropped: {fmt(agg['macroF1_means'].get('macroF1_ours_timing_drop'))}</li>
</ul>
<h2>Prediction vs live (pooled)</h2>
<table>
  <tr><th>Metric</th><th>Typed</th><th>All-blocking</th><th>Win rate (typed &gt; block)</th></tr>
  <tr><td>Spearman ρ (mean)</td><td>{fmt(agg.get('spearman_mean_typed'))}</td><td>{fmt(agg.get('spearman_mean_block'))}</td><td>{fmt(agg.get('spearman_winrate'))}</td></tr>
  <tr><td>Kendall τ (mean)</td><td>{fmt(agg.get('kendall_mean_typed'))}</td><td>{fmt(agg.get('kendall_mean_block'))}</td><td>{fmt(agg.get('kendall_winrate'))}</td></tr>
</table>
<p>95% CIs: ρ_typed={agg.get('spearman_typed_ci95')}, ρ_block={agg.get('spearman_block_ci95')}<br>
τ_typed={agg.get('kendall_typed_ci95')}, τ_block={agg.get('kendall_block_ci95')}</p>
<h2>Top-k hit rate</h2>
<table>
  <tr><th></th><th>Typed</th><th>All-blocking</th></tr>
  <tr><td>P@3 (mean)</td><td>{fmt(agg.get('p_at_3_mean_typed'))}</td><td>{fmt(agg.get('p_at_3_mean_block'))}</td></tr>
  <tr><td>P@5 (mean)</td><td>{fmt(agg.get('p_at_5_mean_typed'))}</td><td>{fmt(agg.get('p_at_5_mean_block'))}</td></tr>
</table>
<p>95% CIs: P@3_typed={agg.get('p_at_3_typed_ci95')}, P@3_block={agg.get('p_at_3_block_ci95')}<br>
P@5_typed={agg.get('p_at_5_typed_ci95')}, P@5_block={agg.get('p_at_5_block_ci95')}</p>
{avail_table_html}
<h2>Downloads</h2>
<ul>
  <li><a href="data/aggregate_summary.json" download>aggregate_summary.json</a></li>
  <li><a href="data/replicas_summary.csv" download>replicas_summary.csv</a></li>
</ul>
"""

    (outdir / "index.html").write_text(html)


if __name__ == "__main__":
    main()
