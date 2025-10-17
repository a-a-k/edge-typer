"""
High-precision rule labels + baselines.

- SemConv-only baseline: async iff any messaging semantics observed on the edge.
- Timing-only baseline: async iff median_lag_ns >= 0 and p_overlap <= 0.25 (tuneable heuristic).
- Rule labels (high-confidence):
    * async if p_messaging >= 0.6 or (p_nonneg_lag >= 0.9 and p_overlap <= 0.1)
    * sync  if p_messaging == 0.0 and p_overlap >= 0.6
Ambiguous otherwise (defer to ML fallback).
"""
from __future__ import annotations

import pandas as pd


def baseline_semconv(feat_semconv: pd.DataFrame) -> pd.DataFrame:
    df = feat_semconv[["src_service", "dst_service", "p_messaging"]].copy()
    df["pred_label"] = df["p_messaging"].map(lambda p: "async" if p > 0.0 else "sync")
    df["pred_score"] = df["p_messaging"].astype(float)
    return df[["src_service", "dst_service", "pred_label", "pred_score"]]


def baseline_timing(feat_timing: pd.DataFrame) -> pd.DataFrame:
    df = feat_timing[["src_service", "dst_service", "median_lag_ns", "p_overlap"]].copy()
    df["pred_label"] = df.apply(
        lambda r: "async" if (r["median_lag_ns"] >= 0 and r["p_overlap"] <= 0.25) else "sync",
        axis=1,
    )
    # Simple score proxy: larger positive lag and lower overlap → higher async score
    df["pred_score"] = df["median_lag_ns"].clip(lower=0) / (1e6) * (1 - df["p_overlap"])
    return df[["src_service", "dst_service", "pred_label", "pred_score"]]


def rule_labels(feat_semconv: pd.DataFrame, feat_timing: pd.DataFrame) -> pd.DataFrame:
    df = feat_semconv.merge(
        feat_timing[["src_service", "dst_service", "median_lag_ns", "p_overlap", "p_nonneg_lag"]],
        on=["src_service", "dst_service"],
        how="left",
    ).fillna({"median_lag_ns": 0, "p_overlap": 0.0, "p_nonneg_lag": 0.0})

    def label_row(r):
        # High-confidence async
        if r["p_messaging"] >= 0.6 or (r["p_nonneg_lag"] >= 0.9 and r["p_overlap"] <= 0.1):
            return "async", "high"
        # High-confidence sync
        if r["p_messaging"] == 0.0 and r["p_overlap"] >= 0.6:
            return "sync", "high"
        # Ambiguous
        return "unknown", "low"

    lab = df.apply(lambda r: label_row(r), axis=1, result_type="expand")
    df["rule_label"] = lab[0]
    df["rule_conf"] = lab[1]
    return df[
        [
            "src_service", "dst_service",
            "rule_label", "rule_conf",
            "p_messaging", "link_ratio", "median_lag_ns", "p_overlap", "p_nonneg_lag",
        ]
    ]
