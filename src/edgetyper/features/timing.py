"""
Timing/overlap features at the service→service edge level.

Definitions per event:
- lag_ns = down_start_ns - up_end_ns  (time gap from upstream completion to downstream start)
- overlap = 1 if down_start_ns < up_end_ns else 0  (downstream starts before upstream ends)

Aggregates per edge:
- median_lag_ns, p_overlap (fraction of events with overlap), p_nonneg_lag (share with lag >= 0)

Rationale:
- Synchronous RPC typically shows substantial overlap (client active while server runs).
- Async messaging typically shows down_start >= up_end (little/no overlap; often positive lag).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def features_timing(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            columns=[
                "src_service", "dst_service",
                "median_lag_ns", "p_overlap", "p_nonneg_lag",
            ]
        )

    ev = events.copy()
    # Compute per-event lag and overlap
    ev["lag_ns"] = (ev["down_start_ns"].astype("Int64") - ev["up_end_ns"].astype("Int64")).astype("Int64")
    ev["overlap"] = (ev["down_start_ns"] < ev["up_end_ns"]).astype(int)

    grp = ev.groupby(["src_service", "dst_service"], as_index=False)
    out = grp.agg(
        median_lag_ns=("lag_ns", lambda s: int(np.nanmedian(s.dropna().astype("int64"))) if s.notna().any() else 0),
        p_overlap=("overlap", lambda s: float(np.mean(s)) if len(s) > 0 else 0.0),
        p_nonneg_lag=("lag_ns", lambda s: float((s.fillna(0) >= 0).mean()) if len(s) > 0 else 0.0),
    )
    return out
