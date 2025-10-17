"""
Semantic-convention features at the service→service edge level.

We derive protocol/semantics-facing features from the per-interaction `events` table
and the edge aggregates produced by `graph.build`.

Key signals (all per edge):
- p_messaging: fraction of events classified as 'messaging' (vs 'rpc').
- link_ratio: fraction of events with links present on either side.
- any_messaging_semconv: boolean proxy for messaging SemConv presence (events.kind == 'messaging').

References:
  - OTel messaging semconv (producer/consumer, destination, operation).
  - OTel span links (used in async/batch pipelines).
"""
from __future__ import annotations

import pandas as pd


def features_semconv(events: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """
    Input:
      events: columns = [src_service, dst_service, kind, has_links_up, has_links_down, ...]
      edges:  columns = [src_service, dst_service, n_events, n_rpc, n_messaging, n_with_links]

    Output: one row per (src_service, dst_service) with:
      src_service, dst_service, n_events, n_rpc, n_messaging,
      p_messaging, link_ratio, any_messaging_semconv
    """
    if events.empty and edges.empty:
        return pd.DataFrame(
            columns=[
                "src_service", "dst_service", "n_events", "n_rpc", "n_messaging",
                "p_messaging", "link_ratio", "any_messaging_semconv",
            ]
        )

    # Compute link counts from events (both sides)
    if not events.empty:
        link_counts = (
            events.assign(
                any_link=(events["has_links_up"].fillna(False) | events["has_links_down"].fillna(False)).astype(bool)
            )
            .groupby(["src_service", "dst_service"], as_index=False)["any_link"]
            .sum()
            .rename(columns={"any_link": "n_events_with_links"})
        )
    else:
        link_counts = pd.DataFrame(columns=["src_service", "dst_service", "n_events_with_links"])

    # Merge with edge aggregates
    df = edges.merge(link_counts, on=["src_service", "dst_service"], how="left")
    df["n_events_with_links"] = df["n_events_with_links"].fillna(0).astype(int)

    # Ratios
    df["p_messaging"] = (df["n_messaging"] / df["n_events"]).where(df["n_events"] > 0, 0.0)
    df["link_ratio"] = (df["n_events_with_links"] / df["n_events"]).where(df["n_events"] > 0, 0.0)
    df["any_messaging_semconv"] = (df["n_messaging"] > 0)

    keep_cols = [
        "src_service", "dst_service", "n_events", "n_rpc", "n_messaging",
        "p_messaging", "link_ratio", "any_messaging_semconv",
    ]
    return df[keep_cols]
