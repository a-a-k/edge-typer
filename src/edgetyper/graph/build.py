"""
Build service→service interactions and edges from a normalized spans table.

Inputs
------
A pandas.DataFrame 'spans' as produced by edgetyper.io.otlp_json.read_otlp_json with columns:
  trace_id, span_id, parent_span_id, service_name, span_name, span_kind,
  start_ns, end_ns, duration_ns, has_links, links_count,
  messaging_system, messaging_destination, messaging_operation,
  http_method, rpc_system

Outputs
-------
events_df: one row per inferred interaction with columns:
  src_service, dst_service, kind ('rpc'|'messaging'),
  up_start_ns, up_end_ns, down_start_ns, down_end_ns,
  messaging_destination (for messaging), has_links_up, has_links_down

edges_df: one row per service→service pair with aggregates:
  src_service, dst_service, n_events, n_rpc, n_messaging, n_with_links

Notes
-----
- RPC pairing follows the common pattern where the SERVER span's parentSpanId equals the CLIENT span's spanId
  within the same trace. (Cross-process propagation sets the server's parent to the client span.) 
- Messaging pairing groups by messaging destination (topic/queue) and greedily matches a PRODUCER with the
  earliest CONSUMER whose start >= producer end (approximate causality for queued async work).
- This module *does not* attempt to handle batching or fan-out beyond 1:1 greedy pairing; the features stage
  will use aggregate rates (e.g., fan-out) computed from the events.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd


@dataclass(frozen=True)
class EventsAndEdges:
    events: pd.DataFrame
    edges: pd.DataFrame


def _safe_int(x):
    try:
        return int(x) if x is not None else None
    except Exception:
        return None


def build(spans: pd.DataFrame, *, emit_broker_edges: bool = True, broker_service_name: str = "kafka") -> EventsAndEdges:
    # ---- Guard rails ----
    required_cols = {
        "trace_id", "span_id", "parent_span_id", "service_name", "span_kind",
        "start_ns", "end_ns", "has_links", "links_count",
        "messaging_destination",
    }
    missing = sorted(required_cols - set(spans.columns))
    if missing:
        raise ValueError(f"spans df missing columns: {missing}")

    # Normalize times to ints (ns); some exporters may have strings
    for col in ("start_ns", "end_ns"):
        spans[col] = spans[col].map(_safe_int)

    # ---- RPC interactions: CLIENT ↔ SERVER via parent-child within same trace ----
    clients = spans.loc[spans["span_kind"] == "CLIENT", [
        "trace_id", "span_id", "service_name", "start_ns", "end_ns", "has_links"
    ]].rename(columns={
        "service_name": "src_service",
        "start_ns": "up_start_ns",
        "end_ns": "up_end_ns",
        "has_links": "has_links_up",
    })

    servers = spans.loc[spans["span_kind"] == "SERVER", [
        "trace_id", "parent_span_id", "span_id", "service_name", "start_ns", "end_ns", "has_links"
    ]].rename(columns={
        "service_name": "dst_service",
        "start_ns": "down_start_ns",
        "end_ns": "down_end_ns",
        "has_links": "has_links_down",
    })

    rpc_pairs = servers.merge(
        clients,
        how="inner",
        left_on=["trace_id", "parent_span_id"],
        right_on=["trace_id", "span_id"],
        suffixes=("_srv", "_cli"),
        copy=False,
    )

    rpc_events = pd.DataFrame({
        "src_service": rpc_pairs["src_service"],
        "dst_service": rpc_pairs["dst_service"],
        "kind": "rpc",
        "up_start_ns": rpc_pairs["up_start_ns"],
        "up_end_ns": rpc_pairs["up_end_ns"],
        "down_start_ns": rpc_pairs["down_start_ns"],
        "down_end_ns": rpc_pairs["down_end_ns"],
        "messaging_destination": None,
        "has_links_up": rpc_pairs["has_links_up"],
        "has_links_down": rpc_pairs["has_links_down"],
    })

    # ---- Messaging interactions: PRODUCER ↔ CONSUMER by destination/topic ----
    producers = spans.loc[
        (spans["span_kind"] == "PRODUCER") & spans["messaging_destination"].notna(),
        ["service_name", "start_ns", "end_ns", "has_links", "messaging_destination"],
    ].rename(columns={
        "service_name": "src_service",
        "start_ns": "up_start_ns",
        "end_ns": "up_end_ns",
        "has_links": "has_links_up",
    }).sort_values(["messaging_destination", "up_end_ns"]).reset_index(drop=True)

    consumers = spans.loc[
        (spans["span_kind"] == "CONSUMER") & spans["messaging_destination"].notna(),
        ["service_name", "start_ns", "end_ns", "has_links", "messaging_destination"],
    ].rename(columns={
        "service_name": "dst_service",
        "start_ns": "down_start_ns",
        "end_ns": "down_end_ns",
        "has_links": "has_links_down",
    }).sort_values(["messaging_destination", "down_start_ns"]).reset_index(drop=True)

    msg_events = []
    # Greedy 1:1 matching within each destination (topic/queue)
    for dest, prod_grp in producers.groupby("messaging_destination", sort=False):
        cons_grp = consumers[consumers["messaging_destination"] == dest]
        if cons_grp.empty or prod_grp.empty:
            continue
        i = j = 0
        while i < len(prod_grp) and j < len(cons_grp):
            p_end = prod_grp.iloc[i]["up_end_ns"]
            c_start = cons_grp.iloc[j]["down_start_ns"]
            if p_end is None or c_start is None:
                # Move forward conservatively
                i += 1
                j += 1
                continue
            if c_start >= p_end:
                # Found a plausible consumer after producer → record event
                row = {
                    "src_service": prod_grp.iloc[i]["src_service"],
                    "dst_service": cons_grp.iloc[j]["dst_service"],
                    "kind": "messaging",
                    "up_start_ns": prod_grp.iloc[i]["up_start_ns"],
                    "up_end_ns": prod_grp.iloc[i]["up_end_ns"],
                    "down_start_ns": cons_grp.iloc[j]["down_start_ns"],
                    "down_end_ns": cons_grp.iloc[j]["down_end_ns"],
                    "messaging_destination": dest,
                    "has_links_up": prod_grp.iloc[i]["has_links_up"],
                    "has_links_down": cons_grp.iloc[j]["has_links_down"],
                }
                msg_events.append(row)
                i += 1
                j += 1
            else:
                # Consumer starts before this producer ended → advance consumer pointer
                j += 1

    msg_events_df = pd.DataFrame(msg_events, columns=[
        "src_service", "dst_service", "kind",
        "up_start_ns", "up_end_ns", "down_start_ns", "down_end_ns",
        "messaging_destination", "has_links_up", "has_links_down",
    ])
  
    extra = []
    if emit_broker_edges:
        # Produce producer→kafka
        if not producers.empty:
            for _, p in producers.iterrows():
                extra.append({
                    "src_service": p["src_service"],
                    "dst_service": broker_service_name,
                    "kind": "messaging",
                    "up_start_ns": p["up_start_ns"],
                    "up_end_ns": p["up_end_ns"],
                    # set downstream start=end to avoid NaNs in timing features
                    "down_start_ns": p["up_end_ns"],
                    "down_end_ns": p["up_end_ns"],
                    "messaging_destination": p["messaging_destination"],
                    "has_links_up": p["has_links_up"],
                    "has_links_down": False,
                })
        # Consume kafka→consumer
        if not consumers.empty:
            for _, c in consumers.iterrows():
                extra.append({
                    "src_service": broker_service_name,
                    "dst_service": c["dst_service"],
                    "kind": "messaging",
                    # set upstream end=start to avoid NaNs
                    "up_start_ns": c["down_start_ns"],
                    "up_end_ns": c["down_start_ns"],
                    "down_start_ns": c["down_start_ns"],
                    "down_end_ns": c["down_end_ns"],
                    "messaging_destination": c["messaging_destination"],
                    "has_links_up": False,
                    "has_links_down": c["has_links_down"],
                })

    broker_events_df = pd.DataFrame(extra, columns=[
        "src_service","dst_service","kind","up_start_ns","up_end_ns","down_start_ns","down_end_ns",
        "messaging_destination","has_links_up","has_links_down",
    ])

    # ---- Concatenate events, then aggregate to edges ----
    parts = [df for df in (rpc_events, msg_events_df, broker_events_df) if not df.empty]
    events_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[...])

    # ---- Concatenate events, then aggregate to edges ----
    events_df = pd.concat([rpc_events, msg_events_df], ignore_index=True) if not rpc_events.empty else msg_events_df
    if events_df.empty:
        # Return empty frames with expected columns
        events_df = pd.DataFrame(columns=[
            "src_service", "dst_service", "kind",
            "up_start_ns", "up_end_ns", "down_start_ns", "down_end_ns",
            "messaging_destination", "has_links_up", "has_links_down",
        ])

    # Aggregate to edges
    def _sum_bool(col: pd.Series) -> int:
        return int(col.fillna(False).astype(bool).sum())

    grp = events_df.groupby(["src_service", "dst_service"], as_index=False)
    edges_df = grp.agg(
        n_events=("kind", "count"),
        n_rpc=("kind", lambda s: int((s == "rpc").sum())),
        n_messaging=("kind", lambda s: int((s == "messaging").sum())),
        n_with_links=("has_links_up", _sum_bool),
    )
    # Add link info from downstream side too
    edges_df["n_with_links"] += grp["has_links_down"].agg(_sum_bool)["has_links_down"].values

    return EventsAndEdges(events=events_df, edges=edges_df)
