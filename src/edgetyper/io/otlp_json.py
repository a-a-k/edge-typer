"""
OTLP-JSON trace reader (compatible with the OpenTelemetry Collector 'file' exporter).

- Accepts files containing one or more concatenated JSON documents.
- Flattens Resource attributes and Span attributes.
- Normalizes span kinds to {'CLIENT','SERVER','PRODUCER','CONSUMER','INTERNAL','UNSPECIFIED'}.
- Returns a pandas.DataFrame with one row per span and selected attributes.

References:
  - OTLP JSON mapping (Protobuf JSON for TracesData)
  - OTel span kinds (CLIENT/SERVER/PRODUCER/CONSUMER)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Tuple

import json
import orjson
import pandas as pd


# --- Span kind normalization ----------------------------------------------------

_NUMERIC_KIND_TO_NAME = {
    0: "UNSPECIFIED",
    1: "INTERNAL",
    2: "SERVER",
    3: "CLIENT",
    4: "PRODUCER",
    5: "CONSUMER",
}

_STR_KIND_ALIASES = {
    "SPAN_KIND_UNSPECIFIED": "UNSPECIFIED",
    "SPAN_KIND_INTERNAL": "INTERNAL",
    "SPAN_KIND_SERVER": "SERVER",
    "SPAN_KIND_CLIENT": "CLIENT",
    "SPAN_KIND_PRODUCER": "PRODUCER",
    "SPAN_KIND_CONSUMER": "CONSUMER",
    # Sometimes exporters emit bare names:
    "UNSPECIFIED": "UNSPECIFIED",
    "INTERNAL": "INTERNAL",
    "SERVER": "SERVER",
    "CLIENT": "CLIENT",
    "PRODUCER": "PRODUCER",
    "CONSUMER": "CONSUMER",
}


def _norm_span_kind(kind_value: Any) -> str:
    if isinstance(kind_value, int):
        return _NUMERIC_KIND_TO_NAME.get(kind_value, "UNSPECIFIED")
    if isinstance(kind_value, str):
        return _STR_KIND_ALIASES.get(kind_value.upper(), "UNSPECIFIED")
    return "UNSPECIFIED"


# --- Attribute flattening -------------------------------------------------------

def _attr_value_to_python(v: Dict[str, Any]) -> Any:
    """
    Convert an OTel AttributeValue (oneof) to a native Python value.
    Example forms:
      {"stringValue": "checkout"}
      {"intValue": "1"}         # note: numbers often encoded as strings in JSON
      {"boolValue": true}
      {"doubleValue": 1.23}
      {"arrayValue": {"values": [ ... ]}}
      {"kvlistValue": {"values": [{"key":"k","value":{...}}, ...]}}
    """
    if not isinstance(v, dict):
        return v
    if "stringValue" in v:
        return v["stringValue"]
    if "boolValue" in v:
        return bool(v["boolValue"])
    if "intValue" in v:
        # Protobuf JSON may serialize integers as strings
        try:
            return int(v["intValue"])
        except Exception:
            return v["intValue"]
    if "doubleValue" in v:
        try:
            return float(v["doubleValue"])
        except Exception:
            return v["doubleValue"]
    if "arrayValue" in v and isinstance(v["arrayValue"], dict):
        return [_attr_value_to_python(item) for item in v["arrayValue"].get("values", [])]
    if "kvlistValue" in v and isinstance(v["kvlistValue"], dict):
        out = {}
        for kv in v["kvlistValue"].get("values", []):
            k = kv.get("key")
            out[k] = _attr_value_to_python(kv.get("value"))
        return out
    return v


def _flatten_attributes(attrs: Any) -> Dict[str, Any]:
    """
    Handle both canonical list-of-kv and already-flattened dict forms.
    """
    flat: Dict[str, Any] = {}
    if isinstance(attrs, list):
        for item in attrs:
            k = item.get("key")
            v = _attr_value_to_python(item.get("value"))
            if k is not None:
                flat[k] = v
    elif isinstance(attrs, dict):
        # Some exporters map directly to dict
        for k, v in attrs.items():
            flat[k] = v
    return flat


def _extract_service_name(resource: Dict[str, Any], default_attr: str = "service.name") -> str:
    # Resource may look like: {"attributes":[{"key":"service.name","value":{"stringValue":"frontend"}}, ...]}
    attrs = _flatten_attributes(resource.get("attributes", {}))
    # Prefer the provided key; fall back to common alternates.
    for key in (default_attr, "telemetry.sdk.name", "service"):
        if key in attrs and isinstance(attrs[key], (str, int, float, bool)):
            return str(attrs[key])
    return "unknown"


# --- JSON document iteration ----------------------------------------------------

def _iter_json_documents(path: Path) -> Generator[Dict[str, Any], None, None]:
    """
    Yield one JSON object at a time from a file that may contain either:
      - a single large JSON document, or
      - multiple concatenated JSON documents (no commas between), or
      - one JSON per line (NDJSON-like).
    """
    data = path.read_bytes()

    # First, try a single-shot fast parse via orjson.
    try:
        obj = orjson.loads(data)
        if isinstance(obj, dict):
            yield obj
            return
        if isinstance(obj, list):
            # Some exporters might write a list; handle that too.
            for item in obj:
                if isinstance(item, dict):
                    yield item
            return
    except Exception:
        pass

    # Fallback: use stdlib JSONDecoder.raw_decode to peel off one object at a time.
    txt = data.decode("utf-8", errors="ignore")
    dec = json.JSONDecoder()
    i = 0
    n = len(txt)
    while i < n:
        # Skip whitespace and newlines
        while i < n and txt[i].isspace():
            i += 1
        if i >= n:
            break
        try:
            obj, end = dec.raw_decode(txt, i)
        except json.JSONDecodeError:
            # Try line-based as last resort
            line_end = txt.find("\n", i)
            if line_end == -1:
                break
            i = line_end + 1
            continue
        if isinstance(obj, dict):
            yield obj
        i = end


# --- Public API ----------------------------------------------------------------

def read_otlp_json(path: Path, service_attr_key: str = "service.name") -> pd.DataFrame:
    """
    Read an OTLP-JSON file (Collector 'file' exporter) and return a DataFrame of spans.

    Columns:
      - trace_id, span_id, parent_span_id
      - service_name, span_name, span_kind
      - start_ns, end_ns, duration_ns
      - has_links, links_count
      - messaging_system, messaging_destination, messaging_operation
      - http_method, rpc_system
    """
    rows: List[Dict[str, Any]] = []

    for doc in _iter_json_documents(path):
        # Expected top-level envelope: TracesData -> resourceSpans[]
        resource_spans = doc.get("resourceSpans", [])
        for r in resource_spans:
            service_name = _extract_service_name(r.get("resource", {}), default_attr=service_attr_key)
            scope_spans = r.get("scopeSpans") or r.get("instrumentationLibrarySpans") or []
            for s in scope_spans:
                spans = s.get("spans", [])
                for sp in spans:
                    attrs = _flatten_attributes(sp.get("attributes", {}))
                    links = sp.get("links", []) or []
                    start_ns = _parse_unix_nano(sp.get("startTimeUnixNano"))
                    end_ns = _parse_unix_nano(sp.get("endTimeUnixNano"))
                    duration_ns = end_ns - start_ns if (end_ns and start_ns) else None

                    rows.append(
                        {
                            "trace_id": sp.get("traceId"),
                            "span_id": sp.get("spanId"),
                            "parent_span_id": sp.get("parentSpanId"),
                            "service_name": service_name,
                            "span_name": sp.get("name"),
                            "span_kind": _norm_span_kind(sp.get("kind")),
                            "start_ns": start_ns,
                            "end_ns": end_ns,
                            "duration_ns": duration_ns,
                            "has_links": bool(links),
                            "links_count": len(links),
                            # Selected attributes commonly used downstream:
                            "messaging_system": attrs.get("messaging.system"),
                            "messaging_destination": attrs.get("messaging.destination")
                            or attrs.get("messaging.destination.name")
                            or attrs.get("messaging.kafka.topic"),
                            "messaging_operation": attrs.get("messaging.operation")
                            or attrs.get("messaging.operation.name"),
                            "http_method": attrs.get("http.method"),
                            "rpc_system": attrs.get("rpc.system"),
                        }
                    )

    df = pd.DataFrame.from_records(rows)
    # Basic cleanup
    if not df.empty:
        # Normalize IDs to lowercase hex strings if present
        for col in ("trace_id", "span_id", "parent_span_id"):
            if col in df.columns:
                df[col] = df[col].astype("string").str.lower()
        # Service name as string
        df["service_name"] = df["service_name"].astype("string")
        df["span_kind"] = df["span_kind"].astype("category")
    return df


def _parse_unix_nano(value: Any) -> int | None:
    """
    Convert various Protobuf-JSON representations of time to integer nanoseconds.
    Accepts int, str-encoded int, or None.
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        # Some exporters emit as string; tolerate separators/spaces
        s = value.strip()
        try:
            return int(s)
        except Exception:
            return None
    # Unexpected type
    return None
