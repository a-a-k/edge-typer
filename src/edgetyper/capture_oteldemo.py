import argparse, json, sys, time
from urllib.parse import urljoin
import requests

def _is_json(r: requests.Response) -> bool:
    ctype = r.headers.get("content-type", "")
    # Accept json or tempo/jaeger compatible responses even if mislabelled
    if "json" in ctype.lower():
        return True
    # Some proxies mislabel; try a light parse to confirm
    try:
        _ = r.json()
        return True
    except Exception:
        return False

def _try_services(base: str, prefix: str, timeout=20):
    """Return (ok, services, full_prefix, debug)."""
    services_url = urljoin(base.rstrip("/") + "/", prefix.strip("/") + "/services")
    r = requests.get(services_url, timeout=timeout)
    dbg = f"GET {services_url} -> {r.status_code}"
    if r.ok and _is_json(r):
        try:
            data = r.json()
            svcs = sorted(data.get("data", []))
            if isinstance(svcs, list) and svcs:
                return True, svcs, prefix.strip("/"), dbg
            # Even if empty, treat as ok to unblock; Jaeger returns {"data": []} until traffic ramps
            return True, svcs, prefix.strip("/"), dbg + " (empty data)"
        except Exception as e:
            return False, [], "", dbg + f" (json error: {e})"
    return False, [], "", dbg + " (non-json or bad status)"

def discover_api_prefix(base: str) -> tuple[str, list[str], list[str]]:
    """
    Try common JSON API roots behind the demo's Envoy:
      1) /jaeger/ui/api  (common in OTel Demo docs UI path)
      2) /jaeger/api     (base-path set on jaeger-query)
      3) /api            (direct jaeger-query without prefix)
    Returns (prefix, services, debug_lines).
    """
    attempts = ["/jaeger/ui/api", "/jaeger/api", "/api"]
    debug = []
    for p in attempts:
        ok, svcs, prefix, dbg = _try_services(base, p)
        debug.append(dbg)
        if ok:
            return prefix, svcs, debug
    raise RuntimeError("No working Jaeger JSON API prefix found.\n" + "\n".join(debug))

def fetch_traces(base: str, prefix: str, service: str, lookback: str, limit: int, timeout=40):
    # /api/traces?service=<svc>&lookback=30m&limit=200
    traces_url = urljoin(base.rstrip("/") + "/", f"{prefix.strip('/')}/traces")
    params = {"service": service, "lookback": lookback, "limit": str(limit)}
    r = requests.get(traces_url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json().get("data", [])

def flatten_jaeger(data: list[dict]) -> list[dict]:
    out = []
    for trace in data or []:
        spans = trace.get("spans", [])
        procs = trace.get("processes", {})
        for s in spans:
            pid = s.get("processID")
            proc = procs.get(pid, {})
            svc = proc.get("serviceName")
            tags = {t.get("key"): t.get("value") for t in s.get("tags", []) if "key" in t}
            refs = [{"refType": r.get("refType"), "spanID": r.get("spanID")} for r in s.get("references", [])]
            out.append({
                "traceId": s.get("traceID"),
                "spanId": s.get("spanID"),
                "parentSpanId": next((r["spanID"] for r in refs if r["refType"] == "CHILD_OF"), None),
                "service": svc,
                "operation": s.get("operationName"),
                "startTimeMicros": s.get("startTime"),
                "durationMicros": s.get("duration"),
                "span.kind": str(tags.get("span.kind", "")).lower(),
                "messaging.system": tags.get("messaging.system"),
                "rpc.system": tags.get("rpc.system"),
                "tags": tags,
            })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="Root, e.g. http://localhost:8080")
    ap.add_argument("--lookback", default="30m")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # 1) Discover API prefix & services
    prefix, services, debug = discover_api_prefix(args.base_url)
    print("[capture] API discovery:", *debug, sep="\n  ")
    if not services:
        # No services yet; traffic may still be warming up. Keep a short grace period.
        print("[capture] No services returned yet; sleeping 15s and retrying service list...")
        time.sleep(15)
        _, services, debug2 = discover_api_prefix(args.base_url)
        print("[capture] retry:", *debug2, sep="\n  ")

    # 2) Pull traces per service and write JSONL
    import json as _json
    count = 0
    with open(args.out, "w") as f:
        for svc in services or ["frontend", "checkout", "product-catalog", "kafka", "accounting", "fraud-detection"]:
            data = fetch_traces(args.base_url, prefix, svc, args.lookback, args.limit)
            for rec in flatten_jaeger(data):
                f.write(_json.dumps(rec) + "\n")
                count += 1

    print(f"[capture] wrote {count} spans to {args.out} via prefix=/{prefix}")

if __name__ == "__main__":
    main()
