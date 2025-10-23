import argparse, json, time
from datetime import timedelta
import requests

def fetch_services(base_url: str) -> list[str]:
    r = requests.get(f"{base_url}/api/services", timeout=20)
    r.raise_for_status()
    return sorted(r.json().get("data", []))

def fetch_traces(base_url: str, service: str, lookback: str, limit: int) -> list[dict]:
    # Example: /api/traces?service=frontend&lookback=30m&limit=200
    params = {"service": service, "lookback": lookback, "limit": str(limit)}
    r = requests.get(f"{base_url}/api/traces", params=params, timeout=40)
    r.raise_for_status()
    return r.json().get("data", [])

def flatten_jaeger(data: list[dict]) -> list[dict]:
    """Flatten Jaeger traces into per-span records with essential fields."""
    out = []
    for trace in data:
        spans = trace.get("spans", [])
        procs = trace.get("processes", {})
        for s in spans:
            pid = s.get("processID")
            proc = procs.get(pid, {})
            svc = proc.get("serviceName")
            tags = {t["key"]: t.get("value") for t in s.get("tags", []) if "key" in t}
            refs = [{"refType": r.get("refType"), "spanID": r.get("spanID")} for r in s.get("references", [])]
            out.append({
                "traceId": s.get("traceID"),
                "spanId": s.get("spanID"),
                "parentSpanId": next((r["spanID"] for r in refs if r["refType"] == "CHILD_OF"), None),
                "service": svc,
                "operation": s.get("operationName"),
                "startTimeMicros": s.get("startTime"),
                "durationMicros": s.get("duration"),
                "span.kind": tags.get("span.kind", "").lower(),  # "client","server","producer","consumer"
                "messaging.system": tags.get("messaging.system"),
                "rpc.system": tags.get("rpc.system"),
                "tags": tags,
            })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="e.g., http://localhost:8080/jaeger")
    ap.add_argument("--lookback", default="30m")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    svcs = fetch_services(args.base_url)
    with open(args.out, "w") as f:
        for svc in svcs:
            traces = fetch_traces(args.base_url, svc, args.lookback, args.limit)
            for rec in flatten_jaeger(traces):
                f.write(json.dumps(rec) + "\n")

if __name__ == "__main__":
    main()
