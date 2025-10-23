import argparse, csv, json
from collections import defaultdict

def load_spans(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def build_edges(spans):
    # Index spans by (traceId, spanId)
    by_id = {}
    traces = defaultdict(list)
    for s in spans:
        by_id[(s["traceId"], s["spanId"])] = s
        traces[s["traceId"]].append(s)

    # Derive caller->callee service edges from CHILD_OF relationships
    edges = defaultdict(lambda: {"count":0, "evidence":[]})
    for t, group in traces.items():
        for s in group:
            par = s.get("parentSpanId")
            if not par:
                continue
            parent = by_id.get((s["traceId"], par))
            if not parent:
                continue
            caller = parent.get("service")
            callee = s.get("service")
            if not caller or not callee or caller == callee:
                continue

            # Timing overlap evidence (needs parent+child window)
            ps, pd = parent.get("startTimeMicros"), parent.get("durationMicros")
            cs, cd = s.get("startTimeMicros"), s.get("durationMicros")
            overlap_ratio = None
            if all(isinstance(x, int) for x in [ps, pd, cs, cd]):
                p1, p2 = ps, ps + pd
                c1, c2 = cs, cs + cd
                inter = max(0, min(p2,c2) - max(p1,c1))
                overlap_ratio = inter / max(cd, 1)

            # SemConv evidence
            semconv_async = (s.get("span.kind") in ("producer","consumer")
                             or parent.get("span.kind") in ("producer","consumer")
                             or s.get("messaging.system") is not None
                             or parent.get("messaging.system") is not None)

            key = (caller, callee)
            rec = edges[key]
            rec["count"] += 1
            rec["evidence"].append({"semconv_async": semconv_async, "overlap_ratio": overlap_ratio})
    return edges

def decide_label(evidences, mode: str):
    # Simple, transparent rule:
    #  - SemConv-only: async if any semconv_async True
    #  - Timing-only:  async if median overlap_ratio < 0.1
    #  - Ours:         async if (any semconv_async) or (median overlap_ratio < 0.1)
    import statistics
    ov = [e["overlap_ratio"] for e in evidences if e["overlap_ratio"] is not None]
    med_ov = statistics.median(ov) if ov else None
    sem = any(e["semconv_async"] for e in evidences)

    if mode == "semconv":
        return "async" if sem else "blocking", med_ov, sem
    if mode == "timing":
        return ("async" if (med_ov is not None and med_ov < 0.10) else "blocking"), med_ov, sem
    # ours
    return ("async" if sem or (med_ov is not None and med_ov < 0.10) else "blocking"), med_ov, sem

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["ours","semconv","timing"], default="ours")
    args = ap.parse_args()

    edges = build_edges(load_spans(args.spans))

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src","dst","label","events","median_overlap","any_semconv"])
        for (src, dst), rec in sorted(edges.items()):
            label, med_ov, sem = decide_label(rec["evidence"], args.mode)
            w.writerow([src, dst, label, rec["count"], med_ov if med_ov is not None else "", int(sem)])

if __name__ == "__main__":
    main()
