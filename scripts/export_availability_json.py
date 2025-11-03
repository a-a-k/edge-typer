#!/usr/bin/env python3
# scripts/export_availability_json.py
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import pandas as pd

def load_csv(path: Path, cols: list[str]) -> pd.DataFrame:
    if not path or not path.exists():
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df.loc[:, cols].copy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--typed", required=True, type=Path)      # availability_typed.csv
    ap.add_argument("--blocking", required=True, type=Path)   # availability_block.csv
    ap.add_argument("--live", type=Path)                      # live_availability.csv (optional)
    ap.add_argument("--out", required=True, type=Path)        # out/availability.json
    args = ap.parse_args()

    t = load_csv(args.typed,    ["entrypoint","p_fail","R_model"])
    b = load_csv(args.blocking, ["entrypoint","p_fail","R_model"])
    l = load_csv(args.live,     ["entrypoint","p_fail","R_live"]) if args.live else pd.DataFrame(columns=["entrypoint","p_fail","R_live"])

    # Canonicalize types
    for df in (t,b):
        df["entrypoint"] = df["entrypoint"].astype(str)
        df["p_fail"] = pd.to_numeric(df["p_fail"], errors="coerce")
        df["R_model"] = pd.to_numeric(df["R_model"], errors="coerce")
    if not l.empty:
        l["entrypoint"] = l["entrypoint"].astype(str)
        l["p_fail"] = pd.to_numeric(l["p_fail"], errors="coerce")
        l["R_live"] = pd.to_numeric(l["R_live"], errors="coerce")

    # Build output
    out = {
        "schema": "edgetyper-availability@v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "p_fail_grid": sorted(pd.unique(pd.concat([t["p_fail"], b["p_fail"], l["p_fail"] if not l.empty else pd.Series([])])).tolist()),
        "model": {
            "typed": t.rename(columns={"R_model":"R"}).to_dict(orient="records"),
            "all_blocking": b.rename(columns={"R_model":"R"}).to_dict(orient="records"),
        },
        "live": l.to_dict(orient="records") if not l.empty else [],
    }

    # Joined convenience table with absolute errors (if live is present)
    if not l.empty:
        J = (t.rename(columns={"R_model":"R_model_typed"})
               .merge(b.rename(columns={"R_model":"R_model_block"}), on=["entrypoint","p_fail"], how="outer")
               .merge(l, on=["entrypoint","p_fail"], how="inner"))
        if not J.empty:
            J["err_typed"]  = (J["R_model_typed"] - J["R_live"]).abs()
            J["err_block"]  = (J["R_model_block"] - J["R_live"]).abs()
            out["joined"] = J.to_dict(orient="records")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[export_availability_json] wrote {args.out}")

if __name__ == "__main__":
    main()
