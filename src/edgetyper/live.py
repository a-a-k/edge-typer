from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import re
import yaml

_ENTRYPOINT_PREFIX = re.compile(r"entry:([^:]+):")


def _normalize_cols(df: pd.DataFrame) -> dict[str, str]:
    return {
        c.lower().strip().replace(" ", "").replace("_", "").replace("#", ""): c
        for c in df.columns
    }


def _load_entrypoint_filter(path: Optional[Path]) -> Optional[set[str]]:
    if not path:
        return None
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        col = "entrypoint" if "entrypoint" in df.columns else df.columns[0]
        vals = [str(x).strip() for x in df[col].dropna().tolist() if str(x).strip()]
        return set(vals) if vals else None
    except Exception:
        text = path.read_text(encoding="utf-8")
        vals = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return set(vals) if vals else None


def _load_rules(targets_yaml: Optional[Path]) -> list[tuple[str, re.Pattern[str], Optional[str]]]:
    if not targets_yaml or not targets_yaml.exists():
        return []
    cfg = yaml.safe_load(targets_yaml.read_text()) or {}
    rules: list[tuple[str, re.Pattern[str], Optional[str]]] = []
    for ep, plist in (cfg.get("entrypoints") or {}).items():
        for rule in (plist or []):
            pat = rule.get("name_regex") or rule.get("re")
            if not pat:
                continue
            meth = rule.get("method")
            try:
                cre = re.compile(str(pat))
            except Exception:
                cre = re.compile(str(pat), re.I)
            rules.append((str(ep), cre, (str(meth).upper() if meth else None)))
    return rules


def compute_live_availability(
    stats_path: Path,
    failures_path: Optional[Path] = None,
    targets_yaml: Optional[Path] = None,
    entrypoints_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, list[str], Counter]:
    """
    Return (df, missing_entrypoints, unmatched_requests) where df has columns:
    entrypoint, R_live, n_total, n_5xx, n_timeout, n_socket.
    """
    if not stats_path.exists():
        raise FileNotFoundError(f"Locust stats CSV not found: {stats_path}")

    stats = pd.read_csv(stats_path)
    cols = _normalize_cols(stats)
    name_col = cols.get("name", "Name")
    method_col = cols.get("method")
    req_col = (
        cols.get("requests")
        or cols.get("requestcount")
        or cols.get("totalrequestcount")
        or cols.get("requests")
        or cols.get("count")
    )
    if req_col is None:
        numeric_cols = [c for c in stats.columns if pd.api.types.is_numeric_dtype(stats[c])]
        req_col = numeric_cols[0] if numeric_cols else stats.columns[0]

    df_stats = stats.copy()
    df_stats[name_col] = df_stats[name_col].astype(str)
    df_stats = df_stats[~df_stats[name_col].str.contains("Aggregated|Total", case=False, na=False)]

    # Failures CSV (optional)
    if failures_path and failures_path.exists():
        fails = pd.read_csv(failures_path)
        fcols = _normalize_cols(fails)
        f_name = fcols.get("name", "Name")
        f_method = fcols.get("method")
        f_err = fcols.get("error", "Error")
        f_occ = fcols.get("occurrences") or fcols.get("count") or "Occurrences"

        def _bucket(msg: object) -> str:
            s = str(msg)
            if re.search(r"\b5\d\d\b", s) or re.search(r"status\s*code\s*5\d\d", s, re.I):
                return "5xx"
            if re.search(r"timeout|timed\s*out|readtimeout|connecttimeout", s, re.I):
                return "timeout"
            if re.search(r"connection|refused|reset|broken\s*pipe|socket|dns|ssl|remote|protocol", s, re.I):
                return "socket"
            return "other"

        fails[f_name] = fails[f_name].astype(str)
        fails["_bucket"] = fails[f_err].map(_bucket)
        keys = [f_name]
        if f_method and f_method in fails.columns:
            fails[f_method] = fails[f_method].astype(str)
            keys.insert(0, f_method)
        pivot = (
            fails.pivot_table(index=keys, columns="_bucket", values=f_occ, aggfunc="sum", fill_value=0)
            .reset_index()
            .rename(columns={"5xx": "n_5xx", "timeout": "n_timeout", "socket": "n_socket"})
        )
    else:
        pivot = pd.DataFrame(columns=[name_col, "n_5xx", "n_timeout", "n_socket"])
        if method_col:
            pivot[method_col] = ""

    rules = _load_rules(targets_yaml)
    eps_filter = _load_entrypoint_filter(entrypoints_path)

    merge_keys = [name_col]
    if method_col and method_col in df_stats.columns and method_col in pivot.columns:
        merge_keys.insert(0, method_col)
    dfm = df_stats.merge(pivot, on=merge_keys, how="left").fillna({"n_5xx": 0, "n_timeout": 0, "n_socket": 0})

    acc: dict[str, dict[str, float]] = {}
    unmatched = Counter()
    matched_eps: set[str] = set()

    for _, row in dfm.iterrows():
        name = str(row[name_col])
        method = str(row.get(method_col, "")).upper() if method_col else ""
        m = _ENTRYPOINT_PREFIX.match(name)
        ep = m.group(1) if m else None
        if ep is None and rules:
            for ep_name, cre, meth in rules:
                if meth and meth != method:
                    continue
                if cre.search(name):
                    ep = ep_name
                    break
        if eps_filter is not None and ep and ep not in eps_filter:
            continue
        if ep is None:
            total_requests = float(pd.to_numeric(row.get(req_col), errors="coerce") or 0.0)
            unmatched[name] += max(1, int(total_requests)) if total_requests > 0 else 1
            continue

        n_total = float(pd.to_numeric(row[req_col], errors="coerce") or 0.0)
        if n_total <= 0:
            continue
        n_5xx = float(row.get("n_5xx", 0.0) or 0.0)
        n_to = float(row.get("n_timeout", 0.0) or 0.0)
        n_so = float(row.get("n_socket", 0.0) or 0.0)
        slot = acc.setdefault(ep, {"n_total": 0.0, "n_5xx": 0.0, "n_timeout": 0.0, "n_socket": 0.0})
        slot["n_total"] += n_total
        slot["n_5xx"] += n_5xx
        slot["n_timeout"] += n_to
        slot["n_socket"] += n_so
        matched_eps.add(ep)

    rows = []
    for ep, vals in acc.items():
        bad = vals["n_5xx"] + vals["n_timeout"] + vals["n_socket"]
        total = vals["n_total"]
        R = 0.0 if total <= 0 else max(0.0, min(1.0, 1.0 - (bad / total)))
        rows.append({
            "entrypoint": ep,
            "R_live": float(R),
            "n_total": int(total),
            "n_5xx": int(vals["n_5xx"]),
            "n_timeout": int(vals["n_timeout"]),
            "n_socket": int(vals["n_socket"]),
        })

    missing = sorted(eps_filter - matched_eps) if eps_filter else []
    return pd.DataFrame(rows), missing, unmatched
