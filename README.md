# EdgeTyper — Typed dependency graphs for resilience experiments

> **Goal.** Test whether a **typed** service‑dependency graph (edges labeled *BLOCKING* vs *ASYNC*) predicts a system’s resilience/availability more accurately than an **untyped** (all‑blocking) graph, while keeping data collection and computation lightweight.

This repository contains:

* a CLI (`edgetyper`) that turns traces into a typed service graph, builds chaos plans, and runs a Monte‑Carlo availability estimator; 
* an aggregator (`scripts/aggregate_replicas.py`) that merges per‑replica outputs into a static site;
* an optional live pipeline that **post‑processes Locust CSV files** to compute live availability on the *same* grid (entrypoint × failure rate) as the model estimates, enabling a direct, apples‑to‑apples comparison.

The availability estimator follows the **topological model** and **live success definition** described in the attached study:

* model = **blocking connectivity + replication**, estimated via Monte‑Carlo reachability (Algorithm1);
* live success rate = (1 - \frac{#5xx + #socket\ errors + #timeouts}{#total\ requests}) under fixed‑rate windows;
* failure fractions (p_{\text{fail}} \in {0.1,0.3,0.5,0.7,0.9}). 

---

## Table of contents

1. [Quick start](#quick-start)
2. [Pipeline overview](#pipeline-overview)
3. [CLI commands](#cli-commands)
4. [Live availability from Locust CSV](#live-availability-from-locust-csv)
5. [Aggregating replicas into a site](#aggregating-replicas-into-a-site)
6. [Data contracts (file formats)](#data-contracts-file-formats)
7. [Interpreting results](#interpreting-results)
8. [Design notes & hypothesis test](#design-notes--hypothesis-test)
9. [Troubleshooting](#troubleshooting)
10. [License & citation](#license--citation)

---

## Quick start

Below is the **minimal end‑to‑end** sequence for one replica. It assumes you have OTLP‑JSON traces (`spans.json`) and a running target for Locust. All paths are examples; adjust as needed.

```bash
# 0) Setup
python -m pip install -e .            # install edgetyper (CLI)
mkdir -p out                          # workspace

# 1) Parse traces → Parquet
edgetyper extract \
  --input traces/spans.json \
  --out out/spans.parquet                                    

# 2) Build service graph (events + edges)
edgetyper graph \
  --spans out/spans.parquet \
  --out-events out/events.parquet \
  --out-edges  out/edges.parquet                             

# 3) Featurize edges
edgetyper featurize \
  --events out/events.parquet \
  --edges  out/edges.parquet \
  --out    out/features.parquet                              

# 4) Label edges (rules+ML fallback)
edgetyper label \
  --features out/features.parquet \
  --out out/pred_ours.csv                                   

# 5) Chaos plans (typed & all‑blocking)
edgetyper plan --edges out/edges.parquet --pred out/pred_ours.csv --out out/plan_physical.csv
edgetyper plan --edges out/edges.parquet --pred out/pred_ours.csv --out out/plan_all_blocking.csv --assume-all-blocking 

# 6) Monte‑Carlo availability (typed & all‑blocking)
edgetyper resilience --edges out/edges.parquet --pred out/pred_ours.csv --out out/availability_typed.csv
edgetyper resilience --edges out/edges.parquet --pred out/pred_ours.csv --out out/availability_block.csv --assume-all-blocking 

# 7) Live windows with Locust (fixed‑rate, multiple p_fail)
#    For each p in 0.1 0.3 0.5 0.7 0.9: run Locust with --csv <prefix>, then post‑process:
edgetyper availability-live \
  --locust-prefix out/locust_0.3 \
  --targets out/live_targets.yaml \
  --entrypoints out/entrypoints.txt \
  --p-fail 0.3 \
  --out out/live_availability.csv --append                  

# 8) Aggregate one or more replicas into a site (availability‑only mode)
python scripts/aggregate_replicas.py --replicas-dir runs --outdir site  # set AVAIL_ONLY=1 to hide rank/CI blocks
```

**Why these steps:** the estimator implements **Algorithm1** (Monte‑Carlo reachability + replication) and live success follows the **5xx/timeout/socket** definition; both come from the study and enable like‑for‑like comparison. 

---

## Pipeline overview

**Discovery → Typing → Planning → Estimation → Live → Aggregation**

* **Discovery/Graph**: extract service interactions from traces and render a directed service‑level graph.
* **Typing**: classify each edge as `ASYNC` or `BLOCKING` (rules with ML fallback), producing `pred_ours.csv`.
* **Plan**: compute IBS/DBS per target under typed vs all‑blocking assumptions.
* **Resilience (model)**: simulate **endpoint availability** over failure fractions (p_{\text{fail}}), accounting for replication; output per‑entrypoint `R_model`. 
* **Live**: drive fixed‑rate windows; from Locust CSVs, compute (R_{\text{live}}) on the same grid (entrypoint × (p_{\text{fail}})). 
* **Aggregate**: average across replicas, compare model vs live (MAE & win‑rate), and publish a static site.

---

## CLI commands

### `edgetyper extract`

Parse **OTLP‑JSON** into `spans.parquet`. Inputs must be Jaeger/OTel JSON with standard fields. 

### `edgetyper graph`

Builds **events** and **service edges** Parquet files. Options support adding producer→broker edges when desired. 

### `edgetyper featurize`

Computes feature tables combining **SemConv** and **timing** signals; optional masking lets you ablate a signal to probe robustness. 

### `edgetyper label`

Generates `pred_ours.csv` with `pred_label ∈ {async,sync,uncertain}` and a score; has an `--uncertain-threshold` to keep low‑confidence edges as `uncertain`. 

### `edgetyper eval`

Benchmarks predictions against **ground truth** (CSV or YAML) and emits `metrics_*.json` (classification report, confusion matrix, counts). Use `edgetyper report` to render a metrics site. 

### `edgetyper plan`

Computes **IBS** (blocking upstream) and **DBS** (async ingress) per target; supports `--assume-all-blocking` to build the baseline plan. Outputs CSV sorted by IBS/DBS. 

### `edgetyper resilience`

**Monte‑Carlo availability** estimator (Algorithm1). Inputs: `edges.parquet`, `pred_ours.csv`, optional replicas and entrypoints; outputs `availability_*.csv`. Use `--assume-all-blocking` to produce the untyped baseline. 

> **Model (Algorithm1).** For each sample: independently fail replicas with probability (p_{\text{fail}}); a service is alive if ≥1 replica survives; a request to entrypoint `e` succeeds if there exists a path of **blocking** edges from `e` to leaves **within the alive subgraph**. Repeat and average. 

### `edgetyper observe`

Segments trace windows and emits `observations.json` (counts per service and per kind) for sanity‑checking live runs; used by the older “rank‑vs‑live” view in `edgetyper report`. 

### `edgetyper report`

Renders a **metrics/coverage** site from `metrics_*.json` with optional dataset snapshot, coverage vs ground truth, chaos plan preview, and live sanity. (Independent of the availability aggregator.) 

### `edgetyper availability-live`  *(added for live comparison without changing Locust)*

Post‑processes **Locust CSV** outputs into `live_availability.csv` using the **5xx + timeouts + socket** definition from the study. You pass a `--locust-prefix` (the same you used with `locust --csv`), and optional `--targets` YAML to map Locust `Name` to entrypoints; optionally filter with `--entrypoints`. Each run appends rows for a specific `--p-fail`. 

---

## Live availability from Locust CSV

You do **not** need to modify `locustfile.py`. The flow is:

1. For each (p_{\text{fail}}) in `{0.1,0.3,0.5,0.7,0.9}`, inject failures (kill a fraction of replicas) and run a **fixed‑rate** Locust window (`--csv <prefix>`, `--csv-full-history`, constant `-R`, fixed `-d`) as in the study. 
2. Run `edgetyper availability-live` with:

   * `--locust-prefix <prefix>` (reads `<prefix>_stats.csv` and `<prefix>_failures.csv`),
   * `--targets live_targets.yaml` (maps Locust `Name` to entrypoint),
   * `--entrypoints entrypoints.txt` (optional; keep model & live in sync),
   * `--p-fail <value>`, `--out live_availability.csv --append`.

> **Want per-endpoint fidelity?** Run `edgetyper entrypoints-from-locust --locust-prefix <prefix> --out-entrypoints entrypoints.csv --out-targets live_targets.yaml`. This emits a 1:1 mapping for every `Name` in the Locust CSV, so both the live pipeline and the Monte-Carlo model operate on the exact same endpoint grid. The GitHub Actions workflows call this command automatically, so the “fire-and-forget” run already snapshots exact endpoints for each replica.
> Entries with fewer than 1 request in the Locust stats are skipped to avoid expecting live data that never arrives.

> **Sampling guardrail (CI).** In the matrix workflow we pass `--min-requests 25` to drop endpoints that only ever see a handful of Locust calls; otherwise the validation step fails because there is no live signal to compare against. Adjust the threshold if your workload drives different traffic volumes.

**Example `live_targets.yaml`:**

```yaml
entrypoints:
  frontend:
    - name_regex: "^/$|^/api/.*"
  payments:
    - name_regex: "^/payments"
```

> **Note.** `src/edgetyper/targets.yaml` ships with a catch‑all mapping for the OpenTelemetry Demo. The CLI now warns when Locust requests fail to match your entrypoint rules (and errors if nothing matches), so you can adjust the YAML/filters before aggregating.

**What the command computes:** for each mapped entrypoint, it sums `#Requests` and categorizes failures from `_failures.csv` into **5xx**, **timeout**, **socket**. It writes one row per `(entrypoint, p_fail)` with:

```
entrypoint,p_fail,R_live,n_total,n_5xx,n_timeout,n_socket
```

where (R_{\text{live}} = 1 - \frac{(\text{5xx}+\text{timeout}+\text{socket})}{\text{total}}). 

---

## Aggregating replicas into a site

Use the Python script (not the `report` command) to build a **lightweight availability site**. Two modes:

* **Availability‑only** (recommended while validating live): set `AVAIL_ONLY=1`.
  The page shows **pooled Monte‑Carlo availability** (typed vs all‑blocking). If each `replicate-*` directory also contains `live_availability.csv`, the page additionally shows **“Availability: model vs live (pooled)”** with MAE by `p_fail`, overall MAE, and **win‑rate** (share of cells where typed is closer to live).
  The aggregator also surfaces **label coverage (async vs uncertain)**, an **Interpretation** paragraph that states whether typed edges beat all-blocking (with MAE/win-rate deltas), and bootstrap **95% CIs** for all reported metrics.

```bash
# Aggregate N replica folders (replicate-1/, replicate-2/, …) into `site/`
AVAIL_ONLY=1 python scripts/aggregate_replicas.py \
  --replicas-dir runs \
  --outdir site
```

Outputs under `site/`:

* `index.html`, plus CSVs in `site/data/`:
  `availability_typed_pooled.csv`, `availability_block_pooled.csv`,
  if live present: `availability_join_pooled.csv`, `availability_errors_by_p.csv`,
  `availability_errors_overall.csv`, `availability_errors_by_replica.csv`.

---

## Data contracts (file formats)

**`edges.parquet`** (from `edgetyper graph`) — service edges with volumes (events/rpc/messaging). 

**`features.parquet`** — feature vectors per edge (SemConv, timing). 

**`pred_ours.csv`** — `src_service,dst_service,pred_label,pred_score`. 

**`plan_physical.csv` / `plan_all_blocking.csv`** — per‑target `IBS,DBS,…`. 

**`availability_typed.csv` / `availability_block.csv`**

```
entrypoint,p_fail,R_model
Frontend,0.3,0.305
...
```

(Produced by `edgetyper resilience`.) 

**`live_availability.csv`**

```
entrypoint,p_fail,R_live,n_total,n_5xx,n_timeout,n_socket
Frontend,0.3,0.300,10000,50,15,5
...
```

(Produced by `edgetyper availability-live`; definition per the study.) 

---

## Interpreting results

* **Monte‑Carlo availability (pooled)** compares model outputs `R_model` averaged over replicas:
  look for systematic gaps between **typed** and **all‑blocking** lines; small gaps suggest that ASYNC edges rarely constrain end‑to‑end reachability on critical cuts.

* **Model vs live (pooled)** compares `R_model` vs `R_live` per `(entrypoint,p_fail)`:

  * **MAE_typed** vs **MAE_block**: lower is better.
  * **Win‑rate (typed)**: share of cells where `|R_typed-R_live| < |R_block-R_live|` (ties count half).
    A higher win‑rate/MAE advantage for typed supports the hypothesis; the opposite rejects it.

---

## Design notes & hypothesis test

* The estimator implements **blocking‑reachability + replication** (Algorithm1), exactly as in the accompanying study; we use it **twice**: once on the **typed** graph and once on an **all‑blocking** baseline produced by the same code path (`--assume-all-blocking`).  
* Live success is computed from **5xx + transport‑level** errors (timeouts, socket) collected under fixed‑rate load windows; this avoids conflating 3xx and most 4xx with unavailability. 
* To avoid aggregation bias, the aggregator joins on the **intersection** of `(replica, entrypoint, p_fail)` present in **all three** datasets (typed, all‑blocking, live) before computing errors.

**Testing the hypothesis.** After running at least one full replicate with live windows:

1. open `site/index.html`;
2. check **“Availability: model vs live (pooled)”**: if **MAE_typed < MAE_block** and **win‑rate_typed > 0.5**, typed **outperforms**; otherwise the hypothesis is **rejected** for that run.

---

## Troubleshooting

* **No live section on the page.** Ensure each `replicate-*` directory has `live_availability.csv` with the columns described above and matching `entrypoint,p_fail` to `availability_*.csv`.
* **Entry points mismatch.** Provide the same list to both model (`edgetyper resilience --entrypoints …`) and live (`edgetyper availability-live --entrypoints …`). Otherwise the join will drop missing rows. 
* **Locust names don’t match.** Add `live_targets.yaml` regex rules for `Name` values in `<prefix>_stats.csv`.
* **Replicas unknown.** Omit `--replicas` to assume 1 per service, or supply a CSV `service,replicas`. 

---

## License & citation

* Code: see `LICENSE` in this repository.
* If you use this in a paper or talk, please cite:

> **Model discovery and graph simulation: a lightweight gateway to chaos engineering.**
> *WIP*

---
