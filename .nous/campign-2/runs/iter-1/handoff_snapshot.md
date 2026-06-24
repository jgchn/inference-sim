# Handoff — Campaign 2, Iteration 1

## Goal

Implement and compare two configuration search strategies (random vs hierarchical decomposition) for discovering the Pareto frontier of throughput (tokens_per_sec) vs tail latency (e2e_p99_ms) in BLIS's 86,400-configuration space. Produce hypervolume measurements proving which strategy is more efficient at 500 evaluations.

## Key Discoveries

1. **Execution speed:** BLIS runs in ~86ms per invocation (timed 10 sequential runs at 0.864s total). At 500 evals, total wall time is ~43s — well within the 2-minute budget.

2. **Search space size:** 86,400 total configurations. Breakdown: TP=1 → 46,080 configs (8 instance options), TP=2 → 23,040 (4 instance options), TP=4 → 11,520 (2 instance options), TP=8 → 5,760 (1 instance option).

3. **Metric differentiation at rate=50:** Throughput ranges from 1,832 tok/s (TP=1, 1 inst, batch=32) to 14,232 tok/s (TP=8, 1 inst, batch=512). Tail latency ranges from 4,719ms to 51,406ms. This 8x throughput / 11x latency range confirms rate=50 produces meaningful Pareto tradeoffs.

4. **TP dominance confirmed:** At rate=50, TP=8/1-inst consistently dominates TP=1/1-inst regardless of other knobs. But TP=4/2-inst vs TP=8/1-inst shows genuine tradeoffs (11,985 vs 14,232 tok/s; 6,326 vs 4,719ms e2e_p99).

5. **Hardware name is "H100"** (not "h100-sxm" — that fails with "GPU not found"). Valid hardware names: `A100-80`, `A100-SXM`, `H100`, `L40S`.

6. **Determinism:** Same seed produces byte-identical metrics. Different seeds produce slightly different workloads but preserve overall behavior patterns.

7. **metrics-path writes cluster-level JSON only** (single dict, not array). Stdout still emits per-instance + cluster blocks even when metrics-path is set. For programmatic extraction, either parse stdout (like existing `parse_cluster_metrics`) or use metrics-path for single-run JSON.

## System Interface

- **Build:** `go build -o blis main.go` (binary already exists at `./blis`)
- **Run baseline:** `./blis run --model qwen3-14b --hardware H100 --tp 2 --num-instances 2 --rate 50 --num-requests 200 --seed 42 --latency-model trained-physics --scheduler fcfs --max-num-running-reqs 128 --max-num-scheduled-tokens 4096 --block-size-in-tokens 16 --routing-policy round-robin --admission-policy always-admit --preemption-policy fcfs`
- **Output format:** Stdout emits `=== Simulation Metrics ===\n{...json...}` blocks (one per instance + one for "cluster"). Parse cluster block for aggregate metrics. Alternatively, `--metrics-path file.json` writes cluster-level `MetricsOutput` JSON.
- **Baseline result:** `tokens_per_sec = 8195.97`, `e2e_p99_ms = 10602.11` at rate=50 with TP=2, 2 instances.

## Code Map

- `sim/metrics_utils.go:57-87` — `MetricsOutput` struct. Check here for exact JSON field names.
- `cmd/root.go:2038` — `--metrics-path` flag registration. Check here if metrics output behavior is wrong.
- `cmd/root.go:184` — `metricsPath` variable. The flag is only on `blis run` (not replay).
- `search_blis.py:212-222` — `parse_cluster_metrics()` function. Proven stdout parsing pattern for extracting cluster metrics from multi-instance output. Reuse this regex approach.
- `search_blis.py:188-209` — `build_blis_cmd()` function. Shows exact flag mapping from config dict to CLI args.
- `search_blis.py:269-283` — `sample_random_config()` function. Shows valid option arrays for each knob.

## Code Targets

### h-main: `search_pareto.py` (new file)

Create a self-contained Python script (~300-400 lines) implementing:
1. Configuration space definition (reuse patterns from `search_blis.py:60-120`)
2. `run_blis(config)` → parse stdout for tokens_per_sec and e2e_p99_ms
3. `random_search(budget=500)` — uniform sampling
4. `hierarchical_search(budget=500)` — Phase 1 (15 evals: each TP×instances with max-throughput profile) + Phase 2 (485 evals: LHS within top-3 tiers)
5. `compute_hypervolume(points, reference)` — 2D hypervolume calculation
6. `is_pareto_optimal(points)` — Pareto dominance filter
7. Main entry with `--strategy`, `--budget`, `--rate`, `--seed`, `--output` args

**Reference point for hypervolume:** Use (0, max_observed_e2e_p99 × 1.1) — ensures all non-dominated points contribute.

### h-control-negative: Same script, `--rate 5`

Run both strategies at rate=5 where all configs are unsaturated. No code changes — just a different `--rate` argument.

### h-robustness: Same script, `--seed` varies

Run hierarchical search with seeds 42, 123, 456, 789, 1024. No code changes — just different `--seed` arguments.

## What I Tried That Didn't Work

- `--hardware h100-sxm` — fails with "GPU not found". Must use `H100`.
- Running TP=8 with 2 instances — the simulator doesn't enforce TP×instances≤8 as a hard constraint. The 8-GPU limit is a logical constraint we must enforce in the search algorithm.
- Using `--metrics-path` alone for parsing — it only writes when the run completes successfully and doesn't include per-instance data. For the search script, parsing stdout is more robust (matches `search_blis.py` proven approach).

## What I Excluded and Why

- **Multi-objective TPE (Optuna NSGA-II):** Excluded from iteration 1 to keep the comparison clean (random vs one structured approach). Will be a natural candidate for iteration 2 if hierarchical wins.
- **Multi-rate experiments:** Fixed at rate=50 to isolate the search algorithm comparison from workload regime effects. Later iterations can test portability.
- **Workload presets (chatbot, summarization, etc.):** Using default "distribution" workload for iteration 1. Presets change prompt/output length distributions which shifts TP crossover points.
- **Multiple seeds for each eval:** Each config is evaluated once (seed=42). Multi-seed noise reduction is a refinement for later iterations. The determinism guarantee means single-seed is sufficient for comparing search strategies.

## Evolution of Thinking

Started by assuming rate=30 would be sufficient for differentiation. Discovered that at rate=30, TP=8/1-inst dominates everything on both objectives simultaneously (no real tradeoff). At rate=50, genuine Pareto tradeoffs emerge because high-TP configs approach saturation while low-TP configs are deeply saturated. This creates a front where some configs offer moderate throughput with low latency, and others offer maximum throughput with higher latency.

Initially considered using `--fitness-weights` for a scalarized objective, but realized Pareto frontier discovery requires the raw bi-objective values. The fitness-weights mechanism is useful for single-point optimization but not for frontier characterization.

## Current Status

- **Validated:** BLIS executes correctly at rate=50 with all knob combinations tested. Output parsing is straightforward. Execution time (~86ms) supports 500 evals in ~43s.
- **Uncertain:** Whether Latin Hypercube Sampling within a tier is the best Phase 2 strategy (vs random within-tier, or adaptive). Also uncertain whether 3 tiers is optimal or if 4-5 would capture more of the frontier.
- **Suggested next:** If hierarchical wins, iteration 2 should test (a) Bayesian optimization (NSGA-II via Optuna) as a third arm, (b) portability across rate=20 and rate=100, (c) whether the advantage holds on A100-SXM hardware.

## Warnings & Constraints

1. **stdout includes all instances:** When `--num-instances > 1`, stdout emits one JSON block per instance plus one for "cluster". The parser must find the cluster block specifically. Use the proven regex from `search_blis.py:214`: `r"=== Simulation Metrics ===\s*(\{.*?\})"` with `re.DOTALL`.

2. **TP=1 with single instance at rate=50 is deeply saturated:** e2e_p99 exceeds 50s. These configs complete successfully but with extreme latency. Don't set a timeout below 60s or they'll be falsely killed.

3. **`--routing-policy weighted` requires routing-scorers:** When instances>1 and routing=weighted, it uses default scorer weights. No additional flag needed for basic operation.

4. **`--long-prefill-token-threshold 0` means disabled** (no chunked prefill). Non-zero values enable chunking at that threshold.

5. **Ground truth budget:** To compute a reliable hypervolume reference, enumerate 2 full tiers (TP=2/4-inst = 1440 configs + TP=4/2-inst = 1440 configs = 2880). But the 2000-eval budget is tight. A practical alternative: enumerate TP=4/2-inst (1440 configs) as reference for that tier, and use the full 500-eval strategies' combined results as an approximation of the global frontier.
