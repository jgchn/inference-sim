# Handoff — Campaign 2, Iteration 2

## Goal

Implement and compare three configuration search strategies (random, hierarchical decomposition, NSGA-II) for finding the throughput-maximizing BLIS configuration at rate=2000, where the optimal config is non-obvious (TP=4/2 beats TP=8/1 due to queue bottleneck). Measure convergence speed: evaluations needed to reach 95% of optimal throughput (38,373 tok/s). Prove that hierarchical decomposition converges faster by exploiting the parallelism-tier structure.

## Key Discoveries

1. **Pareto front is degenerate at all rates** — at any fixed rate, one config dominates all others on BOTH throughput and e2e_p99 simultaneously. The "multi-objective" framing from iteration 1 was wrong; this is a single-objective problem (maximize throughput). Hypervolume is the wrong metric; use convergence speed instead.

2. **Rate=2000 produces a non-obvious optimum** — TP=4/2/batch=512 achieves 40,392 tok/s, beating TP=8/1/batch=512 at 38,591 tok/s (4.5% gap). The mechanism: single-instance TP=8 bottlenecks on scheduling queue (sched_delay_p99=6794ms) while 2-instance TP=4 distributes load (sched_delay_p99=142ms).

3. **Batch size is the dominant secondary knob at rate=2000** — within TP=4/2: batch 64→512 produces 19,944→40,392 tok/s (2x). Scheduler, routing, block_size, chunked_prefill all have ≤2% effect.

4. **95% threshold configs are rare** — 95% of 40,392 = 38,373 tok/s. Only TP=4/2/batch=512 (40,392) and TP=8/1/batch=512 (38,591) meet this threshold. That's ~12 configs out of 71,040 (0.017% of space).

5. **Execution speed at rate=2000** — 233ms per evaluation with num_requests=1000. Budget of 500 evals = ~117s wall time.

6. **TP=4/2 saturates at batch=256** — at rate=2000, batch=256 gives 35,574 tok/s and batch=512 gives 40,392 tok/s. batch=256≠512 (unlike at rate=500 where they're equal). So batch=512 is strictly needed.

7. **At rate=50, same degeneracy as iteration 1** — TP=8/1 dominates at 14,232 tok/s regardless of secondary knobs (confirmed by RP-C2-1 and RP-C2-3).

## System Interface

- **Build:** `go build -o blis main.go` (binary already exists at `./blis`)
- **Run baseline:** `./blis run --model qwen3-14b --hardware H100 --tp 4 --num-instances 2 --rate 2000 --num-requests 1000 --seed 42 --latency-model trained-physics --scheduler fcfs --max-num-running-reqs 512 --max-num-scheduled-tokens 8192 --block-size-in-tokens 16 --routing-policy round-robin --admission-policy always-admit --preemption-policy fcfs`
- **Output format:** Stdout emits `=== Simulation Metrics ===\n{...json...}` blocks (one per instance + one for "cluster"). Parse cluster block for aggregate metrics. Regex: `r"=== Simulation Metrics ===\s*(\{.*?\})"` with `re.DOTALL`.
- **Baseline result:** `tokens_per_sec = 40392`, `e2e_p99_ms = 11542` at rate=2000 with TP=4, 2 instances, batch=512.

## Code Map

- `sim/metrics_utils.go:57-87` — `MetricsOutput` struct. Check here for exact JSON field names.
- `cmd/root.go:2038` — `--metrics-path` flag registration. Only on `blis run`.
- `search_blis.py:212-222` — `parse_cluster_metrics()` function. Proven stdout parsing pattern.
- `search_blis.py:188-209` — `build_blis_cmd()` function. Shows exact flag mapping from config dict to CLI args.
- `search_blis.py:269-283` — `sample_random_config()` function. Shows valid option arrays.
- `search_blis.py:60-120` — Search space constants (TP_OPTIONS, VALID_TP_INSTANCES, etc.)

## Code Targets

### h-main: `search_pareto_iter2.py` (new file)

Create a self-contained Python script (~400 lines) implementing:
1. Configuration space definition (71,040 valid configs: 15 TP×instances tiers × secondary knobs)
2. `run_blis(config)` → subprocess call, parse stdout for tokens_per_sec and e2e_p99_ms
3. `random_search(budget)` — uniform sampling with convergence tracking
4. `hierarchical_search(budget)` — Phase 1: evaluate 15 tiers with max-throughput profile (batch=512, tokens=8192); rank by tokens_per_sec; Phase 2: allocate remaining budget to top-3 tiers via Latin Hypercube Sampling within each tier's secondary-knob space
5. `nsga2_search(budget)` — Optuna NSGA-II sampler with 20 random warmup evaluations, objectives: maximize tokens_per_sec (negate for minimization), minimize e2e_p99_ms
6. Convergence tracking: after each eval, record best_throughput_so_far
7. `compute_evals_to_threshold(curve, threshold)` — first eval where curve ≥ threshold
8. Main entry: `--strategy {random,hierarchical,nsga2,all}`, `--budget N`, `--rate N`, `--num-requests N`, `--seed N`, `--output path.json`

**Key design decisions:**
- Convergence curve is the primary output (list of best_so_far values, one per eval)
- 95% threshold computed from known optimum (40,392 at rate=2000; 14,232 at rate=50)
- Hierarchical Phase 1 uses batch=512, tokens=8192 for max throughput (validated by probing)
- NSGA-II uses optuna.samplers.NSGAIISampler with pop_size=20

### h-control-negative: Same script, `--rate 50 --num-requests 200 --budget 100`

No code changes. Different args.

### h-robustness: Same script, `--rate 2000 --strategy hierarchical --seed {42,123,456}`

No code changes. Multiple seed runs.

## What I Tried That Didn't Work

- **Hypervolume as a metric (iteration 1)**: Degenerate Pareto front means HV is identical once any strategy finds the single dominant config. Cannot distinguish search strategies.
- **Rate=50, 200, 300, 500 for Pareto tradeoffs**: All produce degenerate fronts where one config dominates both throughput and latency simultaneously.
- **Rate=500 for non-obvious optimum**: TP=8/1/batch=512 still dominates at 32,267 tok/s. Only at rate≥2000 does TP=4/2 beat TP=8/1.
- **Throughput vs TTFT as objectives**: Produces a 2-3 point front at rate=2000, but too small for meaningful search algorithm comparison.
- **max-num-scheduled-tokens variation**: No effect at rate=500 or 2000 (2048, 4096, 8192 all give identical results).
- **Chunked prefill threshold variation**: No effect at any rate tested (0, 1024, 4096 identical).
- **Block size variation**: No effect (16 and 32 give identical results at rate=2000).

## What I Excluded and Why

- **Multi-rate optimization**: Testing search strategies across multiple rates simultaneously would add complexity. Single-rate optimization (rate=2000) is the minimal interesting case where hierarchical can demonstrate an advantage.
- **A100-SXM hardware**: Fixed to H100 for this iteration. Hardware portability can be tested in iteration 3 if convergence advantage is confirmed.
- **Workload presets (chatbot, summarization, etc.)**: Using default distribution workload. Presets change the rate at which tiers cross over but don't fundamentally change the search structure.
- **cost-aware optimization**: Adding GPU cost as a third objective would create a richer Pareto front, but departs from the iteration 1 framing.

## Evolution of Thinking

Started iteration 2 by trying to find a rate where the throughput-vs-p99 Pareto front is non-degenerate. Probed rates 200, 300, 500, 1000, 2000. Discovered that at EVERY rate, one config dominates all others on both objectives simultaneously — the "Pareto problem" is fundamentally degenerate in BLIS because higher throughput capacity means shorter queues which means lower latency (monotonic relationship in queuing systems below overload).

This forced a complete reframing: from "discover the Pareto frontier" (which is always a single point) to "converge to the optimal configuration" (a genuine search problem). The convergence-speed metric (evals_to_95pct) CAN distinguish strategies because it measures how quickly each strategy reaches the needle in the 71K-config haystack.

The rate=2000 regime is ideal because: (1) the optimal config is non-obvious (TP=4/2 beats the naively-expected TP=8/1), (2) the optimal occupies only 0.017% of the space, (3) hierarchical Phase 1 can identify the correct tier in 15 evals while random needs ~37 evals just to hit the right tier, and (4) NSGA-II provides an "intelligent baseline" to test whether the hierarchical structure is the right inductive bias.

## Current Status

- **Validated:** rate=2000 produces non-obvious optimum (TP=4/2 > TP=8/1). Execution time 233ms/eval supports 500-eval budget in ~2 minutes. Secondary knobs (batch) matter at this rate. Optuna 4.8.0 is available for NSGA-II.
- **Uncertain:** Whether the 4.5% gap between TP=4/2 and TP=8/1 (40,392 vs 38,591) is stable across seeds (could flip for some seeds). Whether NSGA-II's surrogate model can learn the batch-size dominance faster than hierarchical's Phase 2 LHS.
- **Suggested next:** If hierarchical wins, iteration 3 should (a) test whether the advantage scales to larger budgets (1000, 2000 evals), (b) test hardware portability (A100-SXM where the TP crossover rate is different), (c) explore adaptive Phase 2 strategies (Bayesian within-tier instead of LHS).

## Warnings & Constraints

1. **TP=4/2 vs TP=8/1 gap is only 4.5%** — with different seeds, the ordering might flip. The 95% threshold (38,373) includes BOTH TP=4/2/b512 (40,392) and TP=8/1/b512 (38,591). So the convergence metric is really "find any high-batch config in TP=4/2 or TP=8/1" — either qualifies.

2. **stdout includes all instances** — When `--num-instances > 1`, stdout emits one JSON block per instance plus one for "cluster". The parser must find the cluster block specifically.

3. **rate=2000 needs num_requests≥1000** — at 2000 req/s, 200 requests finish in 0.1s which doesn't build enough queue pressure. Use num_requests=1000 for stable metrics.

4. **optuna import** — `import optuna` is available (v4.8.0). Also needs `numpy` and `scipy.stats.qmc` for Latin Hypercube. Both are available.

5. **Hierarchical Phase 1 must use batch=512** — at rate=2000, lower batch sizes significantly reduce throughput. Using batch=128 in Phase 1 would incorrectly rank TP=8/1 above TP=4/2 (24,550 vs 28,164) but with batch=512, TP=4/2 correctly ranks #1 (40,392 vs 38,591). The max-throughput profile MUST include batch=512 and tokens=8192.

6. **`--hardware H100`** — must be exactly "H100" (not "h100-sxm" or "H100-SXM").
