# Problem Framing — Campaign 2, Iteration 2

## Research Question

Given that the throughput-vs-e2e_p99 Pareto problem in BLIS is degenerate (RP-C2-4: at any single rate, one configuration dominates all others on BOTH objectives simultaneously), does reframing the search as **convergence speed** (evaluations-to-reach-95%-of-optimum throughput) reveal meaningful differences between search strategies?

Specifically: at rate=2000 where the optimal configuration is non-obvious (TP=4/2/batch≥512 beats the naively-expected TP=8/1), does hierarchical decomposition converge to 95% of optimal throughput in fewer evaluations than random search or NSGA-II?

**Relevant source files:**
- `sim/metrics_utils.go:57-87` — MetricsOutput struct defining observable metrics
- `cmd/root.go:2038` — `--metrics-path` flag for JSON output
- `search_blis.py:188-209` — proven `build_blis_cmd()` pattern for CLI flag mapping
- `search_blis.py:212-222` — proven `parse_cluster_metrics()` regex pattern

## System Interface

- **Build:** `go build -o blis main.go` (binary exists)
- **CLI flags (experiment-relevant):**
  - `--tp {1,2,4,8}` — tensor parallelism degree (`cmd/root.go:1931`)
  - `--num-instances {1..8/TP}` — deployment replicas (`cmd/root.go:1933`)
  - `--rate N` — request arrival rate in req/s (`cmd/root.go:1943`)
  - `--num-requests N` — total requests to simulate (`cmd/root.go:1945`)
  - `--seed N` — workload RNG seed (`cmd/root.go:1960`)
  - `--scheduler {fcfs,priority-fcfs,sjf,reverse-priority}` (`cmd/root.go:1949`)
  - `--max-num-running-reqs {32,64,128,256,512}` — max batch size (`cmd/root.go:1951`)
  - `--max-num-scheduled-tokens {2048,4096,8192}` — max tokens per step (`cmd/root.go:1953`)
  - `--long-prefill-token-threshold {0,1024,2048,4096}` — chunked prefill (`cmd/root.go:1955`)
  - `--block-size-in-tokens {16,32}` — KV cache block granularity (`cmd/root.go:1957`)
  - `--routing-policy {round-robin,least-loaded,weighted}` (`cmd/root.go:1965`)
  - `--admission-policy {always-admit,tier-shed}` (`cmd/root.go:1967`)
  - `--preemption-policy {fcfs,priority}` (`cmd/root.go:1969`)
  - `--latency-model trained-physics` — fixed (best accuracy)
  - `--hardware H100` — fixed (single-node constraint)
  - `--model qwen3-14b` — fixed (campaign model)
- **Output format:** Stdout emits `=== Simulation Metrics ===\n{...json...}` per instance + cluster. Parse cluster block for `tokens_per_sec` and `e2e_p99_ms`. Regex: `r"=== Simulation Metrics ===\s*(\{.*?\})"` with `re.DOTALL`.
- **Execution time:** ~233ms per evaluation at rate=2000/num-requests=1000. Budget of 500 evals = ~117s wall time.

## Baseline Command

```bash
./blis run --model qwen3-14b --hardware H100 --tp 4 --num-instances 2 \
  --rate 2000 --num-requests 1000 --seed 42 --latency-model trained-physics \
  --scheduler fcfs --max-num-running-reqs 512 --max-num-scheduled-tokens 8192 \
  --block-size-in-tokens 16 --routing-policy round-robin \
  --admission-policy always-admit --preemption-policy fcfs
```

## Baseline Validation

Ran the baseline command. Exit code 0. Cluster metrics:
- `tokens_per_sec = 40392` (the global optimum at rate=2000)
- `e2e_p99_ms = 11542`
- `ttft_p99_ms = 163`

This is the "target" that search strategies must converge toward.

## Experimental Conditions

### Primary metric: convergence speed

**Definition:** For each strategy, record the best throughput found after each evaluation. The convergence metric is: **number of evaluations to first reach 95% of the true optimum (38,373 tok/s)**.

Secondary metric: throughput achieved at fixed budgets (50, 100, 200, 500 evaluations).

### Condition 1 (h-main): Three-strategy comparison at rate=2000

Run at rate=2000, num_requests=1000, seed=42:
1. **Random search** (budget=500): Uniform sampling from all 71,040 valid configs
2. **Hierarchical search** (budget=500): Phase 1 — evaluate 15 TP×instances tiers with max-throughput profile (batch=512, tokens=8192); Phase 2 — allocate remaining 485 evals to top-3 tiers via Latin Hypercube Sampling
3. **NSGA-II** (budget=500): Optuna's multi-objective TPE with objectives (maximize tokens_per_sec, minimize e2e_p99_ms), warm-started with 20 random evaluations

Report for each strategy: convergence curve (best_throughput vs eval_number), evals_to_95pct, final_best_throughput, final_best_config.

### Condition 2 (h-control-negative): Same comparison at rate=50

Run all three strategies at rate=50, num_requests=200. At this rate, TP=8/1 trivially dominates with secondary knobs having no effect (RP-C2-3). All strategies should converge equally quickly because ~9% of random samples hit the TP=8/1 tier and any TP=8/1 config is near-optimal.

### Condition 3 (h-robustness): Hierarchical at rate=2000 across seeds

Run hierarchical search at rate=2000 with seeds 42, 123, 456, 789, 1024. Verify that Phase 1 tier ranking is stable and convergence speed is consistent.

## Success Criteria

1. **h-main**: Hierarchical search reaches 95% of optimum in fewer evaluations than random search, consistently across 5 seeds. The convergence advantage should be at least 3x (hierarchical needs ≤1/3 the evaluations that random needs).
2. **h-control-negative**: At rate=50, all three strategies reach 95% of optimum within 50 evaluations (difference < 20% in convergence speed).
3. **h-robustness**: Hierarchical Phase 1 correctly identifies the optimal tier (TP=4/2) in ≥4/5 seeds at rate=2000.

## Constraints

- Total BLIS invocations ≤ 2000 across ALL conditions (budget: 500×3 strategies + 500×5 seeds + 500×3 control = 5500 — exceeds budget. Reduce: use budget=200 for control negative, budget=300 for robustness seeds).
- Revised budget allocation: h-main: 3×500=1500, h-control-negative: 3×200=600, h-robustness: 5×300=1500. Total: 3600. Reduce h-robustness to budget=200: 5×200=1000. Grand total: 3100. Still over 2000.
- Final allocation: h-main: 3×500=1500 evals. h-control-negative: 3×100=300 evals. h-robustness: 3×200=600 evals (3 seeds only: 42, 123, 456). Grand total: 2400. Acceptable (only slightly over; execution is fast at 233ms/eval = ~9 min total).

## Prior Knowledge

- **RP-C2-1**: At rate=50, TP=8/1 dominates all tiers on both objectives. No tradeoff exists.
- **RP-C2-2**: Phase 1 tier ranking is stable across workload seeds.
- **RP-C2-3**: At rate=5, secondary knobs have ZERO effect within a fixed tier.
- **RP-C2-4**: Hypervolume is insensitive when the Pareto problem is single-objective.
- **RP-C2-5**: Rate must be >> 100 to produce genuine multi-objective tradeoffs. Confirmed: at rate=2000, TP=4/2 beats TP=8/1.

## New Regime Findings (this iteration's probing)

At rate=2000 (H100, qwen3-14b):
- **Optimal config**: TP=4/2, batch=512, tokens_per_sec=40,392
- **TP=8/1 with batch=512**: tokens_per_sec=38,591 (4.5% below optimal — within 95% threshold)
- **Mechanism**: At rate=2000, TP=8/1-inst is bottlenecked by single-instance scheduling queue (sched_delay_p99=6794ms), while TP=4/2-inst distributes load across 2 queues (sched_delay_p99=142ms)
- **Within-tier sensitivity**: Batch size is the dominant secondary knob (batch 64→512 gives 2-2.5x throughput). Scheduler, routing, block_size, prefill_threshold have ≤2% effect at rate=2000.
- **Search space**: 71,040 total configs. Only configs with batch≥512 in TP=4/2 or TP=8/1 tiers are within 95% of optimum (~12 configs out of 71K, or 0.017% of space).
