# Handoff — Campaign 3, Iteration 3

## Goal

Test whether adding `total_kv_blocks` as a searchable parameter creates a genuinely harder 4-objective search problem where NSGA-II outperforms random search. The 4 objectives: maximize rps, minimize ttft_p99, minimize gpu_count, minimize kv_blocks. Budget: 200 evaluations per arm. Three arms: NSGA-II 4-obj, Random 4-obj, NSGA-II 3-obj (control confirming iter 2 result).

## Key Discoveries

1. **`--total-kv-blocks` is PER INSTANCE.** With tp=4,i=1 and blocks=5000, only 5000 blocks exist. With tp=2,i=2 and blocks=5000, each instance gets 5000 → 10000 total. This creates a non-obvious TP↔blocks interaction that reverses dominance. In iter 2 (1M blocks), tp=4,i=1 dominated tp=2,i=2. In KV-constrained regime: tp=4,i=1 at blocks=5000 gives (16.76 rps, 2640ms ttft, 77 preemptions) while tp=2,i=2 gives (14.83 rps, 39ms ttft, 0 preemptions). Neither dominates — genuine Pareto tradeoff.

2. **KV cliff positions differ by GPU tier.** tp=2,i=2: cliff between blocks=4000 (no stress) and blocks=3000 (75 preemptions). tp=4,i=1: already stressed at blocks=5000 (77 preemptions); stress-free at blocks=10000. tp=1,i=1: stressed even at blocks=5000 (107 preemptions) due to throughput saturation causing long queue residence. tp=8,i=1: mild stress at blocks=5000 (33 preemptions).

3. **Scheduler differentiates ONLY under KV stress.** At tp=2,i=2,blocks=3000: FCFS gives (12.92, 3437) while SJF gives (13.34, 4839). SJF has better rps (+3%) but worse ttft (+41%) due to more preemptions (102 vs 75). At blocks≥4000 (no stress): FCFS and SJF produce identical results.

4. **Batch=128/256/512 produce IDENTICAL results under KV stress.** At blocks=3000,tp=2,i=2: batch=128, 256, and 512 all give (12.92, 3437, 75 preemptions). KV capacity is the binding constraint, not batch limit. Only batch=32 (dominated: 10.65, 7910, 0 preemptions) and batch=64 (13.00, 3411, 64 preemptions) differ.

5. **Routing policy has minimal effect (<3%) even under KV stress.** At blocks=3000: round-robin=13.14, weighted=12.92, least-loaded=13.27. The difference is within noise. Preemption policy (fcfs vs priority) produces IDENTICAL results.

6. **Evaluation time scales with sim duration.** tp=2,i=2 at blocks=5000: ~200ms wall time. tp=1,i=1 at blocks=1500: ~400ms wall time (longer sim due to throughput limitation). 200 evals ≈ 40-80 seconds total.

7. **4-objective reference point must accommodate tp=1 at low blocks.** Worst observed: tp=1,i=1,blocks=1500 gives ttft_p99=48934ms. Reference point: (0, 50000, 9, 11000).

## System Interface

- **Build:** `go build -o blis main.go` (binary already exists)
- **Run baseline:**
  ```bash
  ./blis run --model qwen/qwen3-14b --hardware H100 --latency-model trained-physics \
    --num-requests 200 --rate 50 --tp 2 --num-instances 2 --scheduler fcfs \
    --max-num-running-reqs 128 --max-num-scheduled-tokens 4096 \
    --long-prefill-token-threshold 0 --block-size-in-tokens 16 \
    --routing-policy weighted \
    --routing-scorers "precise-prefix-cache:2,queue-depth:1,kv-utilization:1" \
    --admission-policy always-admit --preemption-policy fcfs \
    --gpu-memory-utilization 0.9 --prefix-tokens 512 --total-kv-blocks 5000 \
    --seed 42 --metrics-path $TMPDIR/baseline.json
  ```
- **Output format:** JSON at `--metrics-path`, key fields: `responses_per_sec` (float), `ttft_p99_ms` (float), `preemption_count` (int). Fourth objective `gpu_count = tp * num_instances`.
- **Baseline result:** `responses_per_sec=14.83`, `ttft_p99_ms=39.13`, exit code 0

## Code Map

| Location | What | When to look |
|----------|------|--------------|
| `sim/metrics_utils.go:57` | `MetricsOutput` struct — all JSON field names | If metrics JSON parsing fails |
| `sim/metrics.go:66` | `SaveResults()` — computes percentiles, writes JSON | If output values seem wrong |
| `cmd/root.go:946` | `--total-kv-blocks` flag definition (default 1000000) | If flag name/default issues |
| `cmd/root.go:184` | `metricsPath` variable | If `--metrics-path` flag issues |
| `sim/kv/cache.go:222-253` | KV block pre-check triggering preemption | If preemption counts seem wrong |
| `sim/batch_formation.go:225-304` | `preemptForTokens()` — victim selection | Understanding preemption behavior |

## Code Targets

All three arms create new Python files in `inputs/` — no BLIS source modifications needed.

| Arm | File | Purpose |
|-----|------|---------|
| h-main | `.nous/campaign-3/runs/iter-3/inputs/search_nsga2_4obj.py` | NSGA-II with 4 objectives, pop=40, 5 gens |
| h-control-negative | `.nous/campaign-3/runs/iter-3/inputs/search_random_4obj.py` | Random search, 200 evals, 4D HV tracking |
| h-robustness | `.nous/campaign-3/runs/iter-3/inputs/search_nsga2_3obj.py` | NSGA-II on original 3-obj space (1M blocks) |

All three must:
- Implement constraint-aware config generation (TP*inst<=8, routing only for multi-instance)
- Evaluate via subprocess: `./blis run ... --metrics-path $TMPDIR/metrics_<id>.json`
- Parse JSON output for `responses_per_sec` and `ttft_p99_ms`; derive `gpu_count = tp * num_instances`
- For 4-obj arms: also record `total_kv_blocks` as 4th objective
- Compute hypervolume (4D or 3D) with appropriate reference point
- Track convergence: record cumulative HV every 40 evaluations
- Output: `pareto_front.json` and `convergence.json` to the `results/` directory

**Parameter space for 4-objective arms:**
```python
params = {
    'tp': [1, 2, 4, 8],
    'num_instances': None,  # derived: range(1, 8//tp + 1)
    'scheduler': ['fcfs', 'sjf'],
    'max_num_running_reqs': [32, 64, 128, 256, 512],
    'total_kv_blocks': [2000, 3000, 4000, 5000, 7500, 10000],
    'block_size_in_tokens': [16, 32],
    'routing_policy': ['round-robin', 'least-loaded', 'weighted'],
    'routing_scorers': [
        'precise-prefix-cache:2,queue-depth:1,kv-utilization:1',
        'queue-depth:1,kv-utilization:1',
        'precise-prefix-cache:2,load-balance:1',
        'load-balance:1,kv-utilization:1',
    ],
    'gpu_memory_utilization': [0.85, 0.9, 0.95],
}
```

**Fixed for all evals:**
```
--model qwen/qwen3-14b --hardware H100 --latency-model trained-physics
--num-requests 200 --rate 50 --prefix-tokens 512 --seed 42
--max-num-scheduled-tokens 4096 --long-prefill-token-threshold 0
--admission-policy always-admit --preemption-policy fcfs
```

**Reference points:**
- 4-obj: (0, 50000, 9, 11000)
- 3-obj: (0, 50000, 9) — ttft ref increased from iter 2's 40000 because tp=1 at rate=50 can exceed 40000ms

## What I Tried That Didn't Work

1. **Preemption policy differentiation** — fcfs vs priority preemption produces IDENTICAL results (same rps, ttft, preemption_count) at all tested configurations. The policy only matters when specific request priorities exist in the workload.

2. **Routing policy differentiation in KV-stressed regime** — round-robin (13.14), weighted (12.92), least-loaded (13.27) differ by <3% at blocks=3000. Not enough to create search difficulty.

3. **Multi-prefix workload spec with blocks=3000** — token lengths in the workload spec (mean=256) are shorter than synthesis mode (mean~500 + 512 prefix), so blocks=3000 doesn't stress the multi-prefix workload. Would need to set blocks even lower (~1500) or increase token lengths.

4. **Chunked prefill threshold** — threshold=1024 vs threshold=0 produces <1% difference at blocks=3000 (77 vs 75 preemptions). Prefill chunking doesn't meaningfully interact with KV stress in this regime.

5. **Batch=256 and batch=512 differentiation under KV stress** — both produce identical results to batch=128 at blocks=3000 because KV is the binding constraint. The batch limit is never reached.

## What I Excluded and Why

1. **Parallel evaluation workers** — at 200-400ms/eval, 200 evals take 40-80s serially. No parallelism needed for the 3-minute budget.

2. **Multiple seeds per evaluation** — determinism (INV-6) makes single-seed stable. Would add 3-5x cost without value.

3. **TPE as a third algorithm** — iter 2 conclusively showed TPE performs WORSE than random on BLIS's conditional space (RP-4). Not worth retesting.

4. **Preemption policy and admission policy in sweep** — neither differentiates. Fixed at `--preemption-policy fcfs --admission-policy always-admit`.

5. **max_num_scheduled_tokens in sweep** — did not test explicitly, but at rate=50 with this workload, it's unlikely to differentiate given that batch capacity isn't the bottleneck. Fixed at 4096.

6. **long_prefill_token_threshold in sweep** — <1% effect. Fixed at 0.

## Evolution of Thinking

Started iteration 3 following iter 2's suggestion to "test a workload with prefix sharing where routing policy actually matters." Probed extensively with `--prefix-tokens 512` and KV constraints. Found that routing STILL doesn't meaningfully differentiate (<3% effect), but discovered something far more interesting: **KV blocks being per-instance creates a non-linear TP↔blocks interaction that REVERSES the iter 2 dominance finding.**

In iter 2: tp=4,i=1 always dominated tp=2,i=2 (better rps AND ttft). In the KV-constrained regime: tp=4,i=1 gets 16.76 rps but 2640ms ttft (stressed), while tp=2,i=2 gets 14.83 rps but 39ms ttft (no stress). This creates a genuine Pareto tradeoff within the same GPU tier that depends non-linearly on the blocks parameter.

This insight led to the 4-objective formulation: by making `total_kv_blocks` both a searchable parameter and a minimized objective (proxy for memory cost), we get a richer Pareto surface where the optimal blocks value depends on the TP/instance configuration. The search algorithm must discover WHICH (tp, inst, blocks) combinations avoid the preemption cliff — a conditional relationship that NSGA-II's crossover might exploit better than uniform random sampling.

## Current Status

- **Validated:** 4-objective frontier exists with genuine tradeoffs; evaluation timing works; baseline commands confirmed; KV cliff positions measured for all GPU tiers
- **Uncertain:** Whether the 4-objective Pareto density is low enough (<30%) for NSGA-II to show measurable convergence advantage at 200 evals. The worry is that routing/block_size/gpu_mem_util still don't differentiate, inflating Pareto density.
- **Suggested next:** If NSGA-II still shows no advantage on 4-obj, iteration 4 should (a) remove non-differentiating parameters (routing, block_size, gpu_util) to reduce noise, (b) try a workload with heterogeneous request sizes (some long, some short) that makes scheduler ordering impactful, (c) consider a constraint-satisfaction formulation where feasibility is itself hard to achieve (e.g., ttft_p99 < 1000ms AND rps > 15 as constraints, with remaining objectives optimized)

## Warnings & Constraints

1. **`--metrics-path` must use `$TMPDIR`**, not `/tmp/`. Sandbox prevents writing to `/tmp/` directly.
2. **`--total-kv-blocks` is PER INSTANCE.** Multi-instance configs have total_blocks × num_instances effective capacity. This is the key interaction — don't assume it's a global limit.
3. **tp=1,i=1 is always KV-stressed at rate=50** even at blocks=5000 (107 preemptions). The throughput bottleneck (5.6 rps vs rate=50) causes massive queue buildup → many concurrent requests → KV overflow. Don't expect tp=1 to ever achieve low TTFT at this rate regardless of blocks.
4. **blocks=5000 and blocks=4000 produce IDENTICAL results for tp=2,i=2** (both have 0 preemptions, same metrics). The cliff is between 4000 and 3000 for this config. For tp=4,i=1, the cliff is higher (between 5000 and 10000).
5. **Reference point (0, 50000, 9, 11000)** must be consistent across 4-obj arms. 50000ms accommodates worst case (tp=1,blocks=1500 gives 48934ms). 11000 blocks is above max (10000).
6. **Hardware name is exactly `H100`** — not "h100-sxm".
7. **Batch≥128 produces identical results under KV stress.** When KV is the binding constraint, increasing batch beyond ~64-128 has zero effect because concurrency is limited by available KV blocks, not batch capacity.
