# Handoff — Campaign 3, Iteration 4

## Goal

Test whether removing non-differentiating parameters (routing_policy, block_size_in_tokens, gpu_memory_utilization) from the NSGA-II search space amplifies convergence speed over random search. The reduced space has 900 configs (5 parameters) vs 25200 (9 parameters). Run exhaustive ground truth (all 900) to provide absolute reference, then compare NSGA-II vs random at 200-eval budget.

## Key Discoveries

1. **Reduced space = 900 configs, exhaustible in 74 seconds.** Average eval time is 83ms (not 200ms as estimated in iter 3). This allows running the complete exhaustive sweep as ground truth within the 3-minute budget.

2. **Batch=32 is strongly differentiating even WITHOUT KV stress.** At tp=2,i=2,blocks=5000: batch=32 gives (10.70 rps, 7979ms ttft) vs batch=512 gives (15.19 rps, 39ms ttft). The small batch cap creates a scheduling bottleneck (max 32 concurrent requests) causing massive queueing at rate=50.

3. **Scheduler differentiates at batch=64 under KV stress.** At tp=2,i=2,blocks=3000,batch=64: SJF gives (13.64, 4369) vs FCFS (13.14, 3435). SJF trades +4% rps for +27% worse ttft — a genuine Pareto tradeoff. At batch≥128, scheduler has no effect.

4. **Within-tier Pareto density is 30% but with heavy degeneracy.** In the tp=2,i=2 tier (60 configs): 18 are Pareto-optimal but map to only ~5 distinct metric outcomes. This means batch={128,256,512} all produce identical results under KV stress (KV is the binding constraint), and kv={4000,5000,7500,10000} all produce identical results (no stress above cliff).

5. **tp=4,i=2 at blocks=3000 is a sweet spot: (21.03 rps, 1192ms ttft, 43 preemptions).** Uses 8 GPUs but achieves high throughput with moderate KV stress. The per-instance semantics mean 2 instances × 3000 blocks = 6000 total blocks, enough to avoid severe stress at this throughput level.

6. **Evaluation time varies by config.** tp=4,i=1 at blocks=5000: 61ms. tp=4,i=2 at blocks=3000: 90ms. tp=1,i=1: 84ms. All well under 200ms.

7. **Reference point (0, 50000, 9, 11000) remains valid.** tp=1,i=1 at rate=50 still produces the worst TTFT values, but the reduced space excludes extreme low-blocks configs for tp=1 that might exceed 50000ms. Worst probed: tp=2,i=2,sjf,batch=64,blocks=2000 gives ttft_p99=8824ms.

## System Interface

- **Build:** `go build -o blis main.go` (binary already exists at `./blis`)
- **Run baseline:**
  ```bash
  ./blis run --model qwen/qwen3-14b --hardware H100 --latency-model trained-physics \
    --num-requests 200 --rate 50 --prefix-tokens 512 --seed 42 \
    --max-num-scheduled-tokens 4096 --long-prefill-token-threshold 0 \
    --admission-policy always-admit --preemption-policy fcfs \
    --block-size-in-tokens 16 --routing-policy round-robin \
    --gpu-memory-utilization 0.9 \
    --tp 2 --num-instances 2 --scheduler fcfs \
    --max-num-running-reqs 128 --total-kv-blocks 3000 \
    --metrics-path $TMPDIR/baseline.json
  ```
- **Output format:** JSON at `--metrics-path`, key fields: `responses_per_sec` (float), `ttft_p99_ms` (float), `preemption_count` (int). Fourth objective `gpu_count = tp * num_instances`.
- **Baseline result:** `responses_per_sec=13.14`, `ttft_p99_ms=3399.2`, exit code 0

## Code Map

| Location | What | When to look |
|----------|------|--------------|
| `sim/metrics_utils.go:57` | `MetricsOutput` struct — all JSON field names | If metrics JSON parsing fails |
| `sim/metrics.go:66` | `SaveResults()` — computes percentiles, writes JSON | If output values seem wrong |
| `cmd/root.go:946` | `--total-kv-blocks` flag definition (default 1000000) | If flag name/default issues |
| `cmd/root.go:184` | `metricsPath` variable | If `--metrics-path` flag issues |
| `cmd/root.go:919` | `--tp` flag | If TP issues |
| `cmd/root.go:908` | `--num-instances` flag | If instance count issues |
| `sim/kv/cache.go:222-253` | KV block pre-check triggering preemption | If preemption counts seem wrong |
| `sim/batch_formation.go:225-304` | `preemptForTokens()` — victim selection | Understanding preemption behavior |

## Code Targets

All four arms are Python files in `inputs/` — no BLIS source modifications needed.

| Arm | File | Purpose |
|-----|------|---------|
| h-main | `.nous/campaign-3/runs/iter-4/inputs/search_nsga2_reduced.py` | NSGA-II with 5 params, pop=40, 4 gens, track HV vs exhaustive ref |
| h-control-negative | `.nous/campaign-3/runs/iter-4/inputs/search_random_reduced.py` | Random search, 200 evals, same reduced space, track HV vs exhaustive ref |
| h-robustness | `.nous/campaign-3/runs/iter-4/inputs/search_exhaustive.py` | All 900 configs, compute true Pareto front and reference HV |
| h-ablation | `.nous/campaign-3/runs/iter-4/inputs/search_nsga2_full.py` | NSGA-II on iter-3's full 9-param space (25200 configs), 200 evals |

All must:
- Evaluate via subprocess: `./blis run ... --metrics-path $TMPDIR/metrics_<id>.json`
- Parse JSON for `responses_per_sec` and `ttft_p99_ms`; derive `gpu_count = tp * num_instances`
- Record `total_kv_blocks` as 4th objective
- Compute 4D hypervolume with reference point (0, 50000, 9, 11000)
- Track convergence: record cumulative HV every 40 evaluations
- Output: `pareto_front.json` and `convergence.json` to `results/<arm-type>/`

**Reduced parameter space:**
```python
params = {
    'tp': [1, 2, 4, 8],
    'num_instances': None,  # derived: range(1, 8//tp + 1)
    'scheduler': ['fcfs', 'sjf'],
    'max_num_running_reqs': [32, 64, 128, 256, 512],
    'total_kv_blocks': [2000, 3000, 4000, 5000, 7500, 10000],
}
```

**Full parameter space (for h-ablation, same as iter 3):**
```python
params = {
    'tp': [1, 2, 4, 8],
    'num_instances': None,  # derived
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

**Fixed for all evals (all arms):**
```
--model qwen/qwen3-14b --hardware H100 --latency-model trained-physics
--num-requests 200 --rate 50 --prefix-tokens 512 --seed 42
--max-num-scheduled-tokens 4096 --long-prefill-token-threshold 0
--admission-policy always-admit --preemption-policy fcfs
```

**Additional fixed for reduced-space arms (h-main, h-control-negative, h-robustness):**
```
--block-size-in-tokens 16 --routing-policy round-robin --gpu-memory-utilization 0.9
```

**Reference point:** (0, 50000, 9, 11000) for all arms.

**IMPORTANT for h-main and h-ablation NSGA-II implementations:**
- Pop size = 40, generations = 4 (total = 40 + 4×40 = 200)
- Multi-instance configs (inst > 1): use `--routing-policy round-robin` for reduced space. For full space, allow routing sweep.
- Single-instance configs (inst == 1): always `--routing-policy round-robin`, no routing scorers.
- Crossover: uniform per-gene exchange with tp→instances constraint enforcement.
- Mutation: 15% per-gene probability.

## What I Tried That Didn't Work

1. **Preemption policy differentiation** — fcfs vs priority preemption produces IDENTICAL results. Fixed at fcfs.

2. **Routing policy differentiation even under KV stress** — <3% effect at blocks=3000. Not worth sweeping.

3. **batch={128,256,512} differentiation under KV stress** — All three produce identical results when KV is the binding constraint. Only batch=32 and batch=64 create distinct outcomes under KV stress. Under NO stress, batch=32 is strongly differentiated (10.70 vs 15.19 rps), but batch≥128 all converge.

4. **Chunked prefill threshold** — <1% effect. Fixed at 0.

5. **TPE as algorithm** — Worse than random on BLIS's conditional space (RP-4). Excluded.

6. **Multi-prefix workload at blocks=3000** — Token lengths too short to stress KV. Would need blocks < 1500.

## What I Excluded and Why

1. **Parallel evaluation workers** — 83ms/eval × 1500 total evals = 125s. No parallelism needed.

2. **Multiple seeds** — Determinism (INV-6) makes single-seed stable. No value.

3. **Reducing batch options to {32, 64, 128}** — Considered removing {256, 512} since they're equivalent to 128, but keeping them tests whether NSGA-II learns to AVOID wasting evals on equivalent configs. This is actually part of what crossover should exploit.

4. **Pop size tuning** — Kept at 40 for comparability with iter 3. Could be interesting for iter 5 (smaller pop = faster generations for smaller space).

5. **Priority-fcfs and reverse-priority schedulers** — Not tested but likely create similar tradeoffs to sjf. Left for future iterations.

## Evolution of Thinking

Started with the iter 3 suggestion to "remove non-differentiating parameters." Initial expectation: this would reduce Pareto density and create a harder problem. After probing, realized the mechanism is different — the problem difficulty (Pareto density) may not change much, but NSGA-II's EFFICIENCY at exploiting the structure improves because every crossover/mutation touches a fitness-relevant gene.

Key insight shift: The experiment tests crossover EFFICIENCY, not problem DIFFICULTY. A 900-config space at 22% coverage (200/900) is actually easier for random than the 25200-config space at 0.8% coverage (200/25200). So random might also converge faster on the reduced space. The real test is whether the RELATIVE speed advantage of NSGA-II over random increases — i.e., whether the gap grows from 40 evals to 80+ evals.

Secondary insight: The exhaustive ground truth (feasible at 74s) eliminates the iter-3 ambiguity entirely. No more "95% of own-final vs shared reference" confusion — there's one absolute reference: the true Pareto front.

## Current Status

- **Validated:** Reduced space produces meaningful Pareto tradeoffs; exhaustive sweep feasible in 74s; eval timing confirmed at 83ms average; scheduler and batch both differentiate in specific regimes; reference point valid.
- **Uncertain:** Whether 200/900 = 22% random coverage is still "too easy" for random (iter-3 was 200/25200 = 0.8%). If random trivially covers the reduced space, the gap won't grow. Counter-argument: the STRUCTURE of the reduced space (cliff interactions, conditional constraints) means random must get lucky to hit the right (tp, kv_blocks) combinations, while NSGA-II can inherit good tp from one parent and good kv from another.
- **Suggested next:** If reduced space shows clear NSGA-II speed advantage (>80 eval gap), iteration 5 should test portability across workload types (heterogeneous request sizes, different rate) or different models. If advantage is still small, consider: (a) reduce budget to 100 evals where 100/900 = 11% coverage makes random much harder, (b) add more kv_blocks granularity (every 500 blocks from 1000-10000) to increase space size while keeping all params meaningful.

## Warnings & Constraints

1. **`--metrics-path` must use `$TMPDIR`**, not `/tmp/`. Sandbox prevents writing to `/tmp/` directly.
2. **`--total-kv-blocks` is PER INSTANCE.** tp=4,i=2 at blocks=3000 means 6000 total blocks across cluster.
3. **Hardware name is exactly `H100`** — not "h100-sxm".
4. **The BLIS binary is at `./blis`** in the repo root, NOT in a worktree path. If running in a worktree, adjust path accordingly.
5. **batch≥128 produces identical results under KV stress.** When KV is the binding constraint, increasing batch has zero effect.
6. **kv={4000,5000,7500,10000} produce IDENTICAL results for tp=2,i=2** — all above the cliff. Only {2000, 3000} create stress for this tier.
7. **For the full-space arm (h-ablation), keep the routing logic:** single-instance → round-robin, multi-instance → sweep routing_policy, weighted → add scorer flags.
8. **Reference point (0, 50000, 9, 11000) must be consistent across ALL arms** for meaningful HV comparison.
9. **Convergence tracking: every 40 evals.** Record: eval count, cumulative HV, Pareto front size. Output as `convergence.json`.
