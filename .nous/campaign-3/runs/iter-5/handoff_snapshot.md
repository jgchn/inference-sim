# Handoff — Campaign 3, Iteration 5

## Goal

Test whether including `block_size_in_tokens` as a searchable parameter (correcting iter-4's erroneous exclusion) amplifies NSGA-II's convergence advantage over random search. The corrected space has 1800 configs (6 parameters) vs iter-4's 900 (5 parameters). The block_size × kv_blocks × tp three-way interaction should give NSGA-II's crossover operator more exploitable structure, while the lower coverage ratio (11.1% vs 22%) hampers random sampling.

## Key Discoveries

1. **block_size=32 creates 92x TTFT improvement for tp=8 at kv=3000.** Confirmed: block_size=16 → (rps=21.35, ttft=2323ms, 107 preemptions); block_size=32 → (rps=26.28, ttft=25ms, 0 preemptions). Mechanism: effective KV capacity = 3000×32=96k tokens (above stress threshold) vs 3000×16=48k tokens (below threshold).

2. **Corrected space = 1800 configs, exhaustible in ~128s.** Average eval time is 71ms (faster than iter-4's 83ms — likely variance or different config mix). Breakdown: tp=1 (960), tp=2 (480), tp=4 (240), tp=8 (120).

3. **Coverage at budget=200: 11.1% (vs iter-4's 22%).** This halved coverage should make random significantly less effective while NSGA-II's guided search exploits the interaction structure.

4. **Reference point (0, 50000, 9, 11000) remains valid.** Worst probed: tp=1,inst=1,batch=32,kv=2000,bs=16 → ttft_p99=41934ms (below 50000 reference). kv range [2000-10000] within 11000 reference.

5. **Rate=100 creates much stronger saturation.** tp=2,inst=1,kv=3000,bs=16 at rate=100 → rps=8.54, ttft=13270ms, 114 preemptions. But tp=4,inst=2,kv=5000,bs=16 at rate=100 → rps=26.78, ttft=29ms, 0 preemptions. Dramatic differentiation across tiers. Reserved for iter-6 portability testing.

6. **block_size=32 is still insufficient at rate=100 for tp=4,inst=1.** tp=4,inst=1,kv=3000,bs=32 at rate=100 → rps=18.45, ttft=3370ms, 77 preemptions. The 96k token capacity (per instance) isn't enough when throughput demand doubles. This confirms rate affects the stress threshold.

7. **4 objectives unchanged from iter-4.** Using (neg_rps, ttft_p99, gpu_count, total_kv_blocks). block_size is a parameter that affects performance but its "cost" is captured indirectly via kv_blocks — lower blocks achievable when block_size=32.

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
| `cmd/root.go:978` | `--total-kv-blocks` flag definition (default 1000000) | If flag name/default issues |
| `cmd/root.go:28` | `--block-size-in-tokens` flag (default 16) | If block size flag issues |
| `cmd/root.go:184` | `metricsPath` variable | If `--metrics-path` flag issues |
| `cmd/root.go:919` | `--tp` flag | If TP issues |
| `cmd/root.go:908` | `--num-instances` flag | If instance count issues |
| `sim/kv/cache.go:199-205` | Block allocation with BlockSizeTokens | Understanding capacity semantics |
| `sim/kv/cache.go:222-253` | KV block pre-check triggering preemption | If preemption counts seem wrong |
| `sim/batch_formation.go:225-304` | `preemptForTokens()` — victim selection | Understanding preemption behavior |

## Code Targets

All four arms are Python files in `inputs/` — no BLIS source modifications needed.

| Arm | File | Purpose |
|-----|------|---------|
| h-robustness | `.nous/campaign-3/runs/iter-5/inputs/search_exhaustive.py` | All 1800 configs, compute true Pareto front and reference HV |
| h-main | `.nous/campaign-3/runs/iter-5/inputs/search_nsga2_corrected.py` | NSGA-II with 6 genes (includes block_size), pop=40, 4 gens, track HV vs exhaustive ref |
| h-control-negative | `.nous/campaign-3/runs/iter-5/inputs/search_random_corrected.py` | Random search, 200 evals, corrected space, track HV vs exhaustive ref |
| h-ablation | `.nous/campaign-3/runs/iter-5/inputs/search_budget100.py` | NSGA-II (pop=20, 4 gens) + random (100 evals) on corrected space |

All must:
- Evaluate via subprocess: `./blis run ... --metrics-path $TMPDIR/metrics_<id>.json`
- Parse JSON for `responses_per_sec` and `ttft_p99_ms`; derive `gpu_count = tp * num_instances`
- Record `total_kv_blocks` as 4th objective (minimize)
- Compute 4D hypervolume with reference point (0, 50000, 9, 11000)
- Track convergence: record cumulative HV every 40 evaluations (every 20 for h-ablation)
- Output: `pareto_front.json` and `convergence.json` to `results/<arm-type>/`

**Corrected parameter space (all arms):**
```python
params = {
    'tp': [1, 2, 4, 8],
    'num_instances': None,  # derived: range(1, 8//tp + 1)
    'scheduler': ['fcfs', 'sjf'],
    'max_num_running_reqs': [32, 64, 128, 256, 512],
    'total_kv_blocks': [2000, 3000, 4000, 5000, 7500, 10000],
    'block_size_in_tokens': [16, 32],
}
```

Gene encoding: `(tp_idx, inst_idx, sched_idx, batch_idx, kv_idx, block_size_idx)` — 6 genes.

**Fixed for all evals (all arms):**
```
--model qwen/qwen3-14b --hardware H100 --latency-model trained-physics
--num-requests 200 --rate 50 --prefix-tokens 512 --seed 42
--max-num-scheduled-tokens 4096 --long-prefill-token-threshold 0
--admission-policy always-admit --preemption-policy fcfs
--routing-policy round-robin --gpu-memory-utilization 0.9
```

**Reference point:** (0, 50000, 9, 11000) for all arms.

**NSGA-II parameters (h-main):** Pop size = 40, generations = 4 (total = 200). Mutation rate = 15%.
**NSGA-II parameters (h-ablation):** Pop size = 20, generations = 4 (total = 100). Mutation rate = 15%.

**Crossover for corrected space:** Uniform per-gene exchange (6 genes), then repair:
- Clamp tp_idx to [0,3]
- Clamp inst_idx to [0, 8//tp - 1] (enforces TP×instances ≤ 8 constraint)
- Clamp block_size_idx to [0,1]
- Other indices clamped to valid ranges

## What I Tried That Didn't Work

1. **Preemption policy differentiation** — fcfs vs priority preemption produces IDENTICAL results. Fixed at fcfs.

2. **Routing policy differentiation even under KV stress** — <3% effect at blocks=3000. Not worth sweeping.

3. **batch={128,256,512} differentiation under KV stress** — All three produce identical results when KV is the binding constraint. Only batch=32 and batch=64 create distinct outcomes under KV stress. Under NO stress, batch=32 is strongly differentiated (10.70 vs 15.19 rps), but batch≥128 all converge.

4. **Chunked prefill threshold** — <1% effect. Fixed at 0.

5. **TPE as algorithm** — Worse than random on BLIS's conditional space (RP-4). Excluded.

6. **Multi-prefix workload at blocks=3000** — Token lengths too short to stress KV. Would need blocks < 1500.

7. **Fixing block_size=16 as "non-differentiating"** (iter-4's reduced space) — WRONG. block_size=32 is dramatically differentiating for tp=8 (92x TTFT improvement at kv=3000). This was iter-4's key refuted assumption.

## What I Excluded and Why

1. **Parallel evaluation workers** — 71ms/eval average, exhaustive takes 128s. No parallelism needed.

2. **Multiple seeds** — Determinism (INV-6) makes single-seed stable. No value.

3. **Rate=100 variant** — Validated that it creates much stronger saturation (8.54 rps at tp=2,i=1 vs 13.14 at rate=50). Reserved for iter-6 portability testing after confirming the algorithm advantage on the corrected space.

4. **kv_memory_tokens as objective** (blocks × block_size) — Considered replacing raw total_kv_blocks with effective memory cost. Rejected for this iteration to maintain comparability with iter-4's objective formulation. Could be tested in iter-6.

5. **gpu_memory_utilization sweep** — Confirmed <2% effect across {0.85, 0.9, 0.95} in iter-3. Fixed at 0.9.

6. **Priority-fcfs and reverse-priority schedulers** — Not tested but likely create similar tradeoffs to sjf. Left for future iterations.

## Evolution of Thinking

Started from iter-4's refutation: block_size was wrongly excluded from the "reduced space." The h-ablation showed full-space NSGA-II (including block_size) exceeds the reduced-space exhaustive Pareto front by 10.4%. This means the iter-4 experiment was testing NSGA-II on an artificially constrained space.

Key insight: The experiment doesn't need to change the OBJECTIVE formulation — just include block_size as a searchable parameter. block_size affects performance (through effective KV capacity) but its "cost" is captured through the existing kv_blocks objective. Configs with block_size=32 achieve good performance at LOWER kv_blocks, making them Pareto-optimal on the cost dimension.

The three-way interaction (tp × kv_blocks × block_size) is the mechanism NSGA-II should exploit: crossover can inherit "use block_size=32" from one parent and "use low kv_blocks" from another, discovering that this combination works for high-TP tiers. Random must discover each such combination independently.

## Current Status

- **Validated:** Corrected space (1800 configs) feasible to exhaust in 128s; block_size interaction confirmed empirically; reference point valid; eval timing stable at 71ms average.
- **Uncertain:** Whether 11.1% coverage is "hard enough" for random. iter-4 showed random reaches 96.1% at 22% coverage. At 11.1%, random might still reach ~93-95% if the Pareto front is accessible from diverse starting points. The prediction (random < 95%) depends on block_size creating genuine access barriers.
- **Suggested next:** If the gap exceeds 60 evals, iter-6 should test rate portability (rate=100, where saturation creates even stronger differentiation). If gap is still ~47 evals (not amplified), consider: (a) reduce budget to 100 evals as the primary test (the h-ablation should answer this), or (b) test whether NSGA-II's advantage generalizes to a different model (different KV capacity thresholds).

## Warnings & Constraints

1. **`--metrics-path` must use `$TMPDIR`**, not `/tmp/`. Sandbox prevents writing to `/tmp/` directly.
2. **`--total-kv-blocks` is PER INSTANCE.** tp=4,i=2 at blocks=3000 means each instance has 3000 blocks.
3. **Hardware name is exactly `H100`** — not "h100-sxm".
4. **The BLIS binary is at `./blis`** in the repo root.
5. **batch≥128 produces identical results under KV stress.** When KV is the binding constraint, increasing batch has zero effect.
6. **block_size_in_tokens is the 6th gene.** Gene encoding: `(tp_idx, inst_idx, sched_idx, batch_idx, kv_idx, block_size_idx)`. block_size_idx ∈ {0, 1} mapping to {16, 32}.
7. **Reference point (0, 50000, 9, 11000) must be consistent across ALL arms** for meaningful HV comparison.
8. **Convergence tracking:** Every 40 evals for h-main/h-control-negative/h-robustness. Every 20 evals for h-ablation (budget=100).
9. **h-ablation outputs two convergence traces** (NSGA-II and random) in a single results directory. Name them `convergence_nsga2.json` and `convergence_random.json`.
10. **The corrected space includes multi-instance configs but fixes routing to round-robin** (routing was shown non-differentiating in earlier iterations). No routing scorer flags needed.
