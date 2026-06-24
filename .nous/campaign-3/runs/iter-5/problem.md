# Problem Framing — Iteration 5

## Research Question

Does including `block_size_in_tokens` as a searchable parameter (correcting iter-4's erroneous exclusion) amplify NSGA-II's convergence speed advantage over random search? The three-way interaction (tp × kv_blocks × block_size) should provide NSGA-II's crossover operator with more exploitable structure, while the lower coverage ratio (11.1% vs iter-4's 22%) hampers random sampling.

Relevant source files:
- `sim/kv/cache.go:199-205` — Block allocation uses `BlockSizeTokens` to determine effective capacity
- `sim/kv/cache.go:222-253` — KV precheck triggers preemption when capacity exceeded
- `cmd/root.go:978` — `--total-kv-blocks` flag definition
- `cmd/root.go:28` — `--block-size-in-tokens` flag (default 16)

## System Interface

- **Build:** `go build -o blis main.go` (binary already present at `./blis`)
- **CLI flags relevant to experiment:**
  - `--tp` (cmd/root.go:919): tensor parallelism degree
  - `--num-instances` (cmd/root.go:908): cluster instance count
  - `--scheduler` (cmd/root.go:924): scheduling policy [fcfs, sjf]
  - `--max-num-running-reqs` (cmd/root.go:981): max batch size
  - `--total-kv-blocks` (cmd/root.go:978): KV cache block count per instance
  - `--block-size-in-tokens` (cmd/root.go:28): tokens per KV block (default 16)
  - `--metrics-path` (cmd/root.go:184): write JSON metrics to file
- **Output format:** JSON at `--metrics-path`. Key fields: `responses_per_sec`, `ttft_p99_ms`, `preemption_count`.
- **Derived objectives:** `gpu_count = tp * num_instances` (minimize), `total_kv_blocks` (minimize, represents memory cost).

## Baseline Command

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

## Baseline Validation

Exit code 0. Output at `$TMPDIR/baseline.json`. Key metric: `responses_per_sec=13.14`, `ttft_p99_ms=3399.2`. Consistent with iter-4 validation.

Critical validation of the block_size interaction:
- tp=8, kv=3000, block_size=16 → rps=21.35, ttft_p99=2323.1ms, preemptions=107
- tp=8, kv=3000, block_size=32 → rps=26.28, ttft_p99=25.5ms, preemptions=0

This 92x TTFT difference from a single parameter change (same kv_blocks cost) confirms block_size is a dominant parameter for high-TP tiers (RP-5, RP-8).

## Experimental Conditions

**Corrected search space (1800 configurations):**
```python
params = {
    'tp': [1, 2, 4, 8],
    'num_instances': derived,  # range(1, 8//tp + 1)
    'scheduler': ['fcfs', 'sjf'],
    'max_num_running_reqs': [32, 64, 128, 256, 512],
    'total_kv_blocks': [2000, 3000, 4000, 5000, 7500, 10000],
    'block_size_in_tokens': [16, 32],
}
```

**4 objectives (same formulation as iter-4 for comparability):**
1. responses_per_sec (maximize → negate for minimization)
2. ttft_p99_ms (minimize)
3. gpu_count = tp × num_instances (minimize)
4. total_kv_blocks (minimize)

**Reference point:** (0, 50000, 9, 11000) — unchanged.

**Fixed flags (all arms):**
```
--model qwen/qwen3-14b --hardware H100 --latency-model trained-physics
--num-requests 200 --rate 50 --prefix-tokens 512 --seed 42
--max-num-scheduled-tokens 4096 --long-prefill-token-threshold 0
--admission-policy always-admit --preemption-policy fcfs
--routing-policy round-robin --gpu-memory-utilization 0.9
```

Note: `--block-size-in-tokens` and `--total-kv-blocks` are now BOTH swept.

### Arm 1: h-robustness (Exhaustive Ground Truth)

Run all 1800 configurations. Compute true Pareto front, Pareto density, and reference hypervolume. Estimated wall time: ~128s (71ms avg per eval).

### Arm 2: h-main (NSGA-II, 6-gene, budget=200)

NSGA-II with pop_size=40, 4 generations (200 total evals). Gene encoding adds block_size_idx (6th gene). Crossover: uniform per-gene exchange with tp→instances constraint repair. Mutation: 15% per gene. Track convergence every 40 evals.

### Arm 3: h-control-negative (Random, budget=200)

Uniform random sampling from the 1800-config space, 200 evals. Track convergence every 40 evals.

### Arm 4: h-ablation (Budget=100, both algorithms)

NSGA-II with pop_size=20, 4 generations (100 total evals) AND random with 100 evals. Tests whether halving the budget (5.6% coverage) amplifies the gap. Track convergence every 20 evals.

## Success Criteria

1. **Primary (h-main vs h-control-negative):** NSGA-II's convergence speed gap to 95% of exhaustive HV exceeds iter-4's 47-eval gap. Target: >60 eval gap.
2. **Secondary (h-main final quality):** NSGA-II achieves >97% of exhaustive HV at eval=200.
3. **Ablation (h-ablation):** At budget=100, NSGA-II final HV exceeds random final HV by >5% of exhaustive reference (vs ~1.7% at budget=200 in iter-4).
4. **Robustness (h-robustness):** Pareto density < 26.4% (iter-4's density), confirming the block_size interaction creates dominance relationships that thin the front.

## Constraints

- All evaluations use `--metrics-path $TMPDIR/...` (sandbox constraint).
- Budget = 200 evals for main/control arms, 100 for ablation.
- Total wall time per arm ≤ 180s.
- Reference point (0, 50000, 9, 11000) unchanged for cross-iteration comparability.
- NSGA-II seed = 42 for reproducibility.

## Prior Knowledge

- **RP-5:** block_size is dominant for high-TP configurations (multiplies effective KV capacity).
- **RP-6:** NSGA-II's convergence advantage over random is ~47 evals at budget=200 on 900-config space.
- **RP-7:** Adding kv_blocks as searchable reduces Pareto density from ~65% to ~26%.
- **RP-8:** Effective KV capacity = total_kv_blocks × block_size. Stress threshold for tp=8 at rate=50 is ~80-96k tokens.
- **Iter-4 h-ablation refutation:** block_size=32 unlocks tp=8 at low kv_blocks (3000×32=96k tokens vs 3000×16=48k tokens). Full space outperforms reduced space by 10.4% HV.
