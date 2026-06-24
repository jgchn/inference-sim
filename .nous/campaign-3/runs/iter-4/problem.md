# Problem Framing — Iteration 4

## Research Question

Does removing non-differentiating parameters from the search space amplify NSGA-II's convergence speed advantage over random search?

Iteration 3 established that NSGA-II achieves +4.4% final hypervolume over random on the 4-objective space, but convergence speed advantage was only 40 evaluations (1 generation) on a shared reference. The hypothesis: non-differentiating parameters (routing_policy, block_size_in_tokens, gpu_memory_utilization) dilute crossover signal by adding search dimensions without informative gradients. Removing them should create a focused space where crossover exclusively operates on the structurally meaningful genes (tp, num_instances, scheduler, max_num_running_reqs, total_kv_blocks), amplifying convergence speed.

Key mechanism: NSGA-II's crossover exchanges gene values between parents. When 5 of 9 genes are non-informative (changing them doesn't change fitness), crossover events that flip only non-informative genes are wasted evolutionary steps. Reducing from 9 to 5 parameters means every crossover event has higher probability of modifying a fitness-relevant gene.

Source files implementing the mechanism:
- `sim/batch_formation.go:225-304` — preemption logic that makes kv_blocks interact with tp/instances
- `sim/kv/cache.go:222-253` — KV block pre-check creating the cliff behavior
- `cmd/root.go:946` — `--total-kv-blocks` flag (per-instance semantics)
- `sim/scheduler.go` — FCFS vs SJF ordering affecting preemption patterns

## System Interface

- **Build:** `go build -o blis main.go` (binary pre-built at `./blis`)
- **CLI flags relevant to experiment:**
  - `--tp` (int): Tensor parallelism degree. Defined at `cmd/root.go:919`.
  - `--num-instances` (int): Cluster instance count. Defined at `cmd/root.go:908`.
  - `--scheduler` (string): Instance scheduler {fcfs, sjf}. Defined at `cmd/root.go:935`.
  - `--max-num-running-reqs` (int): Max concurrent requests/batch. Defined at `cmd/root.go:930`.
  - `--total-kv-blocks` (int): KV cache blocks PER INSTANCE. Defined at `cmd/root.go:946`.
  - `--metrics-path` (string): Native JSON output for metrics. Defined at `cmd/root.go:184`.
- **Output format:** JSON at `--metrics-path`. Key fields: `responses_per_sec` (float), `ttft_p99_ms` (float), `preemption_count` (int).
- **Objectives:** Maximize `responses_per_sec`, minimize `ttft_p99_ms`, minimize `gpu_count` (= tp × num_instances), minimize `total_kv_blocks`.

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

Ran successfully. Exit code 0. Output at `$TMPDIR/baseline.json`:
- `responses_per_sec=13.14`
- `ttft_p99_ms=3399.2`
- `preemption_count=78`

Additional timing validation: average evaluation time across 5 diverse configs is 83ms. 900 exhaustive evaluations complete in ~74 seconds.

## Experimental Conditions

### Arm 1: h-main — NSGA-II on reduced space (200 evals)

**Reduced parameter space (5 parameters, 900 total configs):**
- `tp`: [1, 2, 4, 8]
- `num_instances`: derived, range(1, 8//tp + 1)
- `scheduler`: [fcfs, sjf]
- `max_num_running_reqs`: [32, 64, 128, 256, 512]
- `total_kv_blocks`: [2000, 3000, 4000, 5000, 7500, 10000]

**Fixed (non-differentiating):**
- `block_size_in_tokens`: 16
- `routing_policy`: round-robin
- `gpu_memory_utilization`: 0.9

NSGA-II with pop_size=40, 4 generations (40 init + 4×40 = 200 total evals). Track cumulative HV every 40 evals against exhaustive-reference HV.

### Arm 2: h-control-negative — Random on reduced space (200 evals)

Same reduced parameter space. Uniform random sampling, 200 evals. Track cumulative HV every 40 evals against same exhaustive-reference HV.

### Arm 3: h-robustness — Exhaustive sweep (900 evals, ground truth)

Evaluate ALL 900 configs in the reduced space. Compute the TRUE Pareto front and reference hypervolume. This provides the absolute ground truth that arms 1 and 2 are measured against, eliminating the iter-3 ambiguity of "95% of which reference?"

### Arm 4: h-ablation — NSGA-II on full space (200 evals)

Same as iter-3's NSGA-II (9 parameters, 25200 configs). Used as control to confirm that the reduced space amplifies the advantage. Track cumulative HV every 40 evals. Reference: the reduced-space exhaustive HV is not directly comparable; instead compare convergence RATE (eval-to-95%) between arms 1 and 4 using each arm's own exhaustive reference.

**Note:** Since arm 4's space is different, we compute its HV using the 4D reference point (0, 50000, 9, 11000) — same as iter 3. The comparison metric is convergence speed (which eval reaches 95% of respective ceiling), not absolute HV magnitude.

## Success Criteria

1. **Primary:** NSGA-II on reduced space (arm 1) reaches 95% of exhaustive HV at least 80 evaluations (2 generations) before random (arm 2). This would demonstrate that parameter reduction amplifies the convergence speed gap from the 40-eval gap observed in iter 3.

2. **Secondary:** NSGA-II on reduced space (arm 1) reaches 95% of exhaustive HV in fewer evaluations than NSGA-II on full space (arm 4) reaches 95% of its own ceiling. This confirms the dilution mechanism.

3. **Sanity check:** Exhaustive sweep (arm 3) produces a Pareto density < 30% across the full 900-config space (confirming non-trivial search difficulty).

## Constraints

- Total BLIS invocations: 900 (exhaustive) + 200 (NSGA-II reduced) + 200 (random) + 200 (NSGA-II full) = 1500. At 83ms average, total wall time ≈ 125 seconds. Well within 3-minute budget.
- All evals use `--seed 42` for determinism (INV-6).
- `--metrics-path` must use `$TMPDIR` (sandbox constraint).
- Reference point for 4-obj HV: (0, 50000, 9, 11000) — consistent with iter 3.
- `--total-kv-blocks` is PER INSTANCE — critical for interpreting cross-tier results.

## Prior Knowledge

- **RP-5:** TP, num_instances, AND total_kv_blocks are dominant parameters. Routing, block_size, gpu_memory_utilization are non-differentiating. (Confirmed iter 3)
- **RP-6:** NSGA-II's convergence advantage manifests as final quality (+4-5% HV), not speed. (Iter 3: 40-eval gap on shared reference)
- **RP-7:** 4-obj Pareto density ~12-23% (reduced from 65% in 3-obj). (Confirmed iter 3)
- **RP-3:** Batch ≥ 256 is threshold for tier-optimal TTFT at rate=50. (Iter 2/3)
- **RP-4:** TPE performs worse than random on conditional spaces. (Excluded from this iter)

**New probed findings for this iteration:**
- Batch=32 at blocks=5000 (no KV stress): (10.70 rps, 7979 ttft) — strongly dominated. Batch IS differentiating in the reduced space.
- Batch=512 at blocks=5000: (15.19 rps, 39 ttft) — confirms batch=32 creates massive scheduling delay.
- SJF vs FCFS at batch=64, blocks=3000: SJF (13.64, 4369) vs FCFS (13.14, 3435) — scheduler creates tradeoffs under joint batch+KV constraint.
- Within-tier (tp=2,i=2) Pareto density: 30% (18/60), but many are degenerate (equivalent metrics). Actual distinct outcomes: ~5 unique Pareto points.
- tp=4,i=2 at blocks=3000: (21.03 rps, 1192 ttft) — 8 GPU config with moderate stress.
