# Problem Framing — Iteration 9

## Research Question

What structural property of the BLIS configuration landscape determines whether NSGA-II or random search converges faster to the Pareto front?

Iterations 7-8 established that **Pareto density is NOT a universal predictor** of the convergence gap: Qwen consistently favors NSGA-II (+4.6 at rate=50, +31.0 at rate=75) while Llama consistently favors random (-22.9 at rate=50, -25.3 at rate=75), regardless of their density values. RP-14 proposes that "landscape structure (gradient exploitability)" is the causal driver, specifically that **KV threshold effects create dominance gradients** that NSGA-II's crossover can exploit.

This iteration tests RP-14's mechanism directly by:
1. Quantifying landscape structure via **Neighbor Dominance Rate (NDR)** — the fraction of single-parameter-change neighbor pairs where one config Pareto-dominates the other.
2. Extending the model comparison to rate=100 (3rd rate data point per model).
3. Running a **causal ablation**: removing the KV cliff entirely (fix kv_blocks=10000) and verifying that NSGA-II's advantage vanishes.

Relevant source implementing the KV threshold mechanism:
- `sim/kv/cache.go:199-205` — block allocation with BlockSizeTokens
- `sim/kv/cache.go:126` — GetCachedBlocks hash lookup

## System Interface

- **Build:** `go build -o blis main.go` (binary already exists at `./blis`)
- **CLI flags (code evidence):**
  - `--tp`: `cmd/root.go:919`
  - `--num-instances`: `cmd/root.go:908`
  - `--rate`: `cmd/root.go:967`
  - `--total-kv-blocks`: `cmd/root.go:978`
  - `--block-size-in-tokens`: `cmd/root.go:28`
  - `--metrics-path`: `cmd/root.go:184`
  - `--scheduler`: `cmd/root.go:934`
  - `--max-num-running-reqs`: `cmd/root.go:945`
- **Output:** JSON at `--metrics-path`. Key fields: `responses_per_sec` (float), `ttft_p99_ms` (float), `preemption_count` (int). Fourth objective `gpu_count = tp * num_instances` is derived.
- **Objectives (all minimized):** `[-responses_per_sec, ttft_p99_ms, gpu_count, total_kv_blocks]`
- **Reference point:** `(0, 50000, 9, 11000)` for 4-objective arms; `(0, 50000, 9)` for 3-objective control-negative.

## Baseline Command

```bash
./blis run --model qwen/qwen3-14b --hardware H100 --latency-model trained-physics \
  --num-requests 200 --rate 100 --prefix-tokens 512 --seed 42 \
  --max-num-scheduled-tokens 4096 --long-prefill-token-threshold 0 \
  --admission-policy always-admit --preemption-policy fcfs \
  --routing-policy round-robin --gpu-memory-utilization 0.9 \
  --tp 2 --num-instances 2 --scheduler fcfs \
  --max-num-running-reqs 128 --total-kv-blocks 3000 \
  --block-size-in-tokens 16 --metrics-path $TMPDIR/baseline_qwen_r100.json
```

## Baseline Validation

Executed at rate=100 for both models. Exit code 0 for all probes. Key observations:

| Config | Model | rps | ttft_p99 | preemptions | wall time |
|--------|-------|-----|----------|-------------|-----------|
| tp=2,i=2,kv=3000,bs=16 | Qwen | 13.53 | 4776.3ms | 108 | 56ms |
| tp=2,i=2,kv=3000,bs=16 | Llama | 20.85 | 2564.0ms | 90 | ~50ms |
| tp=2,i=2,kv=10000,bs=16 | Qwen | 16.28 | 39.2ms | 0 | ~56ms |
| tp=2,i=2,kv=10000,bs=16 | Llama | 24.15 | 31.3ms | 0 | ~50ms |
| tp=1,i=1,kv=2000,bs=16 (worst) | Qwen | 3.43 | 43760.6ms | 0 | ~56ms |
| tp=1,i=1,kv=2000,bs=16 (worst) | Llama | 5.54 | 26605.2ms | 2 | ~50ms |
| tp=4,i=1,kv=7500,bs=16 | Qwen | 19.56 | 2655.2ms | 0 | ~56ms |
| tp=8,i=1,kv=7500,bs=16 | Qwen | 30.24 | 1122.1ms | 0 | ~56ms |

**KV cliff magnitude at rate=100:** Qwen TTFT ratio (kv=3000 vs 10000) = 122x. Llama = 83x. Both within reference point bounds (worst TTFT < 50000ms).

**Cliff-free validation:** kv≥7500 produces 0 preemptions for ALL TP configurations at rate=100 (verified for tp=4 and tp=8).

## Experimental Conditions

### h-robustness: Exhaustive sweep with full results (both models, rate=100)

All 1800 configs evaluated for each model. Unlike iter-7/iter-8 which only saved Pareto fronts, this sweep saves ALL per-config objective vectors. This enables computation of landscape structure metrics.

**Output:** For each model: all_results JSON (1800 entries with config + objectives), Pareto front, density, exhaustive HV, **Neighbor Dominance Rate (NDR)** with per-axis breakdown.

NDR definition: For all config pairs differing in exactly 1 parameter (Hamming distance 1), compute the fraction where one Pareto-dominates the other. Per-axis NDR computes this separately for each of the 6 parameters.

### h-main: Algorithm comparison on Qwen at rate=100

NSGA-II (pop=40, 4 gens, total=200 evals, seed=42) vs random (200 evals, seed=43) on the full 1800-config Qwen space. All evaluations served from h-robustness cache (0 additional BLIS calls). Convergence tracked every 40 evals. Reference point: (0, 50000, 9, 11000).

### h-ablation: Algorithm comparison on Llama at rate=100

Same as h-main but for Llama. Same parameters, same cache mechanism.

### h-control-negative: Cliff-free Qwen (kv fixed=10000, 3 objectives)

Qwen at rate=100, but with kv_blocks FIXED at 10000 (not a searchable parameter). Search space:
- tp: [1,2,4,8], num_instances: derived, scheduler: [fcfs,sjf], max_num_running_reqs: [32,64,128,256,512], block_size_in_tokens: [16,32]
- Total: 300 configs (15 tier combos × 2 × 5 × 2)
- 3-objective formulation: [-rps, ttft_p99, gpu_count] (kv_blocks removed as objective since constant)
- Reference point: (0, 50000, 9)
- NSGA-II: pop=20, 4 gens, total=100 evals, seed=42
- Random: 100 evals, seed=43
- Exhaustive sweep of all 300 configs (from h-robustness cache subset)

## Success Criteria

1. **h-robustness:** Qwen NDR > Llama NDR at rate=100 (confirms structural difference exists)
2. **h-main:** NSGA-II convergence gap > 0 for Qwen at rate=100 (gap defined as: random_95pct_eval - nsga2_95pct_eval)
3. **h-ablation:** NSGA-II convergence gap < 0 for Llama at rate=100 (random reaches 95% HV first)
4. **h-control-negative:** NSGA-II convergence gap ≈ 0 or negative for cliff-free Qwen (removing KV cliff removes NSGA-II advantage)
5. **Consistency check:** The kv_blocks axis has the highest per-axis NDR for Qwen (confirming it's the primary source of exploitable gradient)

## Constraints

- All configs must produce ttft_p99 < 50000ms (reference point bound). Validated: worst case = 43761ms.
- Determinism (INV-6): single seed, byte-identical results.
- Wall time for exhaustive sweeps: ~56ms × 3600 = ~200s (within budget).
- Algorithm runs use cached results: ~0s additional BLIS calls.
- Reference point (0, 50000, 9, 11000) for 4-objective arms; (0, 50000, 9) for 3-objective control-negative.
- NSGA-II pop=40/4gens (h-main/h-ablation) and pop=20/4gens (h-control-negative) match coverage ratios.

## Prior Knowledge

- **RP-11:** NSGA-II advantage is determined by model architecture (landscape structure), not Pareto density. Qwen=positive, Llama=negative.
- **RP-12:** No universal density breakeven exists. The density value alone is insufficient to predict the gap sign.
- **RP-14:** The causal driver is landscape structure (gradient exploitability). Sharp KV threshold effects favor NSGA-II.
- **RP-5/RP-8:** KV threshold is the dominant differentiator in BLIS's search space. Effective capacity = total_kv_blocks × block_size_in_tokens.
- **RP-6:** NSGA-II's convergence advantage is decisive in the full 6-parameter 4-objective space (>80-eval gap at rate=50).
- **Iter-8 reference data:** Qwen rate=75 density=6.06%, gap=+31.0. Llama rate=75 density=4.83%, gap=-25.3.
- **Iter-7 reference data:** Qwen rate=50 density=4.5%, gap=+4.6. Llama rate=50 density=5.33%, gap=-22.9.
