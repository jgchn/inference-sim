# Handoff — Campaign 3, Iteration 9

## Goal

Test whether **landscape structure (Neighbor Dominance Rate)** is the causal mechanism behind the NSGA-II vs random convergence gap, as proposed in RP-14. Specifically: (1) quantify NDR for both models at rate=100, (2) verify Qwen-positive and Llama-negative gap patterns extend to rate=100, and (3) prove causality by showing that removing the KV cliff (fixing kv_blocks=10000) eliminates NSGA-II's advantage for Qwen.

## Key Discoveries

1. **Both models work correctly at rate=100.** Qwen: ~56ms/eval, Llama: ~50ms/eval. All configs stay within reference point bounds (worst Qwen ttft=43761ms < 50000, worst Llama ttft=26605ms < 50000).

2. **KV cliff is dramatically sharper at rate=100.** Qwen tp=2,i=2: TTFT ratio between kv=3000 and kv=10000 is **122x** (4776ms → 39ms). Llama same config: **83x** (2564ms → 31ms). At rate=75 (iter-8), these ratios were lower (~90x for Qwen). Higher load amplifies the cliff.

3. **Preemption count increases at rate=100.** Qwen tp=2,i=2,kv=3000,bs=16: 108 preemptions (vs ~90 at rate=75). This confirms rate=100 pushes KV stress harder, creating more dominance relationships among neighbors.

4. **kv≥7500 is preemption-free at rate=100 for ALL TP tiers.** Even tp=8,i=1,kv=7500 shows 0 preemptions. This validates the "cliff-free" subspace for the control-negative arm.

5. **The cliff-free space still has meaningful TTFT differentiation.** tp=4,i=1,kv=7500: TTFT=2655ms (queueing-dominated). tp=8,i=1,kv=7500: TTFT=1122ms. tp=2,i=2,kv=10000: TTFT=39ms. Differentiation comes from throughput capacity, not KV stress — a SMOOTH gradient rather than a cliff.

6. **Previous iteration data (used as fixed references):**
   - Qwen rate=50: density=4.5%, gap=+4.6 (iter-7)
   - Qwen rate=75: density=6.06%, gap=+31.0 (iter-8)
   - Llama rate=50: density=5.33%, gap=-22.9 (iter-7)
   - Llama rate=75: density=4.83%, gap=-25.3 (iter-8)

7. **Previous exhaustive sweeps only saved Pareto fronts.** Iter-7/iter-8 results do NOT contain full per-config objective vectors. Iter-9's exhaustive sweeps must save ALL 1800 configs' results to enable NDR computation.

## System Interface

- **Build:** `go build -o blis main.go` (binary already exists at `./blis`)
- **Run baseline (Qwen, rate=100):**
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
- **Output format:** JSON at `--metrics-path`, key fields: `responses_per_sec` (float), `ttft_p99_ms` (float), `preemption_count` (int). Fourth objective `gpu_count = tp * num_instances`.
- **Baseline result (Qwen, rate=100):** `responses_per_sec=13.53`, `ttft_p99_ms=4776.3`, exit code 0, wall time 56ms.

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
| `cmd/root.go:967` | `--rate` flag | If rate parsing issues |
| `cmd/hfconfig.go:280` | Model name resolution | If model loading fails |
| `model_configs/llama-3.1-8b-instruct/config.json` | Llama: 32 layers, 8 KV heads | If KV capacity calculations seem wrong |
| `model_configs/qwen3-14b/config.json` | Qwen: 40 layers, 8 KV heads | For comparison reference |
| `sim/kv/cache.go:199-205` | Block allocation with BlockSizeTokens | Understanding capacity semantics |

## Code Targets

All arms are Python files in `inputs/` — no BLIS source modifications needed.

| Arm | File | Purpose |
|-----|------|---------|
| h-robustness (Qwen) | `.nous/campaign-3/runs/iter-9/inputs/search_exhaustive_r100.py` | All 1800 configs for Qwen at rate=100, saving ALL results |
| h-robustness (Llama) | `.nous/campaign-3/runs/iter-9/inputs/search_exhaustive_llama_r100.py` | All 1800 configs for Llama at rate=100, saving ALL results |
| h-main | `.nous/campaign-3/runs/iter-9/inputs/search_algorithms_qwen_r100.py` | NSGA-II + Random on Qwen at rate=100 (reads cached results) |
| h-ablation | `.nous/campaign-3/runs/iter-9/inputs/search_algorithms_llama_r100.py` | NSGA-II + Random on Llama at rate=100 (reads cached results) |
| h-control-negative | `.nous/campaign-3/runs/iter-9/inputs/search_cliff_free_qwen_r100.py` | NSGA-II + Random on Qwen cliff-free space (kv=10000, 300 configs, 3 objectives) |
| shared | `.nous/campaign-3/runs/iter-9/inputs/blis_common.py` | Shared utilities (adapted from iter-8 with rate=100 and NDR computation) |

**Execution order:**
1. h-robustness scripts must run FIRST (both exhaustive sweeps) — they produce all_results JSON files
2. h-main and h-ablation can run in parallel (read from all_results cache files)
3. h-control-negative can run after h-robustness (reads Qwen all_results for the kv=10000 subset)

All must:
- Use `blis_common.py` (iter-9 version with rate=100 flag sets)
- Evaluate via subprocess: `./blis run ... --metrics-path $TMPDIR/blis_iter9_<id>_<pid>.json`
- Parse JSON for `responses_per_sec` and `ttft_p99_ms`; derive `gpu_count = tp * num_instances`
- Record `total_kv_blocks` as 4th objective (minimize) — except h-control-negative which uses 3 objectives
- Compute hypervolume with appropriate reference point
- Track convergence: record cumulative HV every 40 evaluations (h-main/h-ablation) or every 20 evaluations (h-control-negative)
- Output: JSON results to `results/<arm-type>/`

**h-robustness outputs (critical):**
- `results/h-robustness/all_results_qwen_r100.json` — ALL 1800 configs with objectives
- `results/h-robustness/all_results_llama_r100.json` — ALL 1800 configs with objectives
- `results/h-robustness/pareto_front_qwen_r100.json` — Pareto front + density + HV + NDR metrics
- `results/h-robustness/pareto_front_llama_r100.json` — same for Llama

**h-main outputs:**
- `results/h-main/convergence_qwen_r100.json` — convergence checkpoints for both algorithms

**h-ablation outputs:**
- `results/h-ablation/convergence_llama_r100.json` — convergence checkpoints for both algorithms

**h-control-negative outputs:**
- `results/h-control-negative/convergence_cliff_free_qwen_r100.json` — convergence checkpoints + exhaustive metrics for cliff-free space

**Corrected parameter space (h-robustness, h-main, h-ablation — same as iter-8):**
```python
params = {
    'tp': [1, 2, 4, 8],
    'num_instances': None,  # derived: range(1, 8//tp + 1)
    'scheduler': ['fcfs', 'sjf'],
    'max_num_running_reqs': [32, 64, 128, 256, 512],
    'total_kv_blocks': [2000, 3000, 4000, 5000, 7500, 10000],
    'block_size_in_tokens': [16, 32],
}
# Total: 1800 configs
```

**Control-negative parameter space:**
```python
params = {
    'tp': [1, 2, 4, 8],
    'num_instances': None,  # derived: range(1, 8//tp + 1)
    'scheduler': ['fcfs', 'sjf'],
    'max_num_running_reqs': [32, 64, 128, 256, 512],
    'block_size_in_tokens': [16, 32],
}
# total_kv_blocks FIXED at 10000 (not a searchable parameter)
# Total: 300 configs
# 3 objectives: [-rps, ttft_p99, gpu_count]
# Reference point: (0, 50000, 9)
```

**NSGA-II parameters:**
- h-main/h-ablation: Pop size = 40, generations = 4 (total = 200). Mutation rate = 15%. Seed = 42.
- h-control-negative: Pop size = 20, generations = 4 (total = 100). Mutation rate = 15%. Seed = 42.
- Random seeds: 43 for all random search arms.

**NDR computation (in h-robustness):**
```python
def compute_ndr(all_results):
    """Compute Neighbor Dominance Rate for all Hamming-1 pairs.
    
    Returns:
      overall_ndr: fraction of Hamming-1 pairs with dominance relationship
      per_axis_ndr: dict mapping param_name -> fraction for pairs differing on that axis
      per_axis_counts: dict mapping param_name -> (dominated_pairs, total_pairs)
    """
    # For each pair of configs differing in exactly 1 parameter:
    #   Check if dominates(a, b) or dominates(b, a)
    #   Track which parameter axis differs
```

**Reference points:**
- 4-objective (h-robustness/h-main/h-ablation): (0, 50000, 9, 11000)
- 3-objective (h-control-negative): (0, 50000, 9)

**HV lower bounds:**
- 4-objective: (-100, 0, 1, 2000)
- 3-objective: (-100, 0, 1)

## What I Tried That Didn't Work

1. **Preemption policy differentiation** — fcfs vs priority preemption produces IDENTICAL results. Fixed at fcfs.

2. **Routing policy differentiation even under KV stress** — <3% effect at blocks=3000. Not worth sweeping.

3. **batch={128,256,512} differentiation under KV stress** — All three produce identical results when KV is the binding constraint.

4. **Chunked prefill threshold** — <1% effect. Fixed at 0.

5. **TPE as algorithm** — Worse than random on BLIS's conditional space (RP-4). Excluded.

6. **Pareto density as universal predictor** — Refuted in iter-8. Density does NOT determine gap sign across models.

7. **Monotonic density-vs-rate assumption** — Does NOT hold across models (increases for Qwen, decreases for Llama).

8. **Using only Pareto front data for landscape metrics** — NDR requires ALL configs' objective values, not just Pareto-optimal ones. Previous iterations only saved Pareto fronts. Must re-run exhaustive sweep saving all results.

## What I Excluded and Why

1. **Rate=50 and rate=75 exhaustive re-runs** — Would give NDR at all 3 rates, but adds 7200 evals (~400s) for data we already have convergence gaps for. The rate=100 measurement alone is sufficient to test the mechanism because we have the iter-7/iter-8 gaps as fixed reference. If NDR explains the rate=100 gaps, the causal claim is supported.

2. **Multiple seeds** — Determinism (INV-6) makes single-seed stable. No value.

3. **Different NSGA-II pop sizes for control-negative** — Would test interaction between pop size and landscape structure, but adds complexity. Pop=20 with 4 gens gives comparable coverage ratio to pop=40/4gens on the larger space.

4. **Llama control-negative arm** — Since Llama already shows random-wins at all rates tested, a cliff-free comparison is less informative (gap should remain negative regardless). The critical test is Qwen, where removing the cliff should FLIP the gap from positive to zero/negative.

5. **Fitting a functional form (NDR → gap)** — Requires 5+ data points. With only 2 NDR measurements (rate=100 for both models) and 4 historical gap values without NDR, fitting is premature. Deferred to iter-10 if NDR proves predictive.

6. **gpu_memory_utilization sweep** — Confirmed <2% effect in iter-3. Fixed at 0.9.

## Evolution of Thinking

Started from iter-8's conclusion: "Pareto density is NOT a universal predictor. Model architecture is an independent factor." The suggested next was: "investigate what structural property beyond density explains the residual."

RP-14 already proposes the mechanism: "sharp KV threshold effects create gradients that NSGA-II's crossover can exploit." But this was stated as a post-hoc explanation in iter-8, never directly tested.

The key insight for iter-9: to prove a CAUSAL mechanism, you need an ABLATION — remove the proposed cause and show the effect vanishes. The KV cliff is the proposed cause. Fixing kv_blocks=10000 removes it completely. If NSGA-II's advantage vanishes for Qwen in the cliff-free space, the mechanism is confirmed.

Additionally, NDR provides a QUANTITATIVE metric that captures the mechanism. Unlike density (which measures the OUTPUT of landscape structure), NDR measures the STRUCTURE itself: how many single-step moves create dominance relationships. This is directly related to crossover exploitability.

The rate=100 data point serves dual purposes: (1) extends the time series to confirm Qwen-positive/Llama-negative patterns persist at extreme load, and (2) provides the raw data for NDR computation at a rate where the KV cliff is maximally sharp (providing the strongest signal).

## Current Status

- **Validated:** Both models produce correct output at rate=100; all configs within reference point bounds; timing ~56ms/eval; cliff-free space (kv≥7500) confirmed preemption-free for all TP tiers.
- **Uncertain:** Exact NDR values for both models at rate=100 (predicted: Qwen > Llama, kv_blocks axis dominant for Qwen). Exact gap magnitude at rate=100 (predicted: Qwen > +31, Llama < -25). Whether cliff-free gap is exactly 0 or slightly negative.
- **Suggested next:** If NDR correctly predicts gap sign and the control-negative confirms causality: (a) compute NDR at rate=50/75 by re-running exhaustive sweeps with full results, (b) fit NDR→gap functional form across all 6 data points, (c) validate on a third model architecture if available.

## Warnings & Constraints

1. **`--metrics-path` must use `$TMPDIR`**, not `/tmp/`. Sandbox prevents writing to `/tmp/` directly.
2. **`--total-kv-blocks` is PER INSTANCE.** tp=4,i=2 at blocks=3000 means each instance has 3000 blocks.
3. **Hardware name is exactly `H100`** — not "h100-sxm".
4. **The BLIS binary is at `./blis`** in the repo root.
5. **Model name for Llama is `meta-llama/llama-3.1-8b-instruct`** (with `meta-llama/` prefix).
6. **Model name for Qwen is `qwen/qwen3-14b`** (with `qwen/` prefix).
7. **Reference point consistency:** (0, 50000, 9, 11000) for 4-objective arms; (0, 50000, 9) for 3-objective control-negative.
8. **Execution order matters:** h-robustness (both exhaustive sweeps) must complete before ALL other arms.
9. **h-robustness MUST save ALL per-config results** (not just Pareto front). This is a critical difference from iter-7/iter-8. The all_results files enable NDR computation and allow algorithm arms to run from cache.
10. **Algorithm arms load results from h-robustness JSON files.** The eval_config function should check an in-memory cache populated from the all_results file at startup. Zero additional BLIS subprocess calls needed.
11. **h-control-negative uses 3 objectives and a DIFFERENT reference point.** The HV computation, Pareto dominance, and convergence tracking must use the 3-objective formulation. Do NOT accidentally use the 4-objective reference point.
12. **blis_common.py has RATE=100 hardcoded in the flag sets.** Different from iter-8 (rate=75).
13. **Convergence tracking:** Every 40 evals for h-main/h-ablation (5 checkpoints). Every 20 evals for h-control-negative (5 checkpoints at budget=100).
14. **The 95% threshold for gap computation:** gap = random_eval_at_95pct - nsga2_eval_at_95pct. Use linear interpolation between checkpoints if 95% falls between tracked evaluations.
15. **NDR computation can be expensive** (O(n²) pairs for 1800 configs). Optimize by iterating only over Hamming-1 pairs, not all pairs. For 6 parameters with varying cardinalities, total Hamming-1 pairs ≈ 1800 × (3+varied+1+4+5+1)/2 ≈ ~12000 pairs per model. Fast enough.
