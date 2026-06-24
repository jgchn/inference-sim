# Handoff — Campaign 3, Iteration 7

## Goal

Test whether NSGA-II's convergence speed advantage over random search transfers across model architectures. Compare Llama-3.1-8B (smaller, 32 layers, lower KV per token) against Qwen3-14B (larger, 40 layers, higher KV per token) at the same rate=50 and same 1800-config parameter space. The prediction: NSGA-II advantage is portable but reduced in magnitude because Llama's weaker KV stress interaction increases Pareto density.

## Key Discoveries

1. **Llama-3.1-8B has weaker block_size interaction than Qwen3-14B.** At tp=8,kv=3000: block_size=32 vs 16 gives 40x TTFT improvement for Llama (22ms vs 883ms) vs 92x for Qwen (25ms vs 2323ms). The smaller model needs ~80% of Qwen's KV per token (32/40 layers ratio), shifting the stress threshold lower.

2. **Llama-3.1-8B has higher throughput ceiling.** Best config at rate=50: tp=8,i=1,kv=3000,bs=32 → 31.67 rps (vs ~26 rps for Qwen at same config). The smaller model processes tokens faster.

3. **Worst case for Llama at rate=50: tp=1,i=1,batch=32,kv=2000,bs=16 → rps=5.51, ttft=24764ms.** Compute-limited (0 preemptions, scheduling delay dominates). This is higher rps than Qwen's worst (3.43 rps at rate=100), confirming the smaller model's compute advantage.

4. **Reference point (0, 50000, 9, 11000) remains valid for Llama.** Worst ttft probed: 24764ms < 50000. Best rps: 31.67 (negated: -31.67 > -100). Max gpu_count: 8 < 9. Max kv_blocks: 10000 < 11000.

5. **Scheduler differentiation for Llama is moderate.** At tp=2,i=2,kv=3000,bs=16: fcfs gives rps=19.62, ttft=1741ms vs sjf gives rps=20.15, ttft=1908ms (~3% rps difference, ~10% TTFT difference).

6. **KV stress gradient for Llama at tp=8,bs=16:** kv=2000→149 preemptions/2607ms ttft; kv=3000→79/883ms; kv=4000→25/373ms; kv=5000→0/~22ms (interpolated from tp=4,i=2,kv=5000). Stress-free threshold: ~4500 blocks at bs=16 or ~2500 blocks at bs=32 for tp=8.

7. **Wall-clock time: ~88ms per eval for Llama.** Total budget: 2600 evals × 88ms ≈ 229s (under 4 minutes).

## System Interface

- **Build:** `go build -o blis main.go` (binary already exists at `./blis`)
- **Run baseline (Llama, rate=50):**
  ```bash
  ./blis run --model meta-llama/llama-3.1-8b-instruct --hardware H100 --latency-model trained-physics \
    --num-requests 200 --rate 50 --prefix-tokens 512 --seed 42 \
    --max-num-scheduled-tokens 4096 --long-prefill-token-threshold 0 \
    --admission-policy always-admit --preemption-policy fcfs \
    --block-size-in-tokens 16 --routing-policy round-robin \
    --gpu-memory-utilization 0.9 \
    --tp 2 --num-instances 2 --scheduler fcfs \
    --max-num-running-reqs 128 --total-kv-blocks 3000 \
    --metrics-path $TMPDIR/baseline_llama.json
  ```
- **Output format:** JSON at `--metrics-path`, key fields: `responses_per_sec` (float), `ttft_p99_ms` (float), `preemption_count` (int). Fourth objective `gpu_count = tp * num_instances`.
- **Baseline result (Llama, rate=50):** `responses_per_sec=19.62`, `ttft_p99_ms=1741.3`, `preemption_count=50`, exit code 0

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
| `cmd/hfconfig.go:280` | Model name resolution | If model loading fails |
| `model_configs/llama-3.1-8b-instruct/config.json` | Llama architecture: 32 layers, 8 KV heads | If KV capacity calculations seem wrong |
| `model_configs/qwen3-14b/config.json` | Qwen architecture: 40 layers, 8 KV heads | For comparison reference |
| `sim/kv/cache.go:199-205` | Block allocation with BlockSizeTokens | Understanding capacity semantics |
| `sim/kv/cache.go:222-253` | KV block pre-check triggering preemption | If preemption counts seem wrong |
| `sim/batch_formation.go:225-304` | `preemptForTokens()` — victim selection | Understanding preemption behavior |

## Code Targets

All four arms are Python files in `inputs/` — no BLIS source modifications needed.

| Arm | File | Purpose |
|-----|------|---------|
| h-robustness | `.nous/campaign-3/runs/iter-7/inputs/search_exhaustive_llama.py` | All 1800 configs with Llama at rate=50, compute true Pareto front and reference HV |
| h-main | `.nous/campaign-3/runs/iter-7/inputs/search_nsga2_llama.py` | NSGA-II with Llama at rate=50, pop=40, 4 gens, track HV vs exhaustive ref |
| h-control-negative | `.nous/campaign-3/runs/iter-7/inputs/search_random_llama.py` | Random search with Llama at rate=50, 200 evals, track HV vs exhaustive ref |
| h-ablation | `.nous/campaign-3/runs/iter-7/inputs/search_qwen_reverify.py` | NSGA-II + Random on Qwen3-14B at rate=50 (same-session re-verification) |

All must:
- Use `blis_common.py` (adapted for iter-7 with configurable model name)
- Evaluate via subprocess: `./blis run ... --metrics-path $TMPDIR/metrics_<id>.json`
- Parse JSON for `responses_per_sec` and `ttft_p99_ms`; derive `gpu_count = tp * num_instances`
- Record `total_kv_blocks` as 4th objective (minimize)
- Compute 4D hypervolume with reference point (0, 50000, 9, 11000)
- Track convergence: record cumulative HV every 40 evaluations
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

**Fixed for h-robustness, h-main, h-control-negative (Llama arms):**
```
--model meta-llama/llama-3.1-8b-instruct --hardware H100 --latency-model trained-physics
--num-requests 200 --rate 50 --prefix-tokens 512 --seed 42
--max-num-scheduled-tokens 4096 --long-prefill-token-threshold 0
--admission-policy always-admit --preemption-policy fcfs
--routing-policy round-robin --gpu-memory-utilization 0.9
```

**Fixed for h-ablation (Qwen re-verification):**
Same as above but `--model qwen/qwen3-14b`.

**Reference point:** (0, 50000, 9, 11000) for ALL arms.

**NSGA-II parameters:** Pop size = 40, generations = 4 (total = 200). Mutation rate = 15%.

**Crossover:** Uniform per-gene exchange (6 genes), then repair:
- Clamp tp_idx to [0,3]
- Clamp inst_idx to [0, 8//tp - 1] (enforces TP×instances ≤ 8 constraint)
- Clamp block_size_idx to [0,1]
- Other indices clamped to valid ranges

**MC hypervolume parameters:**
- MC_SAMPLES = 200,000
- MC_SEED = 42
- HV_LO = (-100.0, 0.0, 1.0, 2000.0) — lower bounds for MC sampling box

## What I Tried That Didn't Work

1. **Preemption policy differentiation** — fcfs vs priority preemption produces IDENTICAL results. Fixed at fcfs.

2. **Routing policy differentiation even under KV stress** — <3% effect at blocks=3000. Not worth sweeping.

3. **batch={128,256,512} differentiation under KV stress** — All three produce identical results when KV is the binding constraint.

4. **Chunked prefill threshold** — <1% effect. Fixed at 0.

5. **TPE as algorithm** — Worse than random on BLIS's conditional space (RP-4). Excluded.

6. **Monotonic density-vs-rate assumption** — Density INCREASES with rate (4.5% → 6.06% → 6.89%), opposite of initial prediction. The mechanism: higher load opens new tradeoff regions rather than concentrating the frontier.

7. **Monotonic gap-vs-rate assumption** — NSGA-II gap is non-monotonic across rates (91→48→57 for rates 50→75→100). Rate-specific landscape structure matters more than a simple density relationship.

## What I Excluded and Why

1. **Parallel evaluation workers** — 88ms/eval average, total ~229s sequential. No parallelism needed within budget.

2. **Multiple seeds** — Determinism (INV-6) makes single-seed stable. No value.

3. **Different rate for Llama** — Rate=50 matches Qwen iter-5's well-characterized baseline (density=4.5%, gap=91). Using the same rate isolates the model variable.

4. **Reduced kv_blocks range for more stress** — Would stress Llama more (matching Qwen's stress profile) but changes the config count, breaking the 1800-config comparability.

5. **gpu_memory_utilization sweep** — Confirmed <2% effect across {0.85, 0.9, 0.95} in iter-3. Fixed at 0.9.

6. **Rate portability re-test with Llama** — Would test Llama at rate=75/100, but the primary question (model portability) is cleanest with a single rate. Deferred to iter-8 if model portability is confirmed.

7. **Qwen3-14B exhaustive re-run** — Using iter-5's established HV (5.857e10) as reference for h-ablation rather than re-running 1800 exhaustive evals. Determinism guarantees the same result.

## Evolution of Thinking

Started from iter-6's suggestion: "test model portability (different model with different KV thresholds, e.g., Llama-3.1-8B with lower memory footprint)." The key question was whether the convergence advantage is a property of NSGA-II on this PARAMETER SPACE (portable) or a property of NSGA-II on Qwen3-14B's specific LANDSCAPE (model-specific).

Probing revealed:
- Llama has the SAME block_size interaction at kv=3000 (stress with bs=16, no stress with bs=32)
- But the interaction is WEAKER (40x TTFT improvement vs 92x for Qwen)
- The stress gradient is smoother (preemptions: 149→79→25→0 for kv=2000→3000→4000→5000 at tp=8,bs=16)

This led to the prediction: NSGA-II advantage transfers (same interaction structure) but is reduced (weaker binary dominance → higher density → random more competitive). This is testable by comparing gap and density directly against Qwen at the same rate.

Added h-ablation (Qwen re-verification in same session) to eliminate potential confounds from different numpy versions, binary changes, etc. between iter-5 and iter-7.

## Current Status

- **Validated:** Llama-3.1-8B evaluations work correctly (88ms wall-clock, proper metrics output, model name resolution confirmed); reference point valid; block_size interaction exists but is weaker; KV stress gradient confirmed across kv_blocks values; same parameter space (1800 configs) applies.
- **Uncertain:** Exact Pareto density for Llama at rate=50 — predicted 6-12% based on the weaker block_size interaction making fewer configs dominated, but could be lower if Llama's higher throughput ceiling concentrates the rps dimension.
- **Suggested next:** If model portability is confirmed with reduced magnitude (gap < 91 evals for Llama): iter-8 should test whether the density→gap relationship holds as a QUANTITATIVE PREDICTOR (can we predict the gap from density alone, regardless of model/rate?). If portability is refuted (gap ≈ Qwen's 91): investigate what landscape structure Llama creates that compensates for lower density.

## Warnings & Constraints

1. **`--metrics-path` must use `$TMPDIR`**, not `/tmp/`. Sandbox prevents writing to `/tmp/` directly.
2. **`--total-kv-blocks` is PER INSTANCE.** tp=4,i=2 at blocks=3000 means each instance has 3000 blocks.
3. **Hardware name is exactly `H100`** — not "h100-sxm".
4. **The BLIS binary is at `./blis`** in the repo root.
5. **Model name for Llama is `meta-llama/llama-3.1-8b-instruct`** (with `meta-llama/` prefix, resolves to `model_configs/llama-3.1-8b-instruct/`).
6. **Model name for Qwen is `qwen/qwen3-14b`** (with `qwen/` prefix).
7. **block_size_in_tokens is the 6th gene.** Gene encoding: `(tp_idx, inst_idx, sched_idx, batch_idx, kv_idx, block_size_idx)`. block_size_idx ∈ {0, 1} mapping to {16, 32}.
8. **Reference point (0, 50000, 9, 11000) must be consistent across ALL arms for meaningful HV comparison.**
9. **Convergence tracking:** Every 40 evals for all arms.
10. **h-ablation uses iter-5's Qwen exhaustive HV (5.857e10) as reference** — do NOT re-run exhaustive sweep for Qwen. Just run NSGA-II and random, compare convergence against the known reference.
11. **The corrected space includes multi-instance configs but fixes routing to round-robin.** No routing scorer flags needed.
12. **blis_common.py must accept a MODEL parameter** (unlike iter-6 which hardcoded qwen/qwen3-14b). The `eval_config` function needs a `model` kwarg or the model should be configurable at module level.
13. **BLIS binary path for executor:** Use the working copy `./blis` or build fresh with `go build -o blis main.go`. Do NOT hardcode absolute paths.
14. **h-ablation Qwen verification:** Uses SAME RNG seeds as iter-5 (nsga2 seed=42, random seed=43). Determinism means results should be BIT-IDENTICAL to iter-5 if nothing has changed.
