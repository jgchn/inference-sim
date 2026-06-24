# Handoff — Campaign 3, Iteration 10

## Goal

Test whether **Crossover Yield Advantage** (crossover_yield / pareto_density) is the mechanistic predictor of NSGA-II convergence gap sign across model architectures and load levels. Specifically: (1) compute crossover yield at rate=50 for both models, (2) show yield_advantage > 5 predicts positive gap and yield_advantage < 3 predicts negative gap, and (3) validate with a null model that destroys Pareto spatial structure.

## Key Discoveries

1. **Crossover yield at rate=100 is ~64-66% for both models.** Qwen: 66.18% (3309/5000 offspring are Pareto). Llama: 63.98% (3199/5000). Both have yield advantage ~9.6-10.9x over density, explaining universal NSGA-II advantage at rate=100.

2. **Cliff-free Pareto front is a perfect Cartesian product.** All 32 Pareto configs are the full cross-product of {tp:1,2,4,8} × {inst:1} × {sched:fcfs,sjf} × {batch:256,512} × {bs:16,32}. This gives 100% crossover yield (any crossover of Pareto parents stays within the Cartesian product).

3. **Full-space Pareto closure ratio is only 16.15%.** Out of 768 possible combinations of values that appear in the Pareto front, only 124 are actually Pareto-optimal. This partial closure produces the observed ~66% crossover yield.

4. **Block_size=32 dominates the Pareto front at rate=100:** 116/124 (93.5%) of Qwen Pareto configs use block_size=32. This extreme concentration is the single largest contributor to high crossover yield — when both parents have block_size=32, offspring inherit it.

5. **Rate=50 timing validated:** Qwen ~99ms/eval, Llama ~107ms/eval at rate=50. Total exhaustive sweep (1800×2 = 3600 evals) at 8 workers: ~50 seconds estimated wall time.

6. **KV stress at rate=50 is model-dependent.** Qwen tp=2,i=2,kv=3000: 78 preemptions, ttft_p99=3399ms. Llama same config: 50 preemptions, ttft_p99=1741ms. Llama is significantly less stressed (50% fewer preemptions, 49% lower TTFT).

7. **Previous iteration data available for rate=100:**
   - `runs/iter-9/results/h-robustness/all_results_qwen_r100.json` — ALL 1800 configs, objectives as [-rps, ttft_p99, gpu_count, kv_blocks]
   - `runs/iter-9/results/h-robustness/all_results_llama_r100.json` — ALL 1800 configs
   - Qwen rate=100: density=6.89%, 124 Pareto, gap=+62.3
   - Llama rate=100: density=5.89%, 106 Pareto, gap=+40.3
   - Cliff-free (Qwen, kv=10000): density=10.67%, 32 Pareto, gap=+29.0

## System Interface

- **Build:** `go build -o blis main.go` (binary already exists at `./blis`)
- **Run baseline (Qwen, rate=50):**
  ```bash
  ./blis run --model qwen/qwen3-14b --hardware H100 --latency-model trained-physics \
    --num-requests 200 --rate 50 --prefix-tokens 512 --seed 42 \
    --max-num-scheduled-tokens 4096 --long-prefill-token-threshold 0 \
    --admission-policy always-admit --preemption-policy fcfs \
    --routing-policy round-robin --gpu-memory-utilization 0.9 \
    --tp 2 --num-instances 2 --scheduler fcfs \
    --max-num-running-reqs 128 --total-kv-blocks 3000 \
    --block-size-in-tokens 16 --metrics-path $TMPDIR/baseline_qwen_r50.json
  ```
- **Output format:** JSON at `--metrics-path`, key fields: `responses_per_sec` (float), `ttft_p99_ms` (float), `preemption_count` (int). Fourth objective `gpu_count = tp * num_instances`.
- **Baseline result (Qwen, rate=50):** `responses_per_sec=13.14`, `ttft_p99_ms=3399.2`, exit code 0, wall time 99ms.

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
| `model_configs/llama-3.1-8b-instruct/config.json` | Llama: 32 layers, 8 KV heads | KV capacity differences |
| `model_configs/qwen3-14b/config.json` | Qwen: 40 layers, 8 KV heads | KV capacity differences |
| `sim/kv/cache.go:199-205` | Block allocation with BlockSizeTokens | Understanding capacity semantics |

## Code Targets

All arms are Python files in `inputs/` — no BLIS source modifications needed.

| Arm | File | Purpose |
|-----|------|---------|
| h-robustness (Qwen) | `.nous/campaign-3/runs/iter-10/inputs/search_exhaustive_r50.py` | All 1800 configs for Qwen at rate=50, saving ALL results + crossover yield |
| h-robustness (Llama) | `.nous/campaign-3/runs/iter-10/inputs/search_exhaustive_llama_r50.py` | All 1800 configs for Llama at rate=50, saving ALL results + crossover yield |
| h-main | `.nous/campaign-3/runs/iter-10/inputs/crossover_yield_analysis.py` | Compute yield advantage across all 4 conditions (reads cached results) |
| h-ablation | `.nous/campaign-3/runs/iter-10/inputs/search_algorithms_r50.py` | NSGA-II + Random on both models at rate=50 (reads cached results) |
| h-control-negative | `.nous/campaign-3/runs/iter-10/inputs/null_model_validation.py` | Random Pareto labeling yield computation (reads cached results) |
| shared | `.nous/campaign-3/runs/iter-10/inputs/blis_common.py` | Shared utilities (adapted from iter-9 with rate=50 and crossover yield functions) |

**Execution order:**
1. h-robustness scripts must run FIRST (both exhaustive sweeps) — they produce all_results JSON files
2. h-main, h-ablation, and h-control-negative can run in parallel after h-robustness (all read from cached files)

All scripts:
- Use `blis_common.py` (iter-10 version with rate=50 flag sets and crossover yield computation)
- Evaluate via subprocess: `./blis run ... --metrics-path $TMPDIR/blis_iter10_<id>_<pid>.json`
- Parse JSON for `responses_per_sec` and `ttft_p99_ms`; derive `gpu_count = tp * num_instances`
- Record `total_kv_blocks` as 4th objective (minimize)
- Compute hypervolume with reference point (0, 50000, 9, 11000)
- Output: JSON results to `results/<arm-type>/`

**h-robustness outputs (critical):**
- `results/h-robustness/all_results_qwen_r50.json` — ALL 1800 configs with objectives
- `results/h-robustness/all_results_llama_r50.json` — ALL 1800 configs with objectives
- `results/h-robustness/pareto_front_qwen_r50.json` — Pareto front + density + NDR + crossover yield metrics
- `results/h-robustness/pareto_front_llama_r50.json` — same for Llama

**h-main outputs:**
- `results/h-main/crossover_yield_comparison.json` — yield, yield_advantage, pareto_entropy, closure_ratio for all 4 conditions + gap correlation

**h-ablation outputs:**
- `results/h-ablation/convergence_qwen_r50.json` — convergence checkpoints for both algorithms
- `results/h-ablation/convergence_llama_r50.json` — convergence checkpoints for both algorithms

**h-control-negative outputs:**
- `results/h-control-negative/null_model_results.json` — null yield distribution (100 trials) for each condition + observed/null ratio

**Corrected parameter space (same as iter-9):**
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

**Crossover yield computation:**
```python
def compute_crossover_yield(all_results, n_samples=10000, seed=42):
    """Sample crossover offspring from Pareto parent pairs.
    Returns yield (fraction of valid offspring that are Pareto-optimal)."""
    # 1. Build lookup: config_key -> objectives
    # 2. Identify Pareto set via get_nondominated
    # 3. For each sample: pick 2 random Pareto parents, uniform crossover
    #    (for each param, pick one parent's value with 50% probability)
    # 4. Normalize offspring (clamp num_instances to 8//tp)
    # 5. Look up offspring in cache, check if Pareto-optimal
    # crossover_yield = pareto_offspring / total_valid_offspring
    # yield_advantage = crossover_yield / pareto_density
```

**Null model computation (h-control-negative):**
```python
def compute_null_yield(all_results, n_trials=100, n_samples=5000, seed=42):
    """Randomly assign Pareto labels (preserving density), compute yield."""
    # For each trial:
    #   1. Randomly select N configs as "Pareto" (N = actual Pareto size)
    #   2. Compute crossover yield using these random "Pareto" labels
    # Return distribution of null yields across trials
```

**NSGA-II parameters (h-ablation):**
- Pop size = 40, generations = 4, total budget = 200
- Mutation rate = 15%, NSGA-II seed = 42, random seed = 43
- Convergence checkpoints every 40 evals (5 checkpoints)
- Reference point: (0, 50000, 9, 11000)
- Same methodology as iter-9 `blis_common.py:run_algorithm_comparison()`

**Rate=100 cached data (for h-main analysis):**
- `../iter-9/results/h-robustness/all_results_qwen_r100.json`
- `../iter-9/results/h-robustness/all_results_llama_r100.json`
- Already-computed values: Qwen yield=66.18%, Llama yield=63.98%

## What I Tried That Didn't Work

1. **Preemption policy differentiation** — fcfs vs priority preemption produces IDENTICAL results. Fixed at fcfs.
2. **Routing policy differentiation** — <3% effect. Not worth sweeping.
3. **Chunked prefill threshold** — <1% effect. Fixed at 0.
4. **TPE as algorithm** — Worse than random on BLIS's conditional space (RP-4). Excluded.
5. **Pareto density as universal predictor** — Refuted in iter-8. Density does NOT determine gap sign.
6. **NDR as gap differentiator** — Refuted in iter-9. Both models have identical NDR (~35.6%) at rate=100 despite different gaps.
7. **Overall NDR predicting gap magnitude** — Refuted (RP-15). NDR equality does not imply gap equality.
8. **KV cliff as sole NSGA-II advantage mechanism** — Refuted in iter-9. Removing cliff (kv=10000) reduces gap from +62.3 to +29, but doesn't eliminate it. Multiple axes contribute.
9. **Using only Pareto front data for landscape metrics** — NDR and crossover yield require ALL configs' objective values.

## What I Excluded and Why

1. **Rate=75 exhaustive sweep** — Would give a third data point but adds 3600 evals. The rate=50 vs rate=100 contrast is the most informative (maximum separation between stress regimes for Llama).

2. **Multiple seeds for algorithm comparison** — BLIS is deterministic (INV-6). The only randomness is in algorithm sampling. Single seed gives stable results per iter-9 evidence.

3. **Llama control-negative (cliff-free space at rate=50)** — Less informative because Llama is predicted to be NOT KV-stressed at rate=50 anyway. The cliff-free manipulation would show minimal change.

4. **Fitting a regression model (yield → gap)** — With 4 data points (plus cliff-free = 5), fitting is premature. Iter-10 tests the SIGN prediction. If successful, iter-11 can collect more rate points for quantitative modeling.

5. **Per-TP-tier crossover yield** — Could decompose yield by GPU tier but adds complexity without changing the primary prediction. Deferred if needed for diagnostics.

6. **Third model architecture** — Would strengthen the claim but no third model config is readily available in the repo.

## Evolution of Thinking

**Iter-9 started with:** "NDR is the causal mechanism behind NSGA-II advantage. Higher NDR → bigger gap."
**Iter-9 found:** NDR is identical between models (~35.6%) at rate=100, yet gaps differ (+62.3 vs +40.3). NDR doesn't differentiate. KV cliff removal doesn't eliminate advantage.

**Iter-10 reframes:** The question isn't "how steep is the landscape?" (NDR) but "how reliably does crossover exploit the landscape?" (crossover yield). NDR measures local gradient existence; crossover yield measures whether NSGA-II's specific operator (recombination of Pareto parents) can leverage that gradient.

The key insight: a landscape can have identical NDR (same fraction of neighbor pairs with dominance) but different yield if the Pareto front is distributed differently. When Pareto configs cluster (forming a near-Cartesian-product), crossover between them is reliable. When they scatter uniformly, crossover is no better than random.

This explains all prior results:
- Rate=100 both models: High stress → Pareto front concentrated on high-kv/high-batch/block32 → high yield → NSGA-II wins
- Rate=50 Llama: Low stress → Pareto front scattered (many configs perform similarly) → low yield → random wins
- Rate=50 Qwen: Moderate stress → Pareto front partially concentrated → moderate yield → small positive gap
- Cliff-free: Perfect Cartesian product → 100% yield → NSGA-II wins despite high density

## Current Status

- **Validated:** Both models produce correct output at rate=50; timing ~100ms/eval; crossover yield computation works on iter-9 data; cliff-free Cartesian product structure confirmed.
- **Uncertain:** Exact crossover yield values at rate=50 for both models (the primary measurement). Whether Llama's Pareto front at rate=50 is truly scattered (high entropy) vs concentrated on different axes than expected. Whether the yield advantage threshold (predicted ~4-5) cleanly separates positive from negative gaps.
- **Suggested next:** If yield advantage correctly predicts gap sign: (a) collect rate=75 data for a 6-point dataset, (b) investigate whether there's a smooth functional form (e.g., yield_advantage → gap is monotonically increasing), (c) propose yield advantage as a practical pre-screening metric for deciding whether to use NSGA-II or random on a new configuration space.

## Warnings & Constraints

1. **`--metrics-path` must use `$TMPDIR`**, not `/tmp/`. Sandbox prevents writing to `/tmp/` directly.
2. **`--total-kv-blocks` is PER INSTANCE.** tp=4,i=2 at blocks=3000 means each instance has 3000 blocks.
3. **Hardware name is exactly `H100`** — not "h100-sxm".
4. **The BLIS binary is at `./blis`** in the repo root. Build with `go build -o blis main.go` if needed.
5. **Model name for Llama is `meta-llama/llama-3.1-8b-instruct`** (with `meta-llama/` prefix).
6. **Model name for Qwen is `qwen/qwen3-14b`** (with `qwen/` prefix).
7. **Reference point: (0, 50000, 9, 11000)** for all 4-objective analyses.
8. **Execution order matters:** h-robustness (both exhaustive sweeps) must complete before ALL other arms.
9. **h-robustness MUST save ALL per-config results** (not just Pareto front). The all_results files enable crossover yield computation and algorithm comparison from cache.
10. **Algorithm arms load results from h-robustness JSON files.** Zero additional BLIS subprocess calls needed after h-robustness completes.
11. **blis_common.py has RATE=50 hardcoded in the flag sets.** This is the critical difference from iter-9 (rate=100).
12. **Convergence tracking:** Every 40 evals for h-ablation (5 checkpoints at budget=200).
13. **The 95% threshold for gap computation:** gap = random_eval_at_95pct - nsga2_eval_at_95pct. Use linear interpolation between checkpoints.
14. **Crossover yield sampling:** Use 10000 samples per condition for statistical stability. Random seed=42 for reproducibility.
15. **Null model:** 100 trials of random Pareto labeling. Report mean ± std of null yield.
16. **Iter-9 results path:** Use relative path `../iter-9/results/h-robustness/all_results_*.json` or absolute paths based on the run directory.
