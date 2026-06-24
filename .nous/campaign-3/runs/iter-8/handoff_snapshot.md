# Handoff — Campaign 3, Iteration 8

## Goal

Test whether Pareto density is a universal quantitative predictor of the NSGA-II vs random convergence gap, independent of what causes the density (model architecture vs arrival rate). Add two new data points by running both models at rate=75, then assess whether all four (density, gap) pairs — from iter-7 at rate=50 and iter-8 at rate=75 — fall on a single monotonically decreasing curve.

## Key Discoveries

1. **Both models work correctly at rate=75.** Qwen3-14B: ~61ms/eval, Llama-3.1-8B: ~50ms/eval. All configs stay within reference point bounds (worst Qwen ttft=43186ms < 50000, worst Llama ttft=26002ms < 50000).

2. **KV stress is stronger at rate=75 than rate=50.** Qwen tp=2,i=2,kv=3000,bs=16: preemptions=90 at rate=75 (vs ~50 at rate=50). Higher load increases KV contention, potentially creating more dominance relationships.

3. **The metric range at rate=75 is wide.** Qwen rps spans 3.43-29.92 (8.7x), ttft spans 26-43186ms (1660x). Llama rps spans 5.53-37.39 (6.8x), ttft spans 23-26002ms (1130x). Sufficient variation for meaningful Pareto discrimination.

4. **Llama worst case at rate=75 (tp=1,i=1,bs=16,kv=2000,batch=32) has 2 preemptions.** The extreme KV stress at rate=75 even at tp=1 (which was 0-preemption at rate=50) shows rate=75 does push the system harder for Llama.

5. **From iter-7 (same implementation, same BLIS binary):**
   - Qwen rate=50: density=4.5% (81 Pareto-optimal / 1800 total), gap=+4.6 evals
   - Llama rate=50: density=5.33% (96 / 1800), gap=-22.9 evals, exhaustive HV=7.754e10

6. **From RP-10:** Qwen rate=75 expected density ~6.06%, Qwen rate=100 ~6.89%. Pattern is monotonically increasing with rate. Llama rate=75 density is unknown — first measurement.

7. **Iter-7 h-ablation confirmed binary stability.** NSGA-II eval=40 matched iter-5 exactly (87.4%), proving no BLIS binary changes between iterations.

## System Interface

- **Build:** `go build -o blis main.go` (binary already exists at `./blis`)
- **Run baseline (Qwen, rate=75):**
  ```bash
  ./blis run --model qwen/qwen3-14b --hardware H100 --latency-model trained-physics \
    --num-requests 200 --rate 75 --prefix-tokens 512 --seed 42 \
    --max-num-scheduled-tokens 4096 --long-prefill-token-threshold 0 \
    --admission-policy always-admit --preemption-policy fcfs \
    --routing-policy round-robin --gpu-memory-utilization 0.9 \
    --tp 2 --num-instances 2 --scheduler fcfs \
    --max-num-running-reqs 128 --total-kv-blocks 3000 \
    --block-size-in-tokens 16 --metrics-path $TMPDIR/baseline_qwen_r75.json
  ```
- **Output format:** JSON at `--metrics-path`, key fields: `responses_per_sec` (float), `ttft_p99_ms` (float), `preemption_count` (int). Fourth objective `gpu_count = tp * num_instances`.
- **Baseline result (Qwen, rate=75):** `responses_per_sec=13.46`, `ttft_p99_ms=4277.6`, exit code 0

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
| h-robustness (Qwen) | `.nous/campaign-3/runs/iter-8/inputs/search_exhaustive_qwen_r75.py` | All 1800 configs with Qwen at rate=75 |
| h-robustness (Llama) | `.nous/campaign-3/runs/iter-8/inputs/search_exhaustive_llama_r75.py` | All 1800 configs with Llama at rate=75 |
| h-main | `.nous/campaign-3/runs/iter-8/inputs/search_algorithms_qwen_r75.py` | NSGA-II + Random on Qwen at rate=75 |
| h-ablation | `.nous/campaign-3/runs/iter-8/inputs/search_algorithms_llama_r75.py` | NSGA-II + Random on Llama at rate=75 |

**Execution order:** h-robustness scripts must run FIRST (both exhaustive sweeps), then h-main and h-ablation can run in parallel (they read from h-robustness results).

All must:
- Use `blis_common.py` (iter-8 version with rate=75 flag sets)
- Evaluate via subprocess: `./blis run ... --metrics-path $TMPDIR/blis_iter8_<id>_<pid>.json`
- Parse JSON for `responses_per_sec` and `ttft_p99_ms`; derive `gpu_count = tp * num_instances`
- Record `total_kv_blocks` as 4th objective (minimize)
- Compute 4D hypervolume with reference point (0, 50000, 9, 11000)
- Track convergence: record cumulative HV every 40 evaluations
- Output: JSON results to `results/<arm-type>/`

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

**NSGA-II parameters:** Pop size = 40, generations = 4 (total = 200). Mutation rate = 15%. Seeds: NSGA-II=42, Random=43.

**Reference point:** (0, 50000, 9, 11000) for ALL arms.

## What I Tried That Didn't Work

1. **Preemption policy differentiation** — fcfs vs priority preemption produces IDENTICAL results. Fixed at fcfs.

2. **Routing policy differentiation even under KV stress** — <3% effect at blocks=3000. Not worth sweeping.

3. **batch={128,256,512} differentiation under KV stress** — All three produce identical results when KV is the binding constraint.

4. **Chunked prefill threshold** — <1% effect. Fixed at 0.

5. **TPE as algorithm** — Worse than random on BLIS's conditional space (RP-4). Excluded.

6. **Monotonic density-vs-rate assumption** — Density INCREASES with rate (4.5% → 6.06% → 6.89%), opposite of initial prediction. The mechanism: higher load opens new tradeoff regions rather than concentrating the frontier.

7. **Monotonic gap-vs-rate assumption** — NSGA-II gap is non-monotonic across rates (91→48→57 for rates 50→75→100) in earlier iterations. But those used a DIFFERENT NSGA-II implementation. Iter-7's implementation shows Qwen gap=+4.6 at rate=50 (not 91), so the earlier gap numbers are not comparable.

8. **Assuming iter-5 gap (91 evals) is reproducible** — Iter-7 h-ablation showed gap=4.6 evals with iter-7's implementation (single offspring per tournament vs iter-5's paired offspring). The absolute gap value is implementation-dependent; only RELATIVE comparisons within the same implementation are valid.

## What I Excluded and Why

1. **Rate=100 condition** — Would give a third rate point per model, but 4 data points (2 models × 2 rates) is sufficient to test whether density alone predicts gap. Rate=100 deferred to iter-9 if the relationship holds.

2. **Parallel evaluation workers** — ~60ms/eval average at rate=75, total ~240s sequential for exhaustive sweeps. Manageable within budget.

3. **Multiple seeds** — Determinism (INV-6) makes single-seed stable. No value.

4. **Different NSGA-II pop sizes** — Would test whether pop size interacts with density, but adds complexity. Fixed at pop=40 matching iter-7.

5. **gpu_memory_utilization sweep** — Confirmed <2% effect in iter-3. Fixed at 0.9.

6. **Iter-7 data re-run** — Iter-7 results (rate=50 for both models) are used as fixed reference points. No need to re-run since BLIS binary is unchanged and deterministic.

## Evolution of Thinking

Started from iter-7's suggestion: "test whether the density→gap relationship holds as a QUANTITATIVE PREDICTOR."

The critical insight from iter-7: the density breakeven is ~5%. Below 5%, NSGA-II has a small positive gap (barely faster). Above 5%, random is decisively faster. But we only have TWO data points, and they conflate model architecture with density (Qwen=4.5% vs Llama=5.33% — different models at the same rate).

To disentangle model from density, iter-8 uses RATE as the independent variable. By running both models at rate=75 (where density should be higher for both), we get:
- Qwen at two densities (rate=50 → 4.5%, rate=75 → ~6%)
- Llama at two densities (rate=50 → 5.33%, rate=75 → ?)

If density is the universal predictor, both models at their higher density should show more negative gaps, and points from different models at the SAME density should show similar gaps. If not, model architecture is an independent factor.

## Current Status

- **Validated:** Both models produce correct output at rate=75; pipeline works end-to-end; reference point valid for rate=75 conditions; timing validated (~60ms/eval).
- **Uncertain:** Exact Pareto density for Llama at rate=75 (predicted > 5.33% from RP-10 pattern, but magnitude unknown). Exact shape of the density→gap function (could be linear, exponential, step-function near 5%).
- **Suggested next:** If density is confirmed as universal predictor: iter-9 should (a) add rate=100 data points for both models (6 total points), and (b) fit a functional form (linear, logistic, etc.) to establish a closed-form predictor. If density is NOT a universal predictor (model-specific offsets exist): investigate what structural property beyond density explains the residual.

## Warnings & Constraints

1. **`--metrics-path` must use `$TMPDIR`**, not `/tmp/`. Sandbox prevents writing to `/tmp/` directly.
2. **`--total-kv-blocks` is PER INSTANCE.** tp=4,i=2 at blocks=3000 means each instance has 3000 blocks.
3. **Hardware name is exactly `H100`** — not "h100-sxm".
4. **The BLIS binary is at `./blis`** in the repo root.
5. **Model name for Llama is `meta-llama/llama-3.1-8b-instruct`** (with `meta-llama/` prefix).
6. **Model name for Qwen is `qwen/qwen3-14b`** (with `qwen/` prefix).
7. **Reference point (0, 50000, 9, 11000) must be consistent across ALL arms.**
8. **Execution order matters:** h-robustness (both exhaustive sweeps) must complete before h-main and h-ablation, which read the exhaustive HV from the robustness results files.
9. **h-robustness outputs TWO files:** `pareto_front_qwen_r75.json` and `pareto_front_llama_r75.json`, both in `results/h-robustness/`.
10. **The iter-7 reference data (rate=50) is NOT re-run.** Use these fixed values: Qwen density=4.5%, gap=+4.6; Llama density=5.33%, gap=-22.9, HV=7.754e10. Qwen iter-5 HV=5.857e10.
11. **blis_common.py has RATE=75 hardcoded in the flag sets.** Do not mix with iter-7's blis_common.py which uses rate=50.
12. **Convergence tracking:** Every 40 evals for algorithm arms. This gives 5 checkpoints (40, 80, 120, 160, 200).
13. **The cache_key_prefix ensures no cache collision** between Qwen and Llama evaluations within the same process. Different scripts use different prefixes.
