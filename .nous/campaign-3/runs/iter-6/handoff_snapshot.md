# Handoff — Campaign 3, Iteration 6

## Goal

Test whether NSGA-II's convergence advantage over random search (91-eval gap at rate=50, iter-5) generalizes to higher load regimes (rate=100 and rate=75). The corrected 6-parameter, 4-objective, 1800-config space is held constant — only the arrival rate changes. This tests portability and the monotonicity hypothesis: higher load → more saturation → sparser Pareto front → larger algorithm advantage.

## Key Discoveries

1. **Rate=100 dramatically stresses mid-range configs.** tp=2,inst=2,kv=3000,bs=16: ttft_p99=4776ms at rate=100 (vs 3399ms at rate=50). Preemptions: 108 (vs lower at rate=50). The same config with block_size=32: ttft=39ms, 0 preemptions. The block_size interaction is STRONGER at rate=100.

2. **Best config at rate=100: tp=8,inst=1,kv=5000,bs=32 → rps=32.01, ttft=27.5ms.** Compared to rate=50 where the best was ~26 rps, rate=100 allows higher-TP configs to demonstrate their throughput advantage since more requests arrive to saturate them.

3. **Worst config at rate=100: tp=1,inst=1,batch=32,kv=2000,bs=16 → rps=3.43, ttft=43760ms.** This is throughput-limited (not KV-limited — 0 preemptions). The single tp=1 GPU can only process ~3.4 rps regardless of incoming rate. The TTFT spike is purely from scheduling delay (43742ms).

4. **Reference point (0, 50000, 9, 11000) remains valid at rate=100.** Worst ttft_p99 probed: 43760ms < 50000. Best rps: 32.01 (negated: -32.01 > -100 lower bound). Max gpu_count: 8 < 9. Max kv_blocks: 10000 < 11000.

5. **Rate=75 intermediate probes confirm progressive stress.** tp=2,inst=2,kv=3000,bs=16: ttft=4278ms at rate=75 (between rate=50's 3399ms and rate=100's 4776ms). 90 preemptions (between rate=50's ~52 and rate=100's 108). Monotonic progression confirmed.

6. **Eval wall-clock time at rate=100: 52-76ms.** Same ballpark as rate=50's 71ms average. The 1800-config exhaustive sweep at rate=100 will take ~130s. Total experiment budget (4400 evals for both rates): ~317s sequential, within 6 minutes.

7. **tp=4,inst=2,kv=3000,bs=32 at rate=100 = tp=4,inst=2,kv=5000,bs=16 at rate=100.** Both: rps=26.78, ttft=29.6ms. This confirms block_size=32 at kv=3000 (effective 96k tokens) ≈ block_size=16 at kv=5000 (effective 80k tokens) for this tier. The dominance relationship: kv=3000,bs=32 dominates kv=5000,bs=16 on the kv_blocks objective.

## System Interface

- **Build:** `go build -o blis main.go` (binary already exists at `./blis`)
- **Run baseline (rate=100):**
  ```bash
  ./blis run --model qwen/qwen3-14b --hardware H100 --latency-model trained-physics \
    --num-requests 200 --rate 100 --prefix-tokens 512 --seed 42 \
    --max-num-scheduled-tokens 4096 --long-prefill-token-threshold 0 \
    --admission-policy always-admit --preemption-policy fcfs \
    --block-size-in-tokens 16 --routing-policy round-robin \
    --gpu-memory-utilization 0.9 \
    --tp 2 --num-instances 2 --scheduler fcfs \
    --max-num-running-reqs 128 --total-kv-blocks 3000 \
    --metrics-path $TMPDIR/baseline.json
  ```
- **Output format:** JSON at `--metrics-path`, key fields: `responses_per_sec` (float), `ttft_p99_ms` (float), `preemption_count` (int). Fourth objective `gpu_count = tp * num_instances`.
- **Baseline result (rate=100):** `responses_per_sec=13.53`, `ttft_p99_ms=4776.3`, exit code 0

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
| h-robustness | `.nous/campaign-3/runs/iter-6/inputs/search_exhaustive_rate100.py` | All 1800 configs at rate=100, compute true Pareto front and reference HV |
| h-main | `.nous/campaign-3/runs/iter-6/inputs/search_nsga2_rate100.py` | NSGA-II at rate=100, pop=40, 4 gens, track HV vs exhaustive ref |
| h-control-negative | `.nous/campaign-3/runs/iter-6/inputs/search_random_rate100.py` | Random search at rate=100, 200 evals, track HV vs exhaustive ref |
| h-ablation | `.nous/campaign-3/runs/iter-6/inputs/search_rate75.py` | Exhaustive + NSGA-II + Random at rate=75 (intermediate point) |

All must:
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

**Fixed for all evals (rate=100 arms):**
```
--model qwen/qwen3-14b --hardware H100 --latency-model trained-physics
--num-requests 200 --rate 100 --prefix-tokens 512 --seed 42
--max-num-scheduled-tokens 4096 --long-prefill-token-threshold 0
--admission-policy always-admit --preemption-policy fcfs
--routing-policy round-robin --gpu-memory-utilization 0.9
```

**Fixed for h-ablation (rate=75):** Same as above but `--rate 75`.

**Reference point:** (0, 50000, 9, 11000) for ALL arms at ALL rates.

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

3. **batch={128,256,512} differentiation under KV stress** — All three produce identical results when KV is the binding constraint. Only batch=32 and batch=64 create distinct outcomes under KV stress. Under NO stress, batch=32 is strongly differentiated (10.70 vs 15.19 rps), but batch≥128 all converge.

4. **Chunked prefill threshold** — <1% effect. Fixed at 0.

5. **TPE as algorithm** — Worse than random on BLIS's conditional space (RP-4). Excluded.

6. **Multi-prefix workload at blocks=3000** — Token lengths too short to stress KV. Would need blocks < 1500.

7. **Fixing block_size=16 as "non-differentiating"** (iter-4's reduced space) — WRONG. block_size=32 is dramatically differentiating for high-TP (92x TTFT improvement at kv=3000 for tp=8, 120x improvement at rate=100 for tp=2,inst=2 kv=3000).

## What I Excluded and Why

1. **Parallel evaluation workers** — 72ms/eval average, exhaustive takes ~130s. No parallelism needed for correctness; sequential is fine within budget.

2. **Multiple seeds** — Determinism (INV-6) makes single-seed stable. No value.

3. **Different model (e.g., Llama)** — Suggested in iter-5 as alternative next step. Deferred to iter-7 since rate portability is the higher-priority test per iter-5's "Suggested next."

4. **kv_memory_tokens as objective** (blocks × block_size) — Would better capture true memory cost but breaks comparability with iter-5's objective formulation. Deferred.

5. **gpu_memory_utilization sweep** — Confirmed <2% effect across {0.85, 0.9, 0.95} in iter-3. Fixed at 0.9.

6. **Budget=100 sub-experiment** — Iter-5's h-ablation already established the budget=100 behavior at rate=50 (3.1% gap). Rate portability at budget=200 is the primary question; budget=100 at different rates is secondary.

7. **rate=150 or rate=200** — Probed but not needed. rate=100 already creates dramatic differentiation (43760ms worst vs 27.5ms best TTFT). Higher rates risk configs where even the best cannot keep up, collapsing the performance dimension.

## Evolution of Thinking

Started from iter-5's clear next step: "test rate portability." The key question was whether stronger saturation helps or hurts NSGA-II relative to random.

Initial hypothesis: stronger saturation helps NSGA-II because it concentrates the Pareto front, giving crossover more exploitable structure.

Probe findings confirmed: at rate=100, the landscape differentiation is MUCH stronger (120x TTFT range at rate=100 vs ~1600x at rate=50 — actually similar ratios). The critical finding is that MORE configs become stressed at rate=100 (tp=2,inst=2,kv=3000,bs=16 goes from moderate stress at rate=50 to heavy stress at rate=100). This should reduce Pareto density.

Added rate=75 ablation to test monotonicity — probes confirm progressive stress (ttft goes 3399 → 4278 → 4776 as rate goes 50 → 75 → 100 for the same config). This provides a three-point curve for the relationship between load intensity and algorithm advantage.

Design decision: h-ablation combines exhaustive + NSGA-II + random at rate=75 in one arm (rather than three separate arms), since the intermediate point's value is comparative, not standalone.

## Current Status

- **Validated:** rate=100 evaluations work correctly (76ms wall-clock, proper metrics output); reference point valid at both rate=75 and rate=100; progressive stress confirmed across rates; 1800-config space unchanged; same NSGA-II/random algorithm implementations applicable.
- **Uncertain:** Whether Pareto density at rate=100 is lower or higher than rate=50's 4.5%. The hypothesis says lower (more dominance), but it's possible that rate=100 creates new tradeoff dimensions (e.g., tp=1 configs that are terrible at rate=100 might still be Pareto-optimal on the cost dimension because nothing at gpu_count=1 works well, creating a degenerate "you can't buy good performance cheaply" frontier). This would INCREASE density.
- **Suggested next:** If rate portability is confirmed (gap >= 60 evals at rate=100): iter-7 should test model portability (different model with different KV thresholds, e.g., Llama-3.1-8B with lower memory footprint). If portability is refuted: investigate why the landscape at rate=100 differs (possibly the binary nature of good/bad at high load makes crossover less effective than at moderate load where gradients exist).

## Warnings & Constraints

1. **`--metrics-path` must use `$TMPDIR`**, not `/tmp/`. Sandbox prevents writing to `/tmp/` directly.
2. **`--total-kv-blocks` is PER INSTANCE.** tp=4,i=2 at blocks=3000 means each instance has 3000 blocks.
3. **Hardware name is exactly `H100`** — not "h100-sxm".
4. **The BLIS binary is at `./blis`** in the repo root.
5. **batch≥128 produces identical results under KV stress.** When KV is the binding constraint, increasing batch has zero effect.
6. **block_size_in_tokens is the 6th gene.** Gene encoding: `(tp_idx, inst_idx, sched_idx, batch_idx, kv_idx, block_size_idx)`. block_size_idx ∈ {0, 1} mapping to {16, 32}.
7. **Reference point (0, 50000, 9, 11000) must be consistent across ALL arms and ALL rates** for meaningful HV comparison.
8. **Convergence tracking:** Every 40 evals for all arms.
9. **h-ablation runs exhaustive first, then NSGA-II and random.** The exhaustive sweep at rate=75 provides the normalization baseline for h-ablation's convergence tracking. Output: `convergence_nsga2.json`, `convergence_random.json`, and `pareto_front.json` (exhaustive).
10. **The corrected space includes multi-instance configs but fixes routing to round-robin** (routing was shown non-differentiating in earlier iterations). No routing scorer flags needed.
11. **At rate=100, tp=1,inst=1 configs are throughput-limited (not KV-limited).** preemption_count=0 despite terrible TTFT. The bottleneck is pure compute throughput, not KV capacity. This is different from rate=50 where some tp=1 configs did experience KV pressure.
12. **BLIS binary path for executor:** Use the working copy `./blis` or build fresh with `go build -o blis main.go`. Do NOT hardcode absolute paths to `.nous-experiments/` binaries from previous iterations.
