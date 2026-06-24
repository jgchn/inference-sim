# Handoff — Campaign 3, Iteration 2

## Goal

Implement and compare three multi-objective configuration search algorithms (NSGA-II, random, TPE) in a 3-objective space (maximize rps, minimize ttft_p99, minimize gpu_count) over BLIS's discrete parameter space. Measure convergence rate: evaluations needed to reach 95% of final hypervolume. Rate=50 with 500 requests creates genuine differentiation across GPU tiers.

## Key Discoveries

1. **2-objective (rps, ttft_p99) Pareto frontier is always a single point.** At any fixed workload/rate, the highest-capacity config (max TP, max instances) dominates all others on both throughput AND latency. This is why iteration 1 failed — all methods found the same single optimal point immediately. More hardware always improves both metrics because it reduces queueing (lower TTFT) and processes faster (higher rps).

2. **3-objective formulation with gpu_count creates a genuine 5-point Pareto frontier.** Verified at rate=50, 500 requests: (10.56, 15221, 1), (15.52, 71.7, 2), (18.74, 4608, 2), (29.74, 35.3, 4), (37.02, 25.1, 8). Each point represents a different cost-performance tradeoff.

3. **Rate=50 with 500 requests creates partial saturation across ALL tiers.** At rate=50: TP=1/i=1 (1 GPU) saturates badly (rps=10.56 << rate). TP=4/i=1 (4 GPUs) partially saturates (rps=29.74 < rate). TP=8/i=1 (8 GPUs) still below rate (rps=37.02 < 50). This ensures every config is under meaningful load.

4. **Evaluation time is ~200ms per 500-request run.** 550 serial evaluations complete in ~110 seconds. No parallelism needed to fit within 3-minute wall time budget.

5. **Within a GPU tier, one TP/instance combo dominates.** At 4 GPUs: tp=4/i=1 dominates tp=2/i=2 and tp=1/i=4 on both rps and ttft (when batch is large enough). At 8 GPUs: tp=8/i=1 dominates tp=4/i=2 and tp=2/i=4. Higher TP with fewer instances is always better within a tier.

6. **Batch size is the key intra-tier differentiator.** At tp=2/i=2 rate=50: batch=32 gives (12.6, 23446), batch=128 gives (24.0, 1283), batch=256 gives (24.62, 42.8). Below a threshold (~256 for 4-GPU configs at rate=50), batch constrains throughput and causes massive queueing. Above threshold, no further improvement.

7. **Scheduler, max_scheduled_tokens, and routing policy have minimal effect** when batch is adequate. FCFS, SJF, and different routing policies produce identical metrics at rate=50 with sufficient batch. Preemption count is 0 even at gpu-mem-util=0.85.

## System Interface

- **Build:** `go build -o blis main.go` (binary already exists)
- **Run baseline:**
  ```bash
  ./blis run --model qwen/qwen3-14b --hardware H100 --latency-model trained-physics \
    --num-requests 500 --rate 50 --tp 2 --num-instances 2 --scheduler fcfs \
    --max-num-running-reqs 128 --max-num-scheduled-tokens 4096 \
    --long-prefill-token-threshold 0 --block-size-in-tokens 16 \
    --routing-policy least-loaded --admission-policy always-admit \
    --preemption-policy fcfs --gpu-memory-utilization 0.9 \
    --seed 42 --metrics-path $TMPDIR/baseline.json
  ```
- **Output format:** JSON at `--metrics-path`, key fields: `responses_per_sec` (float), `ttft_p99_ms` (float). Third objective `gpu_count` is derived from config: `tp * num_instances`.
- **Baseline result:** `responses_per_sec=23.9895`, `ttft_p99_ms=1283.23`, exit code 0, file size 159KB

## Code Map

| Location | What | When to look |
|----------|------|--------------|
| `sim/metrics_utils.go:57` | `MetricsOutput` struct — all JSON field names | If metrics JSON parsing fails |
| `sim/metrics.go:66` | `SaveResults()` — computes percentiles, writes JSON | If output values seem wrong |
| `cmd/root.go:184` | `metricsPath` variable | If `--metrics-path` flag issues |

## Code Targets

All three arms create new Python files in `inputs/` — no BLIS source modifications needed.

| Arm | File | Purpose |
|-----|------|---------|
| h-main (NSGA-II) | `.nous/campaign-3/runs/iter-2/inputs/search_nsga2_3obj.py` | NSGA-II with 3 objectives and convergence tracking |
| h-control-negative (random) | `.nous/campaign-3/runs/iter-2/inputs/search_random_3obj.py` | Random search with convergence tracking |
| h-robustness (TPE) | `.nous/campaign-3/runs/iter-2/inputs/search_tpe_3obj.py` | Model-based TPE with convergence tracking |

All three must:
- Implement constraint-aware config generation (TP*inst<=8, routing only for multi-instance, prefill < max_scheduled_tokens or 0)
- Evaluate via subprocess: `./blis run ... --metrics-path <tmpfile>`
- Parse JSON output for `responses_per_sec` and `ttft_p99_ms`; derive `gpu_count = tp * num_instances`
- Compute 3D hypervolume with reference point (0, 40000, 9)
- Track convergence: record cumulative HV every 50 evaluations
- Output: `pareto_front.json` and `convergence.json`

## What I Tried That Didn't Work

1. **Rate=80 and rate=150 with 2 objectives** — created differentiation between tiers but tp8-i1 still dominated everything in the (rps, ttft) plane. No amount of rate increase creates a 2-objective tradeoff because more hardware always helps both.

2. **KV-constrained scenario (gpu-mem-util=0.85) to induce preemptions** — zero preemptions even with batch=512 at rate=50. The default workload token lengths don't exceed KV capacity at any tested gpu-mem-util level.

3. **Batch-size as throughput-latency tradeoff** — batch=32 gives lower ITL (10ms) than batch=256 (15ms), suggesting a rps-vs-itl tradeoff. But when using ttft_p99 as the latency objective, larger batch is strictly better (lower queueing → lower TTFT). The ITL tradeoff is real but dominated by the queueing effect in TTFT.

4. **Scheduler/routing differentiation** — FCFS vs SJF, least-loaded vs round-robin produce identical or near-identical metrics at rate=50 with adequate batch. These parameters only matter when the system is in a narrow saturation regime with active queueing.

5. **`/tmp/` for metrics-path** — fails with "operation not permitted" in sandbox. Must use `$TMPDIR`.

## What I Excluded and Why

1. **Parallel evaluation workers** — at 200ms/eval, 550 evals take 110s serially. Parallelism adds complexity without necessity for the 3-minute constraint. Can be added in iteration 3 if budget increases.

2. **Multi-seed evaluation** — determinism (INV-6) means single-seed results are stable. Multi-seed testing is premature until the algorithm comparison question is resolved.

3. **Different models/hardware** — portability testing postponed until we have a working algorithm comparison. If NSGA-II shows advantage at rate=50 on qwen3-14b/H100, iteration 3 can test portability.

4. **Bayesian optimization with Gaussian processes** — GP surrogates work poorly with discrete/categorical parameters. TPE is the standard model-based alternative for discrete spaces.

5. **Preemption-inducing workloads** — would require custom workload specs with very long sequences. The default workload distribution (from seed 42) doesn't trigger preemptions at any tested configuration.

## Evolution of Thinking

Started by assuming higher arrival rates would create a non-trivial 2-objective Pareto frontier (per iter 1's suggestion of rate=80-150). Probed at rate=80, 150 — found the SAME structural problem: tp8-i1 dominates everything because more hardware always improves both rps and ttft simultaneously. This is a fundamental property of the system, not a rate-sensitivity issue.

Realized the missing dimension is COST. In real capacity planning, you can't just use 8 GPUs — there's a budget constraint. Adding gpu_count as a third objective captures this naturally and creates a genuine multi-dimensional frontier with distinct cost-performance tradeoffs at each tier.

Also discovered that within a GPU tier, the parameter space is surprisingly flat — many configs produce identical metrics once batch is above a threshold. The "hard" part of the search is finding the right TP/instance split and adequate batch, not optimizing scheduler/routing. This suggests TPE (which models parameter importance) may outperform NSGA-II (which treats all parameters equally through crossover).

## Current Status

- **Validated:** 3-objective frontier has 5 distinct Pareto points, evaluation timing works, baseline command confirmed
- **Uncertain:** Whether NSGA-II's convergence advantage over random will be measurable given that each Pareto tier is achievable by ~1-3% of configs (random may still find all tiers in <200 evals)
- **Suggested next:** If convergence rates are indistinguishable at 550 evals, iteration 3 should (a) reduce budget to 200 evals where differences are more likely visible, (b) test a workload with prefix sharing where routing policy actually matters, (c) try a weighted scalarization where the single-objective optimum is rarer

## Warnings & Constraints

1. **`--metrics-path` must use `$TMPDIR`**, not `/tmp/`. Sandbox prevents writing to `/tmp/` directly.
2. **Metrics file is ~160KB for 500 requests** (includes per-request array). Parse only top-level fields; do not iterate over the `requests` array.
3. **Exit code is 0 when metrics-path write succeeds.** Previously observed exit=1 was from failed `/tmp/` writes.
4. **Within a GPU tier, results cluster tightly.** Most parameter variations (scheduler, routing, block_size, gpu_mem_util) produce identical metrics once batch >= threshold. Expect many duplicate objective values in the random sample.
5. **Hardware name is exactly `H100`** — not "h100-sxm" or "H100-SXM".
6. **Reference point (0, 40000, 9) must be consistent** across all arms for fair HV comparison. 40000ms is safely above the worst observed ttft_p99 (30056ms at tp1/i1). 9 GPUs is above max (8).
