# Handoff — Campaign 2, Iteration 4

## Goal

Validate that Phase 1 batch-size profile (batch=512) is critically required for correct tier ranking across hardware and rate regimes. Demonstrate that "lean Phase 1" (batch=128) fails in specific conditions (H100 rate=500, A100-SXM rate=5000) while succeeding in others (H100 rate=5000). When lean Phase 1 is correct, measure whether TPE Phase 2 can exploit the 25%+ headroom (batch=128→512) faster than LHS.

## Key Discoveries

1. **Tier ranking is batch-sensitive and hardware-dependent** — At batch=128: H100 rate=500 picks wrong tier (TP=2/4 instead of TP=8/1, 24% deficit); A100-SXM rate=5000 picks wrong tier (TP=4/2 instead of TP=8/1, 0.35% deficit). Standard batch=512 is universally correct.

2. **Optimal tier is rate-dependent on H100** — rate=500: TP=8/1 wins (0.1623). rate=2000: TP=4/2 wins (0.1755). rate=5000: TP=4/2 wins (0.1765). The crossover is between rate=500 and rate=2000. A100-SXM always picks TP=8/1 across all tested rates.

3. **Batch=128→512 headroom is 25-35% on saturated tiers** — H100 TP=4/2 rate=5000: batch=128 gives 0.1405, batch=512 gives 0.1765 (25.6% improvement). A100-SXM TP=8/1 rate=5000: batch=128 gives 0.1021, batch=512 gives 0.1372 (34.4%).

4. **No genuine Pareto tradeoff exists at any tested rate** — batch=512 dominates ALL metrics (highest throughput AND lowest p99_ttft AND lowest p99_e2e) at rates 50-10000. The Pareto problem is degenerate across all regimes.

5. **Composite fitness is 99% throughput-determined** — At rate=5000 batch=512: throughput component=0.439, p99_ttft component=0.002, p99_e2e component=0.00009. Latency contributes negligibly because referenceRPS=100 saturates the throughput sigmoid, while referenceTicks=1000µs makes real latencies (444ms+ = 444000µs) contribute vanishingly.

6. **Unsaturated tiers show zero batch sensitivity** — At rate=500, TP=2/4 (4 instances at ~125 req/s each): batch=128 and batch=512 give IDENTICAL scores (0.1228). Only tiers with per-instance load exceeding batch capacity show batch sensitivity.

7. **Execution speed confirmed: ~220ms per eval** at rate=5000 with 1000 requests. Total experiment (~516 evals) ≈ 1.9 minutes.

## System Interface

- **Build:** `go build -o blis main.go` (binary already exists at `./blis`)
- **Run baseline (H100 rate=5000, optimal config):**
  ```
  ./blis run --model qwen3-14b --hardware H100 --tp 4 --num-instances 2 --rate 5000 --num-requests 1000 --seed 42 --latency-model trained-physics --scheduler fcfs --max-num-running-reqs 512 --max-num-scheduled-tokens 8192 --block-size-in-tokens 16 --routing-policy round-robin --admission-policy always-admit --preemption-policy fcfs --fitness-weights "throughput:0.4,p99_ttft:0.3,p99_e2e:0.3"
  ```
- **Output format:** Stdout emits `=== Simulation Metrics ===\n{json}` blocks + `=== Fitness Evaluation ===\nScore: <float>\n  <component>: <value>...`. Parse `Score:` line with `re.search(r"Score:\s+([\d.]+)", stdout)`.
- **Baseline result:** H100 TP=4/2 batch=512 rate=5000: Score=0.176474. H100 TP=8/1 batch=512 rate=500: Score=0.162349. A100-SXM TP=8/1 batch=512 rate=5000: Score=0.137203.

## Code Map

- `sim/cluster/metrics.go:448-469` — `ComputeFitness()` function. Check here for fitness calculation logic.
- `sim/cluster/metrics.go:475-497` — `extractMetric()`: normalization formulas. Throughput: `rps/(rps+100)`. Latency: `1/(1+ticks/1000)`.
- `sim/cluster/metrics.go:432-434` — Reference constants: `referenceRPS=100`, `referenceTPS=10000`, `referenceTicks=1000`.
- `cmd/root.go:984` — `--fitness-weights` flag registration.
- `search_blis_iter8.py:152-170` — `build_blis_cmd()` function. Proven flag mapping pattern.
- `search_blis_iter8.py:173-183` — `parse_cluster_metrics()` function. Proven stdout parsing.
- `search_blis_iter8.py:186-224` — `run_blis()` function. Shows Score line parsing.
- `search_blis_iter8.py:60-74` — Search space constants (all option arrays).

## Code Targets

### search_blis_iter4_c2.py (new file)

Create a self-contained Python script (~400 lines) implementing:

1. **Configuration space** — Same as iter-8: 4 extremes tiers × secondary knobs.

2. **`run_blis(config, rate, num_requests, hardware, seed)`** → subprocess call, parse `Score:` line. Return float or None.

3. **`standard_phase1(rate, hardware, seed)`** — Evaluate 4 extremes (TP=1/8, TP=2/4, TP=4/2, TP=8/1) with batch=512, tokens=8192, scheduler=fcfs, routing=least-loaded (or round-robin for inst=1). Return list of (tier, score) in eval order. Record convergence curve (best_so_far after each eval).

4. **`lean_phase1(rate, hardware, seed)`** — Same as standard but with batch=128. Return list of (tier, score) in eval order.

5. **`tpe_phase2(tier, rate, hardware, budget, seed)`** — Optuna TPE within the tier's secondary knob space. Fixed TP and num_instances from tier. Explore: scheduler(4), max_running(5), max_tokens(3), prefill(4), block_size(2), routing(3), admission(2), preemption(2). Use TPESampler(seed=seed, n_startup_trials=5). Return convergence curve.

6. **`lhs_phase2(tier, rate, hardware, budget, seed)`** — Latin Hypercube Sample, same space. Return convergence curve.

7. **`evaluate_condition(strategy, rate, hardware, budget, seeds)`** — Run strategy across seeds, collect results.

8. **Main experiment matrix:**
   - h-main: lean Phase 1 + TPE Phase 2, H100 rate=5000, budget=50, seeds=42,123,456
   - h-control-negative: lean Phase 1 + TPE Phase 2, H100 rate=500, budget=50, seeds=42,123,456
   - h-ablation: lean Phase 1 + LHS Phase 2, H100 rate=5000, budget=50, seeds=42,123,456
   - h-robustness: standard Phase 1 + lean Phase 1 (no Phase 2), rate={500,2000,5000} × hardware={H100,A100-SXM}, seeds=42,123,456

9. **Output JSON structure:**
```json
{
  "experiment": "phase1-batch-sensitivity",
  "arms": {
    "h-main": {
      "condition": "lean-tpe H100 rate=5000",
      "seeds": {
        "42": {
          "phase1_scores": {"TP=1/8": 0.119, "TP=2/4": 0.139, "TP=4/2": 0.141, "TP=8/1": 0.129},
          "phase1_winner": [4, 2],
          "phase1_correct": true,
          "phase2_convergence": [...],
          "best_score": 0.176,
          "evals_to_95pct_within_tier": 12,
          "total_evals": 50
        }
      }
    },
    "h-control-negative": { ... },
    "h-ablation": { ... },
    "h-robustness": { ... }
  },
  "reference_optima": {
    "H100_rate500": {"tier": [8, 1], "score": 0.162349},
    "H100_rate2000": {"tier": [4, 2], "score": 0.177728},
    "H100_rate5000": {"tier": [4, 2], "score": 0.176474},
    "A100-SXM_rate500": {"tier": [8, 1], "score": 0.129478},
    "A100-SXM_rate2000": {"tier": [8, 1], "score": 0.137186},
    "A100-SXM_rate5000": {"tier": [8, 1], "score": 0.137203}
  }
}
```

**Key implementation notes:**
- BLIS binary at `./blis` (already built)
- Use `--model qwen3-14b` (short form, verified working)
- `--latency-model trained-physics` required on all invocations
- For multi-instance (inst>1): routing=least-loaded in Phase 1. For single-instance: routing=round-robin.
- Phase 2 explores ALL secondary knobs including max_running (batch). The starting point is lean Phase 1's batch=128, but Phase 2 can discover batch=512.
- Parse Score with: `re.search(r"Score:\s+([\d.]+)", stdout)`
- Redirect stderr to DEVNULL (contains logrus diagnostics)
- `num_requests=1000` for rate>=2000, `num_requests=500` for rate=500

## What I Tried That Didn't Work

- **Lean Phase 1 (batch=128) as universal strategy**: Fails on H100 rate=500 (picks TP=2/4 vs correct TP=8/1, 24% deficit) and A100-SXM rate=5000 (picks TP=4/2 vs correct TP=8/1, 0.35% deficit). NOT safe across all conditions.
- **batch=64 for Phase 1**: Even worse — gives wrong tier on H100 rate=5000 too (TP=2/4 beats TP=4/2 at batch=64).
- **batch=32 for Phase 1**: Catastrophic — completely wrong ranking on ALL conditions.
- **batch=256 for Phase 1**: Still wrong on A100-SXM rate=5000 (TP=4/2 > TP=8/1 by 0.1246 vs 0.1230).
- **tier-shed admission for Pareto tradeoff**: No shedding occurs at any tested rate with batch=512 (system handles all requests).
- **priority preemption for tradeoff**: Zero preemptions — KV cache never fills at tested rates.
- **Different fitness weight emphasis**: Irrelevant — batch=512 dominates ALL metrics simultaneously (highest throughput AND lowest latency). Even latency-only weighting would select batch=512.
- **Rate=10000 for new regime**: Gives Score=0.176382 — saturated same as rate=5000, no new behavior.
- **Long prompts (4096 tokens) for KV pressure**: Still no preemption/shedding occurs.

## What I Excluded and Why

- **Multi-model serving**: Campaign goal is single-model optimization. Multi-model adds dimensions not in the sweep matrix.
- **Cost-aware optimization (GPU-hours/fitness)**: Would require changing the fitness function, beyond this iteration's scope.
- **NSGA-II strategy**: Iter-2 showed it provides no reliable advantage (RP-C2-9). TPE for Phase 2 only.
- **qwen3-32b, llama-3.1-8b, L40S**: Campaign 1 (iters 5-10) tested model/hardware portability extensively. Campaign 2 focuses on algorithmic questions with fixed model.
- **Budget scaling (100-500 evals)**: The fundamental question (is lean Phase 1 safe?) doesn't need large budgets. 50 evals per run is sufficient to demonstrate TPE convergence.
- **Chunked prefill, max_tokens, block_size sweeps**: Confirmed zero effect in iter-2 across all rates. Not worth re-testing.

## Evolution of Thinking

Started by investigating whether lean Phase 1 (batch=128) could create Phase 2 headroom — the key iter-3 failure was Phase 2 having nothing to optimize because Phase 1 already used batch=512. Discovered that lean Phase 1 correctly identifies tiers on H100 rate=5000 (0.8% margin) but fails catastrophically on H100 rate=500 (picks wrong tier with 24% unrecoverable deficit).

The mechanism became clear: at rate=500, multi-instance configs (TP=2/4) aren't saturated, so batch doesn't matter for them. But single-instance configs (TP=8/1) ARE saturated at batch=128, making them appear weaker than they actually are. Standard Phase 1 (batch=512) gives TP=8/1 enough capacity to show its true strength.

Also discovered the optimal tier is rate-dependent on H100 (TP=8/1 at rate=500, TP=4/2 at rate≥2000), which was not established in prior iterations. This rate-dependence is hardware-specific — A100-SXM always picks TP=8/1.

Tried to find a genuine Pareto tradeoff (throughput vs latency) by testing different rates, batch sizes, admission/preemption policies, and prompt lengths. Found NONE — batch=512 dominates all metrics simultaneously at every tested rate. The Pareto problem is structurally degenerate for this model/hardware/workload space.

## Current Status

- **Validated:**
  - Lean Phase 1 (batch=128) correctly identifies TP=4/2 on H100 rate=5000 (margin: 0.8%)
  - Lean Phase 1 fails on H100 rate=500 (picks TP=2/4, deficit=24%)
  - Lean Phase 1 fails on A100-SXM rate=5000 (picks TP=4/2, deficit=0.35%)
  - Standard Phase 1 (batch=512) universally correct across all probed conditions
  - Phase 2 headroom from batch=128→512: H100=25.6%, A100-SXM=34.4%
  - Unsaturated tiers show zero batch sensitivity (batch=128=batch=512)
  - Execution time: ~220ms/eval, total experiment ~2 minutes
  - No Pareto tradeoff exists at any rate tested

- **Uncertain:**
  - Whether TPE can reliably discover batch=512 in 15-20 evals (depends on startup trials)
  - Whether lean Phase 1's 0.8% margin on H100 rate=5000 is seed-robust (probed at seed=42 only, but BLIS is deterministic so tiers won't change — only workload distribution might shift scores slightly)
  - The exact rate crossover point on H100 where optimal tier switches from TP=8/1 to TP=4/2 (somewhere between 500 and 2000)

- **Suggested next:** After this iteration validates batch-sensitivity:
  - Iteration 5 could characterize the exact crossover rate on H100 (binary search between 500-2000)
  - Test whether an "adaptive batch" Phase 1 (start at batch=128, detect saturation via score gradient, escalate to batch=512) can be both correct AND leave Phase 2 headroom
  - Explore whether different workload distributions (bimodal prompt lengths, bursty arrivals) create genuine Pareto tradeoffs

## Warnings & Constraints

1. **Phase 1 MUST use batch=512** — This is now triple-validated (RP-C2-5 + iter-4 probes). Lower batch gives wrong tier in at least 2/5 tested conditions. This constraint is FUNDAMENTAL, not conservative.

2. **Lean Phase 1 margin is thin on H100 rate=5000** — TP=4/2 beats TP=2/4 by only 0.8% at batch=128 (0.1405 vs 0.1394). While deterministic across seeds (same workload generator), any small system change could flip this.

3. **A100-SXM deficit is tiny (0.35%)** — lean Phase 1 picks TP=4/2 (max=0.136719) instead of TP=8/1 (optimal=0.137203). This is within noise for practical purposes but technically a failure.

4. **Phase 2 with batch exploration is NOT a real-world use case** — If an algorithm already evaluates batch=512 in Phase 1 (required for correct tier ranking), it already knows batch=512 is the best secondary config. The lean+Phase2 experiment is a controlled test of TPE's learning ability, not a practical algorithm.

5. **num_requests=500 for rate=500** — Using 1000 requests at rate=500 takes 2s of simulated time (fine). Using 500 requests is faster and sufficient for steady-state.

6. **Model name format** — Use `qwen3-14b` on CLI (short form, verified working). Do NOT use `qwen/qwen3-14b`.

7. **Routing for Phase 1** — Multi-instance configs (TP=1/8, TP=2/4, TP=4/2) use `routing=least-loaded`. Single-instance (TP=8/1) uses `routing=round-robin`. This matches iter-3.
