# Handoff — Campaign 2, Iteration 3

## Goal

Implement and compare four configuration search strategies for finding the optimal BLIS configuration using a composite fitness score. Test whether "extremes-first" Phase 1 (4 evals at max-instance per TP level) converges faster than full-sequential Phase 1 (15 evals), and whether TPE-based Phase 2 beats LHS Phase 2 when secondary knobs have meaningful variance. Prove hardware portability by running on both H100 (where TP=4/2 wins) and A100-SXM (where TP=8/1 wins).

## Key Discoveries

1. **Optimal tier is hardware-dependent** — H100 rate=2000: TP=4/2 (score 0.1776) > TP=8/1 (0.1713), gap 3.6%. A100-SXM rate=2000: TP=8/1 (0.1372) ≈ TP=4/2 (0.1368), gap only 0.3%. The crossover never happens on A100-SXM even at rate=8000 — TP=8/1 always dominates.

2. **Max-instance is always optimal within each TP level at rate≥2000** — Verified: TP=2 inst=1→4 gives scores 0.084→0.157 (monotonically increasing). TP=1 inst=1→8 gives 0.045→0.122. This means evaluating only max-instance configs (4 evals) correctly ranks TP levels.

3. **Rate=5000 gives meaningful Phase 2 variance** — On H100 TP=4/2: batch 128→512 gives 0.140→0.176 (26% range). Scheduler adds 2-4% (fcfs=0.176, sjf=0.173). At rate=2000, Phase 2 headroom is only 0.09% — too small for meaningful Phase 2 comparison.

4. **Routing policy doesn't change tier ranking** — On H100 rate=2000: TP=4/2 scores 0.1774-0.1777 across all routing policies. TP=8/1 always 0.1713. Routing is irrelevant for single-instance configs.

5. **A100-SXM saturates at ~23,521 tok/s (TP=8/1) and ~19,725 (TP=4/2)** — These are capacity ceilings reached by rate=3000. The TP=8/1 advantage on A100-SXM is stable across all rates.

6. **Execution speed confirmed: 223ms per eval** at rate=2000 with 1000 requests. Budget of 50 evals = ~11s wall time per run. Total experiment (~1200 evals) ≈ 4.5 minutes.

7. **Fitness-weights composite score** — `--fitness-weights "throughput:0.4,p99_ttft:0.3,p99_e2e:0.3"` produces a single `Score: <float>` line in stdout. Higher is better. This replaces the degenerate multi-objective framing from iter-1/2.

## System Interface

- **Build:** `go build -o blis main.go` (binary already exists at `./blis`)
- **Run baseline (H100):** `./blis run --model qwen3-14b --hardware H100 --tp 4 --num-instances 2 --rate 2000 --num-requests 1000 --seed 42 --latency-model trained-physics --scheduler fcfs --max-num-running-reqs 512 --max-num-scheduled-tokens 8192 --block-size-in-tokens 16 --routing-policy round-robin --admission-policy always-admit --preemption-policy fcfs --fitness-weights "throughput:0.4,p99_ttft:0.3,p99_e2e:0.3"`
- **Run baseline (A100-SXM):** Same command with `--hardware A100-SXM --tp 8 --num-instances 1`
- **Output format:** Stdout emits `=== Simulation Metrics ===\n{...json...}` blocks (one per instance + one for "cluster"). Also emits `Score: <float>` when `--fitness-weights` is specified. Parse Score line with `re.search(r"Score:\s+([\d.]+)", stdout)`.
- **Baseline result:** H100 TP=4/2: Score=0.177562. A100-SXM TP=8/1: Score=0.137186.

## Code Map

- `sim/metrics_utils.go:57-87` — `MetricsOutput` struct. Check here for exact JSON field names.
- `cmd/root.go:2038` — `--metrics-path` flag registration. Only on `blis run`.
- `search_blis_iter8.py:152-170` — `build_blis_cmd()` function. Proven flag mapping pattern including `--fitness-weights`.
- `search_blis_iter8.py:173-183` — `parse_cluster_metrics()` function. Proven stdout parsing pattern.
- `search_blis_iter8.py:186-224` — `run_blis()` function. Shows how to parse Score line and metrics.
- `search_blis_iter8.py:60-73` — Search space constants (all option arrays).
- `search_blis_iter8.py:93-97` — PHASE1_CONFIG_PROFILES (strategic profiles for adaptive Phase 1).

## Code Targets

### search_blis_iter3_c2.py (new file)

Create a self-contained Python script (~350 lines) implementing:

1. **Configuration space** — Same 15 tiers × secondary knobs as iter-8. Use `--fitness-weights "throughput:0.4,p99_ttft:0.3,p99_e2e:0.3"` for composite scoring.

2. **`run_blis(config, ...)`** → subprocess call, parse `Score:` line from stdout. Return score float or None on failure.

3. **`extremes_first_phase1()`** — Evaluate 4 configs: (TP=1/inst=8), (TP=2/inst=4), (TP=4/inst=2), (TP=8/inst=1), all with batch=512, tokens=8192, scheduler=fcfs, block=16, routing=round-robin (or least-loaded for multi-inst), admission=always-admit, preemption=fcfs. Return sorted by score descending.

4. **`sequential_phase1()`** — Evaluate all 15 tiers in VALID_TP_INSTANCES order with same batch=512 profile. Return sorted by score.

5. **`lhs_phase2(top_tier, budget, seed)`** — Latin Hypercube Sample within top tier's secondary knob space (scheduler×max_running×max_tokens×prefill_threshold×block_size×routing×admission×preemption). Fixed TP and num_instances from top tier.

6. **`tpe_phase2(top_tier, budget, seed)`** — Optuna TPE sampler exploring same secondary knob space. Each trial evaluates one config. Use `optuna.samplers.TPESampler(seed=seed, n_startup_trials=5)`.

7. **`random_search(budget, seed)`** — Uniform random sampling from full space (all 15 tiers × secondary knobs).

8. **Convergence tracking** — After each eval (across Phase 1 + Phase 2), record `best_score_so_far`. Compute `evals_to_95pct(curve, optimal_score)`.

9. **Main CLI:**
   - `--strategy {extremes-tpe, extremes-lhs, sequential-lhs, random, all}`
   - `--hardware {H100, A100-SXM}`
   - `--rate N` (default 2000)
   - `--num-requests N` (default 1000)
   - `--budget N` (default 50)
   - `--seeds comma-separated` (default "42,123,456")
   - `--output path.json`

10. **Output JSON structure:**
```json
{
  "hardware": "H100",
  "rate": 2000,
  "budget": 50,
  "results": {
    "extremes-tpe": {
      "seeds": {
        "42": {
          "convergence_curve": [...],
          "best_score": 0.177,
          "best_config": {...},
          "evals_to_95pct": 8,
          "phase1_top_tier": [4, 2],
          "phase1_evals": 4
        }
      }
    }
  },
  "optimal_scores": {"H100": 0.177728, "A100-SXM": 0.137186}
}
```

**Key implementation notes:**
- BLIS_BASE must include `--latency-model trained-physics`
- Phase 1 MUST use batch=512 and tokens=8192 (RP-C2-5 / Warning 5 from iter-2)
- For multi-instance configs, use `routing=least-loaded`; for single-instance use `routing=round-robin`
- `--model qwen3-14b` (note: not `qwen/qwen3-14b` — the iter-8 script uses short name via model_shortname but CLI needs `qwen3-14b`)

**Wait — verify model flag format:**
The iter-2 handoff says `--model qwen/qwen3-14b` but iter-8 uses `MODEL = "qwen/qwen3-14b"`. Both forms work — use `qwen3-14b` (short form, verified working in probes above).

## What I Tried That Didn't Work

- **Hypervolume as metric (iter-1)**: Degenerate Pareto front means HV is identical for all strategies. Use composite fitness score instead.
- **Rate=50, 200, 300, 500 for Pareto tradeoffs (iter-1/2)**: All produce degenerate single-objective. Composite score solves this.
- **Rate=2000 for Phase 2 comparison**: Only 0.09% headroom in secondary knobs — insufficient to distinguish LHS vs TPE. Must use rate=5000 (26% headroom).
- **max-num-scheduled-tokens variation (iter-2)**: No effect at any rate (2048, 4096, 8192 identical).
- **Chunked prefill threshold variation (iter-2)**: No effect at any rate.
- **Block size variation (iter-2)**: No effect (16 and 32 identical).
- **A100-SXM TP crossover search**: Tried rates 2000, 3000, 5000, 8000 — TP=8/1 always dominates on A100-SXM. No crossover exists for this hardware/model.

## What I Excluded and Why

- **NSGA-II strategy**: Iter-2 showed it doesn't provide reliable advantages over hierarchical (RP-C2-9). Seed-dependent luck in warmup explains apparent advantages. Replaced with TPE for Phase 2 only (where it has a clear role as within-tier optimizer).
- **qwen3-32b and L40S hardware**: Campaign 1 (iter-8) already tested portability to these. Focusing campaign 2 on the H100 vs A100-SXM comparison which has the interesting property of different optimal tiers.
- **Budget scaling (1000, 2000 evals)**: The core question (4 vs 15 Phase 1 evals) doesn't need large budgets. 50 evals is sufficient — the advantage is in Phase 1 speed, and Phase 2 needs at most 30 evals to test TPE vs LHS.
- **llama-3.1-8b model**: Single model sufficient to test the algorithm. Model portability was tested in campaign 1.

## Evolution of Thinking

Started by reading iter-2 findings: hierarchical beats random 4.6x but has the structural weakness of fixed ordering (RP-C2-7). The "suggested next" said to test hardware portability and adaptive Phase 2.

First explored A100-SXM and found the optimal tier is different (TP=8/1 vs TP=4/2 on H100). This immediately motivated the "extremes-first" strategy — evaluate max-instance per TP level (4 evals, not 15) and pick the winner. This addresses RP-C2-7 by never wasting evals on sub-max-instance configs.

Then verified that max-instance IS always optimal within each TP level at rate≥2000 (monotonically increasing with instances). This grounds the extremes-first strategy in empirical evidence.

For Phase 2, found that rate=2000 gives only 0.09% headroom — too small to distinguish LHS vs TPE. Escalated to rate=5000 where batch-size effect is 26%. This creates a meaningful test for Bayesian optimization.

Switched from pure throughput metric to composite fitness score (`--fitness-weights`) which was already proven in campaign 1 (iter-8). This gives a single scalar objective that combines throughput, TTFT, and e2e latency.

## Current Status

- **Validated:**
  - Extremes-first Phase 1 correctly identifies optimal tier on both H100 (TP=4/2, score 0.1776) and A100-SXM (TP=8/1, score 0.1372) in 4 evals
  - Full-sequential Phase 1 takes 15 evals on both hardware
  - Rate=5000 gives 26% secondary knob variance for Phase 2 testing
  - Optuna 4.8.0 TPESampler is available
  - Execution time: 223ms/eval at rate=2000
  - Composite fitness score works correctly via `--fitness-weights`

- **Uncertain:**
  - Whether A100-SXM's tiny TP=8/1 vs TP=4/2 gap (0.3%) holds across seeds (could flip)
  - Whether TPE needs more than 5 startup trials to outperform LHS on a 5-option categorical (batch size)
  - Whether rate=5000 changes the TP tier ranking from rate=2000 (verified: it doesn't — still TP=4/2 > TP=8/1 on H100)

- **Suggested next:** If extremes-first + TPE confirms advantage:
  - Iteration 4 could test rate-adaptive strategy selection (low rate → skip Phase 2, high rate → run TPE)
  - Test whether the approach generalizes to multi-model serving (different models on different instances)
  - Explore cost-aware optimization (minimize GPU-hours per unit of fitness) as a third dimension

## Warnings & Constraints

1. **A100-SXM gap is only 0.3%** — TP=8/1 (0.1372) vs TP=4/2 (0.1368) at rate=2000. Some seeds might flip. For h-robustness, accept either TP=8/1 or TP=4/2 as "correct" — the test is about portability, not a specific tier winning.

2. **Model name format** — Use `qwen3-14b` on CLI (short form). The `qwen/qwen3-14b` form also works but probes above used short form.

3. **Phase 1 MUST use batch=512** — Lower batch incorrectly ranks tiers (RP-C2-5). This is critical and validated in iter-2.

4. **Routing for single-instance** — For TP=8/inst=1, routing policy has no effect. For multi-instance, use `least-loaded` as default in Phase 1.

5. **stdout includes all instances** — When `--num-instances > 1`, stdout emits one JSON block per instance plus one for "cluster". Parse the `Score:` line (which is always the cluster-level composite) instead of individual blocks.

6. **rate=5000 needs num_requests≥1000** — same as rate=2000, need enough requests to build steady-state queue pressure.

7. **95% threshold** — Compute as 0.95 × per-hardware optimal. H100 optimal: 0.177728 (weighted routing, TP=4/2, batch=512). A100-SXM optimal: 0.137186 (TP=8/1, batch=512). For rate=5000 H100: optimal is 0.176438 (TP=4/2, batch=512, fcfs).
