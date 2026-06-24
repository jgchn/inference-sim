# Problem Framing — Iteration 7: Lean Bracket (50x Budget Reduction)

## Research Question

Can we reduce the bracket K=1 hierarchical search from 100 evaluations (4 Phase 1 + 96 Phase 2) to 2 evaluations (Phase 1 over TP={4,8} only, no Phase 2) while maintaining 1% accuracy across all tested regimes?

Across 6 iterations and 27 active principles, Phase 1 max-profile achieves within 1% of the global best on every tested regime. TP=1 and TP=2 never win — TP=2's max-profile score is 11-19% below the winner across all rates (100-5000) and hardware (H100, A100-SXM, L40S). Phase 2 TPE contributes at most 0.7% improvement (routing discovery in multi-instance regimes), always within the 1% tolerance.

The "lean bracket" prunes TP=1 and TP=2 from Phase 1 and eliminates Phase 2 entirely, reducing total evaluations from 100 to 2 — a 50x budget reduction with proportional wall-time savings (~200ms total search time vs ~10s).

**Key source files:**
- `search_blis_iter6.py` (from iter-6 h-main.patch): Search algorithm implementation. `PHASE1_TP_CONFIGS` at line ~120 defines the TP evaluation order. `run_adaptive_hierarchical_search()` at line ~611 implements the two-phase search.
- `sim/cluster/metrics.go:418-498`: Fitness computation (`ComputeFitness()`).
- `cmd/root.go:947-984`: All swept parameter flag definitions.

## System Interface

- **Build:** `go build -o blis main.go` (binary already exists)
- **CLI flags relevant to experiment:**
  - `--tp` (int): tensor parallelism degree. `cmd/root.go:947`.
  - `--num-instances` (int): deployment replicas. `cmd/root.go:948`.
  - `--hardware` (string): GPU type (H100, A100-SXM, L40S). `cmd/root.go:956`.
  - `--rate` (float64): arrival rate in req/s. `cmd/root.go:940`.
  - `--seed` (int64): workload RNG seed. `cmd/root.go:938`.
  - `--fitness-weights` (string): scalarization weights. `cmd/root.go:1761-1765`.
  - `--max-num-running-reqs` (int): max concurrent requests per instance. `cmd/root.go:965`.
  - `--max-num-scheduled-tokens` (int): max batch tokens. `cmd/root.go:967`.
  - `--long-prefill-token-threshold` (int): chunked prefill threshold. `cmd/root.go:969`.
- **Output format:** JSON metrics between `=== Simulation Metrics ===` markers on stdout. `Score: <float>` in `=== Fitness Evaluation ===` section when `--fitness-weights` is provided.
- **Code evidence for fitness keys:** `sim/cluster/metrics.go:420-426` (8 valid keys), `sim/cluster/metrics.go:428-435` (reference scale constants).

## Baseline Command

```bash
./blis run --model qwen/qwen3-14b --hardware H100 --latency-model trained-physics \
  --num-requests 1000 --rate 500 --seed 42 --tp 4 --num-instances 2 \
  --scheduler fcfs --max-num-running-reqs 512 --max-num-scheduled-tokens 8192 \
  --long-prefill-token-threshold 4096 --block-size-in-tokens 16 \
  --routing-policy least-loaded --admission-policy always-admit \
  --preemption-policy fcfs \
  --fitness-weights "throughput:0.4,p99_ttft:0.3,p99_e2e:0.3"
```

## Baseline Validation

Ran the baseline command. Exit code 0. Score: 0.178063 (H100 qwen rate=500, TP=4/2inst, max-profile). This is the Phase 1 max-profile score for the TP=4 winner, within 0.68% of the global best (0.179282).

Additional validation probes:
- H100 qwen rate=500 TP=8/1inst max-profile: 0.169720 (4.7% below TP=4)
- H100 qwen rate=1000 TP=2/4inst: 0.158936 (11.5% below winner TP=4=0.179572)
- H100 qwen rate=2000 TP=2/4inst: 0.158202 (10.9% below winner TP=4=0.177470)
- H100 qwen rate=5000 TP=2/4inst: 0.155617 (11.3% below winner TP=4=0.175535)
- H100 qwen rate=500 MIN-profile TP=2/4inst: 0.082056 (WINS, 6.6% above TP=4=0.076957)

## Experimental Conditions

### Condition 1: h-main — Lean Bracket Accuracy (4 regimes, 5 seeds each)

Modify the search script to support lean bracket mode:
1. Add `--tp-candidates` flag: "all" (default, TP=1,2,4,8) or "lean" (TP=4,8 only)
2. Add `--skip-phase2` flag (boolean): when set, return Phase 1 result without running Phase 2

Run lean bracket (`--tp-candidates lean --skip-phase2 --phase1-profiles bracket --phase1-k 1`) on:
- H100 qwen3-14b rate=500 (TP=4 winner, evals_to_best=1)
- H100 qwen3-14b rate=100 (TP=8 winner, evals_to_best=2)
- A100-SXM qwen3-14b rate=500 (TP=4 winner, evals_to_best=1)
- A100-SXM qwen3-14b rate=100 (TP=8 winner, evals_to_best=2)

5 search seeds each. Results should be identical across seeds (deterministic, no RNG in lean bracket).

### Condition 2: h-control-negative — Min-Profile Lean Bracket (wrong winner)

Run lean bracket with min-profile instead of max-profile (`--tp-candidates lean --skip-phase2 --phase1-profiles bracket-min-only`) on H100 qwen rate=500.

This should select TP=2/4inst (wrong winner) because min-profile (mr=32) creates a capacity-limited regime where instance count dominates over bandwidth. Since TP=2 is excluded from lean bracket's TP candidates, the algorithm will erroneously select whichever of TP=4/TP=8 scores highest with min-profile — but neither will match the actual best config (TP=4/2inst with max-profile).

### Condition 3: h-ablation — Lean Bracket + Mini Phase 2 (routing recovery)

Run lean bracket with mini Phase 2 (`--tp-candidates lean --phase1-profiles bracket --phase1-k 1 --budget 12`) on A100-SXM qwen rate=200 (the crossover regime where routing policy matters).

Phase 1 max-profile gives 0.134488 (least-loaded). Phase 2 with 10 trials may discover round-robin/weighted routing (0.135440). From prior data, Phase 2 discovers routing improvement at trial 1-6 for seeds 44-46, but at trial 19-37 for seeds 42-43.

### Condition 4: h-robustness — Lean Bracket Across Workload Seeds

Run lean bracket on H100 qwen rate=500 with 5 blis_seeds (42-46), 5 search seeds each (25 combinations).

## Success Criteria

1. **h-main**: Lean bracket achieves within 1% of global best on all 4 regimes (all seeds). evals_to_best=1 for TP=4-winner regimes, evals_to_best=2 for TP=8-winner regimes. All 5 search seeds produce identical results per regime (determinism).
2. **h-control-negative**: Min-profile lean bracket selects a suboptimal configuration. Best score is >5% below the max-profile lean bracket best score, confirming the max-profile is essential.
3. **h-ablation**: With budget=12 (2 Phase 1 + 10 Phase 2), at least 3/5 search seeds find a score > Phase 1 max-profile (0.134488) via routing discovery. Scores remain within 1% of global best (0.135440) regardless.
4. **h-robustness**: 25/25 (blis_seed × search_seed) combinations achieve within 1% of the per-seed global best on H100 qwen rate=500.

## Constraints

- Each BLIS evaluation takes ~70-100ms on H100 (RP-5). Lean bracket (2 evals) completes in ~200ms total search time.
- Active principles RP-1 through RP-27 apply. No violations expected; lean bracket is consistent with all established principles.
- Determinism (INV-6): Same BLIS seed produces identical results regardless of search seed for lean bracket (no Phase 2 RNG).
- The 1% tolerance threshold (WITHIN_BEST_THRESHOLD = 0.99) is the campaign standard for evals_to_best.

## Prior Knowledge

- **RP-1**: TP is the dominant knob. Lean bracket focuses Phase 1 on the two competitive TP levels (4, 8).
- **RP-15/16**: Bracket K=1 with max-profile achieves 100% Phase 1 correctness across all tested regimes. Lean bracket inherits this since it still evaluates TP=4 and TP=8 with max-profile.
- **RP-17**: Min-profile selects wrong TP winner. Used in h-control-negative.
- **RP-24**: TP winner is rate-dependent (TP=8 at low rates, TP=4 at high rates). Lean bracket covers both candidates.
- **RP-3**: Large batch/token settings (max-profile) are secondary but consistent contributors. Lean bracket uses max-profile, capturing these.
- **Campaign data (new probes)**: TP=2 gap is 11-19% below winner across rates 100-5000 on all hardware. TP=1 gap is 32-38%. Neither ever approaches the winner.
