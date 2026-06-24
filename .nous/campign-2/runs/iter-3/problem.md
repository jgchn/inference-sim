# Problem Framing — Campaign 2, Iteration 3

## Research Question

Can an "extremes-first" Phase 1 strategy (4 evaluations: one max-instance config per TP level) identify the optimal parallelism tier as accurately as full sequential Phase 1 (15 evaluations), while being portable across hardware where the optimal tier differs?

Secondary: Does Bayesian Phase 2 (Optuna TPE sampler) converge to the within-tier optimum faster than Latin Hypercube Sampling when secondary knobs have meaningful variance (rate=5000)?

Grounded in:
- `search_blis_iter8.py:76-82` — PHASE1_TP_CONFIGS (sequential 4-tier ordering)
- `search_blis_iter8.py:84-88` — LEAN_PHASE1_TP_CONFIGS (lean bracket from campaign 1)
- `sim/config.go` — hardware definitions (H100, A100-SXM)

## System Interface

- **Build:** `go build -o blis main.go` (binary already exists at `./blis`)
- **CLI flags (relevant):**
  - `--hardware string` — GPU type ("H100", "A100-SXM")
  - `--tp int` — tensor parallelism degree
  - `--num-instances int` — number of deployment replicas
  - `--rate float` — request arrival rate (req/s)
  - `--num-requests int` — total requests to simulate
  - `--seed int` — RNG seed for determinism
  - `--latency-model string` — latency estimation backend ("trained-physics")
  - `--scheduler string` — scheduling policy
  - `--max-num-running-reqs int` — max batch size
  - `--max-num-scheduled-tokens int` — max batched tokens
  - `--long-prefill-token-threshold int` — chunked prefill threshold
  - `--block-size-in-tokens int` — KV cache block size
  - `--routing-policy string` — request routing policy
  - `--admission-policy string` — admission control policy
  - `--preemption-policy string` — preemption policy
  - `--fitness-weights string` — composite fitness function weights
- **Code evidence:**
  - `cmd/root.go:2038` — `--metrics-path` flag registration
  - `sim/metrics_utils.go:57-87` — `MetricsOutput` struct with JSON field names
- **Output format:** Stdout emits `=== Simulation Metrics ===\n{...json...}` blocks. With `--fitness-weights`, also emits `Score: <float>`. Parse cluster block for multi-instance.
- **Native metric collection:** `--fitness-weights "throughput:0.4,p99_ttft:0.3,p99_e2e:0.3"` produces a composite score in stdout.

## Baseline Command

```bash
./blis run --model qwen3-14b --hardware H100 --tp 4 --num-instances 2 \
  --rate 2000 --num-requests 1000 --seed 42 --latency-model trained-physics \
  --scheduler fcfs --max-num-running-reqs 512 --max-num-scheduled-tokens 8192 \
  --block-size-in-tokens 16 --routing-policy round-robin \
  --admission-policy always-admit --preemption-policy fcfs \
  --fitness-weights "throughput:0.4,p99_ttft:0.3,p99_e2e:0.3"
```

## Baseline Validation

- Exit code: 0
- Score output: `Score: 0.177562`
- Wall time: 223ms
- This is the optimal tier (TP=4/2) with max batch on H100 at rate=2000.

## Experimental Conditions

### Condition 1: Extremes-First Phase 1 + LHS Phase 2 (h-main)

**Phase 1** — Evaluate 4 configs (one per TP level, max-instance, batch=512):
- TP=1/inst=8, TP=2/inst=4, TP=4/inst=2, TP=8/inst=1

**Phase 2** — Allocate remaining budget (budget - 4 evals) to the top-1 tier's secondary knob space via Latin Hypercube Sampling.

Run on BOTH hardware (H100 and A100-SXM), seeds {42, 123, 456}, rate=2000, budget=50.

Compare convergence speed (evals to 95% of per-hardware optimum) against:
- Full-sequential Phase 1 (15 evals) + LHS Phase 2
- Random search (budget=50)

### Condition 2: Extremes-First + Bayesian Phase 2 (h-main rate=5000)

Same Phase 1 as above, but Phase 2 uses Optuna TPE sampler instead of LHS. Test at rate=5000 where secondary knobs have 26% variance (batch 128→512 gives 0.140→0.176).

Run on H100, seeds {42, 123, 456}, budget=50.

Compare Bayesian Phase 2 vs LHS Phase 2 convergence within top tier.

### Condition 3: Control — Rate=50 (h-control-negative)

At rate=50, all secondary knobs have zero effect (RP-C2-3). Extremes-first should identify TP=8/1 in 4 evals (position 4), while full-sequential takes 15. But Phase 2 adds zero value since all configs within the top tier score identically.

Run on H100, seeds {42, 123, 456}, budget=50.

### Condition 4: Hardware portability (h-robustness)

On A100-SXM, the optimal tier is TP=8/1 (score 0.1372) vs H100 where it's TP=4/2 (score 0.1776). The extremes-first strategy must correctly identify the optimal tier on BOTH hardware platforms.

Run extremes-first on A100-SXM, seeds {42, 123, 456}, rate=2000, budget=50.
Verify: top tier identified = TP=8/1 (not TP=4/2).

## Success Criteria

1. **Tier identification accuracy**: Extremes-first correctly identifies the optimal tier in 4 evals on both H100 (TP=4/2) and A100-SXM (TP=8/1) across all seeds.
2. **Convergence speed**: Extremes-first reaches 95% of optimal score consistently faster (fewer evals) than full-sequential Phase 1 on both hardware platforms.
3. **Bayesian Phase 2 advantage**: At rate=5000 where secondary knobs have 26% variance, TPE Phase 2 reaches 95% of within-tier optimum in fewer evals than LHS Phase 2.
4. **Hardware portability**: The same algorithm (no hardware-specific tuning) correctly adapts to different optimal tiers on different hardware.

## Constraints

- Budget: 50 evaluations per strategy per seed (total ~50 × 3 seeds × 4 strategies × 2 hardware = 1200 evals max ≈ 4.5 min wall time)
- Each eval: ~223ms (rate=2000, 1000 requests)
- Must not violate RP-C2-5 (use max batch=512 in Phase 1 for correct tier ranking)
- Must not violate RP-C2-6 (deterministic Phase 1 tier identification)

## Prior Knowledge

Active principles applied:
- **RP-C2-3**: Secondary knobs have zero effect at rate=50 → control-negative validates this
- **RP-C2-5**: Rate must be high enough for genuine tradeoffs → using rate=2000 and rate=5000
- **RP-C2-6**: Hierarchical Phase 1 with batch=512 identifies TP=4/2 in 14 evals (sequential ordering)
- **RP-C2-7**: Fixed ordering is structurally disadvantaged → extremes-first addresses this
- **RP-C2-8**: True qualifying fraction is 2.67% → random needs ~37 evals expected

New observations grounding this design:
- H100 rate=2000: TP=4/2 (0.1776) > TP=8/1 (0.1713), gap 3.6%
- A100-SXM rate=2000: TP=8/1 (0.1372) ≈ TP=4/2 (0.1368), gap 0.3% — optimal tier is hardware-dependent
- Within any TP level, max-instance is always optimal at rate=2000 (monotonically increasing with instances)
- rate=5000 H100: batch 128→512 produces 26% score range within TP=4/2 → Phase 2 has room to optimize
