# Problem Framing — Campaign 2, Iteration 4

## Research Question

**Does the Phase 1 batch-size profile critically determine tier-ranking correctness, and if so, what is the minimum safe batch for correct identification across hardware and rate regimes?**

The finding from iterations 1-3 was that extremes-first Phase 1 (4 evals with batch=512) solves the configuration search in all tested conditions. However, Phase 2 is structurally ineffective because Phase 1 already uses the optimal secondary config (batch=512). This iteration tests whether a "lean" Phase 1 (batch=128) — which would leave genuine Phase 2 headroom (25%+) — can safely identify the correct parallelism tier across all regimes.

**Source files implementing tier ranking mechanism:**
- `sim/cluster/metrics.go:448-469` — `ComputeFitness()`: composite score computation
- `sim/cluster/metrics.go:475-497` — `extractMetric()`: throughput normalization via `rps/(rps+100)`, latency via `1/(1+ticks/1000)`
- `sim/cluster/metrics.go:432-434` — reference constants: `referenceRPS=100`, `referenceTicks=1000`
- `cmd/root.go:984` — `--fitness-weights` flag registration

## System Interface

- **Build:** `go build -o blis main.go` (binary already exists)
- **CLI flags relevant to experiment:**
  - `--tp N` — tensor parallelism degree (`cmd/root.go`, flag registration)
  - `--num-instances N` — number of model instances (`cmd/root.go`)
  - `--max-num-running-reqs N` — maximum batch size (the key variable; controls how many requests can execute simultaneously)
  - `--rate N` — Poisson arrival rate in requests/second
  - `--num-requests N` — total requests to simulate
  - `--hardware {H100, A100-SXM}` — GPU hardware type
  - `--fitness-weights "k:v,..."` — composite scoring weights
  - `--latency-model trained-physics` — required for correct latency estimation
  - `--routing-policy {round-robin, least-loaded, weighted}` — request routing for multi-instance
  - `--scheduler {fcfs, priority-fcfs, sjf, reverse-priority}` — scheduling policy
  - `--seed N` — deterministic RNG seed (INV-6)
- **Output format:** Stdout emits `=== Simulation Metrics ===\n{...json...}` blocks (per-instance + cluster). With `--fitness-weights`, also emits `=== Fitness Evaluation ===\nScore: <float>\n  <component>: <value>\n...`
- **Code evidence:**
  - `sim/cluster/metrics.go:432` — `referenceRPS = 100.0`
  - `sim/cluster/metrics.go:434` — `referenceTicks = 1000.0` (1ms in microseconds)
  - `sim/cluster/metrics.go:479` — throughput normalization: `rps / (rps + 100)`
  - `sim/cluster/metrics.go:484` — latency normalization: `1 / (1 + p99_ticks / 1000)`

## Baseline Command

```bash
./blis run --model qwen3-14b --hardware H100 --tp 4 --num-instances 2 \
  --rate 5000 --num-requests 1000 --seed 42 --latency-model trained-physics \
  --scheduler fcfs --max-num-running-reqs 512 --max-num-scheduled-tokens 8192 \
  --block-size-in-tokens 16 --routing-policy round-robin \
  --admission-policy always-admit --preemption-policy fcfs \
  --fitness-weights "throughput:0.4,p99_ttft:0.3,p99_e2e:0.3"
```

## Baseline Validation

- Exit code: 0
- Output: Score: 0.176474
- Fitness components: throughput=0.439436, p99_ttft=0.002246, p99_e2e=0.000086
- Raw metrics: responses_per_sec=78.39, ttft_p99_ms=444.31, e2e_p99_ms=11677.15
- Execution time: ~220ms per evaluation

## Experimental Conditions

### Condition 1: Standard Phase 1 (batch=512) — Tier Ranking Baseline

Evaluate all 4 extremes-first tiers (TP=1/8, TP=2/4, TP=4/2, TP=8/1) with batch=512, across rate={500, 2000, 5000} × hardware={H100, A100-SXM} × seed={42, 123, 456}.

Commands vary by tier/rate/hardware. Example for TP=4/2 H100 rate=5000:
```bash
./blis run --model qwen3-14b --hardware H100 --tp 4 --num-instances 2 \
  --rate 5000 --num-requests 1000 --seed 42 --latency-model trained-physics \
  --scheduler fcfs --max-num-running-reqs 512 --max-num-scheduled-tokens 8192 \
  --block-size-in-tokens 16 --routing-policy least-loaded \
  --admission-policy always-admit --preemption-policy fcfs \
  --fitness-weights "throughput:0.4,p99_ttft:0.3,p99_e2e:0.3"
```

Total evals: 4 tiers × 3 rates × 2 hardware × 3 seeds = 72

### Condition 2: Lean Phase 1 (batch=128) — Tier Ranking Test

Same as Condition 1 but with `--max-num-running-reqs 128`.

Total evals: 72 (same matrix)

### Condition 3: Lean Phase 1 + TPE Phase 2 (H100 rate=5000 only)

After lean Phase 1 identifies top tier (TP=4/2 on H100), run TPE Phase 2 within that tier exploring all secondary knobs. Budget=46 Phase 2 evals (50 total - 4 Phase 1).

Phase 2 search space within fixed tier: scheduler (4) × batch (5) × routing (3) × max_tokens (3) × block_size (2) × admission (2) × preemption (2) × prefill (4) = 5,760 combinations.

Seeds: {42, 123, 456}. Total evals: 3 × (4 + 46) = 150

### Condition 4: Lean Phase 1 + LHS Phase 2 (H100 rate=5000 only)

Same as Condition 3 but using Latin Hypercube Sampling instead of TPE for Phase 2.

Total evals: 150

### Condition 5: Standard Phase 1 (no Phase 2) as Upper Bound

Extremes-first with batch=512 only. Already gives near-optimal (99.5%). This serves as the "ceiling" comparator.

Total evals per run: 4. Total across all conditions: 4 × 3 rates × 2 hardware × 3 seeds = 72

**Grand total BLIS invocations:** 72 + 72 + 150 + 150 + 72 = 516 (~2 minutes at 220ms/eval)

## Success Criteria

1. **Tier ranking accuracy**: Standard Phase 1 (batch=512) identifies the same top tier across all 3 seeds for each (rate, hardware) condition (100% agreement). Lean Phase 1 (batch=128) shows ≥1 disagreement with standard in at least 2 conditions.

2. **Deficit quantification**: When lean Phase 1 identifies the wrong tier, the maximum achievable score within that tier (Phase 2 ceiling) is measurably lower than the standard Phase 1 score (deficit > 1%).

3. **TPE vs LHS convergence**: When lean Phase 1 is correct (H100 rate=5000), TPE Phase 2 reaches 95% of within-tier optimal in fewer evaluations than LHS Phase 2 across at least 2/3 seeds.

4. **Phase 2 headroom**: Within the correctly-identified tier on H100 rate=5000, the batch=128 baseline → batch=512 optimal gap is ≥20% (verifying genuine Phase 2 value).

## Constraints

- Total BLIS invocations ≤ 600 (budget from campaign: keep under 2000 across ALL conditions)
- Each eval ~220ms → total runtime ~2.2 minutes
- Phase 1 MUST use batch=512 for correctness (RP-C2-5, strengthened by this iteration)
- Deterministic results via --seed (INV-6)
- TP × instances ≤ 8 (hardware constraint)
- Single-instance configs use routing=round-robin; multi-instance use routing=least-loaded

## Prior Knowledge

Active principles that apply:
- **RP-C2-5**: At rate≥2000, secondary knobs have zero throughput effect within a tier. This iteration extends: batch DOES affect composite fitness via latency components at all rates.
- **RP-C2-6**: At rate=2000 H100, Phase 1 reaches 95% in 14 evals (sequential). Extremes-first does it in 3-4.
- **RP-C2-10**: Extremes-first identifies optimal tier in 3-4 evals with batch=512.
- **RP-C2-11**: Hardware-adaptive — H100 picks TP=4/2, A100-SXM picks TP=8/1.

New findings from this iteration's probes (to be validated by experiment):
- Tier ranking is batch-sensitive: batch=128 gives wrong tier on H100 rate=500 and A100-SXM rate=5000
- Optimal tier is rate-dependent on H100: TP=8/1 at rate=500, TP=4/2 at rate≥2000
- Phase 2 headroom from batch=128→512 is 25-35% when system is saturated
