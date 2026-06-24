# Problem Framing — Iteration 5: Simulation Duration Confound

## Research Question

Is `num_requests` (simulation duration) the primary confound in Phase 1 tier ranking, and does using a sufficient duration (≥2000 requests) eliminate the need for batch=512 as a Phase 1 prerequisite?

Prior iterations established that lean Phase 1 (batch=128) fails at certain rates/hardware while standard Phase 1 (batch=512) is "universally correct." However, probing reveals that at nreq=2000 (longer simulation), TP=4/2 wins **every condition** regardless of batch size, rate, or hardware — and the prior "rate-dependent crossover" (TP=8/1 at low rates, TP=4/2 at high rates) was entirely an artifact of short simulations (nreq=500) not reaching steady-state queuing behavior.

The mechanism: when arrival rate >> system capacity (rate=500 vs capacity≈75 rps), short simulations don't build up enough queue for latency metrics to reflect steady state. Single-instance configs (TP=8/1) appear artificially strong during transients because their faster per-request speed matters when queues are short. Once queues reach equilibrium, multi-instance configs win on both aggregate throughput AND latency distribution.

Key source files:
- `sim/cluster/metrics.go:150-152` — throughput = completed_requests / (SimEndedTime / 1e6)
- `sim/cluster/metrics.go:475-479` — fitness normalization: rps / (rps + 100)
- `sim/cluster/metrics.go:483-484` — latency normalization: 1 / (1 + ticks/1000)

## System Interface

- **Build:** `go build -o blis main.go` (binary already exists at `./blis`)
- **CLI flags relevant to experiment:**
  - `--num-requests`: Number of requests to generate (controls simulation duration = num_requests / rate)
  - `--rate`: Arrival rate in requests/second
  - `--tp`: Tensor parallelism degree
  - `--num-instances`: Number of serving instances
  - `--max-num-running-reqs`: Maximum batch size (the "batch" knob: 128=lean, 512=standard)
  - `--hardware`: GPU hardware type (H100, A100-SXM)
  - `--fitness-weights`: Composite fitness metric weights
  - `--seed`: Workload generation seed
- **Code evidence:**
  - `--num-requests`: `cmd/root.go` (cobra flag registration)
  - `--max-num-running-reqs`: `cmd/root.go` (batch size knob)
  - Throughput computation: `sim/cluster/metrics.go:150-152`
  - Fitness computation: `sim/cluster/metrics.go:448-469`
- **Output format:** Stdout emits `=== Fitness Evaluation ===\nScore: <float>`. Parse with `re.search(r"Score:\s+([\d.]+)", stdout)`.

## Baseline Command

```bash
./blis run --model qwen3-14b --hardware H100 --tp 4 --num-instances 2 \
  --rate 2000 --num-requests 2000 --seed 42 \
  --latency-model trained-physics --scheduler fcfs \
  --max-num-running-reqs 512 --max-num-scheduled-tokens 8192 \
  --block-size-in-tokens 16 --routing-policy least-loaded \
  --admission-policy always-admit --preemption-policy fcfs \
  --fitness-weights "throughput:0.4,p99_ttft:0.3,p99_e2e:0.3"
```

## Baseline Validation

Exit code: 0. Score: 0.196464. Wall time: ~510ms.
This is TP=4/2 on H100 at rate=2000 with nreq=2000 — the steady-state optimal configuration.

Reference optima at nreq=2000 (batch=512, seed=42):
- H100 rate=500: TP=4/2, score=0.193654
- H100 rate=2000: TP=4/2, score=0.196464
- H100 rate=5000: TP=4/2, score=0.196875
- A100-SXM rate=500: TP=4/2, score=0.153247
- A100-SXM rate=2000: TP=4/2, score=0.154864
- A100-SXM rate=5000: TP=4/2, score=0.153813

**Key observation:** TP=4/2 wins ALL 6 conditions at nreq=2000. No rate-dependent crossover exists in steady state.

## Experimental Conditions

### h-main: Lean Phase 1 at nreq=2000

For each condition (rate × hardware × seed):
- Run all 4 extremes tiers (TP=1/8, TP=2/4, TP=4/2, TP=8/1) with batch=128, nreq=2000
- Record which tier wins
- Compare to standard Phase 1 (batch=512, nreq=2000) reference

Matrix: rates={500, 2000, 5000} × hardware={H100, A100-SXM} × seeds={42, 123, 456}
Total evals: 4 tiers × 6 conditions × 3 seeds = 72 evals (lean) + 72 evals (standard reference) = 144 evals

### h-control-negative: Transient regime at nreq=250

Same matrix but with nreq=250 (deeply transient, <4× system capacity).
Prediction: tier rankings will be inconsistent and wrong.
Total evals: 4 tiers × 6 conditions × 3 seeds × 2 batch sizes = 144 evals

### h-ablation: nreq vs batch orthogonal test

Compare four Phase 1 configurations at fixed condition (H100 rate=500, seeds=42,123,456):
- {batch=128, nreq=500} — "lean + short" (iter-4's failure mode)
- {batch=512, nreq=500} — "standard + short" (iter-4's "correct" approach)
- {batch=128, nreq=2000} — "lean + long" (this iteration's proposed approach)
- {batch=512, nreq=2000} — "standard + long" (full reference)

Test whether nreq or batch is the dominant factor for correct ranking.
Total evals: 4 tiers × 4 configs × 3 seeds = 48 evals

### h-robustness: Convergence threshold

Sweep nreq={250, 500, 750, 1000, 1500, 2000} at fixed condition (H100 rate=500, batch=512, seed=42) to characterize exactly when the ranking flips from TP=8/1-dominant to TP=4/2-dominant.
Total evals: 4 tiers × 6 nreq values = 24 evals

**Total experiment budget:** ~360 evaluations × ~350ms average = ~2.1 minutes

## Success Criteria

1. At nreq=2000, lean Phase 1 (batch=128) selects the SAME tier as standard Phase 1 (batch=512) in ≥16/18 condition×seed combinations (≥89% agreement)
2. At nreq=250, tier rankings disagree with nreq=2000 reference in ≥50% of conditions
3. In the orthogonal test: {batch=128, nreq=2000} matches {batch=512, nreq=2000} but {batch=512, nreq=500} does NOT — proving nreq is the dominant factor
4. The tier ranking at nreq=2000 is deterministic across seeds (same winner for ≥2/3 seeds in each condition)

## Constraints

- Each BLIS eval at nreq=2000 costs ~500ms wall time. Total budget: ~360 evals in ~3 minutes.
- Must use `--latency-model trained-physics` on all invocations.
- Model: `qwen3-14b` (short form, verified).
- Routing: `least-loaded` for multi-instance, `round-robin` for single-instance.
- Seeds: 42, 123, 456 (matching iter-4 for comparability).

## Prior Knowledge

Active principles that this iteration challenges or refines:
- **RP-C2-5** (Phase 1 MUST use batch=512): This iteration hypothesizes that batch=512 was compensating for insufficient num_requests. At nreq=2000, batch=128 may be sufficient.
- **RP-C2-11** (Standard Phase 1 correct on H100 across all rates): The "correct" reference depends on nreq. At nreq=2000, the reference changes — TP=4/2 wins everywhere including rate=500.
- **RP-C2-7** (TP=8/1 optimal at rate=50): Likely an artifact of short simulation at low rates.

Principles that remain valid:
- **RP-C2-2** (Phase 1 tier ranking is seed-stable): Expected to hold at nreq=2000.
- **RP-C2-6** (14 evals for sequential Phase 1): Still valid as an eval count, though nreq now affects the per-eval cost.
- **RP-C2-14** (TPE converges in 2-3 evals): Still valid for within-tier optimization.
