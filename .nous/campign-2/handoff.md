# Handoff — Campaign 2, Iteration 5

## Goal

Validate that simulation duration (num_requests) is the primary confound in Phase 1 tier ranking. Demonstrate that at nreq≥2000, lean Phase 1 (batch=128) correctly identifies TP=4/2 as universally optimal — eliminating the batch=512 prerequisite (RP-C2-5) and the apparent rate-dependent crossover. Characterize the minimum nreq needed for correct ranking.

## Key Discoveries

1. **TP=4/2 is universally optimal at nreq=2000** — Wins ALL 6 conditions (3 rates × 2 hardware) at batch=512, seed=42. Scores: H100 0.194-0.197, A100-SXM 0.153-0.155. No rate-dependent crossover exists in steady state.

2. **The "rate-dependent crossover" was a num_requests artifact** — At rate=500 with nreq=500 (iter-4 setting): TP=8/1 wins (0.162 vs 0.150). At nreq=1000: TP=4/2 wins (0.178 vs 0.170). The crossover point is between nreq=500 and nreq=750 at rate=500. The system was never reaching steady-state queuing at nreq=500.

3. **All tested rates are deeply oversaturated** — System capacity is ~73-75 rps regardless of hardware. All tested rates (500-5000) are 6.7-67× capacity. The system is always saturated; the only question is whether the simulation runs long enough to reveal it.

4. **Lean Phase 1 works at nreq=2000** — On A100-SXM rate=5000 (iter-4's lean failure case): at nreq=2000, lean Phase 1 picks TP=4/2 (0.115 vs 0.106), matching standard. The A100-SXM "tie zone" from iter-4 was also a nreq artifact.

5. **batch=512 was compensating for short simulations** — Standard Phase 1 (batch=512, nreq=500) happened to identify a "correct" tier only because higher batch capacity artificially inflated throughput during the transient. In steady state, both batch=128 and batch=512 produce the same tier ranking.

6. **TP=8/1's horrible p99 TTFT (~5500ms) vs TP=4/2's 45ms** — At nreq=1000 rate=500, TP=8/1 has scheduling_delay_p99=5502ms while TP=4/2 has 45ms. The single instance builds up a massive queue. This is invisible at nreq=500 because the queue hasn't fully formed yet.

7. **Wall time scales linearly with nreq** — nreq=500: ~165ms, nreq=1000: ~273ms, nreq=2000: ~510ms per eval. The experiment (~360 evals at mixed nreq) will take ~2-3 minutes total.

## System Interface

- **Build:** `go build -o blis main.go` (binary already exists at `./blis`)
- **Run baseline (steady-state optimal):**
  ```
  ./blis run --model qwen3-14b --hardware H100 --tp 4 --num-instances 2 --rate 2000 --num-requests 2000 --seed 42 --latency-model trained-physics --scheduler fcfs --max-num-running-reqs 512 --max-num-scheduled-tokens 8192 --block-size-in-tokens 16 --routing-policy least-loaded --admission-policy always-admit --preemption-policy fcfs --fitness-weights "throughput:0.4,p99_ttft:0.3,p99_e2e:0.3"
  ```
- **Output format:** Stdout emits `=== Simulation Metrics ===\n{json}` blocks + `=== Fitness Evaluation ===\nScore: <float>\n  <component>: <value>...`. Parse `Score:` line with `re.search(r"Score:\s+([\d.]+)", stdout)`.
- **Baseline result:** H100 TP=4/2 batch=512 rate=2000 nreq=2000: Score=0.196464 (steady-state optimum).

## Code Map

- `sim/cluster/metrics.go:150-152` — `RequestsPerSec = CompletedRequests / (SimEndedTime / 1e6)`. This is the throughput formula. SimEndedTime = time of last request completion. Check here if throughput values seem wrong.
- `sim/cluster/metrics.go:448-469` — `ComputeFitness()` function. Weighted sum of normalized metrics.
- `sim/cluster/metrics.go:475-479` — `extractMetric()` for throughput: `rps/(rps+referenceRPS)` where referenceRPS=100.
- `sim/cluster/metrics.go:483-484` — `extractMetric()` for latency: `1/(1+ticks/referenceTicks)` where referenceTicks=1000.
- `sim/cluster/metrics.go:431-434` — Reference constants: `referenceRPS=100`, `referenceTPS=10000`, `referenceTicks=1000`.
- `cmd/root.go` — All CLI flag registrations (num-requests, rate, tp, num-instances, max-num-running-reqs, etc.).

## Code Targets

No code changes in this iteration — all arms vary CLI parameters only (num_requests and max-num-running-reqs).

## What I Tried That Didn't Work

- **Using `-oP` (Perl regex) in grep on macOS** — macOS grep doesn't support `-P` flag. Use standard grep patterns instead.
- **Rate sweep 1800-1960 with nreq=750 for crossover** — Got non-monotone results because nreq=750 is in the transition zone. The crossover appeared at rate=1920-1950 but was actually an artifact of the borderline nreq.
- **The iter-4 "reference optima" at nreq=500** — All rate=500 references are now known to be transient-state (wrong). The correct reference at nreq=2000 is TP=4/2 for all conditions.

## What I Excluded and Why

- **Phase 2 (TPE/LHS) optimization** — Not relevant to this iteration's question. Iter-4 already proved TPE converges in 2-3 evals. This iteration focuses purely on whether Phase 1 needs batch=512.
- **Secondary knob sweeps** — Already shown to have near-zero effect in iters 2-4. Not retesting.
- **Multi-model/multi-hardware expansion** — Campaign 2 focuses on algorithmic questions with fixed model (qwen3-14b). Would add noise without answering the nreq question.
- **Rate=50 testing** — Even more oversaturated (50/75 ≈ 67% utilization actually UNDER capacity!). Would need separate analysis for unsaturated regime. Deferred to future iteration.
- **nreq>2000** — Probes show nreq=2000 already stabilizes. Diminishing returns beyond that.

## Evolution of Thinking

Started by trying to characterize the exact "crossover rate" on H100 where TP=8/1 yields to TP=4/2 (binary search between 500-2000). Found the crossover at rate=1920-1950 with nreq=750. But then noticed inconsistency: with nreq=1000 at rate=1500, TP=4/2 already won (contradicting earlier probes at nreq=750).

This led to testing num_requests as a variable rather than a fixed parameter. The result was dramatic: at rate=500, switching from nreq=500 to nreq=1000 flipped the winner from TP=8/1 to TP=4/2. The "crossover rate" wasn't a property of the system — it was a property of the measurement duration.

Further investigation showed ALL tested rates (500-5000) are deeply oversaturated relative to system capacity (~75 rps). The system ALWAYS eventually favors multi-instance configs; the only question was whether the simulation ran long enough to reveal steady-state behavior. The batch=512 "requirement" was an indirect fix: higher batch capacity during the transient phase compensated for the short simulation by allowing more concurrent processing during the brief arrival window.

The key realization: RP-C2-5 ("Phase 1 MUST use batch=512") was solving the wrong problem. The real problem was insufficient simulation duration. With nreq=2000, batch=128 works correctly because the system is fully saturated and the batch capacity ceiling is irrelevant (queue is always larger than batch size).

## Current Status

- **Validated:**
  - TP=4/2 is universally optimal at nreq=2000 (6/6 conditions, seed=42)
  - Lean Phase 1 (batch=128) at nreq=2000 agrees with standard on A100-SXM rate=5000 (iter-4 failure case)
  - Lean Phase 1 at nreq=1000 agrees with standard on H100 rate=500 (iter-4 failure case)
  - Wall time: ~510ms per eval at nreq=2000
  - System capacity is ~73-75 rps regardless of rate/hardware/batch (saturated regime)
  - nreq=500→750 is the H100 rate=500 transition zone (batch=512, seed=42)

- **Uncertain:**
  - Whether lean Phase 1 at nreq=2000 agrees with standard in ALL 18 condition×seed combinations (need seeds 123, 456)
  - Exact minimum nreq for each rate/hardware combo (probed seed=42 only)
  - Whether the A100-SXM "tie zone" (TP=4/2 vs TP=8/1 gap <1%) resolves cleanly at nreq=2000 across seeds
  - Whether rate=50 is actually unsaturated (50 req/s < 75 rps capacity) — would be a different regime entirely

- **Suggested next:**
  - After validating nreq=2000 correctness across seeds: simplify the canonical search algorithm to just {lean Phase 1, nreq=2000, 4 evals} with no Phase 2 needed
  - Investigate the unsaturated regime (rate < capacity ≈ 75 rps) — does a genuine Pareto tradeoff emerge there?
  - Test whether different workload distributions (non-Poisson arrivals, bimodal prompt lengths) create regimes where TP=4/2 is NOT universally optimal
  - Consider whether the fitness function (with referenceRPS=100) is appropriate — rps values of 73-75 give throughput component of 0.42-0.43, which is well below saturation. A lower referenceRPS would change the relative importance of throughput vs latency.

## Warnings & Constraints

1. **All iter-4 reference optima are invalid for nreq=2000** — The references in iter-4 handoff (H100_rate500: TP=8/1 score=0.162349, etc.) are measured at nreq=500-1000. At nreq=2000, TP=4/2 wins everything with different scores. Use the new references from this iteration.

2. **num_requests MUST match between conditions for fair comparison** — Never compare a score at nreq=500 with one at nreq=2000. The absolute scores change with nreq (more requests → higher throughput because steady-state rps > transient rps).

3. **Rate=500 is NOT "low load"** — All tested rates are deeply oversaturated (rate >> capacity of ~75 rps). Rate=500 means 6.7× overload. The system is always in queuing steady state after sufficient warmup.

4. **Model name format** — Use `qwen3-14b` on CLI (short form, verified working). Do NOT use `qwen/qwen3-14b`.

5. **Routing for multi-instance** — Use `routing=least-loaded` for TP=1/8, TP=2/4, TP=4/2 (multi-instance). Use `routing=round-robin` for TP=8/1 (single-instance only).

6. **Redirect stderr to DEVNULL** — Contains logrus diagnostics that interfere with stdout parsing.

7. **Score parsing** — Use `re.search(r"Score:\s+([\d.]+)", stdout)`. The Score line appears after the Fitness Evaluation header.

8. **Expected scores at nreq=2000 (batch=512, seed=42):**
   - H100: TP=4/2 scores 0.194-0.197 (rate-independent)
   - H100: TP=8/1 scores 0.182-0.183 (rate-independent, plateau)
   - A100-SXM: TP=4/2 scores 0.153-0.155
   - A100-SXM: TP=8/1 scores 0.147-0.147
