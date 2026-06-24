# Handoff — Iteration 1: Configuration Search Algorithm

## Goal

Implement and evaluate three configuration search strategies (random, TPE/Bayesian, Latin Hypercube) for the BLIS simulator's 86,400-configuration search space. The Python search script should find near-optimal configurations within minutes by calling `./blis run` as a subprocess and parsing fitness scores from stdout.

## Key Discoveries

1. **BLIS is extremely fast**: 200-request runs complete in ~77ms, 500 requests in ~163ms. This means 1000 evaluations take ~2 minutes wall-clock. Exhaustive search of all 86,400 configs would take ~2.4 hours.

2. **TP is the dominant performance knob**: At fixed total GPU count (8), increasing TP consistently improves fitness. Quick sweep: TP=1/8inst → Score=0.028, TP=2/4inst → 0.043, TP=4/2inst → 0.059, TP=8/1inst → 0.072. The ~2.5× score difference from TP alone is the largest single-parameter effect.

3. **Routing and admission are no-ops for single-instance configs**: Verified that changing `--routing-policy` from `round-robin` to `least-loaded` produces identical scores when `--num-instances 1`. This means 3 parameters (routing, admission, preemption) are wasted evaluations for any single-instance config.

4. **Fitness scoring normalizes metrics to [0,1]**: Throughput uses `value/(value+100)` (referenceRPS=100), latency uses `1/(1+value/1000)` (referenceTicks=1000μs). Defined in `sim/cluster/metrics.go:428-498`.

5. **Fitness score is printed to stdout but NOT in the JSON metrics file**: The `--metrics-path` JSON contains raw metrics (responses_per_sec, e2e_p99_ms, etc.) but not the computed fitness. The `=== Fitness Evaluation ===` section with Score appears only on stdout. The search script should parse stdout, not the JSON file.

6. **GPU name is "H100" not "h100_sxm"**: Available GPUs are `A100-80, A100-SXM, H100, L40S` (error message from invalid GPU name).

7. **No preemptions observed at tested loads**: Even at rate=200 with 1 instance and max_batch=64, preemption_count=0. The preemption-policy parameter may not matter at these load levels with default workload distributions.

## System Interface

- **Build:** `go build -o blis main.go` (binary already exists)
- **Run baseline:**
  ```bash
  ./blis run --model qwen/qwen3-14b --hardware H100 --latency-model trained-physics \
    --num-requests 200 --rate 50 --seed 42 --tp 1 --num-instances 1 \
    --scheduler fcfs --max-num-running-reqs 256 --max-num-scheduled-tokens 2048 \
    --long-prefill-token-threshold 0 --block-size-in-tokens 16 \
    --routing-policy round-robin --admission-policy always-admit \
    --preemption-policy fcfs \
    --fitness-weights "throughput:0.4,p99_ttft:0.3,p99_e2e:0.3"
  ```
- **Output format:** JSON metrics on stdout between `=== Simulation Metrics ===` and `=== Fitness Evaluation ===`. Fitness score follows as `Score: <float>`.
- **Baseline result:** Score=0.020, responses_per_sec=5.24, e2e_p99_ms=35062

## Code Map

- `cmd/root.go:947-984` — All swept parameter flag definitions. Check here if a flag name or default is wrong.
- `cmd/root.go:1761-1765` — Fitness weight parsing and ComputeFitness call. Check here if fitness isn't being computed.
- `sim/cluster/metrics.go:418-498` — `validFitnessKeysList()`, `ComputeFitness()`, `extractMetric()`. Check here if fitness scores look wrong or a metric key is rejected.
- `sim/cluster/metrics.go:428-435` — Reference scale constants. Check here if normalization seems off.
- `cmd/root.go:1727-1728` — `SaveResults()` call for metrics JSON output. Check here if `--metrics-path` isn't working.

## Code Targets

The experiment requires creating a new Python script (`search_blis.py`), not modifying BLIS source code. The code_changes in the bundle describe the script to create:

- **`search_blis.py`** (new file): Python search script using subprocess to call `./blis run`. Must handle: (1) the conditional TP × instances constraint, (2) stdout parsing of fitness score, (3) three search strategies (random, TPE, LHS), (4) JSON output of results.

## What I Tried That Didn't Work

- **`--hardware h100_sxm`**: Fatal error. Correct name is `H100` (case-sensitive, no suffix).
- **Parsing fitness from `--metrics-path` JSON**: The JSON file does not contain the fitness score. Must parse stdout or recompute fitness from raw metrics in Python (recomputing is more robust).
- **Expecting preemptions at rate=200**: Even with `--max-num-running-reqs 64` and 1 instance at rate=200, preemption_count=0. Preemption may require specific scheduling pressure that the default workload distribution doesn't trigger at these parameters.

## What I Excluded and Why

- **Genetic algorithms / evolutionary search**: Too complex for iteration 1. Start simple with random + TPE + LHS, then consider evolutionary methods in later iterations if TPE's advantage is confirmed.
- **Multi-objective Pareto search (NSGA-II)**: The built-in fitness function already scalarizes the multi-objective problem. True Pareto search is a natural follow-up for iteration 2.
- **Rate as a swept parameter**: Fixed at 50 req/s for the main experiment and 200 req/s for the robustness arm. Sweeping rate adds a dimension that confounds configuration comparison.
- **PD disaggregation knobs** (prefill-instances, decode-instances): Out of scope for the defined search space. Would be interesting for iteration 2.
- **Workload variation**: Fixed at distribution mode with default token counts. Different workload profiles could change the optimal configuration but are out of scope for iteration 1.

## Evolution of Thinking

Started assuming the search space would be too large for any meaningful search. Discovered that BLIS is so fast (~100ms/eval) that even 1000 evaluations take only ~2 minutes. This shifted the question from "can we search at all?" to "can intelligent search find optima faster than random sampling, and by how much?"

The TP dominance finding was surprising — expected more interaction between parameters, but TP alone accounts for most of the fitness variation. This suggests the effective dimensionality is much lower than 10, which should make TPE very effective.

The routing/admission no-op discovery for single-instance configs means the effective search space has structure: 8 of 15 TP/instance combos are single-instance, where 3 parameters don't matter. A smart search algorithm should discover this automatically through its probabilistic model.

## Current Status

- **Validated:** BLIS runs correctly, baseline command works, fitness weights produce parseable output, wall-clock timing is ~100ms per evaluation
- **Uncertain:** Whether preemption_policy actually affects anything at the tested loads (may need higher contention); whether the fitness landscape has enough structure for TPE to exploit beyond the TP dominance effect
- **Suggested next:** (1) After running the experiment, analyze whether TPE's advantage comes mainly from learning the TP dominance or from finer-grained parameter interactions. (2) Try multi-objective Pareto search (NSGA-II via Optuna) instead of scalarized fitness. (3) Explore rate as an additional dimension to see if optimal configs change under different load levels. (4) Consider warm-starting from iteration 1 results.

## Warnings & Constraints

- **Stderr contains warnings**: BLIS prints timing and config warnings to stderr. Always use `2>/dev/null` or `2>stderr.log` when parsing stdout for metrics.
- **Seed affects workload, not config evaluation**: `--seed 42` controls random request generation. For the search algorithm's own randomness, use a separate seed.
- **TP=0 is invalid**: The `--tp` flag defaults to 0 but auto-detects from model config. When scripting, always set `--tp` explicitly to avoid the auto-detection warnings.
- **Fitness score format on stdout**: `Score: 0.019249` — note the exact format with 6 decimal places. Parse with regex `r'Score:\s+([\d.]+)'`.
- **vllm-version auto-detection**: Warnings about "Finding default values of vLLM version" appear on stderr. Harmless but noisy.
