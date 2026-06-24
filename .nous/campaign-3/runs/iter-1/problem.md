# Problem Framing: Multi-Objective Configuration Search for BLIS

## Research Question

Can a structured multi-objective search algorithm (e.g., NSGA-II or model-based) discover Pareto-optimal configurations of BLIS with hypervolume ratio >= 95% of exhaustive search, using <= 2000 evaluations versus the full ~294K configuration space?

The search operates over BLIS's CLI flags (TP, replicas, scheduler, batch size, token budget, chunked prefill, block size, routing policy, scorer weights, admission policy, preemption policy, GPU memory utilization) and optimizes two objectives: maximize throughput (`responses_per_sec`) and minimize tail latency (`ttft_p99_ms`).

Key source files:
- `sim/metrics_utils.go:57` — `MetricsOutput` struct defining all observable metrics
- `sim/metrics.go:66` — `SaveResults()` computes and writes metrics JSON
- `cmd/root.go:184` — `metricsPath` flag definition
- `cmd/root.go:2038` — `--metrics-path` flag registration

## System Interface

- **Build command:** `go build -o blis main.go` (binary already built at `./blis`)
- **Run command:** `./blis run` with CLI flags
- **Output:** `--metrics-path <file>` writes cluster-aggregate `MetricsOutput` JSON
- **Determinism:** Fixed `--seed` produces byte-identical output (INV-6)
- **Evaluation time:** ~52ms per 100-request simulation (measured)

### CLI Flags (Search Space)

| Flag | Type | Values | Source |
|------|------|--------|--------|
| `--tp` | int | {1, 2, 4, 8} | cmd/root.go |
| `--num-instances` | int | 1..floor(8/TP) | cmd/root.go |
| `--scheduler` | string | fcfs, priority-fcfs, sjf, reverse-priority | cmd/root.go |
| `--max-num-running-reqs` | int | {32, 64, 128, 256, 512} | cmd/root.go |
| `--max-num-scheduled-tokens` | int | {2048, 4096, 8192} | cmd/root.go |
| `--long-prefill-token-threshold` | int | {0, 1024, 2048, 4096} (must be < max-num-scheduled-tokens or 0) | cmd/root.go |
| `--block-size-in-tokens` | int | {16, 32} | cmd/root.go |
| `--routing-policy` | string | round-robin, least-loaded, weighted | cmd/root.go |
| `--routing-scorers` | string | 4 curated profiles (only when routing=weighted) | cmd/root.go |
| `--admission-policy` | string | always-admit, tier-shed | cmd/root.go |
| `--preemption-policy` | string | fcfs, priority | cmd/root.go |
| `--gpu-memory-utilization` | float | {0.85, 0.9, 0.95} | cmd/root.go |

### Fixed Flags

| Flag | Value | Reason |
|------|-------|--------|
| `--model` | qwen/qwen3-14b | Fixed for iteration 1 |
| `--hardware` | H100 | Fixed for iteration 1 |
| `--latency-model` | trained-physics | Recommended default |
| `--num-requests` | 100 | Sufficient for stable metrics |
| `--rate` | 20 | Creates meaningful saturation differential |
| `--seed` | 42 | Deterministic baseline |

### Constraints (Pruning Rules)

1. `TP * num-instances <= 8` (single-node H100 SXM)
2. If `num-instances == 1`: routing-policy, routing-scorers, and admission-policy are irrelevant
3. `routing-scorers` only applies when `routing-policy == weighted`
4. `long-prefill-token-threshold` must be `< max-num-scheduled-tokens` (or 0 = disabled)

### Observable Metrics (Objectives)

From `MetricsOutput` (sim/metrics_utils.go:57):
- `responses_per_sec` — throughput (maximize)
- `ttft_p99_ms` — P99 time to first token (minimize)

Additional metrics available for secondary analysis:
- `tokens_per_sec`, `e2e_p99_ms`, `itl_p99_ms`, `scheduling_delay_p99_ms`, `preemption_count`

## Baseline Command

```bash
./blis run \
  --model qwen/qwen3-14b \
  --hardware H100 \
  --latency-model trained-physics \
  --num-requests 100 \
  --rate 20 \
  --tp 2 \
  --num-instances 2 \
  --scheduler fcfs \
  --max-num-running-reqs 128 \
  --max-num-scheduled-tokens 4096 \
  --long-prefill-token-threshold 0 \
  --block-size-in-tokens 16 \
  --routing-policy least-loaded \
  --admission-policy always-admit \
  --preemption-policy fcfs \
  --gpu-memory-utilization 0.9 \
  --seed 42 \
  --metrics-path /tmp/baseline_metrics.json
```

## Baseline Validation

- **Exit code:** 0
- **Output file:** cluster-aggregate JSON written to `--metrics-path`
- **Key metric values:** `responses_per_sec = 7.41`, `ttft_p99_ms = 33.2`
- **Execution time:** ~52ms

## Experimental Conditions

### Condition A: Random Search (Baseline)

Uniformly sample 500 valid configurations from the search space (respecting pruning rules). Evaluate each with `./blis run`. Compute Pareto front and hypervolume.

### Condition B: NSGA-II Evolutionary Search

Run NSGA-II with population_size=50, generations=10 (= 500 evaluations matching budget of Condition A). Use the discrete parameter space as the genome. Measure hypervolume convergence over generations.

### Condition C: NSGA-II with 2x Budget (1000 evaluations)

Same as Condition B but with 20 generations (1000 evaluations). Tests whether additional budget significantly improves hypervolume.

### Condition D: Latin Hypercube Sampling (Stratified Random)

500 evaluations using Latin Hypercube sampling across the parameter space. Tests whether stratification alone (without evolutionary selection pressure) improves over pure random.

## Success Criteria

1. **Hypervolume ratio:** At least one structured approach (B, C, or D) achieves hypervolume ratio >= 95% of a large random sample (2000 evals) within its evaluation budget.
2. **Efficiency:** NSGA-II (Condition B) achieves higher hypervolume than random (Condition A) at the same budget (500 evals), demonstrating value of selection pressure.
3. **Budget sensitivity:** Condition C (1000 evals) achieves higher hypervolume than Condition B (500 evals), confirming diminishing returns are not immediate.
4. **Wall time:** All conditions complete within 3 minutes wall time (at rate of ~52ms/eval, 2000 evals serial = ~104s, well within budget).

## Constraints

- Total BLIS invocations across all conditions: <= 2000
- Single model (qwen/qwen3-14b) and hardware (H100)
- Single arrival rate (20 req/s) and request count (100)
- Fixed seed per configuration for determinism
- Python implementation required

## Prior Knowledge

This is the first iteration. No prior findings or active principles to reference. The search space has been characterized:
- Full exhaustive space: ~294K configurations
- Evaluation time: ~52ms per config
- Objective range observed: throughput 2.92–10.27 rps, TTFT P99 25.3–16,266 ms
- System is deterministic with fixed seed
