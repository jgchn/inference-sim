# Problem Framing: Convergence Rate of Multi-Objective Search in 3D Objective Space

## Research Question

Does NSGA-II converge to the 3-objective Pareto frontier faster than random sampling when GPU cost is included as a third objective?

Iteration 1 established that with 2 objectives (rps, ttft_p99) on H100 hardware, the Pareto frontier collapses to a single point regardless of arrival rate — the highest-capacity config (max TP, max instances) dominates all others on both throughput and latency simultaneously. All three search algorithms (random, NSGA-II, LHS) achieved identical hypervolume immediately.

Iteration 2 introduces a 3rd objective — minimize GPU count (TP * num_instances) — creating a genuine multi-dimensional Pareto frontier where cost-performance tradeoffs exist across hardware tiers. The hypothesis: in this richer objective space, NSGA-II's selection pressure provides measurable convergence advantage over random sampling.

Key source files:
- `sim/metrics_utils.go:57` — `MetricsOutput` struct defining all observable JSON metrics
- `sim/metrics.go:66` — `SaveResults()` computes percentiles and writes metrics JSON
- `cmd/root.go:184` — `metricsPath` flag variable definition

## System Interface

- **Build command:** `go build -o blis main.go` (binary already exists at `./blis`)
- **Run command:** `./blis run` with CLI flags
- **Output:** `--metrics-path <file>` writes cluster-aggregate `MetricsOutput` JSON
- **Determinism:** Fixed `--seed` produces byte-identical output (INV-6)
- **Evaluation time:** ~200ms per 500-request simulation (measured)

### CLI Flags (Search Space)

| Flag | Type | Values | Constraint |
|------|------|--------|-----------|
| `--tp` | int | {1, 2, 4, 8} | TP * instances <= 8 |
| `--num-instances` | int | 1..floor(8/TP) | TP * instances <= 8 |
| `--scheduler` | string | fcfs, priority-fcfs, sjf, reverse-priority | — |
| `--max-num-running-reqs` | int | {32, 64, 128, 256, 512} | — |
| `--max-num-scheduled-tokens` | int | {2048, 4096, 8192} | — |
| `--long-prefill-token-threshold` | int | {0, 1024, 2048, 4096} | Must be < max-num-scheduled-tokens OR 0 |
| `--block-size-in-tokens` | int | {16, 32} | — |
| `--routing-policy` | string | round-robin, least-loaded, weighted | Only when instances > 1 |
| `--routing-scorers` | string | 4 profiles | Only when routing=weighted |
| `--admission-policy` | string | always-admit, tier-shed | Only when instances > 1 |
| `--preemption-policy` | string | fcfs, priority | — |
| `--gpu-memory-utilization` | float | {0.85, 0.9, 0.95} | — |

### Fixed Flags

| Flag | Value | Reason |
|------|-------|--------|
| `--model` | qwen/qwen3-14b | Fixed for iteration 2 |
| `--hardware` | H100 | Fixed for iteration 2 |
| `--latency-model` | trained-physics | Recommended default |
| `--num-requests` | 500 | Sustained load for steady-state metrics |
| `--rate` | 50 | Creates partial saturation across all GPU tiers |
| `--seed` | 42 | Deterministic (no multi-seed needed) |

### Objectives (3-objective formulation)

| Objective | Source | Direction |
|-----------|--------|-----------|
| `responses_per_sec` | BLIS JSON output | Maximize |
| `ttft_p99_ms` | BLIS JSON output | Minimize |
| `gpu_count` | Derived: TP * num_instances | Minimize |

## Baseline Command

```bash
./blis run --model qwen/qwen3-14b --hardware H100 --latency-model trained-physics \
  --num-requests 500 --rate 50 --tp 2 --num-instances 2 --scheduler fcfs \
  --max-num-running-reqs 128 --max-num-scheduled-tokens 4096 \
  --long-prefill-token-threshold 0 --block-size-in-tokens 16 \
  --routing-policy least-loaded --admission-policy always-admit \
  --preemption-policy fcfs --gpu-memory-utilization 0.9 \
  --seed 42 --metrics-path $TMPDIR/baseline.json
```

## Baseline Validation

- **Exit code:** 0
- **Eval time:** 198ms
- **Metrics file:** Written successfully (159,734 bytes)
- **Key metrics:** `responses_per_sec=23.9895`, `ttft_p99_ms=1283.2274`
- **GPU count:** 4 (TP=2 * instances=2)

## Experimental Conditions

All arms share:
- Search space: ~294K valid configurations (full combinatorial with constraints)
- Evaluation budget: 550 evaluations per arm
- 3 objectives: maximize rps, minimize ttft_p99, minimize gpu_count
- Reference point for hypervolume: (rps=0, ttft_p99=40000, gpu_count=9)
- Convergence metric: evaluations needed to reach 95% of final hypervolume

### Arm 1: NSGA-II (h-main)
- Population size: 50, generations: 11 (= 550 evals total)
- Discrete crossover (uniform) and mutation (random single-parameter swap)
- Constraint repair: reject invalid offspring, regenerate
- Track hypervolume after each generation (every 50 evals)
- Implementation: Python with pymoo or custom NSGA-II

### Arm 2: Random Search (h-control-negative)
- 550 independent uniform random configurations
- Track cumulative hypervolume every 50 evaluations
- Same constraint-aware sampling as NSGA-II initial population

### Arm 3: TPE / Model-Based (h-robustness)
- Tree-structured Parzen Estimator (multi-objective variant)
- 550 sequential evaluations with model updates
- Track hypervolume every 50 evaluations
- Implementation: Optuna with MOTPESampler or custom implementation

## Success Criteria

1. **Primary:** NSGA-II reaches 95% of its final hypervolume in fewer evaluations than random (measured at 50-eval intervals). Expected: NSGA-II ≤ 200 evals, random ≥ 300 evals.
2. **Secondary:** The 3-objective Pareto frontier contains ≥ 4 distinct non-dominated points spanning at least 3 different GPU tiers.
3. **Validation:** All arms achieve the same final hypervolume (±2%) at 550 evaluations, confirming the frontier is discoverable by all methods (just at different rates).

## Constraints

- Total evaluations per arm: 550 (achievable in ~110 seconds at 200ms/eval)
- Wall time budget: 3 minutes per arm
- No parallelism required (serial execution sufficient)
- Python implementation (subprocess calls to ./blis)
- No BLIS source modifications needed

## Prior Knowledge

- **RP-1:** Multi-objective comparison requires a non-trivial Pareto frontier. Addressed by adding the gpu_count objective.
- **RP-2:** Validate objective diversity with probe. Done: 5 distinct Pareto points observed across tiers.
- **RP-3:** Rate=20 was too low (created degenerate single-point frontier). Rate=50 with 500 requests creates genuine saturation differentiation.
- **Iteration 1 root cause:** 2-objective space (rps, ttft_p99) with free hardware has a single Pareto-optimal point because more hardware always improves both metrics. Adding cost as objective resolves this.
