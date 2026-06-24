# Problem Framing: Configuration Search Algorithm for BLIS

## Research Question

What search algorithm can efficiently discover the Pareto frontier of BLIS configurations across 10 knobs (86,400 total combinations) within minutes on CPU-only hardware?

The search space consists of 10 configuration parameters (TP, replicas, scheduler, max batch size, max batched tokens, chunked prefill threshold, block size, routing policy, admission policy, preemption policy) with a constraint that `TP × replicas ≤ 8` (single H100 SXM node). BLIS evaluations are extremely fast (~100ms per run with 200 requests), making the system amenable to sample-efficient search algorithms.

**Key source files:**
- `cmd/root.go:947-984` — CLI flag definitions for all swept parameters
- `sim/cluster/metrics.go:418-498` — Fitness computation: valid keys, normalization, weighted scoring
- `sim/cluster/metrics.go:428-435` — Reference scales for metric normalization (referenceRPS=100, referenceTPS=10000, referenceTicks=1000)

## System Interface

### Build command
```bash
go build -o blis main.go
```

### CLI flags relevant to the experiment

| Flag | Type | Code Reference | Semantics |
|------|------|---------------|-----------|
| `--model` | string | `cmd/root.go:935` | LLM model name |
| `--hardware` | string | `cmd/root.go:936` | GPU type (valid: A100-80, A100-SXM, H100, L40S) |
| `--latency-model` | string | `cmd/root.go:942` | Backend: roofline, trained-physics |
| `--tp` | int | `cmd/root.go:957` | Tensor parallelism degree |
| `--num-instances` | int | `cmd/root.go:963` | Cluster replicas |
| `--scheduler` | string | `cmd/root.go:977` | fcfs, priority-fcfs, sjf, reverse-priority |
| `--max-num-running-reqs` | int | `cmd/root.go:947` | Max concurrent requests in batch (default 256) |
| `--max-num-scheduled-tokens` | int | `cmd/root.go:948` | Max new tokens per step (default 2048) |
| `--long-prefill-token-threshold` | int | `cmd/root.go:952` | Chunked prefill trigger (0=disabled) |
| `--block-size-in-tokens` | int | `cmd/root.go:951` | KV cache block size |
| `--routing-policy` | string | `cmd/root.go:973` | round-robin, least-loaded, weighted, always-busiest |
| `--admission-policy` | string | `cmd/root.go:966` | always-admit, tier-shed, etc. |
| `--preemption-policy` | string | `cmd/root.go:978` | fcfs, priority |
| `--fitness-weights` | string | `cmd/root.go:984` | Weighted objective: key:value pairs |
| `--metrics-path` | string | `cmd/root.go:2038` | JSON output file for aggregate metrics |
| `--num-requests` | int | `cmd/root.go:938` | Number of requests to generate |
| `--rate` | float | `cmd/root.go:961` | Arrival rate (req/s) |
| `--seed` | int | `cmd/root.go:956` | RNG seed for determinism |

### Output format

**stdout** (deterministic): JSON metrics block between `=== Simulation Metrics ===` header, followed by `=== Fitness Evaluation ===` with score and per-component breakdown when `--fitness-weights` is set.

**`--metrics-path`**: JSON file containing aggregate metrics and per-request details. Does NOT include fitness score — fitness must be computed from raw metrics by the search script or parsed from stdout.

### Observable metrics (from `sim/cluster/metrics.go:420-426`)

Fitness-compatible keys: `throughput`, `tokens_per_sec`, `p99_ttft`, `p50_ttft`, `mean_ttft`, `p99_e2e`, `p50_e2e`, `mean_e2e`.

Additional output fields: `completed_requests`, `preemption_count`, `dropped_unservable`, `timed_out_requests`, `scheduling_delay_p99_ms`.

## Baseline Command

```bash
./blis run \
  --model qwen/qwen3-14b \
  --hardware H100 \
  --latency-model trained-physics \
  --num-requests 200 \
  --rate 50 \
  --seed 42 \
  --tp 1 \
  --num-instances 1 \
  --scheduler fcfs \
  --max-num-running-reqs 256 \
  --max-num-scheduled-tokens 2048 \
  --long-prefill-token-threshold 0 \
  --block-size-in-tokens 16 \
  --routing-policy round-robin \
  --admission-policy always-admit \
  --preemption-policy fcfs \
  --fitness-weights "throughput:0.4,p99_ttft:0.3,p99_e2e:0.3" \
  --metrics-path /tmp/blis_baseline.json
```

## Baseline Validation

Validated: exit code 0. Output:
- `responses_per_sec`: 5.24 (TP=1, 1 instance, rate=200, 200 requests)
- `e2e_p99_ms`: 35062 ms
- `ttft_p99_ms`: 22557 ms
- Fitness Score: 0.0200

Quick TP sweep (8 GPUs, 100 requests, rate=50):
- TP=1, 8 instances: Score=0.028
- TP=2, 4 instances: Score=0.043
- TP=4, 2 instances: Score=0.059
- TP=8, 1 instance:  Score=0.072

Wall-clock timings:
- 50 requests: 22ms
- 200 requests: 77ms
- 500 requests: 163ms

## Experimental Conditions

### Condition 1: Random Search (Baseline)
Python script that uniformly samples configurations from the search space, runs BLIS for each, and collects the Pareto frontier. Evaluated at budgets of 100, 200, 500, and 1000 evaluations.

### Condition 2: Bayesian Optimization (TPE via Optuna)
Python script using Optuna's TPE sampler to search the same space, with the fitness score as the optimization objective. Same evaluation budgets. Optuna handles the TP × instances constraint via a conditional search space.

### Condition 3: Latin Hypercube Sampling (LHS)
Python script using stratified sampling via Latin Hypercube to ensure better coverage of the space. Same evaluation budgets.

All conditions:
- Fixed workload: `--num-requests 200 --rate 50 --seed 42`
- Fixed objective: `--fitness-weights "throughput:0.4,p99_ttft:0.3,p99_e2e:0.3"`
- Fixed model/hardware: `--model qwen/qwen3-14b --hardware H100 --latency-model trained-physics`
- Output: best fitness score found, Pareto set size, wall-clock time

### Key observations that shape the design

1. **Routing/admission are no-ops for single-instance configs**: Verified that routing-policy does not affect metrics when `--num-instances 1` (identical scores for round-robin vs least-loaded). The search algorithm should know this to avoid wasting evaluations.

2. **TP is the dominant knob**: Higher TP consistently improves per-request latency (lower TTFT/E2E) at fixed total GPU count. The search algorithm should prioritize exploring TP early.

3. **Search space is 86,400 configs**: Exhaustive search takes ~2.4 hours. A good search algorithm should find near-optimal configs in 200-500 evaluations (~20-50 seconds wall clock).

## Success Criteria

1. **Efficiency**: Bayesian optimization (TPE) finds a configuration within 5% of the best-known fitness score using ≤ 200 evaluations, while random search requires > 500 evaluations to reach the same quality.
2. **Speed**: The full search (including Python overhead) completes in under 5 minutes for 500 evaluations.
3. **Pareto quality**: The discovered Pareto frontier from TPE at 200 evaluations dominates or matches the random search frontier at 500 evaluations (measured by hypervolume indicator).

## Constraints

- CPU-only execution (no GPU required — BLIS is a simulator)
- Single H100 SXM node constraint: `TP × num-instances ≤ 8`
- Python search script — must use commonly available libraries (optuna, numpy, scipy)
- Each BLIS evaluation should use `--num-requests 200` for stable metrics (not too few for noisy statistics, not too many for slow runs)
- Deterministic: fixed `--seed 42` across all evaluations for reproducibility

## Prior Knowledge

This is the first iteration. No prior principles or findings exist.
