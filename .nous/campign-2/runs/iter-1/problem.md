# Problem Framing: Pareto-Efficient Configuration Search for BLIS

## Research Question

Given an 86,400-configuration search space (TP × instances × scheduler × batching × routing × admission × preemption), can a structured hierarchical search algorithm discover the Pareto frontier (throughput vs tail latency) within 500 evaluations, achieving ≥95% hypervolume ratio compared to exhaustive enumeration of a representative subspace?

**Mechanism under study:** The search space has hierarchical structure — TP and instance count determine hardware parallelism and dominate throughput capacity, while batching/scheduling knobs fine-tune the throughput-latency tradeoff within a fixed parallelism tier. A hierarchical decomposition that first identifies promising parallelism tiers (Phase 1) then optimizes within-tier knobs (Phase 2) should outperform random search by concentrating budget on productive regions.

**Key source files:**
- `sim/metrics_utils.go:57` — `MetricsOutput` struct defining observable metrics
- `sim/metrics_utils.go:67` — `TokensPerSec` field (throughput objective)
- `sim/metrics_utils.go:73` — `E2EP99Ms` field (tail latency objective)
- `cmd/root.go:184` — `metricsPath` flag definition
- `cmd/root.go:2038` — `--metrics-path` flag registration

## System Interface

- **Build command:** `go build -o blis main.go` (binary already built at `./blis`)
- **Run command:** `./blis run` with configuration flags
- **Output mechanism:** `--metrics-path <file>` writes JSON with cluster-level aggregates; stdout emits per-instance + cluster metrics blocks

### CLI Flags Relevant to Experiment

| Flag | Type | Code Evidence |
|------|------|---------------|
| `--model` | string | `cmd/root.go:2038` |
| `--hardware` | string | Valid: `A100-80`, `A100-SXM`, `H100`, `L40S` |
| `--tp` | int | Tensor parallelism degree |
| `--num-instances` | int | Cluster replicas |
| `--rate` | float | Arrival rate (req/s) |
| `--num-requests` | int | Total requests to simulate |
| `--seed` | int | RNG seed (deterministic) |
| `--latency-model` | string | `trained-physics` recommended |
| `--scheduler` | string | `fcfs`, `priority-fcfs`, `sjf`, `reverse-priority` |
| `--max-num-running-reqs` | int | Max batch size |
| `--max-num-scheduled-tokens` | int | Max batched tokens |
| `--long-prefill-token-threshold` | int | Chunked prefill threshold |
| `--block-size-in-tokens` | int | KV cache block size |
| `--routing-policy` | string | `round-robin`, `least-loaded`, `weighted` |
| `--admission-policy` | string | `always-admit`, `tier-shed` |
| `--preemption-policy` | string | `fcfs`, `priority` |
| `--metrics-path` | string | Output file for MetricsOutput JSON |
| `--fitness-weights` | string | Scalar fitness (not used for Pareto) |

### Metrics Available in Output

From `MetricsOutput` (`sim/metrics_utils.go:57-87`):
- `tokens_per_sec` — aggregate throughput (output tokens/s across cluster)
- `e2e_p99_ms` — 99th percentile end-to-end latency (ms)
- `ttft_p99_ms` — 99th percentile time-to-first-token (ms)
- `responses_per_sec` — request throughput
- `scheduling_delay_p99_ms` — queueing delay
- `preemption_count` — number of preemption events
- `completed_requests`, `still_queued`, `timed_out_requests` — completion status

**Pareto objectives:** Maximize `tokens_per_sec`, minimize `e2e_p99_ms`.

## Baseline Command

```bash
./blis run --model qwen3-14b --hardware H100 --tp 2 --num-instances 2 \
  --rate 50 --num-requests 200 --seed 42 \
  --latency-model trained-physics \
  --scheduler fcfs --max-num-running-reqs 128 \
  --max-num-scheduled-tokens 4096 --block-size-in-tokens 16 \
  --routing-policy round-robin --admission-policy always-admit \
  --preemption-policy fcfs --metrics-path /tmp/baseline.json
```

## Baseline Validation

Executed baseline command. Results:
- **Exit code:** 0
- **Output file:** `/tmp/baseline.json` (valid JSON)
- **Key metrics:** `tokens_per_sec = 8195.97`, `e2e_p99_ms = 10602.11`
- **Execution time:** ~86ms per invocation (measured across 10 runs)
- **Determinism:** Same seed produces identical output (verified)

At rate=50, the search space produces wide differentiation:
- Worst config tested (TP=1, 1 instance, batch=32): tokens_per_sec=1832, e2e_p99=51406ms
- Best config tested (TP=8, 1 instance, batch=512): tokens_per_sec=14232, e2e_p99=4719ms
- ~8x throughput range and ~11x latency range across configs

## Experimental Conditions

### Condition 1: Random Search (Baseline)

Uniformly sample 500 configurations from the full 86,400-point space. Evaluate each at rate=50, num_requests=200, seed=42. Compute the Pareto frontier and hypervolume.

**Command pattern per evaluation:**
```bash
./blis run --model qwen3-14b --hardware H100 --tp {tp} --num-instances {ni} \
  --rate 50 --num-requests 200 --seed 42 --latency-model trained-physics \
  --scheduler {sched} --max-num-running-reqs {max_run} \
  --max-num-scheduled-tokens {max_tok} --long-prefill-token-threshold {prefill} \
  --block-size-in-tokens {blk} --routing-policy {route} \
  --admission-policy {admit} --preemption-policy {preempt} \
  --metrics-path {output_path}
```

### Condition 2: Hierarchical Decomposition Search

Two-phase structured search:
- **Phase 1 (budget: 15 evaluations):** Evaluate each valid (TP, num_instances) pair with a "max-throughput profile" (scheduler=fcfs, max_running=256, max_tokens=8192, block_size=16, routing=least-loaded, admission=always-admit, preemption=fcfs). 15 pairs total. Rank by tokens_per_sec.
- **Phase 2 (budget: 485 evaluations):** Allocate remaining budget to top-3 parallelism tiers proportionally. Within each tier, use Latin Hypercube Sampling across the 7 remaining knobs (scheduler, max_running, max_tokens, prefill_threshold, block_size, routing, admission, preemption) to explore diverse configurations.

### Condition 3: Exhaustive Ground Truth (subset)

Enumerate all configurations for a representative subset of the space to compute true Pareto frontier and reference hypervolume. Use (TP=4, instances=2) tier — 2880 configs. This provides ground truth for the hypervolume ratio metric.

**Total BLIS invocations:** 500 (random) + 500 (hierarchical) + 960 (ground truth subset for TP=2×4inst + TP=4×2inst) = 1960 ≤ 2000 budget.

## Success Criteria

1. **Hypervolume ratio:** Hierarchical search achieves ≥95% hypervolume of the exhaustive ground truth within the budget-matched evaluation count.
2. **Improvement over random:** Hierarchical search discovers ≥20% more hypervolume than random search at the same budget (500 evaluations).
3. **Pareto cardinality:** Hierarchical search finds at least as many Pareto-optimal points as random search.
4. **Wall time:** Both strategies complete within 120 seconds on a single CPU core (budget of 500 × ~100ms = ~50s expected).

## Constraints

- Total BLIS invocations across all conditions: ≤ 2000
- Single model/hardware/rate combo: qwen3-14b / H100 / rate=50
- Each evaluation uses num_requests=200, seed=42 for determinism
- Implementation in Python, invoking `./blis run` as subprocess
- No external dependencies beyond numpy/scipy (for Latin Hypercube Sampling)
- Hardware constraint: TP × num_instances ≤ 8 (single node)

## Prior Knowledge

This is the first iteration of campaign-2. No prior findings exist.

The previous campaign (campaign-1, iterations 1-10 in `search_blis*.py`) established:
- TP is the dominant knob — it determines throughput capacity ceiling
- Hierarchical search with TP-first phase outperforms random for single-objective optimization
- The `parse_cluster_metrics` stdout parsing pattern works reliably
- Execution time is ~80-100ms per BLIS invocation

This iteration extends to **multi-objective Pareto frontier** discovery, which the previous campaign did not address.
