# Handoff — Campaign 3, Iteration 1

## Goal

Implement and compare three multi-objective configuration search algorithms (random, NSGA-II, Latin Hypercube) over BLIS's discrete parameter space. Evaluate each at 500 configurations, measure Pareto front hypervolume. Demonstrate that NSGA-II outperforms random sampling within the same budget.

## Key Discoveries

1. **Evaluation speed is ~52ms per 100-request simulation.** 2000 serial evals complete in ~104 seconds. Parallelism is available but not necessary for the budget constraint.

2. **The search space is ~294K configurations** when fully enumerated with all constraints applied. The TP*instances<=8 constraint and conditional routing/scorer rules significantly prune the combinatorial space.

3. **Rate=20 req/s creates massive objective differentiation.** Weak configs (TP=1, instances=1, batch=32) saturate badly: `ttft_p99=16,266ms, rps=2.92`. Strong configs (TP=4, 2 instances): `ttft_p99=25.3ms, rps=10.27`. This 640x range in TTFT and 3.5x in throughput guarantees a non-trivial Pareto frontier.

4. **The system is fully deterministic with `--seed 42`.** Same config + same seed = byte-identical output. No need for multi-seed averaging in iteration 1 (can be added in later iterations for robustness).

5. **`--metrics-path` writes cluster-aggregate JSON** containing all `MetricsOutput` fields including per-request data. The search algorithm should read `responses_per_sec` and `ttft_p99_ms` from this file.

6. **Available hardware names: A100-80, A100-SXM, H100, L40S.** NOT "h100-sxm" or "H100-SXM" — exactly `H100`.

7. **Multi-instance stdout emits per-instance + cluster blocks.** The `--metrics-path` file always contains the cluster aggregate (instance_id="cluster"), which is the correct one for search objectives.

## System Interface

- **Build:** `go build -o blis main.go` (binary already exists at `./blis`)
- **Run baseline:**
  ```bash
  ./blis run --model qwen/qwen3-14b --hardware H100 --latency-model trained-physics \
    --num-requests 100 --rate 20 --tp 2 --num-instances 2 --scheduler fcfs \
    --max-num-running-reqs 128 --max-num-scheduled-tokens 4096 \
    --long-prefill-token-threshold 0 --block-size-in-tokens 16 \
    --routing-policy least-loaded --admission-policy always-admit \
    --preemption-policy fcfs --gpu-memory-utilization 0.9 \
    --seed 42 --metrics-path /tmp/baseline.json
  ```
- **Output format:** JSON at `--metrics-path`, key fields: `responses_per_sec` (float), `ttft_p99_ms` (float)
- **Baseline result:** `responses_per_sec=7.41`, `ttft_p99_ms=33.2`, exit code 0

## Code Map

| Location | What | When to look |
|----------|------|--------------|
| `sim/metrics_utils.go:57` | `MetricsOutput` struct — all JSON field names | If metrics JSON parsing fails or field names seem wrong |
| `sim/metrics.go:66` | `SaveResults()` — computes percentiles, writes JSON | If output values seem wrong or file isn't written |
| `cmd/root.go:184` | `metricsPath` variable | If `--metrics-path` flag behavior is unexpected |
| `cmd/root.go:1727-1728` | Where cluster metrics are saved | If multi-instance metrics path doesn't work |

## Code Targets

All three arms create new Python files in `inputs/` — no BLIS source modifications needed for iteration 1. The search algorithms invoke `./blis run` as a subprocess.

| Arm | File | Purpose |
|-----|------|---------|
| h-main (nsga2-500) | `.nous/campaign-3/runs/iter-1/inputs/search_nsga2.py` | NSGA-II implementation with discrete operators |
| h-control-negative (random-500) | `.nous/campaign-3/runs/iter-1/inputs/search_random.py` | Uniform random search baseline |
| h-robustness (lhs-500) | `.nous/campaign-3/runs/iter-1/inputs/search_lhs.py` | Latin Hypercube stratified sampling |

All three share a common parameter space definition and constraint validation logic. Consider factoring this into a shared module (`inputs/search_common.py`).

## What I Tried That Didn't Work

1. **`--hardware h100-sxm`** — fails with "GPU not found". The valid name is exactly `H100`.
2. **Parsing stdout JSON directly for multi-instance** — stdout has multiple JSON blocks (per-instance + cluster). Regex splitting works but is fragile. Use `--metrics-path` instead, which reliably writes only the cluster aggregate.
3. **`--total-kv-blocks 500` to induce preemption** — works but is not relevant for the search experiment (we use default KV blocks, preemption is naturally rare at rate=20).

## What I Excluded and Why

1. **Multi-seed evaluation** — determinism means single-seed results are stable. Multi-seed can be added in iteration 2 for robustness testing across workload distributions.
2. **Parallel evaluation workers** — at 52ms/eval, 2000 evals complete in 104s serially. Parallelism adds implementation complexity without necessity for the 3-minute constraint. Can be added in iteration 2.
3. **Higher request counts (500, 1000)** — would increase eval time without changing relative ordering of configurations. 100 requests is sufficient for stable P99 metrics.
4. **Model-based optimization (Bayesian/GP)** — requires continuous surrogate modeling over a discrete space. NSGA-II is more natural for discrete multi-objective problems. Model-based approaches can be tested in iteration 2 if NSGA-II proves effective.
5. **PD disaggregation flags** — adds complexity without relevance to the core search algorithm question.

## Evolution of Thinking

Initially assumed the search space would require parallelism to evaluate within 3 minutes. After measuring 52ms/eval, realized serial evaluation of 2000 configs takes only 104s — well within budget. This means the Python implementation can be simple (subprocess.run in a loop) without needing multiprocessing.

Also initially expected preemption to be a key differentiator between configs. At rate=20 with default KV blocks, preemption is actually zero for all tested configs. The primary differentiator is queueing delay: under-provisioned configs (low TP, small batch) queue requests, causing massive TTFT. This means the Pareto frontier is primarily shaped by the TP*instances compute capacity tradeoff (more GPUs = more throughput + lower latency, but fewer config options due to the <=8 constraint).

## Current Status

- **Validated:** BLIS interface, metrics format, evaluation timing, objective diversity, determinism
- **Uncertain:** Whether NSGA-II's discrete crossover will produce valid offspring efficiently (high constraint-violation rate could waste evaluations)
- **Suggested next:** If NSGA-II shows clear advantage, iteration 2 should (a) test portability to different models/hardware/rates, (b) add parallel evaluation, (c) try model-based approaches (TPE, BOHB) as alternatives

## Warnings & Constraints

1. **The `--metrics-path` file includes a `requests` array** with per-request data. This makes the file large (~50KB for 100 requests). The search algorithm should parse only the top-level fields, not iterate over requests.
2. **Conditional parameters:** When `num-instances=1`, do NOT pass `--routing-policy`, `--routing-scorers`, or `--admission-policy` — they are ignored but could cause confusion. Simpler: always pass them (they're silently ignored for single-instance), but be aware the effective search space is smaller for single-instance configs.
3. **Constraint: `long-prefill-token-threshold < max-num-scheduled-tokens` or 0.** Violation doesn't cause an error but produces meaningless behavior. The search algorithm MUST enforce this during config generation.
4. **Hypervolume computation needs a reference point.** Use the worst observed values plus margin: e.g., `(rps=0, ttft_p99=20000)`. This must be consistent across all conditions for fair comparison.
