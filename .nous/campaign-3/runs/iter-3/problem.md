# Problem Framing — Campaign 3, Iteration 3

## Research Question

Can NSGA-II outperform random search on a multi-objective configuration optimization problem when the parameter space includes non-linear interactions between parameters?

Previous iterations (1-2) showed that on BLIS's default parameter space (1M KV blocks), the Pareto-optimal configuration set is trivially dense (~60% of configs per GPU tier achieve tier-optimal metrics), making all search algorithms converge identically fast. This iteration tests whether adding `total_kv_blocks` as a searchable parameter — creating non-linear batch×blocks×TP interactions — produces a genuinely harder search problem where NSGA-II's selection pressure provides measurable convergence advantage.

Key mechanism: `--total-kv-blocks` is per-instance (`sim/kv/cache.go:47`). With TP=4, inst=1 (4 GPUs, 1 instance), only 1×blocks are available. With TP=2, inst=2 (4 GPUs, 2 instances), 2×blocks are available. This creates a non-obvious TP↔blocks interaction where higher TP gives faster processing but LESS total KV capacity, potentially inducing preemptions that degrade TTFT.

Source files implementing the mechanism:
- `sim/batch_formation.go:225-304` — `preemptForTokens()` triggered when `AllocateKVBlocks` returns false
- `sim/kv/cache.go:222-253` — pre-check: `numNewBlocks + cachedFromFreeList > countFreeBlocks()` → reject
- `cmd/root.go:577-598` — KV capacity auto-calculation, `--total-kv-blocks` flag at line 946

## System Interface

- **Build:** `go build -o blis main.go` (binary already exists at `./blis`)
- **CLI flags relevant to experiment:**
  - `--total-kv-blocks int` (default 1000000) — per-instance KV cache blocks (`cmd/root.go:946`)
  - `--prefix-tokens int` — shared prefix token count for synthesis mode (`cmd/root.go:2024`)
  - `--tp int` — tensor parallelism (`cmd/root.go:938`)
  - `--num-instances int` — number of serving instances (`cmd/root.go:936`)
  - `--max-num-running-reqs int` — max batch size (`cmd/root.go:940`)
  - `--scheduler string` — scheduling policy (`cmd/root.go:942`)
  - `--routing-policy string` — routing selection (`cmd/root.go:960`)
  - `--routing-scorers string` — weighted scorer config (`cmd/root.go:962`)
  - `--metrics-path string` — JSON output file path (`cmd/root.go:184`)
- **Output format:** JSON at `--metrics-path`, key fields: `responses_per_sec` (float), `ttft_p99_ms` (float), `preemption_count` (int). Fourth objective `gpu_count` derived from config: `tp * num_instances`. Fifth parameter `kv_blocks` directly from config.
- **Code evidence:** MetricsOutput struct at `sim/metrics_utils.go:57`; SaveResults at `sim/metrics.go:66`

## Baseline Command

```bash
./blis run --model qwen/qwen3-14b --hardware H100 --latency-model trained-physics \
  --num-requests 200 --rate 50 --tp 2 --num-instances 2 --scheduler fcfs \
  --max-num-running-reqs 128 --max-num-scheduled-tokens 4096 \
  --long-prefill-token-threshold 0 --block-size-in-tokens 16 \
  --routing-policy weighted \
  --routing-scorers "precise-prefix-cache:2,queue-depth:1,kv-utilization:1" \
  --admission-policy always-admit --preemption-policy fcfs \
  --gpu-memory-utilization 0.9 --prefix-tokens 512 --total-kv-blocks 5000 \
  --seed 42 --metrics-path $TMPDIR/baseline.json
```

## Baseline Validation

Exit code 0. Output file produced at `$TMPDIR/baseline.json` (~80KB for 200 requests).
Key metrics: `responses_per_sec=14.83`, `ttft_p99_ms=39.13`, `preemption_count=0`.
With blocks=3000: `responses_per_sec=12.92`, `ttft_p99_ms=3437.13`, `preemption_count=75`.

## Experimental Conditions

### Parameter Space (4-objective formulation)

| Parameter | CLI flag | Values |
|-----------|----------|--------|
| TP | --tp | {1, 2, 4, 8} |
| Replicas | --num-instances | {1..floor(8/TP)} |
| Scheduler | --scheduler | {fcfs, sjf} |
| Max batch size | --max-num-running-reqs | {32, 64, 128, 256, 512} |
| KV blocks | --total-kv-blocks | {2000, 3000, 4000, 5000, 7500, 10000} |
| Block size | --block-size-in-tokens | {16, 32} |
| Routing policy | --routing-policy | {round-robin, least-loaded, weighted} |
| Weighted scorer | --routing-scorers | 4 preset profiles |
| GPU mem util | --gpu-memory-utilization | {0.85, 0.9, 0.95} |

**Constraints:**
- TP * replicas <= 8
- If replicas == 1: routing/scorer not swept (round-robin only)
- Weighted scorer only when routing-policy == weighted

**Fixed parameters:**
- `--model qwen/qwen3-14b --hardware H100 --latency-model trained-physics`
- `--num-requests 200 --rate 50 --prefix-tokens 512 --seed 42`
- `--max-num-scheduled-tokens 4096 --long-prefill-token-threshold 0`
- `--admission-policy always-admit --preemption-policy fcfs`

**Objectives (4D):**
1. maximize `responses_per_sec`
2. minimize `ttft_p99_ms`
3. minimize `gpu_count` (= tp × num_instances)
4. minimize `total_kv_blocks` (memory cost proxy)

**Reference point for hypervolume:** (0, 50000, 9, 11000) — safely dominated by all valid configs.

### Arm 1: h-main (NSGA-II, 4-objective)
NSGA-II with population_size=40, 5 generations (200 total evaluations). Track 4D hypervolume at each generation. Uses constraint-aware crossover respecting TP*inst<=8.

### Arm 2: h-control-negative (Random, 4-objective)
Uniform random search with 200 evaluations on the same 4-objective space. Track 4D hypervolume every 40 evaluations.

### Arm 3: h-robustness (NSGA-II, 3-objective, no KV constraint)
NSGA-II with population=40, 5 generations (200 evals) on the ORIGINAL 3-objective space from iter 2 (no `total_kv_blocks` parameter, fixed at 1M default). Confirms that the 3-obj problem remains trivially easy even at reduced budget.

## Success Criteria

1. **Primary**: h-main (NSGA-II 4-obj) reaches 95% of its final 4D hypervolume in fewer evaluations than h-control-negative (random 4-obj). The convergence gap should be statistically observable (>1 generation = 40 evals difference).
2. **Secondary**: h-main discovers more non-dominated solutions than h-control-negative at the same evaluation count.
3. **Validation**: h-robustness (NSGA-II 3-obj) shows no convergence advantage over what random achieved in iter 2 (both hit 95% at eval≤50), confirming the 3-obj problem remains easy.

## Constraints

- Total evaluations per arm: 200 (budget constraint from iteration strategy)
- Wall time per arm: <120 seconds (200 evals × ~200-400ms each)
- Evaluation via subprocess: `./blis run ... --metrics-path $TMPDIR/metrics_<eval_id>.json`
- Must use `$TMPDIR` for metrics files (sandbox constraint)
- All arms use seed=42 for BLIS (determinism INV-6); algorithm RNG uses different seeds per arm

## Prior Knowledge

- **RP-1/RP-2**: Previous iterations confirmed the standard 3-objective space is too easy (>60% Pareto density per tier). This iteration creates difficulty by adding a parameter with non-linear interactions.
- **RP-3**: Batch≥256 threshold no longer universally applies when KV blocks are constrained. At blocks=3000, batch=128/256/512 all produce identical results because KV is the binding constraint, not batch capacity.
- **RP-4**: TPE performs worse than random due to conditional parameter incompatibility. Not retested.
- **RP-5**: TP/instances remain dominant, but KV blocks create a SECOND dominant dimension that interacts non-linearly with TP. This iteration validates whether that interaction creates genuine search difficulty.

### Key Probe Results

| Config | rps | ttft_p99 | gpu_count | blocks | preemptions |
|--------|-----|----------|-----------|--------|-------------|
| tp=2,i=2 | 14.83 | 39 | 4 | 5000 | 0 |
| tp=4,i=1 | 16.76 | 2640 | 4 | 5000 | 77 |
| tp=4,i=1 | 18.74 | 1105 | 4 | 10000 | 0 |
| tp=8,i=1 | 25.17 | 620 | 8 | 5000 | 33 |
| tp=2,i=2 | 12.92 | 3437 | 4 | 3000 | 75 |
| tp=2,i=2 (sjf) | 13.34 | 4839 | 4 | 3000 | 102 |
| tp=1,i=1 | 5.64 | 15965 | 1 | 5000 | 107 |
| tp=1,i=1 | 3.04 | 48934 | 1 | 1500 | 182 |

**Critical insight**: At gpu_count=4, blocks=5000: tp=4,i=1 (16.76, 2640) and tp=2,i=2 (14.83, 39) are a GENUINE Pareto tradeoff (higher rps vs lower ttft). This tradeoff did NOT exist in iter 2 (1M blocks) where tp=4 dominated on both metrics. The non-linear TP↔blocks interaction reverses dominance relationships.
