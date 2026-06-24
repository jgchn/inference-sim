# Problem Framing — Iteration 6

## Research Question

Does NSGA-II's convergence advantage over random search (established at rate=50 in iter-5: ~91-eval gap to 95% of exhaustive HV) **generalize to higher load regimes** (rate=100, rate=75)?

Iter-5 proved the advantage exists at rate=50 on the 1800-config corrected space (6 parameters, 4 objectives). This iteration tests whether the advantage is portable across load intensities — specifically, whether stronger saturation (rate=100) amplifies or maintains the convergence gap.

The mechanism under test: higher arrival rate shifts the KV stress threshold upward, causing more mid-range configs to become saturated. This concentrates the Pareto front further into the narrow band of high-TP + (block_size=32 OR high kv_blocks) configs, giving NSGA-II's crossover more exploitable structure.

**Key source files:**
- `sim/kv/cache.go:199-205` — Block allocation with BlockSizeTokens (capacity = blocks × block_size)
- `sim/kv/cache.go:222-253` — KV block pre-check triggering preemption
- `sim/batch_formation.go:225-304` — `preemptForTokens()` victim selection
- `cmd/root.go:978` — `--total-kv-blocks` flag definition
- `cmd/root.go:28` — `--block-size-in-tokens` flag (default 16)

## System Interface

- **Build:** `go build -o blis main.go` (binary already exists at `./blis`)
- **CLI flags relevant to experiment:**
  - `--rate <N>` — Request arrival rate (req/s). Defined in `cmd/root.go`.
  - `--tp <N>` — Tensor parallelism degree. `cmd/root.go:919`.
  - `--num-instances <N>` — Instance count. `cmd/root.go:908`.
  - `--scheduler <name>` — Scheduling policy. `cmd/root.go`.
  - `--max-num-running-reqs <N>` — Max batch size. `cmd/root.go`.
  - `--total-kv-blocks <N>` — KV blocks per instance. `cmd/root.go:978`.
  - `--block-size-in-tokens <N>` — Tokens per KV block. `cmd/root.go:28`.
  - `--metrics-path <path>` — Native JSON output (never redirect stdout). `cmd/root.go:184`.
- **Output format:** JSON at `--metrics-path`. Key fields: `responses_per_sec` (float), `ttft_p99_ms` (float), `preemption_count` (int). Fourth objective: `gpu_count = tp * num_instances`.
- **Code evidence:** Metrics struct at `sim/metrics_utils.go:57`, written by `sim/metrics.go:66`.

## Baseline Command

```bash
./blis run --model qwen/qwen3-14b --hardware H100 --latency-model trained-physics \
  --num-requests 200 --rate 100 --prefix-tokens 512 --seed 42 \
  --max-num-scheduled-tokens 4096 --long-prefill-token-threshold 0 \
  --admission-policy always-admit --preemption-policy fcfs \
  --block-size-in-tokens 16 --routing-policy round-robin \
  --gpu-memory-utilization 0.9 \
  --tp 2 --num-instances 2 --scheduler fcfs \
  --max-num-running-reqs 128 --total-kv-blocks 3000 \
  --metrics-path $TMPDIR/baseline_iter6.json
```

## Baseline Validation

Exit code 0. Output at `$TMPDIR/baseline_iter6.json`.
Result: `responses_per_sec=13.53`, `ttft_p99_ms=4776.3`, `preemption_count=108`.
Wall-clock time: 76ms. This confirms: (1) rate=100 stresses the config heavily (vs rate=50's ttft_p99=3399ms for same config), (2) eval times remain fast, (3) the reference point (0, 50000, 9, 11000) remains valid (worst observed ttft_p99=43760ms < 50000).

## Experimental Conditions

All arms use the same 1800-config corrected parameter space as iter-5:
```
tp: [1, 2, 4, 8]
num_instances: range(1, 8//tp + 1)  # derived
scheduler: [fcfs, sjf]
max_num_running_reqs: [32, 64, 128, 256, 512]
total_kv_blocks: [2000, 3000, 4000, 5000, 7500, 10000]
block_size_in_tokens: [16, 32]
```

### h-robustness: Exhaustive sweep at rate=100
- Enumerate all 1800 configs with `--rate 100`
- Compute true Pareto front and exhaustive HV
- Record convergence every 40 evals
- Expected wall time: ~130s (1800 × ~72ms avg)

### h-main: NSGA-II at rate=100, budget=200
- Identical algorithm to iter-5's h-main (pop=40, 4 gens, 15% mutation, uniform crossover)
- Only change: `--rate 100` instead of `--rate 50`
- Track HV convergence vs exhaustive (rate=100) reference every 40 evals

### h-control-negative: Random search at rate=100, budget=200
- Identical to iter-5's h-control-negative but at `--rate 100`
- Track HV convergence every 40 evals

### h-ablation: NSGA-II and Random at rate=75, budget=200
- Both algorithms at `--rate 75` on the same 1800-config space
- Provides intermediate data point for monotonicity analysis
- NSGA-II: pop=40, 4 gens. Random: 200 evals (seed=43).
- Track HV convergence every 40 evals
- Requires its own exhaustive HV reference (from a preliminary exhaustive sweep at rate=75)
- Due to budget constraints, run exhaustive at rate=75 first (~130s), then run both algorithms

## Success Criteria

1. **Portability confirmed:** NSGA-II at rate=100 reaches 95% of exhaustive HV within 200 evals (as it did at rate=50).
2. **Advantage preserved:** The convergence gap (evals for NSGA-II to reach 95% minus evals for random to reach 95%, or ∞ if random never reaches 95%) at rate=100 is >= 60 evals (iter-5 established 91 evals at rate=50).
3. **Pareto density monotonicity:** Pareto density at rate=100 <= Pareto density at rate=50 (4.5%).
4. **Monotonicity of advantage:** If rate=75 data shows gap_75 between gap_50 (91) and gap_100, this confirms monotonic relationship between load and algorithm advantage.
5. **Direction of amplification:** Final HV gap at rate=100 (NSGA-II% - random%) >= 4.9% (iter-5's value at rate=50), indicating the advantage is maintained or amplified.

## Constraints

- Total BLIS invocations: exhaustive_rate100 (1800) + NSGA-II_rate100 (200) + random_rate100 (200) + exhaustive_rate75 (1800) + NSGA-II_rate75 (200) + random_rate75 (200) = 4400 total. All within 3 minutes wall time (estimated 4400 × 72ms ≈ 317s for sequential; exhaustive sweeps can run in separate sequential phases).
- Reference point: (0, 50000, 9, 11000) for ALL arms and rates.
- Seed: 42 for all BLIS evaluations, 42 for NSGA-II RNG, 43 for random search RNG.
- `--metrics-path` must use `$TMPDIR` (sandbox constraint).

## Prior Knowledge

- **RP-1, RP-6, RP-7:** Established at rate=50: Pareto density 4.5%, NSGA-II gap >80 evals, block_size inclusion amplifies advantage.
- **RP-5, RP-8:** block_size × kv_blocks × tp three-way interaction creates threshold effects. At rate=100, the stress threshold shifts higher (need more effective KV capacity).
- **RP-3:** Batch size threshold ~256 at rate=50. At rate=100, higher rates may raise this threshold.
- **Iter-5 findings:** NSGA-II 99.2% vs random 94.3% at budget=200 (rate=50). Gap ~91 evals. Pareto density 4.5%. Exhaustive HV 5.857e+10.
- **Iter-5 probes of rate=100:** tp=2,inst=1,kv=3000,bs=16 → rps=8.54, ttft=13270ms. tp=4,inst=2,kv=5000,bs=16 → rps=26.78, ttft=29ms. Dramatic differentiation confirmed.
