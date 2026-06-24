# Problem Framing — Iteration 8

## Research Question

Is Pareto density a **universal quantitative predictor** of the NSGA-II vs random convergence gap, independent of what causes the density (model architecture vs arrival rate)?

From iter-7, we have two data points measured with the same NSGA-II implementation:
- Qwen3-14B at rate=50: density=4.5%, gap=+4.6 evals (NSGA-II barely faster)
- Llama-3.1-8B at rate=50: density=5.33%, gap=-22.9 evals (random faster)

These suggest a ~5% breakeven, but the two points conflate TWO variables (model architecture AND density). Iter-8 adds two more points by increasing arrival rate (rate=75) for both models. If density is the universal predictor, then:
1. Both models at rate=75 should have higher density than at rate=50 (RP-10 pattern)
2. Both should show negative gaps (random faster) since density > 5%
3. The 4 data points should fall on a single density→gap curve, regardless of model

If the points DON'T align (e.g., Qwen at 6% density shows gap=-5 while Llama at 6% density shows gap=-30), then landscape structure beyond density matters and the predictor is model-specific.

**Source files implementing the mechanism:**
- `sim/metrics_utils.go:57` — `MetricsOutput` struct defining all JSON metrics
- `sim/metrics.go:66` — `SaveResults()` computing percentiles and writing JSON
- `sim/kv/cache.go:199-205` — Block allocation with BlockSizeTokens (drives the KV stress interaction)
- `sim/batch_formation.go:225-304` — `preemptForTokens()` victim selection (creates dominance structure)

## System Interface

- **Build:** `go build -o blis main.go` (binary already current at `./blis`)
- **CLI flags relevant to experiment:**
  - `--model` (string): Model name with org prefix (`cmd/root.go:908`)
  - `--hardware` (string): GPU type (`cmd/root.go:919`)
  - `--rate` (int): Arrival rate in req/s (`cmd/root.go:967`)
  - `--num-requests` (int): Total requests (`cmd/root.go:964`)
  - `--tp` (int): Tensor parallelism (`cmd/root.go:919`)
  - `--num-instances` (int): Instance count (`cmd/root.go:908`)
  - `--scheduler` (string): Scheduling policy (`cmd/root.go:925`)
  - `--max-num-running-reqs` (int): Max batch size (`cmd/root.go:928`)
  - `--total-kv-blocks` (int): KV blocks per instance (`cmd/root.go:978`)
  - `--block-size-in-tokens` (int): Block size (`cmd/root.go:28`)
  - `--metrics-path` (string): JSON output path (`cmd/root.go:184`)
  - `--seed` (int): RNG seed for determinism (`cmd/root.go:961`)
- **Code evidence:** All flags defined in `cmd/root.go` via Cobra flag bindings.
- **Output:** JSON at `--metrics-path` with fields `responses_per_sec` (float), `ttft_p99_ms` (float), `preemption_count` (int). Fourth objective `gpu_count = tp * num_instances` derived from config.

## Baseline Command

```bash
./blis run --model qwen/qwen3-14b --hardware H100 --latency-model trained-physics \
  --num-requests 200 --rate 75 --prefix-tokens 512 --seed 42 \
  --max-num-scheduled-tokens 4096 --long-prefill-token-threshold 0 \
  --admission-policy always-admit --preemption-policy fcfs \
  --routing-policy round-robin --gpu-memory-utilization 0.9 \
  --tp 2 --num-instances 2 --scheduler fcfs \
  --max-num-running-reqs 128 --total-kv-blocks 3000 \
  --block-size-in-tokens 16 --metrics-path $TMPDIR/baseline_qwen_r75.json
```

## Baseline Validation

Executed baseline command. Exit code 0. Output:
- `responses_per_sec=13.46`, `ttft_p99_ms=4277.6`, `preemption_count=90`
- Reference point (0, 50000, 9, 11000) valid: worst Qwen ttft=43186 < 50000, worst Llama ttft=26002 < 50000
- Wall-clock per eval: ~61ms (Qwen rate=75), suitable for 1800-eval exhaustive sweeps in ~110s

## Experimental Conditions

### h-robustness: Exhaustive sweeps at rate=75

Two sub-conditions: enumerate all 1800 configs for each model at rate=75.

**Qwen exhaustive (rate=75):** All 1800 configs with `--model qwen/qwen3-14b --rate 75`. Compute true Pareto front, density, and reference HV.

**Llama exhaustive (rate=75):** All 1800 configs with `--model meta-llama/llama-3.1-8b-instruct --rate 75`. Compute true Pareto front, density, and reference HV.

### h-main: NSGA-II + Random on Qwen at rate=75

Run NSGA-II (pop=40, 4 gens, seed=42) and random search (200 evals, seed=43) on Qwen at rate=75. Track convergence every 40 evals against the exhaustive reference HV from h-robustness. Measure the gap at the 95% HV threshold.

Fixed flags: `--model qwen/qwen3-14b --rate 75` (all other flags same as baseline).

### h-ablation: NSGA-II + Random on Llama at rate=75

Same as h-main but with `--model meta-llama/llama-3.1-8b-instruct --rate 75`. This gives the cross-model data point at the SAME rate, testing whether density (not model) drives the gap.

## Success Criteria

1. **Density measurement:** Both models at rate=75 have Pareto density > rate=50 densities (Qwen > 4.5%, Llama > 5.33%), confirming RP-10's rate-increases-density pattern.
2. **Gap direction:** Both models at rate=75 show NEGATIVE convergence gap (random reaches 95% HV before NSGA-II), since both densities are predicted to be above the ~5% breakeven.
3. **Quantitative prediction test:** All 4 data points (Qwen rate=50, Llama rate=50, Qwen rate=75, Llama rate=75) fall on a monotonically decreasing density→gap curve when plotted. Specifically: higher density → more negative gap, with model identity NOT being an independent predictor (same density from different causes → similar gap).
4. **Reproducibility:** Iter-7 reference data (Qwen rate=50 gap=+4.6, Llama rate=50 gap=-22.9) serves as fixed points for the curve fit.

## Constraints

- Total BLIS invocations: 2×1800 (exhaustive) + 2×400 (NSGA-II+random) = 4400 evals
- Estimated wall-clock: ~5-6 minutes total (within 10-minute budget)
- Reference point: (0, 50000, 9, 11000) for all arms — validated for rate=75 (worst ttft < 50000)
- Same NSGA-II implementation as iter-7 (single offspring per tournament, pop=40, 4 gens)
- Same RNG seeds: NSGA-II seed=42, random seed=43

## Prior Knowledge

- **RP-10:** Pareto density increases with arrival rate: rate=50→4.5%, rate=75→6.06%, rate=100→6.89% (Qwen). Pattern: higher load opens new tradeoff regions.
- **RP-11/RP-12:** NSGA-II speed advantage disappears around 5% density breakeven. At 5.33%, random is 22.9 evals faster.
- **RP-13:** The density→gap mechanism is portable across model architectures (confirmed directionally in iter-7).
- **RP-6:** Including block_size creates a 3-way interaction (tp×kv×block_size) that provides NSGA-II with exploitable gene structure.
- **INV-6:** Determinism — same seed → byte-identical output. No need for multi-seed runs.
