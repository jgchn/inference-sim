# Problem Framing — Iteration 7: Model Portability

## Research Question

Does NSGA-II's convergence speed advantage over random search transfer across model architectures? Specifically, when the same 6-parameter, 4-objective configuration space is applied to Llama-3.1-8B (smaller model, 32 layers, less KV per token) instead of Qwen3-14B (40 layers, more KV per token), does NSGA-II retain its faster convergence to the exhaustive Pareto front?

The mechanism under test: NSGA-II's advantage arises from the tp×kv_blocks×block_size interaction that creates binary stressed/unstressed regions. A smaller model with lower KV demand per token shifts the stress threshold, reducing the binary dominance effect and increasing Pareto density — which should reduce NSGA-II's advantage magnitude while preserving its directional benefit.

**Key source files:**
- `cmd/hfconfig.go:280` — Model name resolution (`meta-llama/llama-3.1-8b-instruct` → `model_configs/llama-3.1-8b-instruct/`)
- `model_configs/llama-3.1-8b-instruct/config.json` — 32 layers, 32 heads, 8 KV heads, hidden=4096
- `model_configs/qwen3-14b/config.json` — 40 layers, 40 heads, 8 KV heads, hidden=5120
- `sim/kv/cache.go:199-205` — Block allocation with BlockSizeTokens
- `sim/kv/cache.go:222-253` — KV block pre-check triggering preemption

## System Interface

- **Build:** `go build -o blis main.go` (binary already exists at `./blis`)
- **CLI flags:** `--model`, `--hardware`, `--latency-model`, `--num-requests`, `--rate`, `--prefix-tokens`, `--seed`, `--max-num-scheduled-tokens`, `--long-prefill-token-threshold`, `--admission-policy`, `--preemption-policy`, `--block-size-in-tokens`, `--routing-policy`, `--gpu-memory-utilization`, `--tp`, `--num-instances`, `--scheduler`, `--max-num-running-reqs`, `--total-kv-blocks`, `--metrics-path`
- **Code evidence:**
  - `cmd/root.go:919` — `--tp` flag
  - `cmd/root.go:908` — `--num-instances` flag
  - `cmd/root.go:978` — `--total-kv-blocks` flag (default 1000000)
  - `cmd/root.go:28` — `--block-size-in-tokens` flag (default 16)
  - `cmd/root.go:184` — `metricsPath` variable
- **Output:** JSON at `--metrics-path`. Key fields: `responses_per_sec` (float), `ttft_p99_ms` (float), `preemption_count` (int).
- **Objectives (all minimized):** `[-responses_per_sec, ttft_p99_ms, gpu_count, total_kv_blocks]`

## Baseline Command

```bash
./blis run --model meta-llama/llama-3.1-8b-instruct --hardware H100 --latency-model trained-physics \
  --num-requests 200 --rate 50 --prefix-tokens 512 --seed 42 \
  --max-num-scheduled-tokens 4096 --long-prefill-token-threshold 0 \
  --admission-policy always-admit --preemption-policy fcfs \
  --block-size-in-tokens 16 --routing-policy round-robin \
  --gpu-memory-utilization 0.9 \
  --tp 2 --num-instances 2 --scheduler fcfs \
  --max-num-running-reqs 128 --total-kv-blocks 3000 \
  --metrics-path $TMPDIR/baseline_llama.json
```

## Baseline Validation

Ran above command. Exit code 0. Output:
- `responses_per_sec = 19.62`
- `ttft_p99_ms = 1741.3`
- `preemption_count = 50`
- Wall-clock: ~88ms

Comparison with Qwen3-14B at identical params (rate=50): Qwen showed ttft_p99≈3399ms and ~107 preemptions. Llama's lower KV demand (32 layers vs 40) produces less stress at the same kv_blocks, confirming the architectural difference.

## Experimental Conditions

All arms use the same 1800-config parameter space as iterations 5-6:
```
tp: [1, 2, 4, 8]
num_instances: [1..8//tp]
scheduler: [fcfs, sjf]
max_num_running_reqs: [32, 64, 128, 256, 512]
total_kv_blocks: [2000, 3000, 4000, 5000, 7500, 10000]
block_size_in_tokens: [16, 32]
```

**Fixed parameters for all evals:**
```
--model meta-llama/llama-3.1-8b-instruct --hardware H100 --latency-model trained-physics
--num-requests 200 --rate 50 --prefix-tokens 512 --seed 42
--max-num-scheduled-tokens 4096 --long-prefill-token-threshold 0
--admission-policy always-admit --preemption-policy fcfs
--routing-policy round-robin --gpu-memory-utilization 0.9
```

### Arm 1: h-robustness — Exhaustive sweep (Llama-3.1-8B)
Enumerate all 1800 configs. Compute true Pareto front, density, and exhaustive HV. Reference point: (0, 50000, 9, 11000). This provides the normalization baseline for all other arms.

### Arm 2: h-main — NSGA-II (Llama-3.1-8B)
NSGA-II with pop=40, 4 generations (200 total evals). Track convergence every 40 evals. Compare against exhaustive HV from h-robustness.

### Arm 3: h-control-negative — Random search (Llama-3.1-8B)
Uniform random sampling, 200 evals. Track convergence every 40 evals. Compare against exhaustive HV.

### Arm 4: h-ablation — Qwen3-14B re-verification at rate=50
Re-run NSGA-II and random (200 evals each) on Qwen3-14B at rate=50 in the same session, using the same blis_common infrastructure but with `--model qwen/qwen3-14b`. This provides a SAME-SESSION comparison point to eliminate any infrastructure variance between iter-5 and iter-7. Uses iter-5's exhaustive HV (5.857e10) as reference.

## Success Criteria

1. **Portability confirmed:** NSGA-II reaches 95% of exhaustive HV before random search for Llama-3.1-8B (directional advantage transfers).
2. **Density prediction:** Pareto density for Llama-3.1-8B > 4.5% (Qwen's density at rate=50). Predicted range: 6-12% based on weaker block_size interaction.
3. **Magnitude prediction:** NSGA-II convergence gap for Llama < 91 evals (Qwen's gap at rate=50). The weaker stress interaction reduces the exploitable structure for crossover.
4. **Cross-model comparison:** When comparing Llama gap vs Qwen gap (from h-ablation re-verification), the Llama gap is smaller, consistent with higher density.

## Constraints

- Reference point (0, 50000, 9, 11000) used for all arms at both models.
- MC hypervolume: 200k samples, seed=42.
- `--metrics-path` must use `$TMPDIR`.
- Total evals: 1800 (exhaustive) + 200 (NSGA-II) + 200 (random) + 400 (h-ablation Qwen verification) = 2600 evals × ~85ms = ~221s. Well within budget.
- BLIS binary at `./blis` in repo root.
- Gene encoding: (tp_idx, inst_idx, sched_idx, batch_idx, kv_idx, block_size_idx).
- NSGA-II: pop=40, 4 gens, mutation=15%, uniform crossover.

## Prior Knowledge

- **RP-5:** block_size is dominant for high-TP configs when KV is constrained (92x TTFT improvement for Qwen at kv=3000). For Llama, probed: block_size=32 vs 16 at tp=8,kv=3000 → 22ms vs 883ms (40x improvement — weaker than Qwen's 92x).
- **RP-6:** In the corrected 6-parameter space, NSGA-II's convergence advantage is decisive at rate=50 for Qwen3-14B (>80-eval gap). This was measured with 4.5% Pareto density.
- **RP-7:** Adding block_size reduces Pareto density dramatically via threshold effects. For Llama, the threshold effect is weaker (32/40 = 80% KV per token), predicting higher density.
- **RP-10:** Pareto density increases with arrival rate for Qwen. At rate=50, Qwen density = 4.5%.
- **RP-11:** NSGA-II advantage decreases with higher Pareto density (6.89% density at rate=100 → only 57-eval gap vs 91-eval gap at 4.5% density).
