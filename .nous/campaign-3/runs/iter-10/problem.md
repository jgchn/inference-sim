# Problem Framing — Iteration 10

## Research Question

Does the **Crossover Yield Advantage** (crossover_yield / pareto_density) predict the direction of the NSGA-II convergence gap across different model architectures and load regimes?

Iter-9 established that at rate=100, NSGA-II has a universal structural advantage for both Qwen (+62.3 evals) and Llama (+40.3 evals). It also showed that overall NDR (~35.6%) and per-axis NDR are nearly identical between models, disqualifying NDR as a differentiator. The remaining open question: **why does Llama show random-wins (gap=-22.9) at rate=50 but NSGA-II-wins at rate=100?**

This iteration tests a new mechanistic hypothesis: NSGA-II's advantage depends on how reliably crossover between Pareto-optimal parents produces Pareto-optimal offspring ("crossover yield"). When crossover yield vastly exceeds Pareto density, NSGA-II's recombination operator is much more efficient than random sampling → positive gap. When yield is close to density, crossover offers no advantage → random can win via superior diversity.

**Evidence grounding (from iter-9 data probe):**
- Qwen rate=100: crossover yield = 66.18%, density = 6.89% → yield advantage = 9.6x → gap = +62.3
- Llama rate=100: crossover yield = 63.98%, density = 5.89% → yield advantage = 10.9x → gap = +40.3
- Cliff-free subspace (kv=10000): crossover yield = 100%, density = 10.67% → yield advantage = 9.4x → gap = +29.0
- Cliff-free Pareto front is a perfect Cartesian product (4 tp × 2 scheduler × 2 batch × 2 block_size, all with instances=1)

All three rate=100 conditions have yield advantage ~9-11x and all show large positive gaps. The prediction: at rate=50, Llama's yield advantage will drop dramatically (close to 1-2x), explaining its negative gap.

**Mechanism** (grounded in code): The Pareto front's spatial clustering determines crossover reliability. At rate=100, both models are KV-stressed: the Pareto front concentrates on block_size=32 (93.5% for Qwen), kv≤5000 (via the kv-as-objective pressure), and batch≥128. This concentration makes crossover between Pareto parents highly likely to produce another Pareto config. At rate=50, Llama (lighter model: 8B params, 32 layers — `model_configs/llama-3.1-8b-instruct/config.json`) has sufficient capacity that most configurations perform comparably well. The Pareto front becomes scattered (high entropy, approaching uniform distribution), crossover yield drops to ~density level, and NSGA-II loses its structural advantage.

## System Interface

- **Build:** `go build -o blis main.go` (binary in repo root)
- **CLI flags used:**
  - `--model` (`cmd/root.go:968`) — model name
  - `--hardware` (`cmd/root.go:961`) — GPU type
  - `--rate` (`cmd/root.go:967`) — arrival rate (req/s)
  - `--tp` (`cmd/root.go:919`) — tensor parallelism
  - `--num-instances` (`cmd/root.go:908`) — instance count
  - `--scheduler` (`cmd/root.go:935`) — scheduling policy
  - `--max-num-running-reqs` (`cmd/root.go:923`) — max batch size
  - `--total-kv-blocks` (`cmd/root.go:978`) — KV blocks per instance
  - `--block-size-in-tokens` (`cmd/root.go:28`) — tokens per KV block
  - `--metrics-path` (`cmd/root.go:184`) — JSON output path
  - `--num-requests` (`cmd/root.go:965`) — total requests
  - `--seed` (`cmd/root.go:969`) — RNG seed
  - `--prefix-tokens` (`cmd/root.go:971`) — prefix token count
  - `--latency-model` (`cmd/root.go:956`) — latency estimation mode
  - `--max-num-scheduled-tokens` (`cmd/root.go:925`) — max scheduled tokens
  - `--long-prefill-token-threshold` (`cmd/root.go:927`) — chunked prefill
  - `--admission-policy` (`cmd/root.go:941`) — admission control
  - `--preemption-policy` (`cmd/root.go:929`) — preemption strategy
  - `--routing-policy` (`cmd/root.go:937`) — request routing
  - `--gpu-memory-utilization` (`cmd/root.go:976`) — GPU memory fraction
- **Output format:** JSON at `--metrics-path` with fields: `responses_per_sec`, `ttft_p99_ms`, `preemption_count`
- **Derived objectives:** `gpu_count = tp * num_instances`, `kv_blocks = total_kv_blocks`

## Baseline Command

```bash
./blis run --model qwen/qwen3-14b --hardware H100 --latency-model trained-physics \
  --num-requests 200 --rate 50 --prefix-tokens 512 --seed 42 \
  --max-num-scheduled-tokens 4096 --long-prefill-token-threshold 0 \
  --admission-policy always-admit --preemption-policy fcfs \
  --routing-policy round-robin --gpu-memory-utilization 0.9 \
  --tp 2 --num-instances 2 --scheduler fcfs \
  --max-num-running-reqs 128 --total-kv-blocks 3000 \
  --block-size-in-tokens 16 --metrics-path $TMPDIR/baseline_qwen_r50.json
```

## Baseline Validation

- **Qwen rate=50, tp=2, inst=2, kv=3000, bs=16:** Exit code 0, wall time 99ms. `responses_per_sec=13.14`, `ttft_p99_ms=3399.2`, `preemption_count=78`. KV-stressed regime confirmed.
- **Qwen rate=50, tp=2, inst=2, kv=10000, bs=16:** Exit code 0, wall time 64ms. `responses_per_sec=15.19`, `ttft_p99_ms=39.4`, `preemption_count=0`. Stress-free baseline.
- **Llama rate=50, tp=2, inst=2, kv=3000, bs=16:** Exit code 0, wall time 107ms. `responses_per_sec=19.62`, `ttft_p99_ms=1741.3`, `preemption_count=50`. Moderate KV stress (lower than Qwen at same config).

## Experimental Conditions

### h-robustness: Exhaustive sweep at rate=50

Run ALL 1800 configs for both models (Qwen + Llama) at rate=50, saving ALL per-config results. This provides the foundational data for all subsequent analysis.

**Parameters:** Same 6-parameter space as iter-9:
- tp: [1, 2, 4, 8], num_instances: derived (1..8//tp)
- scheduler: [fcfs, sjf]
- max_num_running_reqs: [32, 64, 128, 256, 512]
- total_kv_blocks: [2000, 3000, 4000, 5000, 7500, 10000]
- block_size_in_tokens: [16, 32]

**Fixed parameters:** rate=50, num_requests=200, seed=42, all others as baseline.

**Total evaluations:** 3600 (1800 × 2 models).
**Estimated wall time:** ~3 min with 8 workers per model (~100ms/eval × 1800 / 8 = 22.5s per model).

**Outputs:**
- `results/h-robustness/all_results_qwen_r50.json` — all 1800 configs with objectives
- `results/h-robustness/all_results_llama_r50.json` — all 1800 configs with objectives
- `results/h-robustness/pareto_front_qwen_r50.json` — Pareto front + density + NDR + crossover yield
- `results/h-robustness/pareto_front_llama_r50.json` — same for Llama

### h-main: Crossover yield comparison across conditions

Compute crossover yield and yield advantage for 4 conditions:
1. Qwen rate=50 (from h-robustness data)
2. Llama rate=50 (from h-robustness data)
3. Qwen rate=100 (from iter-9 cached: `runs/iter-9/results/h-robustness/all_results_qwen_r100.json`)
4. Llama rate=100 (from iter-9 cached: `runs/iter-9/results/h-robustness/all_results_llama_r100.json`)

**Method:** For each condition, sample 10000 crossover offspring from random pairs of Pareto parents (uniform crossover, normalize for tp/instances constraint). Compute:
- `crossover_yield` = fraction of valid offspring that are Pareto-optimal
- `yield_advantage` = crossover_yield / pareto_density
- `pareto_closure_ratio` = |Pareto set| / |Cartesian product of Pareto-appearing values|
- `pareto_entropy` = mean normalized Shannon entropy across parameter axes

**No BLIS runs needed** — pure analysis from cached data.

**Primary prediction:** Yield advantage rank matches gap sign:
- All conditions with yield_advantage > 5 will have positive NSGA-II gap
- Llama rate=50 will have yield_advantage < 3 (crossover offers limited benefit)
- Qwen rate=50 will have yield_advantage > 5

### h-ablation: Algorithm comparison at rate=50

NSGA-II vs random comparison at rate=50 for both models using iter-9 methodology:
- Pop size = 40, generations = 4, total budget = 200
- Convergence checkpoints every 40 evals
- NSGA-II seed = 42, random seed = 43
- Reference point: (0, 50000, 9, 11000)

**Predictions:**
- Qwen rate=50: positive gap (NSGA-II wins), consistent with iter-7's +4.6
- Llama rate=50: gap ≤ 0 (random wins or tie), consistent with iter-7's -22.9

Uses cached all_results from h-robustness — zero BLIS subprocess calls.

### h-control-negative: Null model validation

Compute crossover yield under a **random Pareto labeling**: randomly designate N configs as "Pareto" (same N as actual Pareto size), compute crossover yield. Repeat 100 times to get distribution. This validates that high observed yield is a meaningful signal from spatial clustering, not a statistical artifact of the crossover operator or space structure.

**Prediction:** Null model crossover yield ≈ pareto_density (within ±1%). Observed yield is 5-10x higher than null, confirming that the Pareto front's spatial structure is the source of yield advantage.

Uses cached all_results — zero BLIS subprocess calls.

## Success Criteria

1. **Primary (yield predicts gap sign):** Across all 4 conditions (Qwen/Llama × rate=50/100), yield_advantage > 5 correctly identifies positive gap, and yield_advantage < 3 correctly identifies negative/zero gap. Zero sign mismatches.

2. **Secondary (yield advantage ordering):** Yield advantage rank-correlates with gap magnitude across all 4 conditions (Spearman ρ > 0.8).

3. **Control validation:** Null model crossover yield is within 2× of Pareto density for all conditions (confirming that high observed yield is structurally meaningful, not a baseline artifact).

4. **Reproducibility:** h-ablation gap values at rate=50 are consistent with iter-7 reports (same sign, magnitude within 50% — since methodology may differ slightly from iter-7's original).

## Constraints

- Total BLIS evaluations: 3600 (exhaustive sweeps only). All algorithm comparisons use cached results.
- Estimated wall time: <5 minutes for BLIS runs + <30 seconds for analysis.
- `--metrics-path` must use `$TMPDIR` (sandbox constraint).
- Reference point: (0, 50000, 9, 11000) for 4-objective space.
- NSGA-II parameters: pop=40, gen=4, mutation=15%, same as iter-9.

## Prior Knowledge

**Applicable principles:**
- RP-1: Pareto density 4.5-5.33% at rate=50. Confirmed by iter-7.
- RP-6: NSGA-II has decisive advantage when density < 5% and exploitable structure exists.
- RP-11: Llama shows rate-dependent behavior — negative gaps at 50-75, positive at 100. Breakpoint between 75-100.
- RP-12: No universal density breakeven predicts gap sign across models.
- RP-14: Multiple exploitable axes contribute to NSGA-II advantage (kv NDR~0.71, batch NDR~0.50, block_size NDR~0.26).
- RP-15: Overall NDR is NOT a reliable gap predictor (identical NDR, different gaps).
- RP-16: At rate=100, NSGA-II has universal advantage for both models.

**From iter-9 analysis (new findings this iteration):**
- Crossover yield at rate=100: ~64-66% for both models
- Yield advantage at rate=100: ~9.6-10.9x for both models
- Cliff-free Pareto front forms a perfect Cartesian product (100% yield, 9.4x advantage)
- Full-space Pareto closure ratio: 16.15% (far from Cartesian product)
- Block_size=32 dominates Pareto front at rate=100 (93.5%)
