# Problem Framing — Iteration 5: Rate-Dependent TP Winner Transition

## Research Question

Does bracket K=1 hierarchical search correctly adapt to rate-dependent TP winner transitions without any algorithm modification?

The campaign (iterations 1–4) validated bracket K=1 across 2 models × 3 hardware platforms, but always at two fixed rate regimes: easy (200 req/rate=200) and hard (1000 req/rate=500). Both regimes produce stable TP winners that don't change with rate. At rate=100 on H100, however, the TP winner transitions from TP=4 (rate≥150) to TP=8 (rate≤125) — a regime never tested in the campaign. This creates a natural experiment: does the same bracket K=1 algorithm, without modification, correctly identify the rate-dependent TP winner?

**Mechanism under study**: At low arrival rate (100 req/s), the system is under-saturated. Individual request throughput dominates because there's no queuing pressure. Single-instance TP=8 with higher per-request bandwidth outperforms 2-instance TP=4. At high arrival rate (≥200 req/s), queuing pressure dominates and 2 instances provide 2× effective concurrent capacity, favoring TP=4/2inst.

**Key source files**:
- `sim/cluster/metrics.go:418-498` — ComputeFitness() and fitness key validation
- `sim/cluster/metrics.go:428-435` — Reference scale constants for normalization
- `cmd/root.go:947-984` — CLI flag definitions for all swept parameters
- `sim/latency/config.go:86-101` — GetHWConfig() hardware validation

## System Interface

- **Build:** `go build -o blis main.go` (binary already exists at project root)
- **CLI flags relevant to experiment:**
  - `--rate` (`cmd/root.go:940`): Request arrival rate (req/s). Controls saturation level.
  - `--num-requests` (`cmd/root.go:939`): Total requests to simulate.
  - `--tp` (`cmd/root.go:947`): Tensor parallelism degree.
  - `--num-instances` (`cmd/root.go:948`): Number of model instances. Constrained: TP × instances ≤ 8.
  - `--max-num-running-reqs` (`cmd/root.go:952`): Max concurrent requests per instance.
  - `--max-num-scheduled-tokens` (`cmd/root.go:953`): Max batched tokens per step.
  - `--long-prefill-token-threshold` (`cmd/root.go:954`): Chunked prefill threshold.
  - `--fitness-weights` (`cmd/root.go:1761`): Composite fitness scoring weights.
  - `--hardware` (`cmd/root.go:956`): GPU hardware type (H100, A100-SXM, L40S).
  - `--seed` (`cmd/root.go:938`): Workload generation RNG seed.
- **Output format:** JSON metrics between `=== Simulation Metrics ===` markers on stdout. Fitness score on stdout as `Score: <float>` in `=== Fitness Evaluation ===` section. Warnings/timing on stderr.

## Baseline Command

```bash
./blis run --model qwen/qwen3-14b --hardware H100 --latency-model trained-physics \
  --num-requests 1000 --rate 100 --seed 42 --tp 8 --num-instances 1 \
  --scheduler fcfs --max-num-running-reqs 512 --max-num-scheduled-tokens 8192 \
  --long-prefill-token-threshold 4096 --block-size-in-tokens 16 \
  --routing-policy round-robin --admission-policy always-admit \
  --preemption-policy fcfs \
  --fitness-weights "throughput:0.4,p99_ttft:0.3,p99_e2e:0.3"
```

## Baseline Validation

Ran successfully with exit code 0. Score: **0.164680** (TP=8/1inst, H100, qwen3-14b, rate=100, 1000 requests, max-profile secondary params).

Full Phase 1 TP sweep at rate=100 (seed=42, max-profile):

| TP | Instances | Score    |
|----|-----------|----------|
| 1  | 8         | 0.108774 |
| 2  | 4         | 0.136114 |
| 4  | 2         | 0.155788 |
| 8  | 1         | 0.164680 |

Monotonically increasing — TP=8 is the clear winner at rate=100.

## Experimental Conditions

### Rate-TP crossover map (H100, qwen3-14b, seed=42, max-profile)

| Rate | TP4/2inst | TP8/1inst | Winner | Margin  |
|------|-----------|-----------|--------|---------|
| 100  | 0.155788  | 0.164680  | TP8    | +5.71%  |
| 125  | 0.162149  | 0.162182  | TP8    | +0.02%  |
| 150  | 0.166271  | 0.164011  | TP4    | +1.38%  |
| 200  | 0.170350  | 0.166195  | TP4    | +2.50%  |
| 500  | 0.178117  | 0.169720  | TP4    | +4.95%  |
| 1000 | 0.180227  | 0.170996  | TP4    | +5.40%  |

Crossover between rate=125 and rate=150. Seed-dependent at rate=125 (4/5 seeds TP=8, 1/5 seed=46 TP=4).

### Multi-seed validation at rate=100 (H100, qwen3-14b, max-profile)

| Seed | TP4/2inst | TP8/1inst | Winner | Margin |
|------|-----------|-----------|--------|--------|
| 42   | 0.155788  | 0.164680  | TP8    | 5.71%  |
| 43   | 0.154721  | 0.164862  | TP8    | 6.55%  |
| 44   | 0.155872  | 0.165671  | TP8    | 6.29%  |
| 45   | 0.158850  | 0.167968  | TP8    | 5.74%  |
| 46   | 0.159548  | 0.167901  | TP8    | 5.24%  |

TP=8 wins at all 5 seeds with 5.24–6.55% margins. Robust regime.

### Multi-seed at rate=125 (crossover point)

| Seed | TP4/2inst | TP8/1inst | Winner | Margin |
|------|-----------|-----------|--------|--------|
| 42   | 0.162149  | 0.162182  | TP8    | 0.02%  |
| 43   | 0.161052  | 0.162267  | TP8    | 0.75%  |
| 44   | 0.161849  | 0.162947  | TP8    | 0.68%  |
| 45   | 0.165885  | 0.167234  | TP8    | 0.81%  |
| 46   | 0.166259  | 0.165980  | TP4    | 0.17%  |

Winner is seed-dependent. All scores are within 1% of each other across TP=4 and TP=8.

### Llama cross-model verification (H100, max-profile, seed=42)

| Rate | TP4/2inst | TP8/1inst | Winner | Margin |
|------|-----------|-----------|--------|--------|
| 100  | 0.177317  | 0.182394  | TP8    | 2.86%  |
| 500  | 0.209195  | 0.194142  | TP4    | 7.75%  |
| 1000 | 0.212400  | 0.195603  | TP4    | 8.59%  |

Same rate-dependent TP transition for llama: TP=8 at rate=100, TP=4 at rate≥500.

### Arm conditions

**h-main**: Bracket K=1 at rate=100, 1000 req, H100 qwen3-14b. 5 search seeds, blis_seed=42. Budget=100.

**h-robustness**: Bracket K=1 at rate=100, 1000 req, H100 qwen3-14b. 5 blis_seeds × 5 search seeds. Budget=100.

**h-ablation**: Bracket K=1 at rate=125 (crossover), 1000 req, H100 qwen3-14b. 5 blis_seeds × 5 search seeds. Budget=100.

**h-control-negative**: Flat TPE at rate=100, 1000 req, H100 qwen3-14b. 5 search seeds, blis_seed=42. Budget=100.

## Success Criteria

1. **h-main**: 5/5 search seeds select TP=8/1inst as Phase 1 winner. evals_to_best=4 on all seeds.
2. **h-robustness**: 25/25 (blis_seed × search_seed) combinations select TP=8/1inst. evals_to_best=4 on all.
3. **h-ablation**: At the crossover (rate=125), both TP=4 and TP=8 scores are within 1% of each other at all seeds. evals_to_best=3 on all seeds (TP=4 at position 3 already exceeds the 1% threshold because it's so close to TP=8).
4. **h-control-negative**: Flat TPE converges with median evals_to_best < 20, faster than typical rate=500 regimes, because the monotonic landscape at rate=100 is easier for TPE's surrogate model.

## Constraints

- Must run blis from project root `/Users/jchen/go/src/inference-sim/inference-sim`.
- Hardware name is case-sensitive: `H100`.
- Model name format: `qwen/qwen3-14b`, `meta-llama/llama-3.1-8b-instruct`.
- GLOBAL_BEST_TABLE uses seed=42 values.
- Parse fitness from stdout (`Score: <float>`), not from JSON file.
- Single-instance configs (TP=8/1inst) have degenerate routing (routing_policy has no effect).
- Active TP=8 budget constraint: TP × instances ≤ 8.

## Prior Knowledge

**Active principles that apply:**
- **RP-1**: TP is the dominant performance knob (~87% of fitness variance). Confirmed: at rate=100, the TP level explains >90% of fitness difference.
- **RP-9**: Phase 1 is deterministic per blis_seed regardless of search_seed. Applies directly to h-robustness predictions.
- **RP-15/16**: Bracket K=1 max-only profile achieves 5/5 Phase 1 correctness across all tested regimes. This iteration extends the claim to rate=100.
- **RP-21**: TP winner is model-dependent on low-bandwidth hardware but not on high-bandwidth (H100). At rate=100, H100 shows TP=8 for both qwen and llama — consistent.
- **RP-22**: TPE convergence reliability depends on margin width AND hardware-specific landscape structure. At rate=100, the 5.7% TP=8 margin predicts good TPE convergence.

**New finding not in principles:** The TP winner is rate-dependent. At low arrival rates (≤125 req/s on H100), individual request throughput dominates and TP=8/1inst wins. At high rates (≥150 req/s), concurrency benefits from 2 instances dominate and TP=4/2inst wins. The crossover rate on H100 is ~125 req/s for qwen3-14b (seed-dependent at the boundary).
