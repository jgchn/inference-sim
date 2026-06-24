# Problem Framing — Iteration 4: A100-SXM Matrix Completion and TPE Margin Threshold Narrowing

## Research Question

Does bracket K=1 hierarchical search correctly identify the TP winner for qwen3-14b on A100-SXM — the last untested cell in the 2×3 (model × hardware) matrix — and does it remain robust at the tightest TP=4 margin observed in the campaign (1.07% at seed=42)? Separately, can we narrow the flat TPE convergence threshold (currently bounded to [2.1%, 5.1%] by RP-22) using A100-SXM qwen's natural seed-by-seed margin variation (1.07%–4.47%)?

Relevant source files:
- `cmd/root.go:947-984` — CLI flag definitions for all swept parameters
- `cmd/root.go:956` — `--hardware` flag definition (valid: H100, A100-SXM, A100-80, L40S)
- `sim/cluster/metrics.go:418-498` — `ComputeFitness()`, `validFitnessKeysList()`
- `sim/cluster/metrics.go:428-435` — Reference scale constants for fitness normalization
- `sim/latency/config.go:86-101` — `GetHWConfig()` hardware name validation
- `hardware_config.json` — GPU specs: A100-SXM (312 TFLOPS, 80 GiB, 2.039 TB/s)

## System Interface

- **Build:** `go build -o blis main.go` (binary already exists from prior iterations)
- **CLI flags relevant to the experiment:**
  - `--tp` (int): tensor parallelism degree — `cmd/root.go:965`
  - `--num-instances` (int): number of serving instances — `cmd/root.go:966`
  - `--max-num-running-reqs` (int): max batch size — `cmd/root.go:961`
  - `--max-num-scheduled-tokens` (int): max batched tokens — `cmd/root.go:962`
  - `--long-prefill-token-threshold` (int): chunked prefill threshold — `cmd/root.go:963`
  - `--scheduler` (string): scheduling policy — `cmd/root.go:958`
  - `--block-size-in-tokens` (int): KV cache block size — `cmd/root.go:964`
  - `--routing-policy` (string): request routing — `cmd/root.go:970`
  - `--admission-policy` (string): admission control — `cmd/root.go:971`
  - `--preemption-policy` (string): preemption strategy — `cmd/root.go:972`
  - `--fitness-weights` (string): scalarized fitness weights — `cmd/root.go:984`
  - `--seed` (int): RNG seed for deterministic workload — `cmd/root.go:950`
  - `--model` (string): model identifier — `cmd/root.go:947`
  - `--num-requests` (int): total requests to generate — `cmd/root.go:949`
  - `--rate` (float): arrival rate — `cmd/root.go:951`
  - `--hardware` (string): GPU type — `cmd/root.go:956`
  - `--latency-model` (string): latency estimation backend — `cmd/root.go:955`
- **Output format:** JSON metrics between `=== Simulation Metrics ===` markers on stdout. With `--fitness-weights`, a `Score: <float>` line follows in `=== Fitness Evaluation ===` section. Parse score with regex `r'Score:\s+([\d.]+)'`.
- **Multi-instance output:** When `--num-instances > 1`, stdout contains multiple JSON blocks. Use the block with `"instance_id": "cluster"` for aggregate metrics.

## Baseline Command

```bash
./blis run --model qwen/qwen3-14b --hardware A100-SXM --latency-model trained-physics \
  --num-requests 1000 --rate 500 --seed 42 --tp 4 --num-instances 2 \
  --scheduler fcfs --max-num-running-reqs 512 --max-num-scheduled-tokens 8192 \
  --long-prefill-token-threshold 4096 --block-size-in-tokens 16 \
  --routing-policy round-robin --admission-policy always-admit \
  --preemption-policy fcfs \
  --fitness-weights "throughput:0.4,p99_ttft:0.3,p99_e2e:0.3"
```

## Baseline Validation

Ran the baseline command. Exit code 0. Output includes JSON metrics followed by fitness evaluation. Score: **0.138820**. This is the global best for qwen3-14b on A100-SXM at 1000 req/rate=500 (seed=42).

Phase 1 max-profile scores (all validated):
- TP=1/8inst: 0.085746
- TP=2/4inst: 0.115450
- TP=4/2inst: 0.138290 (with least-loaded routing; 0.138820 with round-robin)
- TP=8/1inst: 0.136812

TP=4/2inst wins by 1.07% margin over TP=8/1inst (max-profile comparison).

Cross-seed TP=4 vs TP=8 margins (all max-profile, least-loaded):
- seed=42: TP4=0.138290, TP8=0.136812 → margin 1.07%
- seed=43: TP4=0.143114, TP8=0.137934 → margin 3.62%
- seed=44: TP4=0.139425, TP8=0.137708 → margin 1.24%
- seed=45: TP4=0.147744, TP8=0.141707 → margin 4.09%
- seed=46: TP4=0.147057, TP8=0.140490 → margin 4.47%

## Experimental Conditions

### h-main: Bracket K=1 on A100-SXM qwen (single blis_seed)
Bracket K=1 hierarchical search with max-profile Phase 1 on A100-SXM qwen3-14b.
- blis_seed=42, search_seeds=42,43,44,45,46, budget=100
- Command: `python3 search_blis_iter9.py --strategy adaptive-hierarchical --phase1-k 1 --phase1-profiles bracket --hardware A100-SXM --blis-seeds 42 --seeds 42,43,44,45,46 --rate 500 --num-requests 1000 --model qwen/qwen3-14b --budget 100 --output results/h-main/a100_bracket_k1.json`

### h-robustness: Multi-blis-seed bracket K=1
Same as h-main but with all 5 blis_seeds to test margin variation.
- blis_seeds=42,43,44,45,46, search_seeds=42,43,44,45,46, budget=100
- Command: `python3 search_blis_iter9.py --strategy adaptive-hierarchical --phase1-k 1 --phase1-profiles bracket --hardware A100-SXM --blis-seeds 42,43,44,45,46 --seeds 42,43,44,45,46 --rate 500 --num-requests 1000 --model qwen/qwen3-14b --budget 100 --output results/h-robustness/a100_multi_seed.json`

### h-ablation: Flat TPE at tight margin (seed=42, 1.07%)
Flat TPE over the full 10-dimensional search space on the tightest-margin seed.
- blis_seed=42, search_seeds=42,43,44,45,46, budget=100
- Command: `python3 search_blis_iter9.py --strategy tpe --hardware A100-SXM --blis-seeds 42 --seeds 42,43,44,45,46 --rate 500 --num-requests 1000 --model qwen/qwen3-14b --budget 100 --output results/h-ablation/a100_tpe_tight.json`

### h-control-negative: Flat TPE at wide margin (seed=46, 4.47%)
Same flat TPE but at the widest-margin seed. The bracket K=1 advantage should diminish.
- blis_seed=46, search_seeds=42,43,44,45,46, budget=100
- Command: `python3 search_blis_iter9.py --strategy tpe --hardware A100-SXM --blis-seeds 46 --seeds 42,43,44,45,46 --rate 500 --num-requests 1000 --model qwen/qwen3-14b --budget 100 --output results/h-control-negative/a100_tpe_wide.json`

## Success Criteria

1. **Matrix completion**: Bracket K=1 identifies TP=4/2inst on A100-SXM qwen at all tested seeds (confirms TP=4 dominance on ≥2.039 TB/s bandwidth hardware for both models).
2. **Phase 1 correctness**: 25/25 (blis_seed × search_seed) combinations in h-robustness correctly identify TP=4.
3. **Margin threshold narrowing**: Flat TPE at 1.07% margin (h-ablation) shows strictly worse convergence than at 4.47% margin (h-control-negative), providing a data point between the established bounds [2.1%, 5.1%].
4. **Bracket superiority**: Bracket K=1's deterministic evals_to_best (expected 3) beats flat TPE's median evals_to_best at the tight margin (seed=42).

## Constraints

- Budget: 100 evaluations per (strategy, blis_seed, search_seed) combination.
- Hardware: A100-SXM (312 TFLOPS, 80 GiB, 2.039 TB/s). ~250ms per BLIS evaluation (extrapolated from RP-5 L40S estimate, A100 slightly faster).
- The GLOBAL_BEST_TABLE entry for A100-SXM qwen must be 0.138820 (updated from prior 0.138290).
- Must run blis from project root `/Users/jchen/go/src/inference-sim/inference-sim`.

## Prior Knowledge

- **RP-15**: Bracket K=1 achieves 5/5 Phase 1 correctness on all tested regimes (H100 hard llama, A100 easy llama, A100 hard qwen, L40S hard llama, L40S hard qwen, H100 hard qwen).
- **RP-16**: Bracket K=1 (max-only profile) is sufficient for Phase 1 TP identification.
- **RP-19**: At 2.1% TP margin (L40S hard llama), flat TPE fails 4/5 seeds within budget=100.
- **RP-21**: TP winner is model-dependent on L40S (0.864 TB/s): qwen→TP=8, llama→TP=4. On H100 (3.35 TB/s), both favor TP=4.
- **RP-22**: TPE margin threshold for reliable convergence lies between 2.1% and 5.1%.
- A100-SXM (2.039 TB/s) sits between L40S (0.864 TB/s) and H100 (3.35 TB/s). Expected to follow H100 pattern (TP=4 for both models). Confirmed by probing.
