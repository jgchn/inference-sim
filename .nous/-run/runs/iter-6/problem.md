# Problem Framing — Iteration 6: Cross-Hardware Rate Crossover

## Research Question

Does the rate-dependent TP winner transition (TP=8 at low arrival rates, TP=4 at high arrival rates) discovered on H100 in iter-5 also exist on A100-SXM hardware? If so, does bracket K=1 hierarchical search correctly adapt without modification, and how does the crossover rate relate to hardware memory bandwidth?

Relevant source files:
- `sim/latency/config.go:86-101` — `GetHWConfig()` hardware validation
- `hardware_config.json` — GPU specs: H100 (3.35 TB/s), A100-SXM (2.039 TB/s), L40S (0.864 TB/s)
- `sim/cluster/metrics.go:418-498` — Fitness computation and normalization
- `cmd/root.go:947-984` — CLI flag definitions for all swept parameters

## System Interface

- **Build:** `go build -o blis main.go` (binary already exists)
- **CLI flags relevant to experiment:**
  - `--hardware` (`cmd/root.go:956`): GPU type. Values: H100, A100-SXM, L40S (case-sensitive)
  - `--tp` (`cmd/root.go:949`): Tensor parallelism degree
  - `--num-instances` (`cmd/root.go:950`): Deployment replicas (TP x instances <= 8)
  - `--rate` (`cmd/root.go:940`): Arrival rate in req/s
  - `--num-requests` (`cmd/root.go:939`): Total request count
  - `--seed` (`cmd/root.go:938`): Workload RNG seed
  - `--fitness-weights` (`cmd/root.go:1761-1765`): Fitness weight string
  - `--max-num-running-reqs`, `--max-num-scheduled-tokens`, `--long-prefill-token-threshold`: Batch formation knobs
- **Code evidence:** Fitness computation at `sim/cluster/metrics.go:428-435` uses reference scale constants for normalization. Valid fitness keys at `sim/cluster/metrics.go:420-426`.
- **Native output:** Fitness score printed to stdout as `Score: <float>` after `=== Fitness Evaluation ===` section. Cluster-level JSON metrics in `=== Simulation Metrics ===` block with `"instance_id": "cluster"`.

## Baseline Command

```bash
./blis run --model qwen/qwen3-14b --hardware A100-SXM --latency-model trained-physics \
  --num-requests 1000 --rate 100 --seed 42 --tp 8 --num-instances 1 \
  --scheduler fcfs --max-num-running-reqs 512 --max-num-scheduled-tokens 8192 \
  --long-prefill-token-threshold 4096 --block-size-in-tokens 16 \
  --routing-policy round-robin --admission-policy always-admit \
  --preemption-policy fcfs \
  --fitness-weights "throughput:0.4,p99_ttft:0.3,p99_e2e:0.3"
```

## Baseline Validation

Ran the baseline command. Exit code 0. Output: `Score: 0.130754`.

Verified Phase 1 max-profile TP landscape on A100-SXM qwen3-14b at rate=100 (seed=42):
- TP=1/8inst: 0.080450
- TP=2/4inst: 0.105531
- TP=4/2inst: 0.127461
- TP=8/1inst: 0.130754

Monotonically increasing — TP=8 wins by 2.5% over TP=4. Confirmed with seed=43: TP=4=0.129098, TP=8=0.131048 (1.5% margin). Consistent across seeds.

Additional probes establishing rate crossover on A100-SXM qwen3-14b (max-profile, seed=42):
- rate=100: TP=8 wins by 2.5% (0.130754 vs 0.127461)
- rate=150: TP=8 wins by 1.1% (0.133406 vs 0.131902)
- rate=200: Essentially tied (TP=4=0.134488 vs TP=8=0.134486, margin 0.001%)
- rate=500: TP=4 wins (known from RP-23, global best 0.138820)

## Experimental Conditions

All conditions use the search script `search_blis_iter6.py` (from existing iter-6 h-main.patch) with GLOBAL_BEST_TABLE updates for new regimes (see code_changes in bundle).

### Condition 1: h-main — Bracket K=1 on A100-SXM qwen rate=100

```bash
python3 search_blis_iter6.py --strategy adaptive-hierarchical --phase1-k 1 \
    --phase1-profiles bracket --hardware A100-SXM \
    --budget 100 --seeds 42,43,44,45,46 \
    --rate 100 --num-requests 1000 --model qwen/qwen3-14b \
    --output results/h-main/results.json
```

### Condition 2: h-robustness — Multi-seed bracket K=1 on A100-SXM llama rate=100

```bash
for blis_seed in 42 43 44 45 46; do
  python3 search_blis_iter6.py --strategy adaptive-hierarchical --phase1-k 1 \
      --phase1-profiles bracket --hardware A100-SXM \
      --budget 100 --seeds 42,43,44,45,46 --blis-seed $blis_seed \
      --rate 100 --num-requests 1000 --model meta-llama/llama-3.1-8b-instruct \
      --output results/h-robustness/blis_seed_${blis_seed}.json
done
```

### Condition 3: h-ablation — Bracket K=1 at crossover rate=200

```bash
python3 search_blis_iter6.py --strategy adaptive-hierarchical --phase1-k 1 \
    --phase1-profiles bracket --hardware A100-SXM \
    --budget 100 --seeds 42,43,44,45,46 \
    --rate 200 --num-requests 1000 --model qwen/qwen3-14b \
    --output results/h-ablation/results.json
```

### Condition 4: h-control-negative — Flat TPE on A100-SXM qwen at rate=100

```bash
python3 search_blis_iter6.py --strategy tpe --hardware A100-SXM \
    --budget 100 --seeds 42,43,44,45,46 \
    --rate 100 --num-requests 1000 --model qwen/qwen3-14b \
    --output results/h-control-negative/results.json
```

## Success Criteria

1. **h-main**: All 5 search seeds select TP=8/1inst as Phase 1 winner. evals_to_best=4 on all seeds (TP=8 at Phase 1 position 4). Phase 1 determinism confirmed (identical scores across search_seeds per RP-9).

2. **h-robustness**: 25/25 (blis_seed x search_seed) combinations select TP=8/1inst. evals_to_best=4 on all 25 combinations. TP=8 margins range 1.5-4% across blis_seeds.

3. **h-ablation**: evals_to_best=3 on all 5 seeds at rate=200. TP=4 at position 3 exceeds 1% threshold. Phase 1 TP winner is TP=4 or TP=8 (essentially tied at crossover).

4. **h-control-negative**: Flat TPE all 5 seeds find global best. Median evals_to_best < 15 (faster than rate=500 median=26 from RP-10/A100). Bracket K=1 (evals_to_best=4) still outperforms flat TPE median.

## Constraints

- TP x instances <= 8 (single-node GPU constraint)
- BLIS binary must be run from project root directory
- Hardware names are case-sensitive: A100-SXM, H100, L40S
- Fitness score parsed from stdout regex `r'Score:\s+([\d.]+)'`
- Multi-instance output requires parsing cluster-level JSON block (`"instance_id": "cluster"`)
- GLOBAL_BEST_TABLE entries must be added for all new (model, hardware, rate, num_requests) combinations

## Prior Knowledge

Directly relevant active principles:
- **RP-15**: Bracket K=1 achieves Phase 1 correctness across all tested models, hardware, and rates (extended in iter-5 to rate regimes on H100)
- **RP-16**: Bracket K=1 (max-only) is sufficient for Phase 1 TP identification
- **RP-21**: TP winner is model-dependent on low-bandwidth hardware (L40S), model-independent on high-bandwidth (H100, A100-SXM)
- **RP-24**: TP winner is rate-dependent on H100: TP=8 at rate<=125, TP=4 at rate>=150 (qwen3-14b)
- **RP-25**: Bracket K=1 correctly adapts to rate-dependent transitions without algorithm modification (validated on H100 only)
- **RP-9**: Phase 1 is deterministic given fixed blis_seed (identical across search_seeds)
- **RP-13**: TP=8 dominance requires max_running >= 256

New probing data grounding this iteration's design:
- A100-SXM qwen crossover at rate~200 (higher than H100's ~125)
- A100-SXM llama at rate=100: TP=8 wins by 4.3% (wider margin than qwen's 2.5%)
- L40S llama crossover at rate~450; L40S qwen: no crossover (TP=8 always wins)
- Pattern: crossover_rate inversely proportional to memory bandwidth
