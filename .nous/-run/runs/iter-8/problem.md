# Problem Framing — Iteration 8: Model-Size Generalization and Formula Limits

## Research Question

Does the lean bracket search algorithm (TP={4,8}, max-profile, no Phase 2, budget=2) generalize to larger models (qwen3-32b, 2.3x the parameter count of qwen3-14b), and does the simple bandwidth formula (RP-27: `crossover_rate ≈ 400/bandwidth_TB_s`) break for model sizes beyond ~14B?

Iteration 7 established lean bracket as a 50x budget reduction over standard bracket K=1, achieving within 1% of global best in ~0.5s across 2 models (qwen3-14b, llama-3.1-8b) and 2 hardware platforms (H100, A100-SXM). This iteration extends the validation to:
1. A third model (qwen3-32b) that shifts the TP crossover rate significantly higher
2. The final untested hardware platform (L40S)
3. Tests whether reducing simulation horizon (num_requests) can further reduce search time

### Source Code References
- `search_blis_iter7.py:86-89` — LEAN_PHASE1_TP_CONFIGS defining TP={4,8} evaluation order
- `search_blis_iter7.py:618-621` — tp_candidates="lean" filter in `run_adaptive_hierarchical_search()`
- `sim/latency/roofline.go:291-351` — rooflineStepTime: the physics behind TP-dependent step latency
- `sim/latency/trained_physics_model.go:263-284` — TP communication overhead that penalizes higher TP degrees
- `hardware_config.json` — GPU specs used by the formula predictor

## System Interface

- **Build:** `go build -o blis main.go` (binary already exists)
- **CLI flags relevant to the experiment:**
  - `--model`: Model path (e.g., `qwen/qwen3-32b`). Defined at `cmd/root.go:955`.
  - `--hardware`: GPU type (`H100`, `A100-SXM`, `L40S`). Defined at `cmd/root.go:956`. Validated at `sim/latency/config.go:86-101`.
  - `--tp`: Tensor parallelism degree. Defined at `cmd/root.go:959`.
  - `--num-instances`: Number of instances. Defined at `cmd/root.go:960`.
  - `--num-requests`: Total requests to simulate. Defined at `cmd/root.go:937`.
  - `--rate`: Arrival rate in req/s. Defined at `cmd/root.go:940`.
  - `--seed`: Workload RNG seed. Defined at `cmd/root.go:938`.
  - `--fitness-weights`: Fitness score composition. Parsed at `cmd/root.go:1761-1765`.
  - `--max-num-running-reqs`, `--max-num-scheduled-tokens`, `--long-prefill-token-threshold`, `--block-size-in-tokens`: Batch/scheduling params. Defined at `cmd/root.go:947-984`.
  - `--scheduler`, `--routing-policy`, `--admission-policy`, `--preemption-policy`: Policy selection.
- **Output format:** JSON metrics between `=== Simulation Metrics ===` markers on stdout. With `--fitness-weights`, `Score: <float>` follows in `=== Fitness Evaluation ===` section. Parse with regex `r'Score:\s+([\d.]+)'`.
- **Code evidence for fitness computation:** `sim/cluster/metrics.go:418-498` (`ComputeFitness()`).

## Baseline Command

```bash
./blis run --model qwen/qwen3-32b --hardware H100 --latency-model trained-physics \
  --num-requests 1000 --rate 500 --seed 42 --tp 4 --num-instances 2 \
  --scheduler fcfs --max-num-running-reqs 512 --max-num-scheduled-tokens 8192 \
  --long-prefill-token-threshold 4096 --block-size-in-tokens 16 \
  --routing-policy least-loaded --admission-policy always-admit \
  --preemption-policy fcfs \
  --fitness-weights "throughput:0.4,p99_ttft:0.3,p99_e2e:0.3"
```

## Baseline Validation

Ran the baseline command; exit code 0. Output includes:
- `Score: 0.141403` (qwen3-32b, H100, rate=500, TP=4/2inst, max-profile)
- Wall time: ~0.28s per evaluation

Additional validation probes:
- H100 qwen3-32b TP=8/1inst rate=500: Score=0.140230 (TP=4 wins by 0.84%)
- H100 qwen3-32b TP=8/1inst rate=300: Score=0.138983 (TP=8 wins — crossover is between 300-500)
- A100-SXM qwen3-32b TP=8/1inst rate=500: Score=0.108182 (TP=8 always wins, no crossover)
- L40S qwen3-32b TP=8/1inst rate=500: Score=0.064436 (TP=8 always wins)

## Experimental Conditions

### h-main: Lean bracket on qwen3-32b (5 regimes × 5 seeds)

Run lean bracket (TP={4,8}, max-profile, skip-phase2) on qwen3-32b:

| Regime | Hardware | Rate | Expected TP Winner |
|--------|----------|------|--------------------|
| 1 | H100 | 100 | TP=8 |
| 2 | H100 | 300 | TP=8 |
| 3 | H100 | 500 | TP=4 |
| 4 | A100-SXM | 100 | TP=8 |
| 5 | A100-SXM | 500 | TP=8 |

Each regime: 5 search seeds (42-46). Total: 25 runs.

Command per regime (example: H100 rate=500):
```bash
python3 search_blis_iter8.py --strategy adaptive-hierarchical --phase1-k 1 \
    --phase1-profiles bracket --tp-candidates lean --skip-phase2 \
    --hardware H100 --budget 2 --seeds 42,43,44,45,46 \
    --rate 500 --num-requests 1000 --model qwen/qwen3-32b \
    --output results/h-main/h100-rate500.json
```

### h-control-negative: Bandwidth formula fails for 32B models

Run the `formula-predict` strategy across all 18 (model × hardware × rate) combinations in the campaign. The formula predicts `crossover_rate = 400 / bandwidth_TB_s` and picks TP=4 if rate > crossover, else TP=8.

Test matrix (18 regimes):
- 3 models: qwen3-14b, llama-3.1-8b, qwen3-32b
- 3 hardware: H100, A100-SXM, L40S
- 2 rates: 100, 500

Command:
```bash
python3 search_blis_iter8.py --strategy formula-predict \
    --hardware H100 --rate 500 --num-requests 1000 --model qwen/qwen3-32b \
    --output results/h-control-negative/formula.json
```

No BLIS evaluations needed — the formula-predict strategy outputs a prediction and compares against GLOBAL_BEST_TABLE.

### h-ablation: Micro-eval (100 requests) misidentifies TP winner

Run lean bracket with `--num-requests 100` on 4 regimes where the 1000-request lean bracket identifies TP=4:

| Regime | Hardware | Model | Rate | 1000-req winner |
|--------|----------|-------|------|-----------------|
| 1 | H100 | qwen3-14b | 500 | TP=4 |
| 2 | A100-SXM | qwen3-14b | 500 | TP=4 |
| 3 | H100 | llama-3.1-8b | 500 | TP=4 |
| 4 | H100 | qwen3-32b | 500 | TP=4 |

Each regime: 1 seed (42, deterministic). Total: 4 runs.

Command per regime (example):
```bash
python3 search_blis_iter8.py --strategy adaptive-hierarchical --phase1-k 1 \
    --phase1-profiles bracket --tp-candidates lean --skip-phase2 \
    --hardware H100 --budget 2 --seeds 42 \
    --rate 500 --num-requests 100 --model qwen/qwen3-14b \
    --output results/h-ablation/h100-qwen14b-100req.json
```

### h-robustness: Lean bracket on L40S (4 regimes × 5 seeds)

Run lean bracket on L40S for both models at rates 100 and 500:

| Regime | Model | Rate | Expected TP Winner |
|--------|-------|------|--------------------|
| 1 | qwen3-14b | 100 | TP=8 |
| 2 | qwen3-14b | 500 | TP=8 |
| 3 | llama-3.1-8b | 100 | TP=8 |
| 4 | llama-3.1-8b | 500 | TP=4 |

Each regime: 5 search seeds (42-46). Total: 20 runs.

## Success Criteria

1. **h-main**: All 25 (regime × seed) lean bracket runs for qwen3-32b achieve within 1% of per-regime global best.
2. **h-control-negative**: The bandwidth formula (`400/BW`) achieves ≤78% accuracy across 18 regimes (fails on ≥4 regimes). All failures involve 32B models or L40S where the formula's model-size blindness causes miscalls.
3. **h-ablation**: Micro-eval (100 requests) selects the wrong TP winner (TP=8 instead of TP=4) on at least 3 of the 4 tested high-rate regimes.
4. **h-robustness**: All 20 (regime × seed) lean bracket runs on L40S achieve within 1% of per-regime global best.

## Constraints

- RP-28 (lean bracket accuracy): must maintain within-1% accuracy across all regimes
- RP-15/RP-16 (bracket K=1): Phase 1 correctness must hold for the new model
- RP-27 (crossover formula): the formula is explicitly model-size-blind; 32B tests its limits
- RP-5 (evaluation speed): ~70-90ms per eval on H100, ~250ms on L40S
- Max 8 GPUs per node: TP × instances ≤ 8

## Prior Knowledge

Active principles directly relevant:
- **RP-27**: Crossover rate inversely proportional to bandwidth. Validated for 8B/14B but untested for 32B. This iteration tests whether the formula generalizes.
- **RP-28**: Lean bracket achieves within 1% across all tested regimes. This iteration extends to qwen3-32b and L40S.
- **RP-21**: TP winner is model-dependent on L40S. This iteration validates lean bracket handles model-dependent winners on L40S.
- **RP-15/RP-16**: Bracket K=1 is model-agnostic. This iteration adds a third model.
- **RP-9**: Phase 1 is deterministic per blis_seed. Lean bracket inherits this property.
