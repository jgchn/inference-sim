# Problem Framing — Iteration 9: 70B Model Generalization and Model-Aware TP Prediction

## Research Question

Does the lean bracket search algorithm (TP={4,8}, max-profile, budget=2, no Phase 2) generalize to a 4th model family and significantly larger parameter count (llama-3.1-70b, 4.4× the 14B models)? And can a model-size-aware crossover formula (load_index threshold) eliminate the remaining TP prediction failures of the bandwidth-only formula (400/BW)?

**Key code references:**
- `sim/latency/roofline.go:291-351` — rooflineStepTime: physics behind TP-dependent latency (weight transfer time scales with model size and inversely with TP)
- `sim/latency/trained_physics_model.go:263-284` — TP communication overhead formula
- `model_configs/llama-3.1-70b-instruct/config.json` — 70B architecture: 80 layers, hidden=8192, intermediate=28672, 64 attn heads, 8 KV heads
- `hardware_config.json` — GPU specs: H100 (3.35 TB/s), A100-SXM (2.039 TB/s), L40S (0.864 TB/s)

## System Interface

- **Build:** `go build -o blis main.go` (binary already exists)
- **CLI flags relevant to experiment:**
  - `--model meta-llama/llama-3.1-70b-instruct` — 70B model path (`cmd/root.go:947`)
  - `--hardware {H100,A100-SXM,L40S}` — GPU type (`cmd/root.go:956`, validated in `sim/latency/config.go:86-101`)
  - `--tp {4,8}` — tensor parallelism (`cmd/root.go:949`)
  - `--num-instances {1,2}` — deployment replicas (`cmd/root.go:950`)
  - `--num-requests 1000` — simulation horizon (`cmd/root.go:939`)
  - `--rate {100,500}` — arrival rate (`cmd/root.go:940`)
  - `--seed 42` — workload RNG seed (`cmd/root.go:938`)
  - `--fitness-weights "throughput:0.4,p99_ttft:0.3,p99_e2e:0.3"` — composite fitness (`cmd/root.go:1761-1765`)
  - Secondary params: `--max-num-running-reqs`, `--max-num-scheduled-tokens`, `--long-prefill-token-threshold`, `--block-size-in-tokens`, `--scheduler`, `--routing-policy`, `--admission-policy`, `--preemption-policy`
- **Output format:** JSON metrics between `=== Simulation Metrics ===` markers on stdout. With `--fitness-weights`, `Score: <float>` follows in `=== Fitness Evaluation ===` section. Parse with `r'Score:\s+([\d.]+)'`.
- **Multi-instance output:** When `--num-instances > 1`, parse the block with `"instance_id": "cluster"`.

## Baseline Command

```bash
./blis run --model meta-llama/llama-3.1-70b-instruct --hardware H100 --latency-model trained-physics \
  --num-requests 1000 --rate 500 --seed 42 --tp 8 --num-instances 1 \
  --scheduler fcfs --max-num-running-reqs 512 --max-num-scheduled-tokens 8192 \
  --long-prefill-token-threshold 4096 --block-size-in-tokens 16 \
  --routing-policy least-loaded --admission-policy always-admit \
  --preemption-policy fcfs \
  --fitness-weights "throughput:0.4,p99_ttft:0.3,p99_e2e:0.3"
```

## Baseline Validation

Ran the baseline command and observed:
- **Exit code:** 0
- **Score:** 0.091801
- **TP=4/2inst comparison:** Score=0.083221 (TP=8 wins by 10.3%)
- **Wall time:** ~0.267s per evaluation on H100

Multi-seed verification (blis_seeds 42-46) on H100 rate=500:
- TP=8 wins on ALL 5 seeds with margins 2.5%-10.3%
- Scores range: TP=8 from 0.091801 to 0.095769

Additional validation:
- **TP=1 is infeasible:** Fatal error "model overhead (137.07 GiB) exceeds available GPU memory (72.00 GiB)"
- **TP=2 is catastrophically bad:** Score=0.012899 (6.9× worse than TP=8)
- **70B TP=8 wins at ALL rates (100-2000) on H100** with 9-10% margin
- **70B TP=8 wins on ALL hardware** at both rates: H100 (9-10%), A100-SXM (7-13%), L40S (18-20%)

## Experimental Conditions

### h-main: Lean bracket on 70B (6 regimes × 5 search seeds)

Run lean bracket (TP={4,8}, max-profile, skip-phase2, budget=2) on llama-3.1-70b across all hardware/rate combinations.

| # | Hardware | Rate | Expected Winner | GLOBAL_BEST |
|---|----------|------|----------------|-------------|
| 1 | H100 | 100 | TP=8/1inst | 0.088793 |
| 2 | H100 | 500 | TP=8/1inst | 0.091801 |
| 3 | A100-SXM | 100 | TP=8/1inst | 0.063273 |
| 4 | A100-SXM | 500 | TP=8/1inst | 0.063991 |
| 5 | L40S | 100 | TP=8/1inst | 0.034041 |
| 6 | L40S | 500 | TP=8/1inst | 0.034137 |

Commands (per regime):
```bash
python3 search_blis_iter9.py --strategy adaptive-hierarchical --phase1-k 1 \
    --phase1-profiles bracket --tp-candidates lean --skip-phase2 \
    --hardware $HW --budget 2 --seeds 42,43,44,45,46 \
    --rate $RATE --num-requests 1000 --model meta-llama/llama-3.1-70b-instruct \
    --output results/h-main/$HW-rate$RATE.json
```

### h-control-negative: Model-aware formula vs old formula (24 regimes)

Test two formula strategies across the full 4-model × 3-hardware × 2-rate matrix:

1. **Old formula (400/BW):** Predicts crossover_rate = 400/bandwidth. TP=4 if rate > crossover, else TP=8. Model-size-blind.
2. **Model-aware formula:** Computes load_index = model_params_B / bandwidth_TB_s. If load_index > 12, predict TP=8 always (no crossover regime). Otherwise, fall back to 400/BW.

For the 18 regimes already in GLOBAL_BEST_TABLE, use existing scores. For the 6 new 70B regimes, run both formulas.

Old formula predictions for 70B at rate=500: TP=4 on all 3 hardware → ALL WRONG (actual: TP=8).
Model-aware predictions for 70B: load_index > 20 on all hardware → TP=8 always → ALL CORRECT.

Commands:
```bash
# Old formula on 70B (6 regimes)
for hw in H100 A100-SXM L40S; do
  for rate in 100 500; do
    python3 search_blis_iter9.py --strategy formula-predict \
        --hardware $hw --rate $rate --num-requests 1000 \
        --model meta-llama/llama-3.1-70b-instruct \
        --output results/h-control-negative/old-$hw-rate$rate.json
  done
done

# Model-aware formula on 70B (6 regimes)
for hw in H100 A100-SXM L40S; do
  for rate in 100 500; do
    python3 search_blis_iter9.py --strategy model-aware-predict \
        --hardware $hw --rate $rate --num-requests 1000 \
        --model meta-llama/llama-3.1-70b-instruct \
        --output results/h-control-negative/aware-$hw-rate$rate.json
  done
done
```

### h-ablation: Default-profile lean bracket for 70B (6 regimes × 1 seed)

Test whether the max-profile requirement is necessary for 70B. For smaller models, default-profile (mr=256, mt=4096, pf=1024) masks the rate-dependent TP effect — but for 70B, the per-step latency gap is so large that TP=8 should win even with restricted batch sizes.

Commands:
```bash
for hw in H100 A100-SXM L40S; do
  for rate in 100 500; do
    python3 search_blis_iter9.py --strategy adaptive-hierarchical --phase1-k 1 \
        --phase1-profiles strategic --tp-candidates lean --skip-phase2 \
        --hardware $hw --budget 2 --seeds 42 \
        --rate $rate --num-requests 1000 --model meta-llama/llama-3.1-70b-instruct \
        --output results/h-ablation/$hw-rate$rate.json
  done
done
```

### h-robustness: Multi-blis-seed lean bracket on 70B (2 regimes × 5 blis seeds × 5 search seeds)

Test lean bracket robustness across workload seeds on H100 and L40S (highest and lowest bandwidth).

Commands:
```bash
for blis_seed in 42 43 44 45 46; do
  for hw in H100 L40S; do
    python3 search_blis_iter9.py --strategy adaptive-hierarchical --phase1-k 1 \
        --phase1-profiles bracket --tp-candidates lean --skip-phase2 \
        --hardware $hw --budget 2 --seeds 42,43,44,45,46 --blis-seed $blis_seed \
        --rate 500 --num-requests 1000 --model meta-llama/llama-3.1-70b-instruct \
        --output results/h-robustness/$hw-blis$blis_seed.json
  done
done
```

## Success Criteria

1. **h-main:** Lean bracket achieves within 1% of global best on all 6 regimes (30/30 runs). TP=8 wins on all. evals_to_best=2 everywhere.
2. **h-control-negative:** Model-aware formula achieves 24/24 (100%) on the {100, 500} test grid. Old formula achieves ≤18/24 (≤75%). The 6-point gap is entirely from regimes where load_index > 12.
3. **h-ablation:** Default-profile lean bracket selects TP=8 on all 6 regimes (6/6 correct). Scores lower than max-profile but TP winner unchanged.
4. **h-robustness:** TP=8 wins on all 50 combinations (2 regimes × 5 blis seeds × 5 search seeds). All 5 search seeds identical per blis_seed (deterministic).

## Constraints

- TP × instances ≤ 8 (single node, 8 GPU constraint)
- TP=1 is infeasible for 70B (OOM). TP=2 is technically feasible but catastrophically bad (score 0.012899 vs TP=8's 0.091801).
- Minimum evaluation horizon: 1000 requests at rate≥500 (RP-31)
- BLIS seed controls workload, search seed controls algorithm RNG (lean bracket has no RNG, so search seeds are redundant)
- Hardware names are case-sensitive: H100, A100-SXM, L40S
- Must run from project root directory
- Model path: `meta-llama/llama-3.1-70b-instruct`

## Prior Knowledge

Active principles that apply:
- **RP-1:** TP is the dominant performance knob (~87% of fitness variance)
- **RP-3:** Large batch/token budgets (max-profile) improve fitness alongside optimal TP
- **RP-9:** Phase 1 TP identification is deterministic given a fixed BLIS seed
- **RP-15/16:** Bracket K=1 (max-profile) achieves Phase 1 correctness across all tested regimes
- **RP-20:** Lean bracket is robust to BLIS workload variation across seeds
- **RP-27:** 400/BW formula works for 8-14B models but fails for 32B. Expected to fail for 70B.
- **RP-28:** Lean bracket validated across 3 models and 3 hardware platforms (45/45 runs)
- **RP-29:** Min-profile inverts TP ranking — max-profile is the mechanism enabling lean bracket correctness
- **RP-31:** Minimum 1000 requests for reliable TP winner identification at high rates
