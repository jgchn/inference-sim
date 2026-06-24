# Problem Framing — Iteration 10: Workload-Shape Sensitivity of Lean Bracket

## Research Question

Does the lean bracket search algorithm (TP={4,8}, max-profile, budget=2) remain correct when the workload token distribution changes from the default (prompt mean=512, output mean=512) to named workload presets with very different prompt/output token profiles? And does the model-aware formula (RP-27 + load_index threshold) break when the prompt token distribution shifts the TP crossover?

**Code evidence for mechanism:** The TP crossover is driven by the prefill/decode balance in the latency model (`sim/latency/roofline.go:291-351`). Longer prompts increase prefill compute per request, which saturates TP=8/1inst earlier because a single instance must serialize all prefills. At TP=4/2inst, two instances can process prefills in parallel. This shifts the crossover rate lower — or eliminates it entirely for long-prompt workloads.

**Workload presets** are defined in `defaults.yaml:152-191`:
- **chatbot**: prompt=256±100, output=256±100 (short prompts, short outputs)
- **contentgen**: prompt=1024±150, output=1024±200 (medium prompts, medium outputs)
- **summarization**: prompt=4096±500, output=512±150 (long prompts, medium outputs)
- **multidoc**: prompt=10240±1200, output=1536±300 (very long prompts, long outputs)
- **distribution** (default): prompt=512±256, output=512±256

## System Interface

- **Build:** `go build -o blis main.go` (binary already exists)
- **CLI flags relevant to experiment:**
  - `--workload` (`cmd/root.go:2018`): Workload type. Values: `chatbot`, `summarization`, `contentgen`, `multidoc`, `distribution`. Default: `distribution`.
  - `--rate` (`cmd/root.go:2020`): Arrival rate in req/s.
  - `--num-requests` (`cmd/root.go:2021`): Number of requests.
  - `--tp` (`cmd/root.go:947-984`): Tensor parallelism.
  - `--num-instances`: Number of instances. TP × instances ≤ 8.
  - `--hardware` (`cmd/root.go:956`): GPU type. Values: H100, A100-SXM, L40S.
  - `--max-num-running-reqs`, `--max-num-scheduled-tokens`, `--long-prefill-token-threshold`: Batch/token budget params.
  - `--fitness-weights`: Composite fitness function. Format: `throughput:0.4,p99_ttft:0.3,p99_e2e:0.3`.
- **Output format:** JSON metrics between `=== Simulation Metrics ===` markers on stdout. Fitness `Score: <float>` in `=== Fitness Evaluation ===` section.
- **Workload preset loading:** `cmd/root.go:1315-1326` loads preset from `defaults.yaml` when `--workload` is not `distribution`.

## Baseline Command

```bash
./blis run --model qwen/qwen3-14b --hardware H100 --latency-model trained-physics \
  --num-requests 1000 --rate 500 --seed 42 --tp 4 --num-instances 2 \
  --scheduler fcfs --max-num-running-reqs 512 --max-num-scheduled-tokens 8192 \
  --long-prefill-token-threshold 4096 --block-size-in-tokens 16 \
  --routing-policy least-loaded --admission-policy always-admit \
  --preemption-policy fcfs --workload contentgen \
  --fitness-weights "throughput:0.4,p99_ttft:0.3,p99_e2e:0.3"
```

## Baseline Validation

Ran the baseline command. Exit code 0. Score: 0.104103. Wall time ~0.25s.

Also validated all 20 probe regimes (5 workloads × 2 rates × 2 TP levels on H100, plus 10 on A100-SXM). All produced valid scores. Key probe results:

**H100 qwen3-14b max-profile:**
| Workload     | Prompt Mean | rate=100 TP=4 | rate=100 TP=8 | Winner | Margin | rate=500 TP=4 | rate=500 TP=8 | Winner | Margin |
|---|---|---|---|---|---|---|---|---|---|
| chatbot      | 256   | 0.193424 | 0.200393 | TP=8 | 3.6%  | 0.268650 | 0.262441 | TP=4 | 2.4%  |
| distribution | 512   | 0.156101 | 0.164680 | TP=8 | 5.5%  | 0.178063 | 0.169720 | TP=4 | 4.9%  |
| contentgen   | 1024  | 0.099677 | 0.091319 | TP=4 | 9.1%  | 0.104103 | 0.093909 | TP=4 | 10.9% |
| summarization| 4096  | 0.085079 | 0.082694 | TP=4 | 2.9%  | 0.085323 | 0.083077 | TP=4 | 2.7%  |
| multidoc     | 10240 | 0.016893 | 0.016499 | TP=4 | 2.4%  | 0.016903 | 0.016500 | TP=4 | 2.4%  |

**A100-SXM qwen3-14b max-profile:**
| Workload     | rate=100 TP=4 | rate=100 TP=8 | Winner | Margin | rate=500 TP=4 | rate=500 TP=8 | Winner | Margin |
|---|---|---|---|---|---|---|---|---|
| chatbot      | 0.182203 | 0.193078 | TP=8 | 6.0%   | 0.233715 | 0.234138 | TP=8 | 0.18%  |
| summarization| 0.056095 | 0.054623 | TP=4 | 2.7%   | 0.055265 | 0.054632 | TP=4 | 1.2%   |
| contentgen   | 0.072420 | 0.066904 | TP=4 | 8.2%   | 0.070609 | 0.068007 | TP=4 | 3.8%   |
| multidoc     | 0.003357 | 0.007025 | TP=8 | -52%   | 0.003158 | 0.007118 | TP=8 | -56%   |

**Key finding from probes:** Prompt token distribution shifts the TP crossover. With prompt mean ≥ 1024, TP=4/2inst wins at ALL tested rates on H100. The crossover rate from RP-24 (~125 on H100) only applies to the default workload (prompt mean ~512). Multidoc on A100-SXM shows an anomalous TP=8 dominance, likely due to KV cache capacity constraints at TP=4 with very long prompts.

## Experimental Conditions

### Condition 1: h-main — Lean bracket across workload presets on H100
Run lean bracket (TP={4,8}, max-profile, budget=2) for 4 workload presets × 2 rates on H100. 8 regimes total, 5 search seeds each (all deterministic — search seeds produce identical results for lean bracket).

Commands per regime (example for chatbot rate=100):
```bash
python3 search_blis_iter10.py --strategy adaptive-hierarchical --phase1-k 1 \
    --phase1-profiles bracket --tp-candidates lean --skip-phase2 \
    --hardware H100 --budget 2 --seeds 42,43,44,45,46 \
    --rate 100 --num-requests 1000 --model qwen/qwen3-14b \
    --workload chatbot \
    --output results/h-main/h100-chatbot-rate100.json
```

Repeat for: chatbot×{100,500}, contentgen×{100,500}, summarization×{100,500}, multidoc×{100,500}.

### Condition 2: h-control-negative — Model-aware formula on workload presets
Run model-aware-predict on the same 8 H100 regimes. The formula uses load_index and crossover_rate — both are workload-blind.

```bash
python3 search_blis_iter10.py --strategy model-aware-predict \
    --hardware H100 --rate 100 --num-requests 1000 --model qwen/qwen3-14b \
    --workload contentgen \
    --output results/h-control-negative/h100-contentgen-rate100.json
```

Also run on 8 A100-SXM regimes for cross-hardware validation.

### Condition 3: h-robustness — Lean bracket on A100-SXM workload presets
Run lean bracket for 4 workload presets × 2 rates on A100-SXM. 8 regimes, 5 seeds each.

```bash
python3 search_blis_iter10.py --strategy adaptive-hierarchical --phase1-k 1 \
    --phase1-profiles bracket --tp-candidates lean --skip-phase2 \
    --hardware A100-SXM --budget 2 --seeds 42,43,44,45,46 \
    --rate 100 --num-requests 1000 --model qwen/qwen3-14b \
    --workload summarization \
    --output results/h-robustness/a100-summarization-rate100.json
```

### Code Changes (search_blis_iter10.py)

Copy from `search_blis_iter8.py` and make these changes:

1. **Add `--workload` CLI argument** (default "distribution"). Pass through to `build_blis_cmd()` which adds `--workload <value>` to the BLIS command when not "distribution".

2. **Extend GLOBAL_BEST_TABLE key** from `(model, hardware, rate, num_requests)` to `(model, hardware, rate, num_requests, workload)`. All existing entries get `workload="distribution"`. New entries from probes:
   - H100 chatbot: (qwen3-14b, H100, 100, 1000, chatbot)=0.200393, (500)=0.268650
   - H100 contentgen: 0.099677, 0.104103
   - H100 summarization: 0.085079, 0.085323
   - H100 multidoc: 0.016893, 0.016903
   - A100-SXM chatbot: 0.193078, 0.234138
   - A100-SXM contentgen: 0.072420, 0.070609
   - A100-SXM summarization: 0.056095, 0.055265
   - A100-SXM multidoc: 0.007025, 0.007118
   - Plus 70B distribution entries from iter-9

3. **Add MODEL_PARAMS dict** and `run_model_aware_predict()` function (the model-aware formula with load_index threshold from RP-27).

4. **Wire `model-aware-predict`** into argparse choices and dispatch.

## Success Criteria

1. **h-main**: Lean bracket achieves within 1% of global best on all 8 H100 workload regimes (8/8). TP winner matches probe results. evals_to_best=1 on 7 regimes (TP=4 winner), evals_to_best=2 on 1 regime (chatbot rate=100, TP=8 winner).

2. **h-control-negative**: Model-aware formula correctly predicts TP winner on ≤ 5/8 H100 workload regimes. Specifically fails on summarization, contentgen, and multidoc at rate=100 (predicts TP=8, actual TP=4). On A100-SXM, fails on ≤ 4/8 workload regimes.

3. **h-robustness**: Lean bracket achieves within 1% of global best on all 8 A100-SXM workload regimes (8/8), including the multidoc anomaly (TP=8 wins) and the near-tied chatbot rate=500 (margin 0.18%).

## Constraints

- All evals use 1000 requests (RP-31: minimum for reliable TP identification).
- Lean bracket is deterministic — all 5 search seeds produce identical results per regime (RP-9).
- Fitness weights: throughput:0.4, p99_ttft:0.3, p99_e2e:0.3 (consistent with all prior iterations).
- Must run blis from project root.
- Hardware names are case-sensitive (H100, A100-SXM).

## Prior Knowledge

This experiment builds directly on the campaign's accumulated principles:
- **RP-15/RP-16**: Bracket K=1 Phase 1 achieves correct TP identification across models/hardware.
- **RP-24/RP-25**: TP winner is rate-dependent on the default workload.
- **RP-27**: Model-aware formula (load_index + 400/BW) achieves 24/24 on the default workload test grid.
- **RP-28**: Lean bracket achieves 75/75 across 4 models, 3 hardware, 2 rates — all with default workload.

The novel element: **all prior testing used the default workload** (prompt mean=512, output mean=512). This iteration tests whether lean bracket's robustness extends to workloads with 2-20× different prompt token distributions.
