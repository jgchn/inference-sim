# Problem Framing — Iteration 2: L40S Generalization and BLIS Seed Robustness

## Research Question

Does bracket K=1 hierarchical search generalize to L40S hardware (48 GiB, 362 TFLOPS, 0.864 TB/s) — the tightest TP margin observed (2.1%) — and remain robust across different BLIS seed values (different workload realizations)?

Bracket K=1 was validated on H100 and A100-SXM in iter-6, achieving evals_to_best=3 deterministically across 5 search seeds. Two untested axes remain: (1) hardware generalization to L40S, whose low memory bandwidth creates qualitatively different TP tradeoffs, and (2) workload robustness across BLIS seeds, which changes the specific request mix while preserving distributional properties.

Key source files:
- `sim/latency/config.go:86-101` — `GetHWConfig()` hardware validation
- `hardware_config.json` — GPU specs: L40S (362 TFLOPS BF16, 48 GiB, 0.864 TB/s)
- `sim/cluster/metrics.go:418-498` — Fitness computation and valid keys
- `cmd/root.go:938` — `--seed` flag definition (controls workload RNG)
- `cmd/root.go:947-984` — Swept parameter flag definitions

## System Interface

- **Build:** `go build -o blis main.go` (binary already exists)
- **CLI flags relevant to experiment:**
  - `--hardware L40S` — GPU type (`cmd/root.go:956`, validated by `sim/latency/config.go:86-101`)
  - `--seed <int>` — Workload generation RNG seed (`cmd/root.go:938`). Controls request arrival times, prompt/output lengths
  - `--fitness-weights "throughput:0.4,p99_ttft:0.3,p99_e2e:0.3"` — Composite fitness score (`cmd/root.go:1761-1765`, `sim/cluster/metrics.go:428-498`)
  - `--tp`, `--num-instances`, `--max-num-running-reqs`, `--max-num-scheduled-tokens`, `--long-prefill-token-threshold` — Search space knobs (`cmd/root.go:947-984`)
- **Output format:** JSON metrics between `=== Simulation Metrics ===` markers. Fitness score as `Score: <float>` in `=== Fitness Evaluation ===` section. Parse with `r'Score:\s+([\d.]+)'`.

## Baseline Command

```bash
./blis run --model meta-llama/llama-3.1-8b-instruct --hardware L40S --latency-model trained-physics \
  --num-requests 1000 --rate 500 --seed 42 --tp 4 --num-instances 2 \
  --scheduler fcfs --max-num-running-reqs 512 --max-num-scheduled-tokens 8192 \
  --long-prefill-token-threshold 4096 --block-size-in-tokens 16 \
  --routing-policy least-loaded --admission-policy always-admit \
  --preemption-policy fcfs \
  --fitness-weights "throughput:0.4,p99_ttft:0.3,p99_e2e:0.3"
```

## Baseline Validation

Ran the baseline command. Exit code 0. Output produced: `Score: 0.109668`.

L40S max-profile Phase 1 scores (seed=42):
- TP=1/8inst: 0.067260
- TP=2/4inst: 0.089827
- TP=4/2inst: 0.109668 (winner, 2.1% margin over TP=8)
- TP=8/1inst: 0.107420

L40S min-profile Phase 1 scores (seed=42):
- TP=1/8inst: 0.042847
- TP=2/4inst: 0.047530 (winner — different from H100's TP=1/8inst)
- TP=4/2inst: 0.046768
- TP=8/1inst: 0.042522

H100 max-profile TP=4 vs TP=8 across blis_seeds:
- seed=42: TP=4=0.208909, TP=8=0.194108, margin=7.6%
- seed=43: TP=4=0.212830, TP=8=0.195067, margin=9.1%
- seed=44: TP=4=0.210067, TP=8=0.195159, margin=7.6%
- seed=45: TP=4=0.218383, TP=8=0.199351, margin=9.5%
- seed=46: TP=4=0.216127, TP=8=0.198095, margin=9.1%

L40S max-profile TP=4 vs TP=8 across blis_seeds:
- seed=42: TP=4=0.109668, TP=8=0.107420, margin=2.1%
- seed=43: TP=4=0.111504, TP=8=0.108800, margin=2.5%
- seed=44: TP=4=0.109492, TP=8=0.108565, margin=0.85%
- seed=45: TP=4=0.115873, TP=8=0.112039, margin=3.4%
- seed=46: TP=4=0.114381, TP=8=0.110502, margin=3.5%

## Experimental Conditions

### h-main: Bracket K=1 on L40S hard llama

Run bracket K=1 adaptive-hierarchical search on L40S with hard regime (1000 req, rate=500, llama-3.1-8b). 5 search seeds (42-46), blis_seed=42, budget=100.

```bash
python3 search_blis_iter7.py --strategy adaptive-hierarchical --phase1-k 1 \
    --phase1-profiles bracket --hardware L40S \
    --budget 100 --seeds 42,43,44,45,46 --blis-seeds 42 \
    --rate 500 --num-requests 1000 --model meta-llama/llama-3.1-8b-instruct \
    --output results/h-main/l40s_bracket_k1.json
```

### h-robustness: Bracket K=1 on H100 with blis_seeds 42-46

Run bracket K=1 on H100 hard llama across 5 different BLIS seeds (42-46). 5 search seeds (42-46) per blis_seed, budget=100.

```bash
python3 search_blis_iter7.py --strategy adaptive-hierarchical --phase1-k 1 \
    --phase1-profiles bracket --hardware H100 \
    --budget 100 --seeds 42,43,44,45,46 --blis-seeds 42,43,44,45,46 \
    --rate 500 --num-requests 1000 --model meta-llama/llama-3.1-8b-instruct \
    --output results/h-robustness/h100_seed_robustness.json
```

### h-control-negative: Bracket-min-only on L40S

Run bracket-min-only (K=1, min profile) on L40S hard llama. 5 search seeds, blis_seed=42, budget=100.

```bash
python3 search_blis_iter7.py --strategy adaptive-hierarchical --phase1-k 1 \
    --phase1-profiles bracket-min-only --hardware L40S \
    --budget 100 --seeds 42,43,44,45,46 --blis-seeds 42 \
    --rate 500 --num-requests 1000 --model meta-llama/llama-3.1-8b-instruct \
    --output results/h-control-negative/l40s_min_only.json
```

### h-ablation: Flat TPE on L40S (comparison baseline)

Run flat TPE on L40S hard llama for convergence speed comparison. 5 search seeds, blis_seed=42, budget=100.

```bash
python3 search_blis_iter7.py --strategy tpe --hardware L40S \
    --budget 100 --seeds 42,43,44,45,46 --blis-seeds 42 \
    --rate 500 --num-requests 1000 --model meta-llama/llama-3.1-8b-instruct \
    --output results/h-ablation/l40s_flat_tpe.json
```

## Success Criteria

1. **Phase 1 correctness on L40S**: Bracket K=1 selects TP=4/2inst on all 5 search seeds (h-main).
2. **Phase 1 correctness across blis_seeds**: Bracket K=1 selects TP=4/2inst on all 5 blis_seeds x 5 search seeds on H100 (h-robustness).
3. **evals_to_best=3 on L40S**: Bracket K=1 achieves evals_to_best=3 on L40S, matching H100/A100 performance.
4. **evals_to_best=3 across blis_seeds**: Bracket K=1 achieves evals_to_best=3 at each blis_seed on H100.
5. **Negative control fails**: Bracket-min-only on L40S selects wrong TP (TP=2/4inst, not TP=4/2inst).
6. **Bracket faster than flat TPE**: Bracket K=1 achieves lower median evals_to_best than flat TPE on L40S.

## Constraints

- Budget=100 per search seed per blis_seed.
- L40S evaluations take ~250ms per run. H100 evaluations take ~80ms per run.
- Total wall time estimate: h-main (~2.5 min), h-robustness (~12.5 min), h-control-negative (~2.5 min), h-ablation (~2.5 min). Total: ~20 minutes.
- Python dependencies: numpy, optuna, scipy.

## Prior Knowledge

Active principles that apply:
- **RP-9**: Phase 1 is deterministic per blis_seed. All search seeds produce identical Phase 1 results per blis_seed.
- **RP-15**: Bracket K=1 max-first ordering achieves evals_to_best=3 (validated on H100, A100).
- **RP-16**: K=1 (max-only) is sufficient for Phase 1. Min profile is redundant.
- **RP-17**: Min profile selects wrong TP winner due to capacity-limited regime.
- **RP-5**: BLIS evaluations complete in ~70-90ms/run on H100, ~250ms/run on L40S.
- **RP-10**: Flat TPE finds global best within median 26 evals on H100 hard llama.
