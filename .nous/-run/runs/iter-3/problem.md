# Problem Framing — Iteration 3: Multi-Model Generalization of Bracket K=1

## Research Question

Does bracket K=1 hierarchical search correctly identify model-dependent TP winners, and does it remain robust when TP margins approach zero?

Iterations 1-2 validated bracket K=1 on llama-3.1-8b across three hardware types (H100, A100-SXM, L40S) and five BLIS seeds. All tested regimes shared the same TP winner: TP=4/2inst. Iteration 3 tests the first regime where the TP winner is **different** — qwen3-14b on L40S favors TP=8/1inst (not TP=4/2inst), making L40S the only hardware where the TP winner is model-dependent.

Additionally, qwen's TP margins on L40S vary wildly across BLIS seeds (0.03%–7.5%), including two near-zero margins (seed=43: 0.03%, seed=45: 0.11%) that are an order of magnitude tighter than anything previously tested (prior tightest: L40S llama seed=44 at 0.85%).

**Mechanism under study:** Bracket K=1 Phase 1 evaluates all 4 TP configurations at maximum secondary parameters (mr=512, mt=8192, pf=4096). The max profile creates a compute-saturated regime where TP-level differences fully express. The search algorithm is model-agnostic — it simply picks the highest-scoring TP level. The hypothesis is that this model-agnostic Phase 1 correctly identifies different TP winners for different models without any model-specific tuning.

**Source files:**
- `cmd/root.go:947-984` — CLI flag definitions for all swept parameters
- `cmd/root.go:1761-1765` — Fitness weight parsing and ComputeFitness call
- `sim/cluster/metrics.go:418-498` — ComputeFitness, extractMetric, reference scale constants
- `sim/latency/config.go:86-101` — GetHWConfig() hardware validation
- `hardware_config.json` — GPU specs (L40S: 362 TFLOPS, 48 GiB, 0.864 TB/s; H100: 989.5 TFLOPS, 80 GiB, 3.35 TB/s)

## System Interface

- **Build:** `go build -o blis main.go`
- **CLI flags relevant to experiment:**
  - `--model` (string): Model path, e.g. `qwen/qwen3-14b`, `meta-llama/llama-3.1-8b-instruct`. Defined at `cmd/root.go:955`.
  - `--hardware` (string): GPU type. Valid: H100, A100-SXM, A100-80, L40S. Case-sensitive. Defined at `cmd/root.go:956`.
  - `--tp` (int): Tensor parallelism degree. Defined at `cmd/root.go:958`.
  - `--num-instances` (int): Deployment replicas. Defined at `cmd/root.go:959`.
  - `--max-num-running-reqs` (int): Max batch size. Defined at `cmd/root.go:962`.
  - `--max-num-scheduled-tokens` (int): Max batched tokens. Defined at `cmd/root.go:963`.
  - `--long-prefill-token-threshold` (int): Chunked prefill threshold. Defined at `cmd/root.go:964`.
  - `--scheduler` (string): Scheduling policy. Defined at `cmd/root.go:960`.
  - `--seed` (int64): BLIS workload RNG seed. Defined at `cmd/root.go:938`.
  - `--fitness-weights` (string): Weighted fitness computation. Defined at `cmd/root.go:1761`.
  - `--num-requests` (int): Total requests. Defined at `cmd/root.go:947`.
  - `--rate` (float): Arrival rate. Defined at `cmd/root.go:948`.
- **Output format:** JSON metrics between `=== Simulation Metrics ===` markers on stdout. With `--fitness-weights`, `Score: <float>` in `=== Fitness Evaluation ===` section. Parse with `r'Score:\s+([\d.]+)'`.

## Baseline Command

```bash
./blis run --model qwen/qwen3-14b --hardware L40S --latency-model trained-physics \
  --num-requests 1000 --rate 500 --seed 42 --tp 8 --num-instances 1 \
  --scheduler fcfs --max-num-running-reqs 512 --max-num-scheduled-tokens 8192 \
  --long-prefill-token-threshold 4096 --block-size-in-tokens 16 \
  --routing-policy least-loaded --admission-policy always-admit \
  --preemption-policy fcfs \
  --fitness-weights "throughput:0.4,p99_ttft:0.3,p99_e2e:0.3"
```

## Baseline Validation

Ran the baseline command. Exit code 0. Score: 0.084795.

Full Phase 1 max-profile sweep for qwen3-14b on L40S hard (seed=42):
- TP=1/8inst: 0.038284
- TP=2/4inst: 0.063921
- TP=4/2inst: 0.080647
- TP=8/1inst: 0.084795 ← winner, 5.1% margin over TP=4

Verified TP=8 wins at all BLIS seeds (42-46):
- seed=42: TP8=0.084795, TP4=0.080647, margin=5.1%
- seed=43: TP8=0.085773, TP4=0.085751, margin=0.03%
- seed=44: TP8=0.085653, TP4=0.079668, margin=7.5%
- seed=45: TP8=0.088737, TP4=0.088640, margin=0.11%
- seed=46: TP8=0.087580, TP4=0.086878, margin=0.81%

Full Phase 1 max-profile sweep for qwen3-14b on H100 hard (seed=42):
- TP=1/8inst: 0.120822
- TP=2/4inst: 0.155768
- TP=4/2inst: 0.178063 ← winner, 4.9% margin over TP=8
- TP=8/1inst: 0.169720

H100 qwen TP=4 wins at all seeds with comfortable margins:
- seed=42: TP4=0.178063, TP8=0.169720, margin=4.9%
- seed=43: TP4=0.182724, TP8=0.170657, margin=7.1%
- seed=44: TP4=0.179908, TP8=0.170572, margin=5.5%
- seed=45: TP4=0.188345, TP8=0.174972, margin=7.6%
- seed=46: TP4=0.185631, TP8=0.173857, margin=6.8%

## Experimental Conditions

All arms use the iter-8 search script (`search_blis_iter8.py`), copied from iter-7 with one code change: add L40S qwen entry to GLOBAL_BEST_TABLE.

### Code Change (applies to all arms)

**File:** `search_blis_iter8.py` (copy from iter-7/h-main.patch, rename)
**Intent:** Add `("qwen3-14b", "L40S", 500, 1000): 0.084795` to GLOBAL_BEST_TABLE.
**Rationale:** The table is keyed by (model, hardware, rate, num_requests) and used to compute the 1% evals_to_best threshold. L40S qwen entries are missing. The value 0.084795 is the max-profile TP=8/1inst score verified by probing.

### h-main: Bracket K=1 on L40S hard qwen3-14b

```bash
python3 search_blis_iter8.py --strategy adaptive-hierarchical --phase1-k 1 \
    --phase1-profiles bracket --hardware L40S \
    --budget 100 --seeds 42,43,44,45,46 --blis-seeds 42 \
    --rate 500 --num-requests 1000 --model qwen/qwen3-14b \
    --output results/h-main/l40s_qwen_bracket_k1.json
```

### h-robustness: Bracket K=1 on L40S hard qwen3-14b, multi-seed

```bash
python3 search_blis_iter8.py --strategy adaptive-hierarchical --phase1-k 1 \
    --phase1-profiles bracket --hardware L40S \
    --budget 100 --seeds 42,43,44,45,46 --blis-seeds 42,43,44,45,46 \
    --rate 500 --num-requests 1000 --model qwen/qwen3-14b \
    --output results/h-robustness/l40s_qwen_seed_robustness.json
```

### h-control-negative: Bracket K=1 on H100 hard qwen3-14b

```bash
python3 search_blis_iter8.py --strategy adaptive-hierarchical --phase1-k 1 \
    --phase1-profiles bracket --hardware H100 \
    --budget 100 --seeds 42,43,44,45,46 --blis-seeds 42 \
    --rate 500 --num-requests 1000 --model qwen/qwen3-14b \
    --output results/h-control-negative/h100_qwen_bracket_k1.json
```

### h-ablation: Flat TPE on L40S hard qwen3-14b

```bash
python3 search_blis_iter8.py --strategy tpe --hardware L40S \
    --budget 100 --seeds 42,43,44,45,46 --blis-seeds 42 \
    --rate 500 --num-requests 1000 --model qwen/qwen3-14b \
    --output results/h-ablation/l40s_qwen_flat_tpe.json
```

## Success Criteria

1. **h-main Phase 1 correctness**: 5/5 search seeds identify TP=8/1inst as the Phase 1 winner on L40S qwen.
2. **h-main convergence**: evals_to_best=3 on all 5 search seeds (Phase 1 max-profile TP=8 score 0.084795 exceeds 1% threshold 0.083947).
3. **h-robustness Phase 1 correctness**: 25/25 (blis_seed × search_seed) combinations identify TP=8/1inst. Includes near-zero margin seeds (seed=43: 0.03%, seed=45: 0.11%).
4. **h-control-negative reversal**: 5/5 seeds identify TP=4/2inst on H100 (NOT TP=8). The model-dependent TP=8 effect observed on L40S vanishes on H100.
5. **h-ablation comparison**: Flat TPE on L40S qwen has higher median evals_to_best than bracket K=1.

## Constraints

- Budget=100 evaluations per search seed (campaign standard).
- BLIS evaluations ~70-90ms on H100, ~250ms on L40S per RP-5.
- Total wall time estimate: h-main ~2min (5 runs × 25s), h-robustness ~10min (25 runs × 25s), h-control-negative ~45s (5 runs × 9s), h-ablation ~2min (5 runs × 25s).
- All experiments use fitness weights `throughput:0.4,p99_ttft:0.3,p99_e2e:0.3`.

## Prior Knowledge

- **RP-15/RP-16**: Bracket K=1 achieves 5/5 Phase 1 correctness on all tested regimes (H100, A100, L40S for llama). evals_to_best=3.
- **RP-9**: Phase 1 is deterministic per blis_seed. All search seeds produce identical Phase 1 results.
- **RP-19**: At 2.1% margin (L40S llama), flat TPE fails 4/5 seeds within budget=100. At 7.6% margin (H100 llama), TPE median=26.
- **RP-20**: Bracket K=1 robust to BLIS seed variation on H100 llama (25/25 correct).
- Handoff: L40S TP winner is model-dependent (llama→TP=4, qwen→TP=8). This is unique to L40S due to its low bandwidth (0.864 TB/s). qwen3-14b on H100 → TP=4 (same as llama).
