# Handoff — Iteration 2: L40S Generalization and BLIS Seed Robustness

## Goal

Test whether bracket K=1 hierarchical search generalizes to L40S hardware (48 GiB, 362 TFLOPS, 0.864 TB/s — tightest TP margin at 2.1%) and remains robust across BLIS seed variation (seeds 42-46 on H100 hard llama). Run bracket K=1 on L40S hard llama, bracket K=1 on H100 with blis_seeds 42-46, flat TPE on L40S for comparison, and bracket-min-only on L40S as negative control.

## Key Discoveries

1. **L40S has the tightest TP=4 vs TP=8 margin observed**: On L40S hard llama with max profile (mr=512), TP=4/2inst scores 0.1097 vs TP=8/1inst 0.1074 — a 2.1% margin. Compare: H100 hard llama has 7.6-9.5% margin, A100 hard qwen has 1.1% margin. L40S's low bandwidth (0.864 TB/s) makes TP=8/1inst less dominant because weight-loading overhead is proportionally higher.

2. **TP=4 wins on L40S across all tested blis_seeds (42-46)**: Margins range from 0.85% (seed=44) to 3.5% (seed=46). Seed=44 has the tightest margin but TP=4 still wins. The TP ranking is consistent despite workload variation.

3. **TP=4 wins on H100 across all tested blis_seeds (42-46)**: Margins range from 7.6% (seeds 42,44) to 9.5% (seed=45). H100's wide margins make seed variation irrelevant for Phase 1 correctness.

4. **Min profile on L40S selects TP=2/4inst (not TP=1/8inst like H100)**: At mr=32, L40S Phase 1 scores are TP=1=0.0428, TP=2=0.0475, TP=4=0.0468, TP=8=0.0425. TP=2/4inst wins because 4 instances with 2-way TP balances capacity and speed. The wrong-winner mechanism is hardware-specific.

5. **Prefill threshold is a no-op on L40S**: All values produce identical scores at fixed (mr, mt). Same behavior as A100.

6. **L40S evaluations take ~250ms**: Budget=100 at ~250ms/eval → ~25s per search seed. The h-robustness arm (25 runs) takes ~12.5 minutes total.

7. **All Phase 1 TP=4 scores exceed the 1% threshold for global_best on both hardware types**: H100 TP=4 scores (0.2089-0.2184) all exceed 0.99*0.209195=0.207103. L40S TP=4 scores (0.1095-0.1159) all exceed 0.99*0.109668=0.108571. This means evals_to_best=3 is achievable at all tested seeds.

## System Interface

- **Build:** `go build -o blis main.go` (binary already exists)
- **Run baseline (L40S hard llama):**
  ```bash
  ./blis run --model meta-llama/llama-3.1-8b-instruct --hardware L40S --latency-model trained-physics \
    --num-requests 1000 --rate 500 --seed 42 --tp 4 --num-instances 2 \
    --scheduler fcfs --max-num-running-reqs 512 --max-num-scheduled-tokens 8192 \
    --long-prefill-token-threshold 4096 --block-size-in-tokens 16 \
    --routing-policy least-loaded --admission-policy always-admit \
    --preemption-policy fcfs \
    --fitness-weights "throughput:0.4,p99_ttft:0.3,p99_e2e:0.3"
  ```
- **Run baseline (H100 hard llama):**
  ```bash
  ./blis run --model meta-llama/llama-3.1-8b-instruct --hardware H100 --latency-model trained-physics \
    --num-requests 1000 --rate 500 --seed 42 --tp 4 --num-instances 2 \
    --scheduler fcfs --max-num-running-reqs 512 --max-num-scheduled-tokens 8192 \
    --long-prefill-token-threshold 4096 --block-size-in-tokens 16 \
    --routing-policy least-loaded --admission-policy always-admit \
    --preemption-policy fcfs \
    --fitness-weights "throughput:0.4,p99_ttft:0.3,p99_e2e:0.3"
  ```
- **Output format:** JSON metrics between `=== Simulation Metrics ===` markers. With `--fitness-weights`, `Score: <float>` follows in `=== Fitness Evaluation ===` section.
- **Baseline results:** L40S hard llama TP=4/2inst mr=512 Score=0.109668. H100 hard llama TP=4/2inst mr=512 Score=0.208909.

## Code Map

- `cmd/root.go:947-984` — All swept parameter flag definitions. Check here if a flag name or default is wrong.
- `cmd/root.go:956` — `--hardware` flag definition. Valid values: H100, A100-SXM, A100-80, L40S.
- `cmd/root.go:938` — `--seed` flag definition (Int64, default 42). Controls workload generation RNG.
- `cmd/root.go:1761-1765` — Fitness weight parsing and ComputeFitness call. Check here if fitness isn't being computed.
- `sim/latency/config.go:86-101` — `GetHWConfig()` validates hardware names. Check here if a hardware name is rejected.
- `hardware_config.json` — GPU specs: H100 (989.5 TFLOPS, 80 GiB, 3.35 TB/s), A100-SXM (312 TFLOPS, 80 GiB, 2.039 TB/s), L40S (362 TFLOPS, 48 GiB, 0.864 TB/s).
- `sim/cluster/metrics.go:418-498` — `validFitnessKeysList()`, `ComputeFitness()`, `extractMetric()`. Check here if fitness scores look wrong.
- `sim/cluster/metrics.go:420-426` — The 8 valid fitness keys.
- `sim/cluster/metrics.go:428-435` — Reference scale constants for normalization.
- `sim/metrics_utils.go:57-87` — MetricsOutput struct: all JSON fields in the metrics output.

## Code Targets

The experiment modifies the Python search script from iter-6:

- **`search_blis_iter7.py`** (copy from iter-6 h-main.patch, extract the search_blis_iter6.py file):
  1. **Add L40S entries to GLOBAL_BEST_TABLE**: `("llama-3.1-8b", "L40S", 500, 1000): 0.109668` and `("llama-3.1-8b", "L40S", 200, 200): 0.078308`. These are the max-profile scores confirmed by probing.
  2. **Replace `--blis-seed` with `--blis-seeds` flag**: Comma-separated list of BLIS seeds (default "42"). Parse into list of integers. The main loop becomes: for each strategy, for each blis_seed, for each search_seed -> run. Each run result includes `"blis_seed"` field.
  3. **Update summary output**: Add BlisSeed column to the summary table. Include blis_seed in the per-run print output.
  4. **All existing iter-6 features preserved**: `--hardware`, `--phase1-profiles bracket/bracket-min-only`, bracket profile generation, etc.

## What I Tried That Didn't Work

- **`--hardware h100_sxm`**: Fatal error. Correct name is `H100` (case-sensitive, no suffix). [Carried from iter-1]
- **`--hardware A100`**: Fatal error. Must be `A100-SXM` or `A100-80`. [Carried from iter-5]
- **Parsing fitness from `--metrics-path` JSON**: The JSON file does not contain the fitness score. Must parse stdout. [Carried from iter-1]
- **Looking for Pareto tradeoffs across TP levels at 200 requests**: TP=8 dominates all objectives simultaneously at low request counts on both H100 and A100. No tradeoffs exist. [Carried from iter-2]
- **Exhaustive TP sweep with best secondary params assuming TP=8 always wins**: At 1000 requests, TP=4/2inst is the actual best for both models on both H100 and A100. [Carried from iter-3]
- **Random K=2 profiles for Phase 1 on tight-margin regimes**: 3/5 on H100 hard, 2/5 on A100 easy. Fails because random doesn't guarantee mr>=256. [From iter-5]
- **Assuming TP=8 easy dominance is unconditional**: Requires max_running >= 256. With mr<256, TP=4/2inst wins even in easy regimes. [From iter-5]

## What I Excluded and Why

- **qwen3-14b on L40S hard**: TP=8/1inst wins (0.0848 vs 0.0806 for TP=4). Testing this would validate bracket correctly picks TP=8 (different winner from llama on same hardware), but adds experiment scope. Reserve for iter-3 alongside multi-model generalization.
- **L40S easy regime**: TP=8 wins by 26.6% — too easy to be interesting. Bracket will trivially succeed.
- **Combined L40S + blis_seed variation**: Testing both generalization axes simultaneously creates confounds. Each axis is tested independently first (L40S at seed=42, H100 at seeds 42-46).
- **Rate as a continuous parameter**: Would map the TP dominance transition point. Orthogonal to the generalization question. [Carried from iter-3]
- **Early stopping in Phase 1**: Bracket K=1 already uses only 4 Phase 1 evals. Cannot reduce further. [Carried from iter-6]
- **Budget sensitivity (budget=20, 40)**: Bracket's advantage manifests in Phase 1 (evals 1-4), so smaller budgets show proportionally larger relative improvement. The checkpoint analysis captures this. [Carried from iter-5]

## Evolution of Thinking

Iterations 1-6 progressively refined the search algorithm from flat TPE -> hierarchical -> adaptive -> bracket. The algorithm is well-validated on H100 and A100-SXM with a single workload (blis_seed=42). The natural question is: "Is this algorithm genuinely general, or is it overfit to the tested conditions?"

Two axes of generalization were untested: (1) hardware (only H100 and A100 tested) and (2) workload (only seed=42 tested). L40S is the most interesting hardware to test because it has fundamentally different specs (48 GiB vs 80 GiB, 0.864 TB/s vs 3.35 TB/s), creating the tightest TP margin observed (2.1%). BLIS seed variation is the most practical robustness test — users won't always use seed=42.

A key finding during probing: the TP winner on L40S is model-dependent (llama->TP=4, qwen->TP=8), unlike H100/A100 where both models favor TP=4 in the hard regime. This suggests L40S's extreme bandwidth limitation creates different compute-bandwidth tradeoff regimes for different model sizes (8B vs 14B parameters). The bracket algorithm should handle this automatically (it picks the max scorer in Phase 1), but it's a regime not tested before.

The min-profile negative control also reveals a hardware-specific failure mode: on L40S, the wrong winner is TP=2/4inst (not TP=1/8inst as on H100). This confirms that the min profile creates qualitatively different regimes on different hardware, further validating that the max profile is the universal discriminator.

Probing across blis_seeds 42-46 on L40S revealed seed=44 has only 0.85% margin (TP=4=0.1095 vs TP=8=0.1086). This is the tightest margin ever observed in the campaign. Bracket K=1 should still handle it (Phase 1 is deterministic per seed), but it's the closest call yet.

## Current Status

- **Validated:** L40S hardware is supported and produces sensible results. TP=4/2inst wins L40S hard llama at all tested blis_seeds (42-46). TP=4/2inst wins H100 hard llama at all tested blis_seeds (42-46). L40S min-profile picks TP=2/4inst (wrong winner, different from H100). All Phase 1 TP=4 max-profile scores exceed the 1% threshold at all seeds on both hardware types.
- **Uncertain:** (1) Whether flat TPE on L40S converges slower than on H100 due to tighter margins (expected but not measured). (2) The exact global best for L40S — 0.109668 is the max-profile Phase 1 score, Phase 2 TPE may find higher. (3) Whether qwen3-14b on L40S with bracket K=1 correctly picks TP=8 (probed manually but not run through full search). (4) Whether combining L40S + multi-seed (the cross-product) introduces new failure modes beyond the individual axes.
- **Suggested next:** (1) Test qwen3-14b on L40S with bracket K=1 — the TP=8 winner creates a different Phase 1 outcome to validate. (2) Combine L40S + blis_seed variation once both axes are independently validated. (3) Map the rate at which TP dominance transitions from TP=8 to TP=4 on each hardware. (4) Test L40S at blis_seed=44 (0.85% margin) through the full search to see if it's robust. (5) Package the search algorithm as a production tool with bracket K=1 as default.

## Warnings & Constraints

- **Stderr contains warnings**: BLIS prints timing and config warnings to stderr. Always use `2>/dev/null` or capture stderr separately when parsing stdout. [Carried from iter-2]
- **Seed affects workload, not config evaluation**: `--seed 42` controls random request generation. For the search algorithm's own randomness, use a separate seed. [Carried from iter-2]
- **TP=0 is invalid**: Always set `--tp` explicitly to avoid auto-detection warnings. [Carried from iter-2]
- **Fitness score format on stdout**: `Score: 0.019249` — parse with regex `r'Score:\s+([\d.]+)'`. [Carried from iter-2]
- **Multi-instance JSON output**: When `--num-instances > 1`, stdout contains multiple JSON blocks. Always parse the block with `"instance_id": "cluster"` for aggregate metrics. [Carried from iter-2]
- **Hardware name is case-sensitive**: `H100` not `h100`, `A100-SXM` not `a100-sxm`, `L40S` not `l40s`. [Carried from iter-5]
- **Global best is now (model, hardware, rate, num_requests)-specific**: L40S entries added: llama hard=0.109668, llama easy=0.078308. [New in iter-2]
- **Bracket profiles are deterministic**: They don't depend on search_seed (no RNG involved). All 5 search seeds produce identical Phase 1 results per blis_seed. [Carried from iter-6]
- **Profile ordering affects evals_to_best**: Max-first is optimal. [Carried from iter-6]
- **Model name format**: Use full model paths: `qwen/qwen3-14b`, `meta-llama/llama-3.1-8b-instruct`. [Carried from iter-3]
- **A100/L40S prefill_threshold is a no-op for fitness**: All values produce identical scores at fixed (max_running, max_tokens). [Updated in iter-2 to include L40S]
- **vllm-version auto-detection**: Warnings about "Finding default values of vLLM version" appear on stderr. Harmless but noisy. [Carried from iter-2]
- **L40S memory is 48 GiB (not 80 GiB)**: TP=8/1inst on L40S may have different memory constraints than H100/A100. Validated that TP=8/1inst works on L40S with the full sweep. [New in iter-2]
- **L40S TP winner is model-dependent**: llama->TP=4, qwen->TP=8 on L40S hard. This is unique to L40S. [New in iter-2]
- **GLOBAL_BEST_TABLE uses seed=42 values for all seeds**: When running with --blis-seeds, the threshold for evals_to_best is based on the seed=42 global best. For seeds with higher actual global bests, the threshold is conservative (easy to reach). For seeds with lower actual global bests, it may be aggressive. All tested seeds pass. [New in iter-2]
- **L40S seed=44 has 0.85% TP margin**: The tightest margin in the campaign. TP=4=0.1095, TP=8=0.1086. Bracket Phase 1 should still correctly identify TP=4 since it's deterministic, but this is the narrowest gap tested. [New in iter-2]
