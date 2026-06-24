# Handoff — Iteration 4: A100-SXM Matrix Completion and TPE Margin Threshold Narrowing

## Goal

Complete the 2×3 (model × hardware) matrix by testing bracket K=1 on A100-SXM qwen3-14b, and narrow the flat TPE convergence threshold (currently [2.1%, 5.1%]) using A100-SXM qwen's natural seed-by-seed margin variation: seed=42 has 1.07% margin (tightest TP=4 margin in campaign), seed=46 has 4.47% margin. Run bracket K=1 single-seed + multi-seed, and flat TPE at both tight and wide margins.

## Key Discoveries

1. **A100-SXM qwen TP=4 wins at all seeds**: TP=4/2inst is the Phase 1 winner at all 5 blis_seeds (42–46). Margins: seed=42: 1.07%, seed=43: 3.62%, seed=44: 1.24%, seed=45: 4.09%, seed=46: 4.47%. This confirms A100-SXM (2.039 TB/s) follows H100 behavior (TP=4 for both models), not L40S (model-dependent).

2. **Bandwidth threshold confirmed between 0.864 and 2.039 TB/s for qwen**: The TP winner transition for qwen3-14b occurs between L40S (0.864 TB/s, TP=8) and A100-SXM (2.039 TB/s, TP=4). Both H100 and A100 have sufficient bandwidth for TP=4 to dominate.

3. **Global best for A100-SXM qwen is 0.138820 (not 0.138290)**: The prior GLOBAL_BEST_TABLE entry (0.138290) used least-loaded routing. Round-robin routing produces 0.138820 for TP=4/2inst configs (0.38% improvement). This is the first regime where Phase 2 is NOT redundant — the max-profile Phase 1 score (0.138290) is below the true global best.

4. **Seeds 42 and 44 have the tightest TP=4 margins in the campaign**: 1.07% and 1.24% respectively. These are tighter than L40S llama (2.1%, where TPE failed 4/5) and provide a new lower bound for TPE margin threshold testing.

5. **Phase 1 max-profile scores on A100-SXM qwen (seed=42)**: TP1=0.085746, TP2=0.115450, TP4=0.138290, TP8=0.136812. Monotonically increasing to TP=4, then drops at TP=8 (like H100 for both models).

6. **Routing policy matters for 2-instance configs on A100-SXM**: round-robin (0.138820) > least-loaded (0.138290) at TP=4/2inst seed=42. However, this is seed-dependent: at seed=46, least-loaded (0.147057) > round-robin (0.146765). This routing interaction is unique to multi-instance configs.

7. **Secondary params are mostly degenerate within TP=4/2inst**: Scheduler (fcfs/priority-fcfs/sjf/reverse-priority → identical at max-profile), block_size (16/32 → identical with least-loaded), prefill_threshold (0/2048/4096 → identical). Only max_running and max_tokens strongly affect fitness. This matches the L40S qwen pattern.

## System Interface

- **Build:** `go build -o blis main.go` (binary already exists in project root)
- **Run baseline (A100-SXM hard qwen):**
  ```bash
  ./blis run --model qwen/qwen3-14b --hardware A100-SXM --latency-model trained-physics \
    --num-requests 1000 --rate 500 --seed 42 --tp 4 --num-instances 2 \
    --scheduler fcfs --max-num-running-reqs 512 --max-num-scheduled-tokens 8192 \
    --long-prefill-token-threshold 4096 --block-size-in-tokens 16 \
    --routing-policy round-robin --admission-policy always-admit \
    --preemption-policy fcfs \
    --fitness-weights "throughput:0.4,p99_ttft:0.3,p99_e2e:0.3"
  ```
- **Output format:** JSON metrics between `=== Simulation Metrics ===` markers. With `--fitness-weights`, `Score: <float>` follows in `=== Fitness Evaluation ===` section.
- **Baseline result:** A100-SXM qwen TP=4/2inst Score=0.138820.

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

The experiment modifies the Python search script from iter-3's h-main.patch:

- **`search_blis_iter9.py`** (copy from iter-3 h-main.patch, which contains the iter-8 version):
  1. **Update A100-SXM qwen GLOBAL_BEST_TABLE entry**: Change `("qwen3-14b", "A100-SXM", 500, 1000): 0.138290` to `0.138820`. This is the only code change needed. The stale value (0.138290) was the max-profile with least-loaded routing; the true global best (0.138820) uses round-robin routing.
  2. The script already supports `--hardware`, `--model`, `--blis-seeds`, bracket K=1, and flat TPE strategies from iter-3.

## What I Tried That Didn't Work

- **`--hardware h100_sxm`**: Fatal error. Correct name is `H100` (case-sensitive, no suffix). [Carried from iter-1]
- **`--hardware A100`**: Fatal error. Must be `A100-SXM` or `A100-80`. [Carried from iter-5]
- **Parsing fitness from `--metrics-path` JSON**: The JSON file does not contain the fitness score. Must parse stdout. [Carried from iter-1]
- **Looking for Pareto tradeoffs across TP levels at 200 requests**: TP=8 dominates all objectives simultaneously at low request counts on both H100 and A100. No tradeoffs exist. [Carried from iter-2]
- **Exhaustive TP sweep with best secondary params assuming TP=8 always wins**: At 1000 requests, TP=4/2inst is the actual best for both models on both H100 and A100. [Carried from iter-3]
- **Random K=2 profiles for Phase 1 on tight-margin regimes**: 3/5 on H100 hard, 2/5 on A100 easy. Fails because random doesn't guarantee mr>=256. [From iter-5]
- **Assuming TP=8 easy dominance is unconditional**: Requires max_running >= 256. With mr<256, TP=4/2inst wins even in easy regimes. [From iter-5]
- **Running blis from subdirectory**: The blis binary must be run from the project root `/Users/jchen/go/src/inference-sim/inference-sim`. Running from a subdirectory gives "No such file or directory". [Carried from iter-3]
- **grep "Score:" with awk in for-loop subshell**: When the working directory isn't the project root, `./blis` fails silently and score is empty. Always cd to project root first or use absolute path. [Carried from iter-3]
- **Assuming round-robin always beats least-loaded for 2-instance configs**: At seed=42 round-robin wins (0.138820 vs 0.138290), but at seed=46 least-loaded wins (0.147057 vs 0.146765). The routing advantage is seed-dependent. [New in iter-4]

## What I Excluded and Why

- **Third model (qwen3-32b)**: Model config exists but 32B params may cause memory issues on some TP configs (especially TP=1 on L40S 48 GiB). Not worth the debugging risk when A100-SXM qwen3-14b provides cleaner evidence. [Deferred to future iteration]
- **Rate sensitivity mapping**: Testing different rates (e.g., rate=200, rate=1000) to map the TP dominance transition point. Orthogonal to the matrix completion and margin threshold questions. [Carried from iter-3]
- **L40S A100 bandwidth threshold mapping**: The exact bandwidth value where qwen switches from TP=8 to TP=4 lies between 0.864 and 2.039 TB/s. Narrowing this further requires synthetic hardware configs, which BLIS doesn't support. [Excluded — tool limitation]
- **Easy regime (200 req/rate=200) for A100-SXM qwen**: TP=8 likely dominates trivially. Not interesting given the hard-regime focus. [Carried from iter-3]
- **Phase 2 value quantification as a separate arm**: Phase 2 adds 0.38% on A100-SXM qwen (round-robin discovery). This is observable in h-main results without a dedicated arm. [Excluded — observable as a secondary metric]
- **Bracket-min-only on A100-SXM**: Min profile's failure is well-characterized (RP-17). Adding another hardware would not reveal new mechanisms. [Carried from iter-3]

## Evolution of Thinking

Iterations 1–3 established bracket K=1 as the recommended search algorithm and tested it across 2 models × 2 hardware platforms (H100, L40S). The A100-SXM gap was the last untested cell in the model×hardware matrix. Probing revealed a clean result: A100-SXM follows H100 behavior (TP=4 for both models), confirming the bandwidth threshold hypothesis from RP-21.

The more interesting discovery is the margin variation across seeds on A100-SXM. Seeds 42 and 44 produce 1.07% and 1.24% TP margins — the tightest TP=4 margins in the entire campaign (prior tightest TP=4 margin was L40S llama at 2.1%). Meanwhile, seeds 45 and 46 produce 4.09% and 4.47% margins. This creates a natural within-hardware experiment: by comparing flat TPE at tight (seed=42) vs wide (seed=46) margins, we can test whether margin width — not hardware type — is the causal variable for TPE convergence reliability.

Another minor but interesting finding: the Phase 1 max-profile (which uses least-loaded routing for multi-instance configs) does NOT produce the global best on A100-SXM. Round-robin routing provides a 0.38% improvement. This is the first regime in the campaign where Phase 2 TPE adds genuine value beyond Phase 1, making the full hierarchical algorithm (Phase 1 + Phase 2) necessary for finding the true optimum. In all prior regimes, Phase 1 alone found the global best or came within 0.01%.

## Current Status

- **Validated:** A100-SXM qwen Phase 1 max-profile scores at all 5 blis_seeds. TP=4 wins at all seeds. Global best is 0.138820 (round-robin, seed=42). All Phase 1 TP=4 scores exceed the 1% threshold (0.137432) at all seeds. The 2×3 matrix is structurally complete (all cells probed), pending formal bracket K=1 experiment confirmation.
- **Uncertain:** (1) How flat TPE performs at the 1.07% margin — predicted to fail but not directly tested. (2) Whether flat TPE succeeds at 4.47% — predicted yes based on L40S qwen 5.1% analogy but margin effect could be non-linear. (3) Whether Phase 2 TPE consistently finds the round-robin improvement across search seeds — it may be too small (0.38%) for TPE to reliably discover. (4) Whether the routing effect (round-robin vs least-loaded) is consistent across A100-SXM seeds (already found to be seed-dependent).
- **Suggested next:** (1) Third model (qwen3-32b) to test model-size scaling — does a larger model push the bandwidth threshold higher? (2) Rate sensitivity mapping — test rate=100, 200, 1000 to understand how arrival rate affects the TP margin and search difficulty. (3) Package the search algorithm as a production tool with auto-detection of optimal strategy based on margin estimation. (4) Test easy regime (200 req) on A100-SXM for completeness.

## Warnings & Constraints

- **Stderr contains warnings**: BLIS prints timing and config warnings to stderr. Always use `2>/dev/null` or capture stderr separately when parsing stdout. [Carried from iter-2]
- **Seed affects workload, not config evaluation**: `--seed 42` controls random request generation. For the search algorithm's own randomness, use a separate seed. [Carried from iter-2]
- **TP=0 is invalid**: Always set `--tp` explicitly to avoid auto-detection warnings. [Carried from iter-2]
- **Fitness score format on stdout**: `Score: 0.019249` — parse with regex `r'Score:\s+([\d.]+)'`. [Carried from iter-2]
- **Multi-instance JSON output**: When `--num-instances > 1`, stdout contains multiple JSON blocks. Always parse the block with `"instance_id": "cluster"` for aggregate metrics. [Carried from iter-2]
- **Hardware name is case-sensitive**: `H100` not `h100`, `A100-SXM` not `a100-sxm`, `L40S` not `l40s`. [Carried from iter-5]
- **Global best is (model, hardware, rate, num_requests)-specific**: Updated entry: ("qwen3-14b", "A100-SXM", 500, 1000): 0.138820. [Updated iter-4]
- **Bracket profiles are deterministic**: They don't depend on search_seed (no RNG involved). All 5 search seeds produce identical Phase 1 results per blis_seed. [Carried from iter-6]
- **Profile ordering affects evals_to_best**: Max-first is optimal. [Carried from iter-6]
- **Model name format**: Use full model paths: `qwen/qwen3-14b`, `meta-llama/llama-3.1-8b-instruct`. [Carried from iter-3]
- **A100/L40S prefill_threshold is a no-op for fitness**: All values produce identical scores at fixed (max_running, max_tokens). [Carried from iter-2]
- **vllm-version auto-detection**: Warnings about "Finding default values of vLLM version" appear on stderr. Harmless but noisy. [Carried from iter-2]
- **L40S memory is 48 GiB (not 80 GiB)**: TP=8/1inst on L40S may have different memory constraints than H100/A100. [Carried from iter-2]
- **L40S TP winner is model-dependent**: llama→TP=4, qwen→TP=8 on L40S hard. On H100/A100, both models→TP=4. [Updated iter-4 to confirm A100]
- **GLOBAL_BEST_TABLE uses seed=42 values for all seeds**: Threshold is conservative for seeds with higher actual best. All tested seeds pass. [Carried from iter-2]
- **Single-instance configs have degenerate routing**: At TP=8/1inst, routing_policy and block_size have no effect (verified). [Carried from iter-3]
- **Must run blis from project root**: The binary expects model configs relative to the project directory. [Carried from iter-3]
- **Routing policy effect is seed-dependent on A100-SXM**: round-robin > least-loaded at seed=42, but least-loaded > round-robin at seed=46. The global best routing varies by workload realization. [New in iter-4]
- **A100-SXM TP margins vary 4.2× across seeds**: From 1.07% (seed=42) to 4.47% (seed=46). This range spans the L40S llama failure regime (2.1%) and approaches the L40S qwen success regime (5.1%), making A100-SXM the ideal hardware for margin threshold testing. [New in iter-4]
