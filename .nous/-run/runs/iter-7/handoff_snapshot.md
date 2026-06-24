# Handoff — Iteration 7: Lean Bracket (50x Budget Reduction)

## Goal

Test whether bracket K=1 can be reduced from 100 evaluations (4 Phase 1 + 96 Phase 2) to 2 evaluations (Phase 1 over TP={4,8} only, no Phase 2) while maintaining 1% accuracy across all tested regimes. This is a 50x budget reduction (~200ms total search time vs ~10s). The hypothesis is that TP=1 and TP=2 are structurally excluded (11-38% below winner at all rates) and Phase 2 never improves by more than 0.7% (within tolerance). Includes a negative control showing min-profile breaks lean bracket, an ablation testing mini Phase 2 for routing recovery, and a robustness test across 5 blis_seeds.

## Key Discoveries

1. **TP=2 structural exclusion is rate-independent**: Probed TP=2 gap at rates 100, 500, 1000, 2000, 5000 on H100 qwen. Gap is consistently 11-12% below winner at ALL rates. TP=1 gap is 32-38%. Neither converges toward the winner even at rate=5000. This means pruning TP=1 and TP=2 from Phase 1 is safe regardless of arrival rate.

2. **Min-profile inverts TP ranking, confirming mechanism**: With min-profile (mr=32, mt=2048, pf=0) on H100 qwen rate=500, TP=2/4inst=0.082056 WINS over TP=4/2inst=0.076957 and TP=8/1inst=0.062982. This proves lean bracket's correctness depends on the max-profile creating a compute-saturated regime where bandwidth dominates. This is the negative control mechanism.

3. **Phase 2 improvement is always <1%**: Across ALL regimes in iterations 1-6 using bracket K=1 with max-profile: Phase 2 improvement over Phase 1 ranges 0% (single-instance winners) to 0.71% (multi-instance routing discovery). The maximum observed is 0.71% on A100-SXM qwen rate=200 (crossover regime). All within 1% tolerance.

4. **Phase 2 routing discovery timing varies by seed**: On H100 qwen rate=500, Phase 2 finds the routing improvement at trial 1 (seed=46), 3 (seed=44), 6 (seed=45), 19 (seed=43), 37 (seed=42). Mini Phase 2 budget=10 would succeed on 3/5 seeds.

5. **Phase 1 max-profile scores for lean bracket regimes (seed=42)**:
   - H100 qwen rate=500: TP4=0.178063, TP8=0.169720. TP4 wins. Global best=0.179282. Gap=0.68%.
   - H100 qwen rate=100: TP4=not probed (known <TP8), TP8=0.164680. TP8 wins. Global best=0.164680. Gap=0%.
   - A100-SXM qwen rate=500: TP4=0.138290, TP8=0.136812. TP4 wins. Global best=0.138820. Gap=0.38%.
   - A100-SXM qwen rate=100: TP4=0.127461, TP8=0.130754. TP8 wins. Global best=0.130754. Gap=0%.

6. **Lean bracket evals_to_best depends on winner position**: TP=4 is evaluated at position 1 (evals_to_best=1 when TP=4 wins), TP=8 at position 2 (evals_to_best=2 when TP=8 wins). This is a further 1-2 eval improvement over full bracket's 3-4.

7. **Weighted routing can outperform round-robin**: On A100-SXM qwen rate=200, weighted routing (0.135440) beats round-robin (0.134935) and least-loaded (0.134488). This was discovered by Phase 2 TPE in iter-6 h-ablation. The "global best" for this regime should be 0.135440 (weighted, not round-robin as previously noted in RP-23).

## System Interface

- **Build:** `go build -o blis main.go` (binary already exists in project root)
- **Run baseline (H100 qwen rate=500, TP=4 lean bracket winner):**
  ```bash
  ./blis run --model qwen/qwen3-14b --hardware H100 --latency-model trained-physics \
    --num-requests 1000 --rate 500 --seed 42 --tp 4 --num-instances 2 \
    --scheduler fcfs --max-num-running-reqs 512 --max-num-scheduled-tokens 8192 \
    --long-prefill-token-threshold 4096 --block-size-in-tokens 16 \
    --routing-policy least-loaded --admission-policy always-admit \
    --preemption-policy fcfs \
    --fitness-weights "throughput:0.4,p99_ttft:0.3,p99_e2e:0.3"
  ```
- **Output format:** JSON metrics between `=== Simulation Metrics ===` markers. With `--fitness-weights`, `Score: <float>` follows in `=== Fitness Evaluation ===` section.
- **Baseline result:** H100 qwen TP=4/2inst rate=500 Score=0.178063.

## Code Map

- `cmd/root.go:947-984` — All swept parameter flag definitions. Check here if a flag name or default is wrong.
- `cmd/root.go:956` — `--hardware` flag definition. Valid values: H100, A100-SXM, A100-80, L40S.
- `cmd/root.go:938` — `--seed` flag definition (Int64, default 42). Controls workload generation RNG.
- `cmd/root.go:940` — `--rate` flag definition (Float64). Arrival rate in req/s.
- `cmd/root.go:1761-1765` — Fitness weight parsing and ComputeFitness call. Check here if fitness isn't being computed.
- `sim/latency/config.go:86-101` — `GetHWConfig()` validates hardware names. Check here if a hardware name is rejected.
- `hardware_config.json` — GPU specs: H100 (989.5 TFLOPS, 80 GiB, 3.35 TB/s), A100-SXM (312 TFLOPS, 80 GiB, 2.039 TB/s), L40S (362 TFLOPS, 48 GiB, 0.864 TB/s).
- `sim/cluster/metrics.go:418-498` — `validFitnessKeysList()`, `ComputeFitness()`, `extractMetric()`. Check here if fitness scores look wrong.
- `sim/cluster/metrics.go:420-426` — The 8 valid fitness keys.
- `sim/cluster/metrics.go:428-435` — Reference scale constants for normalization.
- `sim/metrics_utils.go:57-87` — MetricsOutput struct: all JSON fields in the metrics output.

## Code Targets

The experiment modifies the Python search script. Start from the iter-6 h-main.patch (which contains search_blis_iter6.py):

- **`search_blis_iter7.py`** (copy from search_blis_iter6.py in the iter-6 h-main.patch):
  1. **Add `--tp-candidates` CLI flag**: choices "all" (default, TP=1,2,4,8) and "lean" (TP=4,8 only). When "lean", filter `PHASE1_TP_CONFIGS` to `[(4, 2), (8, 1)]`. Apply the filter in `run_adaptive_hierarchical_search()` before the Phase 1 loop.
  2. **Add `--skip-phase2` boolean flag**: When set, return the Phase 1 result immediately after the Phase 1 loop completes (before the Phase 2 TPE section). The returned dict should include all Phase 1 metadata (phase1_winner, phase1_tp_scores, etc.).
  3. **Update GLOBAL_BEST_TABLE** with entries from iterations 5-6:
     - Add: `("qwen3-14b", "H100", 100, 1000): 0.164680`
     - Add: `("qwen3-14b", "H100", 125, 1000): 0.162182`
     - Add: `("llama-3.1-8b", "H100", 100, 1000): 0.182394`
     - Add: `("qwen3-14b", "A100-SXM", 100, 1000): 0.130754`
     - Add: `("llama-3.1-8b", "A100-SXM", 100, 1000): 0.160845`
     - Add: `("qwen3-14b", "A100-SXM", 200, 1000): 0.135440`
     - Update: `("qwen3-14b", "A100-SXM", 500, 1000)` from `0.138290` to `0.138820`
  4. **No algorithmic changes to bracket K=1 or Phase 2 TPE** — the core search logic is unchanged.

## What I Tried That Didn't Work

- **`--hardware h100_sxm`**: Fatal error. Correct name is `H100` (case-sensitive, no suffix). [Carried from iter-1]
- **`--hardware A100`**: Fatal error. Must be `A100-SXM` or `A100-80`. [Carried from iter-5]
- **Parsing fitness from `--metrics-path` JSON**: The JSON file does not contain the fitness score. Must parse stdout. [Carried from iter-1]
- **Running blis from subdirectory**: The blis binary must be run from the project root `/Users/jchen/go/src/inference-sim/inference-sim`. [Carried from iter-3]
- **grep "Score:" with awk in for-loop subshell**: When the working directory isn't the project root, `./blis` fails silently and score is empty. Always cd to project root first. [Carried from iter-3]
- **Assuming TP=8 dominance is unconditional on batch size**: Requires max_running >= 256. With mr<256, TP=4/2inst wins even at rate=100. [Carried, RP-13]
- **Assuming round-robin always beats least-loaded**: The routing advantage is seed-dependent. On A100-SXM at rate=200, weighted routing actually beats round-robin. [Updated from iter-4: weighted is the true global best, not round-robin]
- **Assuming rate crossover is H100-specific**: It's universal — A100-SXM crosses at ~200, L40S llama at ~450. [Carried from iter-6]
- **Assuming TP=2 might eventually win at extreme rates**: Probed at rate=1000, 2000, 5000 on H100 qwen — TP=2 gap is constant at ~11%, never converging. [New in iter-7]

## What I Excluded and Why

- **TP=1/8inst probing at extreme rates**: TP=1 is always 32-38% below winner. Including it in probes would waste evaluation budget without informing the hypothesis. [Design choice]
- **L40S lean bracket arms**: L40S qwen has no crossover (TP=8 always wins) and L40S llama's crossover at ~450 hasn't been fully characterized. Including L40S would add regimes without new mechanistic insight. [Deferred — could be added in iter-8]
- **llama model lean bracket arms**: The lean bracket mechanism (TP=4 and TP=8 dominate) is model-agnostic. Testing on 1 model (qwen) across 2 hardware platforms is sufficient. llama could be added for completeness in a future iteration. [Deferred]
- **Analytical TP prediction (zero-eval)**: Could predict TP winner from hardware bandwidth and arrival rate using the formula crossover_rate ≈ 400/bandwidth_TB_s (RP-27). This would eliminate even the 2 Phase 1 evaluations. Deferred because it requires validating the formula across more regimes and model sizes. [Suggested for iter-8]
- **Phase 2 early stopping**: An alternative to mini Phase 2 is to run Phase 2 with early stopping (stop if no improvement in N trials). Not tested because the ablation (budget=12) is simpler and provides the same insight. [Design choice]
- **Budget sweep (budget=5, 10, 20, 50)**: Could systematically find the optimal Phase 2 budget. Not tested because the binary question (Phase 2 needed or not?) is more informative. [Deferred]

## Evolution of Thinking

Iterations 1-6 established bracket K=1 as a robust, hardware-portable, model-agnostic search algorithm achieving evals_to_best of 3-4 across all tested conditions. The natural next question was: **can we go lower?**

The original campaign explored search algorithm efficiency starting from random search (budget=100) through TPE and hierarchical search, ultimately arriving at bracket K=1. But bracket K=1 still uses 4 Phase 1 evaluations (one per TP level) and 96 Phase 2 trials. The accumulated evidence showed that 2 of the 4 TP levels (TP=1, TP=2) never win, and Phase 2 almost never improves over Phase 1.

The key insight from iter-7 probing is that TP=2's structural exclusion is **rate-independent**: even at rate=5000 (10x the standard hard regime), TP=2 remains 11% below the winner. This isn't a marginal empirical finding — it's a structural property of the BLIS latency model where bandwidth-per-GPU dominance over instance-count advantage holds across all practical operating points.

The min-profile probe was the critical validation: it showed that lean bracket's correctness is specifically tied to the max-profile creating a compute-saturated regime. With min-profile, TP=2 actually wins — proving the mechanism is well-understood, not just an observed correlation.

## Current Status

- **Validated:** (1) TP=2 gap is 11-12% across rates 100-5000 on H100 qwen. (2) TP=1 gap is 32-38%. (3) Min-profile inverts TP ranking on H100 (TP=2/4inst wins with mr=32). (4) Phase 1 max-profile is within 1% of global best on ALL regimes in campaign. (5) Phase 2 max improvement is 0.71% (A100-SXM qwen rate=200). (6) Lean bracket baseline command works (H100 qwen rate=500 TP=4/2inst = 0.178063).
- **Uncertain:** (1) Whether lean bracket holds on L40S (not included in h-main arms but expected to work). (2) Whether weighted routing consistently beats round-robin on A100-SXM at crossover — only blis_seed=42 tested. (3) Exact evals_to_best distribution for mini Phase 2 (budget=10) — based on prior data but not directly tested. (4) Whether TP=2 exclusion holds for models larger than 14B params.
- **Suggested next:** (1) **Analytical TP prediction (zero-eval)**: Use crossover_rate ≈ 400/bandwidth_TB_s to predict TP winner without any BLIS evaluations. Would reduce search from 2 evals to 0. (2) **Lean bracket on L40S**: Validate the algorithm on the remaining hardware platform. (3) **Third model (qwen3-32b)**: Test whether larger models change the TP=2 exclusion threshold. (4) **Production `blis search` subcommand**: Package lean bracket as a native BLIS CLI command.

## Warnings & Constraints

- **Stderr contains warnings**: BLIS prints timing and config warnings to stderr. Always use `2>/dev/null` or capture stderr separately when parsing stdout. [Carried from iter-2]
- **Seed affects workload, not config evaluation**: `--seed 42` controls random request generation. For the search algorithm's own randomness, use a separate seed. [Carried from iter-2]
- **TP=0 is invalid**: Always set `--tp` explicitly. [Carried from iter-2]
- **Fitness score format on stdout**: `Score: 0.019249` — parse with regex `r'Score:\s+([\d.]+)'`. [Carried from iter-2]
- **Multi-instance JSON output**: When `--num-instances > 1`, stdout contains multiple JSON blocks. Always parse the block with `"instance_id": "cluster"`. [Carried from iter-2]
- **Hardware name is case-sensitive**: `H100` not `h100`, `A100-SXM` not `a100-sxm`. [Carried]
- **Global best is (model, hardware, rate, num_requests)-specific**: New entries added for rate=100/125/200 regimes. [Updated iter-7]
- **Bracket profiles are deterministic**: They don't depend on search_seed (no RNG involved). All search seeds produce identical Phase 1 results per blis_seed. [Carried]
- **Profile ordering affects evals_to_best**: In lean bracket, TP=4 is evaluated first (position 1), TP=8 second (position 2). [Updated iter-7]
- **Model name format**: Use full model paths: `qwen/qwen3-14b`, `meta-llama/llama-3.1-8b-instruct`. [Carried]
- **Prefill_threshold is a no-op for fitness on some hardware**: All values produce identical scores at fixed (max_running, max_tokens) on A100/L40S. [Carried]
- **vllm-version auto-detection**: Warnings about "Finding default values of vLLM version" appear on stderr. Harmless but noisy. [Carried]
- **Single-instance configs have degenerate routing**: At TP=8/1inst, routing_policy and block_size have no effect. [Carried from iter-3]
- **Must run blis from project root**: The binary expects model configs relative to the project directory. [Carried from iter-3]
- **Routing policy effect is seed-dependent on multi-instance configs**: round-robin vs least-loaded vs weighted varies by workload realization. [Updated: weighted can beat round-robin]
- **Rate-dependent TP winner on H100**: TP=8 wins at rate<=125, TP=4 wins at rate>=150 (qwen3-14b). [Carried from iter-5]
- **Rate-dependent TP winner on A100-SXM**: TP=8 wins at rate<=150, tied at rate=200, TP=4 wins at rate>=500. [Carried from iter-6]
- **L40S has the highest crossover rate**: ~450 for llama (TP=8 at rate<=400, TP=4 at rate=500). Qwen has no crossover on L40S. [Carried from iter-6]
- **Crossover rate inversely proportional to bandwidth**: H100 (3.35 TB/s) ~125, A100-SXM (2.039 TB/s) ~200, L40S (0.864 TB/s) ~450+. [Carried from iter-6]
- **A100-SXM qwen rate=200 global best uses weighted routing**: TP=4/2inst with weighted=0.135440 vs round-robin=0.134935 vs least-loaded=0.134488. [Updated in iter-7]
- **evals_to_best in lean bracket**: 1 when TP=4 wins (position 1), 2 when TP=8 wins (position 2). Full bracket: 3 (TP=4) or 4 (TP=8). [New in iter-7]
- **Default-profile masks rate effect**: With mr=64, TP=4/2inst wins at ALL rates. The rate-dependent TP=8 advantage only appears with max_running>=256 (max-profile). [Carried from iter-5]
- **TP=2 gap is ~11% on H100 qwen regardless of rate**: Probed at rates 100-5000. Gap does not converge. [New in iter-7]
- **Min-profile reverses TP ranking**: With mr=32, TP=2/4inst (0.082056) beats TP=4/2inst (0.076957) on H100 qwen rate=500. Max-profile is essential for lean bracket. [New in iter-7]
