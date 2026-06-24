# Handoff — Iteration 5: Rate-Dependent TP Winner Transition

## Goal

Test whether bracket K=1 hierarchical search correctly adapts to rate-dependent TP winner transitions on H100. At rate=100, TP=8/1inst wins (5.2-6.6% margin), reversing the TP=4/2inst dominance seen at rate≥200. Run bracket K=1 at rate=100 (new TP winner) and rate=125 (crossover point with ~0% margin), plus flat TPE at rate=100 for comparison. Validate across multiple blis_seeds and search_seeds.

## Key Discoveries

1. **Rate-dependent TP winner crossover on H100**: For qwen3-14b with max-profile params, TP=8/1inst wins at rate≤125, TP=4/2inst wins at rate≥150. The crossover is at rate≈125 (seed=42: 0.02% margin). At rate=100, TP=8 wins by 5.71% — a robust, wide margin. At rate=125, the margin is seed-dependent (0.02-0.81% for 4/5 seeds TP=8, seed=46 flips to TP=4 by 0.17%).

2. **Crossover is model-independent on H100**: Llama-3.1-8b shows the same pattern — TP=8 at rate=100 (margin 2.86%), TP=4 at rate≥500 (margin 7.75%). The crossover rate may differ between models but the direction is consistent.

3. **Monotonic fitness landscape at rate=100**: Unlike rate≥200 where TP=4 > TP=8 (non-monotonic), rate=100 shows strictly TP=1 < TP=2 < TP=4 < TP=8. This creates an "easier" landscape for any search algorithm.

4. **evals_to_best=4 at rate=100 vs =3 at rate≥150**: When TP=8 is the winner, evals_to_best is 4 (TP=8 is the 4th evaluated). At rate=125 (crossover), evals_to_best=3 because TP=4 at position 3 exceeds the 1% threshold even though TP=8 is marginally better.

5. **Default-profile TP4 dominance at ALL rates**: With default params (mr=64, mt=4096, pf=0), TP=4/2inst wins at every rate by 14.8-15.7%. The max-profile is essential for revealing the rate-dependent TP=8 advantage at low rates. This confirms RP-13: TP=8 dominance requires max_running >= 256.

6. **Global best values for new regimes (seed=42)**: qwen3-14b H100 rate=100: 0.164680, rate=125: 0.162182. llama-3.1-8b H100 rate=100: 0.182394.

7. **A100-SXM qwen global best correction**: The correct global best for (qwen3-14b, A100-SXM, 500, 1000) is 0.138820 (round-robin routing), not 0.138290 (least-loaded). This was identified in iter-4 but not applied to the iter-6 script.

## System Interface

- **Build:** `go build -o blis main.go` (binary already exists in project root)
- **Run baseline (rate=100, TP=8, H100 qwen):**
  ```bash
  ./blis run --model qwen/qwen3-14b --hardware H100 --latency-model trained-physics \
    --num-requests 1000 --rate 100 --seed 42 --tp 8 --num-instances 1 \
    --scheduler fcfs --max-num-running-reqs 512 --max-num-scheduled-tokens 8192 \
    --long-prefill-token-threshold 4096 --block-size-in-tokens 16 \
    --routing-policy round-robin --admission-policy always-admit \
    --preemption-policy fcfs \
    --fitness-weights "throughput:0.4,p99_ttft:0.3,p99_e2e:0.3"
  ```
- **Output format:** JSON metrics between `=== Simulation Metrics ===` markers. With `--fitness-weights`, `Score: <float>` follows in `=== Fitness Evaluation ===` section.
- **Baseline result:** H100 qwen TP=8/1inst rate=100 Score=0.164680.

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

The experiment modifies the Python search script from iter-6's h-main.patch (which contains search_blis_iter6.py with bracket profiles):

- **`search_blis_iter5.py`** (copy from iter-6 h-main.patch):
  1. **Add GLOBAL_BEST_TABLE entries for rate=100**:
     - `("qwen3-14b", "H100", 100, 1000): 0.164680`
     - `("llama-3.1-8b", "H100", 100, 1000): 0.182394`
  2. **Add GLOBAL_BEST_TABLE entry for rate=125**:
     - `("qwen3-14b", "H100", 125, 1000): 0.162182`
  3. **Update A100-SXM qwen entry**: Change `("qwen3-14b", "A100-SXM", 500, 1000): 0.138290` to `0.138820`.
  4. No algorithmic changes — bracket K=1 is already implemented.

## What I Tried That Didn't Work

- **`--hardware h100_sxm`**: Fatal error. Correct name is `H100` (case-sensitive, no suffix). [Carried from iter-1]
- **`--hardware A100`**: Fatal error. Must be `A100-SXM` or `A100-80`. [Carried from iter-5]
- **Parsing fitness from `--metrics-path` JSON**: The JSON file does not contain the fitness score. Must parse stdout. [Carried from iter-1]
- **Exhaustive TP sweep with best secondary params assuming TP=8 always wins**: At rate≥200, TP=4/2inst is the actual best. TP=8 dominance is rate-conditional. [Updated]
- **Running blis from subdirectory**: The blis binary must be run from the project root `/Users/jchen/go/src/inference-sim/inference-sim`. [Carried from iter-3]
- **grep "Score:" with awk in for-loop subshell**: When the working directory isn't the project root, `./blis` fails silently and score is empty. Always cd to project root first. [Carried from iter-3]
- **Assuming round-robin always beats least-loaded for 2-instance configs**: The routing advantage is seed-dependent. [Carried from iter-4]
- **Random K=2 profiles for Phase 1 on tight-margin regimes**: 3/5 on H100 hard, 2/5 on A100 easy. Fails because random doesn't guarantee mr>=256. [Carried]
- **Assuming TP=8 dominance is unconditional on batch size**: Requires max_running >= 256. With mr<256, TP=4/2inst wins even at rate=100. [Updated from RP-13]

## What I Excluded and Why

- **Third model (qwen3-32b)**: Model config exists but 32B params may cause memory issues on some TP configs. Not worth debugging risk. [Deferred]
- **L40S/A100-SXM at rate=100**: The rate crossover rate likely differs per hardware (different bandwidth creates different TP tradeoffs). Interesting but orthogonal to the core rate-adaptiveness question on H100. [Deferred to future iteration]
- **Llama rate crossover mapping**: Probes show same pattern (TP=8 at rate=100, TP=4 at rate≥500), but the exact crossover rate wasn't narrowed. Could be tested in a future iteration. [Deferred]
- **Easy regime (200 req/200 requests) at rate=100**: Only meaningful if num_requests is also reduced, which creates short-horizon instability. [Excluded]
- **Rate sensitivity with non-H100 hardware**: Testing rate variations on L40S or A100-SXM would require new global_best probing. One hardware is sufficient to establish the rate-adaptiveness property. [Deferred]
- **Phase 2 value quantification**: At rate=100, TP=8/1inst is single-instance, so Phase 2 can only vary scheduler/mr/mt/pf/admission/preemption. The max-profile is likely near-optimal. Phase 2 adds minimal value here. [Not worth a separate arm]

## Evolution of Thinking

Iterations 1-4 established bracket K=1 as the recommended search algorithm across a 2×3 (model × hardware) matrix, always at fixed rate regimes. The implicit assumption was that the TP winner is rate-invariant — determined by model architecture and hardware bandwidth.

This iteration's probing revealed that assumption is wrong. The TP winner on H100 transitions from TP=8 (rate≤125) to TP=4 (rate≥150), creating a previously unknown dimension of variation. The mechanism is intuitive: at low rates, per-request throughput dominates (favoring TP=8's full GPU utilization), while at high rates, concurrency benefits from multiple instances dominate (favoring TP=4/2inst).

The critical question is whether bracket K=1 handles this transition without modification. The answer appears to be yes: the max-profile Phase 1 naturally reveals the rate-dependent competitive dynamics. At rate=100, Phase 1 scores are monotonically increasing with TP (easy landscape, evals_to_best=4). At rate=125 (crossover), both TP=4 and TP=8 exceed the 1% threshold (evals_to_best=3). At rate≥200, TP=4 dominates (evals_to_best=3). The algorithm adapts to rate without any modification.

An unexpected finding: the default-profile (mr=64) shows TP=4 winning at ALL rates, even rate=100. This means the rate-dependent TP=8 advantage only manifests with high batch budgets (RP-13 interaction). The max-profile used in bracket Phase 1 is essential for revealing the true rate-dependent landscape.

## Current Status

- **Validated:** Rate-TP crossover on H100 qwen3-14b at 6 rates (100-1000). Multi-seed TP margin stability at rate=100 (5 seeds, all TP=8 with 5.2-6.6% margin). Multi-seed crossover behavior at rate=125 (4/5 TP=8, 1/5 TP=4). Llama cross-model confirmation of rate-dependent TP transition. Baseline command exits cleanly with correct fitness output.
- **Uncertain:** (1) Exact crossover rate for llama on H100 (between 100 and 500, not narrowed). (2) Whether flat TPE converges faster at rate=100 than rate=500 as predicted (needs experiment). (3) Whether the rate crossover exists on L40S/A100-SXM at similar rates. (4) Whether the h-ablation evals_to_best=3 prediction holds across all 25 seed combinations at rate=125.
- **Suggested next:** (1) L40S/A100 rate crossover mapping — does the crossover rate differ by hardware? (2) Larger model (qwen3-32b) rate sensitivity — does model size shift the crossover rate? (3) Package the search algorithm as a production tool with auto-rate-detection. (4) Rate × hardware interaction matrix — 3 rates × 3 hardware to map the full TP winner surface.

## Warnings & Constraints

- **Stderr contains warnings**: BLIS prints timing and config warnings to stderr. Always use `2>/dev/null` or capture stderr separately when parsing stdout. [Carried from iter-2]
- **Seed affects workload, not config evaluation**: `--seed 42` controls random request generation. For the search algorithm's own randomness, use a separate seed. [Carried from iter-2]
- **TP=0 is invalid**: Always set `--tp` explicitly. [Carried from iter-2]
- **Fitness score format on stdout**: `Score: 0.019249` — parse with regex `r'Score:\s+([\d.]+)'`. [Carried from iter-2]
- **Multi-instance JSON output**: When `--num-instances > 1`, stdout contains multiple JSON blocks. Always parse the block with `"instance_id": "cluster"`. [Carried from iter-2]
- **Hardware name is case-sensitive**: `H100` not `h100`, `A100-SXM` not `a100-sxm`. [Carried]
- **Global best is (model, hardware, rate, num_requests)-specific**: New entries added for rate=100 and rate=125. [Updated iter-5]
- **Bracket profiles are deterministic**: They don't depend on search_seed (no RNG involved). All 5 search seeds produce identical Phase 1 results per blis_seed. [Carried]
- **Profile ordering affects evals_to_best**: Max-first is optimal. [Carried]
- **Model name format**: Use full model paths: `qwen/qwen3-14b`, `meta-llama/llama-3.1-8b-instruct`. [Carried]
- **Prefill_threshold is a no-op for fitness on some hardware**: All values produce identical scores at fixed (max_running, max_tokens) on A100/L40S. [Carried]
- **vllm-version auto-detection**: Warnings about "Finding default values of vLLM version" appear on stderr. Harmless but noisy. [Carried]
- **Single-instance configs have degenerate routing**: At TP=8/1inst, routing_policy and block_size have no effect. [Carried from iter-3]
- **Must run blis from project root**: The binary expects model configs relative to the project directory. [Carried from iter-3]
- **Routing policy effect is seed-dependent on multi-instance configs**: round-robin vs least-loaded varies by workload realization. [Carried from iter-4]
- **Rate-dependent TP winner on H100**: TP=8 wins at rate≤125, TP=4 wins at rate≥150 (qwen3-14b). This is a max-profile phenomenon — default-profile always shows TP=4 winning. [New in iter-5]
- **Rate=125 is the crossover point**: Margin is 0-0.81%, seed-dependent. Both TP=4 and TP=8 are within 1% of each other. [New in iter-5]
- **evals_to_best=4 when TP=8 wins**: Unlike prior iterations where evals_to_best=3 (TP=4 at position 3), TP=8 is at position 4 in the Phase 1 evaluation order. This is an inherent 1-eval penalty for TP=8 being last. [New in iter-5]
- **Default-profile masks rate effect**: With mr=64, TP=4/2inst wins at ALL rates (14.8-15.7% margin). The rate-dependent TP=8 advantage only appears with max_running≥256. [New in iter-5]
