# Handoff — Iteration 9: 70B Model Generalization and Model-Aware Formula

## Goal

Validate that lean bracket (TP={4,8}, max-profile, budget=2, no Phase 2) generalizes to llama-3.1-70b-instruct (4.4× larger than previous largest tested model) across all 3 hardware platforms and 2 arrival rates. Simultaneously, test a model-size-aware crossover formula that adds a load_index threshold (params/BW > 12 → predict TP=8 always) to eliminate the 6 remaining failures of the bandwidth-only formula.

## Key Discoveries

1. **70B TP=8 dominates on ALL hardware at ALL rates**: H100 margins 2.5-10.3% (5 seeds), A100-SXM margins 6.7-13.8% (5 seeds), L40S margins 18-20% (1 seed). No crossover exists at any tested rate (100-2000).

2. **TP=1 is infeasible for 70B**: Fatal OOM error — 137.07 GiB model overhead exceeds 72.00 GiB available GPU memory at TP=1. TP=2 runs but scores 6.9× worse than TP=8 (0.012899 vs 0.091801). Lean bracket's TP={4,8} naturally avoids both.

3. **Model-aware formula achieves 24/24 (100%) on the {100, 500} test grid**: Adding `load_index = params/BW > 12 → TP=8` to the bandwidth formula eliminates all 6 failures. The boundary lies between load_index 9.6 (crossover exists) and 15.7 (no crossover).

4. **Default-profile preserves TP=8 winner for 70B**: With mr=256/mt=4096/pf=1024, TP=8 still wins on H100 rate=500 (0.076004 vs 0.073035, margin 4.1%). Only min-profile (mr=32) inverts the ranking (TP=4 leads by 10.2%).

5. **70B eval wall time is ~0.25-0.27s** across all hardware (similar to 14B/32B). Lean bracket completes in ~0.5s for 70B.

6. **Multi-blis-seed variation for 70B is moderate**: H100 rate=500 scores range 0.091801-0.095769 across seeds 42-46. TP=8 wins on all 5 seeds, but margin varies from 2.5% (seed 45) to 10.3% (seed 42).

## System Interface

- **Build:** `go build -o blis main.go` (binary already exists in project root)
- **Run baseline (70B H100 rate=500, TP=8 lean bracket winner):**
  ```bash
  ./blis run --model meta-llama/llama-3.1-70b-instruct --hardware H100 --latency-model trained-physics \
    --num-requests 1000 --rate 500 --seed 42 --tp 8 --num-instances 1 \
    --scheduler fcfs --max-num-running-reqs 512 --max-num-scheduled-tokens 8192 \
    --long-prefill-token-threshold 4096 --block-size-in-tokens 16 \
    --routing-policy least-loaded --admission-policy always-admit \
    --preemption-policy fcfs \
    --fitness-weights "throughput:0.4,p99_ttft:0.3,p99_e2e:0.3"
  ```
- **Output format:** JSON metrics between `=== Simulation Metrics ===` markers. With `--fitness-weights`, `Score: <float>` follows in `=== Fitness Evaluation ===` section.
- **Baseline result:** 70B H100 TP=8/1inst rate=500 Score=0.091801. Wall time ~0.27s per eval.

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
- `sim/latency/roofline.go:291-351` — rooflineStepTime: physics behind TP-dependent latency.
- `sim/latency/trained_physics_model.go:263-284` — TP communication overhead formula.
- `model_configs/llama-3.1-70b-instruct/config.json` — 70B architecture: 80 layers, hidden=8192, intermediate=28672, 64 attn heads, 8 KV heads.
- `model_configs/qwen3-32b/config.json` — qwen3-32b: 64 layers, hidden=5120, intermediate=25600, 64 heads, 8 KV heads.
- `model_configs/qwen3-14b/config.json` — qwen3-14b: 40 layers, hidden=5120, intermediate=17408, 40 heads, 8 KV heads.
- `model_configs/llama-3.1-8b-instruct/config.json` — llama-3.1-8b: 32 layers, hidden=4096, intermediate=14336, 32 heads, 8 KV heads.

## Code Targets

The experiment modifies the Python search script. Start from `search_blis_iter8.py` in the project root:

- **`search_blis_iter9.py`** (copy from search_blis_iter8.py):
  1. **Update GLOBAL_BEST_TABLE** with entries discovered in iter-9 probes:
     - Add: `("llama-3.1-70b", "H100", 100, 1000): 0.088793`
     - Add: `("llama-3.1-70b", "H100", 500, 1000): 0.091801`
     - Add: `("llama-3.1-70b", "A100-SXM", 100, 1000): 0.063273`
     - Add: `("llama-3.1-70b", "A100-SXM", 500, 1000): 0.063991`
     - Add: `("llama-3.1-70b", "L40S", 100, 1000): 0.034041`
     - Add: `("llama-3.1-70b", "L40S", 500, 1000): 0.034137`
  2. **Add `MODEL_PARAMS` dict** near `HW_BANDWIDTH`:
     ```python
     MODEL_PARAMS = {
         "llama-3.1-8b": 8,
         "qwen3-14b": 14,
         "qwen3-32b": 32,
         "llama-3.1-70b": 70,
     }
     ```
  3. **Add `run_model_aware_predict()` function** (new, after `run_formula_predict()`):
     - Compute `load_index = MODEL_PARAMS[model_shortname] / HW_BANDWIDTH[hardware]`
     - If `load_index > 12.0`: predict TP=8/1inst (no crossover regime)
     - Else: use bandwidth formula `crossover = 400.0 / hw_bandwidth`, predict TP=4/2inst if rate > crossover, else TP=8/1inst
     - Apply max-profile secondaries (same as formula-predict)
     - Run one BLIS eval
     - Return result dict with: `strategy="model-aware-predict"`, `load_index`, `load_index_threshold=12.0`, `no_crossover_regime=(load_index > 12.0)`, plus all fields from formula-predict
  4. **Wire up `model-aware-predict`** in the `--strategy` argparse choice list and dispatch in the main strategy switch (same pattern as `formula-predict`).
  5. **No changes to lean bracket or Phase 1/2 logic** — those are tested via existing flags.

## What I Tried That Didn't Work

- **`--hardware h100_sxm`**: Fatal error. Correct name is `H100` (case-sensitive, no suffix). [Carried from iter-1]
- **`--hardware A100`**: Fatal error. Must be `A100-SXM` or `A100-80`. [Carried from iter-5]
- **Parsing fitness from `--metrics-path` JSON**: The JSON file does not contain the fitness score. Must parse stdout. [Carried from iter-1]
- **Running blis from subdirectory**: The blis binary must be run from the project root `/Users/jchen/go/src/inference-sim/inference-sim`. [Carried from iter-3]
- **grep "Score:" with awk in for-loop subshell**: When the working directory isn't the project root, `./blis` fails silently and score is empty. Always cd to project root first. [Carried from iter-3]
- **Assuming TP=8 dominance is unconditional on batch size**: Requires max_running >= 256. With mr<256, TP=4/2inst wins even at rate=100. [Carried, RP-13]
- **Assuming round-robin always beats least-loaded**: The routing advantage is seed-dependent. On A100-SXM at rate=200, weighted routing actually beats round-robin. [Carried from iter-7]
- **Assuming rate crossover is H100-specific**: It's universal — A100-SXM crosses at ~200, L40S llama at ~450. [Carried from iter-6]
- **Assuming TP=2 might eventually win at extreme rates**: Probed at rate=1000, 2000, 5000 on H100 qwen — TP=2 gap is constant at ~11%, never converging. [Carried from iter-7]
- **Assuming crossover rate is model-size independent**: The `400/BW` formula (RP-27) works for 8-14B models but fails for 32B. [Carried from iter-8]
- **Micro-eval (≤500 requests) for TP winner identification**: Even at 500 requests on H100 qwen rate=500, TP=8 wins (should be TP=4). Queue saturation effects need ≥1000 requests. [Carried from iter-8]
- **Assuming qwen3-32b has a crossover on A100-SXM**: Probed rates 100-2000 — TP=8 always wins. No crossover exists for 32B on A100-SXM. [Carried from iter-8]
- **70B TP=1**: Fatal OOM — model overhead 137.07 GiB > 72 GiB available at TP=1. [New in iter-9]
- **70B TP=2**: Technically runs but scores 0.012899 (6.9× worse than TP=8). Catastrophically bad, not a viable TP level. [New in iter-9]
- **Looking for 70B TP=4 crossover at extreme rates**: Probed rate=1000 and 2000 on H100 — TP=8 still wins with 9-10% margin. Gap doesn't narrow. [New in iter-9]

## What I Excluded and Why

- **TP=2/4inst for 70B in lean bracket**: TP=2 scores 6.9× worse than TP=8. Including it would waste 1 eval per search. Lean bracket's TP={4,8} already covers the only competitive pair. [Infeasible — catastrophic performance]
- **Higher rates for 70B crossover search**: Probed up to rate=2000 on H100 — TP=8 margin stays at 9-10%. The gap doesn't narrow at all, suggesting no crossover exists at any practical rate. [Exhaustive probing — no signal of convergence]
- **qwen2.5-7b-instruct testing**: A ~7B model exists in model_configs but hasn't been tested in the campaign. It would have similar behavior to llama-3.1-8b (similar param count) and doesn't add model-size diversity. [Redundant — param count too similar to 8B]
- **Phase 2 TPE for 70B**: All 70B regimes have TP=8/1inst winner where routing is degenerate (single instance). Phase 2 can only optimize scheduler/batch params, which are already near-optimal with max-profile. Prior iterations show ≤0.71% improvement from Phase 2. [Degenerate routing — Phase 2 cannot help]
- **Intermediate rates (200-400) for 70B on H100**: Would fill in the TP margin curve but won't change lean bracket behavior — TP=8 wins at both 100 and 500, so intermediate rates are guaranteed. [Monotonic — no experimental value]
- **load_index threshold sensitivity analysis**: Threshold=12 cleanly separates crossover (≤9.6) from no-crossover (≥15.7) regimes. Testing threshold values between 10-15 would all give 24/24. The exact value matters less than the mechanism. [Gap is wide — any threshold 10-15 works identically]
- **codellama-34b and yi-34b models**: Available in model_configs but not in the tested model family. Campaign focuses on llama and qwen families where 8B→70B scaling is cleanly characterized. [Out of scope — different model families]

## Evolution of Thinking

The campaign's narrative arc is now clear: iter-1 discovered TP dominance, iter-2-4 built the search algorithm, iter-5-8 validated it across models and hardware, and iter-9 extends to the largest practical model (70B on 8 GPUs). The key insight from 70B probing is that the TP={4,8} choice is even simpler for large models — TP=8 always wins, with no crossover to navigate. This is the opposite of the 8-14B regime where the rate-dependent crossover is the main complexity.

The model-aware formula arose from observing a clean pattern in the accumulated data: the load_index (params/BW) perfectly separates "crossover exists" from "no crossover" regimes. The threshold=12 was derived empirically from the gap between the highest crossover-regime (9.6 for 32B/H100) and the lowest no-crossover-regime (15.7 for 32B/A100-SXM). This single parameter addition turns the formula from 75% to 100% accurate on the test grid.

The practical conclusion: lean bracket is the recommended search algorithm for ANY model size (8B-70B) on ANY hardware (H100, A100-SXM, L40S). For very large models, it's even simpler — TP=8 always wins, so the search is effectively 2 evals confirming the obvious. The model-aware formula could serve as a zero-eval heuristic for TP prediction, with lean bracket as the 2-eval verification step.

## Current Status

- **Validated:** (1) 70B TP=8 dominates on all 3 hardware at rates 100-2000. (2) TP=1 infeasible, TP=2 catastrophically bad for 70B. (3) Model-aware formula (load_index threshold=12) achieves 24/24 on the test grid. (4) Default-profile preserves TP=8 winner for 70B on H100 rate=500. (5) Multi-blis-seed robustness for 70B confirmed on H100 and A100-SXM (5 seeds each). (6) 70B eval wall time ~0.25-0.27s (lean bracket ~0.5s).
- **Uncertain:** (1) Whether default-profile lean bracket preserves TP=8 on A100-SXM and L40S for 70B (only probed H100). (2) Whether load_index threshold=12 generalizes to models beyond the 4 tested (e.g., codellama-34b, yi-34b). (3) Whether the model-aware formula works at intermediate rates (200-400) for 32B/H100 where the old formula would fail. (4) Whether even larger models (405B, requiring multi-node) would need different search strategies.
- **Suggested next:** (1) **Production `blis search` subcommand**: Package lean bracket as a native Go CLI command with auto-detection of model configs, hardware, and model-aware TP prediction. (2) **Workload diversity testing**: Test lean bracket with different workload distributions (shorter/longer prompts, bursty arrivals) to verify TP winner stability. (3) **Multi-objective Pareto re-evaluation**: RP-7 showed degenerate Pareto fronts — re-check with model-aware formula to see if the Pareto front is non-trivial for regimes near the crossover boundary. (4) **Model-size-parameterized crossover formula**: Fit crossover_rate(model_size, bandwidth) using all accumulated data to provide a predicted crossover rate (not just binary exists/doesn't-exist).

## Warnings & Constraints

- **Stderr contains warnings**: BLIS prints timing and config warnings to stderr. Always use `2>/dev/null` or capture stderr separately when parsing stdout. [Carried from iter-2]
- **Seed affects workload, not config evaluation**: `--seed 42` controls random request generation. For the search algorithm's own randomness, use a separate seed. [Carried from iter-2]
- **TP=0 is invalid**: Always set `--tp` explicitly. [Carried from iter-2]
- **Fitness score format on stdout**: `Score: 0.019249` — parse with regex `r'Score:\s+([\d.]+)'`. [Carried from iter-2]
- **Multi-instance JSON output**: When `--num-instances > 1`, stdout contains multiple JSON blocks. Always parse the block with `"instance_id": "cluster"`. [Carried from iter-2]
- **Hardware name is case-sensitive**: `H100` not `h100`, `A100-SXM` not `a100-sxm`. [Carried]
- **Global best is (model, hardware, rate, num_requests)-specific**: New entries added for llama-3.1-70b (all hardware × 2 rates). [Updated iter-9]
- **Bracket profiles are deterministic**: They don't depend on search_seed (no RNG involved). All search seeds produce identical Phase 1 results per blis_seed. [Carried]
- **Profile ordering affects evals_to_best**: In lean bracket, TP=4 is evaluated first (position 1), TP=8 second (position 2). [Carried from iter-7]
- **Model name format**: Use full model paths: `qwen/qwen3-14b`, `qwen/qwen3-32b`, `meta-llama/llama-3.1-8b-instruct`, `meta-llama/llama-3.1-70b-instruct`. [Updated: added 70B]
- **Single-instance configs have degenerate routing**: At TP=8/1inst, routing_policy and block_size have no effect. [Carried from iter-3]
- **Must run blis from project root**: The binary expects model configs relative to the project directory. [Carried from iter-3]
- **Routing policy effect is seed-dependent on multi-instance configs**: round-robin vs least-loaded vs weighted varies by workload realization. [Carried]
- **Rate-dependent TP winner on H100 (qwen3-14b)**: TP=8 wins at rate<=125, TP=4 wins at rate>=150. [Carried from iter-5]
- **Rate-dependent TP winner on A100-SXM (qwen3-14b)**: TP=8 wins at rate<=150, tied at rate=200, TP=4 wins at rate>=500. [Carried from iter-6]
- **L40S qwen3-14b has no crossover**: TP=8 always wins at rates 100-500+. [Carried from iter-8]
- **L40S llama-3.1-8b crossover ~400-500**: TP=8 at rate=100, TP=4 at rate=500. [Carried from iter-8]
- **qwen3-32b H100 crossover ~350**: TP=8 at rate<=300, TP=4 at rate>=350 (with non-monotonicity at rate=450). [Carried from iter-8]
- **qwen3-32b A100-SXM and L40S: no crossover**: TP=8 always wins, even at rate=2000. [Carried from iter-8]
- **RP-27 formula (`400/BW`) fails for 32B+ models**: Predicts crossover at ~119 on H100 for all models, actual is ~350 for 32B and non-existent for 70B. [Updated iter-9]
- **Micro-eval requires ≥1000 requests**: At 500 requests, TP=8 wins where 1000 requests gives TP=4. Queue saturation effects need sustained load. [Carried from iter-8]
- **qwen3-32b eval wall time**: ~0.28s per eval on H100 at 1000 requests (similar to 14B). [Carried from iter-8]
- **Non-monotonic TP winner for qwen3-32b H100**: TP=4 at rate=350, TP=8 at rate=450, TP=4 at rate=500. The crossover band is not clean. [Carried from iter-8]
- **Default-profile masks rate effect for small models**: With mr=256, TP=4/2inst wins at ALL rates for 14B. The rate-dependent TP=8 advantage only appears with max_running>=256 (max-profile). [Carried from iter-5]
- **70B TP=1 is infeasible**: Fatal OOM (137 GiB > 72 GiB). Don't include in search. [New in iter-9]
- **70B TP=2 is catastrophically bad**: Score 0.012899 (6.9× worse than TP=8). Don't include in search. [New in iter-9]
- **70B has NO crossover on any hardware**: TP=8 wins at all rates (100-2000) on H100, A100-SXM, L40S. Margins: H100 9-10%, A100 7-14%, L40S 18-20%. [New in iter-9]
- **Model-aware formula load_index boundary**: Crossover exists at load_index ≤9.6, doesn't exist at load_index ≥15.7. Threshold=12 works for all tested cases. Gap is wide enough that any threshold 10-15 gives identical results. [New in iter-9]
- **Default-profile for 70B**: Preserves TP=8 winner on H100 rate=500 (margin 4.1%). Only min-profile (mr=32) inverts ranking. [New in iter-9]
- **70B multi-seed variation**: H100 rate=500 scores range 0.091801-0.095769 across seeds 42-46. Min TP=8 margin is 2.5% (seed 45). [New in iter-9]
