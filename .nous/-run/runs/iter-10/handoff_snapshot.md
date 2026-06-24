# Handoff — Iteration 10: Workload-Shape Sensitivity of Lean Bracket

## Goal

Test whether lean bracket (TP={4,8}, max-profile, budget=2) correctly identifies the TP winner when the workload token distribution changes from the default (prompt mean=512) to named presets with 2-20× different prompt token distributions. Simultaneously, show that the model-aware formula (RP-27 + load_index threshold) fails on long-prompt workloads because it has no input for prompt token distribution.

## Key Discoveries

1. **Prompt token distribution shifts the TP crossover**: On H100 qwen3-14b, workloads with prompt mean ≥ 1024 (contentgen, summarization, multidoc) have TP=4/2inst winning at ALL rates including rate=100. The rate-dependent crossover (RP-24, ~125 req/s on H100) only applies to short-prompt workloads (chatbot avg=256, distribution avg=512). The crossover rate from prior iterations is workload-conditional, not universal.

2. **Lean bracket is workload-agnostic by construction**: Since it evaluates both TP=4 and TP=8, it always picks the correct winner regardless of workload shape. No algorithm modification needed.

3. **Model-aware formula fails on 3/8 H100 workload regimes**: At rate=100, the formula predicts TP=8 (rate < crossover=119.4) but actual winner is TP=4 for summarization (margin 2.9%), contentgen (margin 9.1%), and multidoc (margin 2.4%). Formula accuracy drops from 100% (default workload) to 62.5% (workload presets) on H100.

4. **Multidoc on A100-SXM shows anomalous TP=8 dominance**: TP=8 wins by 109-125% margin. Very long prompts (avg 10240 tokens) likely hit KV cache capacity constraints at TP=4/2inst on the lower-bandwidth hardware. On H100 (higher bandwidth), TP=4 wins for multidoc.

5. **Chatbot on A100-SXM rate=500 is near-tied**: TP=8 wins by only 0.18% (0.234138 vs 0.233715). Both TP levels exceed the 1% threshold, so lean bracket's evals_to_best=1 regardless of winner.

6. **Workload presets are defined in `defaults.yaml:152-191`**: chatbot (prompt=256), contentgen (prompt=1024), summarization (prompt=4096), multidoc (prompt=10240). The `--workload` flag (`cmd/root.go:2018`) selects the preset.

## System Interface

- **Build:** `go build -o blis main.go` (binary already exists in project root)
- **Run baseline (contentgen, H100, rate=500, TP=4 lean bracket winner):**
  ```bash
  ./blis run --model qwen/qwen3-14b --hardware H100 --latency-model trained-physics \
    --num-requests 1000 --rate 500 --seed 42 --tp 4 --num-instances 2 \
    --scheduler fcfs --max-num-running-reqs 512 --max-num-scheduled-tokens 8192 \
    --long-prefill-token-threshold 4096 --block-size-in-tokens 16 \
    --routing-policy least-loaded --admission-policy always-admit \
    --preemption-policy fcfs --workload contentgen \
    --fitness-weights "throughput:0.4,p99_ttft:0.3,p99_e2e:0.3"
  ```
- **Output format:** JSON metrics between `=== Simulation Metrics ===` markers. With `--fitness-weights`, `Score: <float>` follows in `=== Fitness Evaluation ===` section.
- **Baseline result:** contentgen H100 TP=4/2inst rate=500 Score=0.104103. Wall time ~0.25s per eval.

## Code Map

- `cmd/root.go:2018` — `--workload` flag definition. Valid values: chatbot, summarization, contentgen, multidoc, distribution. Default: distribution.
- `cmd/root.go:1315-1326` — Workload preset loading from defaults.yaml. Check here if preset isn't being applied.
- `defaults.yaml:152-191` — Workload preset definitions (prompt/output token distributions). Check here for exact token mean/stdev/min/max values.
- `cmd/root.go:947-984` — All swept parameter flag definitions. Check here if a flag name or default is wrong.
- `cmd/root.go:956` — `--hardware` flag definition. Valid values: H100, A100-SXM, A100-80, L40S.
- `cmd/root.go:938` — `--seed` flag definition (Int64, default 42). Controls workload generation RNG.
- `cmd/root.go:940` — `--rate` flag definition (Float64). Arrival rate in req/s.
- `cmd/root.go:1761-1765` — Fitness weight parsing and ComputeFitness call.
- `sim/latency/config.go:86-101` — `GetHWConfig()` validates hardware names.
- `hardware_config.json` — GPU specs: H100 (989.5 TFLOPS, 80 GiB, 3.35 TB/s), A100-SXM (312 TFLOPS, 80 GiB, 2.039 TB/s), L40S (362 TFLOPS, 48 GiB, 0.864 TB/s).
- `sim/cluster/metrics.go:418-498` — `validFitnessKeysList()`, `ComputeFitness()`, `extractMetric()`.
- `sim/cluster/metrics.go:420-426` — The 8 valid fitness keys.
- `sim/cluster/metrics.go:428-435` — Reference scale constants for normalization.
- `sim/metrics_utils.go:57-87` — MetricsOutput struct: all JSON fields in the metrics output.
- `sim/latency/roofline.go:291-351` — rooflineStepTime: physics behind TP-dependent latency.
- `sim/latency/trained_physics_model.go:263-284` — TP communication overhead formula.
- `model_configs/qwen3-14b/config.json` — qwen3-14b: 40 layers, hidden=5120, intermediate=17408, 40 heads, 8 KV heads.
- `model_configs/llama-3.1-8b-instruct/config.json` — llama-3.1-8b: 32 layers, hidden=4096, intermediate=14336, 32 heads, 8 KV heads.
- `model_configs/qwen3-32b/config.json` — qwen3-32b: 64 layers, hidden=5120, intermediate=25600, 64 heads, 8 KV heads.
- `model_configs/llama-3.1-70b-instruct/config.json` — llama-3.1-70b: 80 layers, hidden=8192, intermediate=28672, 64 attn heads, 8 KV heads.

## Code Targets

The experiment modifies the Python search script. Start from `search_blis_iter8.py` in the project root:

- **`search_blis_iter10.py`** (copy from search_blis_iter8.py):
  1. **Add `--workload` CLI argument** (default "distribution") to argparse. Store as `args.workload`.
  2. **Update `build_blis_cmd()`** to accept a `workload` parameter and add `--workload <value>` to the BLIS command when not "distribution".
  3. **Extend GLOBAL_BEST_TABLE key** from `(model, hardware, rate, num_requests)` to `(model, hardware, rate, num_requests, workload)`. All existing entries get `workload="distribution"`. New entries:
     ```python
     # H100 workload presets (qwen3-14b)
     ("qwen3-14b", "H100", 100, 1000, "chatbot"):        0.200393,
     ("qwen3-14b", "H100", 500, 1000, "chatbot"):        0.268650,
     ("qwen3-14b", "H100", 100, 1000, "contentgen"):     0.099677,
     ("qwen3-14b", "H100", 500, 1000, "contentgen"):     0.104103,
     ("qwen3-14b", "H100", 100, 1000, "summarization"):  0.085079,
     ("qwen3-14b", "H100", 500, 1000, "summarization"):  0.085323,
     ("qwen3-14b", "H100", 100, 1000, "multidoc"):       0.016893,
     ("qwen3-14b", "H100", 500, 1000, "multidoc"):       0.016903,
     # A100-SXM workload presets (qwen3-14b)
     ("qwen3-14b", "A100-SXM", 100, 1000, "chatbot"):        0.193078,
     ("qwen3-14b", "A100-SXM", 500, 1000, "chatbot"):        0.234138,
     ("qwen3-14b", "A100-SXM", 100, 1000, "contentgen"):     0.072420,
     ("qwen3-14b", "A100-SXM", 500, 1000, "contentgen"):     0.070609,
     ("qwen3-14b", "A100-SXM", 100, 1000, "summarization"):  0.056095,
     ("qwen3-14b", "A100-SXM", 500, 1000, "summarization"):  0.055265,
     ("qwen3-14b", "A100-SXM", 100, 1000, "multidoc"):       0.007025,
     ("qwen3-14b", "A100-SXM", 500, 1000, "multidoc"):       0.007118,
     # 70B distribution entries (from iter-9)
     ("llama-3.1-70b", "H100", 100, 1000, "distribution"):     0.088793,
     ("llama-3.1-70b", "H100", 500, 1000, "distribution"):     0.091801,
     ("llama-3.1-70b", "A100-SXM", 100, 1000, "distribution"): 0.063273,
     ("llama-3.1-70b", "A100-SXM", 500, 1000, "distribution"): 0.063991,
     ("llama-3.1-70b", "L40S", 100, 1000, "distribution"):     0.034041,
     ("llama-3.1-70b", "L40S", 500, 1000, "distribution"):     0.034137,
     ```
  4. **Update all GLOBAL_BEST_TABLE lookups** to include `workload` as the 5th key element. The lookup in main() and in `run_formula_predict()` must pass the workload parameter.
  5. **Add `MODEL_PARAMS` dict** near `HW_BANDWIDTH`:
     ```python
     MODEL_PARAMS = {
         "llama-3.1-8b": 8,
         "qwen3-14b": 14,
         "qwen3-32b": 32,
         "llama-3.1-70b": 70,
     }
     ```
  6. **Add `run_model_aware_predict()` function** (new, after `run_formula_predict()`):
     - Compute `load_index = MODEL_PARAMS[model_shortname] / HW_BANDWIDTH[hardware]`
     - If `load_index > 12.0`: predict TP=8/1inst (no crossover regime)
     - Else: use bandwidth formula `crossover = 400.0 / hw_bandwidth`, predict TP=4/2inst if rate > crossover, else TP=8/1inst
     - Apply max-profile secondaries (same as formula-predict)
     - Run one BLIS eval
     - Return result dict with: `strategy="model-aware-predict"`, `load_index`, `load_index_threshold=12.0`, `no_crossover_regime=(load_index > 12.0)`, plus all fields from formula-predict
  7. **Wire up `model-aware-predict`** in the `--strategy` argparse choice list and dispatch in the main strategy switch (same pattern as `formula-predict`).
  8. **Pass workload through all run functions** — `run_blis()`, `run_formula_predict()`, `run_model_aware_predict()`, and the adaptive-hierarchical path must all propagate the workload to `build_blis_cmd()`.

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
- **70B TP=1**: Fatal OOM — model overhead 137.07 GiB > 72 GiB available at TP=1. [Carried from iter-9]
- **70B TP=2**: Technically runs but scores 0.012899 (6.9× worse than TP=8). Catastrophically bad. [Carried from iter-9]
- **Looking for 70B TP=4 crossover at extreme rates**: Probed rate=1000 and 2000 on H100 — TP=8 still wins with 9-10% margin. [Carried from iter-9]
- **Assuming rate-dependent TP crossover applies to all workloads**: Probed chatbot, contentgen, summarization, multidoc. With prompt mean ≥ 1024, TP=4/2inst wins at rate=100 on H100, contradicting the crossover at ~125 found for default workload. [New in iter-10]
- **Assuming model-aware formula is workload-agnostic**: Formula predicts TP=8 at rate=100 on H100 for all workloads (rate < crossover=119.4). But actual TP=4 wins for 3/4 non-chatbot workloads. Formula accuracy drops from 100% to 62.5% on workload presets. [New in iter-10]

## What I Excluded and Why

- **L40S workload preset testing**: L40S has the most extreme TP=8 dominance for qwen3-14b (no crossover at any rate). Workload shape is unlikely to flip the winner. Would confirm robustness but add 8 more regimes with predicted identical outcome. [Low experimental value — monotonic TP landscape]
- **Other models (llama-3.1-8b, qwen3-32b, 70B) with workload presets**: The workload-shape effect is model-agnostic in mechanism (it's about prefill compute, not model architecture). Testing one model (qwen3-14b) is sufficient to validate the principle. [Would multiply regimes by 4 without new mechanism insight]
- **Custom prompt token distributions**: Could sweep prompt_tokens_mean from 128 to 16384 to find the exact crossover prompt length. This would be interesting but adds complexity — the named presets already demonstrate the effect. [Deferring to next iteration if needed]
- **Bursty arrival processes (gamma, weibull)**: The --workload flag controls token distribution, not arrival process. Testing different arrival distributions requires workload spec YAML files. The TP crossover mechanism is prefill-compute-driven, not arrival-driven. [Orthogonal dimension — future work]
- **Phase 2 TPE for workload presets**: Prior iterations show Phase 2 adds ≤ 0.71% for default workload. With workload presets, the single-instance configs (TP=8) have degenerate routing, and secondary params are already at max-profile values. Phase 2 is unlikely to improve scores. [Low expected value — degenerate for 7/8 winning configs]
- **Workload-aware formula extension**: Could extend the model-aware formula with a prompt_tokens_mean parameter to fix the failures. This is a design task, not an experimental hypothesis — deferring to next iteration. [Design work, not hypothesis testing]

## Evolution of Thinking

Iterations 1-9 systematically validated lean bracket across 4 models × 3 hardware × 2 rates, always using the default workload (prompt mean=512, output mean=512). The implicit assumption was that the TP crossover behavior is workload-independent. Iteration 10 tests this assumption directly.

The probing revealed a clean gradient: as prompt token mean increases (chatbot=256 → distribution=512 → contentgen=1024 → summarization=4096 → multidoc=10240), the TP=4 advantage grows and the crossover rate shifts lower. At prompt mean ≥ 1024, there is no crossover — TP=4 wins at all rates on H100. This means the "crossover rate" from RP-24 is actually a function of three variables: bandwidth, model size, AND prompt token distribution.

The practical implication: lean bracket is more robust than the formula because it directly evaluates instead of predicting. The formula needs a third parameter (workload prompt length) to recover accuracy, but the lean bracket needs nothing — it's already workload-agnostic.

The multidoc anomaly on A100-SXM (TP=8 winning by >100% margin) reveals a second mechanism: KV cache capacity constraints. Very long prompts (10240 avg, 20480 max) create massive KV cache requirements that overwhelm TP=4/2inst on lower-memory-effective hardware. This is qualitatively different from the prefill compute mechanism and suggests that for extreme workloads, the TP choice is driven by memory, not compute.

## Current Status

- **Validated (probes):** (1) TP winner is workload-dependent on H100 — prompt mean ≥ 1024 → TP=4 at all rates. (2) Lean bracket identifies correct winner on all 20 probed regimes (5 workloads × 2 rates × 2 hardware). (3) Model-aware formula fails on 3/8 H100 workload regimes and 4/8 A100-SXM workload regimes. (4) Multidoc on A100-SXM shows anomalous TP=8 dominance (KV cache capacity mechanism). (5) Chatbot on A100-SXM rate=500 is near-tied (0.18% margin).
- **Uncertain:** (1) Whether the prompt-length crossover point is stable across BLIS seeds. (2) Whether secondary params (scheduler, routing) matter more for long-prompt workloads than for default. (3) Whether multidoc anomaly on A100-SXM is due to KV cache capacity or some other mechanism (need to check metrics for dropped requests). (4) Exact prompt token threshold where TP crossover disappears on H100 (between 512 and 1024).
- **Suggested next:** (1) **Workload-aware formula**: Extend model-aware formula with prompt_tokens_mean parameter — compute an effective crossover rate that shifts with prompt length. (2) **Custom distribution sweep**: Sweep prompt_tokens_mean from 128-8192 on H100 to map the exact prompt-length vs crossover-rate surface. (3) **Multi-seed workload robustness**: Verify TP winner stability across blis_seeds for the workload-shifted regimes. (4) **Production blis search**: Package lean bracket with automatic workload detection as a native Go CLI command.

## Warnings & Constraints

- **Stderr contains warnings**: BLIS prints timing and config warnings to stderr. Always use `2>/dev/null` or capture stderr separately when parsing stdout. [Carried from iter-2]
- **Seed affects workload, not config evaluation**: `--seed 42` controls random request generation. For the search algorithm's own randomness, use a separate seed. [Carried from iter-2]
- **TP=0 is invalid**: Always set `--tp` explicitly. [Carried from iter-2]
- **Fitness score format on stdout**: `Score: 0.019249` — parse with regex `r'Score:\s+([\d.]+)'`. [Carried from iter-2]
- **Multi-instance JSON output**: When `--num-instances > 1`, stdout contains multiple JSON blocks. Always parse the block with `"instance_id": "cluster"`. [Carried from iter-2]
- **Hardware name is case-sensitive**: `H100` not `h100`, `A100-SXM` not `a100-sxm`. [Carried]
- **Bracket profiles are deterministic**: They don't depend on search_seed (no RNG involved). All search seeds produce identical Phase 1 results per blis_seed. [Carried]
- **Profile ordering affects evals_to_best**: In lean bracket, TP=4 is evaluated first (position 1), TP=8 second (position 2). [Carried from iter-7]
- **Model name format**: Use full model paths: `qwen/qwen3-14b`, `qwen/qwen3-32b`, `meta-llama/llama-3.1-8b-instruct`, `meta-llama/llama-3.1-70b-instruct`.
- **Single-instance configs have degenerate routing**: At TP=8/1inst, routing_policy and block_size have no effect. [Carried from iter-3]
- **Must run blis from project root**: The binary expects model configs relative to the project directory. [Carried from iter-3]
- **Micro-eval requires ≥1000 requests**: At 500 requests, TP=8 wins where 1000 requests gives TP=4. [Carried from iter-8]
- **Global best is now (model, hardware, rate, num_requests, workload)-specific**: Extended from 4-tuple to 5-tuple in iter-10. [Updated iter-10]
- **--workload flag selects preset from defaults.yaml**: Valid values: chatbot, summarization, contentgen, multidoc, distribution. Default is "distribution". [New in iter-10]
- **Workload presets override individual token flags**: When --workload is set to a preset name, the token distribution from defaults.yaml is used regardless of --prompt-tokens etc. [New in iter-10]
- **Multidoc scores are very low (0.003-0.017)**: The very long prompts (avg 10240) create an extreme workload regime. Scores may be near the floor where small absolute differences look like large relative margins. [New in iter-10]
- **Chatbot on A100-SXM rate=500 is near-tied (0.18%)**: Both TP levels exceed the 1% threshold. The winner may flip with different blis_seeds. [New in iter-10]
- **Model-aware formula crossover for qwen3-14b on H100**: crossover_rate = 400/3.35 = 119.4. On A100-SXM: 400/2.039 = 196.2. These only apply to default/chatbot workloads (prompt mean ≤ 512). [Updated iter-10]
- **Prompt-length TP crossover threshold**: On H100, TP=4 wins at all rates when prompt mean ≥ 1024. The exact threshold is between 512 (distribution, crossover exists) and 1024 (contentgen, no crossover). [New in iter-10]
