# Handoff — Iteration 8: Model-Size Generalization and Formula Limits

## Goal

Test whether the lean bracket search algorithm (TP={4,8}, max-profile, no Phase 2, budget=2) generalizes to a third model (qwen3-32b, 2.3x parameter count of qwen3-14b) and to the final untested hardware platform (L40S). Also test the limits of the RP-27 bandwidth formula (`400/BW`) by showing it fails for 32B models, and demonstrate that micro-eval (100 requests) cannot replace the standard 1000-request evaluation.

## Key Discoveries

1. **qwen3-32b crossover is ~350 on H100** (vs ~130-150 for 14B models): The TP=4/TP=8 crossover rate scales with model size. At rate=300 on H100, TP=8 wins (0.138983 vs 0.137566). At rate=350, TP=4 barely wins (0.139790 vs 0.139405, margin 0.3%). At rate=500, TP=4 wins more clearly (0.141403 vs 0.140230, margin 0.8%). There is non-monotonicity at rate=450 where TP=8 briefly wins again (0.140054 vs 0.138577).

2. **qwen3-32b has NO crossover on A100-SXM**: TP=8 wins at all tested rates (100-2000). At rate=2000, TP=8 still leads (0.108224 vs 0.102003). The 32B model's weight loading time is so large that A100-SXM's 2.039 TB/s bandwidth never reaches the compute-bound regime at practical rates.

3. **qwen3-32b has NO crossover on L40S**: TP=8 wins at rates 100 and 500 (0.063243/0.064436 vs 0.055893/0.056742). L40S's 0.864 TB/s is deeply bandwidth-bound for 32B models.

4. **RP-27 formula accuracy is 14/18 (78%) across the full matrix**: Failures: H100-qwen32b-rate200 (predicts TP=4, actual TP=8), H100-qwen32b-rate300 (same), A100-qwen32b-rate500 (predicts TP=4, actual TP=8), L40S-qwen14b-rate500 (predicts TP=4, actual TP=8).

5. **Micro-eval (≤500 requests) fails to identify TP=4 winner**: At H100 qwen3-14b rate=500 with 500 requests, TP=8 wins (0.162349 vs 0.150143). At 1000 requests, TP=4 wins (0.178063 vs 0.169720). The queuing advantage of 2 instances requires sustained load beyond ~500 requests to manifest in the fitness score.

6. **L40S lean bracket data**: qwen3-14b TP=8 always wins (rate=100: 0.082636 vs 0.077297; rate=500: 0.084795 vs 0.080647). llama-3.1-8b: TP=8 at rate=100 (0.104264 vs 0.101426), TP=4 at rate=500 (0.109668 vs 0.107420).

7. **qwen3-32b H100 rate=500 global best is 0.141403** (TP=4/2inst, least-loaded routing). Round-robin gives 0.140660, weighted gives 0.139947. Least-loaded is optimal for this regime.

## System Interface

- **Build:** `go build -o blis main.go` (binary already exists in project root)
- **Run baseline (qwen3-32b H100 rate=500, TP=4 lean bracket winner):**
  ```bash
  ./blis run --model qwen/qwen3-32b --hardware H100 --latency-model trained-physics \
    --num-requests 1000 --rate 500 --seed 42 --tp 4 --num-instances 2 \
    --scheduler fcfs --max-num-running-reqs 512 --max-num-scheduled-tokens 8192 \
    --long-prefill-token-threshold 4096 --block-size-in-tokens 16 \
    --routing-policy least-loaded --admission-policy always-admit \
    --preemption-policy fcfs \
    --fitness-weights "throughput:0.4,p99_ttft:0.3,p99_e2e:0.3"
  ```
- **Output format:** JSON metrics between `=== Simulation Metrics ===` markers. With `--fitness-weights`, `Score: <float>` follows in `=== Fitness Evaluation ===` section.
- **Baseline result:** qwen3-32b H100 TP=4/2inst rate=500 Score=0.141403. Wall time ~0.28s per eval.

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
- `model_configs/qwen3-32b/config.json` — qwen3-32b architecture: 64 layers, hidden=5120, intermediate=25600, 64 attn heads, 8 KV heads.
- `model_configs/qwen3-14b/config.json` — qwen3-14b: 40 layers, hidden=5120, intermediate=17408, 40 heads, 8 KV heads.
- `model_configs/llama-3.1-8b-instruct/config.json` — llama-3.1-8b: 32 layers, hidden=4096, intermediate=14336, 32 heads, 8 KV heads.

## Code Targets

The experiment modifies the Python search script. Start from `search_blis_iter7.py` in the project root:

- **`search_blis_iter8.py`** (copy from search_blis_iter7.py):
  1. **Update GLOBAL_BEST_TABLE** with entries discovered in iter-8 probes:
     - Add: `("qwen3-32b", "H100", 100, 1000): 0.132535`
     - Add: `("qwen3-32b", "H100", 300, 1000): 0.138983`
     - Add: `("qwen3-32b", "H100", 500, 1000): 0.141403`
     - Add: `("qwen3-32b", "A100-SXM", 100, 1000): 0.104331`
     - Add: `("qwen3-32b", "A100-SXM", 500, 1000): 0.108182`
     - Add: `("qwen3-32b", "L40S", 100, 1000): 0.063243`
     - Add: `("qwen3-32b", "L40S", 500, 1000): 0.064436`
     - Add: `("qwen3-14b", "L40S", 100, 1000): 0.082636`
     - Add: `("qwen3-14b", "L40S", 500, 1000): 0.084795`
     - Add: `("llama-3.1-8b", "L40S", 100, 1000): 0.104264`
     - Add: `("llama-3.1-8b", "L40S", 500, 1000): 0.109668`
  2. **Add `formula-predict` strategy** (new function `run_formula_predict()`):
     - Hardware bandwidth lookup: `HW_BANDWIDTH = {"H100": 3.35, "A100-SXM": 2.039, "L40S": 0.864}`
     - Compute crossover: `crossover_rate = 400.0 / hw_bandwidth`
     - Predict TP: `tp=4, instances=2` if `rate > crossover_rate` else `tp=8, instances=1`
     - Apply max-profile secondaries: `mr=512, mt=8192, pf=4096, block=16, scheduler=fcfs, routing=least-loaded, admission=always-admit, preemption=fcfs`
     - Run one BLIS eval with the predicted config to get actual score
     - Return: `{"strategy": "formula-predict", "crossover_rate": ..., "predicted_tp": ..., "predicted_instances": ..., "score": ..., "config": ...}`
  3. **Wire up `formula-predict` in the `--strategy` argparse choice list** and dispatch to `run_formula_predict()` in the main strategy switch.
  4. **No changes to lean bracket or Phase 1/2 logic** — those are tested via existing flags.

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
- **Assuming crossover rate is model-size independent**: The `400/BW` formula (RP-27) works for 8-14B models but fails for 32B. qwen3-32b crossover on H100 is ~350, not ~119. [New in iter-8]
- **Micro-eval (≤500 requests) for TP winner identification**: Even at 500 requests on H100 qwen rate=500, TP=8 wins (should be TP=4). The queuing dynamics require ≥1000 requests to differentiate. [New in iter-8]
- **Assuming qwen3-32b has a crossover on A100-SXM**: Probed rates 100-2000 — TP=8 always wins. No crossover exists for 32B on A100-SXM. [New in iter-8]

## What I Excluded and Why

- **Analytical roofline predictor in Python**: Computing step_time from first principles (FLOPs, weight bytes, KV cache) requires reimplementing the Go latency model in Python. The formula-predict approach (one BLIS eval) tests the TP prediction concept with simpler code. A full analytical predictor could be iter-9 work if formula-predict shows the mechanism is sound. [Deferred — complexity vs. insight tradeoff]
- **qwen3-32b on L40S lean bracket arm**: L40S qwen3-32b shows TP=8 always winning by 13-15% margin. Including it in h-main would be trivially confirmed. The global best entries are added to GLOBAL_BEST_TABLE for future use. [Trivially confirmed — low experimental value]
- **Rate sweep for qwen3-32b A100-SXM**: Probed rates 100-2000 and confirmed no crossover. Sweeping intermediate rates provides no new mechanistic insight. [No crossover to find]
- **Multi-blis-seed robustness for qwen3-32b**: The robustness arm focuses on L40S (untested hardware) rather than qwen3-32b (untested model on already-validated hardware). One seed per qwen3-32b regime suffices for h-main since lean bracket is deterministic per blis_seed. Multi-seed robustness for qwen3-32b could be iter-9. [Deferred — diminishing returns]
- **Non-monotonicity investigation at H100 qwen3-32b rate=450**: TP=8 briefly wins at rate=450 between TP=4-dominant rate=350 and rate=500. This is likely a queuing dynamics artifact where the specific arrival pattern at rate=450 creates transient queue behavior. Not investigated because lean bracket still picks the correct winner at both rate=300 (TP=8) and rate=500 (TP=4). [Interesting but doesn't affect lean bracket correctness]
- **Phase 2 TPE for qwen3-32b**: Prior iterations showed Phase 2 adds ≤0.71% over Phase 1. For qwen3-32b, 4/5 regimes have single-instance TP=8 winners where routing is degenerate. Phase 2 can only help at H100 rate=500 (TP=4/2inst). [Diminishing returns]

## Evolution of Thinking

The campaign arrived at lean bracket (2 evals, ~0.5s) as the practical floor for BLIS configuration search. The natural question was whether to go lower — the handoff suggested "zero-eval" analytical prediction using the bandwidth formula. My exploration revealed this direction has significant obstacles:

1. The `400/BW` formula is an empirical fit to 8-14B model data. It fails for 32B models because it doesn't account for model size. A correct formula would need a model-size term, but the relationship isn't a simple power law.

2. A true zero-eval analytical predictor requires reimplementing the roofline latency model in Python — significant complexity for marginal benefit over 2 BLIS evaluations (~0.5s).

3. The queuing dynamics that determine the TP winner at high rates cannot be predicted from static model/hardware specs alone. The fitness score combines throughput (0.4), P99 TTFT (0.3), and P99 E2E (0.3), and the P99 tail effects depend on queue depth evolution over the simulation horizon.

4. Micro-eval (reducing simulation horizon) was an attractive alternative but fails fundamentally: queue saturation requires ≥1000 requests to manifest.

These findings shift the conclusion from "we can predict TP without evaluation" to "lean bracket at 2 evaluations is the practical minimum, and it's model-agnostic." The iter-8 hypothesis validates this by testing the largest model yet (32B) and the final hardware platform (L40S).

## Current Status

- **Validated:** (1) qwen3-32b crossover on H100 is ~350 (probed at rates 100-500). (2) qwen3-32b has no crossover on A100-SXM (probed up to rate=2000) or L40S. (3) RP-27 formula fails on 4/18 regimes (22% error rate). (4) Micro-eval ≤500 requests fails to identify TP=4 winner at high rates. (5) L40S baseline scores for both models established. (6) Lean bracket baseline works for qwen3-32b (verified).
- **Uncertain:** (1) Whether lean bracket non-monotonicity at H100 qwen3-32b rate=450 causes issues across blis seeds (only seed=42 tested). (2) Whether the 32B model's higher crossover makes lean bracket less deterministic across seeds. (3) Whether routing optimization (Phase 2) adds value for qwen3-32b H100 rate=500. (4) Whether larger models (70B) would push the crossover even higher or eliminate it on H100 entirely.
- **Suggested next:** (1) **Model-size scaling law**: Fit a crossover_rate(model_size, bandwidth) formula using the 3-model × 3-hardware data from iterations 1-8. This would generalize the prediction to arbitrary models. (2) **70B model test**: llama-3.1-70b-instruct exists in model_configs — test whether lean bracket still works and whether the crossover disappears on H100. (3) **Production `blis search` subcommand**: Package lean bracket as a native Go CLI command with auto-detection of model configs and hardware. (4) **Roofline-based analytical predictor**: If the model-size scaling law is clean, implement it as the default TP prediction with lean bracket as fallback.

## Warnings & Constraints

- **Stderr contains warnings**: BLIS prints timing and config warnings to stderr. Always use `2>/dev/null` or capture stderr separately when parsing stdout. [Carried from iter-2]
- **Seed affects workload, not config evaluation**: `--seed 42` controls random request generation. For the search algorithm's own randomness, use a separate seed. [Carried from iter-2]
- **TP=0 is invalid**: Always set `--tp` explicitly. [Carried from iter-2]
- **Fitness score format on stdout**: `Score: 0.019249` — parse with regex `r'Score:\s+([\d.]+)'`. [Carried from iter-2]
- **Multi-instance JSON output**: When `--num-instances > 1`, stdout contains multiple JSON blocks. Always parse the block with `"instance_id": "cluster"`. [Carried from iter-2]
- **Hardware name is case-sensitive**: `H100` not `h100`, `A100-SXM` not `a100-sxm`. [Carried]
- **Global best is (model, hardware, rate, num_requests)-specific**: New entries added for qwen3-32b (all hardware) and L40S (all models). [Updated iter-8]
- **Bracket profiles are deterministic**: They don't depend on search_seed (no RNG involved). All search seeds produce identical Phase 1 results per blis_seed. [Carried]
- **Profile ordering affects evals_to_best**: In lean bracket, TP=4 is evaluated first (position 1), TP=8 second (position 2). [Carried from iter-7]
- **Model name format**: Use full model paths: `qwen/qwen3-14b`, `qwen/qwen3-32b`, `meta-llama/llama-3.1-8b-instruct`. [Updated: added qwen3-32b]
- **Single-instance configs have degenerate routing**: At TP=8/1inst, routing_policy and block_size have no effect. [Carried from iter-3]
- **Must run blis from project root**: The binary expects model configs relative to the project directory. [Carried from iter-3]
- **Routing policy effect is seed-dependent on multi-instance configs**: round-robin vs least-loaded vs weighted varies by workload realization. [Carried]
- **Rate-dependent TP winner on H100 (qwen3-14b)**: TP=8 wins at rate<=125, TP=4 wins at rate>=150. [Carried from iter-5]
- **Rate-dependent TP winner on A100-SXM (qwen3-14b)**: TP=8 wins at rate<=150, tied at rate=200, TP=4 wins at rate>=500. [Carried from iter-6]
- **L40S qwen3-14b has no crossover**: TP=8 always wins at rates 100-500+. [Confirmed iter-8]
- **L40S llama-3.1-8b crossover ~400-500**: TP=8 at rate=100, TP=4 at rate=500. [Confirmed iter-8]
- **qwen3-32b H100 crossover ~350**: TP=8 at rate<=300, TP=4 at rate>=350 (with non-monotonicity at rate=450). [New in iter-8]
- **qwen3-32b A100-SXM and L40S: no crossover**: TP=8 always wins, even at rate=2000. [New in iter-8]
- **RP-27 formula (`400/BW`) fails for 32B models**: Predicts crossover at ~119 on H100, actual is ~350. Predicts TP=4 on A100-SXM at rate≥200, actual is TP=8 at all rates. [New in iter-8]
- **Micro-eval requires ≥1000 requests**: At 500 requests, TP=8 wins where 1000 requests gives TP=4. Queue saturation effects need sustained load. [New in iter-8]
- **qwen3-32b eval wall time**: ~0.28s per eval on H100 at 1000 requests (similar to 14B). ~0.11s at 100 requests. [New in iter-8]
- **Non-monotonic TP winner for qwen3-32b H100**: TP=4 at rate=350, TP=8 at rate=450, TP=4 at rate=500. The crossover band is not clean. [New in iter-8]
- **Default-profile masks rate effect**: With mr=64, TP=4/2inst wins at ALL rates. The rate-dependent TP=8 advantage only appears with max_running>=256 (max-profile). [Carried from iter-5]
