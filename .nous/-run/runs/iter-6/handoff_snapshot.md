# Handoff — Iteration 6: Cross-Hardware Rate Crossover on A100-SXM

## Goal

Test whether bracket K=1 hierarchical search correctly adapts to rate-dependent TP winner transitions on A100-SXM hardware. At rate=100, TP=8/1inst wins on A100-SXM (2.5% margin for qwen, 4.3% for llama), while at rate=500 TP=4/2inst wins (known from prior iterations). The crossover is at rate~200 for qwen on A100-SXM — higher than H100's ~125 crossover, consistent with A100-SXM's lower bandwidth (2.039 vs 3.35 TB/s). Run bracket K=1 at rate=100 (A100-SXM TP=8 winner), rate=200 (A100-SXM crossover), and flat TPE at rate=100 for comparison. Validate llama model across 5 blis_seeds.

## Key Discoveries

1. **Rate-dependent TP crossover is UNIVERSAL**: The TP=8-at-low-rates / TP=4-at-high-rates transition exists on ALL tested hardware, not just H100. Probed on A100-SXM: TP=8 wins at rate≤150, tied at rate=200, TP=4 wins at rate≥500. Probed on L40S: llama crossover at rate~450, qwen has NO crossover (TP=8 always wins).

2. **Crossover rate scales inversely with bandwidth**: H100 (3.35 TB/s) crosses at ~125, A100-SXM (2.039 TB/s) at ~200, L40S (0.864 TB/s) at ~450 or never. Lower bandwidth means TP=8's bandwidth pooling advantage persists to higher rates because the system needs more bandwidth per request.

3. **A100-SXM qwen max-profile Phase 1 scores at rate=100 (seed=42)**: TP1=0.080450, TP2=0.105531, TP4=0.127461, TP8=0.130754. Monotonically increasing. TP=8 margin: 2.5%. Confirmed seed=43: margin 1.5%.

4. **A100-SXM llama at rate=100 (seed=42)**: TP2=0.136399, TP4=0.154241, TP8=0.160845. TP=8 margin 4.3% — wider than qwen because the smaller 8B model makes better use of full bandwidth pooling.

5. **A100-SXM qwen crossover at rate=200 (seed=42)**: TP4=0.134488, TP8=0.134486 (0.001% margin). Global best is 0.134935 (TP=4 with round-robin routing, discovered by probing). 1% threshold = 0.133586. TP=4 at position 3 exceeds threshold → evals_to_best=3.

6. **L40S landscape**: qwen TP=8 wins at both rate=100 (6.9% margin) and rate=500 (5.1%). No crossover exists. llama shows TP=8 at rate≤400, TP=4 at rate=500 (2.1% margin). Crossover at ~450.

7. **Global best values for new regimes (seed=42)**: (qwen3-14b, A100-SXM, 100, 1000)=0.130754; (qwen3-14b, A100-SXM, 200, 1000)=0.134935; (llama-3.1-8b, A100-SXM, 100, 1000)=0.160845.

## System Interface

- **Build:** `go build -o blis main.go` (binary already exists in project root)
- **Run baseline (rate=100, TP=8, A100-SXM qwen):**
  ```bash
  ./blis run --model qwen/qwen3-14b --hardware A100-SXM --latency-model trained-physics \
    --num-requests 1000 --rate 100 --seed 42 --tp 8 --num-instances 1 \
    --scheduler fcfs --max-num-running-reqs 512 --max-num-scheduled-tokens 8192 \
    --long-prefill-token-threshold 4096 --block-size-in-tokens 16 \
    --routing-policy round-robin --admission-policy always-admit \
    --preemption-policy fcfs \
    --fitness-weights "throughput:0.4,p99_ttft:0.3,p99_e2e:0.3"
  ```
- **Output format:** JSON metrics between `=== Simulation Metrics ===` markers. With `--fitness-weights`, `Score: <float>` follows in `=== Fitness Evaluation ===` section.
- **Baseline result:** A100-SXM qwen TP=8/1inst rate=100 Score=0.130754.

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

The experiment modifies the Python search script from the existing iter-6 h-main.patch (which contains search_blis_iter6.py with bracket profiles):

- **`search_blis_iter6.py`** (from existing h-main.patch):
  1. **Add GLOBAL_BEST_TABLE entries for A100-SXM rate=100**:
     - `("qwen3-14b", "A100-SXM", 100, 1000): 0.130754`
     - `("llama-3.1-8b", "A100-SXM", 100, 1000): 0.160845`
  2. **Add GLOBAL_BEST_TABLE entry for A100-SXM crossover rate=200**:
     - `("qwen3-14b", "A100-SXM", 200, 1000): 0.134935`
  3. **Add H100 rate=100/125 entries from iter-5**:
     - `("qwen3-14b", "H100", 100, 1000): 0.164680`
     - `("qwen3-14b", "H100", 125, 1000): 0.162182`
     - `("llama-3.1-8b", "H100", 100, 1000): 0.182394`
  4. **Update A100-SXM qwen rate=500**: Change `("qwen3-14b", "A100-SXM", 500, 1000): 0.138290` to `0.138820` per RP-23.
  5. No algorithmic changes — bracket K=1 is already implemented.

## What I Tried That Didn't Work

- **`--hardware h100_sxm`**: Fatal error. Correct name is `H100` (case-sensitive, no suffix). [Carried from iter-1]
- **`--hardware A100`**: Fatal error. Must be `A100-SXM` or `A100-80`. [Carried from iter-5]
- **Parsing fitness from `--metrics-path` JSON**: The JSON file does not contain the fitness score. Must parse stdout. [Carried from iter-1]
- **Running blis from subdirectory**: The blis binary must be run from the project root `/Users/jchen/go/src/inference-sim/inference-sim`. [Carried from iter-3]
- **grep "Score:" with awk in for-loop subshell**: When the working directory isn't the project root, `./blis` fails silently and score is empty. Always cd to project root first. [Carried from iter-3]
- **Assuming TP=8 dominance is unconditional on batch size**: Requires max_running >= 256. With mr<256, TP=4/2inst wins even at rate=100. [Carried, RP-13]
- **Assuming round-robin always beats least-loaded**: The routing advantage is seed-dependent. On A100-SXM at rate=200, round-robin gives TP=4 a 0.33% edge over least-loaded. [Carried from iter-4]
- **Assuming rate crossover is H100-specific**: Probing shows it's universal — A100-SXM crosses at ~200, L40S llama at ~450. [New in iter-6]

## What I Excluded and Why

- **L40S arms**: L40S qwen has NO crossover (TP=8 always wins), so there's no rate-adaptive behavior to test. L40S llama crosses at ~450, but establishing the global best at intermediate rates (200-400) would require extensive probing. A100-SXM provides a cleaner test because the crossover is at the well-separated rate=200. [Deferred]
- **Third model (qwen3-32b)**: 32B params may cause memory issues on some TP configs. Not worth debugging risk for this iteration. [Deferred from iter-5]
- **Rate crossover on llama for A100-SXM**: The crossover for llama on A100-SXM exists (TP=8 at rate=100, TP=4 at rate=500) but the exact crossover rate wasn't narrowed. The qwen crossover at rate=200 is sufficient to demonstrate the hardware-specific crossover phenomenon. [Deferred]
- **Phase 2 value quantification at rate=100**: At rate=100 where TP=8/1inst wins, it's single-instance (routing degenerate). Phase 2 can only vary scheduler/mr/mt/pf. Max-profile is likely near-optimal. [Not worth a separate arm]
- **Multi-rate sweep on A100-SXM (3+ rates)**: Would require 3+ arms worth of compute. The binary comparison (rate=100 TP=8 wins, rate=200 tied) is sufficient to establish the crossover. [Design choice]

## Evolution of Thinking

Iterations 1-5 established bracket K=1's robustness across models and hardware at fixed high rates, then discovered the rate-dependent TP transition on H100. The implicit question from iter-5 was: is this transition H100-specific or universal?

Probing answered decisively: the transition is universal. Every hardware platform shows TP=8 winning at low rates and TP=4 (or no transition) at high rates. The critical insight is that the crossover rate scales with 1/bandwidth — a clean physical relationship. H100's high bandwidth (3.35 TB/s) means each instance has enough bandwidth to serve requests efficiently even at lower TP, so the concurrency advantage of multiple instances kicks in at lower rates (~125). L40S's low bandwidth (0.864 TB/s) means the system is bandwidth-starved, so pooling bandwidth via TP=8 helps at much higher rates (~450) or never (qwen, which is more bandwidth-intensive per token due to larger model).

The key experimental question now is whether bracket K=1 handles this correctly on A100-SXM — a hardware platform with intermediate bandwidth and a crossover at a different rate than H100. If it works here without modification, we can confidently claim hardware-agnostic rate-adaptiveness.

## Current Status

- **Validated:** Rate-TP crossover exists on A100-SXM (qwen crossover at ~200, confirmed with 2 blis_seeds). Rate-TP crossover exists on L40S (llama at ~450, qwen none). Baseline command exits cleanly on A100-SXM at rate=100. Phase 1 TP landscape is monotonically increasing on A100-SXM at rate=100 for both models. Global best values established for all new regimes.
- **Uncertain:** (1) Exact TP margins across blis_seeds 43-46 for A100-SXM qwen at rate=100 (probed only 42 and 43). (2) Whether flat TPE convergence speed on A100-SXM at rate=100 matches H100's dramatic median=6. (3) Whether Phase 2 adds value at the A100-SXM rate=200 crossover (round-robin discovery).
- **Suggested next:** (1) L40S rate crossover validation for llama at rate~450 — narrowing the crossover and testing bracket K=1 on the highest-crossover hardware. (2) Bandwidth-crossover model: can we predict the crossover rate from hardware bandwidth and model size alone? (3) Multi-rate bracket K=1 — what if the search algorithm automatically sweeps 2-3 rates to find the crossover? (4) Production packaging: create a standalone `blis search` subcommand with bracket K=1 built in.

## Warnings & Constraints

- **Stderr contains warnings**: BLIS prints timing and config warnings to stderr. Always use `2>/dev/null` or capture stderr separately when parsing stdout. [Carried from iter-2]
- **Seed affects workload, not config evaluation**: `--seed 42` controls random request generation. For the search algorithm's own randomness, use a separate seed. [Carried from iter-2]
- **TP=0 is invalid**: Always set `--tp` explicitly. [Carried from iter-2]
- **Fitness score format on stdout**: `Score: 0.019249` — parse with regex `r'Score:\s+([\d.]+)'`. [Carried from iter-2]
- **Multi-instance JSON output**: When `--num-instances > 1`, stdout contains multiple JSON blocks. Always parse the block with `"instance_id": "cluster"`. [Carried from iter-2]
- **Hardware name is case-sensitive**: `H100` not `h100`, `A100-SXM` not `a100-sxm`. [Carried]
- **Global best is (model, hardware, rate, num_requests)-specific**: New entries added for A100-SXM rate=100 and rate=200. [Updated iter-6]
- **Bracket profiles are deterministic**: They don't depend on search_seed (no RNG involved). All 5 search seeds produce identical Phase 1 results per blis_seed. [Carried]
- **Profile ordering affects evals_to_best**: Max-first is optimal. [Carried]
- **Model name format**: Use full model paths: `qwen/qwen3-14b`, `meta-llama/llama-3.1-8b-instruct`. [Carried]
- **Prefill_threshold is a no-op for fitness on some hardware**: All values produce identical scores at fixed (max_running, max_tokens) on A100/L40S. [Carried]
- **vllm-version auto-detection**: Warnings about "Finding default values of vLLM version" appear on stderr. Harmless but noisy. [Carried]
- **Single-instance configs have degenerate routing**: At TP=8/1inst, routing_policy and block_size have no effect. [Carried from iter-3]
- **Must run blis from project root**: The binary expects model configs relative to the project directory. [Carried from iter-3]
- **Routing policy effect is seed-dependent on multi-instance configs**: round-robin vs least-loaded varies by workload realization. [Carried from iter-4]
- **Rate-dependent TP winner on H100**: TP=8 wins at rate<=125, TP=4 wins at rate>=150 (qwen3-14b). [Carried from iter-5]
- **Rate-dependent TP winner on A100-SXM**: TP=8 wins at rate<=150 (1.1% margin), tied at rate=200 (0.001%), TP=4 wins at rate=500. Crossover at ~200. [New in iter-6]
- **L40S has the highest crossover rate**: ~450 for llama (TP=8 at rate<=400, TP=4 at rate=500). Qwen has no crossover on L40S. [New in iter-6]
- **Crossover rate inversely proportional to bandwidth**: H100 (3.35 TB/s) ~125, A100-SXM (2.039 TB/s) ~200, L40S (0.864 TB/s) ~450+. [New in iter-6]
- **A100-SXM qwen rate=200 global best uses round-robin**: TP=4/2inst with round-robin=0.134935 vs least-loaded=0.134488. [New in iter-6]
- **evals_to_best=4 when TP=8 wins**: TP=8 is at position 4 in Phase 1 evaluation order (TP=1,2,4,8). This is an inherent 1-eval penalty vs TP=4 winning (evals_to_best=3). [Carried from iter-5]
- **Default-profile masks rate effect**: With mr=64, TP=4/2inst wins at ALL rates. The rate-dependent TP=8 advantage only appears with max_running>=256 (max-profile). [Carried from iter-5]
