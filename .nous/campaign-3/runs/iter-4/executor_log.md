Validation passes. Here's the summary of what was found:

## Iteration 4 Results

**Execution**: All 4 arms completed successfully in ~126 seconds total (900 + 3×200 BLIS evaluations).

**Key findings:**

1. **h-robustness** (PARTIALLY_CONFIRMED): Exhaustive sweep confirmed 26.4% Pareto density — slightly above the predicted 8-25%, but the qualitative prediction (non-trivial search) was correct. Exhaustive reference HV = 5.2962e+10.

2. **h-main** (PARTIALLY_CONFIRMED): NSGA-II on reduced space reaches 95% of exhaustive HV at eval=117 vs random at eval=164 — a **47-eval gap**, up from iter-3's 40-eval gap. Direction confirmed, but fell short of the predicted 80-eval gap. The 22% random coverage (200/900) and 26.4% Pareto density limited the gap.

3. **h-control-negative** (CONFIRMED): Random reaches 95% at eval=164 (>160 as predicted). Final HV 3.9% below exhaustive.

4. **h-ablation** (REFUTED — critical discovery): The "non-differentiating" `block_size_in_tokens` is **highly differentiating** for tp=8. Since `total_kv_blocks` counts blocks (not tokens), doubling `block_size` from 16→32 doubles effective KV token capacity: at tp=8,kv=3000 this eliminates all preemptions (107→0) and drops ttft_p99 from 2323ms→25ms (+23% rps). Fixing block_size=16 in the reduced space inadvertently excluded the best tp=8 configurations.

**New principle (RP-8)**: `effective_kv_tokens = total_kv_blocks × block_size_in_tokens`. When sweeping total_kv_blocks, block_size must be treated as a co-variate, not a fixed non-differentiating parameter.