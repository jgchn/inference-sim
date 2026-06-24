Validation passes. Here's a summary of what was designed:

**Iteration 5: Rate-Dependent TP Winner Transition**

The campaign (iters 1-4) validated bracket K=1 across 2 models × 3 hardware at two fixed rate regimes. Probing revealed a previously unknown dimension: **the TP winner on H100 is rate-dependent**, transitioning from TP=8 (rate≤125) to TP=4 (rate≥150) at a crossover near rate≈125 req/s.

**Four arms:**
1. **h-main**: Bracket K=1 at rate=100 — tests whether it correctly identifies the new TP=8 winner (predicted evals_to_best=4)
2. **h-robustness**: 25 seed combinations at rate=100 — verifies robustness across workload realizations
3. **h-ablation**: Rate=125 crossover — tests vanishing-margin behavior where both TP=4 and TP=8 are within 1% (predicted evals_to_best=3 despite ambiguous winner)
4. **h-control-negative**: Flat TPE at rate=100 — tests whether the monotonic landscape at low rate makes TPE converge faster than at rate=500

**Key code change**: Add GLOBAL_BEST_TABLE entries for rate=100 and rate=125 regimes; update the stale A100-SXM qwen entry. No algorithmic changes to bracket K=1.