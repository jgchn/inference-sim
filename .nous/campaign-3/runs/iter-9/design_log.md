Validation passes. Let me summarize the experiment design.

## Iteration 9 Summary

**Research question:** Does landscape structure (measured via Neighbor Dominance Rate) causally determine the NSGA-II vs random convergence gap?

**4 arms:**
1. **h-robustness** — Exhaustive sweep of all 1800 configs for both models at rate=100, saving ALL per-config results. Computes NDR (fraction of Hamming-1 neighbor pairs with dominance) with per-axis breakdown. Predicts Qwen NDR > Llama NDR, with kv_blocks as the most dominant axis for Qwen.

2. **h-main** — NSGA-II vs random on Qwen at rate=100 (full 1800-config space). Predicts positive gap ≥ +31 evals, extending the Qwen-positive pattern from iter-7/iter-8.

3. **h-ablation** — Same comparison for Llama at rate=100. Predicts negative gap (random wins), extending the Llama-negative pattern.

4. **h-control-negative** — The causal test: Qwen at rate=100 with kv_blocks FIXED at 10000 (cliff removed), 300-config 3-objective space. Predicts gap ≈ 0 or negative. If confirmed, this proves the KV cliff is the causal mechanism behind NSGA-II's advantage.

**Key innovation:** This is the first iteration to directly test the MECHANISM (not just correlates). The control-negative ablates the proposed cause and predicts the effect vanishes.