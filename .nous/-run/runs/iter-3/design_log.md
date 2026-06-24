Validation passes. Here's a summary of what was designed:

**Iteration 3: Multi-Model Generalization of Bracket K=1**

Tests whether bracket K=1 correctly identifies model-dependent TP winners — the first time the algorithm faces a different TP winner than all previous iterations.

**Key finding from probing:** L40S is the only hardware where the TP winner is model-dependent: qwen3-14b→TP=8/1inst (5.1% margin) while llama→TP=4/2inst (2.1% margin). On H100, both models→TP=4. Additionally, qwen's TP margins on L40S vary wildly across seeds, with seed=43 at just 0.03% — the tightest margin ever observed in this campaign.

**Four arms:**
- **h-main**: Bracket K=1 on L40S qwen (seed=42, 5 search seeds) — correctly identifies TP=8/1inst
- **h-robustness**: Bracket K=1 on L40S qwen across blis_seeds 42-46 (25 runs) — tests near-zero margins (0.03%, 0.11%)
- **h-control-negative**: Bracket K=1 on H100 qwen — TP=4 should win (L40S-specific effect vanishes on high-bandwidth hardware)
- **h-ablation**: Flat TPE on L40S qwen — comparison at intermediate 5.1% margin

**Code change:** One line added to GLOBAL_BEST_TABLE: `("qwen3-14b", "L40S", 500, 1000): 0.084795`.