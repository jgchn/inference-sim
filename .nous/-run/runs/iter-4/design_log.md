Validation passes. Here's a summary of the iteration 4 design:

**Family**: `a100-qwen-margin-threshold` — completes the 2×3 model×hardware matrix and narrows the TPE convergence margin threshold.

**Key probing results**:
- A100-SXM qwen3-14b: TP=4/2inst wins at all 5 seeds (margins 1.07%–4.47%)
- Global best is 0.138820 (round-robin routing, 0.38% above Phase 1 max-profile) — first regime where Phase 2 adds genuine value
- Bandwidth threshold for qwen TP winner: between L40S (0.864 TB/s, TP=8) and A100 (2.039 TB/s, TP=4)

**Four arms**:
1. **h-main**: Bracket K=1 on A100-SXM qwen (seed=42) — predicts TP=4, evals_to_best=3
2. **h-robustness**: 25 (blis_seed × search_seed) combinations — predicts 25/25 correct
3. **h-ablation**: Flat TPE at 1.07% margin (seed=42) — predicts poor convergence (≤2/5 success), narrowing threshold from below
4. **h-control-negative**: Flat TPE at 4.47% margin (seed=46) — predicts good convergence (≥3/5 within-threshold), demonstrating margin width as the causal variable

**Single code change**: update GLOBAL_BEST_TABLE entry for A100-SXM qwen from 0.138290 → 0.138820.