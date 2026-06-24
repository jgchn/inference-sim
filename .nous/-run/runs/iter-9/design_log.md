Validation passes. Here's a summary of the iter-9 experiment design:

**Research question:** Does lean bracket generalize to 70B models, and can a model-size-aware formula eliminate the remaining TP prediction failures?

**Key probe findings:**
- 70B: TP=8 dominates ALL hardware (H100/A100/L40S) at ALL rates (100-2000) — no crossover exists
- TP=1 is infeasible (OOM), TP=2 is catastrophically bad (6.9× worse)
- Model-aware formula (load_index = params/BW > 12 → TP=8) achieves 24/24 vs old formula's 18/24

**Four arms:**
1. **h-main:** Lean bracket on 70B across 6 regimes × 5 seeds — expects TP=8 everywhere, evals_to_best=2
2. **h-control-negative:** Model-aware formula (100%) vs bandwidth-only formula (75%) on 24-regime matrix
3. **h-ablation:** Default-profile lean bracket for 70B — tests whether max-profile is necessary (probe shows TP=8 wins even at default)
4. **h-robustness:** Multi-blis-seed validation for 70B on H100 and L40S

**Code changes:** New `search_blis_iter9.py` adding 6 GLOBAL_BEST_TABLE entries for 70B, MODEL_PARAMS dict, and `model-aware-predict` strategy with load_index threshold.