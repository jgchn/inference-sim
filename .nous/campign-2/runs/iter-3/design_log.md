Validation passes.

## Summary

Iteration 3 is designed and validated. The experiment tests two advances over iteration 2's hierarchical search:

1. **Extremes-first Phase 1** (4 evals vs 15) — evaluates only max-instance per TP level, addressing the structural ordering weakness (RP-C2-7). Validated that max-instance is always optimal within each TP level at rate≥2000.

2. **Hardware portability** — H100 optimal is TP=4/2, A100-SXM optimal is TP=8/1. The same algorithm must correctly identify both without hardware-specific tuning.

3. **TPE Phase 2** — at rate=5000 where batch size creates 26% score variance, testing whether Bayesian optimization outperforms Latin Hypercube for within-tier refinement.

Four arms: h-main (extremes-first + TPE at rate=5000), h-control-negative (rate=50 where Phase 2 is provably useless), h-ablation (LHS vs TPE Phase 2), h-robustness (A100-SXM hardware portability).