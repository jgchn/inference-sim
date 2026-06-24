Validation passes. Here's a summary of iteration 8's design:

**Research question:** Is Pareto density a universal quantitative predictor of the NSGA-II vs random convergence gap?

**Method:** Run both Qwen3-14B and Llama-3.1-8B at rate=75 (higher density than rate=50), measuring exhaustive Pareto density and convergence gap with the same NSGA-II implementation used in iter-7. This adds two data points to the existing rate=50 results.

**Arms:**
- **h-robustness:** Exhaustive sweeps (1800 configs each) for both models at rate=75 → establishes densities and reference HVs
- **h-main:** NSGA-II + Random on Qwen at rate=75 → primary gap measurement at ~6% density
- **h-ablation:** NSGA-II + Random on Llama at rate=75 → cross-model validation

**Key prediction:** Both models at rate=75 will show NEGATIVE convergence gaps (random faster to 95% HV), since their densities will exceed the ~5% breakeven. All 4 data points (2 models × 2 rates) will fall on a single monotonically decreasing density→gap curve, demonstrating density is model-independent.

**Budget:** ~4400 BLIS evaluations total, ~5 minutes wall-clock.