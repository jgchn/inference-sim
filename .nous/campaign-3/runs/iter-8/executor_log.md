Validation passes. Here's the complete summary of iteration 8:

---

## Iteration 8 — Summary

**Research question:** Is Pareto density a universal quantitative predictor of the NSGA-II vs random convergence gap, independent of model architecture vs arrival rate?

**Answer: No.** The hypothesis is refuted.

### Key findings

| Model | Rate | Density | Gap (evals) |
|-------|------|---------|-------------|
| Qwen3-14B | 50 | 4.5% | +4.6 (NSGA-II faster) |
| Llama-3.1-8B | 50 | 5.33% | -22.9 (random faster) |
| Qwen3-14B | 75 | **6.06%** | **+31.0** (NSGA-II faster) |
| Llama-3.1-8B | 75 | **4.83%** | **-25.3** (random faster) |

**Three prediction failures:**
1. **RP-10 fails for Llama**: Llama's density *decreased* from 5.33% → 4.83% at rate=75 (Qwen correctly increased 4.5% → 6.06%). The rate-increases-density pattern is model-architecture-dependent, not universal.
2. **h-main refuted**: Qwen at 6.06% density shows a *larger* NSGA-II advantage (+31 vs +4.6 at 4.5%), directly inverting RP-11/RP-12's density-reduces-advantage prediction.
3. **No monotone curve**: The four (density, gap) points cannot be fit to a decreasing curve — Qwen moves in the opposite direction from what a universal density predictor requires.

**New principles extracted (RP-10, RP-11, RP-12 updated; RP-14 new):**
- The causal driver is **landscape structure** (gradient exploitability via tp×kv×block_size interactions), not density
- Qwen consistently favors NSGA-II; Llama consistently favors random — model architecture is the primary factor
- Density is a within-model correlate of landscape structure but does not generalize across models