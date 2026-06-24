Validation passes. Here's a summary of the completed experiment:

---

## Iteration 7 Results: Model Portability

**All 4 arms complete. Validation: PASS.**

### Key Findings

**h-robustness (Llama exhaustive sweep):**
- Pareto density: **5.33%** (96/1800 configs), vs Qwen's 4.5% — directionally confirmed (higher), but below predicted range of 6-12%
- Exhaustive HV: **7.754e10** (32% larger than Qwen's 5.857e10, driven by Llama's higher throughput ceiling ~31.7 rps vs ~26 rps)

**h-main (NSGA-II, Llama) — PARTIALLY_CONFIRMED:**
- Convergence: 89.0% → 90.1% → 95.9% → 99.7% → **99.87%** at eval=40/80/120/160/200
- Reaches 95% at ~eval **114**. Random reaches 95% at ~eval **91** — random is **22.9 evals faster**
- Prediction of NSGA-II reaching 95% first: **REFUTED** at 5.33% density; final HV ≥97% prediction: **CONFIRMED**

**h-control-negative (Random, Llama) — CONFIRMED:**
- Reaches 95% at ~eval 91, final HV **98.21%** — both success criteria met

**h-ablation (Qwen re-verification) — PARTIALLY_CONFIRMED:**
- Infrastructure validated: Qwen NSGA-II eval=40 = 87.4% matches iter-5 exactly
- Within-iter-7 Qwen gap: **4.6 evals** (NSGA-II barely faster) vs Llama gap **-22.9 evals** (random faster)
- Cross-model direction **CONFIRMED**: higher density (Llama 5.33%) → smaller NSGA-II advantage

### New Principles
- **RP-12**: ~5% Pareto density is the breakeven where NSGA-II's 95%-threshold speed advantage disappears
- **RP-13**: NSGA-II's advantage scales with density **across model architectures** — Llama (5.33%, gap -22.9) vs Qwen (4.5%, gap +4.6) within same implementation confirms the density-driven portability mechanism
- **RP-11** updated to incorporate both rate-driven and architecture-driven density evidence