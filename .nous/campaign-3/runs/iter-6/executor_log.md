Validation passes. Here's a summary of the iter-6 findings:

**Experiment complete. All artifacts validated.**

## Key Findings

The central prediction was **wrong in direction**: higher arrival rate leads to **higher** Pareto density, not lower.

| Rate | Pareto density | NSGA-II final | Random final | Convergence gap |
|------|---------------|---------------|--------------|-----------------|
| rate=50 (iter-5) | 4.5% | 99.2% | 94.3% | 91 evals |
| rate=75 | 6.06% | 99.45% | 97.33% | ~48 evals |
| rate=100 | 6.89% | 98.77% | 97.31% | ~57 evals |

**What happened:** Higher load expands the Pareto frontier by opening new cost-performance tradeoff regions (throughput-limited low-cost configs become Pareto-optimal, plus the higher throughput ceiling differentiates high-TP configs). This increases Pareto density, which makes random search more competitive — narrowing NSGA-II's advantage rather than widening it.

**What's portable:** NSGA-II's **speed advantage** is portable — it still reaches 95% of exhaustive HV ~57 evals before random at rate=100 (vs 91 at rate=50). But the **final HV gap** shrinks dramatically (1.46% at rate=100 vs 4.9% at rate=50).

**New principles extracted:** RP-10 (Pareto density increases monotonically with load in BLIS) and RP-11 (NSGA-II convergence speed advantage decreases with higher rates). RP-1 and RP-2 updated to reflect rate-dependence of density.