Validation passes. The experiment is complete.

**Summary of findings:**

All three algorithm predictions were **REFUTED** — but the experiment was still valid and highly informative:

1. **NSGA-II (h-main) — REFUTED (regime error):** NSGA-II's initial population already saturated the 3-objective Pareto frontier (HV reached 98.2% at eval=50). The predicted convergence advantage over random never materialized. The root cause: ~60% of within-tier configs achieve tier-optimal metrics (any batch ≥256 out of {32,64,128,256,512} suffices), making the frontier trivially accessible to random sampling. NSGA-II final HV exceeded random by only 0.02%.

2. **Random Search (h-control-negative) — REFUTED (regime error):** Random reached 95% of its final HV at eval=50, not the predicted 300-400 evaluations. The assumed 1-3% per-tier Pareto density was wrong by ~20x — actual density is ~60%.

3. **TPE (h-robustness) — REFUTED (direction error):** TPE converged to a **7.1% lower** final HV than random, performing worse rather than better. Optuna's multivariate TPE cannot handle conditional parameter spaces (num_instances conditioned on tp), causing it to fall back to random sampling for conditional parameters while biasing toward certain GPU tiers.

**Key principle extracted (RP-4, new):** Multi-objective TPE performs worse than random on conditional spaces — a novel finding with practical implications for search algorithm selection on hierarchical configuration spaces.