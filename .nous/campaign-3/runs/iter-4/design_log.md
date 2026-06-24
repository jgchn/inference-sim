Validation passes.

## Summary

**Iteration 4 design complete.** The experiment tests whether removing non-differentiating parameters (routing, block_size, gpu_memory_utilization) amplifies NSGA-II's convergence speed over random search by concentrating crossover on fitness-relevant genes.

Key design elements:
- **Reduced space:** 900 configs (5 params) vs iter-3's 25200 (9 params)
- **Exhaustive ground truth:** All 900 configs evaluated in ~74s — eliminates reference ambiguity
- **4 arms:** NSGA-II reduced (h-main), Random reduced (h-control-negative), Exhaustive (h-robustness), NSGA-II full-space (h-ablation)
- **Primary prediction:** NSGA-II reaches 95% of exhaustive HV ≥80 evals before random (vs 40-eval gap in iter 3)
- **Total budget:** ~1500 evals, ~125s wall time