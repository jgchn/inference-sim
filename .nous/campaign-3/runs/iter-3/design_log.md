Validation passes. Here's a summary of the iteration 3 design:

**Core insight**: Adding `total_kv_blocks` as a searchable parameter creates non-linear interactions that didn't exist in iter 2. KV blocks are **per-instance**, so tp=4,i=1 with blocks=5000 has severe preemption stress (77 preemptions, ttft=2640ms) while tp=2,i=2 with the same blocks has zero stress (ttft=39ms). This reverses the dominance relationship found in iter 2 and creates genuine Pareto tradeoffs within GPU tiers.

**Experiment**: 4-objective formulation (rps↑, ttft↓, gpu_count↓, kv_blocks↓) with 200-eval budget. Three arms:
- **h-main**: NSGA-II predicting faster convergence via crossover exploiting the TP↔blocks cliff structure
- **h-control-negative**: Random search as baseline (uniform sampling wastes budget on dominated regions)
- **h-robustness**: NSGA-II on original 3-obj space (confirming it remains trivially easy, isolating the difficulty to the blocks dimension)