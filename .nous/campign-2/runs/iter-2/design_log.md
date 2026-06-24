Validation passes. 

## Summary

**Iteration 2 design is complete.** The key insight from probing: BLIS's throughput-vs-latency Pareto problem is fundamentally degenerate at every rate (one config always dominates both objectives simultaneously). This forced a reframing from "Pareto frontier discovery" to "convergence speed" — how quickly each strategy finds the throughput-maximizing config.

The experiment tests at rate=2000 where the optimal config is non-obvious (TP=4/2 beats TP=8/1 due to single-instance queue bottleneck), comparing:
1. Random search (baseline)
2. Hierarchical decomposition (Phase 1 tier screening + Phase 2 within-tier LHS)
3. NSGA-II via Optuna (intelligent baseline)

Primary metric: evaluations needed to reach 95% of optimal throughput (38,373 tok/s from a target of 40,392).