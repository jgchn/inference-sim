Validation passes.

## Summary

**Iteration 4 design complete.** The hypothesis bundle tests whether Phase 1 batch-size profile (batch=512 vs batch=128) critically determines tier-ranking correctness:

- **h-main**: Lean Phase 1 (batch=128) + TPE Phase 2 can match standard Phase 1 on H100 rate=5000 where lean correctly identifies the tier (25.6% Phase 2 headroom for TPE to exploit)
- **h-control-negative**: Lean Phase 1 catastrophically fails on H100 rate=500 — picks wrong tier with unrecoverable 24% deficit
- **h-ablation**: TPE Phase 2 converges faster than LHS within the correctly-identified tier (monotone batch→score relationship)
- **h-robustness**: Standard Phase 1 (batch=512) universally identifies correct tier across all 6 (rate, hardware) conditions

Key finding from probes: **the BLIS Pareto problem is structurally degenerate** — batch=512 dominates all metrics simultaneously at every rate tested. The search problem is solved in 4 evaluations with standard Phase 1, making this iteration's focus on characterizing the safety boundary of that approach.