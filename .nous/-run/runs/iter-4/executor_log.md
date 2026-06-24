All five phases complete. Here's a summary of findings from Iteration 4:

---

## Iteration 4 Results Summary

**All 4 arms executed, validation passed.**

### h-main — CONFIRMED
Bracket K=1 on A100-SXM qwen3-14b (blis_seed=42): 5/5 search seeds identify TP=4/2inst with evals_to_best=3 (deterministic). Phase 1 scores exactly matched predictions (TP4=0.138290, TP8=0.136812). Phase 2 TPE finds the round-robin routing improvement (0.138820) in 4/5 seeds. **Matrix completion confirmed: A100-SXM follows H100 behavior (TP=4 for both models).**

### h-robustness — CONFIRMED
25/25 Phase 1 correctness across all 5 blis_seeds × 5 search seeds. evals_to_best=3 for all 25 combinations — perfect determinism even at the two tightest TP margins in the campaign (seed=42: 1.07%, seed=44: 1.24%).

### h-ablation — PARTIALLY CONFIRMED (magnitude discrepancy)
Flat TPE at 1.07% margin (blis_seed=42): unexpectedly, **all 5 seeds pass the within-1% threshold** (predicted ≤2/5). Median evals_to_best=26 (predicted >50). Direction correct — TPE is much slower than bracket (26 vs 3), but not a failure regime. Key new finding: **A100-SXM's landscape is more TPE-friendly than L40S at equivalent relative margins**, because more TP4/2inst configurations exceed the absolute threshold.

### h-control-negative — CONFIRMED
Flat TPE at 4.47% margin (blis_seed=46): 5/5 within-threshold, median evals_to_best=6. Within-hardware comparison confirms **margin width causally affects TPE speed** (6 vs 26 evals at 4.47% vs 1.07%).

### Key New Principles
- **RP-21 extended**: 2×3 model×hardware matrix complete — A100-SXM (2.039 TB/s) follows H100, bandwidth threshold for qwen TP winner is between 0.864 and 2.039 TB/s.
- **RP-22 revised**: TPE convergence threshold is hardware/model-specific. The [2.1%, 5.1%] bound applies to L40S only. On A100-SXM, TPE succeeds even at 1.07% margin.
- **RP-23 (new)**: A100-SXM qwen3-14b global best is 0.138820 via round-robin routing — first regime where Phase 2 adds genuine value over Phase 1.