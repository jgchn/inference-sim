**Validation passed.** Here's a summary of all findings from iteration 2:

---

## Iteration 2 Results Summary

**Research question:** Does bracket K=1 hierarchical search generalize to L40S hardware (2.1% TP margin) and remain robust across BLIS seed variation?

### All 4 Arms Confirmed

| Arm | Result | Key Metric |
|-----|--------|-----------|
| **h-main** (L40S bracket K=1) | CONFIRMED | 5/5 seeds: TP=4/2inst, evals_to_best=3 |
| **h-robustness** (H100 multi-seed) | CONFIRMED | 25/25 (blis_seed×search_seed): TP=4/2inst, evals_to_best=3 |
| **h-control-negative** (L40S min-only) | CONFIRMED | 5/5 seeds: TP=2/4inst (wrong), evals_to_best=101 |
| **h-ablation** (flat TPE on L40S) | CONFIRMED (magnitude deviation) | Median evals_to_best=101 (4/5 seeds fail) vs bracket=3 |

### Key New Findings

1. **Bracket K=1 is hardware-portable**: Works on L40S (2.1% TP margin) as well as H100/A100, always achieving evals_to_best=3.

2. **Bracket K=1 is seed-robust**: 25/25 runs across all blis_seed × search_seed combinations on H100 achieve Phase 1 correctness and evals_to_best=3.

3. **L40S tight margin breaks flat TPE (RP-19)**: At 2.1% TP margin, flat TPE fails in 4/5 seeds within budget=100. The relationship between TP margin size and TPE convergence reliability is non-linear — hierarchical search becomes qualitatively necessary (not just beneficial) at tight margins.

4. **Min-profile failure mode is hardware-specific (RP-17 updated)**: L40S selects TP=2/4inst (wrong) while H100 selects TP=1/8inst (wrong) — different capacity-limited optima reflect different bandwidth tradeoffs.