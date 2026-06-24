All artifacts are present and validation passes. Here's a summary of what was accomplished:

## Summary

**Experiment completed successfully.** All 9 conditions executed across 3 arms.

### Key Findings

**h-main (rate=2000, hard regime):** CONFIRMED with nuance
- Hierarchical: **14 evals** to 95% threshold (deterministic, Phase 1 hits TP=4/2 at eval 14)
- Random: **64 evals** to 95% (4.6x slower than hierarchical, exceeding the 3x success criterion)
- NSGA-II: **8 evals** to 95% (faster than hierarchical at seed=42, but this is lucky random warmup — ~19% chance probability, not systematic)

**h-control-negative (rate=50, easy regime):** PARTIALLY_CONFIRMED
- All 3 strategies converge within 15 evals (<<50 predicted) ✓
- But 5x spread (NSGA-II=3, random=10, hierarchical=15) vs predicted <2x ✗
- Hierarchical is slowest because its fixed Phase 1 order evaluates TP=8/1 last (eval 15)

**h-robustness (seeds 42, 123, 456):** CONFIRMED
- TP=4/2 correctly identified as #1 tier in 3/3 seeds
- evals_to_95pct = **exactly 14** for all seeds (CV = 0%, far below 30% threshold)

### New Principles
- RP-C2-6: Hierarchical Phase 1 deterministically converges in 14 evals at rate=2000 (TP=4/2 at index 13 in tier order)
- RP-C2-7: Fixed-order Phase 1 is structurally disadvantaged when the optimal tier is evaluated late (high-TP tiers evaluated last)
- RP-C2-8: True qualifying config density is ~2.67% (not 0.017%) — both TP=4/2 and TP=8/1 with batch=512 meet the 95% threshold
- RP-C2-9: NSGA-II's apparent advantage over hierarchical at seed=42 is lucky random-warmup sampling, not systematic evolutionary advantage