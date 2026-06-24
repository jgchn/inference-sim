Validation passes. Here's a summary of what was found:

---

## Iteration 4 Results — Phase 1 Batch-Sensitivity

**h-main (CONFIRMED):** Lean Phase 1 (batch=128) correctly identifies TP=4/2 on H100 rate=5000 across all 3 seeds. TPE Phase 2 converges to 95% of optimal in just **2-3 evals** (vs predicted ≤20), reaching or exceeding the reference score of 0.176474.

**h-control-negative (PARTIALLY CONFIRMED):** Lean Phase 1 is always wrong at H100 rate=500 (0/3 seeds pick correct TP=8/1), confirming the failure. However, the mechanism is seed-dependent: seed=42 picks TP=2/4 (unrecoverable, 24.2% deficit), while seeds 123/456 pick TP=4/2 (partially recoverable, 4-8% deficit). The ≥20% unrecoverable deficit prediction holds only when TP=2/4 wins — which depends on a ~0.02% score margin.

**h-ablation (CONFIRMED):** TPE reaches 95% of optimal in 2-3 evals; LHS requires 5-16 evals. TPE wins in all 3/3 seeds, outperforming the predicted 2/3 minimum.

**h-robustness (PARTIALLY CONFIRMED):** Standard Phase 1 (batch=512) is 16/18 correct (89%). It fails on A100-SXM at seed=456 for rates ≥2000, where TP=4/2 and TP=8/1 are within 0.1-1.0% — a genuine tie zone. The prior "A100-SXM always picks TP=8/1" characterization was based on seed=42 only. Lean Phase 1 fails in 4 of 6 conditions as predicted.

**New principles extracted:** RP-C2-11 updated (A100-SXM tie zone discovered), RP-C2-13 (lean Phase 1 safe regime refined), RP-C2-14 (TPE advantage quantified: 25% headroom enables 2-3 eval convergence), RP-C2-5 updated (batch=512 requirement now hardware-rate scoped).