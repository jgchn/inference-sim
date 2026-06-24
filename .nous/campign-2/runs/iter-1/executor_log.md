All 5 phases complete. Here's a summary of what was found and produced:

---

## Summary

**All artifacts validated successfully** (`nous validate: pass`).

### Key Findings

**h-main — REFUTED** (error_type: `regime`): The hypothesis predicted hierarchical search would achieve ≥20% higher hypervolume than random search by concentrating budget on productive TP tiers. Instead, both strategies achieved **identical HV (7.755e+08)**. The root cause: at rate=50, TP=8/1-inst simultaneously achieves the highest throughput (14,232 tok/s) AND lowest latency (4,719ms), strictly dominating all other tiers in both objectives. The Pareto problem degenerates to a single-point frontier. Random search discovers this maximum with ~9% per-eval probability (finds it by eval ~20), so no budget-allocation strategy helps.

**h-control-negative — PARTIALLY_CONFIRMED** (error_type: `regime`): The "advantage vanishes at rate=5" outcome was confirmed (HV ratio = 1.000), but for the wrong reason. The prediction assumed all tiers achieve similar throughput — instead, TP=8/1-inst still dominates at rate=5, AND all secondary knobs produce identical results within each tier (arrival-rate-limited, zero batching dynamics).

**h-robustness — CONFIRMED**: Phase 1 tier ranking is **perfectly stable** across all 5 seeds (42, 123, 456, 789, 1024) — top-3 tiers are invariably `[8x1, 4x2, 4x1]`. HV varies ±7.8% across seeds (workload sampling), but tier ordering is invariant.

### 5 Principles Extracted (RP-C2-1 through RP-C2-5)
The key actionable insight for iteration 2: **genuine Pareto tradeoffs between BLIS parallelism tiers require rate >> 100 req/s** (where TP=8/1-inst approaches saturation, allowing lower-TP multi-instance configs to offer lower latency at reduced throughput). Rate=50 is far below saturation.