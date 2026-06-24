Validation passes. Here's a summary of the iteration 3 experiment findings:

---

## Summary

All four arms completed successfully. Validation passes.

**Key findings:**

1. **h-main (CONFIRMED)** — Bracket K=1 correctly identifies **TP=8/1inst** on L40S qwen3-14b across 5/5 search seeds. This is the first time bracket identified a *different* TP winner than llama — confirmed model-agnostic. Phase 1 scores match predictions exactly. `evals_to_best=4` (not 3 as predicted — TP=8 is the 4th in evaluation order, vs TP=4 being 3rd for llama).

2. **h-robustness (CONFIRMED)** — **25/25** (blis_seed × search_seed) combinations correctly identify TP=8/1inst, including near-zero margin seeds (43: 0.03%, 45: 0.11%). `evals_to_best=3` at seeds 43/45/46 because the runner-up TP=4 itself exceeds the seed=42 calibrated threshold at those seeds, and `evals_to_best=4` at seeds 42/44.

3. **h-control-negative (CONFIRMED)** — 5/5 seeds correctly identify **TP=4/2inst** on H100 qwen3-14b (`evals_to_best=3`). The L40S-specific TP=8 effect completely vanishes on H100, confirming the bandwidth-driven mechanism.

4. **h-ablation (CONFIRMED)** — Flat TPE on L40S qwen achieves median `evals_to_best=6` (4/5 seeds reach exact global best, 5/5 within 1%). Better than llama's L40S performance (1/5) due to wider 5.1% margin, but bracket K=1 (deterministic evals=4) is still faster and more reliable.

**New principles extracted:** RP-21 (model-dependent TP winner on L40S), RP-22 (TPE reliability vs margin width), plus updates to RP-15, RP-16, RP-9.