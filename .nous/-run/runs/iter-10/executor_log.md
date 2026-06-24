Validation passes. Here's a summary of all findings:

---

## Experiment Complete — Iteration 10 Summary

**All 3 arms confirmed, validation passed.**

### h-main (CONFIRMED)
Lean bracket (TP={4,8}, max-profile, budget=2) achieved 8/8 correct on H100 workload presets:
- **chatbot rate=100**: TP=8 wins (0.200393), evals_to_best=2 ✓
- **chatbot rate=500**: TP=4 wins (0.268650), evals_to_best=1 ✓  
- **contentgen, summarization, multidoc at both rates**: TP=4 wins at ALL rates (not just >125 req/s) — confirming that prompt mean ≥1024 eliminates the rate-dependent crossover
- All 5 seeds identical (deterministic), all scores match global best exactly

### h-control-negative (CONFIRMED)
Model-aware formula accuracy drops from 100% (default workload) to **5/8 on H100** (62.5%) and **4/8 on A100-SXM** (50%) with workload presets. Fails exactly as predicted: rate=100 for long-prompt workloads (formula predicts TP=8 based on rate < crossover=119.4, but TP=4 wins because longer prompts shift the effective crossover below 100 req/s). A100-SXM multidoc rate=500 adds an additional failure: TP=4 predicted but TP=8 wins due to KV cache capacity constraints.

### h-robustness (CONFIRMED)
Lean bracket achieved 8/8 correct on A100-SXM workload presets (40/40 seeds):
- TP=8 wins on chatbot (both rates) and multidoc (both rates — KV cache capacity anomaly correctly captured)
- TP=4 wins on contentgen and summarization (both rates)
- Near-tied chatbot rate=500 (0.18% margin): correctly resolved with evals_to_best=1

**New principles extracted:** RP-33 (workload-conditional TP crossover), updates to RP-24 (crossover is workload-conditional), RP-27 (formula workload blindness quantified), RP-28 (lean bracket now validated on 155/155 total runs including workload presets).