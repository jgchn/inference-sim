Validation passes. Let me provide a summary of the findings:

---

## Experiment Complete — Iteration 6: Cross-Hardware Rate Crossover on A100-SXM

**All 4 arms CONFIRMED. Validation: PASS.**

### Key Results

**h-main** (A100-SXM qwen3-14b, rate=100, bracket K=1):
- 5/5 seeds → TP8/1inst winner, evals_to_best=4 deterministically
- Phase 1 scores exactly as predicted: TP1=0.0804, TP2=0.1055, TP4=0.1275, TP8=0.1308 (2.6% margin)
- best_score=0.130754 matches global best on all seeds

**h-robustness** (A100-SXM llama3.1-8b, rate=100, 5×5 blis/search seeds):
- 25/25 combinations → TP8/1inst, evals_to_best=4 on every single combination
- Margins per blis_seed: 4.3%, 5.2%, 5.0%, 4.0%, 3.7% — all well above 1% threshold
- RP-9 (Phase 1 determinism) confirmed for llama on A100-SXM

**h-ablation** (A100-SXM qwen3-14b, rate=200 crossover, bracket K=1):
- 5/5 seeds → evals_to_best=3, TP4 winner at crossover where TP4=TP8=0.1345
- Phase 2 TPE found round-robin routing improvement: best_score=0.135440 > global_best=0.134935

**h-control-negative** (A100-SXM qwen3-14b, rate=100, flat TPE):
- 5/5 seeds find global best; median evals_to_best=6 (range 1-18)
- Bracket K=1 (deterministic 4) outperforms flat TPE (median 6) by 33%

### New Principles Extracted
- **RP-24 updated**: Rate-dependent TP crossover confirmed cross-hardware: A100-SXM crossover at ~200 req/s vs H100's ~125 req/s
- **RP-25 updated**: Bracket K=1 hardware-agnostic rate-adaptiveness confirmed on A100-SXM
- **RP-26 updated**: Monotonic landscape at rate=100 accelerates TPE (median=6) vs rate=500 (median=26) — confirmed on A100-SXM
- **RP-27 new**: Crossover rate scales inversely with memory bandwidth: H100 (3.35 TB/s)→~125, A100-SXM (2.039 TB/s)→~200, L40S (0.864 TB/s)→~450+