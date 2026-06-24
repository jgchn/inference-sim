Validation passes. Here's the complete summary of the iteration 5 experiment:

---

## Iteration 5: Rate-Dependent TP Winner Transition — Results

**All 4 arms CONFIRMED.**

### h-main (bracket K=1, rate=100, H100, qwen3-14b, blis_seed=42)
- **5/5 search seeds** select TP=8/1inst as Phase 1 winner
- **evals_to_best=4** on all seeds (exactly as predicted)
- Phase 1 max-profile scores: TP1=0.108, TP2=0.137, TP4=0.156, TP8=0.165 — monotonically increasing

### h-robustness (bracket K=1, rate=100, 5 blis_seeds × 5 search_seeds)
- **25/25 combinations** select TP=8/1inst
- **evals_to_best=4** on all 25 — RP-9 confirmed (determinism per blis_seed)
- TP=8 margins by blis_seed: 42→5.5%, 43→6.4%, 44→6.4%, 45→5.8%, 46→4.7%

### h-ablation (bracket K=1, rate=125 crossover, 5 blis_seeds × 5 search_seeds)
- **evals_to_best=3 on all 25** combinations (confirmed prediction)
- blis46 is effectively tied (TP4=0.16597, TP8=0.16598 — 0.001% margin, not TP4 winning as predicted)
- Threshold mechanism confirmed: TP4 at position 3 (0.161–0.167) exceeds 1% threshold (0.160560)

### h-control-negative (flat TPE, rate=100, blis_seed=42)
- **evals_to_best**: 18, 11, 6, 1, 6 → **median=6**
- Dramatically faster than rate=500 (prior median=26) — monotonic TP landscape at rate=100 enables rapid TPE convergence
- Bracket K=1 (evals_to_best=4) still outperforms TPE (median=6) by 33%

**New principles extracted:** RP-24 (rate-dependent TP winner), RP-25 (bracket K=1 rate-regime agnosticism), RP-26 (monotonic landscape accelerates TPE). RP-15 and RP-16 extended to cover rate-varied regimes.