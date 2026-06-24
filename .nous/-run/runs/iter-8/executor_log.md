All 5 phases complete. Here's a summary of the findings:

---

## Iteration 8 Results Summary

**h-main — CONFIRMED (25/25 runs at global best)**  
Lean bracket generalizes to qwen3-32b across all 5 regimes on H100 and A100-SXM. The algorithm adapts empirically to the model-size-shifted crossover (~350 req/s for 32B on H100 vs ~130 for 14B) without any modification. evals_to_best=1 when TP=4 wins (H100 rate=500), evals_to_best=2 when TP=8 wins (all other regimes). All 5 seeds identical per regime.

**h-control-negative — PARTIALLY_CONFIRMED (15/18 = 83.3% accuracy, predicted ≤78%)**  
The RP-27 bandwidth formula does fail and fails for the predicted reasons (model-size blindness), but achieves 83.3% not ≤78%. 3 failures (not 4+): A100-SXM qwen3-32b rate=500 (no crossover exists), L40S qwen3-14b rate=500 (no crossover), L40S qwen3-32b rate=500 (no crossover). H100 qwen3-32b rate=500 is a "lucky" correct prediction because TP=4 IS the winner at rate=500, even though the formula's crossover estimate (119 vs true ~350) is wrong.

**h-ablation — CONFIRMED (4/4 misidentifications)**  
Micro-eval with 100 requests selects TP=8 on all 4 high-rate regimes where 1000-req gives TP=4. At 100 requests, TP=8 wins by 16-38% — the opposite of the 1000-req result where TP=4 wins by 0.8-1.5%. Queue saturation requires ≥1000 requests to manifest.

**h-robustness — CONFIRMED (20/20 runs at global best)**  
Lean bracket completes L40S validation: qwen3-14b→TP=8 at both rates, llama-3.1-8b→TP=8 at rate=100 / TP=4 at rate=500, all 5/5 seeds identical per regime. Lean bracket now validated across all 3 hardware platforms and 3 models.

**New principle added: RP-31** — Minimum simulation horizon for reliable TP identification at high rates is ≥1000 requests. The 100-request micro-eval completely inverts the TP winner signal.