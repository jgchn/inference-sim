All 5 phases complete. Here's a summary of the iteration 9 findings:

**Validation: PASS**

**Key findings from this iteration:**

- **h-robustness (PARTIALLY_CONFIRMED):** Qwen NDR (35.6%) and Llama NDR (35.75%) are nearly identical at rate=100 — the prediction that Qwen would have higher overall NDR was wrong. However, `kv_blocks` is the top per-axis NDR for **both** models (~0.71-0.73), confirming the KV cliff mechanism is present in both.

- **h-main (CONFIRMED):** Qwen NSGA-II gap = **+62.3 evals** at rate=100, exceeding the predicted ≥+31. The pattern strengthens with load as expected.

- **h-ablation (REFUTED):** Llama NSGA-II gap = **+40.3 evals** (NSGA-II wins), not the predicted negative gap. Llama's Llama-random-wins pattern breaks at rate=100 as it enters a KV-stressed regime.

- **h-control-negative (REFUTED):** Removing the KV cliff still leaves NSGA-II with a **+29 eval advantage** — multiple other landscape axes (batch size NDR≈0.50, block_size NDR≈0.26, scheduler NDR≈0.22) collectively preserve NSGA-II's structural advantage.

**Updated principles (RP-11, RP-14, RP-15, RP-16, RP-1):** The main revision is that NSGA-II has a universal advantage in BLIS at high load due to multi-axis landscape structure, not solely KV threshold effects. RP-11 now captures the rate-dependent breakpoint for Llama.