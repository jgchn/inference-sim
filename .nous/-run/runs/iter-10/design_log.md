Validation passes. Here's a summary of iteration 10:

**Iteration 10: Workload-Shape Sensitivity of Lean Bracket** — all 3 artifacts written and validated.

**Key finding from probing:** The TP crossover is workload-dependent. With prompt mean >= 1024 (contentgen, summarization, multidoc), TP=4/2inst wins at ALL rates on H100 — the rate-dependent crossover from RP-24 only applies to short-prompt workloads. The model-aware formula (100% on default workload) drops to 62.5% accuracy on workload presets because it has no prompt-length input.

**Three arms:**
- **h-main**: Lean bracket across 4 workload presets x 2 rates on H100 (8 regimes) — predicting 8/8 correct, workload-agnostic by construction
- **h-control-negative**: Model-aware formula fails on 3/8 H100 regimes (summarization, contentgen, multidoc at rate=100) — validates that the mechanism is prompt-length-driven
- **h-robustness**: Lean bracket on A100-SXM (8 regimes) including the multidoc anomaly (TP=8 wins by >100% due to KV cache capacity) and near-tied chatbot rate=500 (0.18% margin)