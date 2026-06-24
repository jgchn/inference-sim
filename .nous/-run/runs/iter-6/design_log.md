Validation passes. Here's a summary of the iteration 6 design:

**Research Question**: Does the rate-dependent TP winner transition exist on A100-SXM, and does bracket K=1 adapt without modification?

**Key Discovery from Probing**: The crossover is universal and scales inversely with memory bandwidth:
- H100 (3.35 TB/s): crossover at ~125 req/s
- A100-SXM (2.039 TB/s): crossover at ~200 req/s  
- L40S (0.864 TB/s): crossover at ~450 req/s (llama) or none (qwen)

**Four Arms**:
1. **h-main**: Bracket K=1 on A100-SXM qwen at rate=100 — predict evals_to_best=4 (TP=8 wins by 2.5%)
2. **h-robustness**: Multi-seed (25 combos) bracket K=1 on A100-SXM llama at rate=100 — predict 25/25 correct (TP=8 wins by 4.3%)
3. **h-ablation**: Bracket K=1 at A100-SXM crossover rate=200 — predict evals_to_best=3 (TP=4 and TP=8 tied, mirrors H100's rate=125 crossover)
4. **h-control-negative**: Flat TPE on A100-SXM at rate=100 — predict median evals_to_best < 15, but bracket K=1 (4) still wins