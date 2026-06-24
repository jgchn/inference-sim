Validation passes.

## Summary

**Iteration 8 designs a 4-arm hypothesis bundle testing lean bracket generalization to larger models and the limits of the bandwidth formula.**

Key findings from exploration:
- **qwen3-32b crossover is ~350 on H100** (vs ~130 for 14B), proving model size matters
- **No crossover on A100-SXM or L40S for 32B** — TP=8 dominates at all rates
- **RP-27 formula (`400/BW`) fails on 4/18 regimes** (22% error rate), all failures involve 32B or L40S
- **Micro-eval (≤500 requests) cannot identify the TP=4 winner** — queue saturation needs sustained load

The bundle:
1. **h-main**: Lean bracket on qwen3-32b across H100/A100-SXM at 5 regimes — tests model-size generalization
2. **h-control-negative**: Bandwidth formula accuracy across 18 regimes — demonstrates model-size blindspot
3. **h-ablation**: Micro-eval (100 req) vs standard (1000 req) — establishes minimum viable simulation horizon
4. **h-robustness**: Lean bracket on L40S for both existing models — completes hardware platform coverage