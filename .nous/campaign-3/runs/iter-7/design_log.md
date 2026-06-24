Validation passed.

## Summary

**Iteration 7 design complete.** The hypothesis bundle tests **model portability** of NSGA-II's convergence advantage:

- **h-robustness**: Exhaustive sweep of all 1800 configs with Llama-3.1-8B at rate=50 → establishes ground truth Pareto density and HV
- **h-main**: NSGA-II on Llama → tests whether convergence advantage transfers (predicted: yes, but with smaller gap than Qwen's 91 evals due to higher density)
- **h-control-negative**: Random search on Llama → baseline comparison
- **h-ablation**: Re-verify Qwen3-14B NSGA-II/random in same session → eliminates infrastructure confounds

Key prediction: Llama's weaker KV stress interaction (40x TTFT improvement from block_size vs 92x for Qwen) will produce higher Pareto density (6-12% vs Qwen's 4.5%), reducing NSGA-II's convergence gap to 40-70 evals while preserving its directional advantage.