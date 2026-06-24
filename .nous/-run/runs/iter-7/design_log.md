Validation passed. Here's a summary of the iteration 7 design:

**Research Question**: Can bracket K=1 be reduced from 100 evaluations to 2 (Phase 1 over TP={4,8} only, no Phase 2) — a 50x budget reduction — while maintaining 1% accuracy?

**Key Evidence Gathered**:
- TP=2 gap is structurally 11% below winner across rates 100-5000 on H100 (never converges)
- TP=1 gap is 32-38% (even more excluded)
- Phase 2 contributes at most 0.71% improvement, always within the 1% tolerance
- Min-profile (mr=32) inverts the TP ranking, confirming the max-profile mechanism

**Arms**:
1. **h-main**: Lean bracket (2 evals) across 4 diverse regimes (H100/A100-SXM × rate=100/500)
2. **h-control-negative**: Min-profile lean bracket → selects wrong winner, validating that max-profile is essential
3. **h-ablation**: Lean bracket + mini Phase 2 (budget=12) on the routing-sensitive crossover regime (A100-SXM rate=200)
4. **h-robustness**: Lean bracket across 5 blis_seeds × 5 search_seeds on H100 hard

**Code Changes**: Single file (search_blis_iter7.py) — add `--tp-candidates lean` flag and `--skip-phase2` flag, plus GLOBAL_BEST_TABLE updates from iterations 5-6. No algorithmic changes to bracket K=1 core.