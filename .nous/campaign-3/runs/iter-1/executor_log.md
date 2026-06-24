Validation passes. Here's a summary of findings:

**All three arms were REFUTED** by the same root cause: the workload regime (rate=20 req/s) is non-saturating for most configurations.

**Key observations:**
- All three methods (random, NSGA-II, LHS) achieved identical hypervolume = **249,720.535**
- The "Pareto front" is a degenerate single point: **(rps=12.50, ttft_p99=22.73ms)**
- ~24% of randomly sampled configs hit this optimum — random found it at **eval #1**
- NSGA-II's hypervolume was flat from generation 0 through generation 10 — selection pressure never engaged
- 500 evaluations at ~60ms each = 30s wall time per condition

**Principles extracted:**
- **RP-1**: Multi-objective search comparisons require a non-trivial Pareto frontier — at rate=20 req/s, the space is degenerate (single dominant point)
- **RP-2**: Before running a search comparison, probe 10-20 configs to verify objective diversity (>5x range)  
- **RP-3**: At rate=20 with qwen/qwen3-14b on H100, configs with TP*instances≥2 are all unsaturated and produce identical metrics; meaningful tradeoffs emerge at rate≥80 req/s

**Iteration 2 recommendation:** Rerun at rate=80-150 req/s to create genuine queueing saturation where NSGA-II's selection pressure can engage.