All artifacts written and validation passes. Here's the summary:

---

**Design complete.** Three artifacts written to `.nous/campaign-3/runs/iter-1/`:

- **problem.md** — Frames the multi-objective search problem over BLIS's ~294K configuration space, with validated baseline command (rps=7.41, ttft_p99=33.2ms at 52ms/eval)
- **bundle.yaml** — Three-arm hypothesis bundle:
  - **h-main (NSGA-II):** 500 evaluations with evolutionary selection pressure should outperform random on hypervolume
  - **h-control-negative (Random):** 500 uniform samples as baseline
  - **h-robustness (LHS):** 500 stratified samples to isolate coverage vs. selection pressure
- **handoff_snapshot.md** — Full exploration context for executor agent (validated commands, dead ends, code map, warnings)

Key design decisions: rate=20 creates a 640x TTFT range across configs, giving a rich Pareto frontier. Serial evaluation of 2000 configs fits in ~104s (well within 3-minute budget), so no parallelism needed for iteration 1.