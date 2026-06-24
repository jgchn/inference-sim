Validation passes. Here's a summary of what was found:

---

## Iteration 5 Results Summary

### What was tested
360 BLIS evaluations across 4 experimental arms, testing whether `nreq=2000` eliminates the need for `batch=512` in Phase 1 tier ranking.

### Key Findings

**h-main: REFUTED** — Lean Phase 1 (`batch=128`) at `nreq=2000` agrees with standard Phase 1 (`batch=512`) in only **6/18 conditions (33%)**, far below the predicted 89%. Lean consistently selects TP=2/4 instead of TP=4/2. The mechanism: at `batch=128`, TP=2/4 has 4×128=512 effective concurrent capacity vs TP=4/2's 2×128=256 — a 2× advantage that persists in steady state. At `batch=512`, both exceed queue demand and TP=4/2's compute efficiency dominates.

**h-control-negative: CONFIRMED** — At `nreq=250`, standard Phase 1 disagrees with the `nreq=2000` reference in **18/18 conditions (100%)**. TP=8/1 dominates every single condition due to transient dynamics.

**h-ablation: PARTIALLY CONFIRMED** — `nreq` IS the dominant factor for the TP=8/1 vs multi-instance boundary: `{batch=512, nreq=500}` → TP=8/1 (3/3 seeds), `{batch=128+512, nreq=2000}` → multi-instance. But within multi-instance, batch size still matters: `batch=128/nreq=2000` picks TP=2/4 while `batch=512/nreq=2000` picks TP=4/2.

**h-robustness: CONFIRMED** — Transition from TP=8/1 to TP=4/2 dominant occurs between nreq=500 (TP=8/1 leads by 8%) and nreq=750 (TP=4/2 leads by 2%, the tie zone). nreq≥1000 gives stable TP=4/2 selection.

### Principle Updates
- **RP-C2-5** updated: `batch=512` remains required even at `nreq=2000`, but for a new reason — it prevents the effective-cluster-capacity artifact (instances × max_batch) from biasing TP=2/4 over TP=4/2.
- **RP-C2-15** (new): nreq≥750 required for correct ranking at rate=500/H100; transition threshold is nreq=750.
- **RP-C2-7** updated: TP=8/1's transient advantage confirmed as simulation-duration artifact, not rate-dependent.
- **RP-C2-16** (new): Lean Phase 1 at nreq=2000 is a valid coarse filter eliminating TP=1/8 and TP=8/1, converging to the competitive {TP=4/2, TP=2/4} set — but cannot distinguish between them.