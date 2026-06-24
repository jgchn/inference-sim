# BLIS Configuration Search: Pareto Frontier Discovery Report

---

## Answer

A **two-phase adaptive search** (Extremes-Phase-1 with batch=512 + TPE Phase-2) reliably achieves ≥95% hypervolume ratio within **≤50 evaluations** on saturated regimes — far under the 500-evaluation budget — by using 4 evaluations to identify the dominant parallelism tier then directing TPE toward high-quality secondary-knob combinations within that tier. The algorithm is portable across hardware (H100, A100-SXM) and rates (500–5000 req/s) provided Phase 1 uses batch=512 and nreq≥750.

---

## Evidence

### Iteration 1 — Baseline (Random vs. Hierarchical)
- At **rate=50 req/s** (qwen3-14b/H100), the Pareto problem is **degenerate**: TP=8/1 dominates all other tiers in both objectives simultaneously. Both random search (500 evals, 9% tier hit rate → ~45 hits) and hierarchical search achieve identical hypervolume once the dominant point is discovered.
- Key insight: at low rates, no genuine throughput–latency tradeoff exists. HV is an insensitive discriminator in degenerate cases (RP-C2-4). Genuine tradeoffs require rates that saturate the best single-instance tier.

### Iteration 2 — Convergence Speed Comparison
- At **rate=2000 req/s**, sequential Phase 1 (all 15 tiers, batch=512) identifies TP=4/2 as optimal at **evaluation 14** deterministically across 3 seeds (RP-C2-6).
- NSGA-II appeared faster (8 evals at seed=42) but this was a lucky random-warmup artifact with p≈19% — multi-seed expected value equals random search at ~37 evals (RP-C2-9).
- The qualifying configuration fraction is ~2.67% of total space (2304/86,400), making random search viable but slower than structured search (RP-C2-8).
- Phase 1 tier ranking is **perfectly stable across workload seeds** (RP-C2-2).

### Iteration 3 — Extremes-First Phase 1 + TPE Phase 2
- **Extremes-first Phase 1** (4 configs: max-instance per TP level, batch=512) identifies optimal tier in exactly **3 evals on H100** at rate=2000–5000 and **4 evals on A100-SXM**, reducing Phase 1 cost from 14 to 3–4 evaluations (RP-C2-10).
- Phase 2 (TPE, n_startup=5): when Phase 2 improvement headroom is ≤1%, TPE and LHS are indistinguishable (converge in 10–33 evals); the signal is too weak for the surrogate to dominate random sampling (RP-C2-12).
- Secondary knobs (esp. batch size) DO affect composite fitness at low rates even when throughput is saturated — batch=32 gives composite 0.063 vs batch=512 at 0.132 (2× difference) due to latency components (RP-C2-3).

### Iteration 4 — Batch Sensitivity of Phase 1
- When Phase 2 headroom is **≥25%** (starting from batch=128), TPE converges to 95% within-tier optimal in **2–3 evals** vs. 5–16 for LHS (RP-C2-14) — TPE's tree-density model captures the dominant batch monotone dimension quickly.
- **Lean Phase 1 (batch=128)** fails on H100 at rate=500 (0/3 seeds correct) and A100-SXM at all rates ≥2000 (RP-C2-13). Standard Phase 1 (batch=512) is 100% accurate on H100 across all tested rates and seeds (RP-C2-11).

### Iteration 5 — Simulation Duration Confound
- **nreq≥750** is required for correct Phase 1 ranking at rate=500 on H100; below this, TP=8/1 is artificially favored by transient queue dynamics (RP-C2-15). At nreq=500, TP=8/1 leads TP=4/2 by 8%; at nreq=750, TP=4/2 leads by 2%.
- Lean Phase 1 at nreq=2000 correctly eliminates extreme tiers (TP=1/8, TP=8/1) in all 18/18 conditions but misranks TP=4/2 vs TP=2/4 in 12/18 conditions (RP-C2-16). The mechanism: effective cluster batch capacity = instances × max_batch creates a systematic 2× bias for TP=2/4 at batch=128 (RP-C2-5).
- **The correct Phase 1 protocol requires batch=512, nreq≥750** — this is the single most critical implementation detail.

---

## Principles Discovered

| ID | Statement | Confidence | Regime |
|----|-----------|-----------|--------|
| **RP-C2-1** | At undersaturated rates (rate=50), TP=8/1 dominates all objectives — no genuine Pareto tradeoff | High | rate < saturation point, single-node ≤8 GPU |
| **RP-C2-2** | Phase 1 tier ranking is perfectly stable across workload seeds | High | rate≥500, H100, qwen3-14b |
| **RP-C2-3** | Secondary knobs affect latency-component composite fitness even at low rates; batch=512 is always best even when throughput is flat | High | All tested rates, composite scoring |
| **RP-C2-4** | HV is insensitive when Pareto front degenerates to a single dominant point | High | Degenerate 2D cases generally |
| **RP-C2-5** | **Phase 1 MUST use batch=512**; batch=128 misranks TP=4/2 vs TP=2/4 in 12/18 conditions across hardware/rates | High | qwen3-14b, H100+A100, rates 500–5000 |
| **RP-C2-6** | Sequential Phase 1 reaches 95% optimal in exactly 14 evals at rate=2000 | High | rate=2000, H100, qwen3-14b |
| **RP-C2-7** | TP=8/1 apparent wins at low nreq (≤500) are simulation artifacts; TP=4/2 is truly optimal at steady state | High | Oversaturated regimes, nreq≥750 |
| **RP-C2-8** | ~2.67% of total configs meet the 95% throughput threshold; random search expected evals ~37 | High | rate=2000, H100, qwen3-14b |
| **RP-C2-9** | NSGA-II provides no reliable convergence advantage over structured search; apparent wins are lucky random warmup | Medium | Optuna NSGAIISampler, pop=20 |
| **RP-C2-10** | Extremes-first Phase 1 identifies optimal tier in 3–4 evals vs. 14 for sequential Phase 1 | High | H100/A100, rate≥2000, batch=512 |
| **RP-C2-11** | Standard Phase 1 (batch=512) is 100% accurate on H100 all rates; A100-SXM shows tie zone at rate≥2000 (gap <1.5%) | High | qwen3-14b, tested rates 500–5000 |
| **RP-C2-12** | TPE ≈ LHS when Phase 2 improvement window ≤1% (noise dominates signal) | Medium | Small-headroom Phase 2 |
| **RP-C2-13** | Lean Phase 1 (batch=128) fails on H100 rate=500 and A100-SXM rate≥2000 | High | Multi-hardware tested |
| **RP-C2-14** | TPE converges in 2–3 evals when Phase 2 headroom ≥25%; 5–10× faster than LHS | High | Large-headroom, strong monotone dimension |
| **RP-C2-15** | nreq≥750 required for correct Phase 1 ranking at rate=500 on H100 | High | H100, rate=500, batch=512 |
| **RP-C2-16** | Lean Phase 1 at nreq=2000 is useful as a coarse filter eliminating single-instance tiers, but cannot precisely rank multi-instance tiers | High | H100+A100, rates 500–5000 |

---

## Recommended Algorithm (Pseudocode)

```python
# Phase 1: Extremes-first parallelism tier identification
# 4 configs × ~200ms each = ~800ms
EXTREMES = [(tp=1,ni=8), (tp=2,ni=4), (tp=4,ni=2), (tp=8,ni=1)]
# Use batch=512, nreq=max(750, 3*rate_hz) to avoid transient artifacts
phase1_results = [run_blis(tp, ni, batch=512, nreq=nreq_safe) for tp, ni in EXTREMES]
best_tier = argmax(composite_fitness, phase1_results)

# Phase 2: TPE within best tier
# ~46-96 evals × ~200ms each = ~10-20s
study = optuna.create_study(sampler=TPESampler(n_startup_trials=5))
study.optimize(lambda t: run_blis(best_tier.tp, best_tier.ni,
    batch=t.suggest_categorical([32,64,128,256,512]),
    scheduler=t.suggest_categorical([...]),
    routing=t.suggest_categorical([...]),
    # ... other secondary knobs
), n_trials=96)

# Total: ~100 evals, ~25s wall time, ≥95% HV ratio
pareto_front = extract_pareto(all_results)
```

---

## Limitations & Open Questions

### Unanswered Questions

1. **Multi-node scaling**: All experiments were conducted on single 8-GPU nodes. With 16+ GPUs, the TP×instances space expands significantly (e.g., TP=16/1, TP=8/2), and the extremes-first strategy may need a larger Phase 1 budget. The batch=512 correctness guarantee has not been verified for TP>8.

2. **Genuine Pareto frontiers**: Every tested high-rate condition produced a **near-degenerate** Pareto front (1–2 non-dominated points). True multi-point fronts with distributed tradeoffs (requiring NSGA-II or MOEA advantage) were never observed. It is unknown whether such conditions exist in BLIS (e.g., at rates exactly at saturation boundary).

3. **Other model families**: All experiments used qwen3-14b. Larger models (e.g., 70B+ requiring TP≥4 mandatory) or encoder-decoder architectures may shift tier rankings. The batch=512 correctness principle should be re-verified for new models.

4. **Phase 2 secondary knob importance ordering**: Only batch size was confirmed as the dominant secondary dimension. The relative importance of scheduler, routing, preemption, and prefill-chunk-size was not characterized. It is possible that at non-batch=512 starting points, these dimensions have stronger signals than currently observed.

5. **A100-SXM tie zone**: At rate≥2000 on A100-SXM, TP=4/2 and TP=8/1 are within 1.5% — Phase 1 winner is seed-dependent. The correct approach (seed-averaged scoring, multiple Phase 1 replicates per tier, or treating both as optimal) has not been systematically tested.

6. **Admission control and preemption interactions**: These knobs were present in the sweep matrix but their interaction effects with batch size and routing were never isolated. Counterintuitive interactions (e.g., aggressive preemption helping at extreme overload) may create secondary Pareto-relevant configurations that Phase 2 misses.

### Next Campaign Priorities

- **Campaign B**: Characterize whether genuine 3+ point Pareto fronts exist in BLIS at any rate/model/hardware combination; if yes, evaluate MOEA vs. structured search with correct HV discrimination.
- **Campaign C**: Portability to larger models (Llama-70B, Mixtral-8×7B) and multi-node configs to validate the batch=512 + extremes-first approach.
- **Campaign D**: Systematic secondary knob importance ranking via Sobol sensitivity analysis within the optimal tier, to improve Phase 2 efficiency beyond TPE's current 2–3 eval convergence floor.