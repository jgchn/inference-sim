Validation passes. All artifacts are correct.

Here's a summary of the iteration 9 results:

**All 4 arms confirmed:**

- **h-main (30/30):** Lean bracket (TP={4,8}, max-profile, budget=2) generalizes perfectly to llama-3.1-70b on all 6 regimes. TP=8 wins with margins 9-20% across all hardware. evals_to_best=2 everywhere (deterministic). Scores exactly match global best.

- **h-control-negative:** Old formula (400/BW) = 18/24 = 75% (3 failures on 70B at rate=500). Model-aware formula (load_index = params/BW, threshold=12) = 24/24 = 100%. All 6 failures of the old formula occur precisely at load_index > 12 regimes. 70B load indices: H100=20.90, A100-SXM=34.33, L40S=81.02.

- **h-ablation (6/6):** Default-profile (mr=256 vs max mr=512) still correctly selects TP=8 on all 70B regimes. Scores are 15-19% below max-profile (outside 1% threshold) but TP winner unchanged — confirming the advantage is structural weight-transfer physics, not batch-size-dependent queue dynamics.

- **h-robustness (50/50):** TP=8 wins across all blis seeds 42-46 on H100 and L40S. Minimum margin: 2.4% (H100 blis_seed=45). All 5 search seeds identical per blis_seed (deterministic).

**New principles:** RP-28 (lean bracket now validated on 4th model family, 75/75 total), RP-27 (model-aware formula reaches 100% via load_index threshold), RP-32 (70B has no crossover on any hardware), RP-20 update (70B robustness confirmed).