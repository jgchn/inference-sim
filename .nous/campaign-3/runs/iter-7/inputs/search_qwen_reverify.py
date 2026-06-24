#!/usr/bin/env python3
"""h-ablation: NSGA-II + Random re-verification on Qwen3-14B at rate=50.

Same session as Llama arms. Uses iter-5 exhaustive HV (58570200000.0) as reference.
Uses same RNG seeds as iter-5 (nsga2=42, random=43) for deterministic comparison.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from blis_common import run_nsga2, run_random, FIXED_FLAGS_QWEN

RESULTS_DIR = '/Users/jchen/go/src/inference-sim/inference-sim/.nous/campaign-3/runs/iter-7/results/h-ablation'

# iter-5 established exhaustive HV for Qwen3-14B at rate=50 (deterministic — BLIS INV-6)
QWEN_EXHAUSTIVE_HV = 58570200000.0


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"h-ablation: Qwen3-14B re-verification at rate=50", flush=True)
    print(f"  Reference exhaustive HV (iter-5): {QWEN_EXHAUSTIVE_HV:.6e}", flush=True)

    # Run NSGA-II on Qwen (seed=42, same as iter-5)
    print("\n--- NSGA-II on Qwen3-14B ---", flush=True)
    nsga2_nd, nsga2_checkpoints = run_nsga2(
        fixed_flags=FIXED_FLAGS_QWEN,
        pop_size=40,
        n_gens=4,
        mutation_rate=0.15,
        rng_seed=42,
        checkpoint_every=40,
        exhaustive_hv=QWEN_EXHAUSTIVE_HV,
        cache_key_prefix='qwen',
    )

    # Run Random on Qwen (seed=43, same as iter-5)
    print("\n--- Random search on Qwen3-14B ---", flush=True)
    rand_nd, rand_checkpoints = run_random(
        fixed_flags=FIXED_FLAGS_QWEN,
        budget=200,
        rng_seed=43,
        checkpoint_every=40,
        exhaustive_hv=QWEN_EXHAUSTIVE_HV,
        cache_key_prefix='qwen',
    )

    nsga2_final_hv = nsga2_checkpoints[-1]['hv'] if nsga2_checkpoints else 0.0
    nsga2_final_pct = nsga2_checkpoints[-1]['pct_of_exhaustive'] if nsga2_checkpoints else None
    rand_final_hv = rand_checkpoints[-1]['hv'] if rand_checkpoints else 0.0
    rand_final_pct = rand_checkpoints[-1]['pct_of_exhaustive'] if rand_checkpoints else None

    print(f"\nQwen re-verification summary:", flush=True)
    print(f"  NSGA-II final HV:     {nsga2_final_hv:.6e} ({nsga2_final_pct:.2f}% of exhaustive)",
          flush=True)
    print(f"  Random final HV:      {rand_final_hv:.6e} ({rand_final_pct:.2f}% of exhaustive)",
          flush=True)
    print(f"  iter-5 NSGA-II ref:   5.8106e+10 (99.21%)", flush=True)
    print(f"  iter-5 Random ref:    5.5242e+10 (94.32%)", flush=True)

    # Save results
    summary = {
        'model': 'qwen/qwen3-14b',
        'reference_exhaustive_hv': QWEN_EXHAUSTIVE_HV,
        'nsga2': {
            'total_evals': nsga2_checkpoints[-1]['eval'] if nsga2_checkpoints else 0,
            'final_hv': nsga2_final_hv,
            'final_pct_of_exhaustive': nsga2_final_pct,
            'pareto_size': len(nsga2_nd),
            'convergence': nsga2_checkpoints,
        },
        'random': {
            'total_evals': rand_checkpoints[-1]['eval'] if rand_checkpoints else 0,
            'final_hv': rand_final_hv,
            'final_pct_of_exhaustive': rand_final_pct,
            'pareto_size': len(rand_nd),
            'convergence': rand_checkpoints,
        },
    }
    with open(os.path.join(RESULTS_DIR, 'pareto_front.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Also save individual convergence for consistency
    with open(os.path.join(RESULTS_DIR, 'convergence.json'), 'w') as f:
        json.dump({
            'nsga2': nsga2_checkpoints,
            'random': rand_checkpoints,
        }, f, indent=2)

    print(f"Results saved to {RESULTS_DIR}", flush=True)


if __name__ == '__main__':
    main()
