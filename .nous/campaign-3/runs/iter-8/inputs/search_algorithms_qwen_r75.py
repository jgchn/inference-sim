#!/usr/bin/env python3
"""h-main: NSGA-II + Random search on Qwen3-14B at rate=75.

Measures convergence gap at the 95% HV threshold against exhaustive reference.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from blis_common import (run_nsga2, run_random, mc_hypervolume,
                         FIXED_FLAGS_QWEN_R75)

RESULTS_DIR = '/Users/jchen/go/src/inference-sim/inference-sim/.nous/campaign-3/runs/iter-8/results/h-main'
EXHAUSTIVE_RESULTS = '/Users/jchen/go/src/inference-sim/inference-sim/.nous/campaign-3/runs/iter-8/results/h-robustness/pareto_front_qwen_r75.json'


def interpolate_95_threshold(checkpoints):
    """Find the eval count where HV first reaches 95% of exhaustive (linear interpolation)."""
    for i, cp in enumerate(checkpoints):
        if cp['pct_of_exhaustive'] is not None and cp['pct_of_exhaustive'] >= 95.0:
            if i == 0:
                return cp['eval']
            prev = checkpoints[i - 1]
            if prev['pct_of_exhaustive'] is None:
                return cp['eval']
            # Linear interpolation
            pct_prev = prev['pct_of_exhaustive']
            pct_curr = cp['pct_of_exhaustive']
            eval_prev = prev['eval']
            eval_curr = cp['eval']
            if pct_curr == pct_prev:
                return eval_curr
            frac = (95.0 - pct_prev) / (pct_curr - pct_prev)
            return eval_prev + frac * (eval_curr - eval_prev)
    return None  # Never reached 95%


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load exhaustive reference
    if not os.path.exists(EXHAUSTIVE_RESULTS):
        sys.exit(f"ERROR: exhaustive results not found at {EXHAUSTIVE_RESULTS}. Run h-robustness first.")
    with open(EXHAUSTIVE_RESULTS) as f:
        exh_data = json.load(f)
    exhaustive_hv = exh_data['exhaustive_hv']
    pareto_density = exh_data['pareto_density']
    print(f"Qwen rate=75 exhaustive: HV={exhaustive_hv:.6e}, density={100*pareto_density:.2f}%", flush=True)

    # Run NSGA-II
    print("\n=== NSGA-II ===", flush=True)
    nsga2_nd, nsga2_cp = run_nsga2(
        FIXED_FLAGS_QWEN_R75, pop_size=40, n_gens=4, mutation_rate=0.15,
        rng_seed=42, checkpoint_every=40, exhaustive_hv=exhaustive_hv,
        cache_key_prefix='qwen_r75')

    # Run Random
    print("\n=== Random Search ===", flush=True)
    random_nd, random_cp = run_random(
        FIXED_FLAGS_QWEN_R75, budget=200, rng_seed=43, checkpoint_every=40,
        exhaustive_hv=exhaustive_hv, cache_key_prefix='qwen_r75')

    # Compute gap at 95% threshold
    nsga2_95 = interpolate_95_threshold(nsga2_cp)
    random_95 = interpolate_95_threshold(random_cp)

    gap = None
    if nsga2_95 is not None and random_95 is not None:
        gap = random_95 - nsga2_95  # positive = NSGA-II faster

    nsga2_final_hv = mc_hypervolume(nsga2_nd)
    random_final_hv = mc_hypervolume(random_nd)

    print(f"\n=== Results ===", flush=True)
    print(f"NSGA-II reaches 95% at eval ~{nsga2_95:.1f}" if nsga2_95 else "NSGA-II: never reached 95%", flush=True)
    print(f"Random reaches 95% at eval ~{random_95:.1f}" if random_95 else "Random: never reached 95%", flush=True)
    print(f"Gap at 95%: {gap:.1f} evals (positive = NSGA-II faster)" if gap else "Gap: N/A", flush=True)
    print(f"NSGA-II final HV: {nsga2_final_hv:.6e} ({100*nsga2_final_hv/exhaustive_hv:.2f}%)", flush=True)
    print(f"Random final HV: {random_final_hv:.6e} ({100*random_final_hv/exhaustive_hv:.2f}%)", flush=True)

    # Save results
    result = {
        'model': 'qwen/qwen3-14b',
        'rate': 75,
        'pareto_density': pareto_density,
        'exhaustive_hv': exhaustive_hv,
        'nsga2': {
            'checkpoints': nsga2_cp,
            'final_hv': nsga2_final_hv,
            'final_hv_pct': 100 * nsga2_final_hv / exhaustive_hv,
            'eval_at_95pct': nsga2_95,
        },
        'random': {
            'checkpoints': random_cp,
            'final_hv': random_final_hv,
            'final_hv_pct': 100 * random_final_hv / exhaustive_hv,
            'eval_at_95pct': random_95,
        },
        'convergence_gap': gap,
        'gap_interpretation': 'positive = NSGA-II reaches 95% first; negative = random reaches 95% first',
    }

    outpath = os.path.join(RESULTS_DIR, 'convergence_qwen_r75.json')
    with open(outpath, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {outpath}", flush=True)


if __name__ == '__main__':
    main()
