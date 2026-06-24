#!/usr/bin/env python3
"""h-control-negative: Random search on 1800-config space with Llama-3.1-8B at rate=50.

200 uniform random evals. Tracks HV convergence vs exhaustive ref.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from blis_common import run_random, FIXED_FLAGS_LLAMA

RESULTS_DIR = '/Users/jchen/go/src/inference-sim/inference-sim/.nous/campaign-3/runs/iter-7/results/h-control-negative'
EXHAUSTIVE_RESULTS = '/Users/jchen/go/src/inference-sim/inference-sim/.nous/campaign-3/runs/iter-7/results/h-robustness/pareto_front.json'


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load exhaustive HV from h-robustness
    exhaustive_hv = None
    if os.path.exists(EXHAUSTIVE_RESULTS):
        with open(EXHAUSTIVE_RESULTS) as f:
            data = json.load(f)
        exhaustive_hv = data.get('exhaustive_hv')
        print(f"Loaded exhaustive HV: {exhaustive_hv:.6e}", flush=True)
    else:
        print("WARNING: exhaustive results not found; pct_of_exhaustive will be null", flush=True)

    print("h-control-negative: Random search on Llama-3.1-8B, 200 evals", flush=True)
    current_nd, checkpoints = run_random(
        fixed_flags=FIXED_FLAGS_LLAMA,
        budget=200,
        rng_seed=43,
        checkpoint_every=40,
        exhaustive_hv=exhaustive_hv,
        cache_key_prefix='llama',
    )

    final_hv = checkpoints[-1]['hv'] if checkpoints else 0.0
    final_pct = checkpoints[-1]['pct_of_exhaustive'] if checkpoints else None
    total_evals = checkpoints[-1]['eval'] if checkpoints else 0

    print(f"\nRandom search summary:", flush=True)
    print(f"  Total evals:        {total_evals}", flush=True)
    print(f"  Final HV:           {final_hv:.6e}", flush=True)
    print(f"  Final pareto size:  {checkpoints[-1]['pareto_size'] if checkpoints else 0}", flush=True)
    if final_pct is not None:
        print(f"  % of exhaustive:    {final_pct:.2f}%", flush=True)

    # Save pareto_front.json
    pareto_out = {
        'algorithm': 'random',
        'model': 'meta-llama/llama-3.1-8b-instruct',
        'total_evals': total_evals,
        'pareto_size': len(current_nd),
        'exhaustive_hv': exhaustive_hv,
        'final_hv': final_hv,
        'final_pct_of_exhaustive': final_pct,
        'pareto_front': [{'objectives': list(p)} for p in current_nd],
    }
    with open(os.path.join(RESULTS_DIR, 'pareto_front.json'), 'w') as f:
        json.dump(pareto_out, f, indent=2)

    with open(os.path.join(RESULTS_DIR, 'convergence.json'), 'w') as f:
        json.dump(checkpoints, f, indent=2)

    print(f"Results saved to {RESULTS_DIR}", flush=True)


if __name__ == '__main__':
    main()
