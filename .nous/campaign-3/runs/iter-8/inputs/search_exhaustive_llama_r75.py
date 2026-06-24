#!/usr/bin/env python3
"""h-robustness (part 2): Exhaustive sweep of 1800 configs for Llama-3.1-8B at rate=75.

Computes true Pareto front, density, and reference HV.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from blis_common import (all_configs, eval_config, pareto_front, mc_hypervolume,
                         FIXED_FLAGS_LLAMA_R75)

RESULTS_DIR = '/Users/jchen/go/src/inference-sim/inference-sim/.nous/campaign-3/runs/iter-8/results/h-robustness'


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    configs = all_configs()
    print(f"Exhaustive sweep: {len(configs)} configs, Llama-3.1-8B at rate=75", flush=True)

    t0 = time.time()
    all_objs = []
    for i, cfg in enumerate(configs):
        objs = eval_config(cfg, FIXED_FLAGS_LLAMA_R75, cache_key_prefix='llama_r75')
        all_objs.append(objs)
        if (i + 1) % 200 == 0:
            print(f"  evaluated {i + 1}/{len(configs)} ({time.time() - t0:.1f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"Exhaustive sweep completed in {elapsed:.1f}s", flush=True)

    # Compute Pareto front
    pf_indices = pareto_front(all_objs)
    pf_objs = [all_objs[i] for i in pf_indices]
    density = len(pf_indices) / len(configs)

    # Compute exhaustive HV
    exhaustive_hv = mc_hypervolume(pf_objs)

    print(f"Pareto front: {len(pf_indices)} configs ({100*density:.2f}% density)", flush=True)
    print(f"Exhaustive HV: {exhaustive_hv:.6e}", flush=True)

    # Save results
    result = {
        'model': 'meta-llama/llama-3.1-8b-instruct',
        'rate': 75,
        'total_configs': len(configs),
        'evaluated': len(configs),
        'pareto_size': len(pf_indices),
        'pareto_density': density,
        'exhaustive_hv': exhaustive_hv,
        'reference_point': [0.0, 50000.0, 9.0, 11000.0],
        'hv_lo_bounds': [-100.0, 0.0, 1.0, 2000.0],
        'elapsed_s': round(elapsed, 1),
        'pareto_front': [
            {
                'config': {k: v for k, v in configs[i].items()
                           if k in ('tp', 'num_instances', 'scheduler',
                                    'max_num_running_reqs', 'total_kv_blocks',
                                    'block_size_in_tokens')},
                'objectives': all_objs[i],
            }
            for i in pf_indices
        ],
    }

    outpath = os.path.join(RESULTS_DIR, 'pareto_front_llama_r75.json')
    with open(outpath, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {outpath}", flush=True)


if __name__ == '__main__':
    main()
