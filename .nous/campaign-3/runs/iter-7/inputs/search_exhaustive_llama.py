#!/usr/bin/env python3
"""h-robustness: Exhaustive sweep of all 1800 configs with Llama-3.1-8B at rate=50.

Computes the true Pareto front, Pareto density, and exhaustive HV.
This provides the normalization baseline for h-main and h-control-negative.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from blis_common import (
    all_configs, eval_config, pareto_front, mc_hypervolume,
    FIXED_FLAGS_LLAMA, REF_POINT, HV_LO, MC_SAMPLES, MC_SEED,
)

RESULTS_DIR = '/Users/jchen/go/src/inference-sim/inference-sim/.nous/campaign-3/runs/iter-7/results/h-robustness'

CHECKPOINT_EVERY = 40


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    configs = all_configs()
    print(f"Exhaustive sweep: {len(configs)} configs, Llama-3.1-8B, rate=50", flush=True)

    all_objs = []
    convergence = []
    t0 = time.time()

    for i, cfg in enumerate(configs):
        objs = eval_config(cfg, FIXED_FLAGS_LLAMA, cache_key_prefix='llama')
        all_objs.append(objs)

        if (i + 1) % CHECKPOINT_EVERY == 0:
            pf_idx = pareto_front(all_objs)
            pf_objs = [all_objs[j] for j in pf_idx]
            hv = mc_hypervolume(pf_objs)
            convergence.append({
                'eval': i + 1,
                'hv': hv,
                'pareto_size': len(pf_idx),
                'elapsed_s': round(time.time() - t0, 1),
            })
            print(f"  {i + 1}/{len(configs)}: hv={hv:.4e}, pareto={len(pf_idx)}, "
                  f"{time.time() - t0:.1f}s", flush=True)
        elif (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{len(configs)} evals, {elapsed:.1f}s elapsed", flush=True)

    # Final Pareto front
    pf_idx = pareto_front(all_objs)
    pf_objs = [all_objs[i] for i in pf_idx]
    pf_configs = [configs[i] for i in pf_idx]
    exhaustive_hv = mc_hypervolume(pf_objs)
    density_pct = 100.0 * len(pf_idx) / len(configs)
    elapsed = time.time() - t0

    print(f"\nExhaustive complete:", flush=True)
    print(f"  Pareto configs: {len(pf_idx)}/{len(configs)} = {density_pct:.2f}%", flush=True)
    print(f"  Exhaustive HV:  {exhaustive_hv:.6e}", flush=True)
    print(f"  Elapsed:        {elapsed:.1f}s", flush=True)

    # Save pareto_front.json
    pareto_out = {
        'total_configs': len(configs),
        'evaluated': len(configs),
        'pareto_size': len(pf_idx),
        'pareto_density': round(density_pct / 100.0, 4),
        'exhaustive_hv': exhaustive_hv,
        'reference_point': list(REF_POINT),
        'hv_lo_bounds': list(HV_LO),
        'pareto_front': [
            {
                'tp': cfg['tp'], 'num_instances': cfg['num_instances'],
                'scheduler': cfg['scheduler'],
                'max_num_running_reqs': cfg['max_num_running_reqs'],
                'total_kv_blocks': cfg['total_kv_blocks'],
                'block_size_in_tokens': cfg['block_size_in_tokens'],
                'gpu_count': cfg['tp'] * cfg['num_instances'],
                'objectives': objs,
            }
            for cfg, objs in zip(pf_configs, pf_objs)
        ],
    }
    with open(os.path.join(RESULTS_DIR, 'pareto_front.json'), 'w') as f:
        json.dump(pareto_out, f, indent=2)

    # Save convergence.json
    with open(os.path.join(RESULTS_DIR, 'convergence.json'), 'w') as f:
        json.dump(convergence, f, indent=2)

    print(f"Results saved to {RESULTS_DIR}", flush=True)
    return exhaustive_hv


if __name__ == '__main__':
    main()
