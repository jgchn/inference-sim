"""Exhaustive sweep for Llama at rate=50 (h-robustness arm)."""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))
from blis_common import (
    MODEL_LLAMA, REF_POINT_4OBJ,
    run_exhaustive, get_nondominated, hypervolume,
    compute_ndr, compute_crossover_yield, compute_pareto_entropy,
    compute_pareto_closure_ratio, config_key
)

RESULTS_DIR = "/Users/jchen/go/src/inference-sim/inference-sim/.nous/campaign-3/runs/iter-10/results/h-robustness"

def main():
    print("=== h-robustness: Llama rate=50 exhaustive sweep ===", file=sys.stderr)

    all_results = run_exhaustive(MODEL_LLAMA, max_workers=8)

    # Save all results
    out_all = os.path.join(RESULTS_DIR, "all_results_llama_r50.json")
    with open(out_all, 'w') as f:
        json.dump(all_results, f)
    print(f"Saved {len(all_results)} configs to {out_all}", file=sys.stderr)

    # Compute Pareto front and metrics
    valid = [r for r in all_results if r['objectives'] is not None]
    all_objs = [r['objectives'] for r in valid]
    pareto_objs = get_nondominated(all_objs)
    pareto_density = len(pareto_objs) / len(valid) if valid else 0.0
    exhaustive_hv = hypervolume(pareto_objs, REF_POINT_4OBJ)

    # NDR
    overall_ndr, per_axis_ndr, per_axis_counts = compute_ndr(all_results)

    # Crossover yield
    yield_metrics = compute_crossover_yield(all_results, n_samples=10000, seed=42)

    # Pareto entropy and closure ratio
    pareto_entropy = compute_pareto_entropy(all_results)
    closure_ratio = compute_pareto_closure_ratio(all_results)

    # Identify Pareto configs
    pareto_set_keys = set()
    lookup = {config_key(r['config']): r['objectives'] for r in valid}
    for r in valid:
        obj = r['objectives']
        if any(all(o[i] == obj[i] for i in range(len(obj))) for o in pareto_objs):
            pareto_set_keys.add(config_key(r['config']))

    pareto_configs_data = [
        r for r in valid if config_key(r['config']) in pareto_set_keys
    ]

    # Block_size distribution in Pareto
    bs_counts = {}
    for r in pareto_configs_data:
        bs = r['config']['block_size_in_tokens']
        bs_counts[bs] = bs_counts.get(bs, 0) + 1

    pareto_front_data = {
        'model': MODEL_LLAMA,
        'rate': 50,
        'total_configs': len(valid),
        'pareto_size': len(pareto_set_keys),
        'pareto_density': pareto_density,
        'exhaustive_hv': exhaustive_hv,
        'ref_point': list(REF_POINT_4OBJ),
        'overall_ndr': overall_ndr,
        'per_axis_ndr': per_axis_ndr,
        'crossover_yield': yield_metrics['crossover_yield'],
        'yield_advantage': yield_metrics['yield_advantage'],
        'yield_valid_offspring': yield_metrics['valid_offspring'],
        'yield_pareto_offspring': yield_metrics['pareto_offspring'],
        'pareto_entropy': pareto_entropy,
        'closure_ratio': closure_ratio,
        'block_size_distribution': bs_counts,
        'pareto_objectives': [r['objectives'] for r in pareto_configs_data],
    }

    out_pareto = os.path.join(RESULTS_DIR, "pareto_front_llama_r50.json")
    with open(out_pareto, 'w') as f:
        json.dump(pareto_front_data, f, indent=2)
    print(f"Saved Pareto front data to {out_pareto}", file=sys.stderr)

    print(f"\n=== Llama rate=50 Summary ===")
    print(f"Total configs: {len(valid)}")
    print(f"Pareto size: {len(pareto_set_keys)} ({pareto_density*100:.2f}%)")
    print(f"Exhaustive HV: {exhaustive_hv:.4f}")
    print(f"Overall NDR: {overall_ndr:.4f}")
    print(f"Per-axis NDR: {per_axis_ndr}")
    print(f"Crossover yield: {yield_metrics['crossover_yield']*100:.2f}%")
    print(f"Yield advantage: {yield_metrics['yield_advantage']:.2f}x")
    print(f"Pareto entropy: {pareto_entropy:.4f}")
    print(f"Closure ratio: {closure_ratio:.4f}")
    print(f"Block size distribution: {bs_counts}")


if __name__ == '__main__':
    main()
