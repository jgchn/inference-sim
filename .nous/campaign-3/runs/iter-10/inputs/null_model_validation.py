"""h-control-negative: Null model crossover yield validation.

Randomly assigns Pareto labels (preserving density) to test that
high observed yield is structurally meaningful, not a statistical artifact.

Uses cached all_results from h-robustness and iter-9.

Outputs:
  - results/h-control-negative/null_model_results.json
"""

import sys
import os
import json
import random

sys.path.insert(0, os.path.dirname(__file__))
from blis_common import (
    get_nondominated, compute_crossover_yield,
    crossover_configs_uniform, normalize_config, config_key, PARAM_SPACE
)

BASE = "/Users/jchen/go/src/inference-sim/inference-sim/.nous/campaign-3/runs"
ITER9_BASE = os.path.join(BASE, "iter-9/results/h-robustness")
ITER10_BASE = os.path.join(BASE, "iter-10/results/h-robustness")
OUT_DIR = os.path.join(BASE, "iter-10/results/h-control-negative")


def compute_null_yield(all_results, n_trials=100, n_samples=5000, seed=42):
    """Randomly assign Pareto labels (same N as actual), compute yield distribution."""
    # Build lookup
    lookup = {}
    for r in all_results:
        if r['objectives'] is not None:
            k = config_key(r['config'])
            lookup[k] = r['config']

    all_keys = list(lookup.keys())
    all_objs_list = []
    for r in all_results:
        if r['objectives'] is not None:
            all_objs_list.append(r['objectives'])

    # Actual Pareto size
    pareto_objs = get_nondominated(all_objs_list)
    actual_pareto_size = len(pareto_objs)
    pareto_density = actual_pareto_size / len(lookup) if lookup else 0.0

    rng = random.Random(seed)
    null_yields = []

    for trial in range(n_trials):
        # Randomly select N configs as "Pareto"
        null_pareto_keys = set(rng.sample(all_keys, actual_pareto_size))
        null_pareto_configs = [lookup[k] for k in null_pareto_keys]

        # Compute crossover yield with null labels
        pareto_offspring = 0
        valid_offspring = 0

        for _ in range(n_samples):
            p1 = rng.choice(null_pareto_configs)
            p2 = rng.choice(null_pareto_configs)
            child = crossover_configs_uniform(p1, p2, rng)
            k = config_key(child)
            if k in lookup:
                valid_offspring += 1
                if k in null_pareto_keys:
                    pareto_offspring += 1

        null_yield = pareto_offspring / valid_offspring if valid_offspring > 0 else 0.0
        null_yields.append(null_yield)

    mean_null = sum(null_yields) / len(null_yields)
    variance = sum((y - mean_null) ** 2 for y in null_yields) / len(null_yields)
    std_null = variance ** 0.5

    return {
        'actual_pareto_size': actual_pareto_size,
        'total_configs': len(lookup),
        'pareto_density': pareto_density,
        'n_trials': n_trials,
        'n_samples_per_trial': n_samples,
        'null_yield_mean': mean_null,
        'null_yield_std': std_null,
        'null_yield_min': min(null_yields),
        'null_yield_max': max(null_yields),
        'null_yields': null_yields,
    }


def main():
    conditions = [
        ('qwen_r50', os.path.join(ITER10_BASE, 'all_results_qwen_r50.json'), 'qwen/qwen3-14b', 50),
        ('llama_r50', os.path.join(ITER10_BASE, 'all_results_llama_r50.json'), 'meta-llama/llama-3.1-8b-instruct', 50),
        ('qwen_r100', os.path.join(ITER9_BASE, 'all_results_qwen_r100.json'), 'qwen/qwen3-14b', 100),
        ('llama_r100', os.path.join(ITER9_BASE, 'all_results_llama_r100.json'), 'meta-llama/llama-3.1-8b-instruct', 100),
    ]

    # Also load observed yield from pareto front files if available
    pareto_front_paths = {
        'qwen_r50': os.path.join(ITER10_BASE, 'pareto_front_qwen_r50.json'),
        'llama_r50': os.path.join(ITER10_BASE, 'pareto_front_llama_r50.json'),
    }

    results = {}
    for cond_id, path, model_name, rate in conditions:
        print(f"\n=== Null model: {model_name} rate={rate} ===", file=sys.stderr)
        with open(path) as f:
            all_results = json.load(f)

        # Observed yield
        obs_yield = compute_crossover_yield(all_results, n_samples=5000, seed=42)

        # Null yield
        null = compute_null_yield(all_results, n_trials=100, n_samples=5000, seed=42)

        obs_vs_null_ratio = (obs_yield['crossover_yield'] / null['null_yield_mean']
                             if null['null_yield_mean'] > 0 else float('inf'))

        print(f"  Observed yield: {obs_yield['crossover_yield']*100:.2f}%", file=sys.stderr)
        print(f"  Null yield: {null['null_yield_mean']*100:.2f}% ± {null['null_yield_std']*100:.2f}%", file=sys.stderr)
        print(f"  Observed/null ratio: {obs_vs_null_ratio:.2f}x", file=sys.stderr)
        print(f"  Density: {null['pareto_density']*100:.2f}%", file=sys.stderr)

        # Prediction check: null yield within 2x of density?
        null_within_2x_density = null['null_yield_mean'] <= 2 * null['pareto_density']

        results[cond_id] = {
            'model': model_name,
            'rate': rate,
            'observed_crossover_yield': obs_yield['crossover_yield'],
            'pareto_density': null['pareto_density'],
            'null_yield_mean': null['null_yield_mean'],
            'null_yield_std': null['null_yield_std'],
            'null_yield_min': null['null_yield_min'],
            'null_yield_max': null['null_yield_max'],
            'observed_vs_null_ratio': obs_vs_null_ratio,
            'null_within_2x_density': null_within_2x_density,
            'actual_pareto_size': null['actual_pareto_size'],
            'total_configs': null['total_configs'],
        }

    # Summary
    all_null_within_2x = all(r['null_within_2x_density'] for r in results.values())

    output = {
        'conditions': results,
        'summary': {
            'all_null_within_2x_density': all_null_within_2x,
            'prediction': 'Null model yield ≈ pareto_density (within 2x), observed yield 5-10x higher',
        }
    }

    out_path = os.path.join(OUT_DIR, 'null_model_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    print("\n=== h-control-negative Summary ===")
    for cond_id, r in results.items():
        print(f"{cond_id}: density={r['pareto_density']*100:.2f}%, "
              f"null_yield={r['null_yield_mean']*100:.2f}%, "
              f"obs_yield={r['observed_crossover_yield']*100:.2f}%, "
              f"ratio={r['observed_vs_null_ratio']:.2f}x, "
              f"null_within_2x={r['null_within_2x_density']}")


if __name__ == '__main__':
    main()
