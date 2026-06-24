"""h-main: Crossover yield comparison across all 4 conditions.

Reads from:
  - results/h-robustness/all_results_qwen_r50.json (iter-10, computed above)
  - results/h-robustness/all_results_llama_r50.json (iter-10, computed above)
  - ../iter-9/results/h-robustness/all_results_qwen_r100.json (cached)
  - ../iter-9/results/h-robustness/all_results_llama_r100.json (cached)

Outputs:
  - results/h-main/crossover_yield_comparison.json
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))
from blis_common import (
    compute_crossover_yield, compute_pareto_entropy, compute_pareto_closure_ratio
)

BASE = "/Users/jchen/go/src/inference-sim/inference-sim/.nous/campaign-3/runs"
ITER9_BASE = os.path.join(BASE, "iter-9/results/h-robustness")
ITER10_BASE = os.path.join(BASE, "iter-10/results/h-robustness")
OUT_DIR = os.path.join(BASE, "iter-10/results/h-main")

# Known gap values from iter-7/iter-9
KNOWN_GAPS = {
    'qwen_r50':  4.6,   # iter-7
    'llama_r50': -22.9, # iter-7
    'qwen_r100':  62.3, # iter-9
    'llama_r100': 40.3, # iter-9
}

def analyze_condition(all_results, model_name, rate, n_samples=10000):
    """Compute all yield metrics for one condition."""
    print(f"\n--- Analyzing {model_name} rate={rate} ---", file=sys.stderr)

    yield_metrics = compute_crossover_yield(all_results, n_samples=n_samples, seed=42)
    pareto_entropy = compute_pareto_entropy(all_results)
    closure_ratio = compute_pareto_closure_ratio(all_results)

    print(f"  Pareto density: {yield_metrics['pareto_density']*100:.2f}%", file=sys.stderr)
    print(f"  Crossover yield: {yield_metrics['crossover_yield']*100:.2f}%", file=sys.stderr)
    print(f"  Yield advantage: {yield_metrics['yield_advantage']:.2f}x", file=sys.stderr)
    print(f"  Pareto entropy: {pareto_entropy:.4f}", file=sys.stderr)
    print(f"  Closure ratio: {closure_ratio:.4f}", file=sys.stderr)

    return {
        'model': model_name,
        'rate': rate,
        'pareto_size': yield_metrics['pareto_size'],
        'total_configs': yield_metrics['total_configs'],
        'pareto_density': yield_metrics['pareto_density'],
        'crossover_yield': yield_metrics['crossover_yield'],
        'yield_advantage': yield_metrics['yield_advantage'],
        'valid_offspring': yield_metrics['valid_offspring'],
        'pareto_offspring': yield_metrics['pareto_offspring'],
        'pareto_entropy': pareto_entropy,
        'closure_ratio': closure_ratio,
        'n_samples': n_samples,
    }


def main():
    conditions = [
        ('qwen_r50', os.path.join(ITER10_BASE, 'all_results_qwen_r50.json'), 'qwen/qwen3-14b', 50),
        ('llama_r50', os.path.join(ITER10_BASE, 'all_results_llama_r50.json'), 'meta-llama/llama-3.1-8b-instruct', 50),
        ('qwen_r100', os.path.join(ITER9_BASE, 'all_results_qwen_r100.json'), 'qwen/qwen3-14b', 100),
        ('llama_r100', os.path.join(ITER9_BASE, 'all_results_llama_r100.json'), 'meta-llama/llama-3.1-8b-instruct', 100),
    ]

    results = {}
    for cond_id, path, model_name, rate in conditions:
        print(f"Loading {path}...", file=sys.stderr)
        with open(path) as f:
            all_results = json.load(f)
        results[cond_id] = analyze_condition(all_results, model_name, rate)

    # Add known gap values
    for cond_id in results:
        results[cond_id]['known_gap'] = KNOWN_GAPS.get(cond_id)
        ya = results[cond_id]['yield_advantage']
        gap = KNOWN_GAPS.get(cond_id)
        if gap is not None:
            results[cond_id]['gap_positive'] = gap > 0
            results[cond_id]['ya_predicts_positive'] = ya > 5
            results[cond_id]['prediction_correct'] = (ya > 5) == (gap > 0)

    # Check predictions
    all_correct = all(
        results[c].get('prediction_correct', False)
        for c in results if results[c].get('known_gap') is not None
    )

    # Rank correlation check
    pairs = [(results[c]['yield_advantage'], results[c]['known_gap'])
             for c in ['qwen_r50', 'llama_r50', 'qwen_r100', 'llama_r100']
             if results[c].get('known_gap') is not None]
    pairs.sort(key=lambda x: x[0])
    yield_ranks = [i for i, _ in enumerate(sorted(pairs, key=lambda x: x[0]))]
    gap_ranks = [sorted([p[1] for p in pairs]).index(p[1]) for p in sorted(pairs, key=lambda x: x[0])]

    output = {
        'conditions': results,
        'summary': {
            'all_sign_predictions_correct': all_correct,
            'sign_prediction_results': {
                c: results[c].get('prediction_correct') for c in results
            },
            'yield_advantage_threshold_used': 5.0,
        }
    }

    out_path = os.path.join(OUT_DIR, 'crossover_yield_comparison.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    print("\n=== Crossover Yield Comparison Summary ===")
    for cond_id in ['qwen_r50', 'llama_r50', 'qwen_r100', 'llama_r100']:
        r = results[cond_id]
        print(f"{cond_id}: density={r['pareto_density']*100:.2f}%, yield={r['crossover_yield']*100:.2f}%, "
              f"ya={r['yield_advantage']:.2f}x, gap={r.get('known_gap')}, "
              f"predict_correct={r.get('prediction_correct')}")


if __name__ == '__main__':
    main()
