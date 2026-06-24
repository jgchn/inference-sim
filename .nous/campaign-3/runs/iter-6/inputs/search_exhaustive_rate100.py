"""h-robustness: Exhaustive sweep at rate=100.

Enumerates all 1800 configs, computes true Pareto front and exhaustive HV.
Records convergence every 40 evals.
Output: results/h-robustness/pareto_front.json
"""
import json
import os
import sys

# Allow importing blis_common from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blis_common as bc

RESULTS_DIR = "/Users/jchen/go/src/inference-sim/inference-sim/.nous/campaign-3/runs/iter-6/results/h-robustness"
RATE = 100

os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"=== h-robustness: Exhaustive sweep at rate={RATE} ===")
result = bc.compute_exhaustive_hv(RATE)

output = {
    'arm': 'h-robustness',
    'rate': RATE,
    'total_configs': result['count'],
    'pareto_count': result['pareto_count'],
    'density_pct': result['density_pct'],
    'hypervolume': result['hypervolume'],
    'convergence': result['convergence'],
    'pareto_configs': result['pareto_configs'],
}

out_path = os.path.join(RESULTS_DIR, 'pareto_front.json')
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nSaved to {out_path}")
print(f"Exhaustive HV: {result['hypervolume']:.6e}")
print(f"Pareto density: {result['density_pct']:.2f}% ({result['pareto_count']}/{result['count']})")
