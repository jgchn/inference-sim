"""h-control-negative: Random search at rate=100, budget=200.

Reads exhaustive HV from h-robustness results for normalization.
Output: results/h-control-negative/convergence.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blis_common as bc

RESULTS_DIR = "/Users/jchen/go/src/inference-sim/inference-sim/.nous/campaign-3/runs/iter-6/results/h-control-negative"
ROBUSTNESS_RESULTS = "/Users/jchen/go/src/inference-sim/inference-sim/.nous/campaign-3/runs/iter-6/results/h-robustness/pareto_front.json"
RATE = 100

os.makedirs(RESULTS_DIR, exist_ok=True)

# Load exhaustive HV from h-robustness
with open(ROBUSTNESS_RESULTS) as f:
    rob = json.load(f)
exhaustive_hv = rob['hypervolume']
print(f"Loaded exhaustive HV (rate={RATE}): {exhaustive_hv:.6e}")

print(f"\n=== h-control-negative: Random search at rate={RATE}, budget=200 ===")
all_results, checkpoints = bc.run_random(
    rate=RATE,
    budget=200,
    rng_seed=43,
    checkpoint_every=40,
)

bc.add_hv_pct(checkpoints, exhaustive_hv)

all_objs = [r[1] for r in all_results]
pf_idx = bc.pareto_front(all_objs)

output = {
    'arm': 'h-control-negative',
    'rate': RATE,
    'algorithm': 'random',
    'total_evals': len(all_results),
    'exhaustive_hv': exhaustive_hv,
    'final_hv': checkpoints[-1]['hv'] if checkpoints else 0,
    'final_hv_pct': checkpoints[-1]['hv_pct'] if checkpoints else 0,
    'final_pareto_size': len(pf_idx),
    'convergence': checkpoints,
}

out_path = os.path.join(RESULTS_DIR, 'convergence.json')
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nSaved to {out_path}")
print(f"Total evals: {len(all_results)}")
print(f"Final HV: {output['final_hv']:.6e} ({output['final_hv_pct']:.2f}% of exhaustive)")
conv_str = [(cp['eval'], f"{cp['hv_pct']:.1f}%") for cp in checkpoints]
print(f"Convergence: {conv_str}")
