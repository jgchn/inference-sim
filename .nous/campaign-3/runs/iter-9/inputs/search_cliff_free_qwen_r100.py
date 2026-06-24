"""h-control-negative: NSGA-II vs Random on cliff-free Qwen (kv=10000 fixed, 3 objectives)."""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blis_common import (
    MODEL_QWEN, REF_POINT_3OBJ, run_algorithm_comparison
)

ALL_RESULTS_PATH = "/Users/jchen/go/src/inference-sim/inference-sim/.nous/campaign-3/runs/iter-9/results/h-robustness/all_results_qwen_r100.json"
OUTPUT_PATH = "/Users/jchen/go/src/inference-sim/inference-sim/.nous/campaign-3/runs/iter-9/results/h-control-negative/convergence_cliff_free_qwen_r100.json"

def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print("=== h-control-negative: Cliff-free Qwen at rate=100 (kv=10000, 3obj) ===", file=sys.stderr)
    with open(ALL_RESULTS_PATH) as f:
        all_results = json.load(f)

    # Filter to kv=10000 only (cliff-free subspace)
    cliff_free = [r for r in all_results if r['config']['total_kv_blocks'] == 10000]
    print(f"Cliff-free configs (kv=10000): {len(cliff_free)}", file=sys.stderr)

    result = run_algorithm_comparison(
        all_results=cliff_free,
        ref_point=list(REF_POINT_3OBJ),
        pop_size=20,
        n_gens=4,
        budget=100,
        checkpoint_interval=20,
        nsga2_seed=42,
        random_seed=43,
        use_3obj=True,   # Drop kv_blocks objective
        fixed_kv=10000,  # Prevent mutation from changing kv
    )

    output = {
        'arm': 'h-control-negative',
        'model': MODEL_QWEN,
        'rate': 100,
        'kv_fixed': 10000,
        'n_objectives': 3,
        **result,
    }
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved convergence results to {OUTPUT_PATH}", file=sys.stderr)
    print(json.dumps({
        'arm': 'h-control-negative',
        'model': MODEL_QWEN,
        'nsga2_95pct_eval': result['nsga2_95pct_eval'],
        'random_95pct_eval': result['random_95pct_eval'],
        'gap': result['gap'],
        'exhaustive_density': result['exhaustive_density'],
    }, indent=2))

if __name__ == '__main__':
    main()
