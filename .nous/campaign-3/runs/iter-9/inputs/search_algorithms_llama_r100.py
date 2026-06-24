"""h-ablation: NSGA-II vs Random on Llama at rate=100 (reads from h-robustness cache)."""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blis_common import (
    MODEL_LLAMA, REF_POINT_4OBJ, run_algorithm_comparison
)

ALL_RESULTS_PATH = "/Users/jchen/go/src/inference-sim/inference-sim/.nous/campaign-3/runs/iter-9/results/h-robustness/all_results_llama_r100.json"
OUTPUT_PATH = "/Users/jchen/go/src/inference-sim/inference-sim/.nous/campaign-3/runs/iter-9/results/h-ablation/convergence_llama_r100.json"

def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print("=== h-ablation: Algorithm comparison Llama at rate=100 ===", file=sys.stderr)
    with open(ALL_RESULTS_PATH) as f:
        all_results = json.load(f)
    print(f"Loaded {len(all_results)} configs from cache", file=sys.stderr)

    result = run_algorithm_comparison(
        all_results=all_results,
        ref_point=list(REF_POINT_4OBJ),
        pop_size=40,
        n_gens=4,
        budget=200,
        checkpoint_interval=40,
        nsga2_seed=42,
        random_seed=43,
        use_3obj=False,
        fixed_kv=None,
    )

    output = {
        'arm': 'h-ablation',
        'model': MODEL_LLAMA,
        'rate': 100,
        **result,
    }
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved convergence results to {OUTPUT_PATH}", file=sys.stderr)
    print(json.dumps({
        'arm': 'h-ablation',
        'model': MODEL_LLAMA,
        'nsga2_95pct_eval': result['nsga2_95pct_eval'],
        'random_95pct_eval': result['random_95pct_eval'],
        'gap': result['gap'],
        'exhaustive_density': result['exhaustive_density'],
    }, indent=2))

if __name__ == '__main__':
    main()
