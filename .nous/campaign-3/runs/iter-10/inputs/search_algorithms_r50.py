"""h-ablation: NSGA-II vs random comparison at rate=50 for both models.

Uses cached all_results from h-robustness. Zero BLIS subprocess calls.

Outputs:
  - results/h-ablation/convergence_qwen_r50.json
  - results/h-ablation/convergence_llama_r50.json
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))
from blis_common import REF_POINT_4OBJ, run_algorithm_comparison

BASE = "/Users/jchen/go/src/inference-sim/inference-sim/.nous/campaign-3/runs/iter-10"
IN_DIR = os.path.join(BASE, "results/h-robustness")
OUT_DIR = os.path.join(BASE, "results/h-ablation")

POP_SIZE = 40
N_GENS = 4
BUDGET = 200
CHECKPOINT_INTERVAL = 40
NSGA2_SEED = 42
RANDOM_SEED = 43


def run_comparison_for_model(all_results_path, model_label, out_path):
    print(f"\n=== h-ablation: {model_label} rate=50 ===", file=sys.stderr)
    with open(all_results_path) as f:
        all_results = json.load(f)

    result = run_algorithm_comparison(
        all_results=all_results,
        ref_point=REF_POINT_4OBJ,
        pop_size=POP_SIZE,
        n_gens=N_GENS,
        budget=BUDGET,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        nsga2_seed=NSGA2_SEED,
        random_seed=RANDOM_SEED,
    )

    output = {
        'model': model_label,
        'rate': 50,
        'algorithm_params': {
            'pop_size': POP_SIZE,
            'n_gens': N_GENS,
            'budget': BUDGET,
            'checkpoint_interval': CHECKPOINT_INTERVAL,
            'nsga2_seed': NSGA2_SEED,
            'random_seed': RANDOM_SEED,
        },
        **result
    }

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {out_path}", file=sys.stderr)

    print(f"\n{model_label} rate=50:")
    print(f"  Exhaustive HV: {result['exhaustive_hv']:.4f}")
    print(f"  Pareto density: {result['exhaustive_density']*100:.2f}%")
    print(f"  NSGA-II 95% eval: {result['nsga2_95pct_eval']:.1f}")
    print(f"  Random 95% eval: {result['random_95pct_eval']:.1f}")
    print(f"  Gap (random - nsga2): {result['gap']:.1f} ({'NSGA-II wins' if result['gap'] > 0 else 'Random wins or tie'})")

    return result


def main():
    qwen_path = os.path.join(IN_DIR, "all_results_qwen_r50.json")
    llama_path = os.path.join(IN_DIR, "all_results_llama_r50.json")

    qwen_result = run_comparison_for_model(
        qwen_path, "qwen/qwen3-14b",
        os.path.join(OUT_DIR, "convergence_qwen_r50.json")
    )
    llama_result = run_comparison_for_model(
        llama_path, "meta-llama/llama-3.1-8b-instruct",
        os.path.join(OUT_DIR, "convergence_llama_r50.json")
    )

    print("\n=== h-ablation Summary ===")
    print(f"Qwen gap:  {qwen_result['gap']:.1f}  (predicted: >0)")
    print(f"Llama gap: {llama_result['gap']:.1f}  (predicted: <=0)")
    qwen_ok = qwen_result['gap'] > 0
    llama_ok = llama_result['gap'] <= 0
    print(f"Qwen prediction: {'CONFIRMED' if qwen_ok else 'REFUTED'}")
    print(f"Llama prediction: {'CONFIRMED' if llama_ok else 'REFUTED'}")


if __name__ == '__main__':
    main()
