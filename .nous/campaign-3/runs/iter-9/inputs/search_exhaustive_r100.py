"""h-robustness: Exhaustive sweep for Qwen at rate=100. Saves ALL per-config results."""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blis_common import (
    MODEL_QWEN, REF_POINT_4OBJ, run_exhaustive,
    get_nondominated, hypervolume, compute_ndr, config_key
)

RESULTS_DIR = "/Users/jchen/go/src/inference-sim/inference-sim/.nous/campaign-3/runs/iter-9/results/h-robustness"

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=== h-robustness: Exhaustive sweep Qwen at rate=100 ===", file=sys.stderr)
    all_results = run_exhaustive(MODEL_QWEN, max_workers=8, verbose=True)

    # Save all results
    all_results_path = os.path.join(RESULTS_DIR, "all_results_qwen_r100.json")
    with open(all_results_path, 'w') as f:
        json.dump(all_results, f)
    print(f"Saved {len(all_results)} results to {all_results_path}", file=sys.stderr)

    # Compute Pareto front and metrics
    valid = [r for r in all_results if r['objectives'] is not None]
    all_obj = [r['objectives'] for r in valid]
    pareto_nd = get_nondominated(all_obj)
    density = len(pareto_nd) / len(all_obj)
    exh_hv = hypervolume(pareto_nd, list(REF_POINT_4OBJ))

    print(f"\nQwen results:", file=sys.stderr)
    print(f"  Total configs: {len(all_obj)}", file=sys.stderr)
    print(f"  Pareto size: {len(pareto_nd)}", file=sys.stderr)
    print(f"  Pareto density: {density*100:.2f}%", file=sys.stderr)
    print(f"  Exhaustive HV: {exh_hv:.4f}", file=sys.stderr)

    # Compute NDR
    print("Computing NDR...", file=sys.stderr)
    overall_ndr, per_axis_ndr, per_axis_counts = compute_ndr(all_results)
    print(f"  Overall NDR: {overall_ndr:.4f}", file=sys.stderr)
    for ax, ndr in sorted(per_axis_ndr.items(), key=lambda x: -x[1]):
        dom, tot = per_axis_counts[ax]
        print(f"    {ax}: {ndr:.4f} ({dom}/{tot})", file=sys.stderr)

    # Save summary
    pareto_path = os.path.join(RESULTS_DIR, "pareto_front_qwen_r100.json")
    summary = {
        'model': MODEL_QWEN,
        'rate': 100,
        'total_configs': len(all_obj),
        'pareto_size': len(pareto_nd),
        'pareto_density': density,
        'exhaustive_hv': exh_hv,
        'ref_point': list(REF_POINT_4OBJ),
        'pareto_objectives': pareto_nd,
        'ndr_overall': overall_ndr,
        'ndr_per_axis': per_axis_ndr,
        'ndr_per_axis_counts': {k: list(v) for k, v in per_axis_counts.items()},
    }
    with open(pareto_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved Pareto/NDR summary to {pareto_path}", file=sys.stderr)

    # Print final summary to stdout
    print(json.dumps({
        'model': MODEL_QWEN,
        'pareto_density': density,
        'exhaustive_hv': exh_hv,
        'ndr_overall': overall_ndr,
        'ndr_per_axis': per_axis_ndr,
    }, indent=2))

if __name__ == '__main__':
    main()
