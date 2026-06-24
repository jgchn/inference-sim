"""h-ablation: Exhaustive + NSGA-II + Random at rate=75.

Runs in sequence: exhaustive → NSGA-II → random.
Output:
  results/h-ablation/pareto_front.json   (exhaustive Pareto)
  results/h-ablation/convergence_nsga2.json
  results/h-ablation/convergence_random.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blis_common as bc

RESULTS_DIR = "/Users/jchen/go/src/inference-sim/inference-sim/.nous/campaign-3/runs/iter-6/results/h-ablation"
RATE = 75

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Phase 1: Exhaustive at rate=75 ─────────────────────────────────────────
print(f"=== h-ablation phase 1: Exhaustive sweep at rate={RATE} ===")
result = bc.compute_exhaustive_hv(RATE)
exhaustive_hv_75 = result['hypervolume']

ex_output = {
    'arm': 'h-ablation',
    'rate': RATE,
    'phase': 'exhaustive',
    'total_configs': result['count'],
    'pareto_count': result['pareto_count'],
    'density_pct': result['density_pct'],
    'hypervolume': result['hypervolume'],
    'convergence': result['convergence'],
    'pareto_configs': result['pareto_configs'],
}
ex_path = os.path.join(RESULTS_DIR, 'pareto_front.json')
with open(ex_path, 'w') as f:
    json.dump(ex_output, f, indent=2)
print(f"Exhaustive saved to {ex_path}")
print(f"  HV={exhaustive_hv_75:.6e}, density={result['density_pct']:.2f}%")

# ── Phase 2: NSGA-II at rate=75 ─────────────────────────────────────────────
# Reset eval cache between phases so we get fresh counts (but cache hits OK)
print(f"\n=== h-ablation phase 2: NSGA-II at rate={RATE} ===")
all_results_nsga2, ckpts_nsga2 = bc.run_nsga2(
    rate=RATE,
    pop_size=40,
    n_gens=4,
    mutation_rate=0.15,
    rng_seed=42,
    checkpoint_every=40,
)
bc.add_hv_pct(ckpts_nsga2, exhaustive_hv_75)

all_objs_nsga2 = [r[1] for r in all_results_nsga2]
pf_idx_nsga2 = bc.pareto_front(all_objs_nsga2)

nsga2_out = {
    'arm': 'h-ablation',
    'rate': RATE,
    'algorithm': 'nsga2',
    'pop_size': 40,
    'n_gens': 4,
    'total_evals': len(all_results_nsga2),
    'exhaustive_hv': exhaustive_hv_75,
    'final_hv': ckpts_nsga2[-1]['hv'] if ckpts_nsga2 else 0,
    'final_hv_pct': ckpts_nsga2[-1]['hv_pct'] if ckpts_nsga2 else 0,
    'final_pareto_size': len(pf_idx_nsga2),
    'convergence': ckpts_nsga2,
}
nsga2_path = os.path.join(RESULTS_DIR, 'convergence_nsga2.json')
with open(nsga2_path, 'w') as f:
    json.dump(nsga2_out, f, indent=2)
print(f"NSGA-II saved to {nsga2_path}")
print(f"  Final HV={nsga2_out['final_hv']:.6e} ({nsga2_out['final_hv_pct']:.2f}%)")

# ── Phase 3: Random at rate=75 ───────────────────────────────────────────────
print(f"\n=== h-ablation phase 3: Random search at rate={RATE} ===")
all_results_rand, ckpts_rand = bc.run_random(
    rate=RATE,
    budget=200,
    rng_seed=43,
    checkpoint_every=40,
)
bc.add_hv_pct(ckpts_rand, exhaustive_hv_75)

all_objs_rand = [r[1] for r in all_results_rand]
pf_idx_rand = bc.pareto_front(all_objs_rand)

rand_out = {
    'arm': 'h-ablation',
    'rate': RATE,
    'algorithm': 'random',
    'total_evals': len(all_results_rand),
    'exhaustive_hv': exhaustive_hv_75,
    'final_hv': ckpts_rand[-1]['hv'] if ckpts_rand else 0,
    'final_hv_pct': ckpts_rand[-1]['hv_pct'] if ckpts_rand else 0,
    'final_pareto_size': len(pf_idx_rand),
    'convergence': ckpts_rand,
}
rand_path = os.path.join(RESULTS_DIR, 'convergence_random.json')
with open(rand_path, 'w') as f:
    json.dump(rand_out, f, indent=2)
print(f"Random saved to {rand_path}")
print(f"  Final HV={rand_out['final_hv']:.6e} ({rand_out['final_hv_pct']:.2f}%)")

# ── Summary ─────────────────────────────────────────────────────────────────
print(f"\n=== h-ablation summary at rate={RATE} ===")
print(f"Exhaustive: HV={exhaustive_hv_75:.6e}, density={result['density_pct']:.2f}%")
print(f"NSGA-II final: {nsga2_out['final_hv_pct']:.2f}%")
print(f"Random final:  {rand_out['final_hv_pct']:.2f}%")
print(f"Gap (NSGA-II - Random): {nsga2_out['final_hv_pct'] - rand_out['final_hv_pct']:.2f}%")
