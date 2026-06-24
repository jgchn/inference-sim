#!/usr/bin/env python3
"""h-robustness: Exhaustive sweep of all 1800 configurations."""

import subprocess, json, os, sys, time
import numpy as np

BLIS = '/Users/jchen/go/src/inference-sim/inference-sim/.nous-experiments/iter-5-9ae74af5/blis'
RESULTS_DIR = '/Users/jchen/go/src/inference-sim/inference-sim/.nous/campaign-3/runs/iter-5/results/h-robustness'
TMPDIR = os.environ.get('TMPDIR', '/tmp')
REF = np.array([0.0, 50000.0, 9.0, 11000.0])
MC_SAMPLES = 200_000
MC_SEED = 42
HV_LO = np.array([-100.0, 0.0, 1.0, 2000.0])

FIXED_FLAGS = [
    '--model', 'qwen/qwen3-14b', '--hardware', 'H100',
    '--latency-model', 'trained-physics', '--num-requests', '200',
    '--rate', '50', '--prefix-tokens', '512', '--seed', '42',
    '--max-num-scheduled-tokens', '4096', '--long-prefill-token-threshold', '0',
    '--admission-policy', 'always-admit', '--preemption-policy', 'fcfs',
    '--routing-policy', 'round-robin', '--gpu-memory-utilization', '0.9',
]

TP_VALUES = [1, 2, 4, 8]
SCHED_VALUES = ['fcfs', 'sjf']
BATCH_VALUES = [32, 64, 128, 256, 512]
KV_VALUES = [2000, 3000, 4000, 5000, 7500, 10000]
BS_VALUES = [16, 32]


def all_configs():
    configs = []
    for tp in TP_VALUES:
        for inst in range(1, 8 // tp + 1):
            for sched in SCHED_VALUES:
                for batch in BATCH_VALUES:
                    for kv in KV_VALUES:
                        for bs in BS_VALUES:
                            configs.append((tp, inst, sched, batch, kv, bs))
    return configs


def evaluate(tp, inst, sched, batch, kv, bs, idx):
    mf = os.path.join(TMPDIR, f'blis_iter5_exh_{idx}.json')
    cmd = [BLIS, 'run'] + FIXED_FLAGS + [
        '--tp', str(tp), '--num-instances', str(inst),
        '--scheduler', sched, '--max-num-running-reqs', str(batch),
        '--total-kv-blocks', str(kv), '--block-size-in-tokens', str(bs),
        '--metrics-path', mf,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=60)
        if r.returncode != 0 or not os.path.exists(mf):
            return None
        with open(mf) as f:
            d = json.load(f)
        os.unlink(mf)
        return (-d['responses_per_sec'], d['ttft_p99_ms'], float(tp * inst), float(kv))
    except Exception as e:
        print(f"  WARN eval({tp},{inst},{sched},{batch},{kv},{bs}): {e}", file=sys.stderr)
        return None


def pareto_update(nd, new_pt):
    """Update non-dominated set with new point. Returns new nd list."""
    for p in nd:
        if all(p[k] <= new_pt[k] for k in range(4)) and any(p[k] < new_pt[k] for k in range(4)):
            return nd  # new_pt is dominated
    new_nd = [p for p in nd if not (
        all(new_pt[k] <= p[k] for k in range(4)) and any(new_pt[k] < p[k] for k in range(4))
    )]
    new_nd.append(new_pt)
    return new_nd


def hypervolume_mc(pareto_pts):
    """Monte Carlo HV estimate (minimization) using fixed sampling bounds."""
    if not pareto_pts:
        return 0.0
    pts = np.array(pareto_pts, dtype=np.float64)  # (n, 4)
    rng = np.random.default_rng(MC_SEED)
    box_vol = float(np.prod(REF - HV_LO))
    dominated = 0
    batch = 50_000
    for _ in range(MC_SAMPLES // batch):
        s = rng.uniform(HV_LO, REF, size=(batch, 4))
        # s is dominated if any Pareto point p has p <= s in all dims
        dom = np.any(np.all(pts[np.newaxis, :, :] <= s[:, np.newaxis, :], axis=2), axis=1)
        dominated += int(np.sum(dom))
    return box_vol * dominated / MC_SAMPLES


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    configs = all_configs()
    print(f"Total configs: {len(configs)}", file=sys.stderr)

    all_results = []
    current_nd = []
    convergence = []
    t0 = time.time()

    for idx, cfg in enumerate(configs):
        tp, inst, sched, batch, kv, bs = cfg
        obj = evaluate(tp, inst, sched, batch, kv, bs, idx)
        if obj is not None:
            all_results.append({'config': list(cfg), 'objectives': list(obj)})
            current_nd = pareto_update(current_nd, obj)

        if (idx + 1) % 40 == 0 or idx + 1 == len(configs):
            hv = hypervolume_mc(current_nd)
            elapsed = time.time() - t0
            convergence.append({
                'eval': idx + 1,
                'hv': hv,
                'pareto_size': len(current_nd),
                'elapsed_s': round(elapsed, 1),
            })
            print(f"Eval {idx+1}/{len(configs)}: HV={hv:.4e} Pareto={len(current_nd)} t={elapsed:.0f}s",
                  file=sys.stderr)

    exhaustive_hv = hypervolume_mc(current_nd)
    pareto_density = len(current_nd) / len(all_results) if all_results else 0.0

    obj_to_config = {tuple(r['objectives']): r['config'] for r in all_results}
    pareto_front_out = [
        {'objectives': list(p), 'config': obj_to_config.get(tuple(p))}
        for p in current_nd
    ]

    out = {
        'total_configs': len(configs),
        'evaluated': len(all_results),
        'pareto_size': len(current_nd),
        'pareto_density': round(pareto_density, 4),
        'exhaustive_hv': exhaustive_hv,
        'reference_point': list(REF),
        'hv_lo_bounds': list(HV_LO),
        'pareto_front': pareto_front_out,
    }
    with open(os.path.join(RESULTS_DIR, 'pareto_front.json'), 'w') as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(RESULTS_DIR, 'convergence.json'), 'w') as f:
        json.dump(convergence, f, indent=2)

    print(f"\nExhaustive HV: {exhaustive_hv:.6e}")
    print(f"Pareto density: {pareto_density:.1%} ({len(current_nd)}/{len(all_results)})")
    print(f"Results saved to {RESULTS_DIR}")


if __name__ == '__main__':
    main()
