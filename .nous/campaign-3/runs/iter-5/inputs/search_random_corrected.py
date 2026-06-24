#!/usr/bin/env python3
"""h-control-negative: Random search on 1800-config corrected space. 200 evals."""

import subprocess, json, os, sys, time
import numpy as np

BLIS = '/Users/jchen/go/src/inference-sim/inference-sim/.nous-experiments/iter-5-9ae74af5/blis'
RESULTS_DIR = '/Users/jchen/go/src/inference-sim/inference-sim/.nous/campaign-3/runs/iter-5/results/h-control-negative'
EXHAUSTIVE_RESULTS = '/Users/jchen/go/src/inference-sim/inference-sim/.nous/campaign-3/runs/iter-5/results/h-robustness/pareto_front.json'
TMPDIR = os.environ.get('TMPDIR', '/tmp')
REF = np.array([0.0, 50000.0, 9.0, 11000.0])
MC_SAMPLES = 200_000
MC_SEED = 42
HV_LO = np.array([-100.0, 0.0, 1.0, 2000.0])
BUDGET = 200

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


def random_config(rng):
    tp_idx = int(rng.integers(0, 4))
    tp = TP_VALUES[tp_idx]
    inst = int(rng.integers(1, 8 // tp + 1))
    sched = SCHED_VALUES[int(rng.integers(0, 2))]
    batch = BATCH_VALUES[int(rng.integers(0, 5))]
    kv = KV_VALUES[int(rng.integers(0, 6))]
    bs = BS_VALUES[int(rng.integers(0, 2))]
    return tp, inst, sched, batch, kv, bs


def evaluate(tp, inst, sched, batch, kv, bs, idx):
    mf = os.path.join(TMPDIR, f'blis_iter5_rnd_{idx}.json')
    cmd = [BLIS, 'run'] + FIXED_FLAGS + [
        '--tp', str(tp), '--num-instances', str(inst),
        '--scheduler', sched, '--max-num-running-reqs', str(batch),
        '--total-kv-blocks', str(kv), '--block-size-in-tokens', str(bs),
        '--metrics-path', mf,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=60)
        if r.returncode != 0 or not os.path.exists(mf):
            return (0.0, 50000.0, float(tp * inst), float(kv))
        with open(mf) as f:
            d = json.load(f)
        os.unlink(mf)
        return (-d['responses_per_sec'], d['ttft_p99_ms'], float(tp * inst), float(kv))
    except Exception as e:
        print(f"  WARN eval({tp},{inst},...): {e}", file=sys.stderr)
        return (0.0, 50000.0, float(tp * inst), float(kv))


def pareto_update(nd, new_pt):
    for p in nd:
        if all(p[k] <= new_pt[k] for k in range(4)) and any(p[k] < new_pt[k] for k in range(4)):
            return nd
    return [p for p in nd if not (
        all(new_pt[k] <= p[k] for k in range(4)) and any(new_pt[k] < p[k] for k in range(4))
    )] + [new_pt]


def hypervolume_mc(pareto_pts):
    if not pareto_pts:
        return 0.0
    pts = np.array(pareto_pts, dtype=np.float64)
    rng_mc = np.random.default_rng(MC_SEED)
    box_vol = float(np.prod(REF - HV_LO))
    dominated = 0
    batch = 50_000
    for _ in range(MC_SAMPLES // batch):
        s = rng_mc.uniform(HV_LO, REF, size=(batch, 4))
        dom = np.any(np.all(pts[np.newaxis, :, :] <= s[:, np.newaxis, :], axis=2), axis=1)
        dominated += int(np.sum(dom))
    return box_vol * dominated / MC_SAMPLES


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = np.random.default_rng(42)

    exhaustive_hv = None
    if os.path.exists(EXHAUSTIVE_RESULTS):
        with open(EXHAUSTIVE_RESULTS) as f:
            exhaustive_hv = json.load(f).get('exhaustive_hv')
        print(f"Loaded exhaustive HV: {exhaustive_hv:.6e}", file=sys.stderr)

    current_nd = []
    convergence = []
    t0 = time.time()

    for idx in range(BUDGET):
        tp, inst, sched, batch, kv, bs = random_config(rng)
        obj = evaluate(tp, inst, sched, batch, kv, bs, idx)
        current_nd = pareto_update(current_nd, obj)

        if (idx + 1) % 40 == 0 or idx + 1 == BUDGET:
            hv = hypervolume_mc(current_nd)
            pct = (hv / exhaustive_hv * 100) if exhaustive_hv else None
            entry = {
                'eval': idx + 1,
                'hv': hv,
                'pareto_size': len(current_nd),
                'pct_of_exhaustive': round(pct, 2) if pct is not None else None,
                'elapsed_s': round(time.time() - t0, 1),
            }
            convergence.append(entry)
            print(f"Eval {idx+1}: HV={hv:.4e} Pareto={len(current_nd)}"
                  + (f" ({pct:.1f}%)" if pct else ""), file=sys.stderr)

    pareto_out = {
        'total_evals': BUDGET,
        'pareto_size': len(current_nd),
        'exhaustive_hv': exhaustive_hv,
        'final_hv': convergence[-1]['hv'],
        'final_pct_of_exhaustive': convergence[-1]['pct_of_exhaustive'],
        'reference_point': list(REF),
        'pareto_front': [{'objectives': list(p)} for p in current_nd],
    }
    with open(os.path.join(RESULTS_DIR, 'pareto_front.json'), 'w') as f:
        json.dump(pareto_out, f, indent=2)
    with open(os.path.join(RESULTS_DIR, 'convergence.json'), 'w') as f:
        json.dump(convergence, f, indent=2)

    print(f"\nTotal evals: {BUDGET}")
    print(f"Final HV: {convergence[-1]['hv']:.6e}")
    if exhaustive_hv:
        print(f"% of exhaustive: {convergence[-1]['pct_of_exhaustive']:.1f}%")
    print(f"Results saved to {RESULTS_DIR}")


if __name__ == '__main__':
    main()
