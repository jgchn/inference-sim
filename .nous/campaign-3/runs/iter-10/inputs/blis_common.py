"""Shared utilities for BLIS iter-10 experiments (rate=50, crossover yield analysis)."""

import os
import json
import subprocess
import sys
import random
import math
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

BLIS_BIN = "/Users/jchen/go/src/inference-sim/inference-sim/.nous-experiments/iter-10-455a8670/blis"
RATE = 50
MODEL_QWEN = "qwen/qwen3-14b"
MODEL_LLAMA = "meta-llama/llama-3.1-8b-instruct"
HARDWARE = "H100"

COMMON_FLAGS = [
    "--latency-model", "trained-physics",
    "--hardware", HARDWARE,
    "--num-requests", "200",
    "--rate", str(RATE),
    "--prefix-tokens", "512",
    "--seed", "42",
    "--max-num-scheduled-tokens", "4096",
    "--long-prefill-token-threshold", "0",
    "--admission-policy", "always-admit",
    "--preemption-policy", "fcfs",
    "--routing-policy", "round-robin",
    "--gpu-memory-utilization", "0.9",
]

# Model-specific config folder overrides (for models not in the worktree's model_configs/)
MODEL_CONFIG_FOLDER = {
    MODEL_LLAMA: "/Users/jchen/go/src/inference-sim/inference-sim/model_configs/llama-3.1-8b-instruct",
}

PARAM_SPACE = {
    'tp': [1, 2, 4, 8],
    'scheduler': ['fcfs', 'sjf'],
    'max_num_running_reqs': [32, 64, 128, 256, 512],
    'total_kv_blocks': [2000, 3000, 4000, 5000, 7500, 10000],
    'block_size_in_tokens': [16, 32],
}

REF_POINT_4OBJ = (0.0, 50000.0, 9.0, 11000.0)


def get_max_instances(tp):
    return 8 // tp


def enumerate_configs(fixed_kv=None):
    """Enumerate all valid configs. fixed_kv pins total_kv_blocks."""
    configs = []
    for tp in PARAM_SPACE['tp']:
        for num_instances in range(1, get_max_instances(tp) + 1):
            for scheduler in PARAM_SPACE['scheduler']:
                for max_reqs in PARAM_SPACE['max_num_running_reqs']:
                    kv_list = [fixed_kv] if fixed_kv is not None else PARAM_SPACE['total_kv_blocks']
                    for kv in kv_list:
                        for bs in PARAM_SPACE['block_size_in_tokens']:
                            configs.append({
                                'tp': tp,
                                'num_instances': num_instances,
                                'scheduler': scheduler,
                                'max_num_running_reqs': max_reqs,
                                'total_kv_blocks': kv,
                                'block_size_in_tokens': bs,
                            })
    return configs


def config_key(config):
    return (config['tp'], config['num_instances'], config['scheduler'],
            config['max_num_running_reqs'], config['total_kv_blocks'],
            config['block_size_in_tokens'])


def run_blis(config, model, eval_id=0):
    """Run BLIS and return (config, objectives) or (config, None) on failure."""
    tmpdir = os.environ.get('TMPDIR', '/tmp')
    tid = threading.get_ident() % 100000
    metrics_path = os.path.join(tmpdir, f"blis_iter10_{eval_id}_{os.getpid()}_{tid}.json")

    extra_flags = []
    if model in MODEL_CONFIG_FOLDER:
        extra_flags = ["--model-config-folder", MODEL_CONFIG_FOLDER[model]]

    cmd = [BLIS_BIN, "run"] + COMMON_FLAGS + extra_flags + [
        "--model", model,
        "--tp", str(config['tp']),
        "--num-instances", str(config['num_instances']),
        "--scheduler", config['scheduler'],
        "--max-num-running-reqs", str(config['max_num_running_reqs']),
        "--total-kv-blocks", str(config['total_kv_blocks']),
        "--block-size-in-tokens", str(config['block_size_in_tokens']),
        "--metrics-path", metrics_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[WARN] BLIS failed for {config}: {result.stderr[:100]}", file=sys.stderr)
        return config, None

    try:
        with open(metrics_path) as f:
            metrics = json.load(f)
        try:
            os.remove(metrics_path)
        except Exception:
            pass
        rps = metrics['responses_per_sec']
        ttft_p99 = metrics['ttft_p99_ms']
        gpu_count = float(config['tp'] * config['num_instances'])
        kv_blocks = float(config['total_kv_blocks'])
        return config, [-rps, ttft_p99, gpu_count, kv_blocks]
    except Exception as e:
        print(f"[WARN] Metrics parse error: {e}", file=sys.stderr)
        return config, None


def run_exhaustive(model, fixed_kv=None, max_workers=8, verbose=True):
    """Run exhaustive sweep. Returns list of {config, objectives} dicts."""
    configs = enumerate_configs(fixed_kv=fixed_kv)
    if verbose:
        print(f"Running {len(configs)} configs for {model} (workers={max_workers})...", file=sys.stderr)

    results = [None] * len(configs)

    def eval_one(args):
        i, cfg = args
        return i, run_blis(cfg, model, eval_id=i)

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(eval_one, (i, cfg)): i for i, cfg in enumerate(configs)}
        for future in as_completed(futures):
            i, (cfg, obj) = future.result()
            results[i] = {'config': cfg, 'objectives': obj}
            completed += 1
            if verbose and completed % 200 == 0:
                print(f"  {completed}/{len(configs)} done", file=sys.stderr)

    failed = sum(1 for r in results if r['objectives'] is None)
    if verbose:
        print(f"Done. {len(configs) - failed}/{len(configs)} successful.", file=sys.stderr)
    return results


# ──────────────────────────────────────────────────────────
# Pareto / Hypervolume utilities
# ──────────────────────────────────────────────────────────

def pareto_dominates(a, b):
    """True if a dominates b (all <=, at least one <). Minimization."""
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def get_nondominated(points):
    """Return non-dominated subset (minimization)."""
    nd = []
    for p in points:
        if not any(pareto_dominates(q, p) for q in nd):
            nd = [q for q in nd if not pareto_dominates(p, q)]
            nd.append(p)
    return nd


def hypervolume(points, ref):
    """Hypervolume indicator (minimization), bounded by ref."""
    if not points:
        return 0.0
    d = len(ref)
    valid = [p for p in points if all(p[i] < ref[i] for i in range(d))]
    if not valid:
        return 0.0
    if d == 1:
        return float(ref[0] - min(p[0] for p in valid))

    nd = get_nondominated(valid)
    nd.sort(key=lambda p: p[0])

    hv = 0.0
    prev = ref[0]
    for i in range(len(nd) - 1, -1, -1):
        p = nd[i]
        width = prev - p[0]
        if width > 0:
            slice_pts = [q[1:] for q in nd[: i + 1]]
            hv += width * hypervolume(slice_pts, ref[1:])
        prev = p[0]
    return hv


# ──────────────────────────────────────────────────────────
# NDR computation
# ──────────────────────────────────────────────────────────

def compute_ndr(all_results):
    """Compute Neighbor Dominance Rate (Hamming-1 pairs only)."""
    lookup = {}
    for r in all_results:
        if r['objectives'] is not None:
            lookup[config_key(r['config'])] = r['objectives']

    param_axes = ['tp', 'num_instances', 'scheduler', 'max_num_running_reqs',
                  'total_kv_blocks', 'block_size_in_tokens']

    total_pairs = 0
    dominated_pairs = 0
    per_axis_total = defaultdict(int)
    per_axis_dom = defaultdict(int)
    seen_pairs = set()

    for r in all_results:
        if r['objectives'] is None:
            continue
        cfg = r['config']
        obj_a = r['objectives']
        key_a = config_key(cfg)

        for axis in param_axes:
            if axis == 'num_instances':
                alternatives = list(range(1, get_max_instances(cfg['tp']) + 1))
            elif axis == 'tp':
                alternatives = PARAM_SPACE['tp']
            else:
                alternatives = PARAM_SPACE[axis]

            for alt_val in alternatives:
                if alt_val == cfg[axis]:
                    continue

                neighbor = dict(cfg)
                neighbor[axis] = alt_val
                if axis == 'tp':
                    if neighbor['num_instances'] > 8 // alt_val:
                        continue

                key_b = config_key(neighbor)
                if key_b not in lookup:
                    continue

                pair = tuple(sorted([key_a, key_b]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                obj_b = lookup[key_b]
                dominated = pareto_dominates(obj_a, obj_b) or pareto_dominates(obj_b, obj_a)

                per_axis_total[axis] += 1
                total_pairs += 1
                if dominated:
                    per_axis_dom[axis] += 1
                    dominated_pairs += 1

    overall_ndr = dominated_pairs / total_pairs if total_pairs > 0 else 0.0
    per_axis_ndr = {
        ax: per_axis_dom[ax] / per_axis_total[ax] if per_axis_total[ax] > 0 else 0.0
        for ax in param_axes
    }
    per_axis_counts = {
        ax: (per_axis_dom[ax], per_axis_total[ax]) for ax in param_axes
    }
    return overall_ndr, per_axis_ndr, per_axis_counts


# ──────────────────────────────────────────────────────────
# Crossover yield analysis
# ──────────────────────────────────────────────────────────

def normalize_config(config):
    """Clamp num_instances to valid range for tp."""
    c = dict(config)
    c['num_instances'] = max(1, min(c['num_instances'], 8 // c['tp']))
    return c


def crossover_configs_uniform(p1, p2, rng):
    """Uniform crossover of two configs. Returns ONE offspring."""
    child = {}
    params = ['tp', 'num_instances', 'scheduler', 'max_num_running_reqs',
              'total_kv_blocks', 'block_size_in_tokens']
    for param in params:
        child[param] = p1[param] if rng.random() < 0.5 else p2[param]
    return normalize_config(child)


def compute_crossover_yield(all_results, n_samples=10000, seed=42):
    """Sample crossover offspring from Pareto parent pairs.

    Returns dict with yield, yield_advantage, density, pareto_size, etc.
    """
    # Build lookup: config_key -> (config, objectives)
    lookup = {}
    for r in all_results:
        if r['objectives'] is not None:
            k = config_key(r['config'])
            lookup[k] = (r['config'], r['objectives'])

    all_objs = [v[1] for v in lookup.values()]
    pareto_objs = get_nondominated(all_objs)
    pareto_set_keys = set()
    for v in lookup.values():
        cfg, obj = v
        if any(all(o[i] == obj[i] for i in range(len(obj))) for o in pareto_objs):
            pareto_set_keys.add(config_key(cfg))

    pareto_configs = [lookup[k][0] for k in pareto_set_keys]
    total_configs = len(lookup)
    pareto_size = len(pareto_set_keys)
    pareto_density = pareto_size / total_configs if total_configs > 0 else 0.0

    if pareto_size < 2:
        return {
            'crossover_yield': 0.0,
            'yield_advantage': 0.0,
            'pareto_density': pareto_density,
            'pareto_size': pareto_size,
            'total_configs': total_configs,
            'n_samples': n_samples,
            'valid_offspring': 0,
            'pareto_offspring': 0,
        }

    rng = random.Random(seed)
    pareto_offspring = 0
    valid_offspring = 0

    for _ in range(n_samples):
        p1 = rng.choice(pareto_configs)
        p2 = rng.choice(pareto_configs)
        child = crossover_configs_uniform(p1, p2, rng)
        k = config_key(child)
        if k in lookup:
            valid_offspring += 1
            if k in pareto_set_keys:
                pareto_offspring += 1

    crossover_yield = pareto_offspring / valid_offspring if valid_offspring > 0 else 0.0
    yield_advantage = crossover_yield / pareto_density if pareto_density > 0 else 0.0

    return {
        'crossover_yield': crossover_yield,
        'yield_advantage': yield_advantage,
        'pareto_density': pareto_density,
        'pareto_size': pareto_size,
        'total_configs': total_configs,
        'n_samples': n_samples,
        'valid_offspring': valid_offspring,
        'pareto_offspring': pareto_offspring,
    }


def compute_pareto_entropy(all_results):
    """Compute mean normalized Shannon entropy of parameter values across Pareto configs."""
    lookup = {}
    for r in all_results:
        if r['objectives'] is not None:
            k = config_key(r['config'])
            lookup[k] = (r['config'], r['objectives'])

    all_objs = [v[1] for v in lookup.values()]
    pareto_objs = get_nondominated(all_objs)
    pareto_set_keys = set()
    for v in lookup.values():
        cfg, obj = v
        if any(all(o[i] == obj[i] for i in range(len(obj))) for o in pareto_objs):
            pareto_set_keys.add(config_key(cfg))

    pareto_configs = [lookup[k][0] for k in pareto_set_keys]
    if not pareto_configs:
        return 0.0

    axes = {
        'tp': PARAM_SPACE['tp'],
        'scheduler': PARAM_SPACE['scheduler'],
        'max_num_running_reqs': PARAM_SPACE['max_num_running_reqs'],
        'total_kv_blocks': PARAM_SPACE['total_kv_blocks'],
        'block_size_in_tokens': PARAM_SPACE['block_size_in_tokens'],
    }

    entropies = []
    for axis, values in axes.items():
        counts = defaultdict(int)
        for cfg in pareto_configs:
            counts[cfg[axis]] += 1
        total = sum(counts.values())
        probs = [counts[v] / total for v in values if counts[v] > 0]
        if len(probs) <= 1:
            h = 0.0
        else:
            h = -sum(p * math.log(p) for p in probs)
            max_h = math.log(len(values))
            h = h / max_h if max_h > 0 else 0.0
        entropies.append(h)

    return sum(entropies) / len(entropies) if entropies else 0.0


def compute_pareto_closure_ratio(all_results):
    """Compute |Pareto set| / |Cartesian product of Pareto-appearing values|."""
    lookup = {}
    for r in all_results:
        if r['objectives'] is not None:
            k = config_key(r['config'])
            lookup[k] = (r['config'], r['objectives'])

    all_objs = [v[1] for v in lookup.values()]
    pareto_objs = get_nondominated(all_objs)
    pareto_set_keys = set()
    for v in lookup.values():
        cfg, obj = v
        if any(all(o[i] == obj[i] for i in range(len(obj))) for o in pareto_objs):
            pareto_set_keys.add(config_key(cfg))

    pareto_configs = [lookup[k][0] for k in pareto_set_keys]
    if not pareto_configs:
        return 0.0

    axes = ['tp', 'num_instances', 'scheduler', 'max_num_running_reqs',
            'total_kv_blocks', 'block_size_in_tokens']
    value_sets = {}
    for ax in axes:
        value_sets[ax] = set(cfg[ax] for cfg in pareto_configs)

    cartesian_size = 1
    for ax in axes:
        cartesian_size *= len(value_sets[ax])

    return len(pareto_set_keys) / cartesian_size if cartesian_size > 0 else 0.0


# ──────────────────────────────────────────────────────────
# NSGA-II
# ──────────────────────────────────────────────────────────

def nsga2_fast_nondominated_sort(objectives):
    """Return list of fronts (each front = list of indices)."""
    n = len(objectives)
    domination_count = [0] * n
    dominated_set = [[] for _ in range(n)]
    fronts = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            if pareto_dominates(objectives[i], objectives[j]):
                dominated_set[i].append(j)
                domination_count[j] += 1
            elif pareto_dominates(objectives[j], objectives[i]):
                dominated_set[j].append(i)
                domination_count[i] += 1
        if domination_count[i] == 0:
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        k += 1
        fronts.append(next_front)

    return [f for f in fronts if f]


def crowding_distance(objectives):
    """Crowding distance for a front's objectives."""
    n = len(objectives)
    if n <= 2:
        return [float('inf')] * n
    d = len(objectives[0])
    distances = [0.0] * n
    for m in range(d):
        sorted_idx = sorted(range(n), key=lambda i: objectives[i][m])
        distances[sorted_idx[0]] = float('inf')
        distances[sorted_idx[-1]] = float('inf')
        mn = objectives[sorted_idx[0]][m]
        mx = objectives[sorted_idx[-1]][m]
        if mx == mn:
            continue
        for k in range(1, n - 1):
            distances[sorted_idx[k]] += (
                objectives[sorted_idx[k + 1]][m] - objectives[sorted_idx[k - 1]][m]
            ) / (mx - mn)
    return distances


def tournament_select_one(objectives, fronts, rng):
    """Binary tournament selection. Returns index."""
    ranks = {}
    for rank, front in enumerate(fronts):
        for i in front:
            ranks[i] = rank
    cd_map = {}
    for front in fronts:
        fobj = [objectives[i] for i in front]
        fcd = crowding_distance(fobj)
        for k, i in enumerate(front):
            cd_map[i] = fcd[k]

    n = len(objectives)
    a, b = rng.randrange(n), rng.randrange(n)
    ra, rb = ranks.get(a, 999), ranks.get(b, 999)
    if ra < rb:
        return a
    if rb < ra:
        return b
    return a if cd_map.get(a, 0) >= cd_map.get(b, 0) else b


def crossover_configs(p1, p2, rng):
    """Uniform crossover of two configs."""
    c1, c2 = dict(p1), dict(p2)
    params = ['tp', 'num_instances', 'scheduler', 'max_num_running_reqs',
              'total_kv_blocks', 'block_size_in_tokens']
    for param in params:
        if rng.random() < 0.5:
            c1[param], c2[param] = c2[param], c1[param]
    return normalize_config(c1), normalize_config(c2)


def mutate_config(config, rng, mutation_rate=0.15, fixed_kv=None):
    """Mutate config parameters with given probability each."""
    c = dict(config)
    if rng.random() < mutation_rate:
        c['tp'] = rng.choice(PARAM_SPACE['tp'])
    if rng.random() < mutation_rate:
        c['num_instances'] = rng.randint(1, 8 // c['tp'])
    if rng.random() < mutation_rate:
        c['scheduler'] = rng.choice(PARAM_SPACE['scheduler'])
    if rng.random() < mutation_rate:
        c['max_num_running_reqs'] = rng.choice(PARAM_SPACE['max_num_running_reqs'])
    if fixed_kv is None and rng.random() < mutation_rate:
        c['total_kv_blocks'] = rng.choice(PARAM_SPACE['total_kv_blocks'])
    if rng.random() < mutation_rate:
        c['block_size_in_tokens'] = rng.choice(PARAM_SPACE['block_size_in_tokens'])
    return normalize_config(c)


def random_config(rng, fixed_kv=None):
    """Sample a random valid config."""
    tp = rng.choice(PARAM_SPACE['tp'])
    return {
        'tp': tp,
        'num_instances': rng.randint(1, 8 // tp),
        'scheduler': rng.choice(PARAM_SPACE['scheduler']),
        'max_num_running_reqs': rng.choice(PARAM_SPACE['max_num_running_reqs']),
        'total_kv_blocks': fixed_kv if fixed_kv is not None else rng.choice(PARAM_SPACE['total_kv_blocks']),
        'block_size_in_tokens': rng.choice(PARAM_SPACE['block_size_in_tokens']),
    }


def run_algorithm_comparison(all_results, ref_point, pop_size, n_gens, budget,
                              checkpoint_interval, nsga2_seed, random_seed,
                              use_3obj=False, fixed_kv=None):
    """Run NSGA-II and random search using pre-computed cache."""
    cache = {}
    for r in all_results:
        if r['objectives'] is not None:
            obj = r['objectives']
            if use_3obj:
                obj = obj[:3]
            cache[config_key(r['config'])] = obj

    all_configs = list(cache.keys())
    exhaustive_nd = get_nondominated(list(cache.values()))
    exhaustive_hv = hypervolume(exhaustive_nd, ref_point)
    target_hv = 0.95 * exhaustive_hv

    print(f"  Exhaustive HV={exhaustive_hv:.4f}, target (95%)={target_hv:.4f}", file=sys.stderr)
    print(f"  Exhaustive Pareto size={len(exhaustive_nd)}, density={len(exhaustive_nd)/len(cache)*100:.2f}%", file=sys.stderr)

    def lookup(cfg):
        k = config_key(cfg)
        return cache.get(k)

    def find_95pct_eval(checkpoints, target):
        prev_eval, prev_hv = 0, 0.0
        for eval_count, hv in checkpoints:
            if hv >= target:
                if prev_hv >= target:
                    return prev_eval
                if hv == prev_hv:
                    return eval_count
                frac = (target - prev_hv) / (hv - prev_hv)
                return prev_eval + frac * (eval_count - prev_eval)
            prev_eval, prev_hv = eval_count, hv
        return budget + 1

    # ── NSGA-II ──
    rng_nsga2 = random.Random(nsga2_seed)
    evaluated = {}
    nsga2_checkpoints = []

    def get_unique_config(rng, fixed_kv=None):
        for _ in range(1000):
            cfg = random_config(rng, fixed_kv=fixed_kv)
            k = config_key(cfg)
            if k in cache and k not in evaluated:
                return cfg
        for k in all_configs:
            if k not in evaluated:
                cfg = {
                    'tp': k[0], 'num_instances': k[1], 'scheduler': k[2],
                    'max_num_running_reqs': k[3], 'total_kv_blocks': k[4],
                    'block_size_in_tokens': k[5],
                }
                return cfg
        return None

    population = []
    pop_objectives = []
    for _ in range(pop_size):
        cfg = get_unique_config(rng_nsga2, fixed_kv=fixed_kv)
        if cfg is None:
            break
        obj = lookup(cfg)
        if obj is None:
            continue
        evaluated[config_key(cfg)] = obj
        population.append(cfg)
        pop_objectives.append(obj)

    current_nd = get_nondominated(list(evaluated.values()))
    current_hv = hypervolume(current_nd, ref_point)
    nsga2_checkpoints.append((len(evaluated), current_hv))

    for gen in range(n_gens):
        fronts = nsga2_fast_nondominated_sort(pop_objectives)
        offspring = []
        offspring_obj = []

        attempts = 0
        while len(offspring) < pop_size and attempts < pop_size * 20:
            attempts += 1
            p1_idx = tournament_select_one(pop_objectives, fronts, rng_nsga2)
            p2_idx = tournament_select_one(pop_objectives, fronts, rng_nsga2)
            child1, child2 = crossover_configs(population[p1_idx], population[p2_idx], rng_nsga2)
            for child in [child1, child2]:
                child = mutate_config(child, rng_nsga2, fixed_kv=fixed_kv)
                child = normalize_config(child)
                k = config_key(child)
                if k in cache and k not in evaluated:
                    obj = lookup(child)
                    evaluated[k] = obj
                    offspring.append(child)
                    offspring_obj.append(obj)
                    if len(offspring) >= pop_size:
                        break

        while len(offspring) < pop_size:
            cfg = get_unique_config(rng_nsga2, fixed_kv=fixed_kv)
            if cfg is None:
                break
            obj = lookup(cfg)
            if obj is None:
                continue
            evaluated[config_key(cfg)] = obj
            offspring.append(cfg)
            offspring_obj.append(obj)

        combined = population + offspring
        combined_obj = pop_objectives + offspring_obj
        combined_fronts = nsga2_fast_nondominated_sort(combined_obj)

        new_pop = []
        new_pop_obj = []
        for front in combined_fronts:
            if len(new_pop) + len(front) <= pop_size:
                for i in front:
                    new_pop.append(combined[i])
                    new_pop_obj.append(combined_obj[i])
            else:
                remaining = pop_size - len(new_pop)
                fobj = [combined_obj[i] for i in front]
                fcd = crowding_distance(fobj)
                sorted_front = sorted(range(len(front)), key=lambda k: -fcd[k])
                for k in sorted_front[:remaining]:
                    new_pop.append(combined[front[k]])
                    new_pop_obj.append(combined_obj[front[k]])
                break

        population = new_pop
        pop_objectives = new_pop_obj

        current_nd = get_nondominated(list(evaluated.values()))
        current_hv = hypervolume(current_nd, ref_point)
        nsga2_checkpoints.append((len(evaluated), current_hv))

    nsga2_95pct = find_95pct_eval(nsga2_checkpoints, target_hv)

    # ── Random Search ──
    rng_rand = random.Random(random_seed)
    random_evaluated = {}
    random_checkpoints = []

    shuffled = list(all_configs)
    rng_rand.shuffle(shuffled)
    shuffled = shuffled[:budget]

    for i, k in enumerate(shuffled):
        obj = cache[k]
        random_evaluated[k] = obj
        if (i + 1) % checkpoint_interval == 0:
            nd = get_nondominated(list(random_evaluated.values()))
            hv = hypervolume(nd, ref_point)
            random_checkpoints.append((i + 1, hv))

    if not random_checkpoints or random_checkpoints[-1][0] != budget:
        nd = get_nondominated(list(random_evaluated.values()))
        hv = hypervolume(nd, ref_point)
        random_checkpoints.append((budget, hv))

    random_95pct = find_95pct_eval(random_checkpoints, target_hv)

    gap = random_95pct - nsga2_95pct

    print(f"  NSGA-II 95% eval: {nsga2_95pct:.1f}", file=sys.stderr)
    print(f"  Random  95% eval: {random_95pct:.1f}", file=sys.stderr)
    print(f"  Gap (random - nsga2): {gap:.1f}", file=sys.stderr)

    return {
        'exhaustive_hv': exhaustive_hv,
        'exhaustive_pareto_size': len(exhaustive_nd),
        'exhaustive_total_configs': len(cache),
        'exhaustive_density': len(exhaustive_nd) / len(cache),
        'target_hv': target_hv,
        'nsga2_checkpoints': nsga2_checkpoints,
        'random_checkpoints': random_checkpoints,
        'nsga2_95pct_eval': nsga2_95pct,
        'random_95pct_eval': random_95pct,
        'gap': gap,
    }
