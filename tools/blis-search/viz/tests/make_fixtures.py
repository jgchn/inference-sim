#!/usr/bin/env python3
"""Generate the committed viz test fixtures. Deterministic: same bytes every run.

Usage:
    .venv/bin/python tools/blis-search/viz/tests/make_fixtures.py
    .venv/bin/python tools/blis-search/viz/tests/make_fixtures.py --from-real search_output.json
"""
import argparse
import copy
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")

SCHEDULERS = ["fcfs", "sjf", "priority-fcfs", "reverse-priority"]
ROUTING = ["round-robin", "least-loaded", "weighted", "always-busiest"]
ADMISSION = ["always-admit", "token-bucket", "tier-shed"]

PARAM_FLAGS = {
    "tp": "--tp",
    "replicas": "--num-instances",
    "scheduler": "--scheduler",
    "max_batch": "--max-num-running-reqs",
    "max_scheduled_tokens": "--max-num-scheduled-tokens",
    "long_prefill": "--long-prefill-token-threshold",
    "block_size": "--block-size-in-tokens",
    "gpu_mem_util": "--gpu-memory-utilization",
    "routing_policy": "--routing-policy",
    "admission_policy": "--admission-policy",
    "preemption_policy": "--preemption-policy",
    "snapshot_refresh": "--snapshot-refresh-interval",
    "scorer_profile": "--routing-scorers",
    "flow_control": "--flow-control",
    "saturation_detector": "--saturation-detector",
    "queue_depth_threshold": "--queue-depth-threshold",
    "kv_util_threshold": "--kv-cache-util-threshold",
    "dispatch_order": "--dispatch-order",
    "max_gateway_queue_depth": "--max-gateway-queue-depth",
    "request_ttl": "--request-ttl",
    "pd_decider": "--pd-decider",
    "prefill_instances": "--prefill-instances",
    "decode_instances": "--decode-instances",
    "pd_transfer_bandwidth": "--pd-transfer-bandwidth",
}

SPEC = {
    "model": "Qwen/Qwen3-32B",
    "hardware": "H100",
    "latency-model": "trained-physics",
    "seed": "42",
    "workload": "chatbot",
    "rate": "50",
    "num-requests": "2000",
}


def make_eval(rng):
    """One synthetic {config, metrics} evaluation with realistic knob coupling."""
    tp = rng.choice([1, 2, 4, 8])
    replicas = rng.choice(list(range(1, 8 // tp + 1)))
    routing = "round-robin" if replicas == 1 else rng.choice(ROUTING)
    fc = rng.random() < 0.4
    config = {
        "tp": tp,
        "replicas": replicas,
        "gpus_used": tp * replicas,
        "scheduler": rng.choice(SCHEDULERS),
        "max_batch": rng.choice([32, 64, 128, 256, 512]),
        "max_scheduled_tokens": rng.choice([2048, 4096, 8192]),
        "long_prefill": rng.choice([0, 1024, 2048, 4096]),
        "block_size": rng.choice([16, 32]),
        "gpu_mem_util": rng.choice([0.85, 0.9, 0.95]),
        "routing_policy": routing,
        "admission_policy": "always-admit" if replicas == 1 else rng.choice(ADMISSION),
        "preemption_policy": rng.choice(["fcfs", "priority"]),
        "snapshot_refresh": 50000,
        "scorer_profile": "queue-depth:2,kv-utilization:1" if routing == "weighted" else None,
        "flow_control": fc,
        "saturation_detector": "utilization" if fc else None,
        "queue_depth_threshold": rng.choice([5, 10, 20]) if fc else None,
        "kv_util_threshold": rng.choice([0.8, 0.9]) if fc else None,
        "dispatch_order": rng.choice(["fcfs", "priority"]) if fc else None,
        "max_gateway_queue_depth": 500 if fc else None,
        "request_ttl": 10000000 if fc else None,
        "pd_decider": "never",
        "prefill_instances": 0,
        "decode_instances": 0,
        "pd_transfer_bandwidth": 25.0,
    }
    gpus = tp * replicas
    # Throughput rises with GPUs and is helped by weighted routing; TTFT falls
    # with GPUs and is helped by flow control. The jitter keeps the cloud thick.
    rps = 18.0 * gpus ** 0.75 * (1.12 if routing == "weighted" else 1.0) * (0.8 + 0.4 * rng.random())
    ttft = 400.0 / gpus ** 0.9 * (0.88 if fc else 1.0) * (0.7 + 0.6 * rng.random())
    e2e = ttft + 900.0 + 300.0 * rng.random()
    itl = 3.5 + 2.5 * rng.random()
    metrics = {
        "responses_per_sec": round(rps, 6),
        "ttft_p99_ms": round(ttft, 5),
        "gpus_used": gpus,
        "completed_requests": 2000,
        "still_queued": 0,
        "still_running": 0,
        "injected_requests": 2000,
        "total_input_tokens": 497382,
        "total_output_tokens": 495611,
        "tokens_per_sec": round(rps * 248.0, 1),
        "e2e_mean_ms": round(e2e, 1),
        "e2e_p90_ms": round(e2e * 1.5, 1),
        "e2e_p95_ms": round(e2e * 1.6, 1),
        "e2e_p99_ms": round(e2e * 1.8, 1),
        "ttft_mean_ms": round(ttft * 0.7, 1),
        "ttft_p90_ms": round(ttft * 0.9, 1),
        "ttft_p95_ms": round(ttft * 0.95, 1),
        "itl_mean_ms": round(itl, 2),
        "itl_p90_ms": round(itl * 1.1, 2),
        "itl_p95_ms": round(itl * 1.15, 2),
        "itl_p99_ms": round(itl * 1.25, 2),
        "scheduling_delay_p99_ms": round(ttft * 0.45, 1),
        "preemption_count": rng.choice([0, 0, 0, 3, 11]),
        "dropped_unservable": 0,
        "length_capped_requests": 0,
        "timed_out_requests": 0,
        "vllm_estimated_duration_s": round(2000.0 / rps, 3),
    }
    return {"config": config, "metrics": metrics}


def dominates(a, b, objectives):
    """True when a dominates b (a is no worse everywhere, strictly better once)."""
    strictly = False
    for obj in objectives:
        av, bv = a["metrics"][obj["metric"]], b["metrics"][obj["metric"]]
        if obj["direction"] == "maximize":
            if av < bv:
                return False
            if av > bv:
                strictly = True
        else:
            if av > bv:
                return False
            if av < bv:
                strictly = True
    return strictly


def front_of(evals, objectives):
    return [e for e in evals if not any(dominates(o, e, objectives) for o in evals if o is not e)]


def satisfies(entry, constraints):
    for c in constraints:
        val = entry["metrics"].get(c["metric"])
        if val is None:
            return False
        if c["op"] == "<" and val >= c["threshold"]:
            return False
        if c["op"] == ">" and val <= c["threshold"]:
            return False
    return True


def stats(n_valid, convergence_eval, hv):
    return {
        "budget_used": n_valid,
        "valid_evals": n_valid,
        "convergence_eval": convergence_eval,
        "wall_time_seconds": 41.7,
        "final_hv": hv,
        "strategy": "diversity",
        "phase1_evals": convergence_eval,
        "phase2_evals": n_valid - convergence_eval,
    }


def base_output(evals, objectives, *, all_results=True, slo=None, with_flags=True):
    evals = copy.deepcopy(evals)   # callers mutate the returned fixture in place
    front = front_of(evals, objectives)
    out = {
        "pareto_front": [{"config": e["config"], "metrics": e["metrics"]} for e in front],
        "search_stats": stats(len(evals), 18, 0.2317),
        "objectives": objectives,
        "spec": dict(SPEC),
    }
    if with_flags:
        out["param_flags"] = dict(PARAM_FLAGS)
        out["space_file"] = "full-stack.yaml"
    if all_results:
        out["all_results"] = [{"config": e["config"], "metrics": e["metrics"]} for e in evals]
    if slo is not None:
        out["slo_feasible"] = [e for e in out["pareto_front"] if satisfies(e, slo)]
        out["slo_constraints"] = slo
    return out


def write(name, obj):
    path = os.path.join(FIXTURES, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=True)
        f.write("\n")
    print("wrote %s (%d bytes)" % (name, os.path.getsize(path)))


TWO_OBJ = [
    {"metric": "responses_per_sec", "direction": "maximize"},
    {"metric": "ttft_p99_ms", "direction": "minimize"},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-real", default=None,
                    help="path to a real search_output.json to subsample into real_subsample.json")
    args = ap.parse_args()

    os.makedirs(FIXTURES, exist_ok=True)
    rng = random.Random(20260820)
    evals = [make_eval(rng) for _ in range(60)]
    # The degenerate fixtures need a front and a cloud, not a big cloud.
    small = evals[:14]

    slo = [
        {"metric": "ttft_p99_ms", "op": "<", "threshold": 200.0},
        {"metric": "responses_per_sec", "op": ">", "threshold": 40.0},
    ]
    write("two_obj_slo.json", base_output(evals, TWO_OBJ, slo=slo))

    # Front-only mode, and deliberately pre-§7 (no param_flags / space_file).
    write("front_only.json", base_output(evals, TWO_OBJ, all_results=False, with_flags=False))

    three = TWO_OBJ + [{"metric": "gpus_used", "direction": "minimize"}]
    write("three_obj.json", base_output(evals, three))

    four = three + [{"metric": "e2e_p99_ms", "direction": "minimize"}]
    write("four_obj.json", base_output(evals, four))

    write("single_objective.json",
          base_output(small, [{"metric": "responses_per_sec", "direction": "maximize"}]))

    single = base_output(small, TWO_OBJ)
    single["pareto_front"] = single["pareto_front"][:1]
    write("single_member_front.json", single)

    impossible = [{"metric": "ttft_p99_ms", "op": "<", "threshold": 0.5}]
    write("empty_slo_feasible.json", base_output(small, TWO_OBJ, slo=impossible))

    # One point missing an objective metric, one with NaN, one with +inf.
    dirty = base_output(small, TWO_OBJ)
    del dirty["all_results"][3]["metrics"]["ttft_p99_ms"]
    dirty["all_results"][7]["metrics"]["ttft_p99_ms"] = float("nan")
    dirty["all_results"][11]["metrics"]["responses_per_sec"] = float("inf")
    write("dirty_metrics.json", dirty)

    # A zero-range objective: every point reports the same throughput.
    flat = base_output(small, TWO_OBJ)
    for entry in flat["all_results"]:
        entry["metrics"]["responses_per_sec"] = 42.0
    flat["pareto_front"] = front_of(
        [{"config": e["config"], "metrics": e["metrics"]} for e in flat["all_results"]], TWO_OBJ)
    write("constant_objective.json", flat)

    # Exactly one distinct GPU count across the plotted set: the color channel is
    # wasted and the caption must say so.
    one_gpu = base_output([e for e in evals if e["metrics"]["gpus_used"] == 8][:12], TWO_OBJ)
    write("single_gpu_count.json", one_gpu)

    # No GPU dimension derivable at all: gpus_used absent from config and metrics,
    # and tp/replicas absent too, so the channel is dropped.
    nogpu = base_output(small, TWO_OBJ)
    for bucket in ("pareto_front", "all_results"):
        for entry in nogpu[bucket]:
            entry["metrics"].pop("gpus_used", None)
            for knob in ("gpus_used", "tp", "replicas"):
                entry["config"].pop(knob, None)
    write("no_gpu_dimension.json", nogpu)

    # The rejection case: search/pareto_search.py's unrelated schema.
    write("foreign_schema.json", {
        "best_objectives": {"responses_per_sec": 169.0},
        "pareto_front": [{"objectives": {"responses_per_sec": 169.0}, "flags": ["--tp", "8"]}],
        "recommended_config": {"tp": 8},
        "main": {"pareto_front": []},
        "transfer": None,
    })

    if args.from_real:
        with open(args.from_real) as f:
            real = json.load(f)
        allr = real.get("all_results", [])
        if len(allr) < 60:
            raise SystemExit("--from-real file has only %d all_results" % len(allr))
        stride = len(allr) // 60
        sub = [allr[i * stride] for i in range(60)]
        objectives = real["objectives"]
        out = {
            "pareto_front": front_of(sub, objectives),
            "all_results": sub,
            "objectives": objectives,
            "spec": real["spec"],
            "search_stats": real["search_stats"],
        }
        if "slo_constraints" in real:
            out["slo_constraints"] = real["slo_constraints"]
            out["slo_feasible"] = [e for e in out["pareto_front"]
                                   if satisfies(e, real["slo_constraints"])]
        write("real_subsample.json", out)


if __name__ == "__main__":
    main()
