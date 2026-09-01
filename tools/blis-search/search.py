#!/usr/bin/env python3
"""BLIS Multi-Objective Configuration Search.

Discovers Pareto-optimal configurations for BLIS using random + hierarchical
sampling with HV convergence detection and diversity-biased Phase 2 targeting.

Algorithm developed and validated across 15 iterations of hypothesis-driven
experimentation (see .nous/-run/ for the full campaign).
"""
import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional, Set

from convergence import detect_convergence, get_pareto_gpu_classes, get_pareto_tp_classes
from evaluator import evaluate_batch
from pareto import compute_hypervolume_2d, extract_pareto_front
from search_space import (
    DiversityBiasedSampler,
    HierarchicalRandomSampler,
    get_all_param_flags,
    get_param_values,
    load_search_space,
)


def parse_objectives(objectives_str: Optional[str], space: dict) -> List[Dict]:
    """Parse objectives from CLI string or fall back to search space defaults."""
    if objectives_str:
        objectives = []
        for part in objectives_str.split(","):
            metric, direction = part.strip().split(":")
            objectives.append({"metric": metric.strip(), "direction": direction.strip()})
        return objectives
    return space.get("objectives", [
        {"metric": "responses_per_sec", "direction": "maximize"},
        {"metric": "ttft_p99_ms", "direction": "minimize"},
    ])


def parse_slo(slo_str: Optional[str]) -> List[Dict]:
    """Parse SLO constraints like 'ttft_p99_ms<200,responses_per_sec>50'."""
    if not slo_str:
        return []
    constraints = []
    for part in slo_str.split(","):
        part = part.strip()
        if "<" in part:
            metric, threshold = part.split("<")
            constraints.append({"metric": metric.strip(), "op": "<", "threshold": float(threshold)})
        elif ">" in part:
            metric, threshold = part.split(">")
            constraints.append({"metric": metric.strip(), "op": ">", "threshold": float(threshold)})
    return constraints


def filter_by_slo(results: List[dict], slo: List[Dict]) -> List[dict]:
    """Filter results that satisfy all SLO constraints."""
    if not slo:
        return results
    feasible = []
    for r in results:
        passes = True
        for constraint in slo:
            val = r["metrics"].get(constraint["metric"])
            if val is None:
                passes = False
                break
            if constraint["op"] == "<" and val >= constraint["threshold"]:
                passes = False
                break
            if constraint["op"] == ">" and val <= constraint["threshold"]:
                passes = False
                break
        if passes:
            feasible.append(r)
    return feasible


def run_search(
    space: dict,
    blis_binary: str,
    fixed_params: Dict[str, str],
    objectives: List[Dict],
    reference_point: Dict[str, float],
    budget: int = 500,
    workers: int = 8,
    search_seed: int = 42,
    strategy: str = "diversity",
    convergence_k: int = 30,
    convergence_epsilon: float = 0.001,
    eval_timeout: float = 20.0,
    verbose: bool = False,
) -> dict:
    """Run the two-phase search algorithm.

    Phase 1: Random hierarchical sampling until HV convergence.
    Phase 2: Diversity-biased targeting of under-represented TP classes.
    """
    rng = random.Random(search_seed)
    param_flags = get_all_param_flags(space)
    metric_keys = [obj["metric"] for obj in objectives]
    # Include gpus_used in extraction even if not a Pareto objective
    if "gpus_used" not in metric_keys:
        metric_keys = metric_keys + ["gpus_used"]
    infra_mode = "infrastructure" in space

    phase1_sampler = HierarchicalRandomSampler(rng, space)
    active_sampler = phase1_sampler

    all_results: List[dict] = []
    hv_trajectory: List[float] = []
    start_time = time.time()
    remaining = budget
    valid_eval_count = 0
    phase = 1
    convergence_eval = None
    phase2_sampler = None

    while remaining > 0:
        batch_size = min(workers, remaining)
        batch_configs = []
        attempts = 0
        while len(batch_configs) < batch_size and attempts < batch_size * 100:
            attempts += 1
            cfg = active_sampler.sample()
            if cfg is not None:
                batch_configs.append(cfg)
        if not batch_configs:
            break

        batch_results = evaluate_batch(
            batch_configs, workers, blis_binary, fixed_params, param_flags, metric_keys, eval_timeout
        )

        for res in batch_results:
            if res is not None:
                all_results.append(res)
                valid_eval_count += 1
                pf = extract_pareto_front(all_results, objectives)
                hv = compute_hypervolume_2d(pf, objectives, reference_point)
                hv_trajectory.append(hv)

                if phase2_sampler is not None:
                    if infra_mode:
                        current_classes = get_pareto_gpu_classes(pf)
                    else:
                        current_classes = get_pareto_tp_classes(pf)
                    phase2_sampler.update_pareto(current_classes)

        remaining -= batch_size

        if phase == 1 and strategy == "diversity":
            conv_eval = detect_convergence(hv_trajectory, convergence_k, convergence_epsilon)
            if conv_eval is not None and convergence_eval is None:
                convergence_eval = conv_eval
                phase = 2

                current_pf = extract_pareto_front(all_results, objectives)
                if infra_mode:
                    current_classes = get_pareto_gpu_classes(current_pf)
                else:
                    current_classes = get_pareto_tp_classes(current_pf)

                if verbose:
                    current_hv = compute_hypervolume_2d(current_pf, objectives, reference_point)
                    if infra_mode:
                        from search_space import generate_topologies
                        all_gpu_counts = sorted(set(
                            t["gpus_used"] for t in generate_topologies(space["infrastructure"])
                        ))
                        missing = sorted(gc for gc in all_gpu_counts if gc not in current_classes)
                        label = "GPU tiers"
                    else:
                        missing = sorted(
                            tp for tp in space["parameters"]["tp"]["values"]
                            if tp not in current_classes
                        )
                        label = "TP"
                    print(
                        f"  Phase 1 converged at eval {convergence_eval}, "
                        f"HV={current_hv:.4f}, missing {label}: {missing}",
                        file=sys.stderr,
                        flush=True,
                    )

                phase2_sampler = DiversityBiasedSampler(
                    rng, space, current_classes, verbose=verbose
                )
                active_sampler = phase2_sampler

        if verbose and remaining > 0 and (budget - remaining) % 50 == 0:
            pf = extract_pareto_front(all_results, objectives) if all_results else []
            hv = compute_hypervolume_2d(pf, objectives, reference_point)
            print(
                f"  eval {budget - remaining}/{budget}, valid={valid_eval_count}, "
                f"HV={hv:.4f}, phase={phase}, t={time.time() - start_time:.1f}s",
                file=sys.stderr,
                flush=True,
            )

    wall_time = time.time() - start_time
    pareto_front = extract_pareto_front(all_results, objectives) if all_results else []
    final_hv = compute_hypervolume_2d(pareto_front, objectives, reference_point)

    if verbose:
        if infra_mode:
            gpu_classes = get_pareto_gpu_classes(pareto_front)
            class_info = f"gpu_tiers={sorted(gpu_classes)}"
        else:
            tp_classes = get_pareto_tp_classes(pareto_front)
            class_info = f"tp_classes={sorted(tp_classes)}"
        print(
            f"  DONE: HV={final_hv:.6f}, valid={valid_eval_count}/{budget}, "
            f"pareto_size={len(pareto_front)}, {class_info}, "
            f"t={wall_time:.1f}s",
            file=sys.stderr,
            flush=True,
        )

    return {
        "pareto_front": pareto_front,
        "all_results": all_results,
        "search_stats": {
            "budget_used": budget,
            "valid_evals": valid_eval_count,
            "convergence_eval": convergence_eval,
            "wall_time_seconds": round(wall_time, 2),
            "final_hv": round(final_hv, 6),
            "strategy": strategy,
            "phase1_evals": convergence_eval or valid_eval_count,
            "phase2_evals": valid_eval_count - (convergence_eval or valid_eval_count),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="BLIS multi-objective configuration search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python search.py --model qwen/qwen3-14b --hardware H100 --workload chatbot --rate 200
  python search.py --model qwen/qwen3-14b --hardware H100 --workload summarization \\
    --rate 1000 --budget 500 --strategy diversity --verbose
  python search.py --model qwen/qwen3-14b --hardware H100 --workload chatbot --rate 200 \\
    --slo "ttft_p99_ms<200,responses_per_sec>50"

  # Trace-driven space (--trace-header supplied by the search space YAML):
  # --workload/--rate are not required and are not passed to BLIS.
  python search.py --model qwen/qwen3-14b --hardware H100 \\
    --search-space weka-qwen3-14b.yaml \\
    --blis-binary blis-replay-shim.sh --budget 300 --verbose
""",
    )

    parser.add_argument("--model", required=True, help="LLM model name (e.g. qwen/qwen3-14b)")
    parser.add_argument("--hardware", required=True, help="GPU type (e.g. H100, A100-80)")
    parser.add_argument(
        "--workload", default=None,
        help="Workload preset name. Required unless the search space supplies a trace "
             "(a --trace-header parameter), in which case the trace governs arrivals.",
    )
    parser.add_argument(
        "--rate", type=int, default=None,
        help="Request arrival rate (req/s). Same requirement as --workload.",
    )
    parser.add_argument(
        "--num-requests", type=int, default=None,
        help="Total requests per eval (default 300; omitted entirely in trace mode)",
    )
    parser.add_argument("--seed", type=int, default=42, help="BLIS simulation seed")

    parser.add_argument("--budget", type=int, default=500, help="Max evaluations")
    parser.add_argument("--workers", type=int, default=8, help="Parallel evaluation workers")
    parser.add_argument("--search-seed", type=int, default=1, help="Search RNG seed")
    parser.add_argument(
        "--search-space", default=None,
        help="YAML search space file (default: built-in defaults.yaml)",
    )
    parser.add_argument(
        "--strategy", choices=["diversity", "random"], default="diversity",
        help="Phase 2 strategy: diversity (default) or random (no Phase 2)",
    )
    parser.add_argument("--convergence-k", type=int, default=30, help="Convergence window size")
    parser.add_argument("--convergence-epsilon", type=float, default=0.001, help="HV improvement threshold")
    parser.add_argument("--eval-timeout", type=float, default=20.0, help="Per-eval timeout (seconds)")

    parser.add_argument("--objectives", default=None, help="Objectives as 'metric:direction,...'")
    parser.add_argument("--slo", default=None, help="SLO constraints as 'metric<value,...'")

    parser.add_argument("--output", default=None, help="Output file path (default: stdout)")
    parser.add_argument("--output-all", action="store_true", help="Include all evaluated configs in output (not just Pareto front)")
    parser.add_argument("--blis-binary", default="./blis", help="Path to BLIS binary")
    parser.add_argument("--latency-model", default="trained-physics", help="Latency model backend")
    parser.add_argument("--verbose", action="store_true", help="Print progress to stderr")

    args = parser.parse_args()

    # Load search space
    if args.search_space:
        space_path = args.search_space
    else:
        space_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "defaults.yaml")
    space = load_search_space(space_path)

    # A space that injects a trace (--trace-header) drives `blis replay`, where
    # arrivals come entirely from the trace: --workload/--rate/--num-requests are
    # not just optional but meaningless, so they are neither required nor passed
    # through. Anywhere else they are still mandatory `blis run` inputs.
    trace_mode = "--trace-header" in get_all_param_flags(space).values()
    if not trace_mode:
        missing = [
            flag for flag, val in (("--workload", args.workload), ("--rate", args.rate))
            if val is None
        ]
        if missing:
            parser.error(
                f"{', '.join(missing)} required: the search space {os.path.basename(space_path)} "
                "supplies no --trace-header parameter, so arrivals must come from a workload preset"
            )

    objectives = parse_objectives(args.objectives, space)
    slo = parse_slo(args.slo)

    ref = space.get("hypervolume_reference", {})
    reference_point = {obj["metric"]: ref.get(obj["metric"], 100000.0) for obj in objectives}

    fixed_params = {
        "--model": args.model,
        "--hardware": args.hardware,
        "--latency-model": args.latency_model,
        "--seed": str(args.seed),
    }
    # Only real inputs land in fixed_params — it is both the command-line prefix
    # and the output's `spec` block, so a trace-driven run no longer claims a
    # workload preset it never used.
    if args.workload is not None:
        fixed_params["--workload"] = args.workload
    if args.rate is not None:
        fixed_params["--rate"] = str(args.rate)
    if args.num_requests is not None:
        fixed_params["--num-requests"] = str(args.num_requests)
    elif not trace_mode:
        fixed_params["--num-requests"] = "300"

    if args.verbose:
        if trace_mode:
            traces = [
                v for name, flag in get_all_param_flags(space).items() if flag == "--trace-header"
                for v in get_param_values(space, name)
            ]
            source = "trace=" + (traces[0] if len(traces) == 1 else f"{len(traces)} candidates")
        else:
            source = f"workload={args.workload} rate={args.rate}"
        print(
            f"BLIS Search: model={args.model} hardware={args.hardware} "
            f"{source} budget={args.budget} "
            f"workers={args.workers} strategy={args.strategy}",
            file=sys.stderr,
            flush=True,
        )

    result = run_search(
        space=space,
        blis_binary=args.blis_binary,
        fixed_params=fixed_params,
        objectives=objectives,
        reference_point=reference_point,
        budget=args.budget,
        workers=args.workers,
        search_seed=args.search_seed,
        strategy=args.strategy,
        convergence_k=args.convergence_k,
        convergence_epsilon=args.convergence_epsilon,
        eval_timeout=args.eval_timeout,
        verbose=args.verbose,
    )

    pareto_front = result["pareto_front"]
    slo_feasible = filter_by_slo(pareto_front, slo) if slo else pareto_front

    output = {
        "pareto_front": [
            {"config": r["config"], "metrics": r["metrics"]} for r in pareto_front
        ],
        "search_stats": result["search_stats"],
        "objectives": objectives,
        "spec": {k.lstrip("-"): v for k, v in fixed_params.items()},
        # Self-describing output so a reader can reconstruct the exact `blis run`
        # command: knob names do not map mechanically to flags (replicas ->
        # --num-instances, flow_control is a bare boolean flag).
        "param_flags": get_all_param_flags(space),
        "space_file": os.path.basename(space_path),
    }
    if args.output_all:
        output["all_results"] = [
            {"config": r["config"], "metrics": r["metrics"]} for r in result["all_results"]
        ]
    if slo:
        output["slo_feasible"] = [
            {"config": r["config"], "metrics": r["metrics"]} for r in slo_feasible
        ]
        output["slo_constraints"] = slo

    output_json = json.dumps(output, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
            f.write("\n")
        if args.verbose:
            print(f"  Results written to {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
