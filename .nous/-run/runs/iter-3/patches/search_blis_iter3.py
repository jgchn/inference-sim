#!/usr/bin/env python3
"""
BLIS Configuration Search Script - Iteration 3
Extends iter-2 with --model flag for cross-model comparison and updated global bests
keyed by (model_shortname, rate, num_requests).

Usage:
  python3 search_blis.py --strategy random --budget 100 --seeds 42,43,44,45,46 \
      --rate 200 --output results.json

  # Hierarchical search:
  python3 search_blis.py --strategy hierarchical --budget 100 --seeds 42,43,44,45,46 \
      --rate 200 --output results.json

  # h-main strategies (random+tpe+hierarchical) for a specific model:
  python3 search_blis.py --strategy h-main-strategies --budget 100 --seeds 42,43,44,45,46 \
      --rate 500 --num-requests 1000 --model qwen/qwen3-14b --output results.json

  # Cross-model comparison:
  python3 search_blis.py --strategy h-main-strategies --budget 100 --seeds 42,43,44,45,46 \
      --rate 500 --num-requests 1000 --model meta-llama/llama-3.1-8b-instruct --output results.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import optuna
from scipy.stats import qmc

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Search space definition ────────────────────────────────────────────────

TP_OPTIONS = [1, 2, 4, 8]
VALID_TP_INSTANCES = [
    (tp, ni)
    for tp in TP_OPTIONS
    for ni in range(1, 8 // tp + 1)
]

SCHEDULER_OPTIONS = ["fcfs", "priority-fcfs", "sjf", "reverse-priority"]
MAX_RUNNING_OPTIONS = [32, 64, 128, 256, 512]
MAX_TOKENS_OPTIONS = [2048, 4096, 8192]
PREFILL_THRESHOLD_OPTIONS = [0, 1024, 2048, 4096]
BLOCK_SIZE_OPTIONS = [16, 32]
ROUTING_OPTIONS = ["round-robin", "least-loaded", "weighted"]
ADMISSION_OPTIONS = ["always-admit", "tier-shed"]
PREEMPTION_OPTIONS = ["fcfs", "priority"]

# Phase 1 defaults for hierarchical search (max GPU utilization per TP level)
# (tp, num_instances) - always use max instances for the given TP
PHASE1_TP_CONFIGS = [
    (1, 8),  # TP=1, 8 instances
    (2, 4),  # TP=2, 4 instances
    (4, 2),  # TP=4, 2 instances
    (8, 1),  # TP=8, 1 instance
]

# Global best scores keyed by (model_shortname, rate, num_requests)
# Calibrated values from probing across iterations
GLOBAL_BEST_TABLE = {
    ("qwen3-14b", 200, 200): 0.117806,    # iter-2, blis_seed=42
    ("qwen3-14b", 500, 1000): 0.178468,   # iter-3 probe, blis_seed=42
    ("llama-3.1-8b", 200, 200): 0.143342, # iter-3 probe, blis_seed=42
    ("llama-3.1-8b", 500, 1000): 0.209119, # iter-3 probe, blis_seed=42
}
WITHIN_BEST_THRESHOLD = 0.99  # 1% tolerance

# MODEL is set at startup from --model arg; used by build_blis_cmd
MODEL = "qwen/qwen3-14b"

BLIS_BASE = [
    "./blis", "run",
    "--hardware", "H100",
    "--latency-model", "trained-physics",
]


def model_shortname(model: str) -> str:
    """Extract short model name for GLOBAL_BEST_TABLE lookup.
    qwen/qwen3-14b -> qwen3-14b
    meta-llama/llama-3.1-8b-instruct -> llama-3.1-8b
    """
    name = model.split("/")[-1]
    for suffix in ["-instruct", "-chat", "-hf"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def build_blis_cmd(config: dict, num_requests: int, rate: float, seed: int,
                   fitness_weights: str | None = "throughput:0.4,p99_ttft:0.3,p99_e2e:0.3") -> list[str]:
    cmd = BLIS_BASE + ["--model", MODEL, "--num-requests", str(num_requests),
        "--rate", str(rate),
        "--seed", str(seed),
        "--tp", str(config["tp"]),
        "--num-instances", str(config["num_instances"]),
        "--scheduler", config["scheduler"],
        "--max-num-running-reqs", str(config["max_running"]),
        "--max-num-scheduled-tokens", str(config["max_tokens"]),
        "--long-prefill-token-threshold", str(config["prefill_threshold"]),
        "--block-size-in-tokens", str(config["block_size"]),
        "--routing-policy", config["routing"],
        "--admission-policy", config["admission"],
        "--preemption-policy", config["preemption"],
    ]
    if fitness_weights:
        cmd += ["--fitness-weights", fitness_weights]
    return cmd


def parse_cluster_metrics(stdout: str) -> dict:
    """Parse cluster-level metrics from BLIS stdout. Handles multi-instance output."""
    # Find all JSON blocks between === Simulation Metrics === markers
    blocks = re.findall(r"=== Simulation Metrics ===\s*(\{.*?\})", stdout, re.DOTALL)
    for block_str in blocks:
        try:
            data = json.loads(block_str)
            if data.get("instance_id") == "cluster" or len(blocks) == 1:
                return data
        except json.JSONDecodeError:
            continue
    return {}


def run_blis(config: dict, num_requests: int = 200, rate: float = 200.0,
             seed: int = 42, fitness_weights: str | None = "throughput:0.4,p99_ttft:0.3,p99_e2e:0.3") -> dict | None:
    """Run BLIS with given config. Returns parsed result or None on failure."""
    cmd = build_blis_cmd(config, num_requests, rate, seed, fitness_weights)
    try:
        t0 = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                                cwd=Path(__file__).parent)
        wall_time = time.perf_counter() - t0
        if result.returncode != 0:
            return None

        stdout = result.stdout
        metrics = parse_cluster_metrics(stdout)

        if fitness_weights:
            score_match = re.search(r"Score:\s+([\d.]+)", stdout)
            if not score_match:
                return None
            score = float(score_match.group(1))
            return {"config": config, "score": score, "wall_time_s": wall_time, "metrics": metrics}
        else:
            # Raw metrics mode (for NSGA-II)
            throughput = metrics.get("responses_per_sec", 0.0)
            e2e_p99 = metrics.get("e2e_p99_ms", 1e9)
            ttft_p99 = metrics.get("ttft_p99_ms", 1e9)
            if throughput <= 0:
                return None
            return {
                "config": config,
                "throughput": throughput,
                "e2e_p99": e2e_p99,
                "ttft_p99": ttft_p99,
                "wall_time_s": wall_time,
                "metrics": metrics,
            }
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


# ── Config sampling utilities ──────────────────────────────────────────────

def sample_random_config(rng: np.random.Generator) -> dict:
    tp_ni = VALID_TP_INSTANCES[rng.integers(0, len(VALID_TP_INSTANCES))]
    tp, ni = tp_ni
    return {
        "tp": tp,
        "num_instances": ni,
        "scheduler": SCHEDULER_OPTIONS[rng.integers(0, len(SCHEDULER_OPTIONS))],
        "max_running": MAX_RUNNING_OPTIONS[rng.integers(0, len(MAX_RUNNING_OPTIONS))],
        "max_tokens": MAX_TOKENS_OPTIONS[rng.integers(0, len(MAX_TOKENS_OPTIONS))],
        "prefill_threshold": PREFILL_THRESHOLD_OPTIONS[rng.integers(0, len(PREFILL_THRESHOLD_OPTIONS))],
        "block_size": BLOCK_SIZE_OPTIONS[rng.integers(0, len(BLOCK_SIZE_OPTIONS))],
        "routing": ROUTING_OPTIONS[rng.integers(0, len(ROUTING_OPTIONS))],
        "admission": ADMISSION_OPTIONS[rng.integers(0, len(ADMISSION_OPTIONS))],
        "preemption": PREEMPTION_OPTIONS[rng.integers(0, len(PREEMPTION_OPTIONS))],
    }


def make_phase1_config(tp: int, instances: int) -> dict:
    """Build a phase-1 config for hierarchical search with sensible defaults."""
    routing = "round-robin" if instances == 1 else "least-loaded"
    return {
        "tp": tp,
        "num_instances": instances,
        "scheduler": "fcfs",
        "max_running": 256,
        "max_tokens": 4096,
        "prefill_threshold": 1024,
        "block_size": 16,
        "routing": routing,
        "admission": "always-admit",
        "preemption": "fcfs",
    }


def compute_evals_to_best(evaluations: list[dict], global_best: float) -> int:
    """Find the first eval index (1-based) where best_so_far >= global_best * 0.99."""
    threshold = global_best * WITHIN_BEST_THRESHOLD
    best_so_far = -1.0
    for i, ev in enumerate(evaluations):
        score = ev.get("score", ev.get("scalarized_score", 0.0))
        if score > best_so_far:
            best_so_far = score
        if best_so_far >= threshold:
            return i + 1  # 1-based eval index
    return len(evaluations) + 1  # Never found


# ── Search strategies ──────────────────────────────────────────────────────

def run_random_search(budget: int, search_seed: int, num_requests: int = 200,
                      rate: float = 200.0, blis_seed: int = 42) -> dict:
    rng = np.random.default_rng(search_seed)
    results = []
    best_score = -1.0
    best_config = None

    for i in range(budget):
        config = sample_random_config(rng)
        out = run_blis(config, num_requests=num_requests, rate=rate, seed=blis_seed)
        if out is None:
            continue
        score = out["score"]
        results.append({
            "eval_idx": i,
            "score": score,
            "config": config,
            "wall_time_s": out["wall_time_s"],
            "best_so_far": max(best_score, score),
        })
        if score > best_score:
            best_score = score
            best_config = config

    return {
        "strategy": "random",
        "search_seed": search_seed,
        "budget": budget,
        "best_score": best_score,
        "best_config": best_config,
        "evaluations": results,
    }


def run_tpe_search(budget: int, search_seed: int, num_requests: int = 200,
                   rate: float = 200.0, blis_seed: int = 42) -> dict:
    results = []
    eval_counter = [0]

    def objective(trial: optuna.Trial) -> float:
        tp_ni_idx = trial.suggest_int("tp_ni_idx", 0, len(VALID_TP_INSTANCES) - 1)
        tp, ni = VALID_TP_INSTANCES[tp_ni_idx]
        config = {
            "tp": tp,
            "num_instances": ni,
            "scheduler": trial.suggest_categorical("scheduler", SCHEDULER_OPTIONS),
            "max_running": trial.suggest_categorical("max_running", MAX_RUNNING_OPTIONS),
            "max_tokens": trial.suggest_categorical("max_tokens", MAX_TOKENS_OPTIONS),
            "prefill_threshold": trial.suggest_categorical("prefill_threshold", PREFILL_THRESHOLD_OPTIONS),
            "block_size": trial.suggest_categorical("block_size", BLOCK_SIZE_OPTIONS),
            "routing": trial.suggest_categorical("routing", ROUTING_OPTIONS),
            "admission": trial.suggest_categorical("admission", ADMISSION_OPTIONS),
            "preemption": trial.suggest_categorical("preemption", PREEMPTION_OPTIONS),
        }

        out = run_blis(config, num_requests=num_requests, rate=rate, seed=blis_seed)
        if out is None:
            return 0.0
        score = out["score"]
        current_idx = eval_counter[0]
        eval_counter[0] += 1
        best_so_far = max((r["best_so_far"] for r in results), default=0.0)
        best_so_far = max(best_so_far, score)
        results.append({
            "eval_idx": current_idx,
            "score": score,
            "config": config,
            "wall_time_s": out["wall_time_s"],
            "best_so_far": best_so_far,
        })
        return score

    sampler = optuna.samplers.TPESampler(seed=search_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=budget, show_progress_bar=False)

    best_score = study.best_value if study.best_trial else 0.0
    best_params = study.best_params if study.best_trial else {}
    if "tp_ni_idx" in best_params:
        tp, ni = VALID_TP_INSTANCES[best_params["tp_ni_idx"]]
        best_config = {
            "tp": tp, "num_instances": ni,
            "scheduler": best_params.get("scheduler", "fcfs"),
            "max_running": best_params.get("max_running", 256),
            "max_tokens": best_params.get("max_tokens", 2048),
            "prefill_threshold": best_params.get("prefill_threshold", 0),
            "block_size": best_params.get("block_size", 16),
            "routing": best_params.get("routing", "round-robin"),
            "admission": best_params.get("admission", "always-admit"),
            "preemption": best_params.get("preemption", "fcfs"),
        }
    else:
        best_config = None

    return {
        "strategy": "tpe",
        "search_seed": search_seed,
        "budget": budget,
        "best_score": best_score,
        "best_config": best_config,
        "evaluations": results,
    }


def run_hierarchical_search(budget: int, search_seed: int, num_requests: int = 200,
                             rate: float = 200.0, blis_seed: int = 42) -> dict:
    """
    Two-phase hierarchical search.
    Phase 1: Evaluate 4 configs (one per TP level at max GPU utilization).
    Phase 2: TPE within the winning TP level (remaining budget).
    """
    evaluations = []
    best_score = -1.0
    best_config = None

    # --- Phase 1: evaluate one config per TP level ---
    phase1_results = []
    for tp, instances in PHASE1_TP_CONFIGS:
        if len(evaluations) >= budget:
            break
        config = make_phase1_config(tp, instances)
        out = run_blis(config, num_requests=num_requests, rate=rate, seed=blis_seed)
        if out is None:
            phase1_results.append({"tp": tp, "instances": instances, "score": 0.0})
            continue
        score = out["score"]
        best_so_far = max(best_score, score)
        evaluations.append({
            "eval_idx": len(evaluations),
            "phase": 1,
            "score": score,
            "config": config,
            "wall_time_s": out["wall_time_s"],
            "best_so_far": best_so_far,
        })
        if score > best_score:
            best_score = score
            best_config = config
        phase1_results.append({"tp": tp, "instances": instances, "score": score})

    # Identify winning TP level
    if not phase1_results or all(r["score"] <= 0 for r in phase1_results):
        winning_tp, winning_instances = 8, 1
    else:
        winner = max(phase1_results, key=lambda r: r["score"])
        winning_tp, winning_instances = winner["tp"], winner["instances"]

    # --- Phase 2: TPE within winning TP level ---
    remaining_budget = budget - len(evaluations)
    if remaining_budget <= 0:
        return {
            "strategy": "hierarchical",
            "search_seed": search_seed,
            "budget": budget,
            "best_score": best_score,
            "best_config": best_config,
            "phase1_winner": {"tp": winning_tp, "instances": winning_instances},
            "evaluations": evaluations,
        }

    phase2_results = []
    eval_counter = [0]

    def objective(trial: optuna.Trial) -> float:
        config = {
            "tp": winning_tp,
            "num_instances": winning_instances,
            "scheduler": trial.suggest_categorical("scheduler", SCHEDULER_OPTIONS),
            "max_running": trial.suggest_categorical("max_running", MAX_RUNNING_OPTIONS),
            "max_tokens": trial.suggest_categorical("max_tokens", MAX_TOKENS_OPTIONS),
            "prefill_threshold": trial.suggest_categorical("prefill_threshold", PREFILL_THRESHOLD_OPTIONS),
            "block_size": trial.suggest_categorical("block_size", BLOCK_SIZE_OPTIONS),
            "routing": trial.suggest_categorical("routing", ["round-robin"] if winning_instances == 1
                                                  else ROUTING_OPTIONS),
            "admission": trial.suggest_categorical("admission", ADMISSION_OPTIONS),
            "preemption": trial.suggest_categorical("preemption", PREEMPTION_OPTIONS),
        }

        out = run_blis(config, num_requests=num_requests, rate=rate, seed=blis_seed)
        if out is None:
            return 0.0
        score = out["score"]
        current_idx = eval_counter[0]
        eval_counter[0] += 1
        phase2_results.append({
            "phase2_idx": current_idx,
            "score": score,
            "config": config,
            "wall_time_s": out["wall_time_s"],
        })
        return score

    sampler = optuna.samplers.TPESampler(seed=search_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=remaining_budget, show_progress_bar=False)

    # Merge phase 2 results into evaluations, updating best_so_far
    for p2 in phase2_results:
        best_so_far = max(best_score, p2["score"])
        eval_entry = {
            "eval_idx": len(evaluations),
            "phase": 2,
            "score": p2["score"],
            "config": p2["config"],
            "wall_time_s": p2["wall_time_s"],
            "best_so_far": best_so_far,
        }
        evaluations.append(eval_entry)
        if p2["score"] > best_score:
            best_score = p2["score"]
            best_config = p2["config"]

    # Re-compute best_so_far as running max across all evals
    running_max = -1.0
    for ev in evaluations:
        running_max = max(running_max, ev["score"])
        ev["best_so_far"] = running_max

    return {
        "strategy": "hierarchical",
        "search_seed": search_seed,
        "budget": budget,
        "best_score": best_score,
        "best_config": best_config,
        "phase1_winner": {"tp": winning_tp, "instances": winning_instances},
        "evaluations": evaluations,
    }


def is_dominated(point: tuple, others: list[tuple]) -> bool:
    """Check if point is dominated by any other point (minimization on all objectives)."""
    for other in others:
        if all(o <= p for o, p in zip(other, point)) and any(o < p for o, p in zip(other, point)):
            return True
    return False


def pareto_front(points: list[tuple]) -> list[tuple]:
    """Extract non-dominated points (minimization on all objectives)."""
    front = []
    for i, p in enumerate(points):
        others = [points[j] for j in range(len(points)) if j != i]
        if not is_dominated(p, others):
            front.append(p)
    return front


def deduplicate_pareto(front: list[tuple], tolerance: float = 0.01) -> list[tuple]:
    """Deduplicate Pareto points within tolerance on all objectives."""
    deduped = []
    for p in front:
        is_dup = False
        for q in deduped:
            if all(abs(pi - qi) / (abs(qi) + 1e-10) <= tolerance for pi, qi in zip(p, q)):
                is_dup = True
                break
        if not is_dup:
            deduped.append(p)
    return deduped


def compute_scalarized(throughput: float, e2e_p99: float, ttft_p99: float) -> float:
    """Compute scalarized fitness using same weights as --fitness-weights."""
    REF_RPS = 100.0
    REF_TTFT = 1000.0
    REF_E2E = 1000.0
    w_tp = 0.4 / (1 + REF_RPS / max(throughput, 0.001))
    w_ttft = 0.3 * (1 - min(ttft_p99 / REF_TTFT, 1.0))
    w_e2e = 0.3 * (1 - min(e2e_p99 / REF_E2E, 1.0))
    return w_tp + w_ttft + w_e2e


def run_nsga2_search(budget: int, search_seed: int, num_requests: int = 200,
                     rate: float = 200.0, blis_seed: int = 42) -> dict:
    """
    NSGA-II multi-objective search with 3 objectives:
      maximize throughput (minimize -throughput),
      minimize e2e_p99,
      minimize ttft_p99.
    Returns Pareto front analysis.
    """
    all_evals = []
    eval_counter = [0]

    def objective(trial: optuna.Trial) -> tuple[float, float, float]:
        tp_ni_idx = trial.suggest_int("tp_ni_idx", 0, len(VALID_TP_INSTANCES) - 1)
        tp, ni = VALID_TP_INSTANCES[tp_ni_idx]
        config = {
            "tp": tp,
            "num_instances": ni,
            "scheduler": trial.suggest_categorical("scheduler", SCHEDULER_OPTIONS),
            "max_running": trial.suggest_categorical("max_running", MAX_RUNNING_OPTIONS),
            "max_tokens": trial.suggest_categorical("max_tokens", MAX_TOKENS_OPTIONS),
            "prefill_threshold": trial.suggest_categorical("prefill_threshold", PREFILL_THRESHOLD_OPTIONS),
            "block_size": trial.suggest_categorical("block_size", BLOCK_SIZE_OPTIONS),
            "routing": trial.suggest_categorical("routing", ROUTING_OPTIONS),
            "admission": trial.suggest_categorical("admission", ADMISSION_OPTIONS),
            "preemption": trial.suggest_categorical("preemption", PREEMPTION_OPTIONS),
        }

        out = run_blis(config, num_requests=num_requests, rate=rate, seed=blis_seed,
                       fitness_weights=None)
        if out is None:
            return (1e9, 1e9, 1e9)

        throughput = out["throughput"]
        e2e_p99 = out["e2e_p99"]
        ttft_p99 = out["ttft_p99"]
        scalarized = compute_scalarized(throughput, e2e_p99, ttft_p99)

        current_idx = eval_counter[0]
        eval_counter[0] += 1
        all_evals.append({
            "eval_idx": current_idx,
            "throughput": throughput,
            "e2e_p99": e2e_p99,
            "ttft_p99": ttft_p99,
            "scalarized_score": scalarized,
            "config": config,
            "wall_time_s": out["wall_time_s"],
        })

        # Objectives: minimize neg-throughput, e2e_p99, ttft_p99
        return (-throughput, e2e_p99, ttft_p99)

    sampler = optuna.samplers.NSGAIISampler(population_size=20, seed=search_seed)
    study = optuna.create_study(
        directions=["minimize", "minimize", "minimize"],
        sampler=sampler,
    )
    study.optimize(objective, n_trials=budget, show_progress_bar=False)

    # Extract Pareto front from all valid evaluations
    valid_evals = [e for e in all_evals if e["throughput"] > 0 and e["e2e_p99"] < 1e8]
    points = [(-e["throughput"], e["e2e_p99"], e["ttft_p99"]) for e in valid_evals]
    raw_front = pareto_front(points)
    deduped_front = deduplicate_pareto(raw_front, tolerance=0.01)
    deduped_5pct = deduplicate_pareto(raw_front, tolerance=0.05)

    # Extract Pareto front configs
    front_configs = []
    for fp in raw_front:
        for e in valid_evals:
            if (abs(-e["throughput"] - fp[0]) < 1e-6 and
                    abs(e["e2e_p99"] - fp[1]) < 1e-6):
                front_configs.append({
                    "config": e["config"],
                    "throughput": e["throughput"],
                    "e2e_p99": e["e2e_p99"],
                    "ttft_p99": e["ttft_p99"],
                    "scalarized_score": e["scalarized_score"],
                })
                break

    return {
        "strategy": "nsga2",
        "search_seed": search_seed,
        "budget": budget,
        "n_valid_evals": len(valid_evals),
        "raw_pareto_front_size": len(raw_front),
        "deduped_pareto_size_1pct": len(deduped_front),
        "deduped_pareto_size_5pct": len(deduped_5pct),
        "pareto_front_configs": front_configs,
        "evaluations": all_evals,
    }


def run_lhs_search(budget: int, search_seed: int, num_requests: int = 200,
                   rate: float = 200.0, blis_seed: int = 42) -> dict:
    dims = [
        VALID_TP_INSTANCES,
        SCHEDULER_OPTIONS,
        MAX_RUNNING_OPTIONS,
        MAX_TOKENS_OPTIONS,
        PREFILL_THRESHOLD_OPTIONS,
        BLOCK_SIZE_OPTIONS,
        ROUTING_OPTIONS,
        ADMISSION_OPTIONS,
        PREEMPTION_OPTIONS,
    ]
    sampler_lhs = qmc.LatinHypercube(d=9, seed=search_seed)
    samples = sampler_lhs.random(n=budget)

    results = []
    best_score = -1.0
    best_config = None

    for i, sample in enumerate(samples):
        indices = [int(sample[j] * len(dims[j])) % len(dims[j]) for j in range(9)]
        tp, ni = dims[0][indices[0]]
        config = {
            "tp": tp, "num_instances": ni,
            "scheduler": dims[1][indices[1]],
            "max_running": dims[2][indices[2]],
            "max_tokens": dims[3][indices[3]],
            "prefill_threshold": dims[4][indices[4]],
            "block_size": dims[5][indices[5]],
            "routing": dims[6][indices[6]],
            "admission": dims[7][indices[7]],
            "preemption": dims[8][indices[8]],
        }
        out = run_blis(config, num_requests=num_requests, rate=rate, seed=blis_seed)
        if out is None:
            continue
        score = out["score"]
        results.append({
            "eval_idx": i,
            "score": score,
            "config": config,
            "wall_time_s": out["wall_time_s"],
            "best_so_far": max(best_score, score),
        })
        if score > best_score:
            best_score = score
            best_config = config

    return {
        "strategy": "lhs",
        "search_seed": search_seed,
        "budget": budget,
        "best_score": best_score,
        "best_config": best_config,
        "evaluations": results,
    }


# ── Budget checkpoints ─────────────────────────────────────────────────────

def best_at_budget(evaluations: list[dict], budget: int) -> float:
    scores = [e.get("score", e.get("scalarized_score", 0.0)) for e in evaluations[:budget]]
    return max(scores) if scores else 0.0


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BLIS Configuration Search - Iteration 2")
    parser.add_argument("--strategy",
                        choices=["random", "tpe", "lhs", "hierarchical", "nsga2",
                                 "all", "h-main-strategies"],
                        default="all")
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--blis-seed", type=int, default=42)
    parser.add_argument("--num-requests", type=int, default=200)
    parser.add_argument("--rate", type=float, default=200.0)
    parser.add_argument("--model", default="qwen/qwen3-14b")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoints", default="10,20,40,100")
    args = parser.parse_args()

    # Set global MODEL so build_blis_cmd picks it up
    global MODEL
    MODEL = args.model

    seeds = [int(s) for s in args.seeds.split(",")]
    checkpoints = sorted(set(int(c) for c in args.checkpoints.split(",") if int(c) <= args.budget))
    if args.budget not in checkpoints:
        checkpoints.append(args.budget)

    # Strategy groups
    if args.strategy == "all":
        strategies = ["random", "tpe", "lhs", "hierarchical"]
    elif args.strategy == "h-main-strategies":
        strategies = ["random", "tpe", "hierarchical"]
    else:
        strategies = [args.strategy]

    mshort = model_shortname(args.model)
    global_best = GLOBAL_BEST_TABLE.get((mshort, int(args.rate), args.num_requests), None)

    all_results = {
        "config": {
            "budget": args.budget,
            "seeds": seeds,
            "blis_seed": args.blis_seed,
            "num_requests": args.num_requests,
            "rate": args.rate,
            "model": args.model,
            "global_best": global_best,
        },
        "runs": [],
    }

    t_global_start = time.perf_counter()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    for strategy in strategies:
        for seed in seeds:
            print(f"[{strategy}] seed={seed} budget={args.budget} rate={args.rate} model={mshort}", flush=True)
            t0 = time.perf_counter()

            if strategy == "random":
                run = run_random_search(args.budget, seed, args.num_requests, args.rate, args.blis_seed)
            elif strategy == "tpe":
                run = run_tpe_search(args.budget, seed, args.num_requests, args.rate, args.blis_seed)
            elif strategy == "lhs":
                run = run_lhs_search(args.budget, seed, args.num_requests, args.rate, args.blis_seed)
            elif strategy == "hierarchical":
                run = run_hierarchical_search(args.budget, seed, args.num_requests, args.rate, args.blis_seed)
            elif strategy == "nsga2":
                run = run_nsga2_search(args.budget, seed, args.num_requests, args.rate, args.blis_seed)
            else:
                continue

            wall = time.perf_counter() - t0
            run["wall_time_total_s"] = wall

            # Checkpoint best scores
            if strategy != "nsga2":
                run["checkpoint_best"] = {
                    str(c): best_at_budget(run["evaluations"], c)
                    for c in checkpoints
                }
                run["n_evaluations"] = len(run["evaluations"])

                if global_best is not None:
                    run["evals_to_best"] = compute_evals_to_best(run["evaluations"], global_best)
                    run["checkpoint_evals_to_best"] = {
                        str(c): compute_evals_to_best(run["evaluations"][:c], global_best)
                        for c in checkpoints
                    }

            print(f"  best={run.get('best_score', 'N/A (nsga2)')} "
                  f"evals={run.get('n_evaluations', run.get('n_valid_evals', '?'))} "
                  f"wall={wall:.1f}s", flush=True)

            all_results["runs"].append(run)

            # Flush after each run
            with open(args.output, "w") as f:
                json.dump(all_results, f, indent=2)

    all_results["total_wall_time_s"] = time.perf_counter() - t_global_start

    # Summary
    print("\n=== Summary ===")
    print(f"{'Strategy':<14} {'Seed':<6} {'Best/Front':<12} {'Evals':<8} "
          f"{'EvalsToBest':<14} {'Wall(s)':<10}")
    for run in all_results["runs"]:
        if run["strategy"] == "nsga2":
            best_str = f"Pareto={run['deduped_pareto_size_1pct']}"
            evals = run.get("n_valid_evals", "?")
            etb = "N/A"
        else:
            best_str = f"{run['best_score']:.6f}"
            evals = run.get("n_evaluations", "?")
            etb = str(run.get("evals_to_best", "?"))
        print(f"{run['strategy']:<14} {run['search_seed']:<6} {best_str:<12} "
              f"{str(evals):<8} {etb:<14} {run['wall_time_total_s']:<10.1f}")

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults written to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
