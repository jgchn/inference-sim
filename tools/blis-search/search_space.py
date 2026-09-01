"""Search space loading, validation, and hierarchical samplers.

Data-driven: reads all parameters, constraints, and conditional subsystems
from the YAML config. Supports two modes:

1. Legacy (defaults.yaml): parameters.tp + parameters.replicas with gpu_budget
2. Infrastructure-driven (full-stack.yaml): infrastructure block enumerates all
   valid topologies (tp × instances ≤ gpu_budget) including optional P/D splits.
"""
import itertools
import math
import random as random_module
from typing import Dict, List, Optional, Set

import yaml


def _validate_space(space: dict, yaml_path: str) -> None:
    """Reject search spaces whose topology semantics are ambiguous.

    There is no schema validation for these files (load_search_space is a bare
    yaml.safe_load), so a genuinely ambiguous declaration must fail loudly rather
    than be silently resolved: dp declared under BOTH `infrastructure` and
    `parameters` would have the per-parameter sample overwrite the topology's dp
    in HierarchicalRandomSampler._sample_base_params, silently voiding the GPU
    budget that generate_topologies just enforced.
    """
    infra = space.get("infrastructure") or {}
    params = space.get("parameters") or {}
    if "dp" in infra and "dp" in params:
        raise ValueError(
            f"{yaml_path}: dp is declared in both `infrastructure` and `parameters`. "
            "Pick one: `infrastructure.dp` makes dp part of the topology, so the GPU "
            "budget counts it (gpus_used = tp * dp * replicas); `parameters.dp` samples "
            "it independently of the budget. Declaring both lets the parameter sample "
            "overwrite the topology, silently breaking the budget bound."
        )


def load_search_space(yaml_path: str) -> dict:
    """Load and parse a search space YAML file."""
    with open(yaml_path) as f:
        space = yaml.safe_load(f)
    _validate_space(space, yaml_path)
    return space


def generate_topologies(infra: dict) -> List[dict]:
    """Enumerate all valid topologies from an infrastructure block.

    Each topology is a partial config dict containing tp, replicas, gpus_used,
    optionally dp, and optionally P/D pool fields.

    An instance occupies tp x dp GPUs (its "world"): tensor parallelism shards a
    rank, data parallelism adds ranks. So the budget bound is
    tp * dp * replicas <= gpu_budget and gpus_used is tp * dp * replicas.

    `dp` is an OPT-IN infrastructure axis. Declaring `dp: [...]` makes data
    parallelism part of the topology, so the GPU budget accounts for it and
    gpus_used is correct. Omitting it leaves dp out of the topology entirely
    (no `dp` key, world == tp), which keeps every pre-existing search space
    byte-identical: the enumeration order, the dicts, and gpus_used are unchanged.

    A space may still declare dp under `parameters` instead, which samples it
    INDEPENDENTLY of the topology -- meaning the GPU budget does NOT account for
    it and gpus_used understates the true GPU count by a factor of dp. That form
    is retained for backward compatibility but the infrastructure form is
    preferred whenever dp is a real deployment axis. Declaring dp in BOTH places
    is rejected by load_search_space (see _validate_space).
    """
    gpu_budget = infra["gpu_budget"]
    tp_values = infra.get("tp", [1, 2, 4, 8])
    min_instances = infra.get("min_instances", 1)
    explore_pd = infra.get("explore_pd", False)
    pd_bandwidth = infra.get("pd_transfer_bandwidth", 25.0)

    # dp absent => single implicit rank, and no `dp` key is emitted (BC).
    dp_declared = "dp" in infra
    dp_values = infra["dp"] if dp_declared else [1]

    topologies = []
    for tp in tp_values:
        for dp in dp_values:
            world = tp * dp
            if world > gpu_budget:
                continue
            max_instances = gpu_budget // world
            for n in range(min_instances, max_instances + 1):
                # Aggregated (no P/D)
                topo = {
                    "tp": tp,
                    "replicas": n,
                    "pd_decider": "never",
                    "gpus_used": world * n,
                }
                if dp_declared:
                    topo["dp"] = dp
                topologies.append(topo)
                # P/D splits (pure-disjoint only)
                if explore_pd and n >= 2:
                    for p in range(1, n):
                        pd_topo = {
                            "tp": tp,
                            "replicas": n,
                            "prefill_instances": p,
                            "decode_instances": n - p,
                            "pd_decider": "always",
                            "pd_transfer_bandwidth": pd_bandwidth,
                            "gpus_used": world * n,
                        }
                        if dp_declared:
                            pd_topo["dp"] = dp
                        topologies.append(pd_topo)
    return topologies


def expand_format_b(scorers: List[str], weights: List[int]) -> List[str]:
    """Expand Format B scorer definition into all profile strings."""
    profiles = []
    for r in range(1, len(scorers) + 1):
        for subset in itertools.combinations(scorers, r):
            for weight_combo in itertools.product(weights, repeat=r):
                profile = ",".join(f"{s}:{w}" for s, w in zip(subset, weight_combo))
                profiles.append(profile)
    return profiles


def get_scorer_profiles(space: dict) -> List[str]:
    """Get expanded scorer profiles from conditional_subsystems config."""
    cs = space.get("conditional_subsystems", {})
    scorers_cfg = cs.get("routing_scorers", {})
    fmt_b = scorers_cfg.get("format_b")
    if fmt_b:
        return expand_format_b(fmt_b["scorers"], fmt_b["weights"])
    explicit = scorers_cfg.get("profiles")
    if explicit:
        return explicit
    return []


def get_subsystem_profiles(subsystem_cfg: dict) -> List[str]:
    """Get expanded profiles for any subsystem (format_b or explicit profiles)."""
    fmt_b = subsystem_cfg.get("format_b")
    if fmt_b:
        return expand_format_b(fmt_b["scorers"], fmt_b["weights"])
    explicit = subsystem_cfg.get("profiles")
    if explicit:
        return explicit
    return []


def get_param_values(space: dict, param: str) -> List:
    """Get the list of valid values for a parameter."""
    params = space.get("parameters", {})
    p = params.get(param, {})
    return p.get("values", [])


def get_gpu_budget(space: dict) -> Optional[int]:
    """Get gpu_budget from search space (infrastructure or legacy)."""
    infra = space.get("infrastructure")
    if infra:
        return int(infra["gpu_budget"])
    params = space.get("parameters", {})
    replicas_cfg = params.get("replicas", {})
    budget = replicas_cfg.get("gpu_budget")
    return int(budget) if budget is not None else None


def get_max_replicas_by_tp(space: dict) -> Dict[int, int]:
    """Get the tp -> max_replicas mapping (legacy mode only)."""
    infra = space.get("infrastructure")
    if infra:
        gpu_budget = infra["gpu_budget"]
        tp_values = infra.get("tp", [1, 2, 4, 8])
        # An instance occupies tp x dp GPUs, so the replica ceiling is set by the
        # SMALLEST world a given tp can have (the most permissive dp). Reported per
        # tp because that is this helper's contract; the authoritative per-topology
        # bound is generate_topologies, which enumerates each (tp, dp) separately.
        min_dp = min(infra["dp"]) if "dp" in infra else 1
        return {int(tp): gpu_budget // (int(tp) * min_dp)
                for tp in tp_values if int(tp) * min_dp <= gpu_budget}

    params = space.get("parameters", {})
    replicas_cfg = params.get("replicas", {})
    tp_values = params.get("tp", {}).get("values", [1])

    gpu_budget = replicas_cfg.get("gpu_budget")
    if gpu_budget is not None:
        return {int(tp): int(gpu_budget) // int(tp) for tp in tp_values if int(tp) <= int(gpu_budget)}

    mapping = replicas_cfg.get("max_replicas_by_tp", {})
    return {int(k): int(v) for k, v in mapping.items()}


def get_min_replicas(space: dict) -> int:
    """Get minimum replicas from search space."""
    infra = space.get("infrastructure")
    if infra:
        return int(infra.get("min_instances", 1))
    params = space.get("parameters", {})
    replicas_cfg = params.get("replicas", {})
    return int(replicas_cfg.get("min_value", 1))


def get_param_flags(space: dict) -> Dict[str, str]:
    """Build param_name -> CLI flag mapping from space config."""
    flags = {}
    for name, cfg in space.get("parameters", {}).items():
        if "flag" in cfg:
            flags[name] = cfg["flag"]
    return flags


def get_all_param_flags(space: dict) -> Dict[str, str]:
    """Build complete param_name -> CLI flag mapping including subsystem params."""
    flags = get_param_flags(space)

    # Infrastructure-mode: tp and replicas need flags even though they're not in parameters
    infra = space.get("infrastructure")
    if infra:
        flags["tp"] = "--tp"
        flags["replicas"] = "--num-instances"
        # Only when dp is an infrastructure axis. A space that declares dp under
        # `parameters` already got the flag from get_param_flags above, and a space
        # with no dp at all must not gain a --dp flag (BC: an unflagged key is never
        # emitted, so the command line stays byte-identical).
        if "dp" in infra:
            flags["dp"] = "--dp"
        flags["pd_decider"] = "--pd-decider"
        flags["prefill_instances"] = "--prefill-instances"
        flags["decode_instances"] = "--decode-instances"
        flags["pd_transfer_bandwidth"] = "--pd-transfer-bandwidth"

    cs = space.get("conditional_subsystems", {})
    for sub_name, sub_cfg in cs.items():
        if "flag" in sub_cfg:
            flags[sub_name] = sub_cfg["flag"]
        for nested_name, nested_cfg in sub_cfg.get("parameters", {}).items():
            if isinstance(nested_cfg, dict) and "flag" in nested_cfg:
                flags[nested_name] = nested_cfg["flag"]
        if "flags" in sub_cfg:
            for nested_name, nested_cfg in sub_cfg["flags"].items():
                if isinstance(nested_cfg, dict) and "flag" in nested_cfg:
                    flags[nested_name] = nested_cfg["flag"]
        if "fixed" in sub_cfg:
            for nested_name, nested_cfg in sub_cfg["fixed"].items():
                if isinstance(nested_cfg, dict) and "flag" in nested_cfg:
                    flags[nested_name] = nested_cfg["flag"]
    return flags


def eval_condition(condition: str, cfg: dict) -> bool:
    """Evaluate a condition against a config dict."""
    if " -> " in condition:
        parts = condition.split(" -> ", 1)
        condition = f"not ({parts[0]}) or ({parts[1]})"

    safe_ns = {
        "true": True,
        "false": False,
        "True": True,
        "False": False,
        "__builtins__": {},
    }
    safe_ns.update(cfg)
    try:
        return bool(eval(condition, safe_ns))
    except Exception:
        return False


def eval_range_expr(expr, cfg: dict) -> int:
    """Evaluate a range expression (e.g., 'floor(replicas / 2)')."""
    if isinstance(expr, (int, float)):
        return int(expr)
    safe_ns = {"floor": math.floor, "ceil": math.ceil, "min": min, "max": max, "__builtins__": {}}
    safe_ns.update(cfg)
    try:
        return int(eval(expr, safe_ns))
    except Exception:
        return 0


def is_valid_config(cfg: dict, space: dict) -> bool:
    """Validate a config against the search space constraints."""
    infra = space.get("infrastructure")
    if infra:
        # Infrastructure mode: topology is pre-validated by generate_topologies
        # Only check non-topology constraints
        for constraint in space.get("constraints", []):
            if not eval_condition(constraint, cfg):
                return False
        return True

    # Legacy mode
    tp = cfg.get("tp", 1)
    replicas = cfg.get("replicas", 1)
    long_prefill = cfg.get("long_prefill", 0)
    max_scheduled_tokens = cfg.get("max_scheduled_tokens", 8192)

    max_replicas = get_max_replicas_by_tp(space)
    min_replicas = get_min_replicas(space)
    gpu_budget = get_gpu_budget(space)

    if gpu_budget is not None:
        if tp * replicas > gpu_budget:
            return False
    else:
        if tp * replicas > 8:
            return False
    if replicas < min_replicas or replicas > max_replicas.get(tp, 1):
        return False
    if long_prefill > 0 and long_prefill >= max_scheduled_tokens:
        return False

    for constraint in space.get("constraints", []):
        if not eval_condition(constraint, cfg):
            return False

    return True


def is_scorer_eligible(cfg: dict, space: dict) -> bool:
    """Check if a config is eligible for scorer profile expansion."""
    cs = space.get("conditional_subsystems", {})
    scorers_cfg = cs.get("routing_scorers", {})
    condition = scorers_cfg.get("enabled_when", "routing_policy == 'weighted' and replicas > 1")
    return eval_condition(condition, cfg)


class HierarchicalRandomSampler:
    """Data-driven hierarchical sampler.

    In infrastructure mode: picks a random topology then samples policy params.
    In legacy mode: samples tp/replicas/params respecting ranges and constraints.
    """

    def __init__(self, rng: random_module.Random, space: dict):
        self.rng = rng
        self.space = space
        self.params = space.get("parameters", {})
        self.scorer_profiles = get_scorer_profiles(space)

        # Infrastructure mode: pre-generate topologies
        self._infra = space.get("infrastructure")
        if self._infra:
            self._topologies = generate_topologies(self._infra)
        else:
            self._topologies = None
            self.tp_values = get_param_values(space, "tp")
            self.max_replicas = get_max_replicas_by_tp(space)
            self.min_replicas = get_min_replicas(space)

        self._subsystem_profiles = {}
        cs = space.get("conditional_subsystems", {})
        for sub_name, sub_cfg in cs.items():
            profiles = get_subsystem_profiles(sub_cfg)
            if profiles:
                self._subsystem_profiles[sub_name] = profiles
            for nested_name in ("prefill_routing_scorers", "decode_routing_scorers"):
                nested = sub_cfg.get(nested_name)
                if nested and isinstance(nested, dict):
                    nested_profiles = get_subsystem_profiles(nested)
                    if nested_profiles:
                        self._subsystem_profiles[nested_name] = nested_profiles

    def _sample_param(self, name: str, cfg: dict) -> object:
        """Sample a single parameter value based on its YAML definition."""
        p = self.params.get(name, {})

        if name == "replicas" and not self._infra:
            tp = cfg.get("tp", 1)
            max_r = self.max_replicas.get(tp, 1)
            min_r = self.min_replicas
            if min_r > max_r:
                return None
            return self.rng.randint(min_r, max_r)

        if p.get("type") == "boolean":
            prob = p.get("probability", 0.5)
            return self.rng.random() < prob

        values = p.get("values", [])
        if values:
            return self.rng.choice(values)

        return None

    def _sample_base_params(self) -> dict:
        """Sample all top-level parameters."""
        cfg = {}

        if self._infra:
            # Infrastructure mode: pick a random topology
            topo = self.rng.choice(self._topologies)
            cfg.update(topo)
        else:
            # Legacy mode: sample tp then replicas
            cfg["tp"] = self.rng.choice(self.tp_values)
            cfg["replicas"] = self._sample_param("replicas", cfg)
            if cfg["replicas"] is None:
                return {}

        # Sample all policy parameters
        for name in self.params:
            if name in ("tp", "replicas"):
                continue
            val = self._sample_param(name, cfg)
            if val is not None:
                cfg[name] = val

        return cfg

    def _apply_single_instance_overrides(self, cfg: dict):
        """Force routing/admission to defaults for single-instance configs."""
        if cfg.get("replicas", 1) == 1:
            cfg["routing_policy"] = "round-robin"
            cfg["admission_policy"] = "always-admit"
            if self._infra:
                cfg["pd_decider"] = "never"
                cfg.pop("prefill_instances", None)
                cfg.pop("decode_instances", None)
                cfg.pop("pd_transfer_bandwidth", None)

    def _sample_conditional_subsystems(self, cfg: dict):
        """Sample conditional subsystem parameters based on enabled_when conditions."""
        cs = self.space.get("conditional_subsystems", {})

        # Routing scorers
        scorers_cfg = cs.get("routing_scorers", {})
        condition = scorers_cfg.get("enabled_when", "")
        if condition and eval_condition(condition, cfg) and self.scorer_profiles:
            cfg["scorer_profile"] = self.rng.choice(self.scorer_profiles)
        else:
            cfg["scorer_profile"] = None

        # Flow control
        fc_cfg = cs.get("flow_control", {})
        if fc_cfg:
            fc_condition = fc_cfg.get("enabled_when", "")
            if fc_condition and eval_condition(fc_condition, cfg):
                prob = fc_cfg.get("probability", 0.5)
                cfg["flow_control"] = self.rng.random() < prob
            else:
                cfg["flow_control"] = False

            if cfg.get("flow_control"):
                self._sample_flow_control(cfg, fc_cfg)
            else:
                for param_name in fc_cfg.get("parameters", {}):
                    cfg[param_name] = None

    def _sample_flow_control(self, cfg: dict, fc_cfg: dict):
        """Sample flow control sub-parameters."""
        for param_name, param_cfg in fc_cfg.get("parameters", {}).items():
            if isinstance(param_cfg, dict):
                values = param_cfg.get("values", [])
                if values:
                    cfg[param_name] = self.rng.choice(values)

    def sample(self) -> Optional[dict]:
        for _ in range(200):
            cfg = self._sample_base_params()
            if not cfg:
                continue

            self._apply_single_instance_overrides(cfg)
            self._sample_conditional_subsystems(cfg)

            if not is_valid_config(cfg, self.space):
                continue
            return cfg
        return None


class DiversityBiasedSampler:
    """Post-convergence diversity sampler.

    In infrastructure mode: targets under-represented GPU-count tiers.
    In legacy mode: targets under-represented TP classes (ascending order).
    """

    def __init__(
        self,
        rng: random_module.Random,
        space: dict,
        initial_pareto_classes: Set[int],
        verbose: bool = True,
    ):
        self.rng = rng
        self.space = space
        self.random_sampler = HierarchicalRandomSampler(rng, space)
        self._infra = space.get("infrastructure")

        safe = space.get("diversity_safe", {})
        self.safe_values = safe

        if self._infra:
            # Infrastructure mode: diversity over GPU-count tiers
            self._topologies = generate_topologies(self._infra)
            all_gpu_counts = sorted(set(t["gpus_used"] for t in self._topologies))
            self.missing_classes = sorted(
                gc for gc in all_gpu_counts if gc not in initial_pareto_classes
            )
        else:
            # Legacy mode: diversity over TP classes
            self.tp_values = get_param_values(space, "tp")
            self.max_replicas = get_max_replicas_by_tp(space)
            self.min_replicas = get_min_replicas(space)
            self.missing_classes = sorted(
                tp for tp in self.tp_values if tp not in initial_pareto_classes
            )

        self.current_target_idx = 0
        self.all_found = len(self.missing_classes) == 0

        if verbose:
            label = "GPU tiers" if self._infra else "TP classes"
            print(
                f"    [diversity] Missing {label} to target: {self.missing_classes}",
                flush=True,
            )

    def update_pareto(self, current_pareto_classes: Set[int]):
        if self.all_found:
            return
        while self.current_target_idx < len(self.missing_classes):
            target = self.missing_classes[self.current_target_idx]
            if target in current_pareto_classes:
                self.current_target_idx += 1
            else:
                break
        if self.current_target_idx >= len(self.missing_classes):
            self.all_found = True

    def sample(self) -> Optional[dict]:
        if self.all_found:
            return self.random_sampler.sample()

        target_class = self.missing_classes[self.current_target_idx]

        for _ in range(200):
            if self._infra:
                # Pick a topology matching the target GPU count
                matching = [t for t in self._topologies if t["gpus_used"] == target_class]
                if not matching:
                    return self.random_sampler.sample()
                cfg = dict(self.rng.choice(matching))
            else:
                # Legacy: pick target TP with max replicas
                target_tp = target_class
                max_r = self.max_replicas.get(target_tp, 1)
                min_r = self.min_replicas
                if min_r > max_r:
                    return self.random_sampler.sample()
                cfg = {"tp": target_tp, "replicas": max_r}

            # Sample policy params, preferring safe values
            for name, p_cfg in self.space.get("parameters", {}).items():
                if name in ("tp", "replicas"):
                    continue
                safe_vals = self.safe_values.get(name)
                if safe_vals:
                    cfg[name] = self.rng.choice(safe_vals)
                elif p_cfg.get("type") == "boolean":
                    prob = p_cfg.get("probability", 0.5)
                    cfg[name] = self.rng.random() < prob
                elif "values" in p_cfg:
                    cfg[name] = self.rng.choice(p_cfg["values"])

            self.random_sampler._apply_single_instance_overrides(cfg)
            self.random_sampler._sample_conditional_subsystems(cfg)

            if not is_valid_config(cfg, self.space):
                continue
            return cfg
        return None
