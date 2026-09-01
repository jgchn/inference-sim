"""BLIS subprocess invocation and parallel evaluation."""
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional

from search_space import get_all_param_flags


# Parameters that are topology-derived (handled specially in command building)
_TOPOLOGY_PARAMS = {"tp", "replicas", "pd_decider", "prefill_instances",
                    "decode_instances", "pd_transfer_bandwidth", "gpus_used"}

# Flow control sub-params (skipped when FC is off)
_FC_PARAMS = {"saturation_detector", "queue_depth_threshold", "kv_util_threshold",
              "dispatch_order", "max_gateway_queue_depth", "request_ttl"}


def build_command(
    cfg: dict,
    metrics_path: str,
    blis_binary: str,
    fixed_params: Dict[str, str],
    param_flags: Dict[str, str],
) -> List[str]:
    """Build the BLIS CLI command from a config dict."""
    cmd = [blis_binary, "run"]
    for flag, val in fixed_params.items():
        cmd.extend([flag, str(val)])

    pd_active = cfg.get("pd_decider", "never") != "never"
    fc_active = cfg.get("flow_control") is True

    # Emit topology flags
    if "tp" in cfg:
        cmd.extend(["--tp", str(cfg["tp"])])
    if "replicas" in cfg:
        cmd.extend(["--num-instances", str(cfg["replicas"])])

    # P/D flags (only when active)
    if pd_active:
        cmd.extend(["--pd-decider", str(cfg.get("pd_decider", "always"))])
        pi = cfg.get("prefill_instances", 0)
        di = cfg.get("decode_instances", 0)
        if pi > 0:
            cmd.extend(["--prefill-instances", str(pi)])
        if di > 0:
            cmd.extend(["--decode-instances", str(di)])
        bw = cfg.get("pd_transfer_bandwidth")
        if bw:
            cmd.extend(["--pd-transfer-bandwidth", str(bw)])

    # Emit policy parameters (skip topology and FC sub-params when inactive)
    for param_name, flag in param_flags.items():
        if param_name in _TOPOLOGY_PARAMS:
            continue
        val = cfg.get(param_name)
        if val is None:
            continue
        if val is False:
            continue
        if param_name in _FC_PARAMS and not fc_active:
            continue
        if val is True:
            cmd.append(flag)
        else:
            cmd.extend([flag, str(val)])

    # Scorer profiles (global routing)
    if cfg.get("scorer_profile") and cfg.get("routing_policy") == "weighted":
        cmd.extend(["--routing-scorers", cfg["scorer_profile"]])

    # Flow control requires saturation detector
    if fc_active:
        detector = cfg.get("saturation_detector", "utilization")
        if "--saturation-detector" not in cmd:
            cmd.extend(["--saturation-detector", detector])

    cmd.extend(["--metrics-path", metrics_path])
    return cmd


def evaluate_config(
    cfg: dict,
    blis_binary: str,
    fixed_params: Dict[str, str],
    param_flags: Dict[str, str],
    metric_keys: List[str],
    timeout: float = 20.0,
) -> Optional[dict]:
    """Evaluate a single BLIS config. Returns result dict or None on failure."""
    tmpdir = os.environ.get("TMPDIR", "/tmp")
    tmp_dir = tempfile.mkdtemp(dir=tmpdir)
    metrics_path = os.path.join(tmp_dir, "metrics.json")
    try:
        cmd = build_command(cfg, metrics_path, blis_binary, fixed_params, param_flags)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0 or not os.path.exists(metrics_path):
            return None
        with open(metrics_path) as f:
            metrics = json.load(f)
        extracted = {}
        for key in metric_keys:
            if key == "gpus_used":
                continue
            val = metrics.get(key)
            if val is None:
                return None
            extracted[key] = float(val)

        # Record every other numeric metric BLIS reported, so results carry the
        # full metrics set (e2e/itl percentiles, token counts, preemptions, ...)
        # and not just the search objectives. Skipped: the per-request `requests`
        # array (bulk, not a metric) and non-numeric metadata like `instance_id`.
        for key, val in metrics.items():
            if key in extracted or key == "requests":
                continue
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                continue
            extracted[key] = val

        # Inject gpus_used from config (synthetic, not from blis output)
        if "gpus_used" in cfg:
            extracted["gpus_used"] = cfg["gpus_used"]

        return {"config": cfg, "metrics": extracted}
    except Exception:
        return None
    finally:
        try:
            if os.path.exists(metrics_path):
                os.remove(metrics_path)
            os.rmdir(tmp_dir)
        except Exception:
            pass


def evaluate_batch(
    configs: List[dict],
    workers: int,
    blis_binary: str,
    fixed_params: Dict[str, str],
    param_flags: Dict[str, str],
    metric_keys: List[str],
    timeout: float = 20.0,
) -> List[Optional[dict]]:
    """Evaluate a batch of configs in parallel. Returns results in order."""
    results = [None] * len(configs)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                evaluate_config, cfg, blis_binary, fixed_params, param_flags, metric_keys, timeout
            ): i
            for i, cfg in enumerate(configs)
        }
        for future in as_completed(futures):
            i = futures[future]
            try:
                results[i] = future.result()
            except Exception:
                results[i] = None
    return results
