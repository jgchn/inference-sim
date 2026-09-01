"""Deploy-command construction for the picks table (design §7).

The only module in viz/ that imports code from outside viz/. Knob names do not
map mechanically to CLI flags — `replicas` is `--num-instances`, `flow_control`
is a bare boolean, flow-control sub-params are suppressed when FC is off — so the
flag map is either carried by the results file, reconstructed from the space
YAML, or absent; and absent means no command rather than a guessed one.
"""
import os
import shlex
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# evaluator.py and search_space.py use flat imports (`from search_space import
# ...`), exactly as search.py does, so tools/blis-search/ must be on sys.path
# before they can be imported (D6).
_VIZ_DIR = os.path.dirname(os.path.abspath(__file__))
_SEARCH_DIR = os.path.dirname(_VIZ_DIR)
if _SEARCH_DIR not in sys.path:
    sys.path.insert(0, _SEARCH_DIR)

from evaluator import build_command                                # noqa: E402
from search_space import get_all_param_flags, load_search_space    # noqa: E402

from data import Dataset                                          # noqa: E402

#: Printed in place of a command when the flag map cannot be established (§7 tier 3).
NO_FLAGS_NOTE = "re-run search or pass --space to generate commands"

#: Placeholder handed to build_command, whose trailing --metrics-path pair is then
#: dropped: a deploy command should not write a metrics file.
_METRICS_SENTINEL = "/dev/null"


class CommandError(RuntimeError):
    """A --space that cannot be read or parsed. The CLI turns this into a
    one-line exit rather than a traceback (§9: never a silent crash)."""


@dataclass
class FlagSource:
    """Where the knob -> flag map came from, and what to say when it didn't."""
    flags: Optional[Dict[str, str]]
    origin: str          # "results-file" | "space-file" | "unavailable"
    note: str            # "" unless origin == "unavailable"
    warning: str = ""    # non-empty when a --space was supplied but not needed


def resolve_param_flags(ds: Dataset, space_path: Optional[str] = None) -> FlagSource:
    """§7 precedence: results file, then --space, then nothing — never a guess.

    A results file that already carries param_flags wins over an explicit --space
    (D14): the embedded map describes the search that actually ran, while the YAML
    on disk may have drifted since. Ignoring the flag is reported, not silent.
    """
    if ds.param_flags:
        warning = ""
        if space_path:
            warning = (f"--space {space_path} ignored: {ds.source_name} already "
                       "carries param_flags")
        return FlagSource(dict(ds.param_flags), "results-file", "", warning)
    if space_path:
        try:
            space = load_search_space(space_path)
        except Exception as exc:                       # noqa: BLE001 — re-raised
            raise CommandError(f"--space {space_path}: {exc}") from exc
        return FlagSource(dict(get_all_param_flags(space)), "space-file", "")
    return FlagSource(None, "unavailable", NO_FLAGS_NOTE)


def fixed_params(ds: Dataset) -> Dict[str, str]:
    """Rebuild search.py's fixed_params from `spec`, which stored it with the
    leading dashes stripped (`--num-requests` -> `num-requests`)."""
    return {"--" + str(k): str(v) for k, v in sorted(ds.context.items())}


def deploy_command(
    ds: Dataset, idx: int, flags: Optional[Dict[str, str]],
    blis_binary: str = "./blis",
) -> Optional[List[str]]:
    """The exact blis invocation for one point, or None without a flag map.

    The subcommand follows the space: a space that injects a trace declares
    --trace-header, and those flags exist only on `blis replay` (`blis run` has no
    such flag and exits with "unknown flag"). search.py detects trace mode the same
    way and drops the run-only --workload/--rate/--num-requests from fixed_params,
    so the remaining command is valid `replay` as-is. Without this the printed
    command was unpastable for every trace-driven search.
    """
    if not flags:
        return None
    # build_command emits flags in the map's iteration order. Sorting makes the
    # printed command a function of the config alone, so it cannot change because
    # a space file reordered its knobs (D15).
    ordered = {k: flags[k] for k in sorted(flags)}
    cmd = build_command(dict(ds.points[idx].config), _METRICS_SENTINEL,
                        blis_binary, fixed_params(ds), ordered)
    # build_command always appends the metrics path last; a deploy command has no
    # business writing one. Check rather than trim blindly, so an upstream change
    # to build_command fails loudly here instead of shipping a command that writes
    # a metrics file the user never asked for. Not an assert: `python -O` strips
    # asserts, and this guard must survive that.
    if cmd[-2:] != ["--metrics-path", _METRICS_SENTINEL]:
        raise CommandError(
            "evaluator.build_command no longer ends with the metrics path "
            "(got %r); refusing to emit a command that may write one" % (cmd[-2:],))
    cmd = cmd[:-2]
    # build_command hardcodes the `run` subcommand. Rewrite it for a trace-driven
    # space rather than trimming blindly, and only when the marker flag is actually
    # present, so a workload-driven space is untouched.
    if "--trace-header" in ordered.values() and len(cmd) > 1 and cmd[1] == "run":
        cmd[1] = "replay"
    return cmd


def command_text(cmd: Optional[List[str]]) -> str:
    """One shell-safe line, or the honest note when there is no command."""
    if cmd is None:
        return NO_FLAGS_NOTE
    return " ".join(shlex.quote(a) for a in cmd)
