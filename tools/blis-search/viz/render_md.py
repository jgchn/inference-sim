"""picks.md — the copy-pasteable output (design §3).

Deliberately not a PNG: a picture of a table is neither greppable nor
copy-pasteable, and the commands are the most actionable thing the tool
produces. Every number and every sentence here comes from the Analysis bundle,
so picks.md, the PNGs, and the HTML cannot disagree.
"""
from typing import List

import commands
from analysis import (Analysis, context_line, describe_config, fmt, gpu_count,
                      gpu_source, page_banners, provenance_bits, strip_cells,
                      summary_bits)

#: Glyphs for a knob strip, from empty to full.
_GLYPHS = "░▒▓█"


def _glyph_strip(strip) -> str:
    cells = strip_cells(strip)
    return "".join(_GLYPHS[min(len(_GLYPHS) - 1, int(c * len(_GLYPHS)))] for c in cells)


def document(an: Analysis, flag_source) -> str:
    """The whole picks.md, as one string. No I/O, so it is trivially testable."""
    out: List[str] = [f"# BLIS search picks — {context_line(an.ds)}", "",
                      " · ".join(summary_bits(an.ds)), ""]
    for banner in page_banners(an, flag_source):
        out.append(f"> **{banner}**")
        out.append("")
    out.extend(_picks_section(an, flag_source))
    out.extend(_knobs_section(an))
    out.extend(_provenance(an))
    return "\n".join(out) + "\n"


def _picks_section(an: Analysis, flag_source) -> List[str]:
    ds = an.ds
    out = [f"## Picks ({an.picks.header})", ""]
    if not an.picks.rows:
        out += ["No configs to pick from.", ""]
        return out

    has_gpu = bool(gpu_source(ds))
    head = ["Pick", "Config"] + [f"{o.key} {o.arrow}".strip() for o in ds.objectives]
    if has_gpu:
        head.append("GPUs")
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "|".join(["---"] * len(head)) + "|")
    for row in an.picks.rows:
        cells = [", ".join(row.titles), describe_config(ds, row.index)]
        cells += [fmt(ds.value(ds.points[row.index], o)) for o in ds.objectives]
        if has_gpu:
            cells.append(str(gpu_count(ds, row.index)))
        out.append("| " + " | ".join(cells) + " |")
    out.append("")

    for note in an.picks.notes:
        out.append(f"- {note}")
    if an.picks.notes:
        out.append("")

    for row in an.picks.rows:
        out.append(f"### {', '.join(row.titles)} — {describe_config(ds, row.index)}")
        out.append("")
        cmd = commands.deploy_command(ds, row.index, flag_source.flags)
        if cmd is None:
            out.append(commands.NO_FLAGS_NOTE)
        else:
            out.append("```bash")
            out.append(commands.command_text(cmd))
            out.append("```")
        out.append("")
    return out


def _knobs_section(an: Analysis) -> List[str]:
    out = ["## Which knobs the front exploits", ""]
    if not an.strips:
        out += ["This file records no configuration knobs.", ""]
        return out
    if not an.ds.has_cloud:
        out += ["No dominated cloud to compare against, so these are ranges the "
                "front occupies, not attributions.", ""]
    out.append("| Knob | Front | Separation | What the front does |")
    out.append("|---|---|---|---|")
    for s in an.strips:
        out.append(f"| {s.name} | `{_glyph_strip(s)}` | {s.score:.2f} | {s.headline} |")
    out.append("")
    return out


def _provenance(an: Analysis) -> List[str]:
    """analysis.py decides what the strip says; this decides that it is a
    markdown rule followed by one middot-joined line."""
    return ["---", "", " · ".join(provenance_bits(an)), ""]
