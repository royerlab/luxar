"""A demo must not throw away the provenance of the fit it ships.

``GSplatData.save(include_fitting_info=False)`` does not merely relocate the fit
stats -- ``save_gsplats.split_fitting_info`` leaves ``fitting_info`` as ``None``,
so the ``_FITTING_INFO_KEYS`` whitelist never runs, and because those same keys
are excluded from ``pipeline_info`` they are dropped from the store ENTIRELY.

For the demos this matters more than it looks: several of them save straight to
the file that is then SHIPPED (the cache path and the packaged artifact carry the
same name), so a suppressed save bakes into the next artifact regenerated from a
demo run -- no fitter, no splat count, no runtime, no PSNR, no record of what the
final cull removed. ``fitting/`` is also one of the three groups the loader reads
back into ``stats``, so a figure that was never written cannot be quoted later.
The whitelist is where source-grid stamps land as well, so a suppressing demo
forfeits any compression figure the fitter learns to record.

The one legitimate use is a store that never came from a fit at all, which is
why the exemption below is keyed to a stated reason AND to a call-site count
rather than merely allowed: clearance for one sentinel save is not clearance for
the file it happens to live in.

Scope, stated so it is not mistaken for more: this guard sees the EXPLICIT
suppression only (in either spelling -- the keyword, or the flag's positional
slot). ``write_gsplats_tree`` takes ``fitting_info`` as a keyword that defaults
to ``None``, so a tree write drops the same record by saying nothing at all --
there is no argument for the scan to find. Two demos write trees that way.
``_interop_common.py`` hands its writer imported splats that were never fitted,
so nothing in the whitelist exists to lose. ``demo_gsplats_recipes_tribolium.py``
normally builds on a precomputed base loaded with ``include_stats=False``, but
under ``--recompute`` it re-fits and does hold the fit's stats, so that path
drops them by omission. Whether a caller holding fit stats should be made to
pass them through is the wider question #1600 asks of every in-out rewrite, not
something a demo lint can settle -- so it is out of this guard's reach by
decision, not by oversight.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path

from ._scanned_modules import DEMOS_DIR, scanned_demo_modules

#: The demos PLUS the shared helpers they delegate to — the same set the other
#: demo guards scan. Not a ``demo_*.py`` glob: ``_interop_common.py`` already
#: saves gsplat stores, so an allowlist keyed on the file name would let a
#: suppressing save escape simply by living in a helper.
SCANNED_PATHS = scanned_demo_modules()

#: Call sites allowed to pass ``include_fitting_info=False``: file name ->
#: (number of allowed call sites, why there is no fit to record).
#:
#: NOT a general opt-out list: each entry must describe a store built WITHOUT
#: fitting, where there is no provenance to keep. A real fit belongs nowhere
#: near this mapping. The count is part of the exemption on purpose — a file
#: that legitimately suppresses one sentinel save must not thereby become free
#: to suppress a second, real one.
_NO_FIT_TO_RECORD: dict[str, tuple[int, str]] = {
    "demo_gsplats_4d_nexrad_supercell.py": (
        1,
        "the empty-frame sentinel -- a synthetic single-splat placeholder for a "
        "frame with nothing above the dBZ floor, which the fitter never saw",
    ),
}


#: ``GSplatData.save(path, ordering, encoding_mode, include_fitting_info, ...)``
#: — the flag is the FOURTH positional parameter and is not keyword-only, so a
#: bare ``False`` in that slot suppresses exactly as well as the keyword does.
#: Every call site spells it out today; the scan covers both forms so that stays
#: a style choice rather than the way past this gate.
_SAVE_POSITIONAL_INDEX = 3


def _suppresses(node: ast.Call) -> bool:
    """Does this call pass ``include_fitting_info=False``, spelled either way?"""
    for kw in node.keywords:
        if kw.arg == "include_fitting_info":
            return isinstance(kw.value, ast.Constant) and kw.value.value is False
    if (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "save"
        and len(node.args) > _SAVE_POSITIONAL_INDEX
    ):
        positional = node.args[_SAVE_POSITIONAL_INDEX]
        return isinstance(positional, ast.Constant) and positional.value is False
    return False


def _suppressing_modules(paths: Iterable[Path] = SCANNED_PATHS) -> dict[str, int]:
    """Map module file name -> count of ``include_fitting_info=False`` call sites."""
    found: dict[str, int] = {}
    for path in paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _suppresses(node):
                found[path.name] = found.get(path.name, 0) + 1
    return found


def test_no_demo_discards_the_provenance_of_a_real_fit() -> None:
    """Only the documented no-fit stores may suppress fitting info."""
    suppressing = _suppressing_modules()
    expected = {name: count for name, (count, _) in _NO_FIT_TO_RECORD.items()}
    assert suppressing == expected, (
        "include_fitting_info=False changed. A demo that saves a REAL fit must "
        "keep its provenance -- the flag drops the whitelisted fit stats "
        "entirely (they are excluded from pipeline_info too), so the shipped "
        "dataset states neither what produced it nor how well it did. Counts "
        "are compared, not just file names: an exempt file gets exactly the "
        "documented number of suppressing call sites, no more.\n"
        f"  suppressing now: {sorted(suppressing.items())}\n"
        f"  documented no-fit stores: {sorted(expected.items())}"
    )


def test_every_exemption_states_why_there_is_no_fit() -> None:
    """A bare exemption is how this list would rot into a general opt-out."""
    for name, (count, reason) in _NO_FIT_TO_RECORD.items():
        assert (DEMOS_DIR / name).exists(), (
            f"{name} is exempted but no longer exists -- drop the entry"
        )
        assert count >= 1, f"{name}: an exemption for zero call sites is dead"
        assert len(reason) > 40, f"{name}: give the reason the fit is absent"


def test_the_detector_sees_a_planted_suppression(tmp_path: Path) -> None:
    """Guard against the scan quietly matching nothing (a vacuous gate).

    Runs the real detector over a planted module, so it also pins that a second
    suppressing call site in one file counts as two rather than collapsing into
    the first, and that the positional spelling of the flag counts as well.
    """
    planted = tmp_path / "demo_planted.py"
    planted.write_text(
        "result.save(path, include_fitting_info=False)\n"
        "result.save(other, include_fitting_info=True)\n"
        "result.save(third, include_fitting_info=False)\n"
        # The flag's positional slot -- suppression without the keyword.
        "result.save(fourth, 'hilbert', mode, False)\n"
        "result.save(fifth, 'hilbert', mode, True)\n"
    )
    assert _suppressing_modules([planted]) == {"demo_planted.py": 3}
    assert SCANNED_PATHS, "no demo modules were scanned at all"


def test_suppression_really_drops_the_keys_rather_than_moving_them() -> None:
    """Pin the reason this gate exists, not just the flag's current value."""
    from luxar.gsplats.io.save_gsplats import _FITTING_INFO_KEYS, split_fitting_info

    stats = {k: 1 for k in list(_FITTING_INFO_KEYS)[:3]}
    kept = split_fitting_info(stats, include_fitting_info=True)
    dropped = split_fitting_info(stats, include_fitting_info=False)

    assert kept[0], "fitting info should survive when the flag is on"
    assert not dropped[0], "fitting info should be absent when the flag is off"
    # The point of the gate: they do NOT reappear in the pipeline group.
    pipeline_when_dropped = dropped[3] or {}
    assert not (set(stats) & set(pipeline_when_dropped)), (
        "if suppressed fit keys ever fell through to pipeline/ instead of being "
        "dropped, this gate would be guarding the wrong thing"
    )
