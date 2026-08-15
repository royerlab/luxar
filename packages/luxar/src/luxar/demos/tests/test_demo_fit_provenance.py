"""A demo must not throw away the provenance of the fit it ships.

``GSplatData.save(include_fitting_info=False)`` does not merely relocate the fit
stats -- ``save_gsplats.split_fitting_info`` leaves ``fitting_info`` as ``None``,
so the ``_FITTING_INFO_KEYS`` whitelist never runs, and because those same keys
are excluded from ``pipeline_info`` they are dropped from the store ENTIRELY.

For the demos this matters more than it looks: several of them save straight to
the file that is then SHIPPED (the cache path and the packaged artifact are the
same name), so suppressing the flag means the published dataset can never state
what volume it is a representation of -- no source grid, and therefore no
compression ratio.

The one legitimate use is a store that never came from a fit at all, which is
why the exemption below is keyed to a reason rather than merely allowed.
"""

from __future__ import annotations

import ast

from luxar.demos import registry

DEMO_PATHS = sorted(registry._DEMOS_DIR.glob("demo_*.py"))

#: Call sites allowed to pass ``include_fitting_info=False``, and why.
#:
#: NOT a general opt-out list: each entry must describe a store built WITHOUT
#: fitting, where there is no provenance to keep. A real fit belongs nowhere
#: near this mapping.
_NO_FIT_TO_RECORD = {
    "demo_gsplats_4d_nexrad_supercell.py": (
        "the empty-frame sentinel -- a synthetic single-splat placeholder for a "
        "frame with nothing above the dBZ floor, which the fitter never saw"
    ),
}


def _suppressing_modules() -> dict[str, int]:
    """Map demo file name -> count of ``include_fitting_info=False`` call sites."""
    found: dict[str, int] = {}
    for path in DEMO_PATHS:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg != "include_fitting_info":
                    continue
                if isinstance(kw.value, ast.Constant) and kw.value.value is False:
                    found[path.name] = found.get(path.name, 0) + 1
    return found


def test_no_demo_discards_the_provenance_of_a_real_fit() -> None:
    """Only the documented no-fit stores may suppress fitting info."""
    suppressing = _suppressing_modules()
    assert set(suppressing) == set(_NO_FIT_TO_RECORD), (
        "include_fitting_info=False changed. A demo that saves a REAL fit must "
        "keep its provenance -- the flag drops the source-grid stamps entirely "
        "(they are excluded from pipeline_info too), so the shipped dataset "
        "loses its compression figure.\n"
        f"  suppressing now: {sorted(suppressing)}\n"
        f"  documented no-fit stores: {sorted(_NO_FIT_TO_RECORD)}"
    )


def test_every_exemption_states_why_there_is_no_fit() -> None:
    """A bare exemption is how this list would rot into a general opt-out."""
    for name, reason in _NO_FIT_TO_RECORD.items():
        assert (registry._DEMOS_DIR / name).exists(), (
            f"{name} is exempted but no longer exists -- drop the entry"
        )
        assert len(reason) > 40, f"{name}: give the reason the fit is absent"


def test_the_detector_sees_a_planted_suppression() -> None:
    """Guard against the scan quietly matching nothing (a vacuous gate)."""
    source = "result.save(path, include_fitting_info=False)\n"
    tree = ast.parse(source)
    hits = [
        kw
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg == "include_fitting_info"
        and isinstance(kw.value, ast.Constant)
        and kw.value.value is False
    ]
    assert len(hits) == 1
    assert DEMO_PATHS, "no demo modules were scanned at all"


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
