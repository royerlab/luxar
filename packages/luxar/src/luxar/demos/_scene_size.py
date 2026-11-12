"""Make a demo scene's text state the scene's own on-disk size.

A caption that quotes the served scene's size cannot know it before the scene
is written, since the caption is part of the scene. The demos that quote it
build into a hidden scratch sibling of the served path with an estimate,
measure the result, rewrite the two text attrs with the measured size, and
repeat until the stated figure stops moving (the text barely changes the size,
so the second measurement normally agrees). Only then is the scratch renamed
onto the served path: an interrupted build leaves the previous scene, or none,
never one stating a wrong size.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Callable, Sequence

from arbol import aprint

from luxar.demos._caption import compression_ratio, format_bytes, format_ratio
from luxar.demos.registry import dir_size_bytes

#: Measurements before the stated size must have converged (two normally do).
SIZE_PASSES = 3


def replace_scene_texts(
    store: Path,
    swaps: Sequence[tuple[str, str]],
    *,
    expected: int,
    restamp_hash: bool = False,
) -> None:
    """Apply ``(old, new)`` substring swaps to a scene's description and overlays.

    Attrs only, through the zarr facade, then the root is re-consolidated.
    Raises unless exactly ``expected`` texts changed. ``restamp_hash`` recomputes
    every node's ``content_hash`` (description and overlay text are both
    hashed): needed when ``store`` is the scene that will be served, so a warm
    viewer cache never validates stale text against an unchanged hash. A store
    that ``luxar optimize`` rewrites afterwards does not need it, because the
    optimizer hashes its output from scratch.
    """
    from luxar._zarr_compat import consolidate, group_keys, open_group

    root = open_group(str(store), mode="r+")
    texts = [(root, "description")]
    overlays = root["overlays"]
    texts += [(overlays[name], "text") for name in group_keys(overlays)]
    replaced = 0
    for group, key in texts:
        text = group.attrs.get(key)
        if not isinstance(text, str):
            continue
        new = text
        for old, fresh in swaps:
            new = new.replace(old, fresh)
        if new != text:
            group.attrs[key] = new
            replaced += 1
    if replaced != expected:
        raise RuntimeError(
            f"expected the size in {expected} texts of {store}, found {replaced}"
        )
    if restamp_hash:
        from luxar.io._compiler.finalize.hashing import compute_content_hashes

        compute_content_hashes(root)
    consolidate(root)


def converge_stated_size(
    stated: int,
    size_texts: Callable[[int], tuple[str, ...]],
    write: Callable[[], None],
    measured_path: Path,
    restate: Callable[[int, int], None],
) -> tuple[int, int, bool]:
    """Write, measure and restate until the stated size's texts stop changing.

    Each pass calls ``write()`` (a no-op when ``restate`` edits the measured
    store itself), measures ``measured_path`` and compares the texts
    ``size_texts`` gives for the measured and the stated size. If they differ,
    ``restate(stated, measured)`` rewrites them and the next pass measures
    again, up to :data:`SIZE_PASSES` passes. Returns ``(stated, measured,
    converged)``: when not converged, the store states ``stated``.
    """
    for attempt in range(1, SIZE_PASSES + 1):
        write()
        measured = dir_size_bytes(measured_path)
        converged = size_texts(measured) == size_texts(stated)
        if converged or attempt == SIZE_PASSES:
            break
        restate(stated, measured)
        stated = measured
    return stated, measured, converged


def scratch_paths(
    output_path: Path, step: str, *, keep_extension: bool = False
) -> tuple[Path, Path]:
    """Hidden ``(scratch, retired)`` siblings of ``output_path``.

    ``step`` names the scratch after what writes it (``packing``, ``building``).
    Same directory, so the final rename stays on one filesystem.
    ``keep_extension`` puts ``step`` first so the scratch still ends in
    ``output_path``'s name: ``LuxarZarrCompiler`` appends ``.luxar.zarr`` to a
    path that does not end in it, so a scratch it writes must keep it.
    """
    name = output_path.name
    return (
        output_path.with_name(
            f".{step}.{name}" if keep_extension else f".{name}.{step}"
        ),
        output_path.with_name(f".{output_path.name}.old"),
    )


def remove_scene(path: Path) -> None:
    """Remove a scene directory, or the symlink standing in for one."""
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def promote_scene(scratch: Path, output_path: Path, retired: Path) -> None:
    """Rename a finished ``scratch`` scene onto ``output_path``.

    A directory cannot be renamed onto a non-empty one, so the old scene is
    retired first and removed after. Renames keep every byte, so a size
    measured on ``scratch`` still describes ``output_path``.
    """
    if output_path.exists():
        os.replace(output_path, retired)
    os.replace(scratch, output_path)
    remove_scene(retired)


def report_scene_size(
    raw_bytes: int, raw_label: str, stated: int, measured: int, converged: bool
) -> None:
    """Print the measured scene size against the raw source, and any drift."""
    ratio = format_ratio(compression_ratio(raw_bytes, measured))
    aprint(
        f"scene on disk: {format_bytes(measured)} ({measured:,} bytes), {ratio} "
        f"smaller than the {format_bytes(raw_bytes)} {raw_label} "
        f"({raw_bytes:,} bytes)"
    )
    if not converged:
        aprint(
            f"⚠️  stated size {format_bytes(stated)} did not converge on the "
            f"measured {format_bytes(measured)} in {SIZE_PASSES} passes"
        )
