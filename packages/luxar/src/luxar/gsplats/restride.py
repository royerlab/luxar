"""Sub-sample a gsplat tree's stacked (barrier) axis, preserving its shape.

The operation this exists for: take every ``stride``-th timepoint out of a
timelapse fit and renumber the survivors onto a dense ``0..N-1`` grid, without
disturbing the ``partition -> lod levels -> additive ladder`` structure around
them.

WHY NOT ``gsplat slice``
    That command takes coordinate RANGES, not a stride, and it refuses a
    partition outright (the input must be flattened first). Flattening a
    timelapse fit large enough to want sub-sampling is exactly what
    sub-sampling avoids: the h2afva 253-timepoint archive is 602M splats at its
    finest level, and loading it flat peaked at 115 GB.

WHY RENUMBER
    The viewer navigates a discrete axis by stepping its grid. A strided slice
    that kept the original labels (0, 5, 10, ...) would leave four empty
    positions between every frame, so four steps in five would land on nothing.
    Renumbering to 0..N-1 also lets the encoder's gridded-axis snap store the
    column exactly: the values land on a unit lattice, so the AUTO uint16
    fixed-point encoding is lossless there.

MEMORY
    Read sub-LOD by sub-LOD: peak is one sub-LOD's arrays, not the tree. That is
    the whole reason this is a walk rather than ``load_gsplats`` plus a fancy
    index.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
from arbol import aprint, asection

from luxar._zarr_compat import open_group
from luxar.encoding import EncodingMode
from luxar.encoding.decoder import ArrayDecoder
from luxar.gsplats.gsplat_data import AdditiveSubLOD
from luxar.gsplats.io.save_gsplats import write_gsplats_tree
from luxar.gsplats.tree import GSplatLeaf, GSplatLodGroup, GSplatPartition
from luxar.gsplats.utils.trils import recombine_cholesky

#: How far a stored stacked coordinate may sit from an integer before we refuse
#: to guess which frame it is. Quantization smear on a gridded uint16 axis is
#: ~2e-3; anything approaching a tenth of a frame means the axis is not the
#: integer lattice this function assumes, and rounding it would silently merge
#: or mislabel frames.
_MAX_INTEGRALITY_OFFSET = 0.05

#: Fallback Gaussian truncation radius when no node in the chain declares one.
_DEFAULT_TRUNCATION_RADIUS = 3.0

# NO ATTRIBUTE COPYING HAPPENS HERE, DELIBERATELY.
#
# The ad-hoc script this replaces carried the input's group attrs forward through
# a filter that dropped "stale" counts and per-leaf compositing values. Measured:
# `write_gsplats_tree` ignores arbitrary node `meta` outright -- it stamps every
# node from its own key set, so a probe key set on a leaf or an lod group is
# simply absent afterwards, and `colormap="plasma"` on a leaf reads back as the
# writer's default `"gray"`. That filter therefore never did anything, in either
# direction. Which is also why nothing needs protecting: the output's per-node
# attrs are freshly derived from the arrays, exactly as any new fit's are.
#
# The one channel that DOES reach disk is the writer's own `root_attrs`, which is
# where `root_attrs=` below goes.


def _truncation_radius(*nodes: Any, label: str = "") -> float:
    """Resolve ``truncation_radius`` nearest-first across ``nodes``.

    Reading only the sub-LOD group and defaulting otherwise substitutes the
    constant on any store that declares the radius one level up, changing every
    splat's rendered extent without a word. So walk outward, and only fall back
    when nobody declared it.
    """
    for node in nodes:
        value = dict(node.attrs).get("truncation_radius")
        if value is not None:
            return float(value)
    aprint(
        f"NOTE: {label or 'store'} declares no truncation_radius anywhere; "
        f"using {_DEFAULT_TRUNCATION_RADIUS}"
    )
    return _DEFAULT_TRUNCATION_RADIUS


def _sorted_children(group: Any, prefix: str) -> list[str]:
    return sorted(
        (k for k in group.group_keys() if k.startswith(prefix)),
        key=lambda s: int(s.split("_")[1]),
    )


def _slice_sublod(
    group: Any,
    level: Any,
    root: Any,
    decoder: ArrayDecoder,
    *,
    time_col: int,
    stride: int,
    label: str,
) -> tuple[Optional[AdditiveSubLOD], int, set[int]]:
    """Filter one sub-LOD down to the kept frames.

    Returns ``(sublod or None, splats_read, kept_frame_labels)``.
    """

    def decode(name: str) -> Optional[np.ndarray]:
        return decoder.decode(group[name], root) if name in group else None

    centers = decode("centers")
    if centers is None:
        return None, 0, set()

    stacked = np.asarray(centers[:, time_col], dtype=np.float64)
    rounded = np.rint(stacked)
    offset = np.abs(stacked - rounded)
    if offset.size and offset.max() > _MAX_INTEGRALITY_OFFSET:
        raise ValueError(
            f"{label}: stacked axis is not integral (max offset "
            f"{offset.max():.4f} > {_MAX_INTEGRALITY_OFFSET}). Refusing to "
            f"guess frame labels: a non-integer axis is not a frame index, and "
            f"rounding it would merge or mislabel frames silently."
        )

    keep = (rounded % stride) == 0
    splats_read = int(stacked.size)
    if not keep.any():
        return None, splats_read, set()

    amplitudes = decode("amplitudes")
    cholesky = recombine_cholesky(decode)
    if amplitudes is None or cholesky is None:
        missing = "amplitudes" if amplitudes is None else "cholesky_factors"
        raise ValueError(f"{label}: has a centers array but no {missing}")
    colors = decode("colors")

    kept_centers = np.array(centers[keep], dtype=np.float32)
    kept_centers[:, time_col] = (rounded[keep] / stride).astype(np.float32)

    return (
        AdditiveSubLOD(
            centers=kept_centers,
            amplitudes=np.asarray(amplitudes[keep], dtype=np.float32),
            cholesky_factors=np.asarray(cholesky[keep], dtype=np.float32),
            colors=None if colors is None else np.asarray(colors[keep]),
            truncation_radius=_truncation_radius(group, level, root, label=label),
        ),
        splats_read,
        {int(v) for v in np.unique(rounded[keep])},
    )


def restride_stacked_axis(
    src: str | Path,
    out: str | Path,
    *,
    stride: int,
    time_col: int = 3,
    root_attrs: Optional[Dict[str, Any]] = None,
    derived_from: Optional[str] = None,
    encoding_mode: EncodingMode = EncodingMode.AUTO,
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Write ``out`` holding every ``stride``-th frame of ``src``, renumbered.

    ``src`` must be a ``kind=partition``. Each part may be either a ``kind=lod``
    group of levels or a bare leaf, and either may carry an additive ladder --
    the two shapes a tiled timelapse fit produces before and after its
    substitutive levels are dropped. The output reproduces whichever shape it
    read; only the stacked column is filtered and relabelled.

    ``stride=1`` keeps every frame but still ROUNDS the axis onto its integer
    lattice, which is a real if narrow use: it makes the encoder's gridded snap
    exact on a store whose column carries quantization smear.

    Returns a summary dict with ``n_splats_in``, ``n_splats_out``, ``parts_in``,
    ``parts_out``, ``source_timepoints`` and ``frames``.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    src = Path(src)
    out = Path(out)
    say = progress or aprint

    root = open_group(src, mode="r")
    root_meta = dict(root.attrs)
    decoder = ArrayDecoder()

    part_names = _sorted_children(root, "part_")
    if not part_names:
        raise ValueError(
            f"{src.name} has no part_* groups, so it is not the kind=partition "
            f"tree this expects (root kind={root_meta.get('kind')!r})"
        )

    kept_labels: set[int] = set()
    n_in = n_out = 0
    # Mixed by design: a part reproduces its input shape, lod group or leaf.
    part_nodes: list[Any] = []

    for part_name in part_names:
        part = root[part_name]
        # A part is either a kind=lod group of `child_*` levels, or a LEAF in its
        # own right. Both shapes are real and ship: a tiled fit with per-part LOD
        # produces the former, a tiled fit whose substitutive levels were later
        # dropped produces the latter. Reading only `child_*` silently yields an
        # empty result on the second shape — no error, just nothing kept.
        level_names = _sorted_children(part, "child_") or [""]
        # Annotated as the union the tree accepts, not list[GSplatLeaf]: a list
        # is invariant, so the narrower type would not satisfy `children`.
        leaves: list[Any] = []
        for level_name in level_names:
            level = part[level_name] if level_name else part
            # A level either carries an additive ladder or holds its arrays
            # directly; the empty name selects the level group itself.
            sub_names = _sorted_children(level, "additive_") or [""]
            sublods: list[AdditiveSubLOD] = []
            for sub_name in sub_names:
                group = level[sub_name] if sub_name else level
                sublod, splats_read, labels = _slice_sublod(
                    group,
                    level,
                    root,
                    decoder,
                    time_col=time_col,
                    stride=stride,
                    label="/".join(
                        p for p in (part_name, level_name, sub_name or "(level)") if p
                    ),
                )
                n_in += splats_read
                if sublod is None:
                    continue
                kept_labels.update(labels)
                n_out += int(sublod.centers.shape[0])
                sublods.append(sublod)
            if not sublods:
                continue
            leaves.append(GSplatLeaf(additive_sublods=sublods, meta={}))
        if not leaves:
            say(f"  {part_name}: empty after the filter -- dropped")
            continue
        if level_names == [""]:
            # The part WAS a leaf. Wrapping it in a one-child lod group would
            # invent a level ladder that the input never had, and the viewer
            # would then run its coverage selector over a single option.
            part_nodes.append(leaves[0])
            say(f"  {part_name}: leaf kept (running out={n_out:,})")
        else:
            part_nodes.append(GSplatLodGroup(children=leaves, meta={}))
            say(f"  {part_name}: {len(leaves)} levels kept (running out={n_out:,})")

    if not part_nodes:
        raise ValueError(
            f"nothing survived a stride-{stride} filter of {src.name}: check "
            f"that time_col={time_col} names the stacked axis"
        )

    source_timepoints = sorted(kept_labels)
    node = GSplatPartition(
        children=part_nodes,
        # `max_elements` and `bsp_tree` are real constructor arguments, so unlike
        # `meta` they DO survive: carry the input's, since the parts keep their
        # spatial extents and only their contents thin out.
        max_elements=int(root_meta.get("max_elements", 0) or 0),
        meta={},
        bsp_tree=root_meta.get("bsp_tree"),
    )

    renumbered = stride != 1
    method = (
        f"strided stacked-axis slice, every {stride}th frame"
        if renumbered
        else "stacked axis rounded onto its integer grid (every frame kept)"
    )
    # The caller's root attrs go through the WRITER's `root_attrs`, not the
    # node's `meta`. Only the former reaches the root document: the writer stamps
    # the root from its own key set (type/kind/display_type/max_elements/
    # position_bounds/format_*), so arbitrary node meta is dropped without a
    # word. The ad-hoc script this replaces set `blending_mode` on the node and
    # had to patch the finished store afterwards for exactly that reason.
    stamped = {
        "source_timepoints": source_timepoints,
        "source_stride": int(stride),
        # Name the STORE this was read from, not the file it may go on to
        # replace: those are different claims.
        "source_archive": src.name,
        "stacked_axis_renumbered": renumbered,
        "stacked_axis_rounded_to_grid": True,
    }
    stamped.update(root_attrs or {})
    with asection(f"Writing {out.name}"):
        write_gsplats_tree(
            out,
            node,
            encoding_mode=encoding_mode,
            barrier_dims=[time_col],
            root_attrs=stamped,
            provenance_info={
                "derived_from": derived_from or src.name,
                "method": method,
                "source_timepoints": source_timepoints,
            },
        )

    summary = {
        "n_splats_in": n_in,
        "n_splats_out": n_out,
        "parts_in": len(part_names),
        "parts_out": len(part_nodes),
        "source_timepoints": source_timepoints,
        "frames": len(source_timepoints),
    }
    say(
        f"parts {summary['parts_out']}/{summary['parts_in']}, splats "
        f"{n_out:,}/{n_in:,} ({100.0 * n_out / max(1, n_in):.2f}%), "
        f"frames {summary['frames']}"
    )
    return summary


__all__: Sequence[str] = ["restride_stacked_axis"]
