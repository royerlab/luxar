#!/usr/bin/env python3
"""Check that built scenes carry usable streaming ladders on their big leaves.

A points/lines leaf that loads all-at-once freezes the browser for as long as it
takes to decode and commit — ~85 s on the 9.75M-point DESI demo before its levels
were laddered (royerlab/luxar#808). Additive ladders bound first-paint and
individual commit work; they do not reduce terminal geometry residency. The
viewer folds completed rungs into one cumulative payload, avoiding the former
second retained copy. This walks built scenes and reports, per laddered leaf,
whether the ladder is actually *useful*.

An unladdered sliced leaf is judged by its busiest resident coordinate fetch,
not by its declared all-slices total; leaves without usable slice metadata keep
the declared-total fallback.

A ladder can exist and still be worthless. For example,
``global_rivers_earth/terrain`` once shipped levels of 8 / 56 / 272 / 1174 /
7,998,490 over 8M points: 99.98% of the data remained in one commit. A large
leaf therefore fails when any level exceeds either ``--max-share`` of the total
or the absolute ``--max-level-elements`` commit budget. On a barrier-ordered
sliced leaf, that absolute arm measures the conservative largest per-coordinate
fetch from the level's chunk bounds; unsliced, unindexed, malformed or
mixed-metadata, and ``extend_to_all`` leaves retain the node-level cap. This
bounds-based commit arm is independent of the coordinate-decoding histogram
used by the share arm below, so their reports can differ at chunk boundaries.

A SECOND arm audits nodes the viewer SLICES (any non-displayed dimension). A
ladder's rungs are sized against the whole node, but only one slice is ever on
screen, so rung 0 arrives divided by the slice count — and under a playback
frame budget rung 0 is all the viewer commits, re-paid every tick and never
converging past it. ``gsplats_4d_drosophila_embryogenesis`` shipped with a
median of 45 splats per timepoint at rung 0 and played as an empty screen
(#2374/#2376). Each sliced node therefore fails on either a per-coordinate
ABSOLUTE floor (measured at its SPARSEST slices, not its busiest) or a SHARE
floor on rung 0's fraction of the node, or of the worst part for a partition.

Usage:
    hatch run check-demo-ladders                              # all built demos
    hatch run check-demo-ladders path/to.luxar.zarr ...
    hatch run check-demo-ladders --min-elements 500000
    hatch run check-demo-ladders --require-scenes             # scenes MUST exist

Exit code is non-zero if any leaf fails. The command is part of ``hatch run
check``; on a checkout without built demos it remains a read-only no-op, since
the output directory is gitignored and an empty inventory is the normal state
of a fresh clone and of CI. That run now reports INSPECTED NOTHING instead of
wording that reads like a pass (audit A9-04), and ``--require-scenes`` makes it
a failure for callers that know scenes should be there — the gallery generator
or the pre-upload inventory audit.

A SECOND, independent pass lives behind ``--screen`` (``--screen-only`` to skip
the ladder gate above): the LOD **opening-shot screen** from
:mod:`luxar.io.lod_screening`. It answers a different question — per ``kind=lod``
group, would re-deriving the ladder onto the ``screen-area`` selector actually
make the OPENING framing land on a coarser level? — and it is a REPORT ONLY.
No verdict it produces can change this script's exit code; only a genuine error
(bad arguments) does. That is deliberate: this command is part of ``hatch run
check``, and a screening verdict is an observation about a store, not a defect
in it.

    hatch run check-demo-ladders --screen-only datasets/examples/*.luxar.zarr
    hatch run check-demo-ladders --screen-only --screen-verdict win .../*.zarr
    hatch run check-demo-ladders --screen --screen-aspect 4:3=1.3333
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Iterator, Sequence
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from arbol import aprint, asection

from luxar.encoding.decoder import decode_coordinate_columns
from luxar.io.lod_screening import (
    DEFAULT_ASPECTS,
    DEFAULT_FIT_FOV,
    DEFAULT_VIEWPORT_LONG_PX,
    VERDICT_ORDER,
    print_screen_report,
    screen_stores,
)

#: The ``--screen-aspect`` default, rendered from :data:`DEFAULT_ASPECTS` so the
#: help text and the measured aspects can never drift apart.
DEFAULT_SCREEN_ASPECT_SPEC = ",".join(
    f"{label}={value!r}" for label, value in DEFAULT_ASPECTS
)

#: Unsliced leaves at or below this total, and sliced leaves whose busiest
#: resident coordinate fetch is at or below it, are reported but never failed.
#: Such a commit is small enough that an all-at-once load is invisible.
DEFAULT_MIN_ELEMENTS = 200_000

#: A ladder whose largest level exceeds this share of the total is not a
#: progressive paint, whatever its level count says. Set to 0.6 because that is
#: the strict supremum of the sliver-folded LAST increment's share in
#: ``stream_cuts`` — the degeneracy this gate targets: when the final increment
#: is < chunk/2 it folds into the previous cut, and the folded increment's share
#: approaches (but never reaches) 0.6 in the tightest case (n just under 2.5·c;
#: see ``utils/lod_breakpoints.stream_cuts``). This bounds the FOLDED last
#: increment, NOT every level's share: a user-tuned first chunk with 1.5·c ≤ n <
#: 2·c can push the FIRST level to ~2/3, but that regime is unreachable at the
#: demo default chunk (small relative to n), so 0.6 is the right gate here.
DEFAULT_MAX_SHARE = 0.6

#: A relative share can look healthy while still leaving millions of elements
#: in one commit. Cap every increment independently of the total leaf size.
DEFAULT_MAX_LEVEL_ELEMENTS = 1_000_000

#: Fewer levels than this cannot stream across more than one refinement pass.
DEFAULT_MIN_SUBLODS = 3

#: A sliced first rung below this absolute element count is visually empty in
#: playback even when its share of the whole node looks structurally healthy.
#:
#: Calibrated against OBSERVED playback rather than chosen, on the five sliced
#: demos measured in #2374 (rung-0 order statistics from the built store against
#: what the deployed viewer actually commits while the axis plays):
#:
#:     node               p05 rung-0   observed playback        verdict
#:     drosophila                  7   20-51 of 166,443         blank
#:     nexrad                      4   187-440 of ~10,000       structure gone
#:     zebrafish/endoderm        330   ~1,449 of ~11,159 (13%)  soft, usable
#:     celegans                1,069   ~2,970 of 3,209 (93%)    fine
#:     neuromast/membranes     2,962   12,842-17,465            soft, usable
#:
#: The boundary between "structure gone" and "soft but usable" therefore lies
#: between 4 and 330. 250 sits inside that gap with zebrafish — the closest
#: PASSING store — clearing it by 32%. Do not raise it without re-measuring:
#: at 2,000 (the value before the percentile reduction below) zebrafish and
#: celegans both fail despite playing acceptably.
DEFAULT_MIN_SLICE_FIRST_RUNG = 250

#: A sliced node's rung 0 must also be a usable SHARE of the node (or every
#: drawable part), because playback re-pays rung 0 on every tick and never
#: converges past it. This is the gate form of the authoring contract in
#: ``demos/_lod_policy`` (rung 0 >= ``n / SLICED_LADDER_MAX_DEPTH``, i.e. 12.5%);
#: the 0.10 here leaves that a margin rather than tracking it exactly, so a small
#: rounding change in the ladder builder does not turn the corpus red.
DEFAULT_MIN_SLICE_RUNG_SHARE = 0.10

#: Nodes exempt from the share arm, each keyed to the REASON it is allowed —
#: a shrinking allowlist, not a tolerance. Every entry is a store built before
#: the #2384 contract whose rung 0 is exactly 1/16 of the node, and whose
#: rebuild was CANCELLED rather than postponed: measurement showed these axes
#: are keypress-navigated, which takes the loader's ``refine`` pass rather than
#: the budgeted ``playback`` one, so they stream past rung 0 and converge
#: instead of starving a frame (#2374, #2376).
#: This is deliberately an allowlist and not a warning. A warning here would
#: never clear — nothing is scheduled to rebuild these — and a permanently
#: warning arm stops being a gate at all. As an allowlist the arm bites on
#: anything NEW today, each exemption carries its justification, and the list
#: shrinks to empty if these stores are ever rebuilt for another reason. Each
#: entry also carries a 6% lower bound just below its measured 6.25% share, so a
#: degraded rebuild goes red instead of inheriting a stale path exemption.
SHARE_ARM_EXEMPT: dict[str, tuple[float, str]] = {
    "arxiv_papers_kaggle.luxar.zarr/arxiv_papers_kaggle": (
        0.06,
        "pre-#2384 16-rung ladder (rung 0 = 410,796 of 6,572,730 = 6.25%) over 2 "
        "keypress-navigated coordinates; rebuild cancelled, not deferred",
    ),
    "esm3_protein_landscape.luxar.zarr/proteins": (
        0.06,
        "pre-#2384 16-rung ladder (rung 0 = 71,938 of 1,151,006 = 6.25%) over 2 "
        "keypress-navigated coordinates; rebuild cancelled, not deferred",
    ),
    "human_multiome_peak_umap.luxar.zarr/Cells": (
        0.06,
        "pre-#2384 16-rung ladder (rung 0 = 390,546 of 6,248,730 = 6.25%) over 6 "
        "keypress-navigated coordinates; rebuild cancelled, not deferred",
    ),
    "mouse_multiome_peak_umap.luxar.zarr/Cells": (
        0.06,
        "pre-#2384 16-rung ladder (rung 0 = 72,095 of 1,153,506 = 6.25%) over 6 "
        "keypress-navigated coordinates; rebuild cancelled, not deferred",
    ),
    "zebrahub_multiome_peak_umap.luxar.zarr/Cells": (
        0.06,
        "pre-#2384 16-rung ladder (rung 0 = 280,364 of 4,485,810 = 6.25%) over 7 "
        "keypress-navigated coordinates; rebuild cancelled, not deferred",
    ),
}

LEAF_EXEMPT: dict[str, tuple[int | None, str]] = {
    "gsplats_4d_drosophila_embryogenesis.luxar.zarr/drosophila_nuclei": (
        37_930_613,
        "pinned 2026-08 archive has a measured 37,930,613-element level",
    ),
    "gsplats_recipes_tribolium.luxar.zarr/recipe_flat/flat": (
        None,
        "control: the flat leaf of the six-recipe LOD comparison, unladdered by design",
    ),
}

#: Which order statistic of the per-coordinate histogram decides starvation.
#: A ladder starves at its SPARSEST slice, not its densest, so this reduction
#: must be a low percentile. It replaced ``max()``, which asked whether the
#: BUSIEST coordinate was healthy and so passed any node with one busy slice
#: and hundreds of starved ones. That is not hypothetical: the spread on real
#: stores runs from 2.1x (celegans) to 122x (nexrad) between p05 and max, and
#: ``zebrafish_timelapse/endoderm`` passed the old reduction on a max of 6,648
#: while a twentieth of its timepoints render 330 elements or fewer.
SLICE_STARVATION_PERCENTILE = 0.05


def _element_count(attrs: dict[str, Any]) -> int:
    for key in ("n_points", "n_vertices", "n_splats"):
        if key in attrs:
            return int(attrs[key] or 0)
    return 0


def _slice_ordering(level: Any) -> tuple[list[int], int, str] | None:
    """Return barrier dims, row atom, and bounds array for one level.

    Hidden barrier dimensions are discrete and non-spatial, so the viewer's
    quarter-cell query targets one coordinate. Coordinates closer than the
    stored bounds epsilon are the residual exception handled conservatively by
    :func:`_busiest_slice_elements`.
    """
    attrs = dict(level.attrs)
    if attrs.get("extend_to_all"):
        return None
    if attrs.get("type") == "lines":
        ordering = attrs.get("vertex_ordering")
        if not isinstance(ordering, dict):
            return None
        bounds_name = "vertex_chunk_bounds"
    else:
        ordering = attrs
        bounds_name = "chunk_bounds"
    try:
        dims = [int(dim) for dim in ordering.get("slice_dims", [])]
        chunk_size = int(ordering.get("chunk_size", 0) or 0)
    except (TypeError, ValueError):
        return None
    if not dims or chunk_size < 1 or bounds_name not in level:
        return None
    return dims, chunk_size, bounds_name


def _busiest_slice_elements(level: Any) -> int | None:
    """Conservative largest one-coordinate fetch from chunk bounds.

    Whole chunks are the viewer's fetch atom, so a chunk whose bounds touch two
    coordinates counts toward both. Distinct coordinates must be separated by
    more than the producer's bounds epsilon or their intervals merge here,
    conservatively inflating the result toward the whole-level count.
    """
    ordering = _slice_ordering(level)
    if ordering is None:
        return None
    dims, chunk_size, bounds_name = ordering
    total = _element_count(dict(level.attrs))
    bounds = np.asarray(level[bounds_name][:])
    expected_chunks = (total + chunk_size - 1) // chunk_size
    if (
        total < 1
        or bounds.ndim != 3
        or bounds.shape[0] != expected_chunks
        or bounds.shape[2] < 2
        or len(set(dims)) != len(dims)
        or any(dim < 0 or dim >= bounds.shape[1] for dim in dims)
    ):
        return None
    slice_bounds = bounds[:, dims, :2].astype(np.float64, copy=False)
    if not np.isfinite(slice_bounds).all() or np.any(
        slice_bounds[:, :, 0] > slice_bounds[:, :, 1]
    ):
        return None
    rows = np.minimum(
        chunk_size,
        total - np.arange(expected_chunks, dtype=np.int64) * chunk_size,
    )
    candidates = [np.unique(slice_bounds[:, axis, :]) for axis in range(len(dims))]
    busiest = 0
    for coordinate in product(*candidates):
        point = np.asarray(coordinate, dtype=np.float64)
        contains = np.all(
            (slice_bounds[:, :, 0] <= point) & (point <= slice_bounds[:, :, 1]),
            axis=1,
        )
        busiest = max(busiest, int(rows[contains].sum()))
    return busiest


def _largest_commit_elements(
    leaf: Any, n_sub: int, node_biggest: int
) -> tuple[int, int | None]:
    """Return the cap measurement and optional coordinate-fetch upper bound."""
    if leaf.attrs.get("extend_to_all"):
        return node_biggest, None
    sliced_sizes = [
        _busiest_slice_elements(leaf[f"additive_{i}"]) for i in range(n_sub)
    ]
    if not sliced_sizes or any(size is None for size in sliced_sizes):
        return node_biggest, None
    sliced_biggest = max(int(size) for size in sliced_sizes if size is not None)
    return sliced_biggest, sliced_biggest


def _commit_cap_result(
    leaf: Any,
    n_sub: int,
    node_biggest: int,
    detail: str,
    max_level_elements: int,
) -> tuple[str, tuple[str, str] | None]:
    """Return a detail suffix and absolute-cap failure when present."""
    commit_biggest, sliced_biggest = _largest_commit_elements(leaf, n_sub, node_biggest)
    detail_suffix = ""
    if sliced_biggest is not None:
        detail_suffix = f", largest coordinate fetch {sliced_biggest:,} elements"
    if commit_biggest <= max_level_elements:
        return detail_suffix, None
    measured = (
        f"largest coordinate fetch is {commit_biggest:,} elements"
        if sliced_biggest is not None
        else f"largest level has {commit_biggest:,} elements"
    )
    return (
        detail_suffix,
        (
            "fail",
            f"{detail}{detail_suffix} — {measured}, above the "
            f"{max_level_elements:,} absolute commit cap",
        ),
    )


def _unladdered_result(leaf: Any, total: int, min_elements: int) -> tuple[str, str]:
    """Judge one flat leaf by its resident coordinate fetch when available."""
    sliced_total = None
    if not leaf.attrs.get("extend_to_all"):
        sliced_total = _busiest_slice_elements(leaf)
    measured_total = sliced_total if sliced_total is not None else total
    if measured_total > min_elements:
        measured = (
            f"{measured_total:,} elements in one coordinate fetch"
            if sliced_total is not None
            else f"{measured_total:,} elements in ONE commit"
        )
        return (
            "fail",
            f"{measured} (no ladder) — will block the main thread on load",
        )
    if sliced_total is not None:
        return (
            "skip",
            f"{total:,} elements, largest coordinate fetch "
            f"{sliced_total:,} elements, no ladder (below threshold)",
        )
    return ("skip", f"{total:,} elements, no ladder (below threshold)")


def walk_leaves(group: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    """Yield ``(path, group)`` for every points/lines/gsplats leaf in the tree."""
    attrs = dict(group.attrs)
    node_type = attrs.get("type")
    if node_type in ("points", "lines", "gsplats"):
        yield path or "/", group
        return  # additive_<i> subgroups are internal to the leaf
    for name in group.group_keys():
        yield from walk_leaves(group[name], f"{path}/{name}")


def _leaf_commit_measurement(leaf: Any) -> int | None:
    """Return the element measurement a leaf exemption bounds."""
    attrs = dict(leaf.attrs)
    total = _element_count(attrs)
    n_sub = int(attrs.get("n_additive_sublods", 1) or 1)
    if n_sub <= 1:
        sliced_total = None
        if not leaf.attrs.get("extend_to_all"):
            sliced_total = _busiest_slice_elements(leaf)
        return total if sliced_total is None else sliced_total
    sizes: list[int] = []
    for index in range(n_sub):
        try:
            sizes.append(_element_count(dict(leaf[f"additive_{index}"].attrs)))
        except KeyError:
            return None
    node_biggest = max(sizes, default=0)
    return _largest_commit_elements(leaf, n_sub, node_biggest)[0]


def _leaf_exemption_reason(group: Any, exemption_key: str) -> str | None:
    """Return a matching exemption reason only within its measured ceiling."""
    exemption = LEAF_EXEMPT.get(exemption_key)
    if exemption is None:
        return None
    ceiling, reason = exemption
    if ceiling is None:
        return reason
    measurements = [
        measurement
        for _, leaf in walk_leaves(group)
        if (measurement := _leaf_commit_measurement(leaf)) is not None
    ]
    if measurements and max(measurements) <= ceiling:
        return reason
    return None


def _first_rung_histogram(
    leaf: Any, zarr_root: Any
) -> Counter[tuple[object, ...]] | None:
    """Histogram one sliced leaf's rung 0 by decoded hidden coordinate."""
    attrs = dict(leaf.attrs)
    if int(attrs.get("n_additive_sublods", 1) or 1) <= 1:
        return None
    try:
        rung = leaf["additive_0"]
    except KeyError:
        return Counter()
    ordering = (
        rung.attrs.get("vertex_ordering", {})
        if attrs["type"] == "lines"
        else rung.attrs
    )
    dims = [int(dim) for dim in ordering.get("slice_dims", [])]
    if not dims:
        return None
    coordinate_array = {
        "points": "positions",
        "lines": "vertices",
        "gsplats": "centers",
    }[attrs["type"]]
    if coordinate_array not in rung:
        return Counter()
    columns = decode_coordinate_columns(rung[coordinate_array], dims, zarr_root)
    counts: Counter[tuple[object, ...]] = Counter()
    counts.update(tuple(row) for row in columns)
    return counts


def _node_slice_measurement(
    group: Any, zarr_root: Any
) -> tuple[Counter[tuple[object, ...]], int, float, bool, str | None] | None:
    """Return a sliced histogram, total, worst share, partition flag, and part."""
    attrs = dict(group.attrs)
    if attrs.get("type") in ("points", "lines", "gsplats"):
        histogram = _first_rung_histogram(group, zarr_root)
        if histogram is None:
            return None
        total = _element_count(attrs)
        share = sum(histogram.values()) / total if total else 0.0
        return histogram, total, share, False, None

    child_names = list(group.group_keys())
    if attrs.get("kind") == "partition":
        combined: Counter[tuple[object, ...]] = Counter()
        total = 0
        shares: list[tuple[float, str]] = []
        for child_name in child_names:
            child = group[child_name]
            measurement = _node_slice_measurement(child, zarr_root)
            if measurement is not None:
                child_histogram, child_total, child_share, _, child_worst = measurement
                combined += child_histogram
                total += child_total
                if child_histogram and child_total > 0:
                    worst_name = (
                        child_name
                        if child_worst is None
                        else f"{child_name}/{child_worst}"
                    )
                    shares.append((child_share, worst_name))
        if not combined or not shares:
            return None
        worst_share, worst_name = min(shares, key=lambda item: item[0])
        return combined, total, worst_share, True, worst_name
    if attrs.get("kind") == "lod":
        alternatives = [
            measurement
            for child_name in child_names
            if (measurement := _node_slice_measurement(group[child_name], zarr_root))
            is not None
        ]
        if not alternatives:
            return None
        # Substitutive levels are alternatives, not additive parts. Audit one
        # actual level — the finest by declared element total — so histogram
        # and denominator describe the same drawable subtree.
        return max(alternatives, key=lambda measurement: measurement[1])
    return None


def _node_slice_histogram(
    group: Any, zarr_root: Any
) -> Counter[tuple[object, ...]] | None:
    """Return the sliced rung histogram selected for one geometry node."""
    measurement = _node_slice_measurement(group, zarr_root)
    return None if measurement is None else measurement[0]


def sparsest_slice_elements(histogram: Counter[tuple[object, ...]]) -> int:
    """The rung-0 count at the node's SPARSEST slices.

    Returns the lower 5th-percentile ORDER STATISTIC — an actually observed
    count, never an interpolation between two — so the number in a failure
    message is a slice that exists. With 20 or fewer coordinates the index lands
    on 0 and this is simply the minimum, which is the right reading for a handful
    of slices: there is no tail to discount.

    Coordinates ABSENT from rung 0 do not appear in ``histogram`` and so cannot
    lower this. That is deliberate for an empty coordinate (nothing to draw is
    not starvation) but it does mean a coordinate the node covers while rung 0
    does not is invisible here; when a sliced arm fails, the caller reports
    rung-0 coverage alongside so the gap is at least legible in that verdict.
    """
    if not histogram:
        return 0
    ordered = sorted(histogram.values())
    index = int(SLICE_STARVATION_PERCENTILE * (len(ordered) - 1))
    return ordered[index]


def sliced_rung_histograms(
    root: Any, path: str = "", zarr_root: Any = None
) -> list[tuple[str, Counter[tuple[object, ...]]]]:
    """Yield ``(path, per-coordinate rung-0 histogram)`` for each sliced node."""
    zarr_root = root if zarr_root is None else zarr_root
    return [
        (node_path, histogram)
        for node_path, histogram, _, _, _, _ in sliced_rung_measurements(
            root, path, zarr_root
        )
    ]


def sliced_rung_measurements(
    root: Any, path: str = "", zarr_root: Any = None
) -> list[tuple[str, Counter[tuple[object, ...]], int, float, bool, str | None]]:
    """Yield sliced histograms with totals and worst drawable-part shares/names."""
    zarr_root = root if zarr_root is None else zarr_root
    measurement = _node_slice_measurement(root, zarr_root)
    if measurement is not None:
        histogram, total, share, partitioned, worst_part = measurement
        return [(path or "/", histogram, total, share, partitioned, worst_part)]
    return [
        result
        for name in root.group_keys()
        for result in sliced_rung_measurements(root[name], f"{path}/{name}", zarr_root)
    ]


def sliced_first_rung_counts(
    root: Any, path: str = "", zarr_root: Any = None
) -> list[tuple[str, int]]:
    """Measure each sliced node's rung-0 count at its SPARSEST slices."""
    return [
        (node_path, sparsest_slice_elements(histogram))
        for node_path, histogram in sliced_rung_histograms(root, path, zarr_root)
    ]


def _record_sliced_verdicts(
    root: Any,
    scene_name: str,
    min_first_rung: int,
    results: list[tuple[str, str, str]],
    counts: dict[str, int],
    failures: list[str],
    matched_exemptions: set[str],
    min_rung_share: float = DEFAULT_MIN_SLICE_RUNG_SHARE,
) -> None:
    """Append scene-level sliced-rung failures without inflating ``run_gate``.

    Two independent arms, because they catch different shapes and neither
    subsumes the other:

    * the ABSOLUTE arm asks whether the sparsest slices carry enough elements to
      be worth drawing, and catches a node whose share looks healthy only
      because the node is small;
    * the SHARE arm asks whether rung 0 is a usable fraction of the node at all,
      or of every drawable partition part. It catches a ladder that is simply
      too deep — the authoring defect in #2376, where a rung sized against a
      download budget is divided by the slice count. Its denominators add only
      attr reads to the histogram pass, and the worst-part reduction prevents a
      large healthy part from hiding a thin played frame.
    """
    for (
        path,
        histogram,
        node_total,
        share,
        partitioned,
        worst_part,
    ) in sliced_rung_measurements(root):
        exemption_key = f"{scene_name}{path}"
        if exemption_key in LEAF_EXEMPT:
            matched_exemptions.add(exemption_key)
        count = sparsest_slice_elements(histogram)
        coverage = len(histogram)
        where = (
            f" (rung 0 covers {coverage:,} coordinate{'s' if coverage != 1 else ''})"
        )

        if count <= 0:
            message = "empty rung-0 slice survey"
        elif count < min_first_rung:
            message = (
                f"sparsest rung-0 slices hold {count:,} elements, below the "
                f"{min_first_rung:,} absolute first-paint floor{where}"
            )
        elif node_total and share < min_rung_share:
            share_scope = (
                f"worst part ({worst_part})" if partitioned and worst_part else "node"
            )
            thin = (
                f"rung 0 is {share:.2%} of the {share_scope}, below the "
                f"{min_rung_share:.0%} share floor — the ladder is too deep for a "
                f"sliced node, so playback re-pays a thin rung 0 every tick{where}"
            )
            exemption = SHARE_ARM_EXEMPT.get(f"{scene_name}{path}")
            if exemption is not None and share >= exemption[0]:
                _, reason = exemption
                results.append((path, "warn", f"{thin} [exempt: {reason}]"))
                counts["warn"] += 1
                continue
            message = thin
        else:
            continue
        exemption_reason = _leaf_exemption_reason(root[path], exemption_key)
        if exemption_reason is not None:
            results.append((path, "warn", f"{message} [exempt: {exemption_reason}]"))
            counts["warn"] += 1
            continue
        results.append((path, "fail", message))
        counts["fail"] += 1
        failures.append(f"{scene_name}{path}")


def check_leaf(
    leaf: Any,
    *,
    min_elements: int,
    max_share: float,
    max_level_elements: int,
    min_sublods: int,
) -> tuple[str, str]:
    """Return ``(status, message)`` where status is ok / warn / fail / skip."""
    attrs = dict(leaf.attrs)
    total = _element_count(attrs)
    n_sub = int(attrs.get("n_additive_sublods", 1) or 1)

    if n_sub <= 1:
        return _unladdered_result(leaf, total, min_elements)

    sizes: list[int] = []
    for i in range(n_sub):
        try:
            sizes.append(_element_count(dict(leaf[f"additive_{i}"].attrs)))
        except KeyError:
            return ("fail", f"n_additive_sublods={n_sub} but additive_{i} is missing")

    summed = sum(sizes)
    if total and summed != total:
        return (
            "fail",
            f"levels sum to {summed:,} but the leaf declares {total:,}",
        )

    biggest = max(sizes) if sizes else 0
    share = biggest / summed if summed else 0.0
    detail = f"{n_sub} levels {sizes}, largest {share:.1%} of {summed:,}"

    if summed > min_elements:
        cap_detail, cap_failure = _commit_cap_result(
            leaf, n_sub, biggest, detail, max_level_elements
        )
        detail += cap_detail
        if cap_failure is not None:
            return cap_failure
        if share > max_share:
            return (
                "fail",
                f"{detail} — degenerate ladder, one level is effectively the "
                "whole dataset",
            )
        if n_sub < min_sublods:
            return ("warn", f"{detail} — only {n_sub} levels")
    return ("ok", detail)


def scene_paths(args_paths: Sequence[str]) -> list[Path]:
    """Resolve explicit scenes or inventory the existing demo output directory."""
    if args_paths:
        return [Path(path) for path in args_paths]
    from luxar.utils.paths import get_demos_output_dir

    return sorted(get_demos_output_dir(create=False).glob("*.luxar.zarr"))


def parse_aspects(spec: str) -> list[tuple[str, float]]:
    """Parse a ``--screen-aspect`` list into ``(label, width/height)`` pairs.

    Two spellings per entry, comma-separated: a bare number (``1.7778``), or a
    ``label=value`` pair (``21:9=2.3333``) so the report can name the shape it
    measured. A bare ``W:H`` is also accepted and divided out, since that is how
    aspect ratios are actually written.

    Args:
        spec: The raw flag value.

    Returns:
        The parsed pairs, in the order given.

    Raises:
        ValueError: An entry is not a number, is zero/negative, or is a ``W:H``
            ratio with a zero height. The last one arrives as a
            ``ZeroDivisionError`` and is converted here, so every malformed spec
            leaves this function the same way and ``main`` can route the lot
            through ``parser.error``.
    """
    pairs: list[tuple[str, float]] = []
    for raw in spec.split(","):
        entry = raw.strip()
        if not entry:
            continue
        label, _, value = entry.rpartition("=")
        text = value if label else entry
        if ":" in text:
            width, _, height = text.partition(":")
            try:
                number = float(width) / float(height)
            except ZeroDivisionError:
                raise ValueError(
                    f"aspect {entry!r} has a zero height; a W:H ratio needs a "
                    "non-zero denominator"
                ) from None
        else:
            number = float(text)
        if number <= 0:
            raise ValueError(f"aspect must be > 0, got {entry!r}")
        pairs.append((label or text, number))
    if not pairs:
        raise ValueError("--screen-aspect needs at least one aspect")
    return pairs


def run_screen(paths: Sequence[Path], args: argparse.Namespace) -> None:
    """Run the opening-shot LOD screen and print it. Never affects the exit code."""
    report = screen_stores(
        paths,
        aspects=args.screen_aspects,
        viewport_long_px=args.screen_viewport_long,
        fit_fov=args.screen_fit_fov,
        render_fov=args.screen_render_fov,
    )
    with asection("LOD opening-shot screen (report only — never fails the build)"):
        print_screen_report(report, verdicts=args.screen_verdict or None)


def build_parser() -> argparse.ArgumentParser:
    """The CLI surface: the streaming-ladder gate, plus the opt-in screen."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenes", nargs="*", help="scene paths (default: built demos)")
    parser.add_argument(
        "--require-scenes",
        action="store_true",
        help=(
            "fail instead of passing when no scene was inspected (use wherever "
            "demos ARE expected: gallery generation, the pre-upload audit)"
        ),
    )
    parser.add_argument("--min-elements", type=int, default=DEFAULT_MIN_ELEMENTS)
    parser.add_argument("--max-share", type=float, default=DEFAULT_MAX_SHARE)
    parser.add_argument(
        "--max-level-elements", type=int, default=DEFAULT_MAX_LEVEL_ELEMENTS
    )
    parser.add_argument("--min-sublods", type=int, default=DEFAULT_MIN_SUBLODS)
    parser.add_argument(
        "--min-slice-first-rung",
        type=int,
        default=DEFAULT_MIN_SLICE_FIRST_RUNG,
    )
    parser.add_argument(
        "--min-slice-rung-share",
        type=float,
        default=DEFAULT_MIN_SLICE_RUNG_SHARE,
        # argparse %-formats help text, so a literal percent sign must be
        # doubled or parser construction raises "badly formed help string" —
        # on every invocation, --help included.
        help=(
            "minimum share of a sliced node that rung 0 must carry "
            f"(default {DEFAULT_MIN_SLICE_RUNG_SHARE:.0%})".replace("%", "%%")
        ),
    )
    parser.add_argument(
        "--quiet", action="store_true", help="only print warnings and failures"
    )
    screen = parser.add_argument_group(
        "LOD opening-shot screen (report only, never fails)"
    )
    screen.add_argument(
        "--screen",
        action="store_true",
        help="also run the opening-shot LOD screen (luxar.io.lod_screening)",
    )
    screen.add_argument(
        "--screen-only",
        action="store_true",
        help="run ONLY the screen — the streaming-ladder gate is skipped",
    )
    screen.add_argument(
        "--screen-aspect",
        default=DEFAULT_SCREEN_ASPECT_SPEC,
        help=(
            "comma-separated aspects to measure at, as 'label=value', 'W:H' or a "
            f"bare number (default: {DEFAULT_SCREEN_ASPECT_SPEC})"
        ),
    )
    screen.add_argument(
        "--screen-viewport-long",
        type=int,
        default=DEFAULT_VIEWPORT_LONG_PX,
        help=(
            "pixels on the viewport's LONG axis; affects the legacy diagonal "
            f"metric only (default: {DEFAULT_VIEWPORT_LONG_PX})"
        ),
    )
    screen.add_argument(
        "--screen-fit-fov",
        type=float,
        default=DEFAULT_FIT_FOV,
        help=(
            "vertical FOV the fitted camera DISTANCE is computed at (default: "
            f"{DEFAULT_FIT_FOV:g}, the viewer's own — the cinematic preset "
            "changes the FOV only after the fit)"
        ),
    )
    screen.add_argument(
        "--screen-render-fov",
        type=float,
        default=None,
        help=(
            "fallback vertical FOV for scenes that do not author "
            "viewer_config.camera.fov (default: --screen-fit-fov; pass 63 to "
            "model the cinematic preset)"
        ),
    )
    screen.add_argument(
        "--screen-verdict",
        action="append",
        metavar="BUCKET",
        # Constrained rather than free-form: a typo would otherwise filter every
        # group out and read as "nothing to report", which is the wrong answer.
        choices=VERDICT_ORDER,
        help=(
            "show only these verdict buckets (repeatable); one of "
            f"{', '.join(VERDICT_ORDER)}. Default: all"
        ),
    )
    return parser


def _record_stale_leaf_exemptions(
    inspected_scenes: set[str],
    matched_exemptions: set[str],
) -> list[str]:
    """Fail exact exemptions whose scene exists but whose leaf no longer does."""
    stale_exemptions = sorted(
        key
        for key in LEAF_EXEMPT
        if key.split("/", 1)[0] in inspected_scenes and key not in matched_exemptions
    )
    for key in stale_exemptions:
        aprint(f"❌ stale leaf exemption matched no leaf: {key}")
    return stale_exemptions


def run_gate(paths: Sequence[Path], args: argparse.Namespace) -> int:
    """Run the streaming-ladder and sliced-first-rung audits."""
    counts = {"ok": 0, "warn": 0, "fail": 0, "skip": 0}
    icons = {"ok": "✅", "warn": "⚠️ ", "fail": "❌", "skip": "· "}
    failures: list[str] = []
    matched_exemptions: set[str] = set()
    inspected_scenes: set[str] = set()

    for scene in paths:
        try:
            root = zarr.open(str(scene), mode="r")
        except Exception as exc:
            aprint(f"❌ {scene.name}: cannot open ({exc})")
            counts["fail"] += 1
            failures.append(scene.name)
            continue
        inspected_scenes.add(scene.name)

        results: list[tuple[str, str, str]] = []
        for leaf_path, leaf in walk_leaves(root):
            exemption_key = f"{scene.name}{leaf_path}"
            if exemption_key in LEAF_EXEMPT:
                matched_exemptions.add(exemption_key)
            status, message = check_leaf(
                leaf,
                min_elements=args.min_elements,
                max_share=args.max_share,
                max_level_elements=args.max_level_elements,
                min_sublods=args.min_sublods,
            )
            if status == "fail":
                exemption_reason = _leaf_exemption_reason(leaf, exemption_key)
                if exemption_reason is not None:
                    status = "warn"
                    message = f"{message} [exempt: {exemption_reason}]"
            results.append((leaf_path, status, message))
            counts[status] += 1
            if status == "fail":
                failures.append(f"{scene.name}{leaf_path}")

        _record_sliced_verdicts(
            root,
            scene.name,
            args.min_slice_first_rung,
            results,
            counts,
            failures,
            matched_exemptions,
            args.min_slice_rung_share,
        )

        visible_results = [
            result
            for result in results
            if not args.quiet or result[1] in ("warn", "fail")
        ]
        if visible_results:
            with asection(scene.name):
                for leaf_path, status, message in visible_results:
                    aprint(f"{icons[status]} {leaf_path}: {message}")

    stale_exemptions = _record_stale_leaf_exemptions(
        inspected_scenes, matched_exemptions
    )

    aprint(
        f"{counts['ok']} ok, {counts['warn']} warned, {counts['fail']} failed, "
        f"{counts['skip']} below threshold, "
        f"{len(stale_exemptions)} stale exemption"
        f"{'s' if len(stale_exemptions) != 1 else ''}"
    )
    if failures:
        with asection("Failed leaves"):
            for name in failures:
                aprint(name)
    return int(bool(failures or stale_exemptions))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ladder audit and return a process exit code.

    The screen runs AFTER the gate and its result is discarded on the way to the
    exit code, so it visibly cannot influence the verdict — and under
    ``--screen-only`` the gate does not run at all, which is why that mode always
    exits 0.

    ``--screen-aspect`` is parsed BEFORE anything runs, so a malformed spec
    exits 2 with a usage message instead of dying on a raw traceback halfway
    through — under ``--screen`` that traceback landed after the gate had
    already printed its PASS line.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.screen_aspects = parse_aspects(args.screen_aspect)
    except ValueError as error:
        parser.error(f"--screen-aspect: {error}")

    paths = scene_paths(args.scenes)

    # An empty inventory means this run inspected NOTHING. That is legitimate on
    # a checkout that has never built demos — the output directory is gitignored
    # — so it stays exit 0 by default. But it is not evidence that the ladders
    # are sound, and the old wording ("No scenes found. Build a demo first")
    # read like a tidy pass (audit A9-04). Say plainly that nothing was looked
    # at, and give callers who know demos SHOULD be present a way to enforce it.
    if not paths:
        if args.require_scenes:
            aprint(
                "❌ --require-scenes: INSPECTED NOTHING — no scene found. Build "
                "a demo first, or pass a path explicitly."
            )
            return 1
        aprint(
            "INSPECTED NOTHING — no scene found (this is not a pass). Build a "
            "demo first, pass a path explicitly, or use --require-scenes where "
            "scenes are expected to exist."
        )
        return 0

    if args.screen_only:
        run_screen(paths, args)
        return 0

    exit_code = run_gate(paths, args)
    if args.screen:
        run_screen(paths, args)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
