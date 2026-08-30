#!/usr/bin/env python3
"""Check that built scenes carry usable streaming ladders on their big leaves.

A points/lines leaf that loads all-at-once freezes the browser for as long as it
takes to decode and commit — ~85 s on the 9.75M-point DESI demo before its levels
were laddered (royerlab/luxar#808). This walks built scenes and reports, per
laddered leaf, whether the ladder is actually *useful*.

A ladder can exist and still be worthless. For example,
``global_rivers_earth/terrain`` once shipped levels of 8 / 56 / 272 / 1174 /
7,998,490 over 8M points: 99.98% of the data remained in one commit. A large
leaf therefore fails when any level exceeds either ``--max-share`` of the total
or the absolute ``--max-level-elements`` commit budget.

Usage:
    hatch run check-demo-ladders                              # all built demos
    hatch run check-demo-ladders path/to.luxar.zarr ...
    hatch run check-demo-ladders --min-elements 500000

Exit code is non-zero if any leaf fails. The command is part of ``hatch run
check``; on a checkout without built demos it is a read-only no-op.

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
from pathlib import Path
from typing import Any

import zarr
from arbol import aprint, asection

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

#: Leaves at or below this element count are reported but never failed — a small
#: leaf commits fast enough that an all-at-once load is invisible.
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
DEFAULT_MIN_SLICE_FIRST_RUNG = 1_000


def _element_count(attrs: dict[str, Any]) -> int:
    for key in ("n_points", "n_vertices", "n_splats"):
        if key in attrs:
            return int(attrs[key] or 0)
    return 0


def walk_leaves(group: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    """Yield ``(path, group)`` for every points/lines/gsplats leaf in the tree."""
    attrs = dict(group.attrs)
    node_type = attrs.get("type")
    if node_type in ("points", "lines", "gsplats"):
        yield path or "/", group
        return  # additive_<i> subgroups are internal to the leaf
    for name in group.group_keys():
        yield from walk_leaves(group[name], f"{path}/{name}")


def _first_rung_histogram(leaf: Any) -> Counter[tuple[object, ...]] | None:
    """Histogram one sliced leaf's rung 0 by encoded hidden coordinate."""
    attrs = dict(leaf.attrs)
    if int(attrs.get("n_additive_sublods", 1) or 1) <= 1:
        return None
    try:
        rung = leaf["additive_0"]
    except KeyError:
        return Counter()
    dims = [int(dim) for dim in rung.attrs.get("slice_dims", [])]
    if not dims:
        return None
    if "centers" not in rung:
        return Counter()
    columns = rung["centers"][:, dims]
    counts: Counter[tuple[object, ...]] = Counter()
    counts.update(tuple(row) for row in columns)
    return counts


def _node_slice_histogram(group: Any) -> Counter[tuple[object, ...]] | None:
    """Reduce one geometry node: partition=sum, substitutive LOD=keywise max."""
    attrs = dict(group.attrs)
    if attrs.get("type") in ("points", "lines", "gsplats"):
        return _first_rung_histogram(group)

    children = [group[name] for name in group.group_keys()]
    if attrs.get("kind") == "partition":
        combined: Counter[tuple[object, ...]] = Counter()
        for child in children:
            child_histogram = _node_slice_histogram(child)
            if child_histogram is not None:
                combined += child_histogram
        return combined or None
    if attrs.get("kind") == "lod":
        alternatives = [
            histogram
            for child in children
            if (histogram := _node_slice_histogram(child)) is not None
        ]
        if not alternatives:
            return None
        keys = set().union(*(histogram.keys() for histogram in alternatives))
        return Counter(
            {key: max(histogram[key] for histogram in alternatives) for key in keys}
        )
    return None


def sliced_first_rung_counts(root: Any) -> list[int]:
    """Measure every sliced node alternative's global per-slice rung-0 maximum."""
    histogram = _node_slice_histogram(root)
    if histogram is not None:
        return [max(histogram.values(), default=0)]
    return [
        count
        for name in root.group_keys()
        for count in sliced_first_rung_counts(root[name])
    ]


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
        if total > min_elements:
            return (
                "fail",
                f"{total:,} elements in ONE commit (no ladder) — will block the "
                "main thread on load",
            )
        return ("skip", f"{total:,} elements, no ladder (below threshold)")

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
        if biggest > max_level_elements:
            return (
                "fail",
                f"{detail} — largest level has {biggest:,} elements, above the "
                f"{max_level_elements:,} absolute commit cap",
            )
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


def run_gate(paths: Sequence[Path], args: argparse.Namespace) -> int:
    """The original streaming-ladder audit, unchanged. Returns its exit code."""
    counts = {"ok": 0, "warn": 0, "fail": 0, "skip": 0}
    icons = {"ok": "✅", "warn": "⚠️ ", "fail": "❌", "skip": "· "}
    failures: list[str] = []

    for scene in paths:
        try:
            root = zarr.open(str(scene), mode="r")
        except Exception as exc:
            aprint(f"❌ {scene.name}: cannot open ({exc})")
            counts["fail"] += 1
            failures.append(scene.name)
            continue

        results: list[tuple[str, str, str]] = []
        for leaf_path, leaf in walk_leaves(root):
            status, message = check_leaf(
                leaf,
                min_elements=args.min_elements,
                max_share=args.max_share,
                max_level_elements=args.max_level_elements,
                min_sublods=args.min_sublods,
            )
            results.append((leaf_path, status, message))
            counts[status] += 1
            if status == "fail":
                failures.append(f"{scene.name}{leaf_path}")

        sliced_counts = sliced_first_rung_counts(root)
        for index, count in enumerate(sliced_counts):
            if count <= 0:
                results.append(
                    (f"/sliced-node-{index}", "fail", "empty rung-0 slice survey")
                )
                counts["fail"] += 1
                failures.append(f"{scene.name}/sliced-node-{index}")
            elif count < args.min_slice_first_rung:
                results.append(
                    (
                        f"/sliced-node-{index}",
                        "fail",
                        f"largest rung-0 slice has {count:,} elements, below the "
                        f"{args.min_slice_first_rung:,} absolute first-paint floor",
                    )
                )
                counts["fail"] += 1
                failures.append(f"{scene.name}/sliced-node-{index}")

        visible_results = [
            result
            for result in results
            if not args.quiet or result[1] in ("warn", "fail")
        ]
        if visible_results:
            with asection(scene.name):
                for leaf_path, status, message in visible_results:
                    aprint(f"{icons[status]} {leaf_path}: {message}")

    aprint(
        f"{counts['ok']} ok, {counts['warn']} warned, {counts['fail']} failed, "
        f"{counts['skip']} below threshold"
    )
    if failures:
        with asection("Failed leaves"):
            for name in failures:
                aprint(name)
        return 1
    return 0


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
    if not paths:
        aprint("No scenes found. Build a demo first, or pass a path explicitly.")
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
