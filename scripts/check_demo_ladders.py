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
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import zarr
from arbol import aprint, asection

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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ladder audit and return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenes", nargs="*", help="scene paths (default: built demos)")
    parser.add_argument("--min-elements", type=int, default=DEFAULT_MIN_ELEMENTS)
    parser.add_argument("--max-share", type=float, default=DEFAULT_MAX_SHARE)
    parser.add_argument(
        "--max-level-elements", type=int, default=DEFAULT_MAX_LEVEL_ELEMENTS
    )
    parser.add_argument("--min-sublods", type=int, default=DEFAULT_MIN_SUBLODS)
    parser.add_argument(
        "--quiet", action="store_true", help="only print warnings and failures"
    )
    args = parser.parse_args(argv)

    paths = scene_paths(args.scenes)
    if not paths:
        aprint("No scenes found. Build a demo first, or pass a path explicitly.")
        return 0

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


if __name__ == "__main__":
    sys.exit(main())
