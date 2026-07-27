#!/usr/bin/env python3
"""Check that built scenes carry usable streaming ladders on their big leaves.

A points/lines leaf that loads all-at-once freezes the browser for as long as it
takes to decode and commit — ~85 s on the 9.75M-point DESI demo before its levels
were laddered (royerlab/luxar#808). This walks built scenes and reports, per
laddered leaf, whether the ladder is actually *useful*.

The important check is the last one. A ladder can exist and still be worthless:
``global_rivers_earth/terrain`` shipped ``additive_lod=dict(method="spatial-uniform",
n_lods=5)`` over 8M points and produced levels of 8 / 56 / 272 / 1174 / 7,998,490 —
99.98% of the data in the final commit. It streamed in name only. So a leaf fails
if its largest level is more than ``--max-share`` of the total.

Usage:
    hatch run python scripts/check_demo_ladders.py                 # all built demos
    hatch run python scripts/check_demo_ladders.py path/to.luxar.zarr ...
    hatch run python scripts/check_demo_ladders.py --min-elements 500000

Exit code is non-zero if any leaf fails, so this works as a gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import zarr

#: Leaves at or below this element count are reported but never failed — a small
#: leaf commits fast enough that an all-at-once load is invisible.
DEFAULT_MIN_ELEMENTS = 200_000

#: A ladder whose largest level exceeds this share of the total is not a
#: progressive paint, whatever its level count says.
DEFAULT_MAX_SHARE = 0.5

#: Fewer levels than this cannot stream across more than one refinement pass.
DEFAULT_MIN_SUBLODS = 3


def _element_count(attrs: Dict[str, Any]) -> int:
    for key in ("n_points", "n_vertices", "n_splats"):
        if key in attrs:
            return int(attrs[key] or 0)
    return 0


def walk_leaves(group: Any, path: str = "") -> Iterator[Tuple[str, Any]]:
    """Yield ``(path, group)`` for every points/lines/gsplats leaf in the tree."""
    attrs = dict(group.attrs)
    node_type = attrs.get("type")
    if node_type in ("points", "lines", "gsplats"):
        yield path or "/", group
        return  # additive_<i> subgroups are internal to the leaf
    for name in group.group_keys():
        yield from walk_leaves(group[name], f"{path}/{name}")


def check_leaf(
    path: str,
    leaf: Any,
    *,
    min_elements: int,
    max_share: float,
    min_sublods: int,
) -> Tuple[str, str]:
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

    sizes: List[int] = []
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
        if share > max_share:
            return (
                "fail",
                f"{detail} — degenerate ladder, the last level is effectively the "
                "whole dataset",
            )
        if n_sub < min_sublods:
            return ("warn", f"{detail} — only {n_sub} levels")
    return ("ok", detail)


def scene_paths(args_paths: List[str]) -> List[Path]:
    if args_paths:
        return [Path(p) for p in args_paths]
    from luxar.utils.paths import get_demos_output_dir

    return sorted(get_demos_output_dir().glob("*.luxar.zarr"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenes", nargs="*", help="scene paths (default: built demos)")
    parser.add_argument("--min-elements", type=int, default=DEFAULT_MIN_ELEMENTS)
    parser.add_argument("--max-share", type=float, default=DEFAULT_MAX_SHARE)
    parser.add_argument("--min-sublods", type=int, default=DEFAULT_MIN_SUBLODS)
    parser.add_argument(
        "--quiet", action="store_true", help="only print warnings and failures"
    )
    args = parser.parse_args()

    paths = scene_paths(args.scenes)
    if not paths:
        print("No scenes found. Build a demo first, or pass a path explicitly.")
        return 0

    counts = {"ok": 0, "warn": 0, "fail": 0, "skip": 0}
    icons = {"ok": "✅", "warn": "⚠️ ", "fail": "❌", "skip": "· "}
    failures: List[str] = []

    for scene in paths:
        try:
            root = zarr.open(str(scene), mode="r")
        except Exception as exc:
            print(f"❌ {scene.name}: cannot open ({exc})")
            counts["fail"] += 1
            failures.append(scene.name)
            continue

        header_shown = args.quiet
        if not args.quiet:
            print(f"\n{scene.name}")

        for leaf_path, leaf in walk_leaves(root):
            status, message = check_leaf(
                leaf_path,
                leaf,
                min_elements=args.min_elements,
                max_share=args.max_share,
                min_sublods=args.min_sublods,
            )
            counts[status] += 1
            if status == "fail":
                failures.append(f"{scene.name}{leaf_path}")
            if args.quiet and status in ("ok", "skip"):
                continue
            if not header_shown:
                print(f"\n{scene.name}")
                header_shown = True
            print(f"  {icons[status]} {leaf_path}: {message}")

    print(
        f"\n{counts['ok']} ok, {counts['warn']} warned, {counts['fail']} failed, "
        f"{counts['skip']} below threshold"
    )
    if failures:
        print("\nFailed leaves:")
        for name in failures:
            print(f"  {name}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
