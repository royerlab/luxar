#!/usr/bin/env python3
"""Check that BUILT scenes carry the citation their demo declares.

``test_credited_demo_wires_citation`` already proves every credited demo passes
``citation=`` into its scene constructor. That is a check on the SOURCE, and it
cannot see what actually reached disk.

The gap is real and has bitten. On 2026-08-21 ``datasets/demos/desi_galaxies``
was regenerated from a checkout that predated the wiring, and the rebuilt store
came out with no ``citation`` root attr at all. Nothing complained: the demo
source was correct, the source-level gate passed, the scene loaded, the gallery
tile still showed a credit (the page falls back to the demo metadata), and the
store quietly shipped uncredited. It was noticed two days later, by comparing
stores against ``DEMO_META`` by hand.

That is the failure this closes. A store is the artifact someone downloads, and
for several of these datasets the licence requires attribution to travel with
it, so "the credit is in the source" is not sufficient.

Deliberately NOT a source check duplicated at another layer: it reads the root
attributes of stores on disk and compares them with the registry. Both zarr
layouts are handled, because naming one metadata document blindly is its own
recurring bug — format 2 keeps root attributes in ``.zattrs`` and has no
``zarr.json``, while format 3 nests attributes inside ``zarr.json``.

Usage::

    hatch run check-scene-credits                     # every built demo scene
    hatch run check-scene-credits path/to.luxar.zarr  # specific stores

Exit code is non-zero when a built store contradicts its demo. A checkout with
no generated scenes is a read-only no-op, matching ``check-demo-ladders``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Iterable, Optional

from arbol import aprint

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "packages" / "luxar" / "src"))

from luxar._zarr_compat import read_node_attrs  # noqa: E402
from luxar.demos.registry import iter_demos  # noqa: E402
from luxar.utils.paths import get_demos_output_dir  # noqa: E402

_ATTRIBUTION_FIELDS = ("short", "doi", "license")
Target = tuple[str, Path, Optional[dict]]


def compare(store: Path, declared: Optional[dict]) -> Optional[str]:
    """The problem with this store, or ``None`` when it agrees with the demo.

    ``declared`` is the demo's ``DEMO_META["citation"]``, or ``None`` when the
    registry has no citation. Registry ``None`` is not by itself proof that a
    demo is synthetic, so callers must only pass registered demo outputs.
    """
    attrs = read_node_attrs(store)
    if attrs is None:
        return "no readable root metadata (neither zarr.json nor .zattrs)"
    carried = attrs.get("citation")
    if carried is not None and not isinstance(carried, Mapping):
        return f"has a malformed citation record (expected a mapping): {carried!r}"
    if declared is None:
        if carried is not None:
            return (
                "carries a citation its demo does not declare: "
                f"{carried.get('short')!r}"
            )
        return None
    want = declared["short"]
    if not carried:
        return f"declares {want!r} but the store carries no citation"
    if not carried.get("short"):
        # `validate_citation` requires `short`, so a luxar-written store always
        # has one -- but a hand-edited or foreign store may not, and the generic
        # comparison below would report the confusing "carries None".
        return f"declares {want!r} but the store's citation has no 'short': {carried!r}"
    for field in _ATTRIBUTION_FIELDS:
        if carried.get(field) != declared.get(field):
            return (
                f"citation field {field!r} is {carried.get(field)!r} "
                f"but the demo declares {declared.get(field)!r}"
            )
    return None


def _demo_stores(demos_dir: Path) -> Iterable[tuple[str, Path, Optional[dict]]]:
    """(demo key, store path, declared citation) for every built demo scene."""
    for demo in iter_demos():
        for stem in demo.outputs:
            store = demos_dir / f"{stem}.luxar.zarr"
            if store.is_dir():
                yield demo.key, store, demo.citation


def _store_stem(path: Path) -> str:
    """Return a scene store's output stem for directory or archive spelling."""
    for suffix in (".luxar.zarr.zip", ".luxar.zarr"):
        if path.name.endswith(suffix):
            return path.name.removesuffix(suffix)
    return path.name


def _explicit_targets(stores: Iterable[Path]) -> tuple[list[Target], list[str]]:
    """Resolve named paths to registered demo outputs and skip unsupported ones."""
    by_stem = {
        stem: (demo.key, demo.citation)
        for demo in iter_demos()
        for stem in demo.outputs
    }
    targets = []
    skipped = []
    for store in stores:
        known = by_stem.get(_store_stem(store))
        if known is None:
            skipped.append(f"{store.name}: not a known demo output, skipped")
            continue
        if store.name.endswith(".luxar.zarr.zip"):
            skipped.append(f"{store.name}: archive stores are not inspected, skipped")
            continue
        key, declared = known
        targets.append((key, store, declared))
    return targets, skipped


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "stores",
        nargs="*",
        type=Path,
        help="specific stores (default: all built demos)",
    )
    parser.add_argument(
        "--demos-dir",
        type=Path,
        default=get_demos_output_dir(create=False),
        help="where built demo scenes live",
    )
    args = parser.parse_args(argv)

    if args.stores:
        targets, skipped = _explicit_targets(args.stores)
    else:
        targets = list(_demo_stores(args.demos_dir))
        skipped = []

    if not targets and not skipped:
        aprint("no built demo scenes found; nothing to check")
        return 0

    problems = []
    for key, store, declared in targets:
        problem = compare(store, declared)
        if problem:
            problems.append(f"{key}: {store.name} {problem}")

    aprint(
        f"checked {len(targets)} built scene(s); problems: {len(problems)}; "
        f"skipped: {len(skipped)}"
    )
    for line in problems:
        aprint(f"  {line}")
    for line in skipped:
        aprint(f"  {line}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
