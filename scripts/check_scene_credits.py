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
recurring bug — format 2 keeps ``.zmetadata``/``.zattrs`` and has no
``zarr.json``, format 3 nests attributes inside ``zarr.json``.

Usage::

    hatch run check-scene-credits                     # every built demo scene
    hatch run check-scene-credits path/to.luxar.zarr  # specific stores

Exit code is non-zero when a built store contradicts its demo. A checkout with
no generated scenes is a read-only no-op, matching ``check-demo-ladders``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "packages" / "luxar" / "src"))

from luxar.demos.registry import iter_demos  # noqa: E402

#: Metadata documents in both on-disk layouts. Format 2 has no ``zarr.json``;
#: format 3 has no ``.zattrs``. Probing for the wrong one reports a perfectly
#: good store as unreadable.
_V3_DOC = "zarr.json"
_V2_CONSOLIDATED = ".zmetadata"
_V2_ATTRS = ".zattrs"


def read_root_attrs(store: Path) -> Optional[dict]:
    """Root attributes of a store, whichever zarr layout it uses."""
    v3 = store / _V3_DOC
    if v3.is_file():
        try:
            doc: dict = json.loads(v3.read_text())
        except (OSError, ValueError):
            return None
        return dict(doc.get("attributes", {}))
    consolidated = store / _V2_CONSOLIDATED
    if consolidated.is_file():
        try:
            meta: dict = json.loads(consolidated.read_text())["metadata"]
        except (OSError, ValueError, KeyError):
            return None
        return dict(meta.get(_V2_ATTRS, {}))
    attrs = store / _V2_ATTRS
    if attrs.is_file():
        try:
            loose: dict = json.loads(attrs.read_text())
        except (OSError, ValueError):
            return None
        return dict(loose)
    return None


def compare(store: Path, declared: Optional[dict]) -> Optional[str]:
    """The problem with this store, or ``None`` when it agrees with the demo.

    ``declared`` is the demo's ``DEMO_META["citation"]``, or ``None`` for a
    procedurally generated demo that owes no credit.
    """
    attrs = read_root_attrs(store)
    if attrs is None:
        return "no readable root metadata (neither zarr.json nor .zmetadata/.zattrs)"
    carried = attrs.get("citation")
    want = declared["short"] if declared else None
    if want is None:
        if carried:
            return (
                "carries a citation its demo does not declare: "
                f"{carried.get('short')!r}"
            )
        return None
    if not carried:
        return f"declares {want!r} but the store carries no citation"
    if carried.get("short") != want:
        return f"carries {carried.get('short')!r} but the demo declares {want!r}"
    return None


def _demo_stores(demos_dir: Path) -> Iterable[tuple[str, Path, Optional[dict]]]:
    """(demo key, store path, declared citation) for every built demo scene."""
    for demo in iter_demos():
        for stem in demo.outputs:
            for suffix in (".luxar.zarr", ".luxar.zarr.zip"):
                store = demos_dir / f"{stem}{suffix}"
                # A zipped store is not inspected: it would have to be unpacked,
                # and the built directory is what a regeneration writes.
                if store.is_dir():
                    yield demo.key, store, demo.citation
                    break


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
        default=_REPO_ROOT / "datasets" / "demos",
        help="where built demo scenes live",
    )
    args = parser.parse_args(argv)

    if args.stores:
        by_stem = {}
        for demo in iter_demos():
            for stem in demo.outputs:
                by_stem[stem] = (demo.key, demo.citation)
        targets = []
        for store in args.stores:
            stem = store.name.removesuffix(".luxar.zarr")
            key, declared = by_stem.get(stem, (stem, None))
            targets.append((key, store, declared))
    else:
        targets = list(_demo_stores(args.demos_dir))

    if not targets:
        print("no built demo scenes found; nothing to check")
        return 0

    problems = []
    for key, store, declared in targets:
        problem = compare(store, declared)
        if problem:
            problems.append(f"{key}: {store.name} {problem}")

    print(f"checked {len(targets)} built scene(s); problems: {len(problems)}")
    for line in problems:
        print(f"  {line}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
