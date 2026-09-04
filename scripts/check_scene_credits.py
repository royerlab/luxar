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
    hatch run check-scene-credits --require-scenes    # where demos MUST exist

Exit code is non-zero when a built store contradicts its demo, and also when a
named path is not a demo output at all — a typo or a renamed output previously
reported "checked 0 built scene(s); problems: 0" and exited 0 (audit A9-04).

A checkout with no generated scenes remains a read-only no-op, matching
``check-demo-ladders``: the output directory is gitignored, so an empty
inventory is the normal state of a fresh clone and of CI. It now says INSPECTED
NOTHING rather than wording that reads like a pass, and ``--require-scenes``
turns it into a failure for callers that know better — release prep, or a
demo-build pipeline running this after the build.
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
    if carried is None:
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


def _explicit_targets(
    stores: Iterable[Path],
) -> tuple[list[Target], list[str], list[str]]:
    """Resolve named paths to registered demo outputs and skip unsupported ones.

    Returns ``(targets, skipped, unknown)``. ``unknown`` is reported separately
    from ``skipped`` because the two mean different things to the caller, and
    only one of them is their mistake:

    - **unknown** — the path is not a registered demo output at all. A typo, a
      renamed output, or a store from somewhere else. The caller asked for
      something that cannot be checked and should hear about it.
    - **skipped** — a genuine demo output this tool structurally cannot inspect
      (an archive; the stores are not unpacked). The caller named the right
      thing; the limitation is ours.
    """
    by_stem = {
        stem: (demo.key, demo.citation)
        for demo in iter_demos()
        for stem in demo.outputs
    }
    targets = []
    skipped = []
    unknown = []
    for store in stores:
        known = by_stem.get(_store_stem(store))
        if known is None:
            unknown.append(f"{store.name}: not a known demo output, skipped")
            continue
        if store.name.endswith(".luxar.zarr.zip"):
            skipped.append(f"{store.name}: archive stores are not inspected, skipped")
            continue
        key, declared = known
        targets.append((key, store, declared))
    return targets, skipped, unknown


def _verdict_for_nothing_inspected(
    targets: list[Target],
    skipped: list[str],
    unknown: list[str],
    demos_dir: Path,
    require_scenes: bool,
) -> Optional[int]:
    """The exit code when nothing will be compared, or ``None`` to carry on.

    Three ways a run can inspect nothing, which used to be one silent exit 0
    between them (audit A9-04). They are NOT equivalent, and conflating them is
    what let a typo look like a clean bill of health:

    1. **A named path is not a demo output.** The caller's error — a typo, or an
       output renamed since they wrote the command. Always exit 1. Previously
       this left ``targets`` empty while the skip list stayed populated, so the
       empty-inventory guard never fired, the compare loop ran zero times, and
       the run printed "checked 0 built scene(s); problems: 0" and exited 0.
    2. **Every named store is an archive.** A real demo output this tool cannot
       open. The caller named the right artifact, so the limitation is ours:
       exit 0, but say plainly that nothing was inspected.
    3. **No arguments, empty inventory.** Normal for a fresh checkout and for
       CI, since the output directory is gitignored. Exit 0 with the same
       plain statement.

    ``require_scenes`` escalates 2 and 3 — never 1, which is already an error.
    """
    if unknown:
        aprint(f"❌ {len(unknown)} named path(s) are not demo outputs:")
        for line in unknown:
            aprint(f"  {line}")
        aprint(
            "  Check the spelling against the demo registry — a renamed output "
            "looks exactly like this."
        )
        if not targets:
            if skipped:
                aprint(
                    f"INSPECTED NOTHING — all {len(skipped)} named store(s) "
                    "were skipped:"
                )
                for line in skipped:
                    aprint(f"  {line}")
            return 1

    if targets:
        return None

    if skipped:
        aprint(f"INSPECTED NOTHING — all {len(skipped)} named store(s) were skipped:")
        for line in skipped:
            aprint(f"  {line}")
    elif require_scenes:
        aprint(
            f"❌ --require-scenes: INSPECTED NOTHING — no built demo scene "
            f"found under {demos_dir}."
        )
    else:
        aprint(
            f"no built demo scene found under {demos_dir}; INSPECTED NOTHING "
            "(this is not a pass — build demos, or pass --require-scenes where "
            "they are expected)"
        )
    return 1 if require_scenes else 0


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
    parser.add_argument(
        "--require-scenes",
        action="store_true",
        help=(
            "fail instead of passing when no scene was inspected (use wherever "
            "demos ARE expected to be built: release prep, a demo-build pipeline)"
        ),
    )
    args = parser.parse_args(argv)

    unknown: list[str] = []
    if args.stores:
        targets, skipped, unknown = _explicit_targets(args.stores)
    else:
        targets = list(_demo_stores(args.demos_dir))
        skipped = []

    verdict = _verdict_for_nothing_inspected(
        targets, skipped, unknown, args.demos_dir, args.require_scenes
    )
    if verdict is not None:
        return verdict

    problems = []
    for key, store, declared in targets:
        problem = compare(store, declared)
        if problem:
            problems.append(f"{key}: {store.name} {problem}")

    aprint(
        f"checked {len(targets)} built scene(s); problems: {len(problems)}; "
        f"unrecognised: {len(unknown)}; skipped: {len(skipped)}"
    )
    for line in problems:
        aprint(f"  {line}")
    for line in skipped:
        aprint(f"  {line}")
    return 1 if problems or unknown else 0


if __name__ == "__main__":
    raise SystemExit(main())
