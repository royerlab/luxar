#!/usr/bin/env python3
"""Keep the `Subpackage layering` contract honest.

import-linter enforces the layer order it is given. It cannot tell you whether
that order is still the right one, and it cannot stop someone from "fixing" a
new violation by appending to `ignore_imports`. This script covers both:

1. **The declared order is still the measured minimum.** Every ordering of the
   ambiguous middle packages is scored by how many production edges point UP the
   stack. If a structural change makes a different order strictly better, that is
   worth knowing — either the code moved toward a different architecture, or the
   change was in the wrong direction.

2. **The debt list only shrinks.** The contract's `ignore_imports` are dated
   layering violations, not design. import-linter already refuses an entry that
   matches nothing, so paid-down debt cannot linger; this adds the other
   direction, so new debt cannot be waved through by appending a line.

    python scripts/check_layer_order.py            # gate (exit 1 on regression)
    python scripts/check_layer_order.py --report   # scores + the violating edges

There is no `--update`: the expected debt count is DERIVED from the contract's
own `ignore_imports`, not stored in a baseline file, so there is nothing to
re-baseline. Pay a violation down by deleting the code edge and its ignore
entry; both directions are then checked automatically.

Fails closed: a graph it cannot build, or a contract it cannot find, is an error
rather than a pass.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
CONTRACT_NAME = "Subpackage layering"

#: Packages whose position is not in question, so they are held fixed while the
#: rest are permuted. `cli`/`demos` sit at the top on a wide margin (201 and 343
#: outgoing edges against 2 and 6 incoming); the others import nothing internal
#: at all, so no ordering among them can be violated.
FIXED_TOP = ("cli", "demos")
FIXED_BOTTOM = ("colormaps", "mesh", "shading", "_zarr_compat", "_process")


def _fail(message: str) -> None:
    print(f"❌ {message}", file=sys.stderr)
    raise SystemExit(1)


def _contract() -> dict:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    for contract in data.get("tool", {}).get("importlinter", {}).get("contracts", []):
        if contract.get("name") == CONTRACT_NAME:
            return contract
    _fail(f"no contract named {CONTRACT_NAME!r} in pyproject.toml")
    raise AssertionError("unreachable")


def _declared_order() -> list[str]:
    layers = _contract().get("layers")
    if not layers:
        _fail(f"contract {CONTRACT_NAME!r} declares no layers")
    return [layer.split(".", 1)[1] for layer in layers]


def _production_ignores() -> list[str]:
    """`ignore_imports` entries naming concrete modules, not test globs."""
    return [
        entry
        for entry in _contract().get("ignore_imports", [])
        if ".tests." not in entry and "*" not in entry
    ]


def _edges() -> dict[tuple[str, str], list[tuple[str, str]]]:
    """Production subpackage->subpackage edges, as import-linter sees them."""
    try:
        import grimp
    except ImportError:  # pragma: no cover - grimp ships with import-linter
        _fail("grimp is not installed; `pip install import-linter`")

    # MUST match `exclude_type_checking_imports` in pyproject, or this scores a
    # graph no contract will ever enforce.
    graph = grimp.build_graph(
        "luxar", include_external_packages=False, exclude_type_checking_imports=True
    )

    def subpackage(module: str) -> str | None:
        parts = module.split(".")
        return parts[1] if len(parts) >= 2 else None

    out: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for importer in graph.modules:
        src = subpackage(importer)
        if src is None or ".tests" in importer or importer.endswith(".tests"):
            continue
        for imported in graph.find_modules_directly_imported_by(importer):
            dst = subpackage(imported)
            if dst is None or dst == src or ".tests" in imported:
                continue
            out[(src, dst)].append((importer, imported))
    if not out:
        _fail("the dependency graph came back empty — nothing was scanned")
    return out


def _score(order: list[str], edges) -> tuple[int, dict[tuple[str, str], int]]:
    rank = {name: i for i, name in enumerate(order)}  # 0 = highest layer
    bad: dict[tuple[str, str], int] = {}
    for (src, dst), es in edges.items():
        if src in rank and dst in rank and rank[src] > rank[dst]:
            bad[(src, dst)] = len(es)
    return sum(bad.values()), bad


def _rank_orders(declared: list[str], edges) -> list[tuple[int, list[str]]]:
    middle = [p for p in declared if p not in FIXED_TOP and p not in FIXED_BOTTOM]
    ranked = [
        (
            _score(list(FIXED_TOP) + list(perm) + list(FIXED_BOTTOM), edges)[0],
            list(perm),
        )
        for perm in itertools.permutations(middle)
    ]
    ranked.sort(key=lambda pair: pair[0])
    return ranked


def _print_report(declared, edges, total, bad, ranked) -> None:
    """Human-readable scores and the exact edges behind them."""
    middle = [p for p in declared if p not in FIXED_TOP and p not in FIXED_BOTTOM]
    print(f"declared order ({len(declared)} layers), {total} violating edges:\n")
    for i, name in enumerate(declared):
        print(f"  {i:>2}. luxar.{name}")
    print(f"\nbest of {len(ranked)} middle orderings: {ranked[0][0]} edges")
    for score, perm in ranked[:5]:
        marker = " <- declared" if perm == middle else ""
        print(f"  {score:>4}   {' > '.join(perm)}{marker}")
    print("\nviolating edges by pair:")
    for (src, dst), n in sorted(bad.items(), key=lambda kv: -kv[1]):
        print(f"\n  === {src} -> {dst} ({n}) ===")
        for importer, imported in sorted(edges[(src, dst)]):
            print(f"    {importer} -> {imported}")


def _order_problem(total: int, ranked) -> str | None:
    """The declared order is no longer the cheapest one available."""
    if total <= ranked[0][0]:
        return None
    return (
        f"the declared layer order costs {total} violating edges, but "
        f"{ranked[0][0]} is achievable with {' > '.join(ranked[0][1])}. Either "
        f"reorder the contract's `layers` to match, or -- if the declared order "
        f"is deliberate -- say why in the contract comment. The order is an "
        f"architectural claim, not a score."
    )


def _debt_problem(total: int, ignores: list[str]) -> str | None:
    """The dated debt list disagrees with the violations actually present."""
    if len(ignores) > total:
        return (
            f"{len(ignores)} production `ignore_imports` entries for {total} actual "
            f"violations -- {len(ignores) - total} name edges that no longer exist. "
            f"Delete them; import-linter refuses an ignore that matches nothing."
        )
    if len(ignores) < total:
        return (
            f"{total} production layering violations but only {len(ignores)} are "
            f"listed in `ignore_imports`. A NEW violation must be fixed, not "
            f"appended to the debt list -- that list is dated and only shrinks. "
            f"Run `--report` to see every edge."
        )
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="print scores and edges")
    args = parser.parse_args(argv)

    declared = _declared_order()
    edges = _edges()
    total, bad = _score(declared, edges)
    ranked = _rank_orders(declared, edges)

    if args.report:
        _print_report(declared, edges, total, bad, ranked)
        return 0

    ignores = _production_ignores()
    problems = [
        problem
        for problem in (_order_problem(total, ranked), _debt_problem(total, ignores))
        if problem
    ]
    if problems:
        for problem in problems:
            print(f"❌ {problem}", file=sys.stderr)
        return 1

    print(
        f"✅ layer order is the measured optimum ({total} violating edges), and "
        f"all {len(ignores)} are declared debt."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
