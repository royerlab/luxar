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
   matches nothing, so paid-down debt cannot linger; a committed high-water mark
   prevents new debt from being waved through by appending a matching line.

    python scripts/check_layer_order.py            # gate (exit 1 on regression)
    python scripts/check_layer_order.py --report   # scores + the violating edges

There is no `--update`: `MAX_DEBT` is a ratchet, changed only in review. Pay a
violation down by deleting the code edge and its ignore entry, then lower the
constant; the live equality check also keeps the list consistent with the graph.

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
from typing import Any, NoReturn, TypeAlias, cast

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
CONTRACT_NAME = "Subpackage layering"
MAX_DEBT = 35

ModuleEdge: TypeAlias = tuple[str, str]
LayerPair: TypeAlias = tuple[str, str]
EdgeMap: TypeAlias = dict[LayerPair, list[ModuleEdge]]
BadEdges: TypeAlias = dict[LayerPair, int]
RankedOrders: TypeAlias = list[tuple[int, list[str]]]

#: Packages whose position is not in question, so they are held fixed while the
#: rest are permuted. `cli`/`demos` sit at the top on a wide margin (201 and 343
#: outgoing edges against 2 and 6 incoming); the others import nothing internal
#: at all, so no ordering among them can be violated.
FIXED_TOP = ("cli", "demos")
FIXED_BOTTOM = ("colormaps", "mesh", "shading", "_zarr_compat", "_process")


def _fail(message: str) -> NoReturn:
    print(f"❌ {message}", file=sys.stderr)
    raise SystemExit(1)


def _importlinter_config() -> dict[str, Any]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    config = data.get("tool", {}).get("importlinter", {})
    if not isinstance(config, dict):
        _fail("tool.importlinter must be a TOML table")
    return cast(dict[str, Any], config)


def _contract() -> dict[str, Any]:
    for contract in _importlinter_config().get("contracts", []):
        if isinstance(contract, dict) and contract.get("name") == CONTRACT_NAME:
            return cast(dict[str, Any], contract)
    _fail(f"no contract named {CONTRACT_NAME!r} in pyproject.toml")
    raise AssertionError("unreachable")


def _declared_order() -> list[str]:
    layers = _contract().get("layers")
    if not isinstance(layers, list) or not layers:
        _fail(f"contract {CONTRACT_NAME!r} declares no layers")
    declared = []
    for layer in layers:
        if not isinstance(layer, str) or not layer.startswith("luxar."):
            _fail(f"contract {CONTRACT_NAME!r} has an invalid layer name")
        declared.append(layer.split(".", 1)[1])
    return declared


def _subpackage(module: str) -> str | None:
    parts = module.split(".")
    return parts[1] if len(parts) >= 2 else None


def _production_ignores() -> list[str]:
    """Concrete ignored edges whose endpoints are both declared layers."""
    declared = set(_declared_order())
    ignores = []
    for entry in _contract().get("ignore_imports", []):
        if not isinstance(entry, str) or ".tests." in entry or "*" in entry:
            continue
        importer, imported = entry.split(" -> ", 1)
        if _subpackage(importer) in declared and _subpackage(imported) in declared:
            ignores.append(entry)
    return ignores


def _exclude_type_checking_imports() -> bool:
    value = _importlinter_config().get("exclude_type_checking_imports", False)
    if not isinstance(value, bool):
        _fail("tool.importlinter.exclude_type_checking_imports must be a boolean")
    return value


def _edges() -> EdgeMap:
    """Production subpackage->subpackage edges, as import-linter sees them."""
    try:
        import grimp
    except ImportError:  # pragma: no cover - grimp ships with import-linter
        _fail("grimp is not installed; `pip install import-linter`")

    graph = grimp.build_graph(
        "luxar",
        include_external_packages=False,
        exclude_type_checking_imports=_exclude_type_checking_imports(),
    )

    declared = set(_declared_order())
    discovered = {
        package
        for module in graph.modules
        if (package := _subpackage(module)) is not None
        and package not in {"__main__", "conftest", "tests"}
    }
    if unknown := sorted(discovered - declared):
        _fail(
            "subpackage(s) missing from the layering contract: "
            + ", ".join(f"luxar.{package}" for package in unknown)
        )

    out: EdgeMap = defaultdict(list)
    for importer in graph.modules:
        src = _subpackage(importer)
        if src is None or ".tests" in importer or importer.endswith(".tests"):
            continue
        for imported in graph.find_modules_directly_imported_by(importer):
            dst = _subpackage(imported)
            if dst is None or dst == src or ".tests" in imported:
                continue
            out[(src, dst)].append((importer, imported))
    if not out:
        _fail("the dependency graph came back empty — nothing was scanned")
    return out


def _score(order: list[str], edges: EdgeMap) -> tuple[int, BadEdges]:
    rank = {name: i for i, name in enumerate(order)}  # 0 = highest layer
    bad: BadEdges = {}
    for (src, dst), es in edges.items():
        if src in rank and dst in rank and rank[src] > rank[dst]:
            bad[(src, dst)] = len(es)
    return sum(bad.values()), bad


def _rank_orders(declared: list[str], edges: EdgeMap) -> RankedOrders:
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


def _print_report(
    declared: list[str],
    edges: EdgeMap,
    total: int,
    bad: BadEdges,
    ranked: RankedOrders,
) -> None:
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


def _order_problem(total: int, ranked: RankedOrders) -> str | None:
    """The declared order is no longer the cheapest one available."""
    if total <= ranked[0][0]:
        return None
    return (
        f"the declared layer order costs {total} violating edges, but "
        f"{ranked[0][0]} is achievable with {' > '.join(ranked[0][1])}. Reorder "
        f"the contract's `layers`, or change the imports until the declared order "
        f"is optimal; a comment alone does not override this gate."
    )


def _debt_problem(total: int, ignores: list[str]) -> str | None:
    """The dated debt list disagrees with the violations actually present."""
    if len(ignores) > MAX_DEBT:
        return (
            f"{len(ignores)} production `ignore_imports` entries exceeds the "
            f"committed maximum of {MAX_DEBT}. New layering debt must be fixed, "
            f"not added to the list."
        )
    if len(ignores) > total:
        stale = len(ignores) - total
        return (
            f"{len(ignores)} production `ignore_imports` entries for {total} actual "
            f"violations -- {stale} named edge{'s' if stale != 1 else ''} no longer "
            f"{'exist' if stale != 1 else 'exists'}. "
            f"Delete them; import-linter refuses an ignore that matches nothing."
        )
    if len(ignores) < total:
        return (
            f"{total} production layering violations but only {len(ignores)} are "
            f"listed in `ignore_imports`. A NEW violation must be fixed, not "
            f"appended to the debt list -- that list is dated and only shrinks. "
            f"Run `--report` to see every edge."
        )
    if len(ignores) < MAX_DEBT:
        return (
            f"layering debt fell to {len(ignores)} entries; lower `MAX_DEBT` from "
            f"{MAX_DEBT} to preserve the ratchet."
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
