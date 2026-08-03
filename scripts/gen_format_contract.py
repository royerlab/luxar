#!/usr/bin/env python3
"""Generate the cross-language format-contract projections from one YAML source.

``format-contract/contract.yaml`` is the single source of truth for the format
vocabulary shared by the Python writer (``luxar``) and the TypeScript consumer
(``luxar-viewer``): format versions, encoding-scheme names, node types/kinds,
and the canonical attr/array keys. This script mechanically projects it into:

    packages/luxar/src/luxar/typing_utils/_format_contract.py   (constants + Literals)
    packages/luxar-viewer/src/types/format-contract.ts          (const arrays + unions)

Both generated files carry a generated / DO-NOT-EDIT header and are the
sole authority on their own formatting (the Python file is excluded from ruff,
the TS file matches prettier's ``printWidth: 100`` so ``pnpm format`` is a
no-op).

Usage::

    python scripts/gen_format_contract.py            # regenerate both files
    python scripts/gen_format_contract.py --check     # drift gate (CI)

``--check`` regenerates in memory and diffs against the committed files; it
exits 0 when they match, 1 on drift (printing a unified diff), 2 on a read/parse
error.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO = Path(__file__).resolve().parent.parent
CONTRACT = REPO / "format-contract" / "contract.yaml"
PY_OUT = REPO / "packages/luxar/src/luxar/typing_utils/_format_contract.py"
TS_OUT = REPO / "packages/luxar-viewer/src/types/format-contract.ts"

PY_WIDTH = 88
TS_WIDTH = 100

_EDIT_HINT = (
    "edit format-contract/contract.yaml, then run `make gen-contract` "
    "(or `hatch run gen-contract`)"
)


#: Node types that are containers, not element-bearing geometry leaves. These
#: are the members of ``node_types`` that must NOT appear in ``geometry_types``.
_CONTAINER_NODE_TYPES = ("scene", "group")


def _geometry_types(c: Dict[str, Any]) -> List[str]:
    """Return ``geometry_types``, validated as the leaf subset of ``node_types``.

    The two lists are separate keys so the contract can name the leaf-geometry
    vocabulary directly, but that freedom needs guarding — each rule below
    corresponds to a way the generated projections would otherwise be wrong
    rather than merely odd:

    * **non-empty** — an empty list renders ``Literal[]`` (a Python syntax
      error) and ``export type GeometryTypeName = ;`` (a TypeScript one), so
      codegen would emit files that cannot be imported.
    * **subset of node_types** — a geometry type that is not a node type is
      unrepresentable on disk, since the writer stamps ``type`` from that
      vocabulary.
    * **no containers** — ``scene`` / ``group`` pass the subset rule but are not
      element-bearing, and admitting one would hand the viewer's per-geometry
      dispatch a kind that has no loader.
    * **no duplicates** — a repeat silently widens ``GEOMETRY_TYPES`` while
      leaving the union unchanged, desynchronising the two projections of the
      same key.
    """
    geometry_types = list(c["geometry_types"])

    def fail(problem: str) -> None:
        raise SystemExit(f"contract.yaml: geometry_types {problem}")

    if not geometry_types:
        fail("must not be empty (codegen would emit an empty Literal/union)")

    stray = [t for t in geometry_types if t not in c["node_types"]]
    if stray:
        fail(
            f"entries {stray} are missing from node_types; "
            "every geometry type must also be a node type"
        )

    containers = [t for t in geometry_types if t in _CONTAINER_NODE_TYPES]
    if containers:
        fail(
            f"entries {containers} are container node types, not element-bearing "
            "geometry leaves; remove them"
        )

    duplicates = sorted({t for t in geometry_types if geometry_types.count(t) > 1})
    if duplicates:
        fail(f"contains duplicate entries {duplicates}")

    return geometry_types


# --------------------------------------------------------------------------- #
# Python projection
# --------------------------------------------------------------------------- #
def _py_literal_type(name: str, values: List[str]) -> str:
    """Emit ``Name = Literal["a", "b", ...]`` (inline if it fits, else stacked)."""
    inline = f'{name} = Literal[{", ".join(repr_str(v) for v in values)}]'
    if len(inline) <= PY_WIDTH:
        return inline
    body = "".join(f"    {repr_str(v)},\n" for v in values)
    return f"{name} = Literal[\n{body}]"


def _py_tuple(name: str, elem_type: str, values: List[str]) -> str:
    """Emit ``NAME: Final[tuple[T, ...]] = (...)`` (inline if it fits, else stacked)."""
    lhs = f"{name}: Final[tuple[{elem_type}, ...]] = "
    inline = f'{lhs}({", ".join(repr_str(v) for v in values)})'
    if len(values) != 1 and len(inline) <= PY_WIDTH:
        return inline
    if len(values) == 1:
        return f"{lhs}({repr_str(values[0])},)"
    body = "".join(f"    {repr_str(v)},\n" for v in values)
    return f"{lhs}(\n{body})"


def repr_str(value: str) -> str:
    """Deterministic double-quoted string literal (matches ruff/black)."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_python(c: Dict[str, Any]) -> str:
    scene = c["scene_format"]
    gsplats = c["gsplats_format"]
    node_types = list(c["node_types"])
    node_kinds = list(c["node_kinds"])
    geometry_types = _geometry_types(c)
    encodings = list(c["encodings"])
    attr_keys = list(c["attr_keys"])
    array_names = list(c["array_names"])

    parts: List[str] = []
    parts.append(
        "# @generated by scripts/gen_format_contract.py — DO NOT EDIT BY HAND.\n"
        f"# To change the format contract, {_EDIT_HINT};\n"
        "# `hatch run check-contract` gates against drift.\n"
        '"""Generated cross-language format-contract constants (Python side).\n'
        "\n"
        "Single source of truth: ``format-contract/contract.yaml``. This module is a\n"
        "mechanical projection of it — the writer half of the Python <-> TypeScript\n"
        "format contract (the TS half is ``luxar-viewer/src/types/format-contract.ts``).\n"
        '"""\n'
        "\n"
        "from __future__ import annotations\n"
        "\n"
        "from typing import Final, Literal\n"
    )

    parts.append(
        "# --- scene (.luxar.zarr) format version ---\n"
        f"{_py_literal_type('SceneFormatVersion', scene['supported'])}\n"
        f"SCENE_FORMAT_VERSION: Final[SceneFormatVersion] = {repr_str(scene['current'])}\n"
        f"{_py_tuple('SUPPORTED_SCENE_VERSIONS', 'SceneFormatVersion', scene['supported'])}\n"
    )

    parts.append(
        "# --- standalone gsplats (.gsplats.zarr) node-tree format version ---\n"
        f"{_py_literal_type('GSplatsFormatVersion', gsplats['supported'])}\n"
        f"GSPLATS_FORMAT_VERSION: Final[GSplatsFormatVersion] = {repr_str(gsplats['current'])}\n"
        f"{_py_tuple('SUPPORTED_GSPLATS_VERSIONS', 'GSplatsFormatVersion', gsplats['supported'])}\n"
    )

    parts.append(
        "# --- root-header format_type identifying a standalone gsplats store ---\n"
        f"FORMAT_TYPE_GSPLATS: Final[str] = {repr_str(c['format_type_gsplats'])}\n"
    )

    parts.append(
        "# --- scene-graph node types ---\n"
        f"{_py_literal_type('NodeTypeName', node_types)}\n"
        f"{_py_tuple('NODE_TYPES', 'NodeTypeName', node_types)}\n"
    )

    parts.append(
        "# --- leaf geometry types (the element-bearing subset of NODE_TYPES) ---\n"
        f"{_py_literal_type('GeometryTypeName', geometry_types)}\n"
        f"{_py_tuple('GEOMETRY_TYPES', 'GeometryTypeName', geometry_types)}\n"
    )

    parts.append(
        "# --- specialized-group kinds ---\n"
        f"{_py_literal_type('NodeKind', node_kinds)}\n"
        f"{_py_tuple('NODE_KINDS', 'NodeKind', node_kinds)}\n"
    )

    parts.append(
        "# --- on-disk array encoding scheme names ---\n"
        f"{_py_literal_type('EncodingName', encodings)}\n"
        f"{_py_tuple('ENCODING_NAMES', 'EncodingName', encodings)}\n"
    )

    parts.append(
        "# --- canonical metadata attribute keys ---\n"
        f"{_py_literal_type('AttrKey', attr_keys)}\n"
        f"{_py_tuple('ATTR_KEYS', 'AttrKey', attr_keys)}\n"
    )

    parts.append(
        "# --- canonical gsplats array names ---\n"
        f"{_py_literal_type('ArrayName', array_names)}\n"
        f"{_py_tuple('ARRAY_NAMES', 'ArrayName', array_names)}\n"
    )

    exports = [
        "SceneFormatVersion",
        "SCENE_FORMAT_VERSION",
        "SUPPORTED_SCENE_VERSIONS",
        "GSplatsFormatVersion",
        "GSPLATS_FORMAT_VERSION",
        "SUPPORTED_GSPLATS_VERSIONS",
        "FORMAT_TYPE_GSPLATS",
        "NodeTypeName",
        "NODE_TYPES",
        "NodeKind",
        "NODE_KINDS",
        "EncodingName",
        "ENCODING_NAMES",
        "AttrKey",
        "ATTR_KEYS",
        "ArrayName",
        "ARRAY_NAMES",
    ]
    all_body = "".join(f"    {repr_str(name)},\n" for name in exports)
    parts.append(f"__all__ = [\n{all_body}]\n")

    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# TypeScript projection (prettier printWidth=100, singleQuote, es5 commas)
# --------------------------------------------------------------------------- #
def _ts_str(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _ts_const(name: str, values: List[str], elem_type: str = "string") -> str:
    """Emit ``export const NAME: readonly T[] = [...]``.

    ``elem_type`` mirrors :func:`_py_tuple`'s parameter of the same name. Pass
    the paired union alias so callers can iterate the array *and* index a
    record keyed by it — with the default ``string`` they cannot, and every
    such call site ends up re-declaring the literals locally.

    Left as ``string`` on purpose for the ``SUPPORTED_*_VERSIONS`` allowlists:
    those are matched against untrusted values read off disk, and
    ``readonly T[].includes(someString)`` is a type error in TypeScript. Python
    can narrow the same tuples because ``x in tup`` is not type-checked there.
    """
    lhs = f"export const {name}: readonly {elem_type}[] = "
    inline = f'{lhs}[{", ".join(_ts_str(v) for v in values)}];'
    if len(inline) <= TS_WIDTH:
        return inline
    body = "".join(f"  {_ts_str(v)},\n" for v in values)
    return f"{lhs}[\n{body}];"


def _ts_union(name: str, values: List[str]) -> str:
    inline = f'export type {name} = {" | ".join(_ts_str(v) for v in values)};'
    if len(inline) <= TS_WIDTH:
        return inline
    body = "".join(f"  | {_ts_str(v)}\n" for v in values)
    # Replace the final newline with a semicolon terminator.
    return f"export type {name} =\n{body.rstrip()};"


def render_typescript(c: Dict[str, Any]) -> str:
    scene = c["scene_format"]
    gsplats = c["gsplats_format"]
    node_types = list(c["node_types"])
    node_kinds = list(c["node_kinds"])
    geometry_types = _geometry_types(c)
    encodings = list(c["encodings"])
    attr_keys = list(c["attr_keys"])
    array_names = list(c["array_names"])

    header = (
        "/**\n"
        " * @generated by scripts/gen_format_contract.py — DO NOT EDIT BY HAND.\n"
        f" * To change the format contract, {_EDIT_HINT};\n"
        " * `hatch run check-contract` gates against drift.\n"
        " *\n"
        " * Generated cross-language format-contract constants (TypeScript side).\n"
        " * Single source of truth: format-contract/contract.yaml. This is the\n"
        " * consumer half of the Python <-> TypeScript format contract (the writer\n"
        " * half is luxar/src/luxar/typing_utils/_format_contract.py).\n"
        " */\n"
    )

    blocks = [
        "// --- scene (.luxar.zarr) format version ---\n"
        f"export const SCENE_FORMAT_VERSION = {_ts_str(scene['current'])};\n"
        f"{_ts_const('SUPPORTED_SCENE_VERSIONS', scene['supported'])}\n"
        f"{_ts_union('SceneFormatVersion', scene['supported'])}",
        "// --- standalone gsplats (.gsplats.zarr) node-tree format version ---\n"
        f"export const GSPLATS_FORMAT_VERSION = {_ts_str(gsplats['current'])};\n"
        f"{_ts_const('SUPPORTED_GSPLATS_FORMAT_VERSIONS', gsplats['supported'])}\n"
        f"{_ts_union('GSplatsFormatVersion', gsplats['supported'])}",
        "// --- root-header format_type identifying a standalone gsplats store ---\n"
        f"export const FORMAT_TYPE_GSPLATS = {_ts_str(c['format_type_gsplats'])};",
        "// --- scene-graph node types ---\n"
        f"{_ts_const('NODE_TYPES', node_types, 'NodeTypeName')}\n"
        f"{_ts_union('NodeTypeName', node_types)}",
        "// --- leaf geometry types (the element-bearing subset of NODE_TYPES) ---\n"
        f"{_ts_const('GEOMETRY_TYPES', geometry_types, 'GeometryTypeName')}\n"
        f"{_ts_union('GeometryTypeName', geometry_types)}",
        "// --- specialized-group kinds ---\n"
        f"{_ts_const('NODE_KINDS', node_kinds, 'NodeKind')}\n"
        f"{_ts_union('NodeKind', node_kinds)}",
        "// --- on-disk array encoding scheme names ---\n"
        f"{_ts_const('ENCODING_NAMES', encodings, 'EncodingName')}\n"
        f"{_ts_union('EncodingName', encodings)}",
        "// --- canonical metadata attribute keys ---\n"
        f"{_ts_const('ATTR_KEYS', attr_keys, 'AttrKey')}\n"
        f"{_ts_union('AttrKey', attr_keys)}",
        "// --- canonical gsplats array names ---\n"
        f"{_ts_const('ARRAY_NAMES', array_names, 'ArrayName')}\n"
        f"{_ts_union('ArrayName', array_names)}",
    ]

    return header + "\n" + "\n\n".join(blocks) + "\n"


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def load_contract() -> Dict[str, Any]:
    data = yaml.safe_load(CONTRACT.read_text())
    if not isinstance(data, dict):
        raise TypeError(f"{CONTRACT} must contain a top-level mapping")
    return data


def _diff(path: Path, expected: str) -> List[str]:
    actual = path.read_text() if path.exists() else ""
    if actual == expected:
        return []
    return list(
        difflib.unified_diff(
            actual.splitlines(keepends=True),
            expected.splitlines(keepends=True),
            fromfile=f"{path} (committed)",
            tofile=f"{path} (regenerated)",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed files match the contract (no writes); "
        "exit 1 on drift",
    )
    args = parser.parse_args()

    try:
        contract = load_contract()
    except (OSError, yaml.YAMLError) as exc:
        print(f"error: cannot read {CONTRACT}: {exc}", file=sys.stderr)
        return 2

    outputs = {
        PY_OUT: render_python(contract),
        TS_OUT: render_typescript(contract),
    }

    if args.check:
        drifted = False
        for path, expected in outputs.items():
            diff = _diff(path, expected)
            if diff:
                drifted = True
                print(f"drift: {path.relative_to(REPO)}", file=sys.stderr)
                sys.stderr.writelines(diff)
        if drifted:
            print(
                f"\nformat-contract projections are stale. To fix, {_EDIT_HINT} "
                "and commit the regenerated files.",
                file=sys.stderr,
            )
            return 1
        print("format contract: Python and TypeScript projections are in sync")
        return 0

    for path, expected in outputs.items():
        path.write_text(expected)
        print(f"wrote {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
