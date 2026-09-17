#!/usr/bin/env python3
"""Generate the cross-language format-contract projections from one YAML source.

``format-contract/contract.yaml`` is the single source of truth for the format
vocabulary shared by the Python writer (``luxar``) and the TypeScript consumer
(``@luxar/viewer``): format versions, encoding-scheme names, node types/kinds,
and the canonical attr/array keys. This script mechanically projects it into:

    packages/luxar/src/luxar/typing_utils/_format_contract.py   (constants + Literals)
    packages/luxar-viewer/src/types/format-contract.ts          (const arrays + unions)

Both generated files carry a generated / DO-NOT-EDIT header and are the
sole authority on their own formatting (the Python file is excluded from ruff,
the TS file matches prettier's ``printWidth: 100`` so ``pnpm format`` is a
no-op).

The projection is TABLE-DRIVEN: every contract key is one row of
:data:`VERSION_BLOCKS`, :data:`SCALARS` or :data:`VOCABULARIES`, naming the
Python / TypeScript symbols it renders to, the doc comments both projections
carry, and the validator that must accept it before anything is written. Adding
a vocabulary to the contract is therefore one YAML key plus one table row; the
renderers themselves never change.

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
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, NoReturn

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

#: Shape every format version string must have: ``MAJOR.MINOR`` with no
#: patch component. The version-check policy (``typing_utils/format_version.py``
#: and the viewer's ``data/format-version.ts``) parses exactly this shape, so a
#: contract entry that does not match it would be refused by every reader.
_FORMAT_VERSION_RE = re.compile(r"^\d+\.\d+$")


# --------------------------------------------------------------------------- #
# Validators
# --------------------------------------------------------------------------- #
def _fail(key: str, problem: str) -> NoReturn:
    raise SystemExit(f"contract.yaml: {key} {problem}")


def _unique_nonempty(key: str) -> Callable[[Dict[str, Any]], List[str]]:
    """Return a validator for a plain list key: non-empty, duplicate-free.

    Every vocabulary list is projected to a Python ``Literal[...]`` and a
    TypeScript union, both of which are syntax errors when empty; a duplicate
    silently widens the tuple/array while leaving the union unchanged,
    desynchronising the two projections of one key.
    """

    def validate(c: Dict[str, Any]) -> List[str]:
        raw = c.get(key)
        if not isinstance(raw, list):
            _fail(key, "must be a list")
        values = [str(v) for v in raw]
        if not values:
            _fail(key, "must not be empty (codegen would emit an empty Literal/union)")
        duplicates = sorted({v for v in values if values.count(v) > 1})
        if duplicates:
            _fail(key, f"contains duplicate entries {duplicates}")
        return values

    return validate


def _format_versions(key: str) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Return a validator for a ``{current, supported}`` version block.

    * every entry (``current`` and each ``supported`` member) matches
      ``MAJOR.MINOR`` — the only shape the version-check policy parses;
    * ``supported`` is non-empty and duplicate-free (it renders to a Literal);
    * ``current`` is a member of ``supported`` — the writer must be able to
      read what it writes.
    """

    def validate(c: Dict[str, Any]) -> Dict[str, Any]:
        block = c.get(key)
        if not isinstance(block, dict):
            _fail(key, "must be a mapping with `current` and `supported`")
        current = str(block.get("current", ""))
        supported = _unique_nonempty("supported")({"supported": block.get("supported")})
        malformed = [
            v for v in [current, *supported] if not _FORMAT_VERSION_RE.match(v)
        ]
        if malformed:
            _fail(
                key,
                f"entries {malformed} are not MAJOR.MINOR version strings "
                "(the version-check policy parses exactly that shape)",
            )
        if current not in supported:
            _fail(key, f"current {current!r} is not in supported {supported}")
        return {"current": current, "supported": supported}

    return validate


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


def _loader_types(c: Dict[str, Any]) -> List[str]:
    """Return ``loader_types``, validated as the viewer-drawable subset.

    Where ``geometry_types`` answers *"is this node a geometry leaf?"* (a
    vocabulary), this answers *"can the viewer load and draw it?"* (a
    capability). A type is writable the moment the Python side can emit it, but
    drawable only once it has a loader, a ``GEOMETRY_DESCRIPTORS`` row, a
    ``PARTIAL_EXTEND_TOLERANCE`` row and a hidden-dim tolerance arm — so the two
    lists are allowed to differ, and each consumer must pick the one matching
    its question. Keeping them as one list forces a wrong answer to one of the
    two.

    The rules mirror :func:`_geometry_types`, minus the container check (already
    guaranteed transitively by the subset rule):

    * **non-empty** — an empty list renders ``export type LoaderTypeName = ;``.
    * **subset of geometry_types** — the viewer cannot dispatch on a type the
      writer has no vocabulary for; such an entry is unrepresentable on disk.
    * **no duplicates** — a repeat widens ``LOADER_TYPES`` while leaving the
      union unchanged, desynchronising the two projections of one key.
    """
    loader_types = list(c["loader_types"])
    geometry_types = list(c["geometry_types"])

    def fail(problem: str) -> None:
        raise SystemExit(f"contract.yaml: loader_types {problem}")

    if not loader_types:
        fail("must not be empty (codegen would emit an empty union)")

    stray = [t for t in loader_types if t not in geometry_types]
    if stray:
        fail(
            f"entries {stray} are missing from geometry_types; the viewer cannot "
            "dispatch on a type the writer has no vocabulary for"
        )

    duplicates = sorted({t for t in loader_types if loader_types.count(t) > 1})
    if duplicates:
        fail(f"contains duplicate entries {duplicates}")

    return loader_types


def _subset_of(key: str, superset_key: str) -> Callable[[Dict[str, Any]], List[str]]:
    """Return a validator for a list that must be ``_unique_nonempty`` AND a
    subset of another contract list (``key ⊆ superset_key``)."""

    def validate(c: Dict[str, Any]) -> List[str]:
        values = _unique_nonempty(key)(c)
        superset = [str(v) for v in c.get(superset_key, [])]
        stray = [v for v in values if v not in superset]
        if stray:
            _fail(key, f"entries {stray} are missing from {superset_key}")
        return values

    return validate


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VersionBlock:
    """A ``{current, supported}`` format-version block."""

    yaml_key: str
    py_type: str  # Literal alias over `supported`
    py_current: str  # Final[py_type] = current
    py_supported: str  # Final[tuple[py_type, ...]] = supported
    py_comment: str  # the `# --- ... ---` banner
    ts_current: str
    ts_supported: str
    ts_type: str
    ts_current_doc: str
    ts_supported_doc: str
    ts_type_doc: str


@dataclass(frozen=True)
class Scalar:
    """A single string value (an attr key or a well-known attr value)."""

    yaml_key: str
    py_name: str
    py_comment: str
    ts_name: str
    ts_doc: str


@dataclass(frozen=True)
class Vocabulary:
    """A list of names rendered to a Literal/union plus a tuple/const array."""

    yaml_key: str
    py_type: str
    py_const: str
    py_comment: str
    ts_const: str
    ts_type: str
    ts_const_doc: str
    ts_type_doc: str
    validate: Callable[[Dict[str, Any]], List[str]]


VERSION_BLOCKS: tuple[VersionBlock, ...] = (
    VersionBlock(
        yaml_key="scene_format",
        py_type="SceneFormatVersion",
        py_current="SCENE_FORMAT_VERSION",
        py_supported="SUPPORTED_SCENE_VERSIONS",
        py_comment="scene (.luxar.zarr) format version",
        ts_current="SCENE_FORMAT_VERSION",
        ts_supported="SUPPORTED_SCENE_VERSIONS",
        ts_type="SceneFormatVersion",
        ts_current_doc=(
            "Current `.luxar.zarr` scene format version — the version the Python "
            "writer emits and this build treats as current."
        ),
        ts_supported_doc="Scene format versions declared loadable by this build.",
        ts_type_doc="Union of the supported scene format version strings.",
    ),
    VersionBlock(
        yaml_key="gsplats_format",
        py_type="GSplatsFormatVersion",
        py_current="GSPLATS_FORMAT_VERSION",
        py_supported="SUPPORTED_GSPLATS_VERSIONS",
        py_comment="standalone gsplats (.gsplats.zarr) node-tree format version",
        ts_current="GSPLATS_FORMAT_VERSION",
        ts_supported="SUPPORTED_GSPLATS_FORMAT_VERSIONS",
        ts_type="GSplatsFormatVersion",
        ts_current_doc=(
            "Current standalone `.gsplats.zarr` node-tree format version — the "
            "version the Python writer emits and this build treats as current."
        ),
        ts_supported_doc=(
            "Standalone gsplats node-tree format versions this build can load; the "
            "loader matches a store on-disk format_version against this allowlist."
        ),
        ts_type_doc="Union of the supported standalone gsplats format version strings.",
    ),
)

SCALARS: tuple[Scalar, ...] = (
    Scalar(
        yaml_key="format_type_gsplats",
        py_name="FORMAT_TYPE_GSPLATS",
        py_comment="root-header format_type identifying a standalone gsplats store",
        ts_name="FORMAT_TYPE_GSPLATS",
        ts_doc="Root-header `format_type` value identifying a standalone gsplats store.",
    ),
    Scalar(
        yaml_key="format_type_scene",
        py_name="FORMAT_TYPE_SCENE",
        py_comment="root-header format_type identifying a compiled scene (0.2+)",
        ts_name="FORMAT_TYPE_SCENE",
        ts_doc=(
            "Root-header `format_type` value identifying a compiled `.luxar.zarr` "
            "scene (format 0.2+)."
        ),
    ),
    Scalar(
        yaml_key="legacy_scene_version_attr",
        py_name="LEGACY_SCENE_VERSION_ATTR",
        py_comment=(
            "scene 0.1 root-header version key (read fallback only; never written)"
        ),
        ts_name="LEGACY_SCENE_VERSION_ATTR",
        ts_doc=(
            "The scene 0.1 root-header version key. Readers fall back to it when\n"
            "`format_version` is absent; the writer no longer emits it."
        ),
    ),
    Scalar(
        yaml_key="software_version_attr",
        py_name="SOFTWARE_VERSION_ATTR",
        py_comment=(
            "root-header attr recording the writing luxar.__version__ (provenance\n"
            "#     only; excluded from content_hash by both hashers)"
        ),
        ts_name="SOFTWARE_VERSION_ATTR",
        ts_doc=(
            "Root-header attr recording the Luxar SOFTWARE version that wrote a\n"
            "scene. Provenance only — excluded from `content_hash`, so it never\n"
            "invalidates a viewer cache."
        ),
    ),
    Scalar(
        yaml_key="nd_transform_permutation_key",
        py_name="ND_TRANSFORM_PERMUTATION_KEY",
        py_comment="the one key a categorical nd_transform entry carries",
        ts_name="ND_TRANSFORM_PERMUTATION_KEY",
        ts_doc=(
            "The one key a categorical `nd_transform` entry carries; an affine entry\n"
            "carries `ND_TRANSFORM_AFFINE_KEYS` instead (mutually exclusive)."
        ),
    ),
)

_LOADER_DOC = (
    "Viewer-drawable geometry types: the subset of `GEOMETRY_TYPES` that has a\n"
    "loader, a `GEOMETRY_DESCRIPTORS` row, a `PARTIAL_EXTEND_TOLERANCE` arm and a\n"
    "hidden-dim tolerance arm. A CAPABILITY, not the vocabulary — key dispatch\n"
    "tables on this, not on `GeometryTypeName`, so a not-yet-drawable type cannot\n"
    "resolve to no loader. See contract.yaml::loader_types."
)

VOCABULARIES: tuple[Vocabulary, ...] = (
    Vocabulary(
        yaml_key="node_types",
        py_type="NodeTypeName",
        py_const="NODE_TYPES",
        py_comment="scene-graph node types",
        ts_const="NODE_TYPES",
        ts_type="NodeTypeName",
        ts_const_doc="All scene-graph node type names.",
        ts_type_doc="Union of the scene-graph node type names.",
        validate=_unique_nonempty("node_types"),
    ),
    Vocabulary(
        yaml_key="geometry_types",
        py_type="GeometryTypeName",
        py_const="GEOMETRY_TYPES",
        py_comment="leaf geometry types (the element-bearing subset of NODE_TYPES)",
        ts_const="GEOMETRY_TYPES",
        ts_type="GeometryTypeName",
        ts_const_doc="Leaf geometry types — the element-bearing subset of NODE_TYPES.",
        ts_type_doc="Union of the leaf geometry type names.",
        validate=_geometry_types,
    ),
    Vocabulary(
        yaml_key="loader_types",
        py_type="LoaderTypeName",
        py_const="LOADER_TYPES",
        py_comment=(
            "viewer-drawable geometry types (the subset of GEOMETRY_TYPES with a\n"
            "#     loader / descriptor row / tolerance arm; a CAPABILITY, not the\n"
            "#     vocabulary — see contract.yaml::loader_types)"
        ),
        ts_const="LOADER_TYPES",
        ts_type="LoaderTypeName",
        ts_const_doc=_LOADER_DOC,
        ts_type_doc="Union of the viewer-drawable geometry type names.",
        validate=_loader_types,
    ),
    Vocabulary(
        yaml_key="node_kinds",
        py_type="NodeKind",
        py_const="NODE_KINDS",
        py_comment="specialized-group kinds",
        ts_const="NODE_KINDS",
        ts_type="NodeKind",
        ts_const_doc="Specialized-group kinds (lod, partition).",
        ts_type_doc="Union of the specialized-group kind names.",
        validate=_unique_nonempty("node_kinds"),
    ),
    Vocabulary(
        yaml_key="encodings",
        py_type="EncodingName",
        py_const="ENCODING_NAMES",
        py_comment="on-disk array encoding scheme names",
        ts_const="ENCODING_NAMES",
        ts_type="EncodingName",
        ts_const_doc="All on-disk array encoding scheme names.",
        ts_type_doc="Union of the on-disk array encoding scheme names.",
        validate=_unique_nonempty("encodings"),
    ),
    Vocabulary(
        yaml_key="attr_keys",
        py_type="AttrKey",
        py_const="ATTR_KEYS",
        py_comment="canonical metadata attribute keys",
        ts_const="ATTR_KEYS",
        ts_type="AttrKey",
        ts_const_doc="Canonical metadata attribute keys.",
        ts_type_doc="Union of the canonical metadata attribute key names.",
        validate=_unique_nonempty("attr_keys"),
    ),
    Vocabulary(
        yaml_key="array_names",
        py_type="ArrayName",
        py_const="ARRAY_NAMES",
        py_comment="canonical gsplats array names",
        ts_const="ARRAY_NAMES",
        ts_type="ArrayName",
        ts_const_doc="Canonical gsplats array names.",
        ts_type_doc="Union of the canonical gsplats array names.",
        validate=_unique_nonempty("array_names"),
    ),
    Vocabulary(
        yaml_key="render_attr_keys",
        py_type="RenderAttrKey",
        py_const="RENDER_ATTR_KEYS",
        py_comment=(
            "render / appearance attribute keys (the writer's typo allowlist,\n"
            "#     validation/writing.py::KNOWN_RENDER_ATTRS)"
        ),
        ts_const="RENDER_ATTR_KEYS",
        ts_type="RenderAttrKey",
        ts_const_doc=(
            "Render / appearance attribute keys a caller may author on at least one\n"
            "node type — the Python writer's typo allowlist. `ComposableAttrs` keys\n"
            "must be a subset."
        ),
        ts_type_doc="Union of the render / appearance attribute key names.",
        validate=_unique_nonempty("render_attr_keys"),
    ),
    Vocabulary(
        yaml_key="lod_selectors",
        py_type="LodSelectorName",
        py_const="LOD_SELECTORS",
        py_comment="kind=lod `selector` attr values (units of coverage_fraction)",
        ts_const="LOD_SELECTORS",
        ts_type="LodSelectorName",
        ts_const_doc=(
            "`selector` values a kind=lod group may carry — the units of its\n"
            "children's `coverage_fraction` thresholds."
        ),
        ts_type_doc="Union of the kind=lod `selector` attr values.",
        validate=_unique_nonempty("lod_selectors"),
    ),
    Vocabulary(
        yaml_key="blending_modes",
        py_type="BlendingModeName",
        py_const="BLENDING_MODES",
        py_comment="`blending_mode` attr values (panel-dropdown order)",
        ts_const="BLENDING_MODES",
        ts_type="BlendingModeName",
        ts_const_doc="The canonical `blending_mode` values, in panel-dropdown order.",
        ts_type_doc="Union of the canonical `blending_mode` values.",
        validate=_unique_nonempty("blending_modes"),
    ),
    Vocabulary(
        yaml_key="tone_mappings",
        py_type="ToneMappingName",
        py_const="TONE_MAPPINGS",
        py_comment="`viewer_config.tone_mapping` values",
        ts_const="TONE_MAPPINGS",
        ts_type="ToneMappingName",
        ts_const_doc="Tone-mapping operator names a `viewer_config.tone_mapping` may carry.",
        ts_type_doc="Union of the tone-mapping operator names.",
        validate=_unique_nonempty("tone_mappings"),
    ),
    Vocabulary(
        yaml_key="builtin_colormaps",
        py_type="BuiltinColormapName",
        py_const="BUILTIN_COLORMAP_NAMES",
        py_comment="built-in colormap names a `colormap` attr may carry",
        ts_const="BUILTIN_COLORMAP_NAMES",
        ts_type="BuiltinColormapName",
        ts_const_doc=(
            "Built-in colormap names a `colormap` attr may carry (the `'custom'`\n"
            "sentinel is not a name). Must equal the keys of\n"
            "`rendering/colormap-data.ts::BUILTIN_COLORMAPS`."
        ),
        ts_type_doc="Union of the built-in colormap names.",
        validate=_unique_nonempty("builtin_colormaps"),
    ),
    Vocabulary(
        yaml_key="physical_units",
        py_type="PhysicalUnitName",
        py_const="PHYSICAL_UNITS",
        py_comment="canonical dimension `unit` spellings (= PhysicalUnit enum)",
        ts_const="PHYSICAL_UNITS",
        ts_type="PhysicalUnitName",
        ts_const_doc=(
            "Canonical dimension `unit` spellings on disk (the Python `PhysicalUnit`\n"
            "enum). Input aliases such as `meter` never reach the store."
        ),
        ts_type_doc="Union of the canonical physical-unit spellings.",
        validate=_unique_nonempty("physical_units"),
    ),
    Vocabulary(
        yaml_key="ordering_methods",
        py_type="OrderingMethodName",
        py_const="ORDERING_METHODS",
        py_comment="`ordering` attr values (spatial sort of a node's elements)",
        ts_const="ORDERING_METHODS",
        ts_type="OrderingMethodName",
        ts_const_doc=(
            "`ordering` attr values: how a geometry node's elements were spatially\n"
            "sorted (`none` = not reordered)."
        ),
        ts_type_doc="Union of the `ordering` attr values.",
        validate=_unique_nonempty("ordering_methods"),
    ),
    Vocabulary(
        yaml_key="line_join_styles",
        py_type="LineJoinStyleName",
        py_const="LINE_JOIN_STYLES",
        py_comment="lines-only `join` attr values",
        ts_const="LINE_JOIN_STYLES",
        ts_type="LineJoinStyleName",
        ts_const_doc="Lines-only `join` attr values (joint strategy at a degree-2 joint).",
        ts_type_doc="Union of the line join style names.",
        validate=_unique_nonempty("line_join_styles"),
    ),
    Vocabulary(
        yaml_key="line_types",
        py_type="LineTypeName",
        py_const="LINE_TYPES",
        py_comment="`line_type` attr values on a lines node",
        ts_const="LINE_TYPES",
        ts_type="LineTypeName",
        ts_const_doc="`line_type` attr values: how a lines node's vertices + segments are read.",
        ts_type_doc="Union of the `line_type` attr values.",
        validate=_unique_nonempty("line_types"),
    ),
    Vocabulary(
        yaml_key="nd_transform_affine_keys",
        py_type="NdTransformAffineKey",
        py_const="ND_TRANSFORM_AFFINE_KEYS",
        py_comment="keys an affine nd_transform entry may carry",
        ts_const="ND_TRANSFORM_AFFINE_KEYS",
        ts_type="NdTransformAffineKey",
        ts_const_doc=(
            "Keys an affine `nd_transform` entry may carry; a categorical entry\n"
            "carries `ND_TRANSFORM_PERMUTATION_KEY` instead."
        ),
        ts_type_doc="Union of the affine `nd_transform` entry keys.",
        validate=_unique_nonempty("nd_transform_affine_keys"),
    ),
    Vocabulary(
        yaml_key="dimension_attr_keys",
        py_type="DimensionAttrKey",
        py_const="DIMENSION_ATTR_KEYS",
        py_comment="keys of one `scene_dimensions.dimensions[]` entry",
        ts_const="DIMENSION_ATTR_KEYS",
        ts_type="DimensionAttrKey",
        ts_const_doc=(
            "Keys of one entry in the root `scene_dimensions.dimensions[]` list\n"
            "(`SceneDimensionAttrs` is checked against this at the type level)."
        ),
        ts_type_doc="Union of the per-dimension attr keys.",
        validate=_unique_nonempty("dimension_attr_keys"),
    ),
)


def validate_contract(c: Dict[str, Any]) -> Dict[str, Any]:
    """Run every row's validator; return the normalised values keyed by YAML key.

    Fails (``SystemExit``) on the first malformed key so a bad contract never
    reaches either renderer. Runs the version blocks first, then the scalars,
    then the vocabularies in table order — the same order the projections are
    rendered in.
    """
    resolved: Dict[str, Any] = {}
    for block in VERSION_BLOCKS:
        resolved[block.yaml_key] = _format_versions(block.yaml_key)(c)
    for scalar in SCALARS:
        value = c.get(scalar.yaml_key)
        if not isinstance(value, str) or not value:
            _fail(scalar.yaml_key, "must be a non-empty string")
        resolved[scalar.yaml_key] = value
    for vocab in VOCABULARIES:
        resolved[vocab.yaml_key] = vocab.validate(c)
    return resolved


# --------------------------------------------------------------------------- #
# Python projection
# --------------------------------------------------------------------------- #
def _py_literal_type(name: str, values: List[str]) -> str:
    """Emit ``Name = Literal["a", "b", ...]`` (inline if it fits, else stacked)."""
    inline = f"{name} = Literal[{', '.join(repr_str(v) for v in values)}]"
    if len(inline) <= PY_WIDTH:
        return inline
    body = "".join(f"    {repr_str(v)},\n" for v in values)
    return f"{name} = Literal[\n{body}]"


def _py_tuple(name: str, elem_type: str, values: List[str]) -> str:
    """Emit ``NAME: Final[tuple[T, ...]] = (...)`` (inline if it fits, else stacked)."""
    lhs = f"{name}: Final[tuple[{elem_type}, ...]] = "
    inline = f"{lhs}({', '.join(repr_str(v) for v in values)})"
    if len(values) != 1 and len(inline) <= PY_WIDTH:
        return inline
    if len(values) == 1:
        return f"{lhs}({repr_str(values[0])},)"
    body = "".join(f"    {repr_str(v)},\n" for v in values)
    return f"{lhs}(\n{body})"


def _py_scalar(name: str, value: str) -> str:
    """Emit ``NAME: Final[str] = "value"``."""
    return f"{name}: Final[str] = {repr_str(value)}"


def repr_str(value: str) -> str:
    """Deterministic double-quoted string literal (matches ruff/black)."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_python(c: Dict[str, Any]) -> str:
    r = validate_contract(c)

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

    exports: List[str] = []
    for block in VERSION_BLOCKS:
        v = r[block.yaml_key]
        parts.append(
            f"# --- {block.py_comment} ---\n"
            f"{_py_literal_type(block.py_type, v['supported'])}\n"
            f"{block.py_current}: Final[{block.py_type}] = {repr_str(v['current'])}\n"
            f"{_py_tuple(block.py_supported, block.py_type, v['supported'])}\n"
        )
        exports += [block.py_type, block.py_current, block.py_supported]

    for scalar in SCALARS:
        parts.append(
            f"# --- {scalar.py_comment} ---\n"
            f"{_py_scalar(scalar.py_name, r[scalar.yaml_key])}\n"
        )
        exports.append(scalar.py_name)

    for vocab in VOCABULARIES:
        values = r[vocab.yaml_key]
        parts.append(
            f"# --- {vocab.py_comment} ---\n"
            f"{_py_literal_type(vocab.py_type, values)}\n"
            f"{_py_tuple(vocab.py_const, vocab.py_type, values)}\n"
        )
        exports += [vocab.py_type, vocab.py_const]

    all_body = "".join(f"    {repr_str(name)},\n" for name in exports)
    parts.append(f"__all__ = [\n{all_body}]\n")

    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# TypeScript projection (prettier printWidth=100, singleQuote, es5 commas)
# --------------------------------------------------------------------------- #
def _ts_str(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _ts_jsdoc(doc: str) -> str:
    """Render a JSDoc block (``/** ... */``) with a trailing newline.

    One ``*`` line per line of ``doc`` (blank input lines become a bare `` *``).
    The block sits directly above the declaration it documents, which is what
    the JSDoc-coverage gate (``scripts/check_documentation.py``) credits.
    """
    body = "".join(f" * {line}\n" if line else " *\n" for line in doc.split("\n"))
    return f"/**\n{body} */\n"


def _ts_const(
    name: str, values: List[str], elem_type: str = "string", doc: str = ""
) -> str:
    """Emit ``export const NAME: readonly T[] = [...]``, optionally JSDoc'd.

    ``elem_type`` mirrors :func:`_py_tuple`'s parameter of the same name. Pass
    the paired union alias so callers can iterate the array *and* index a
    record keyed by it — with the default ``string`` they cannot, and every
    such call site ends up re-declaring the literals locally.

    Left as ``string`` on purpose for the ``SUPPORTED_*_VERSIONS`` allowlists:
    those are matched against untrusted values read off disk, and
    ``readonly T[].includes(someString)`` is a type error in TypeScript. Python
    can narrow the same tuples because ``x in tup`` is not type-checked there.

    ``doc`` — when non-empty — prepends a JSDoc block so the generated export is
    counted as documented by the JSDoc-coverage gate.
    """
    prefix = _ts_jsdoc(doc) if doc else ""
    lhs = f"export const {name}: readonly {elem_type}[] = "
    inline = f"{lhs}[{', '.join(_ts_str(v) for v in values)}];"
    if len(inline) <= TS_WIDTH:
        return prefix + inline
    body = "".join(f"  {_ts_str(v)},\n" for v in values)
    return prefix + f"{lhs}[\n{body}];"


def _ts_union(name: str, values: List[str], doc: str = "") -> str:
    """Emit ``export type NAME = 'a' | 'b' | ...``, optionally prefixed by JSDoc."""
    prefix = _ts_jsdoc(doc) if doc else ""
    union = " | ".join(_ts_str(v) for v in values)
    inline = f"export type {name} = {union};"
    if len(inline) <= TS_WIDTH:
        return prefix + inline
    continuation = f"  {union};"
    if len(continuation) <= TS_WIDTH:
        return prefix + f"export type {name} =\n{continuation}"
    body = "".join(f"  | {_ts_str(v)}\n" for v in values)
    # Replace the final newline with a semicolon terminator.
    return prefix + f"export type {name} =\n{body.rstrip()};"


def _ts_scalar(name: str, value: str, doc: str) -> str:
    """Emit a JSDoc'd ``export const NAME = 'value';`` (a literal-typed const)."""
    return _ts_jsdoc(doc) + f"export const {name} = {_ts_str(value)};"


def render_typescript(c: Dict[str, Any]) -> str:
    r = validate_contract(c)

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

    blocks: List[str] = []
    for block in VERSION_BLOCKS:
        v = r[block.yaml_key]
        blocks.append(
            _ts_scalar(block.ts_current, v["current"], block.ts_current_doc)
            + "\n"
            + _ts_const(block.ts_supported, v["supported"], doc=block.ts_supported_doc)
            + "\n"
            + _ts_union(block.ts_type, v["supported"], doc=block.ts_type_doc)
        )
    for scalar in SCALARS:
        blocks.append(_ts_scalar(scalar.ts_name, r[scalar.yaml_key], scalar.ts_doc))
    for vocab in VOCABULARIES:
        values = r[vocab.yaml_key]
        blocks.append(
            _ts_const(vocab.ts_const, values, vocab.ts_type, doc=vocab.ts_const_doc)
            + "\n"
            + _ts_union(vocab.ts_type, values, doc=vocab.ts_type_doc)
        )

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
