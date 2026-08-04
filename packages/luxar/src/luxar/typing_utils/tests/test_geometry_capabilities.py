"""Tests for the per-geometry-type capability table.

The table answers capability questions ("may this type back a kind=lod group?")
that are narrower than the leaf vocabulary ``GEOMETRY_TYPES``. These tests pin
that distinction, since collapsing the two is how a type gets admitted to a code
path that cannot represent it — the defect class this module exists to prevent.
"""

from __future__ import annotations

import pytest

from luxar.typing_utils._format_contract import (
    GEOMETRY_TYPES,
    LOADER_TYPES,
    NODE_TYPES,
)
from luxar.typing_utils.enums import NodeType
from luxar.typing_utils.geometry_capabilities import (
    GEOMETRY_CAPABILITIES,
    lod_capable_types,
    partition_capable_types,
    require_lod_display_type,
    supports_lod,
    supports_partition,
)


def test_every_contract_geometry_type_has_a_row() -> None:
    """The Python stand-in for ``Record<GeometryTypeName, GeometryCapabilities>``.

    The module raises at import if this drifts, so this test mostly documents the
    invariant — but it also fails informatively rather than as a collection error
    if someone weakens the import-time guard.
    """
    assert set(GEOMETRY_CAPABILITIES) == set(GEOMETRY_TYPES)


def test_no_stray_rows() -> None:
    """A row for a type no longer in the contract is also a drift, and caught."""
    assert set(GEOMETRY_CAPABILITIES) - set(GEOMETRY_TYPES) == set()


def test_node_type_enum_matches_contract() -> None:
    """``NodeType`` must stay in step with ``node_types``.

    The enum is hand-written while the contract is generated, so nothing but this
    test couples them — and a missing member would make ``NodeType.validate``
    reject a type the writer can legitimately stamp.
    """
    assert sorted(member.value for member in NodeType) == sorted(NODE_TYPES)


def test_loader_types_is_a_subset_of_geometry_types() -> None:
    """Viewer-drawable is a subset of writable, never the other way round.

    The generator enforces this too; asserting it here means the *meaning* is
    pinned by a test that reads as documentation, not only by a codegen check.
    """
    assert set(LOADER_TYPES) <= set(GEOMETRY_TYPES)


def test_mesh_is_writable_but_not_yet_drawable() -> None:
    """The concrete case the vocabulary/capability split exists for.

    If this ever fails because mesh joined ``loader_types``, that is the phase-3
    switch-on and the assertion should move, not be deleted.
    """
    assert "mesh" in GEOMETRY_TYPES
    assert "mesh" not in LOADER_TYPES


@pytest.mark.parametrize("geometry_type", ["points", "lines", "gsplats"])
def test_established_types_support_lod_and_partition(geometry_type: str) -> None:
    """The three shipped types keep both capabilities.

    A regression here would silently disable working LOD/partition paths, which no
    rendering test would attribute to this table.
    """
    assert supports_lod(geometry_type)
    assert supports_partition(geometry_type)


def test_mesh_supports_neither_lod_nor_partition() -> None:
    """Mesh has no ladder and no spatial split (MESH_NODE_SPEC.md §9)."""
    assert not supports_lod("mesh")
    assert not supports_partition("mesh")


@pytest.mark.parametrize(
    "value",
    ["group", "scene", "custom_marker", "", "POINTS", None, 3, ["points"]],
)
def test_capability_probes_fail_closed(value: object) -> None:
    """Anything outside the vocabulary gets no capability, including non-strings.

    Failing closed is the point: ``display_type`` carries arbitrary strings on
    nested specialized groups and arrives from zarr attrs, so a permissive default
    would grant a geometry capability to a value the writer never recognised.
    """
    assert not supports_lod(value)
    assert not supports_partition(value)


def test_capable_type_lists_are_in_contract_order() -> None:
    """Error messages list types deterministically, in contract order.

    A set-derived order would make the message text vary between runs, which makes
    it untestable and the output noisy in diffs.
    """
    assert lod_capable_types() == tuple(
        t for t in GEOMETRY_TYPES if GEOMETRY_CAPABILITIES[t].lod
    )
    assert partition_capable_types() == tuple(
        t for t in GEOMETRY_TYPES if GEOMETRY_CAPABILITIES[t].partition
    )


def test_require_lod_rejects_incapable_geometry_types() -> None:
    """The shared gate raises for a geometry type with no ladder."""
    with pytest.raises(ValueError, match="display_type for a kind=lod group"):
        require_lod_display_type("mesh", "probe")


@pytest.mark.parametrize(
    "display_type",
    ["points", "lines", "gsplats", "custom_marker", "group", None],
)
def test_require_lod_passes_everything_else(display_type: object) -> None:
    """Non-geometry display types keep passing — LOD stays heterogeneity-tolerant.

    This is the half that is easy to break by "tightening" the gate into an
    allowlist: nested specialized groups legitimately carry marker strings, and an
    allowlist would reject them and break working scenes.
    """
    require_lod_display_type(display_type, "probe")


def test_require_lod_message_names_the_valid_set_and_the_reason() -> None:
    """The error has to be actionable, not just a rejection.

    It names which types ARE allowed (so the caller can pick one) and why this one
    is not (so they do not read it as a bug).
    """
    with pytest.raises(ValueError) as excinfo:
        require_lod_display_type("mesh", "some/node")
    message = str(excinfo.value)
    assert "some/node" in message
    for capable in lod_capable_types():
        assert repr(capable) in message
    assert "LOD ladder" in message
