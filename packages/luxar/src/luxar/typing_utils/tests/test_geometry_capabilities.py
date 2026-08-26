"""Tests for the per-geometry-type capability table.

The table answers capability questions ("may this type back a kind=lod group?")
that are narrower than the leaf vocabulary ``GEOMETRY_TYPES``. These tests pin
that distinction, since collapsing the two is how a type gets admitted to a code
path that cannot represent it — the defect class this module exists to prevent.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from luxar.conftest import find_repo_relative_file
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


def test_capability_table_matches_the_viewer() -> None:
    """Shared writer/viewer capabilities are one decision in two languages.

    The viewer also owns ``pooled`` and ``depthSortable`` render-only flags; the
    cross-language contract is the exhaustive row set plus the shared ``lod`` and
    ``partition`` columns.
    """
    rel = Path("packages/luxar-viewer/src/types/geometry-capabilities.ts")
    start = Path(__file__).resolve()
    source_path = find_repo_relative_file(rel, start)
    assert source_path is not None, (
        f"cannot locate {rel} in any ancestor of {start}. If the viewer file moved, "
        "update this test — do NOT delete the cross-language lock."
    )
    source = source_path.read_text(encoding="utf-8")
    table = re.search(
        r"export const GEOMETRY_CAPABILITIES:[^=]+?=\s*Object\.freeze\(\{"
        r"(?P<body>.*?)^\s*\}\);",
        source,
        re.MULTILINE | re.DOTALL,
    )
    assert table is not None, (
        "cannot find the literal GEOMETRY_CAPABILITIES Object.freeze table in "
        f"{source_path}. If its shape changed, update this parser — do NOT delete "
        "the cross-language lock."
    )
    rows = re.findall(
        r"^\s*(\w+):\s*\{\s*lod:\s*(true|false),\s*"
        r"partition:\s*(true|false),\s*pooled:\s*(true|false),\s*"
        r"depthSortable:\s*(true|false)\s*\},",
        table.group("body"),
        re.MULTILINE,
    )
    viewer_capabilities = {
        name: tuple(value == "true" for value in values[:2]) for name, *values in rows
    }
    assert len(viewer_capabilities) == len(rows), (
        f"duplicate geometry rows in viewer capability table: {rows!r}"
    )
    python_capabilities = {
        name: tuple(capabilities)
        for name, capabilities in GEOMETRY_CAPABILITIES.items()
    }
    assert viewer_capabilities == python_capabilities


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


def test_mesh_is_both_writable_and_drawable() -> None:
    """Mesh completed the phase-3 switch-on.

    This assertion MOVED rather than being deleted, as its predecessor asked: it
    used to read ``"mesh" not in LOADER_TYPES``, pinning the writable-but-not-yet-
    drawable state the vocabulary/capability split was introduced for. Flipping it
    is the switch-on's visible signature on the Python side, and keeping the test
    (rather than dropping it) is what stops mesh silently falling back OUT of
    ``loader_types`` in some later contract edit.

    Note the split itself is NOT obsolete now that the two lists agree on mesh:
    ``loader_types`` remains a capability and ``geometry_types`` a vocabulary, and
    the next writable-before-drawable type will need the gap again.
    """
    assert "mesh" in GEOMETRY_TYPES
    assert "mesh" in LOADER_TYPES


@pytest.mark.parametrize("geometry_type", ["points", "lines", "gsplats"])
def test_established_types_support_lod_and_partition(geometry_type: str) -> None:
    """The three shipped types keep both capabilities.

    A regression here would silently disable working LOD/partition paths, which no
    rendering test would attribute to this table.
    """
    assert supports_lod(geometry_type)
    assert supports_partition(geometry_type)


def test_mesh_supports_both_lod_and_partition() -> None:
    """Mesh's row now says yes twice, and each flag says yes for its own reason.

    Pinned by name because this is the row that keeps the table honest — each flag
    was earned by a separate producer, and neither implies the other.

    ``lod`` is true for the SUBSTITUTIVE mechanism ONLY: a ``kind=lod`` group holds
    levels that REPLACE one another, and mesh decimation
    (``luxar.mesh.decimate``) is the producer that was missing. The ADDITIVE prefix
    ladder is still impossible for a surface — a prefix of an index buffer is a
    holed surface, not a coarse one — and this flag never gated that flavour; it is
    refused in ``adders/mesh.py`` instead.

    ``partition`` is true because a BSP cut runs BETWEEN faces and each part
    re-indexes its own vertices (``luxar.mesh.split``, spec §9.2).
    """
    assert supports_lod("mesh")
    assert supports_partition("mesh")


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
    """The shared gate raises for a geometry type with no ladder.

    Every type in the contract is LOD-capable today, so the only way to exercise
    the refusal is to simulate a future one. Keeping the test rather than deleting
    it with the last excluded type is deliberate: the gate is the single chokepoint
    all three display-type routes call, and an unexercised chokepoint is one that
    quietly stops working.
    """
    from unittest.mock import patch

    with patch(
        "luxar.typing_utils.geometry_capabilities.supports_lod", return_value=False
    ):
        with pytest.raises(ValueError, match="display_type for a kind=lod group"):
            require_lod_display_type("mesh", "probe")


@pytest.mark.parametrize(
    "display_type",
    ["points", "lines", "gsplats", "mesh", "custom_marker", "group", None],
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
    from unittest.mock import patch

    with patch(
        "luxar.typing_utils.geometry_capabilities.supports_lod", return_value=False
    ):
        with pytest.raises(ValueError) as excinfo:
            require_lod_display_type("mesh", "some/node")
    message = str(excinfo.value)
    assert "some/node" in message
    for capable in lod_capable_types():
        assert repr(capable) in message
    assert "LOD ladder" in message


def test_the_lod_refusal_message_stays_GENERIC() -> None:
    """No type-specific rationale may be attached to this rejection.

    The message once appended a mesh explanation (additive-vs-substitutive, QEM
    decimation) whenever the refused type was ``mesh``. That became dead when mesh
    gained its ladder, and it was always the wrong shape: this guard fires for any
    LOD-less geometry type, and a paragraph about surfaces reads as a confidently
    wrong diagnostic about whichever type the caller actually named.

    Simulated by making a LOD-capable type look LOD-less — which is now the only
    way to reach the branch at all, since every contract type is capable.
    """
    from unittest.mock import patch

    with patch(
        "luxar.typing_utils.geometry_capabilities.supports_lod", return_value=False
    ):
        with pytest.raises(ValueError) as excinfo:
            require_lod_display_type("points", "kind=lod group 'g'")
        message = str(excinfo.value)
        assert "kind=lod group must be one of" in message
        assert "QEM" not in message, "no type-specific rationale belongs here"
        assert "connected surface" not in message

        # Same generic text for mesh — no special case survives.
        with pytest.raises(ValueError) as mesh_exc:
            require_lod_display_type("mesh", "kind=lod group 'g'")
        assert "QEM" not in str(mesh_exc.value)
