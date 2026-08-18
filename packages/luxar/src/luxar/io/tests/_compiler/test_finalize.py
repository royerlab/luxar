"""Direct unit tests for the finalize-time passes in luxar.io._compiler.finalize."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync

from luxar._zarr_compat import create_array
from luxar.core.group.lod.group import MAX_COVERAGE_FRACTION
from luxar.io._compiler.finalize.hashing import (
    _payload_terms,
    compute_content_hashes,
)
from luxar.io._compiler.finalize.lod_backfill import (
    finalize_lod_display_types,
    finalize_lod_position_bounds,
    warn_one_part_partition_anchors,
)
from luxar.io._compiler.finalize.validation import validate_discrete_dimension_ranges
from luxar.typing_utils._format_contract import GEOMETRY_TYPES
from luxar.typing_utils.geometry_capabilities import lod_capable_types


def _lod_tree() -> zarr.Group:
    """Root with a kind=lod wrapper holding two gsplat leaves (no parent bounds)."""
    root = zarr.group()
    lod = root.create_group("lodgrp")
    lod.attrs["kind"] = "lod"
    coarse = lod.create_group("coarse")
    coarse.attrs["type"] = "gsplats"
    coarse.attrs["position_bounds"] = {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}
    fine = lod.create_group("fine")
    fine.attrs["type"] = "gsplats"
    fine.attrs["position_bounds"] = {"min": [-1.0, 0.0, 0.0], "max": [1.0, 2.0, 1.0]}
    return root


def test_finalize_lod_position_bounds_backfills_union() -> None:
    root = _lod_tree()
    assert "position_bounds" not in dict(root["lodgrp"].attrs)
    finalize_lod_position_bounds(root)
    bounds = dict(root["lodgrp"].attrs)["position_bounds"]
    assert bounds["min"] == [-1.0, 0.0, 0.0]
    assert bounds["max"] == [1.0, 2.0, 1.0]


def test_finalize_lod_position_bounds_never_overwrites() -> None:
    root = _lod_tree()
    authored = {"min": [-9.0, -9.0, -9.0], "max": [9.0, 9.0, 9.0]}
    root["lodgrp"].attrs["position_bounds"] = authored
    finalize_lod_position_bounds(root)
    assert dict(root["lodgrp"].attrs)["position_bounds"] == authored


def _partition_tree() -> zarr.Group:
    """Root with a kind=partition wrapper holding two gsplat-leaf parts (no parent
    bounds) — the shape a grafted partition (add_gsplats_from_file) produces, since
    add_partition_group does not compute the children-union the standalone writer
    stamps at write time."""
    root = zarr.group()
    part = root.create_group("partgrp")
    part.attrs["kind"] = "partition"
    p0 = part.create_group("part_0")
    p0.attrs["type"] = "gsplats"
    p0.attrs["position_bounds"] = {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}
    p1 = part.create_group("part_1")
    p1.attrs["type"] = "gsplats"
    p1.attrs["position_bounds"] = {"min": [5.0, -2.0, 0.0], "max": [6.0, 1.0, 3.0]}
    return root


def test_finalize_position_bounds_backfills_partition_wrapper() -> None:
    """A grafted kind=partition wrapper missing position_bounds must be back-filled
    with the union of its parts (regression: graft dropped wrapper bounds, losing
    partition-unit culling / graft-vs-standalone parity)."""
    root = _partition_tree()
    assert "position_bounds" not in dict(root["partgrp"].attrs)
    finalize_lod_position_bounds(root)
    bounds = dict(root["partgrp"].attrs)["position_bounds"]
    assert bounds["min"] == [0.0, -2.0, 0.0]
    assert bounds["max"] == [6.0, 1.0, 3.0]


def test_finalize_position_bounds_partition_never_overwrites() -> None:
    root = _partition_tree()
    authored = {"min": [-9.0, -9.0, -9.0], "max": [9.0, 9.0, 9.0]}
    root["partgrp"].attrs["position_bounds"] = authored
    finalize_lod_position_bounds(root)
    assert dict(root["partgrp"].attrs)["position_bounds"] == authored


def test_finalize_lod_display_types_resolves_from_finest() -> None:
    root = _lod_tree()
    finalize_lod_display_types(root)
    assert dict(root["lodgrp"].attrs)["display_type"] == "gsplats"


def test_finalize_lod_display_types_uses_child_index_not_name_order() -> None:
    """The finest child is resolved by ``child_index`` (coarsest→finest), not by
    alphabetical name order. With >=10 children, name order puts ``child_10``
    before ``child_2`` — so a name-sorted "last" child is NOT the finest. Mixing
    leaf types across levels makes the mis-resolution observable: pre-fix the
    name-sorted last child (child_9, a ``points`` leaf) wins; post-fix the
    highest-child_index child (child_11, the actual finest ``gsplats`` leaf) wins.
    """
    root = zarr.group()
    lod = root.create_group("lodgrp")
    lod.attrs["kind"] = "lod"
    # 12 children, child_index 0..11 (coarsest→finest). All but the true finest
    # are 'points'; the finest (child_index 11) is 'gsplats'. Name-sorting yields
    # child_9 last (a 'points' leaf) — the wrong answer.
    for i in range(12):
        c = lod.create_group(f"child_{i}")
        c.attrs["type"] = "gsplats" if i == 11 else "points"
        c.attrs["child_index"] = i
    finalize_lod_display_types(root)
    assert dict(root["lodgrp"].attrs)["display_type"] == "gsplats"


def test_finalize_lod_display_types_falls_back_to_name_order_without_child_index() -> (
    None
):
    """Legacy trees with no ``child_index`` keep the historical name-order
    behaviour (finest = last by name) so nothing regresses for old data."""
    root = zarr.group()
    lod = root.create_group("lodgrp")
    lod.attrs["kind"] = "lod"
    for i in range(3):  # child_0..child_2, no child_index
        c = lod.create_group(f"child_{i}")
        c.attrs["type"] = "lines" if i == 2 else "points"
    finalize_lod_display_types(root)
    assert dict(root["lodgrp"].attrs)["display_type"] == "lines"


def test_compute_content_hashes_is_deterministic_and_stamps_attrs() -> None:
    root = _lod_tree()
    h1 = compute_content_hashes(root)
    assert isinstance(h1, str) and len(h1) > 0
    assert dict(root.attrs)["content_hash"] == h1
    # Recompute on an identical fresh tree → same root hash (determinism).
    h2 = compute_content_hashes(_lod_tree())
    assert h1 == h2


def test_compute_content_hashes_changes_with_data() -> None:
    root = _lod_tree()
    create_array(
        root["lodgrp"]["fine"], "centers", data=np.ones((3, 3), dtype=np.float32)
    )
    h_with = compute_content_hashes(root)
    assert h_with != compute_content_hashes(_lod_tree())


# A minimal valid PNG (1x1 red pixel) — no PIL needed.
_TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xcf"
    b"\xc0\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _overlay_image_store(
    root_path: Path,
    payload: bytes | None,
    filename: str = "image.png",
    zarr_format: int = 3,
) -> zarr.Group:
    """Real on-disk store with one `overlays/logo` image-overlay group.

    ``payload`` is written as a plain file inside the group's directory (exactly
    what ``write_overlay`` does); ``None`` writes no file at all.

    ``zarr_format`` is pinned explicitly because which metadata DOCUMENT exists
    on disk depends on it — `.zattrs` at format 2, `zarr.json` at format 3 — and
    a case naming a document the store does not have tests nothing. Explicit also
    keeps the cases below independent of `LUXAR_ZARR_FORMAT` (which raw
    `zarr.open_group` never consulted anyway).
    """
    root = zarr.open_group(str(root_path), mode="w", zarr_format=zarr_format)
    logo = root.create_group("overlays").create_group("logo")
    logo.attrs["type"] = "overlay_image"
    logo.attrs["image_file"] = filename
    if payload is not None:
        image_dir = root_path / "overlays" / "logo"
        image_dir.mkdir(parents=True, exist_ok=True)
        (image_dir / filename).write_bytes(payload)
    return root


def test_compute_content_hashes_covers_overlay_payload_bytes(tmp_path: Path) -> None:
    """Issue #1720: an overlay PNG is neither an array nor a group, so its bytes
    used to sit outside the hash entirely — two stores differing only in the
    image stamped one digest, and `content_hash` is supposed to fingerprint the
    content it covers."""
    store_path = tmp_path / "scene.luxar.zarr"
    root = _overlay_image_store(store_path, _TINY_PNG)
    h_before = compute_content_hashes(root)

    # Recomputing over unchanged bytes reproduces the digest (the previous stamp
    # is excluded from the walk by design).
    assert compute_content_hashes(root) == h_before

    # Overwrite the payload at the SAME filename, touching nothing else. The
    # edit is SAME-LENGTH so this pins the bytes: a longer payload would still
    # fail on the length prefix alone, with the bytes never folded in.
    (store_path / "overlays" / "logo" / "image.png").write_bytes(
        _TINY_PNG[:-1] + b"\x83"
    )
    assert compute_content_hashes(root) != h_before


def test_compute_content_hashes_absent_payload_differs_from_empty(
    tmp_path: Path,
) -> None:
    """A missing named file must not hash the same as a zero-byte one."""
    h_absent = compute_content_hashes(_overlay_image_store(tmp_path / "a", None))
    h_empty = compute_content_hashes(_overlay_image_store(tmp_path / "b", b""))
    assert h_absent != h_empty


def test_compute_content_hashes_payload_free_digest_is_unchanged() -> None:
    """A tree with no payload attrs must hash byte-identically to what the
    pre-payload implementation produced — the payload step only ever adds terms
    for groups that actually name a payload file.

    The digest below was captured by running the implementation with the payload
    step commented out, over the tree built right here. The tree is local on
    purpose: pinning a digest over a helper shared with other tests would turn
    any legitimate edit to that helper into a failure of this tripwire. It must
    only ever change deliberately — a moving digest means the hash inputs
    changed, and every already-cached dataset re-downloads once.
    """
    root = zarr.group()
    root.attrs["type"] = "scene"
    child = root.create_group("pts")
    child.attrs["type"] = "points"
    child.attrs["visible"] = True
    create_array(child, "positions", data=np.arange(6, dtype=np.float32).reshape(3, 2))
    assert "image_file" not in dict(child.attrs)  # really the payload-free path
    assert compute_content_hashes(root) == "1a023e2b06f57a09"


def test_payload_terms_are_prefix_free(tmp_path: Path) -> None:
    """The payload block must be injective on its own.

    This is the collision `_payload_terms` documents: without the key and name
    length prefixes, both stores below fold the identical ``image_filea7:5:hello``.
    Both names are readable, so no sentinel is involved either way.
    """
    a = _overlay_image_store(tmp_path / "a", b"5:hello", filename="a")
    b = _overlay_image_store(tmp_path / "b", b"hello", filename="a7:")

    def terms(root: zarr.Group) -> bytes:
        group = root["overlays/logo"]
        return b"".join(_payload_terms(group, dict(group.attrs)))

    assert terms(a) != terms(b)
    # The framing itself, spelled out: length, key, length, name, length, bytes.
    assert terms(a) == b"payload:10:image_file1:a7:5:hello"
    assert terms(b) == b"payload:10:image_file3:a7:5:hello"


@pytest.mark.parametrize(
    ("filename", "zarr_format", "sentinel"),
    [
        # Rejected by NAME (semantic): not one path component, or a document
        # this very walk stamps.
        ("../../../evil.png", 3, b"unsafe:"),
        ("sub/evil.png", 3, b"unsafe:"),
        ("sub\\evil.png", 3, b"unsafe:"),  # normalize_path turns `\` into `/`
        (".", 3, b"unsafe:"),
        ("..", 3, b"unsafe:"),
        ("zarr.json", 3, b"unsafe:"),  # the v3 doc — exists only at format 3
        (".zattrs", 2, b"unsafe:"),  # the v2 doc — exists only at format 2
        # Rejected by the STORE (mechanical): `LocalStore.get` catches only the
        # not-found family, so these reach the payload step as exceptions.
        ("x" * 300 + ".png", 3, b"unreadable:"),  # OSError(ENAMETOOLONG)
        ("a\x00b.png", 3, b"unreadable:"),  # ValueError: embedded null byte
        # A lone surrogate round-trips zarr attrs (`json.dumps` escapes it), so
        # the walk really does meet such a name; encoding it must not raise, and
        # the store's own UnicodeEncodeError is a ValueError.
        ("im\ud800age.png", 3, b"unreadable:"),
    ],
)
def test_compute_content_hashes_tolerates_unreadable_payload_names(
    tmp_path: Path, filename: str, zarr_format: int, sentinel: bytes
) -> None:
    """A payload name the walk cannot read must fold in deterministically.

    Pins only what the payload step owns: no raise (one escaping would reach
    ``finalize()``, which stamps the store ``incomplete``), a convergent digest,
    and which of the two sentinels was folded — ``unsafe:`` for a name refused
    semantically, ``unreadable:`` for the store's own verdict. That a change to
    the *name* moves the hash comes from step 2, which hashes the attrs JSON, not
    from here — do not read this test as covering it.
    """
    root = _overlay_image_store(
        tmp_path / "scene", None, filename=filename, zarr_format=zarr_format
    )

    first = compute_content_hashes(root)  # must not raise
    assert compute_content_hashes(root) == first

    group = root["overlays/logo"]
    assert sentinel in b"".join(_payload_terms(group, dict(group.attrs)))


def test_compute_content_hashes_reads_the_payload_through_the_store() -> None:
    """The read goes through the store, not the filesystem — so it works on a
    store with no filesystem at all."""
    root = zarr.group()  # MemoryStore
    logo = root.create_group("overlays").create_group("logo")
    logo.attrs["image_file"] = "image.png"

    def _write(payload: bytes) -> None:
        buffer = default_buffer_prototype().buffer.from_bytes(payload)
        sync((logo.store_path / "image.png").set(buffer))

    _write(_TINY_PNG)
    h_before = compute_content_hashes(root)
    # A same-length edit, so this pins the BYTES and not just the length prefix.
    _write(_TINY_PNG[:-1] + b"\x83")
    assert compute_content_hashes(root) != h_before


def test_compiled_scene_hash_tracks_overlay_image_bytes(tmp_path: Path) -> None:
    """End-to-end: two scenes compiled from the same script and differing only
    in their overlay PNG get different root content hashes."""
    from luxar import Dimensions, LuxarZarrCompiler

    def _compile(path: Path, payload: bytes) -> str:
        with LuxarZarrCompiler(path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points("pts", positions=np.zeros((2, 3), dtype=np.float32))
            scene.add_image(payload, position=(0.9, 0.05))
        return str(zarr.open_group(str(path), mode="r").attrs["content_hash"])

    # Same-length payloads, so this pins the bytes end to end: a length-changing
    # edit would separate the two digests on the length prefix alone.
    h_a = _compile(tmp_path / "a.luxar.zarr", _TINY_PNG)
    h_b = _compile(tmp_path / "b.luxar.zarr", _TINY_PNG[:-1] + b"\x83")
    assert h_a != h_b


def test_validate_discrete_dimension_ranges_noop_without_bounds() -> None:
    # No scene_bounds → returns immediately without raising.
    validate_discrete_dimension_ranges(zarr.group(), None)


def _store_with_discrete_time(range_: tuple[float, float]) -> zarr.Group:
    """Root with a 4D scene: [time (discrete, hidden, step=1), z, y, x]."""
    from luxar.core.dimensions import Dimension, Dimensions

    dims = Dimensions(
        [
            Dimension(
                name="time", range=range_, step=1.0, display=False, discrete=True
            ),
            Dimension(name="z", range=(0, 10)),
            Dimension(name="y", range=(0, 10)),
            Dimension(name="x", range=(0, 10)),
        ]
    )
    root = zarr.group()
    root.attrs["scene_dimensions"] = dims.to_dict()
    return root


def test_validate_discrete_ranges_tolerance_matches_viewer_quarter_step() -> None:
    """The misalignment warning threshold mirrors the viewer's discrete query
    tolerance (0.25 × step — see DISCRETE_TOLERANCE_FRACTION in
    tolerance-computer.ts), NOT the legacy 0.5 × step. Data starting 0.3 below
    the declared range min is inside the legacy half-step tolerance (no warning
    pre-fix) but outside the quarter-step one, and a viewer initialized at the
    range min would show no data — so it MUST warn."""
    import warnings as _warnings

    bounds = {"min": [1.3, 0.0, 0.0, 0.0], "max": [5.0, 10.0, 10.0, 10.0]}
    store = _store_with_discrete_time(range_=(1.0, 5.0))
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        validate_discrete_dimension_ranges(store, bounds)
    assert any("range starting at" in str(w.message) for w in caught), (
        f"Expected misalignment warning for 0.3-step offset, got: {[str(w.message) for w in caught]}"
    )


def test_validate_discrete_ranges_warns_on_off_grid_data() -> None:
    """Discrete data more than a quarter-step off the absolute k*step grid can
    pass the viewer's half-step visibility gate while its (epsilon-padded)
    chunks are never fetched by the quarter-step query — it would silently not
    display. The declared range matches the data exactly here, so the
    range-edge checks stay silent; only the on-grid check must fire."""
    import warnings as _warnings

    # Data at 1.3, ..., 5.3 (uniform 0.3 grid offset), range matching exactly.
    bounds = {"min": [1.3, 0.0, 0.0, 0.0], "max": [5.3, 10.0, 10.0, 10.0]}
    store = _store_with_discrete_time(range_=(1.3, 5.3))
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        validate_discrete_dimension_ranges(store, bounds)
    messages = [str(w.message) for w in caught]
    assert not any("range starting at" in m for m in messages), messages
    assert any("off the step grid" in m for m in messages), (
        f"Expected off-grid warning for 0.3 grid offset, got: {messages}"
    )


def test_validate_discrete_ranges_warns_on_off_grid_data_without_step() -> None:
    """Regression (deep-double-check): a discrete dimension WITHOUT a declared
    step is not exempt from the on-grid check — the viewer defaults a missing
    step to 1.0 (scene-dims-manager.ts `step: dim.step || 1.0`) and snaps
    navigation/queries to the integer grid, so step-less data at non-integer
    values silently never displays. Pre-fix the check was gated on
    `if dim.step:` and this compiled warning-free."""
    import warnings as _warnings

    from luxar.core.dimensions import Dimension, Dimensions

    dims = Dimensions(
        [
            # No step declared — viewer treats the grid as integers.
            Dimension(name="time", range=(0.5, 4.5), display=False, discrete=True),
            Dimension(name="z", range=(0, 10)),
            Dimension(name="y", range=(0, 10)),
            Dimension(name="x", range=(0, 10)),
        ]
    )
    root = zarr.group()
    root.attrs["scene_dimensions"] = dims.to_dict()

    bounds = {"min": [0.5, 0.0, 0.0, 0.0], "max": [4.5, 10.0, 10.0, 10.0]}
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        validate_discrete_dimension_ranges(root, bounds)
    messages = [str(w.message) for w in caught]
    assert any("off the step grid" in m for m in messages), (
        f"Expected off-grid warning for step-less discrete data at x.5, got: {messages}"
    )
    assert any("no step is declared" in m for m in messages), messages


def test_validate_discrete_ranges_on_grid_data_is_silent() -> None:
    """Exactly on-grid data (and data within a quarter-step of the grid)
    triggers no off-grid warning."""
    import warnings as _warnings

    bounds = {"min": [1.0, 0.0, 0.0, 0.0], "max": [5.0, 10.0, 10.0, 10.0]}
    store = _store_with_discrete_time(range_=(1.0, 5.0))
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        validate_discrete_dimension_ranges(store, bounds)
    assert not caught, [str(w.message) for w in caught]


def test_validate_discrete_ranges_within_quarter_step_is_silent() -> None:
    """A range min within the quarter-step tolerance of the data start is
    reachable by the viewer's on-grid query — no warning."""
    import warnings as _warnings

    bounds = {"min": [1.2, 0.0, 0.0, 0.0], "max": [5.0, 10.0, 10.0, 10.0]}
    store = _store_with_discrete_time(range_=(1.0, 5.0))
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        validate_discrete_dimension_ranges(store, bounds)
    assert not any("range starting at" in str(w.message) for w in caught), (
        f"No warning expected within quarter-step tolerance, got: {[str(w.message) for w in caught]}"
    )


def _lod_tree_with_unrecognized_leaf() -> zarr.Group:
    """kind=lod wrapper (no display_type) over a leaf whose ``type`` is unknown.

    ``resolve()`` early-returns only for the geometry types it hardcodes. Any
    other leaf falls through to the "recurse into the finest child" branch — and
    a LEAF group's ``keys()`` lists its ARRAYS, not sub-groups.
    """
    root = zarr.group()
    lod = root.create_group("lodgrp")
    lod.attrs["kind"] = "lod"
    leaf = lod.create_group("leaf")
    leaf.attrs["type"] = "some_future_type"
    leaf.attrs["child_index"] = 0
    create_array(leaf, "vertices", data=np.zeros((3, 3), dtype=np.float32))
    create_array(leaf, "faces", data=np.zeros((1, 3), dtype=np.uint32))
    return root


def test_display_type_backfill_survives_unrecognized_leaf_type() -> None:
    """An unknown leaf ``type`` must not crash the finalize pass.

    Regression guard: ``resolve()`` used to pick the alphabetically-last ARRAY
    ("vertices") as the "finest child" and recurse into it, then call ``.keys()``
    on a ``zarr.Array`` — which does not have it — raising a bare AttributeError
    from deep inside finalize. Resolution should simply yield nothing.
    """
    root = _lod_tree_with_unrecognized_leaf()

    finalize_lod_display_types(root)  # must not raise

    # Nothing resolvable ⇒ no display_type is invented for the wrapper.
    assert "display_type" not in dict(root["lodgrp"].attrs)


def test_display_type_backfill_ignores_arrays_when_picking_finest_child() -> None:
    """Array siblings must never shadow a real child GROUP.

    A plain wrapper holding both arrays and a genuine geometry sub-group must
    resolve through the sub-group. Sorting by name alone would pick "zz_data".
    """
    root = zarr.group()
    lod = root.create_group("lodgrp")
    lod.attrs["kind"] = "lod"
    wrapper = lod.create_group("wrapper")
    wrapper.attrs["type"] = "group"
    create_array(wrapper, "zz_data", data=np.zeros((2, 2), dtype=np.float32))
    leaf = wrapper.create_group("aa_leaf")
    leaf.attrs["type"] = "lines"

    finalize_lod_display_types(root)

    assert dict(root["lodgrp"].attrs).get("display_type") == "lines"


def test_display_type_backfill_resolves_every_lod_capable_geometry_type() -> None:
    """Leaf resolution is driven by the contract, not a local literal.

    Pins the single-sourcing: an LOD-capable geometry type added to
    ``contract.yaml`` is resolvable here without editing this module.

    Parametrised over ``lod_capable_types()`` rather than ``GEOMETRY_TYPES``,
    because the two are no longer the same set. Being a geometry leaf (the
    vocabulary) and being usable as a ``kind=lod`` group's ``display_type`` (a
    capability) are different questions, and a type can answer yes to the first
    and no to the second — see the rejection test below for the other half.
    """
    for geometry_type in lod_capable_types():
        root = zarr.group()
        lod = root.create_group("lodgrp")
        lod.attrs["kind"] = "lod"
        leaf = lod.create_group("leaf")
        leaf.attrs["type"] = geometry_type

        finalize_lod_display_types(root)

        assert dict(root["lodgrp"].attrs).get("display_type") == geometry_type


def test_display_type_backfill_rejects_every_lod_incapable_geometry_type() -> None:
    """A geometry type with no LOD ladder must not be back-filled onto a lod group.

    The other half of the test above, and the reason the guard exists: without it
    the back-fill happily stamps ``display_type='mesh'``, producing a ``kind=lod``
    group that no viewer path can load — written with no error at all. Failing
    here instead means the broken store is never produced.

    Parametrised over the complement so it covers a fifth type automatically, and
    skips cleanly (rather than passing vacuously) if every geometry type ever
    becomes LOD-capable.
    """
    incapable = [t for t in GEOMETRY_TYPES if t not in lod_capable_types()]
    if not incapable:
        pytest.skip("every contract geometry type is LOD-capable")

    for geometry_type in incapable:
        root = zarr.group()
        lod = root.create_group("lodgrp")
        lod.attrs["kind"] = "lod"
        leaf = lod.create_group("leaf")
        leaf.attrs["type"] = geometry_type

        with pytest.raises(ValueError, match="display_type for a kind=lod group"):
            finalize_lod_display_types(root)

        # And nothing was stamped on the way out.
        assert "display_type" not in dict(root["lodgrp"].attrs)


def test_display_type_backfill_ignores_a_typed_array_sibling() -> None:
    """A zarr ARRAY carrying a ``type`` attr must never win finest-child.

    The nastier sibling of the crash case: arrays can hold attrs, so an array
    named after the real child and stamped with a recognised geometry ``type``
    used to be picked as the finest child and resolve EARLY — no exception,
    just the wrong ``display_type`` written to the wrapper.
    """
    root = zarr.group()
    lod = root.create_group("lodgrp")
    lod.attrs["kind"] = "lod"
    leaf = lod.create_group("aaa_real_child")
    leaf.attrs["type"] = "lines"
    stray = create_array(lod, "zzz_array", data=np.zeros((2, 2), dtype=np.float32))
    stray.attrs["type"] = "gsplats"

    finalize_lod_display_types(root)

    assert dict(root["lodgrp"].attrs).get("display_type") == "lines"


def test_position_bounds_backfill_ignores_array_siblings() -> None:
    """The bounds pass must also walk/resolve child GROUPS only.

    Sibling of the display-type guard: both passes recurse through the tree, and
    a stray array beside real children would be descended into — ``resolve``
    looking for its ``position_bounds`` and ``walk`` looking for its ``kind`` —
    dying on ``Array.keys()`` either way.
    """
    root = zarr.group()
    lod = root.create_group("lodgrp")
    lod.attrs["kind"] = "lod"
    create_array(lod, "stray", data=np.zeros((2, 2), dtype=np.float32))
    for name, bounds in (
        ("a", {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}),
        ("b", {"min": [-1.0, 0.0, 0.0], "max": [1.0, 2.0, 1.0]}),
    ):
        child = lod.create_group(name)
        child.attrs["type"] = "points"
        child.attrs["position_bounds"] = bounds

    finalize_lod_position_bounds(root)

    assert dict(root["lodgrp"].attrs)["position_bounds"] == {
        "min": [-1.0, 0.0, 0.0],
        "max": [1.0, 2.0, 1.0],
    }


# ────────────────────────────────────────────────────────────────────────
# warn_one_part_partition_anchors — the check no PRODUCER can make
# ────────────────────────────────────────────────────────────────────────


def _anchored_ladder(parent: zarr.Group, name: str, finest: float) -> None:
    """A kind=lod group of two mesh children whose finest threshold is ``finest``."""
    lod = parent.create_group(name)
    lod.attrs["kind"] = "lod"
    lod.attrs["display_type"] = "mesh"
    for i, coverage in enumerate((0.0, finest)):
        child = lod.create_group(f"child_{i}")
        child.attrs["type"] = "mesh"
        child.attrs["coverage_fraction"] = coverage


def _partition_of_ladders(n_parts: int, finest: float) -> zarr.Group:
    root = zarr.group()
    part = root.create_group("tiled")
    part.attrs["kind"] = "partition"
    part.attrs["display_type"] = "mesh"
    part.attrs["max_elements"] = 100
    for i in range(n_parts):
        _anchored_ladder(part, f"part_{i}", finest)
    return root


def test_warns_for_a_tile_anchored_ladder_under_a_one_part_partition(capsys) -> None:
    # The undetectable-at-derive-time case: the scene adder anchored part 0 at
    # 4.0 before it could know part 1 would never arrive.
    warn_one_part_partition_anchors(_partition_of_ladders(1, MAX_COVERAGE_FRACTION))
    out = capsys.readouterr().out
    assert "tiled/part_0" in out
    assert "ONE part" in out
    assert "coverage_fractions" in out, "the message must name the way out"


def test_does_not_warn_for_a_genuine_multi_part_partition(capsys) -> None:
    # Two parts IS a tiling, so 4.0 is the correct anchor there.
    warn_one_part_partition_anchors(_partition_of_ladders(2, MAX_COVERAGE_FRACTION))
    assert capsys.readouterr().out == ""


def test_does_not_warn_for_a_whole_object_ladder(capsys) -> None:
    # A one-part partition holding a 1.0-anchored ladder is exactly right.
    warn_one_part_partition_anchors(_partition_of_ladders(1, 1.0))
    assert capsys.readouterr().out == ""


def test_does_not_warn_without_a_partition_ancestor(capsys) -> None:
    # A hand-authored 4.0 ladder outside any partition is the author's business.
    root = zarr.group()
    _anchored_ladder(root, "ladder", MAX_COVERAGE_FRACTION)
    warn_one_part_partition_anchors(root)
    assert capsys.readouterr().out == ""


def test_blames_the_nearest_partition_not_an_outer_one_part_wrapper(capsys) -> None:
    """A real tiling nested inside a one-part wrapper must not be blamed.

    Its ladders switch on a genuine tile, so their 4.0 anchor is correct; only
    the ladder bound DIRECTLY to the lone part is wrong.
    """
    root = zarr.group()
    outer = root.create_group("outer")
    outer.attrs["kind"] = "partition"
    inner = outer.create_group("only_part")
    inner.attrs["kind"] = "partition"
    for i in range(2):
        _anchored_ladder(inner, f"tile_{i}", MAX_COVERAGE_FRACTION)

    warn_one_part_partition_anchors(root)

    assert capsys.readouterr().out == ""


def test_does_not_warn_for_a_one_part_partition_inside_a_real_tiling(capsys) -> None:
    """The inverse nesting: a lone part sitting INSIDE a genuine tiling.

    Both gsplat writers OR an incoming binding in rather than overwriting it
    (``under_partition or len(children) > 1``), so the ladder here really is
    inside one of the outer partition's two tiles and 4.0 is the anchor they
    derive for it. Blaming the nearest wrapper alone would report that as a
    mistake.
    """
    root = zarr.group()
    outer = root.create_group("tiled")
    outer.attrs["kind"] = "partition"
    _anchored_ladder(outer, "part_0", MAX_COVERAGE_FRACTION)
    inner = outer.create_group("part_1")
    inner.attrs["kind"] = "partition"
    _anchored_ladder(inner, "only_part", MAX_COVERAGE_FRACTION)

    warn_one_part_partition_anchors(root)

    assert capsys.readouterr().out == ""


def test_warns_once_per_offending_group(capsys) -> None:
    root = zarr.group()
    part = root.create_group("tiled")
    part.attrs["kind"] = "partition"
    holder = part.create_group("only_part")  # a plain group between the two
    _anchored_ladder(holder, "ladder", MAX_COVERAGE_FRACTION)

    warn_one_part_partition_anchors(root)

    assert capsys.readouterr().out.count("tiled/only_part/ladder") == 1
