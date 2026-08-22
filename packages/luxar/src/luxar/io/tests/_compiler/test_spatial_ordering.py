"""Tests for luxar.io._compiler.spatial_ordering: direct unit tests, plus one
end-to-end compiler round trip through the scene API."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.encoding.decoder import ArrayDecoder
from luxar.io._compiler.context import OrderingCtx
from luxar.io._compiler.spatial_ordering.lines import build_lines_ordering
from luxar.io._compiler.spatial_ordering.points import (
    build_points_ordering,
    write_points_ordering_to_zarr,
)
from luxar.io._ordering.bounds import _BARRIER_BOUND_EPS, _store_outward_f32
from luxar.io._ordering.points import sort_points_compound
from luxar.io.ordering import sort_splats_spatial
from luxar.io.reader import DEFAULT_COMP
from luxar.typing_utils.constants import DEFAULT_POINT_RADIUS


def _disabled() -> OrderingCtx:
    return OrderingCtx(enable_spatial_index=False, ordering_method="morton")


def _enabled() -> OrderingCtx:
    return OrderingCtx(enable_spatial_index=True, ordering_method="morton")


def test_build_points_ordering_none_when_disabled() -> None:
    pos = np.random.rand(10, 3).astype(np.float32)
    assert build_points_ordering(pos, 10, 3, None, _disabled(), zarr.group()) is None


def test_build_points_ordering_none_without_scene_dimensions() -> None:
    pos = np.random.rand(10, 3).astype(np.float32)
    # enabled, but the store carries no scene_dimensions attr → skipped
    assert build_points_ordering(pos, 10, 3, None, _enabled(), zarr.group()) is None


def test_build_lines_ordering_none_when_disabled() -> None:
    verts = np.random.rand(6, 3).astype(np.float32)
    segs = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.uint32)
    out = build_lines_ordering(verts, segs, 1.0, 6, 3, 3, _disabled(), zarr.group())
    assert out is None


def test_write_points_ordering_to_zarr_roundtrip() -> None:
    g = zarr.group()
    chunk_bounds = np.array(
        [[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]], dtype=np.float32
    )  # (1, 3, 2)
    ordering_data = {
        "ordering": "morton",
        "slice_dims": [],
        "ordering_dims": [0, 1, 2],
        "ordering_min": [0.0, 0.0, 0.0],
        "ordering_max": [1.0, 1.0, 1.0],
        "ordering_bits_per_dim": 10,
        "chunk_size": 1024,
        "chunk_bounds": chunk_bounds,
    }
    write_points_ordering_to_zarr(g, ordering_data, DEFAULT_COMP)
    assert g.attrs["ordering"] == "morton"
    assert g.attrs["chunk_size"] == 1024
    assert "chunk_bounds" in g
    np.testing.assert_allclose(g["chunk_bounds"][:], chunk_bounds)


def test_stored_chunk_bounds_carry_the_default_radius_without_radii(
    tmp_path: Path,
) -> None:
    """End-to-end: a radii-less node's ON-DISK chunk_bounds carry the default pad.

    Every other test of this behaviour calls ``compute_chunk_bounds_points``
    directly; this one drives the whole authoring path and then reads the stored
    array back. The writer path that sees ``radii=None`` is
    ``write_points_multi_lod``, reached whenever ``add_points`` builds an
    additive ladder before the authoring default is materialized — the
    ``additive_lod=`` used here, the default stream ladder ``substitutive_lod=``
    composes for its finest child, or ``partition=`` parts carrying a ladder of
    their own. A plain ``add_points`` (and a bare ``partition=``, whose parts
    recurse back through it) materializes ``DEFAULT_POINT_RADIUS`` first.

    The data pins one point at the spatial origin and one at ``(4, 4, 4)``, so
    whichever additive level each lands in, the tightest stored bound across all
    levels is exactly ``[0 - r, 4 + r]`` per spatial axis. The old code padded a
    no-radii chunk by ``max(1% of the chunk's range, 0.01) <= 0.04`` here, so the
    ``-0.5`` / ``4.5`` asserted below cannot be produced by it.

    The ladder is SEEDED: ``additive_lod`` defaults to ``method="random"``, and
    an unseeded split decides whether one level holds both pinned extremes —
    which is the only case where the quantisation pad below is nonzero, so the
    assertions were nondeterministic without it.
    """
    dims = Dimensions(
        [
            Dimension("time", unit="s", display=False, discrete=True),
            Dimension("x", unit="um", display=True),
            Dimension("y", unit="um", display=True),
            Dimension("z", unit="um", display=True),
        ]
    )

    rng = np.random.default_rng(0)
    positions = np.zeros((64, 4), dtype=np.float32)
    positions[:, 0] = rng.integers(0, 4, 64)
    positions[:, 1:] = (rng.random((64, 3)) * 4.0).astype(np.float32)
    # Pin the per-axis extremes so the expected bounds are exact.
    positions[0] = [0.0, 0.0, 0.0, 0.0]
    positions[1] = [3.0, 4.0, 4.0, 4.0]

    out = tmp_path / "no_radii.luxar.zarr"
    with LuxarZarrCompiler(out, enable_spatial_index=True) as compiler:
        scene = compiler.create_scene(dimensions=dims)
        scene.add_points("pts", positions, additive_lod={"n_lods": 3, "seed": 1655})

    node = zarr.open_group(out, mode="r")["pts"]
    levels = [node[k] for k in sorted(node.group_keys())]
    assert levels, "expected additive sub-LOD levels under the points node"

    for level in levels:
        # No max_radius is stamped without radii — this is what makes the
        # reader fall back to defaultMaxRadius=0.1, and why the bound has to be
        # honest on its own.
        assert level.attrs["has_radii"] is False
        assert "max_radius" not in level.attrs

    bounds = np.concatenate([level["chunk_bounds"][:] for level in levels], axis=0)
    assert bounds.shape[1:] == (4, 2)

    # Every bound also carries the encoder's uint16 round-trip slack on top of
    # the pad below (issue #1655): the spatial axes span 4 units, so that is at
    # most 4/131070 = 3.1e-5 per axis, plus up to one float32 ULP from the
    # OUTWARD store that narrows the float64 interval (`_store_outward_f32`) —
    # and it can only widen a bound, never tighten it. The barrier axis is a
    # gridded integer time index, so its slack is exactly 0.
    _quant = 4.0 / 131070.0
    _ulp = float(np.spacing(np.float32(4.5)))

    # Spatial axes (1, 2, 3): padded by exactly DEFAULT_POINT_RADIUS = 0.5.
    # Asserted DIRECTIONALLY rather than symmetrically: a bound that lost the
    # radius fails the tight side, a bound wider than radius + slack + one ULP
    # fails the loose side.
    for dim in (1, 2, 3):
        lo = bounds[:, dim, 0].min()
        hi = bounds[:, dim, 1].max()
        assert -DEFAULT_POINT_RADIUS - _quant - _ulp <= lo <= -DEFAULT_POINT_RADIUS
        assert (
            4.0 + DEFAULT_POINT_RADIUS
            <= hi
            <= 4.0 + DEFAULT_POINT_RADIUS + _quant + _ulp
        )

    # Discrete/barrier axis 0: only the float-boundary epsilon, never the
    # radius — and no quantisation slack either, because a gridded integer axis
    # is snapped to round-trip bit-exactly.
    assert bounds[:, 0, 0].min() == pytest.approx(0.0 - _BARRIER_BOUND_EPS)
    assert bounds[:, 0, 1].max() == pytest.approx(3.0 + _BARRIER_BOUND_EPS)


def _decode(node: zarr.Group, name: str) -> np.ndarray:
    """Read an array back through the REAL decode path the viewer uses."""
    return ArrayDecoder().decode(node[name], node)


def _violations(
    coords: np.ndarray, bounds: np.ndarray, chunk_size: int
) -> tuple[int, int]:
    """Rows (and chunks) whose coordinate falls outside its own chunk's bound.

    ``coords`` is in the SAME row order the chunk grid was built from, so rows
    ``[k·chunk_size, (k+1)·chunk_size)`` belong to chunk ``k``.
    """
    bad_rows = 0
    bad_chunks = 0
    for k in range(bounds.shape[0]):
        block = coords[k * chunk_size : (k + 1) * chunk_size]
        outside = (block < bounds[k, :, 0]) | (block > bounds[k, :, 1])
        n = int(outside.any(axis=1).sum())
        bad_rows += n
        bad_chunks += 1 if n else 0
    return bad_rows, bad_chunks


def _wide_scene_dims(barrier: str) -> Dimensions:
    """3 displayed spatial dims + one non-displayed barrier dim."""
    return Dimensions(
        [
            Dimension(barrier, unit="s", display=False, discrete=True),
            Dimension("x", unit="um", display=True),
            Dimension("y", unit="um", display=True),
            Dimension("z", unit="um", display=True),
        ]
    )


def test_stored_points_bounds_contain_the_DECODED_positions(tmp_path: Path) -> None:
    """End-to-end: every decoded position lies inside its own chunk's bound.

    The bound builders see the AUTHORED positions, but under the default AUTO
    encoding the positions are stored as per-axis uint16 fixed point, so what
    the reader compares its slice query against is a position that can have
    moved half a quantum — ``extent/131070``, i.e. 7.6e-3 on the 1000-wide axes
    used here. A chunk whose own extremum moved outward is a chunk the reader
    never fetches for a query at that edge: the points vanish, silently
    (issue #1655). The compiler now pads the bounds by the encoder's own
    round-trip slack, so containment holds against the DECODED positions.

    ``radii=0.0`` (the accepted degenerate-points contract) is deliberate: the
    default ``DEFAULT_POINT_RADIUS`` pad of 0.5 is 65× the slack and would hide
    the defect entirely. The barrier axis carries only ``_BARRIER_BOUND_EPS``
    (1e-3), which is 7.6× too small on its own — see the count asserted below.

    Measured on the unpatched code (the compiler passing ``coord_slack=None``):
    6 of 6 chunks and 18 of 12,000 rows fell outside their own bound.
    """
    rng = np.random.default_rng(1655)
    n = 12_000
    positions = np.empty((n, 4), dtype=np.float32)
    # Non-gridded barrier axis with a wide extent: irregular acquisition
    # timestamps, not a stacked integer index. Nothing snaps it exact.
    positions[:, 0] = rng.random(n) * 1000.0
    positions[:, 1:] = rng.random((n, 3)) * 1000.0

    out = tmp_path / "wide_points.luxar.zarr"
    with LuxarZarrCompiler(out, enable_spatial_index=True) as compiler:
        scene = compiler.create_scene(dimensions=_wide_scene_dims("timestamp"))
        scene.add_points("pts", positions, radii=0.0)

    node = zarr.open_group(out, mode="r")["pts"]
    assert node["positions"].attrs["encoding"]["name"] == "linear_perchannel_u16"
    decoded = _decode(node, "positions")
    bounds = node["chunk_bounds"][:]
    chunk_size = int(node.attrs["chunk_size"])
    assert bounds.shape[0] > 1, "want several chunks, not one"

    bad_rows, bad_chunks = _violations(decoded, bounds, chunk_size)
    assert (bad_rows, bad_chunks) == (0, 0)

    # Not vacuous: the pad really is the half-quantum, on every axis. On the
    # barrier axis it lands ON TOP of _BARRIER_BOUND_EPS (which is 7.6x too
    # small on its own), not instead of it. The tolerance is one float32 ULP at
    # 1000 (6.1e-5), which the outward store can add.
    slack = 1000.0 / 131070.0
    for d in range(4):
        expected = slack + (_BARRIER_BOUND_EPS if d == 0 else 0.0)
        pad = float(bounds[:, d, 1].max()) - float(decoded[:, d].max())
        assert pad == pytest.approx(expected, abs=1e-4)


def test_stored_lines_bounds_contain_the_DECODED_vertices(tmp_path: Path) -> None:
    """The same claim for both Lines bound sets (vertex AND segment).

    Vertex bounds carry no footprint pad at all, so they are the tightest
    surface in the codebase and the most exposed to the quantisation gap;
    segment bounds carry only the endpoint width, here 1e-4 (widths must be
    > 0), 76× smaller than the slack.

    Measured on the unpatched code: 15 of 15 vertex chunks (57 of 40,000 rows)
    and 5 of 5 segment chunks (19 of 20,000 segment END points) fell outside
    their own bound. Segment START points happened to survive here — the sort
    key leads with them, so a chunk's start-point extremes tend to be the global
    ones, which quantize exactly; the ends scatter and do not. Both are asserted
    anyway: which endpoint is exposed is an accident of the ordering.
    """
    rng = np.random.default_rng(16550)
    n_seg = 20_000
    base = rng.random((n_seg, 4)) * 1000.0
    # Short, spatially coherent segments — an incoherent polyline would give
    # every chunk a bound spanning the whole volume and prove nothing.
    ends = base + rng.standard_normal((n_seg, 4)) * 2.0
    vertices = np.empty((2 * n_seg, 4), dtype=np.float32)
    vertices[0::2] = base
    vertices[1::2] = ends

    out = tmp_path / "wide_lines.luxar.zarr"
    with LuxarZarrCompiler(out, enable_spatial_index=True) as compiler:
        scene = compiler.create_scene(dimensions=_wide_scene_dims("timestamp"))
        scene.add_lines("lns", vertices, widths=1e-4, line_type="segments")

    node = zarr.open_group(out, mode="r")["lns"]
    assert node["vertices"].attrs["encoding"]["name"] == "linear_perchannel_u16"
    decoded = _decode(node, "vertices")
    segments = np.asarray(_decode(node, "segments")).astype(np.int64)

    vertex_chunk = int(node.attrs["vertex_ordering"]["chunk_size"])
    vertex_bounds = node["vertex_chunk_bounds"][:]
    assert vertex_bounds.shape[0] > 1
    assert _violations(decoded, vertex_bounds, vertex_chunk) == (0, 0)

    # Segment bounds are in D-space over BOTH endpoints of each segment, so
    # check each endpoint against its segment chunk's bound.
    segment_chunk = int(node.attrs["segment_ordering"]["chunk_size"])
    segment_bounds = node["segment_chunk_bounds"][:]
    for endpoint in (0, 1):
        coords = decoded[segments[:, endpoint]]
        assert _violations(coords, segment_bounds, segment_chunk) == (0, 0)


def test_stored_points_bounds_contain_the_DECODED_radii(tmp_path: Path) -> None:
    """Every decoded point footprint stays inside its stored chunk bound."""
    rng = np.random.default_rng(1871)
    n = 20_000
    positions = (rng.random((n, 3)) * 1000.0).astype(np.float32)
    radii = rng.uniform(0.1, 5.0, n).astype(np.float32)

    out = tmp_path / "quantised_radii.luxar.zarr"
    with LuxarZarrCompiler(out, enable_spatial_index=True) as compiler:
        scene = compiler.create_scene(
            dimensions=Dimensions(
                [
                    Dimension("x", unit="um", display=True),
                    Dimension("y", unit="um", display=True),
                    Dimension("z", unit="um", display=True),
                ]
            )
        )
        scene.add_points("pts", positions, radii=radii)

    node = zarr.open_group(out, mode="r")["pts"]
    assert node["radii"].attrs["encoding"]["name"] == "bounded_scalar_uint8"
    decoded_positions = _decode(node, "positions")
    decoded_radii = _decode(node, "radii")
    sort_order, _ = sort_points_compound(
        positions,
        [
            Dimension("x", unit="um", display=True),
            Dimension("y", unit="um", display=True),
            Dimension("z", unit="um", display=True),
        ],
    )
    assert float((decoded_radii - radii[sort_order]).max()) > 0.009

    bounds = node["chunk_bounds"][:]
    chunk_size = int(node.attrs["chunk_size"])
    for k in range(bounds.shape[0]):
        start = k * chunk_size
        stop = min(start + chunk_size, n)
        block = decoded_positions[start:stop]
        radius = decoded_radii[start:stop, None]
        assert bool((block - radius >= bounds[k, :, 0]).all())
        assert bool((block + radius <= bounds[k, :, 1]).all())


def test_stored_points_bounds_contain_geolog_DECODED_radii(tmp_path: Path) -> None:
    rng = np.random.default_rng(1871001)
    n = 20_000
    positions = (rng.random((n, 3)) * 1000.0).astype(np.float32)
    radii = np.geomspace(1e-6, 100.0, n, dtype=np.float32)

    out = tmp_path / "geolog_radii.luxar.zarr"
    with LuxarZarrCompiler(out, enable_spatial_index=True) as compiler:
        scene = compiler.create_scene(
            dimensions=Dimensions(
                [
                    Dimension("x", unit="um", display=True),
                    Dimension("y", unit="um", display=True),
                    Dimension("z", unit="um", display=True),
                ]
            )
        )
        scene.add_points("pts", positions, radii=radii)

    node = zarr.open_group(out, mode="r")["pts"]
    assert node["radii"].attrs["encoding"]["name"] == "geolog_scalar_uint16"
    decoded_positions = _decode(node, "positions")
    decoded_radii = _decode(node, "radii")
    sort_order, _ = sort_points_compound(
        positions,
        [
            Dimension("x", unit="um", display=True),
            Dimension("y", unit="um", display=True),
            Dimension("z", unit="um", display=True),
        ],
    )
    assert float((decoded_radii - radii[sort_order]).max()) > 0.01
    bounds = node["chunk_bounds"][:]
    chunk_size = int(node.attrs["chunk_size"])
    for k in range(bounds.shape[0]):
        start = k * chunk_size
        stop = min(start + chunk_size, n)
        block = decoded_positions[start:stop]
        radius = decoded_radii[start:stop, None]
        assert bool((block - radius >= bounds[k, :, 0]).all())
        assert bool((block + radius <= bounds[k, :, 1]).all())


def test_footprint_writes_do_not_cross_deduplicate_on_sharpness(tmp_path: Path) -> None:
    rng = np.random.default_rng(17511871)
    n = 5000
    positions = (rng.random((n, 3)) * 1000.0).astype(np.float32)
    footprint = np.linspace(0.4, 0.5, n, dtype=np.float32)

    out = tmp_path / "footprint_dedup.luxar.zarr"
    with LuxarZarrCompiler(out, enable_spatial_index=True) as compiler:
        scene = compiler.create_scene(
            dimensions=Dimensions(
                [
                    Dimension("x", unit="um", display=True),
                    Dimension("y", unit="um", display=True),
                    Dimension("z", unit="um", display=True),
                ]
            )
        )
        scene.add_points("point_source", positions, sharpness=footprint)
        scene.add_points("point_target", positions, radii=footprint)
        scene.add_lines(
            "line_source",
            positions,
            widths=1.0,
            sharpness=footprint,
            line_type="segments",
        )
        scene.add_lines(
            "line_target", positions, widths=footprint, line_type="segments"
        )

    store = zarr.open_group(out, mode="r")
    assert store["point_target/radii"].attrs["encoding"]["name"] == (
        "bounded_scalar_uint8"
    )
    assert store["line_target/widths"].attrs["encoding"]["name"] == (
        "bounded_scalar_uint8"
    )


def test_stored_line_bounds_contain_the_DECODED_widths(tmp_path: Path) -> None:
    """Every decoded segment footprint stays inside its stored chunk bound."""
    rng = np.random.default_rng(18710)
    n_segments = 20_000
    starts = rng.random((n_segments, 3)) * 1000.0
    ends = starts + rng.standard_normal((n_segments, 3)) * 2.0
    vertices = np.empty((2 * n_segments, 3), dtype=np.float32)
    vertices[0::2] = starts
    vertices[1::2] = ends
    widths = rng.uniform(0.1, 5.0, 2 * n_segments).astype(np.float32)

    out = tmp_path / "quantised_widths.luxar.zarr"
    with LuxarZarrCompiler(out, enable_spatial_index=True) as compiler:
        scene = compiler.create_scene(
            dimensions=Dimensions(
                [
                    Dimension("x", unit="um", display=True),
                    Dimension("y", unit="um", display=True),
                    Dimension("z", unit="um", display=True),
                ]
            )
        )
        scene.add_lines("lns", vertices, widths=widths, line_type="segments")

    node = zarr.open_group(out, mode="r")["lns"]
    assert node["widths"].attrs["encoding"]["name"] == "bounded_scalar_uint8"
    decoded_vertices = _decode(node, "vertices")
    decoded_widths = _decode(node, "widths")
    segments = np.asarray(_decode(node, "segments")).astype(np.int64)
    vertex_sort_order, _ = sort_points_compound(
        vertices,
        [
            Dimension("x", unit="um", display=True),
            Dimension("y", unit="um", display=True),
            Dimension("z", unit="um", display=True),
        ],
    )
    sorted_authored = widths[vertex_sort_order]
    assert float((decoded_widths - sorted_authored).max()) > 0.009

    bounds = node["segment_chunk_bounds"][:]
    chunk_size = int(node.attrs["segment_ordering"]["chunk_size"])
    for k in range(bounds.shape[0]):
        block = segments[k * chunk_size : (k + 1) * chunk_size]
        p1 = decoded_vertices[block[:, 0]]
        p2 = decoded_vertices[block[:, 1]]
        width = np.maximum(decoded_widths[block[:, 0]], decoded_widths[block[:, 1]])[
            :, None
        ]
        assert bool((np.minimum(p1 - width, p2 - width) >= bounds[k, :, 0]).all())
        assert bool((np.maximum(p1 + width, p2 + width) <= bounds[k, :, 1]).all())


def test_stored_lines_bounds_contain_LUT_ELIGIBLE_decoded_vertices(
    tmp_path: Path,
) -> None:
    """A LUT-ELIGIBLE lines node is still quantised, so it still needs the pad.

    ``ArrayEncoder.encodes_as_lut`` answers the LUT question alone; ``encode``
    also skips LUT under an explicit ``allow_lut=False``, and the lines writer
    passes exactly that for ``vertices`` (the spatial-index loader reads that
    array as raw chunked zarr). So a vertices array with ≤256 distinct values —
    a coarse irregular palette, e.g. coordinates snapped to a measured stage
    grid — is NOT a LUT on disk: it goes to ``linear_perchannel_u16`` like any
    other. Asking the encoder for the slack with the default ``allow_lut=True``
    reported "exact" and both bound sets got zero pad, which is precisely the
    silent geometry loss #1655 is about, with vertex bounds (no footprint pad at
    all) fully exposed.

    Measured on THIS data with the glue asking ``allow_lut=True``: 30 of 44
    vertex chunks (8,203 of 120,000 rows) and 7 of 15 segment chunks (487 of
    60,000 segment END points) fell outside their own bound; with
    ``allow_lut=False`` all four counts are 0. Segment START points survive
    here for the same reason as in the test above — the sort key leads with
    them — and are asserted anyway.
    """
    rng = np.random.default_rng(1655001)
    # 250 irregularly spaced values: LUT-eligible (≤256 distinct, and
    # size >= 4K), but on no regular grid, so the per-axis grid snap cannot
    # rescue any axis either.
    palette = np.sort(rng.random(250) * 1000.0).astype(np.float32)
    n_seg = 60_000
    base_idx = rng.integers(0, palette.size, (n_seg, 4))
    # Short segments: neighbouring palette rungs, so chunk bounds stay tight.
    end_idx = np.clip(base_idx + rng.integers(-2, 3, (n_seg, 4)), 0, palette.size - 1)
    vertices = np.empty((2 * n_seg, 4), dtype=np.float32)
    vertices[0::2] = palette[base_idx]
    vertices[1::2] = palette[end_idx]

    out = tmp_path / "lut_lines.luxar.zarr"
    with LuxarZarrCompiler(out, enable_spatial_index=True) as compiler:
        scene = compiler.create_scene(dimensions=_wide_scene_dims("timestamp"))
        scene.add_lines("lns", vertices, widths=1e-4, line_type="segments")

    node = zarr.open_group(out, mode="r")["lns"]
    # The premise: eligible for a LUT, but not stored as one.
    from luxar.encoding.encoder import ArrayEncoder
    from luxar.encoding.semantic_types import SemanticType

    assert ArrayEncoder().encodes_as_lut(vertices, SemanticType.COORDINATE)
    assert node["vertices"].attrs["encoding"]["name"] == "linear_perchannel_u16"

    decoded = _decode(node, "vertices")
    segments = np.asarray(_decode(node, "segments")).astype(np.int64)

    vertex_chunk = int(node.attrs["vertex_ordering"]["chunk_size"])
    vertex_bounds = node["vertex_chunk_bounds"][:]
    assert vertex_bounds.shape[0] > 1
    assert _violations(decoded, vertex_bounds, vertex_chunk) == (0, 0)

    segment_chunk = int(node.attrs["segment_ordering"]["chunk_size"])
    segment_bounds = node["segment_chunk_bounds"][:]
    for endpoint in (0, 1):
        coords = decoded[segments[:, endpoint]]
        assert _violations(coords, segment_bounds, segment_chunk) == (0, 0)


def test_lut_encoded_points_bounds_get_no_slack(tmp_path: Path) -> None:
    """The complement: a points node that REALLY stores a LUT is exempt.

    ``write_positions`` leaves ``allow_lut`` at its default, so a LUT-eligible
    positions array is stored verbatim as ``lut_uint8`` and round-trips
    bit-exactly — no pad is needed and none is added. Pinning that keeps the
    exemption a deliberate choice rather than an accident: if the positions
    writer ever blocks LUT the way the lines writer does, this test goes red at
    the same time as the glue's ``allow_lut`` argument becomes wrong.
    """
    rng = np.random.default_rng(1655002)
    palette = np.sort(rng.random(250) * 1000.0).astype(np.float32)
    positions = palette[rng.integers(0, palette.size, (12_000, 3))].astype(np.float32)

    dims = Dimensions(
        [
            Dimension("x", unit="um", display=True),
            Dimension("y", unit="um", display=True),
            Dimension("z", unit="um", display=True),
        ]
    )
    out = tmp_path / "lut_points.luxar.zarr"
    with LuxarZarrCompiler(out, enable_spatial_index=True) as compiler:
        scene = compiler.create_scene(dimensions=dims)
        scene.add_points("pts", positions, radii=0.0)

    node = zarr.open_group(out, mode="r")["pts"]
    assert node["positions"].attrs["encoding"]["name"] == "lut_uint8"

    decoded = _decode(node, "positions")
    bounds = node["chunk_bounds"][:]
    chunk_size = int(node.attrs["chunk_size"])
    assert bounds.shape[0] > 1

    # Exact round trip — every decoded value is verbatim one of the authored
    # palette rungs — so containment is trivially satisfied...
    assert bool(np.isin(decoded, palette).all())
    assert _violations(decoded, bounds, chunk_size) == (0, 0)
    # ...and the bound is the authored interval itself, with NO slack: a padded
    # bound here would mean the exemption had been dropped.
    for k in range(bounds.shape[0]):
        block = decoded[k * chunk_size : (k + 1) * chunk_size]
        for d in range(3):
            lo32, hi32 = _store_outward_f32(
                float(block[:, d].min()), float(block[:, d].max())
            )
            assert bounds[k, d, 0] == lo32
            assert bounds[k, d, 1] == hi32


def test_ordering_without_a_dataset_ctx_gets_authored_bounds(tmp_path: Path) -> None:
    """A DIRECT caller of the glue (ordering ENABLED, ``dataset_ctx=None``).

    Documented behaviour: with no encoder to ask, the builders get
    ``coord_slack=None`` and the bounds are the AUTHORED ones, byte for byte.
    Everything in-tree passes a ``dataset_ctx``, so without this the contract
    the two glue docstrings state was untested.
    """
    del tmp_path
    rng = np.random.default_rng(1655003)
    positions = (rng.random((4_000, 3)) * 1000.0).astype(np.float32)

    store = zarr.group()
    store.attrs["scene_dimensions"] = Dimensions(
        [
            Dimension("x", unit="um", display=True),
            Dimension("y", unit="um", display=True),
            Dimension("z", unit="um", display=True),
        ]
    ).to_dict()

    out = build_points_ordering(positions, 4_000, 3, 0.0, _enabled(), store)
    assert out is not None
    sorted_positions = out["sorted_positions"]
    bounds = out["chunk_bounds"]
    chunk_size = out["chunk_size"]
    for k in range(bounds.shape[0]):
        block = sorted_positions[k * chunk_size : (k + 1) * chunk_size]
        for d in range(3):
            lo32, hi32 = _store_outward_f32(
                float(block[:, d].min()), float(block[:, d].max())
            )
            assert bounds[k, d, 0] == lo32
            assert bounds[k, d, 1] == hi32

    # Lines likewise: both bound sets are authored-coordinate bounds.
    n_seg = 2_000
    base = rng.random((n_seg, 3)) * 1000.0
    ends = base + rng.standard_normal((n_seg, 3)) * 2.0
    vertices = np.empty((2 * n_seg, 3), dtype=np.float32)
    vertices[0::2] = base
    vertices[1::2] = ends
    segments = np.arange(2 * n_seg, dtype=np.uint32).reshape(n_seg, 2)

    lines_out = build_lines_ordering(
        vertices, segments, 1e-4, 2 * n_seg, 3, n_seg, _enabled(), store
    )
    assert lines_out is not None
    sorted_vertices = lines_out["sorted_vertices"]
    vbounds = lines_out["vertex_chunk_bounds"]
    vchunk = lines_out["vertex_ordering"]["chunk_size"]
    for k in range(vbounds.shape[0]):
        block = sorted_vertices[k * vchunk : (k + 1) * vchunk]
        for d in range(3):
            lo32, hi32 = _store_outward_f32(
                float(block[:, d].min()), float(block[:, d].max())
            )
            assert vbounds[k, d, 0] == lo32
            assert vbounds[k, d, 1] == hi32


def test_gridded_barrier_axis_bounds_are_untouched(tmp_path: Path) -> None:
    """A stacked integer time axis gets slack 0 — its bounds do not move.

    The complement of the two tests above, and the reason the slack is
    PER-AXIS: an ordinary integer time/channel axis is grid-snapped by the
    encoder and round-trips bit-exactly, so widening its barrier bound would
    buy nothing and cost over-fetch into the neighbouring category. Its stored
    bound must still be exactly ``[t_min - eps, t_max + eps]`` — pinned here
    chunk by chunk against the outward float32 store, not just in aggregate —
    while the wide spatial axes beside it do get the pad.
    """
    rng = np.random.default_rng(99)
    n = 12_000
    positions = np.empty((n, 4), dtype=np.float32)
    positions[:, 0] = rng.integers(0, 10, n)  # gridded: 10 timepoints
    positions[:, 1:] = rng.random((n, 3)) * 1000.0

    out = tmp_path / "gridded_time.luxar.zarr"
    with LuxarZarrCompiler(out, enable_spatial_index=True) as compiler:
        scene = compiler.create_scene(dimensions=_wide_scene_dims("time"))
        scene.add_points("pts", positions, radii=0.0)

    node = zarr.open_group(out, mode="r")["pts"]
    decoded = _decode(node, "positions")
    bounds = node["chunk_bounds"][:]
    chunk_size = int(node.attrs["chunk_size"])

    # The gridded axis stores exactly, so its bound is the authored interval
    # plus the float-boundary epsilon and nothing else.
    for k in range(bounds.shape[0]):
        block = decoded[k * chunk_size : (k + 1) * chunk_size, 0]
        lo32, hi32 = _store_outward_f32(
            float(block.min()) - _BARRIER_BOUND_EPS,
            float(block.max()) + _BARRIER_BOUND_EPS,
        )
        assert bounds[k, 0, 0] == lo32
        assert bounds[k, 0, 1] == hi32

    # ...while the continuous spatial axes beside it still got the pad, so the
    # per-axis-ness of the slack is what is being pinned, not its absence.
    slack = 1000.0 / 131070.0
    for d in (1, 2, 3):
        pad = float(bounds[:, d, 1].max()) - float(decoded[:, d].max())
        assert pad == pytest.approx(slack, rel=0.05)
    assert _violations(decoded, bounds, chunk_size) == (0, 0)


def test_stored_gsplat_bounds_contain_decoded_continuous_barrier_centers(
    tmp_path: Path,
) -> None:
    """A non-gridded barrier bound contains the quantized center it indexes.

    AUTO stores centers as per-axis uint16 fixed point. On the continuous
    1000-wide time axis below, its half-quantum is about 7.6e-3: wider than
    ``_BARRIER_BOUND_EPS`` but far below the sigma rail's 5.0 threshold. Before
    the bound used the encoder's round-trip slack, this deterministic corpus
    put 11 decoded centers outside 9 of 12 chunk bounds, with a worst excursion
    of 6.26e-3.
    """
    rng = np.random.default_rng(0)
    n_splats = 12_000
    centers = np.empty((n_splats, 4), dtype=np.float32)
    centers[:, 0] = rng.uniform(0.0, 1000.0, n_splats).astype(np.float32)
    centers[:, 1:] = rng.uniform(-10.0, 10.0, (n_splats, 3)).astype(np.float32)
    centers[0, 0] = 0.0
    centers[1, 0] = 1000.0

    cholesky = np.zeros((n_splats, 10), dtype=np.float32)
    cholesky[:, [0, 2, 5, 9]] = 5.0

    out = tmp_path / "continuous_barrier_gsplats.luxar.zarr"
    with LuxarZarrCompiler(out, enable_spatial_index=True) as compiler:
        compiler.create_scene(dimensions=_wide_scene_dims("time"))
        compiler.write_gsplats(
            "gsplats",
            centers=centers,
            amplitudes=1.0,
            cholesky_factors=cholesky,
        )

    node = zarr.open_group(out, mode="r")["gsplats"]
    assert node.attrs["slice_dims"] == [0]
    assert node["centers"].attrs["encoding"]["name"] == "linear_perchannel_u16"

    decoded = _decode(node, "centers")
    bounds = node["chunk_bounds"][:]
    chunk_size = int(node.attrs["chunk_size"])
    assert len(bounds) > 1
    assert _violations(decoded, bounds, chunk_size) == (0, 0)

    sort_indices, _ = sort_splats_spatial(centers, method="hilbert", slice_dims=[0])
    authored = centers[sort_indices, 0]
    slack = 1000.0 / 131070.0
    barrier_pad = _BARRIER_BOUND_EPS + slack
    assert barrier_pad == pytest.approx(0.00862951094834821)
    for k in range(bounds.shape[0]):
        block = authored[k * chunk_size : (k + 1) * chunk_size]
        lo32, hi32 = _store_outward_f32(
            float(block.min()) - barrier_pad,
            float(block.max()) + barrier_pad,
        )
        assert bounds[k, 0, 0] == lo32
        assert bounds[k, 0, 1] == hi32

    # The large sigma footprint already covers spatial-axis displacement; this
    # regression is specifically the barrier axis that gets no sigma expansion.
    spatial = decoded[:, 1:]
    spatial_bounds = bounds[:, 1:, :]
    assert _violations(spatial, spatial_bounds, chunk_size) == (0, 0)


def test_gridded_gsplat_barrier_bounds_keep_only_the_epsilon(tmp_path: Path) -> None:
    """Grid-snapped gsplat centers get no coordinate-slack over-padding."""
    rng = np.random.default_rng(1870)
    n_splats = 12_000
    centers = np.empty((n_splats, 4), dtype=np.float32)
    centers[:, 0] = rng.integers(0, 10, n_splats)
    centers[:, 1:] = rng.uniform(-10.0, 10.0, (n_splats, 3)).astype(np.float32)
    cholesky = np.zeros((n_splats, 10), dtype=np.float32)
    cholesky[:, 0] = 1e-7
    cholesky[:, [2, 5, 9]] = 5.0

    out = tmp_path / "gridded_barrier_gsplats.luxar.zarr"
    with LuxarZarrCompiler(out, enable_spatial_index=True) as compiler:
        compiler.create_scene(dimensions=_wide_scene_dims("time"))
        compiler.write_gsplats("gsplats", centers, 1.0, cholesky)

    node = zarr.open_group(out, mode="r")["gsplats"]
    assert node["centers"].attrs["encoding"]["name"] == "linear_perchannel_u16"
    decoded = _decode(node, "centers")
    bounds = node["chunk_bounds"][:]
    chunk_size = int(node.attrs["chunk_size"])
    for k in range(bounds.shape[0]):
        block = decoded[k * chunk_size : (k + 1) * chunk_size, 0]
        lo32, hi32 = _store_outward_f32(
            float(block.min()) - _BARRIER_BOUND_EPS,
            float(block.max()) + _BARRIER_BOUND_EPS,
        )
        assert bounds[k, 0, 0] == lo32
        assert bounds[k, 0, 1] == hi32


def test_lut_gsplat_barrier_bounds_keep_only_the_epsilon(tmp_path: Path) -> None:
    """LUT-stored centers are exact and receive no coordinate-slack pad."""
    rng = np.random.default_rng(1872)
    n_splats = 12_000
    palette = np.array(
        [0.0, 3.7, 19.0, 55.0, 132.5, 610.0, 799.5, 1000.0],
        dtype=np.float32,
    )
    centers = rng.choice(palette, size=(n_splats, 4)).astype(np.float32)
    cholesky = np.zeros((n_splats, 10), dtype=np.float32)
    cholesky[:, [0, 2, 5, 9]] = 5.0

    out = tmp_path / "lut_barrier_gsplats.luxar.zarr"
    with LuxarZarrCompiler(out, enable_spatial_index=True) as compiler:
        compiler.create_scene(dimensions=_wide_scene_dims("time"))
        compiler.write_gsplats("gsplats", centers, 1.0, cholesky)

    node = zarr.open_group(out, mode="r")["gsplats"]
    assert node["centers"].attrs["encoding"]["name"] == "lut_uint8"
    decoded = _decode(node, "centers")
    bounds = node["chunk_bounds"][:]
    chunk_size = int(node.attrs["chunk_size"])
    assert _violations(decoded, bounds, chunk_size) == (0, 0)
    for k in range(bounds.shape[0]):
        block = decoded[k * chunk_size : (k + 1) * chunk_size, 0]
        lo32, hi32 = _store_outward_f32(
            float(block.min()) - _BARRIER_BOUND_EPS,
            float(block.max()) + _BARRIER_BOUND_EPS,
        )
        assert bounds[k, 0, 0] == lo32
        assert bounds[k, 0, 1] == hi32


def test_escalated_gsplat_barrier_bounds_keep_only_the_epsilon(tmp_path: Path) -> None:
    """Float32-escalated centers are exact and receive no second slack pad."""
    rng = np.random.default_rng(1871)
    n_splats = 12_000
    centers = np.empty((n_splats, 4), dtype=np.float32)
    centers[:, 0] = rng.uniform(0.0, 1000.0, n_splats).astype(np.float32)
    centers[:, 1:] = rng.uniform(-10.0, 10.0, (n_splats, 3)).astype(np.float32)
    centers[0, 0] = 0.0
    centers[1, 0] = 1000.0
    cholesky = np.zeros((n_splats, 10), dtype=np.float32)
    cholesky[:, 0] = 1e-7
    cholesky[:, [2, 5, 9]] = 5.0

    out = tmp_path / "escalated_barrier_gsplats.luxar.zarr"
    with pytest.warns(UserWarning, match="stored as float32"):
        with LuxarZarrCompiler(out, enable_spatial_index=True) as compiler:
            compiler.create_scene(dimensions=_wide_scene_dims("time"))
            compiler.write_gsplats("gsplats", centers, 1.0, cholesky)

    node = zarr.open_group(out, mode="r")["gsplats"]
    assert node["centers"].attrs["encoding"]["name"] == "float32"
    decoded = _decode(node, "centers")
    bounds = node["chunk_bounds"][:]
    chunk_size = int(node.attrs["chunk_size"])
    for k in range(bounds.shape[0]):
        block = decoded[k * chunk_size : (k + 1) * chunk_size, 0]
        lo32, hi32 = _store_outward_f32(
            float(block.min()) - _BARRIER_BOUND_EPS,
            float(block.max()) + _BARRIER_BOUND_EPS,
        )
        assert bounds[k, 0, 0] == lo32
        assert bounds[k, 0, 1] == hi32
