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
        scene.add_points("pts", positions, additive_lod={"n_lods": 3})

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
    # most 4/131070 = 3.1e-5 per axis, and it can only widen a bound. The
    # barrier axis is a gridded integer time index, so its slack is exactly 0.
    _quant = 4.0 / 131070.0

    # Spatial axes (1, 2, 3): padded by exactly DEFAULT_POINT_RADIUS = 0.5.
    for dim in (1, 2, 3):
        lo = bounds[:, dim, 0].min()
        hi = bounds[:, dim, 1].max()
        assert lo <= -DEFAULT_POINT_RADIUS
        assert hi >= 4.0 + DEFAULT_POINT_RADIUS
        assert lo == pytest.approx(0.0 - DEFAULT_POINT_RADIUS, abs=_quant)
        assert hi == pytest.approx(4.0 + DEFAULT_POINT_RADIUS, abs=_quant)

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
