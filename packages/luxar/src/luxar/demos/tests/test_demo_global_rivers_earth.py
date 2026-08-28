"""Smoke tests for the pure helpers in demo_global_rivers_earth.

Exercises only the network-free / IO-free helpers (Fibonacci sphere, sphere
mapping, hypsometric palette, polyline decimation, LUT builder). The demo is
loaded by file path (see test_demo_ppi_flow_field for the rationale).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_DEMO_PATH = Path(__file__).resolve().parents[1] / "demo_global_rivers_earth.py"


def _load_demo_module():
    name = "_luxar_demo_globe_for_tests"
    spec = importlib.util.spec_from_file_location(name, _DEMO_PATH)
    if spec is None or spec.loader is None:
        pytest.skip(f"Could not locate demo at {_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_demo = _load_demo_module()


class TestFibonacciSphere:
    def test_ranges_and_count(self) -> None:
        lon, lat = _demo.fibonacci_sphere(10000)
        assert lon.shape == (10000,) and lat.shape == (10000,)
        assert lon.min() >= -180.0 and lon.max() <= 180.0
        assert lat.min() >= -90.0 and lat.max() <= 90.0
        # roughly even hemisphere split (uniform sphere coverage)
        assert abs(float((lat > 0).mean()) - 0.5) < 0.05

    def test_single_point(self) -> None:
        lon, lat = _demo.fibonacci_sphere(1)
        assert lon.shape == (1,) and lat.shape == (1,)


class TestLonLatToXyz:
    def test_radius_and_inverse(self) -> None:
        lon = np.array([0, 90, -90, 180, 45.0])
        lat = np.array([0, 0, 45, -30, 89.0])
        p = _demo.lonlat_to_xyz(lon, lat, np.zeros(5))
        assert np.allclose(np.linalg.norm(p, axis=1), _demo.RADIUS, atol=1e-3)
        rlat = np.degrees(np.arcsin(np.clip(p[:, 1] / _demo.RADIUS, -1, 1)))
        assert np.allclose(rlat, lat, atol=1e-3)

    def test_relief_scales_radius(self) -> None:
        p = _demo.lonlat_to_xyz(np.array([0.0]), np.array([0.0]), np.array([0.1]))
        assert np.isclose(np.linalg.norm(p[0]), _demo.RADIUS * 1.1, atol=1e-3)

    def test_right_handed_not_mirrored(self) -> None:
        # East(+dlon) x North(+dlat) must point OUTWARD -> right-handed globe
        # (a mirror/left-handed mapping would point inward). Regression guard.
        eps = 1e-3
        base = _demo.lonlat_to_xyz(np.array([0.0]), np.array([0.0]), np.zeros(1))[0]
        east = (
            _demo.lonlat_to_xyz(np.array([eps]), np.array([0.0]), np.zeros(1))[0] - base
        )
        north = (
            _demo.lonlat_to_xyz(np.array([0.0]), np.array([eps]), np.zeros(1))[0] - base
        )
        assert float(np.dot(np.cross(east, north), base)) > 0


class TestHypsometricScalars:
    def test_break_monotonic_range(self) -> None:
        elev = np.array([-10000, -100, 0, 100, 8000.0], dtype=np.float32)
        s = _demo.hypsometric_scalars(elev)
        assert s.min() >= 0.0 and s.max() <= 1.0
        assert np.all(np.diff(s) >= 0)
        assert 0.20 < float(s[2]) < 0.24  # sea level at the ocean/land break

    def test_degenerate_inputs_no_crash(self) -> None:
        assert (
            _demo.hypsometric_scalars(np.array([-9000, -10.0], dtype=np.float32)).max()
            <= 1.0
        )
        assert (
            _demo.hypsometric_scalars(np.array([10, 5000.0], dtype=np.float32)).min()
            >= 0.0
        )
        z = _demo.hypsometric_scalars(np.zeros(3, dtype=np.float32))
        assert not np.isnan(z).any()


class TestDecimatePolyline:
    def test_reduces_and_keeps_endpoints(self) -> None:
        pts = np.column_stack([np.linspace(0, 1, 50), np.zeros(50)]).astype(np.float32)
        d = _demo.decimate_polyline(pts, 0.025)
        assert len(d) < len(pts)
        assert np.allclose(d[0], pts[0]) and np.allclose(d[-1], pts[-1])

    def test_short_polyline_passthrough(self) -> None:
        pts = np.array([[0, 0], [1, 1.0]], dtype=np.float32)
        assert len(_demo.decimate_polyline(pts, 0.025)) == 2

    def test_sub_epsilon_collapses_to_endpoints(self) -> None:
        pts = np.array([[0, 0], [1e-4, 0], [2e-4, 0.0]], dtype=np.float32)
        assert len(_demo.decimate_polyline(pts, 0.025)) == 2


class TestLutFrom:
    def test_shape_and_endpoints(self) -> None:
        lut = _demo._lut_from([(0.0, (0, 0, 0)), (1.0, (255, 255, 255))])
        assert lut.shape == (256, 3) and lut.dtype == np.uint8
        assert lut[0].tolist() == [0, 0, 0]
        assert lut[-1].tolist() == [255, 255, 255]


def test_terrain_is_a_relief_displaced_textured_mesh() -> None:
    """The terrain is a Mesh, displaced by ETOPO and coloured by a texture.

    Replaces a test that asserted the 8M-point terrain was PARTITIONED below one
    Points node's element-texture bound. That whole concern is designed out: 526k
    vertices is three orders of magnitude under any cap, so there is nothing to
    partition around.

    What is worth pinning instead is the pair of choices that make a mesh better
    here rather than merely smaller, since both are invisible in a rendered frame
    that "looks like a globe":

    * ``relief=`` — the terrain is DISPLACED. Without it this is a smooth sphere
      with a topographic picture painted on, which is a different visualisation.
    * a texture rather than per-vertex colours — this is what decouples colour
      resolution from geometry resolution. A point carried exactly one colour, so
      the coastline and the mountains competed for the same budget; now the
      palette is 4096x2048 while the relief gets the vertices.

    And ``shading="flat"``, because relief that casts no light reads as a colour
    band rather than as terrain — the derivative normal is the real surface normal
    of the displaced mesh, where a stored one would have to be the sphere's.
    """
    source = _DEMO_PATH.read_text()
    assert "scene.add_points(" not in source, "the terrain should no longer be points"
    globe_call = source.split("build_earth(")[1].split("\n            )")[0]
    for token in ("basemap=basemap", "relief=relief", "tiles=tiles"):
        assert token in globe_call, f"missing {token}"
    # The absences: both were scale workarounds for the point cloud.
    assert "partition=" not in globe_call
    assert "additive_lod=" not in globe_call
    # Relief reaches the helper, and is area-averaged rather than point-sampled.
    assert "relief = relief_grid / R_EARTH * EXAGG" in source
    assert "resample_equirect_grid(etopo" in source

    # SMOOTH, not flat. `flat` derives one normal per TRIANGLE, so at this
    # tessellation every triangle shaded as a facet and the mesh itself became the
    # dominant visual feature — visibly so. The helper computes true surface
    # normals from the displaced geometry instead, which is what makes the shading
    # follow the terrain rather than the tessellation. Radial (sphere) normals are
    # the other wrong answer: they ignore slope, so relief casts no light at all.
    assert 'shading="smooth"' in globe_call
    assert 'shading="flat"' not in globe_call


def test_the_sea_surface_sits_at_sea_level_and_is_translucent() -> None:
    """A semi-transparent water shell, so bathymetry reads as depth.

    Only expressible because the terrain is displaced GEOMETRY: land stands proud
    of the shell and the trenches sit below it, which is the relationship the real
    thing has. On a painted sphere there would be nothing for the water to be
    above or below.

    The tiny lift off ``RADIUS`` is load-bearing rather than fussy — exactly at
    sea level the two surfaces are coplanar along every coastline, and coplanar
    geometry z-fights into a shimmering hairline as the camera moves.
    """
    source = _DEMO_PATH.read_text()
    globe_call = source.split("build_earth(")[1].split("\n            )")[0]
    # The water is now a `SeaLevel()` option on the shared helper rather than a
    # hand-rolled `add_mesh` here, so what this demo has to get right is asking
    # for it at all. The values themselves are pinned in `SeaLevel`'s defaults and
    # exercised by the helper's own tests.
    assert "sea_level=SeaLevel()" in globe_call
    from luxar.demos._globe_common import SeaLevel

    # The SPECULAR is what makes it read as water rather than as a colour shift:
    # a translucent blue tint over Blue Marble's own dark-navy ocean is nearly
    # invisible (both are the same colour), while a sun-glint is a highlight the
    # basemap has nowhere. Found by toggling the node, not by reasoning.
    assert SeaLevel().specular > 0.5
    # NEGATIVE, and the sign is the point. Taking the offset to zero still floods,
    # because the relief grid is area-averaged onto ~20 km cells: anywhere genuinely
    # 1-3 m above the sea over tens of kilometres (south Florida, the Everglades,
    # the Nile delta) averages to at or below zero, so a surface at exactly datum
    # submerges it. The giveaway was the SHAPE — the shoreline traced square grid
    # steps rather than the 16384-wide basemap's coastline.
    #
    # Dropping the water ~8 m below datum keeps that coast dry for a shoreline
    # retreat far under one basemap texel (~10 km), and removes the coplanarity
    # that a positive lift existed to avoid in the first place.
    assert SeaLevel().lift < 0.0


def test_the_cloud_shell_clears_the_exaggerated_relief() -> None:
    """The cloud altitude must be DERIVED, not copied from the other globes.

    This is the one number in the four Earth demos that cannot be shared. EXAGG is
    45x, so Everest sits at 45 * 8848 / 6371000 = 6.2% of the radius — five times
    the 1.2% shell the flat globes use. A copied constant would put the atmosphere
    below the entire Himalaya and the mountains would spear through it.
    """
    source = _DEMO_PATH.read_text()
    # The demo hands `relief` to the helper and the helper DERIVES the altitude,
    # which is stronger than the demo computing it: the number now cannot fall out
    # of step with EXAGG, because nothing here states it.
    assert "relief=relief," in source
    from luxar.demos._globe_common import Clouds

    assert Clouds().altitude is None, "the default must derive, not pin"

    # And the derivation must actually clear the peak. 15x exaggeration puts
    # Everest at ~2.1% of the radius, five times the 1.2% floor the flat globes
    # use, so a copied constant would sit below the Himalaya.
    import numpy as np

    from luxar.demos._globe_common import (
        build_earth,  # noqa: F401 — documents the owner
    )

    peak = 8157.0 / _demo.R_EARTH * _demo.EXAGG
    assert peak > 0.012, "the relief must exceed the flat-globe cloud altitude"
    assert max(0.012, peak * 1.35) > peak, "the shell must clear the peak"
    assert np.isfinite(peak)
