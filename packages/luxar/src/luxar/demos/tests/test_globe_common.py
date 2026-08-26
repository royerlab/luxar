"""Shared globe helpers: the UV sphere's frame, winding and seam.

These are the invariants that make a textured globe *interchangeable* with the
point cloud it replaces. Every data overlay in the four Earth demos is placed by
:func:`lonlat_to_xyz`, so a sphere built in a different frame — mirrored, rolled,
or with v inverted — renders a plausible planet with the earthquakes in the sea.
That failure passes every static gate and every "does it render" check, which is
why the mapping is asserted numerically here rather than reviewed by eye.
"""

from __future__ import annotations

import numpy as np
import pytest

from luxar.demos._globe_common import encode_globe_texture, uv_sphere


def _to_lonlat(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Invert ``lonlat_to_xyz``: x = r cosLat cosLon, y = r sinLat, z = -r cosLat sinLon."""
    r = np.linalg.norm(v, axis=1)
    lat = np.degrees(np.arcsin(np.clip(v[:, 1] / r, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(-v[:, 2], v[:, 0]))
    return lon, lat


def test_uvs_encode_the_same_lon_lat_as_the_positions() -> None:
    """A vertex's UV must address the texel its own position corresponds to.

    The single most important property here, and the one an eyeball cannot
    confirm: a globe whose v is inverted or whose u is rolled still renders a
    recognisable Earth. Only the residual against the position's own lon/lat
    catches it.

    Tolerance is expressed in TEXELS of the 2048x1024 basemap these demos use,
    since that is the scale at which an offset becomes visible.
    """
    vertices, _faces, uvs, _normals = uv_sphere(128, 64, 1.0)
    lon, lat = _to_lonlat(vertices)

    # Away from the poles only. At lat = +/-90 longitude is undefined — every
    # meridian meets there — so `arctan2` returns an arbitrary value while the
    # authored u legitimately spans the whole range. That is the degenerate pole
    # column `uv_sphere` documents, not a mapping error, and including it would
    # make this test assert something false.
    interior = np.abs(lat) < 89.0

    expected_u = (lon[interior] + 180.0) / 360.0
    # v runs TOP-DOWN: v = 0 is the texture's first row = +90 latitude.
    expected_v = (90.0 - lat[interior]) / 180.0
    # u wraps: the duplicated seam column carries u = 1 for the same point
    # `arctan2` reports as lon = -180 (u = 0).
    du = np.min(
        np.abs(
            np.stack(
                [
                    uvs[interior, 0] - expected_u,
                    uvs[interior, 0] - expected_u - 1.0,
                    uvs[interior, 0] - expected_u + 1.0,
                ]
            )
        ),
        axis=0,
    )
    dv = np.abs(uvs[interior, 1] - expected_v)

    assert du.max() < 0.5 / 2048, f"u is off by {du.max() * 2048:.3f} texels"
    assert dv.max() < 0.5 / 1024, f"v is off by {dv.max() * 1024:.3f} texels"


def test_v_zero_is_the_north_pole() -> None:
    """v runs TOP-DOWN: v = 0 is the texture's first row, which is +90 latitude.

    Its own test because this is the failure that shipped. The first cut used
    ``v = (lat + 90) / 180`` — bottom-up, the OpenGL convention — and rendered a
    latitude-mirrored Earth while every numeric check passed, because the checks
    derived the sampler's row from ``texture.flipY`` and WebGL silently ignores
    that flag for an ``ImageBitmap``. The visible symptom was earthquakes landing
    in the wrong ocean; the geometry was never wrong.

    So the convention is pinned as a DIRECTION here, independently of any model of
    the sampler: north must have the SMALLER v, because the viewer forces
    ``flipY = false`` and an equirectangular image's row 0 is +90.
    """
    vertices, _f, uvs, _n = uv_sphere(16, 8, 1.0)
    _lon, lat = _to_lonlat(vertices)
    north = lat > 80.0
    south = lat < -80.0
    assert uvs[north, 1].max() < uvs[south, 1].min(), (
        "v must increase SOUTHWARD — a bottom-up v renders the globe mirrored "
        "in latitude"
    )
    # And the poles reach the full range rather than stopping short.
    assert uvs[:, 1].min() == pytest.approx(0.0)
    assert uvs[:, 1].max() == pytest.approx(1.0)


def test_faces_wind_counter_clockwise_seen_from_outside() -> None:
    """Every triangle's geometric normal points away from the centre.

    The winding contract (MESH_NODE_SPEC section 3.2) in the form that matters
    for a closed surface: get it backwards and a single-sided globe renders
    inside-out — you see the far hemisphere's interior, which reads as a hollow
    shell rather than as an error. It is also what lets these demos pass
    `double_sided=False` and halve their fragment work.
    """
    vertices, faces, _uv, _n = uv_sphere(24, 12, 1.0)
    tri = vertices[faces]
    geometric = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    outward = tri.mean(axis=1)
    dots = np.einsum("ij,ij->i", geometric, outward)
    # The pole rows are degenerate quads, so their cross product is ~zero and
    # carries no orientation to check.
    nondegenerate = np.abs(dots) > 1e-9
    assert nondegenerate.sum() > 0
    assert np.all(dots[nondegenerate] > 0), (
        f"{(dots[nondegenerate] <= 0).sum()} triangles wind inward"
    )


def test_stored_normals_agree_with_the_geometry() -> None:
    """The returned normals are outward unit vectors, aligned with the surface."""
    vertices, _f, _uv, normals = uv_sphere(16, 8, 2.5)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)
    # On an undisplaced sphere the radial normal IS the surface normal.
    radial = vertices / np.linalg.norm(vertices, axis=1)[:, None]
    assert np.allclose(normals, radial, atol=1e-5)


def test_the_seam_column_is_duplicated_in_uv_but_not_in_space() -> None:
    """lon = -180 and lon = +180 are one place with two texture coordinates.

    This duplication is the entire reason for preferring a UV sphere over the
    icosphere the test fixtures use. Without it the quads spanning the dateline
    would interpolate u from ~1 back to ~0 and smear the whole texture backwards
    across one band.
    """
    n_lon, n_lat = 12, 6
    vertices, _f, uvs, _n = uv_sphere(n_lon, n_lat, 1.0)
    cols = n_lon + 1
    first = vertices[0::cols]
    last = vertices[cols - 1 :: cols]
    assert np.allclose(first, last, atol=1e-6), "the seam columns are not coincident"
    assert np.allclose(uvs[0::cols, 0], 0.0)
    assert np.allclose(uvs[cols - 1 :: cols, 0], 1.0)


def test_relief_displaces_radially() -> None:
    """A per-vertex relief scales each vertex along its own radius.

    What makes the rivers demo's 45x-exaggerated terrain expressible as a mesh at
    all: the displacement has to be the same function `lonlat_to_xyz` applies to
    the points, or the surface and the rivers drawn on it would separate.
    """
    n_lon, n_lat = 8, 4
    flat, _f, _uv, _n = uv_sphere(n_lon, n_lat, 1.0)
    relief = np.full((n_lat + 1, n_lon + 1), 0.25)
    bumped, _f2, _uv2, _n2 = uv_sphere(n_lon, n_lat, 1.0, relief=relief)
    assert np.allclose(bumped, flat * 1.25, atol=1e-6)


@pytest.mark.parametrize("bad", [(2, 4), (8, 1)])
def test_degenerate_grids_are_refused(bad: tuple[int, int]) -> None:
    """Fewer than 3 longitudes or 2 latitudes cannot close a surface."""
    with pytest.raises(ValueError, match="n_lon >= 3 and n_lat >= 2"):
        uv_sphere(bad[0], bad[1], 1.0)


def test_encoded_texture_round_trips_through_the_codec() -> None:
    """`encode_globe_texture` produces bytes a decoder recognises as JPEG."""
    rng = np.random.default_rng(0)
    tex = rng.integers(0, 256, size=(32, 64, 3), dtype=np.uint8)
    blob = encode_globe_texture(tex)
    assert blob.dtype == np.uint8 and blob.ndim == 1
    # JPEG SOI marker. Checked because the encoding attr the writer stamps is a
    # DECLARATION: the viewer picks its decode path from it without inspecting
    # the bytes, so a mislabelled payload fails in the browser, not here.
    assert blob[0] == 0xFF and blob[1] == 0xD8
    assert blob.size < tex.nbytes, "the encoded form should be smaller than raw"
