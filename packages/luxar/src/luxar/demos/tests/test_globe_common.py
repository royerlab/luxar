"""Shared globe helpers: the UV sphere's frame, winding and seam.

These are the invariants that make a textured globe *interchangeable* with the
point cloud it replaces. Every data overlay in the four Earth demos is placed by
:func:`lonlat_to_xyz`, so a sphere built in a different frame — mirrored, rolled,
or with v inverted — renders a plausible planet with the earthquakes in the sea.
That failure passes every static gate and every "does it render" check, which is
why the mapping is asserted numerically here rather than reviewed by eye.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from luxar.demos import _globe_common
from luxar.demos._globe_common import (
    add_textured_globe,
    build_earth,
    encode_globe_texture,
    resample_equirect_grid,
    uv_sphere,
)


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


@pytest.mark.parametrize(
    ("tiles", "expected_wrap"),
    [(1, None), (2, "clamp")],
)
def test_globe_wrap_only_overrides_the_tiled_case(
    monkeypatch: pytest.MonkeyPatch, tiles: int, expected_wrap: str | None
) -> None:
    """A whole globe keeps repeat-u/clamp-v; longitude bands clamp both axes."""

    class Target:
        def __init__(self) -> None:
            self.meshes: list[dict[str, object]] = []

        def add_group(self, _name: str, **_kwargs: object) -> "Target":
            return self

        def add_mesh(self, _name: str, **kwargs: object) -> None:
            self.meshes.append(kwargs)

    monkeypatch.setattr(
        _globe_common,
        "encode_texture",
        lambda image, **_kwargs: (
            np.array([1], dtype=np.uint8),
            "jpeg",
            image.shape[1],
            image.shape[0],
            image.shape[2],
        ),
    )
    target = Target()

    add_textured_globe(
        target,
        "earth",
        basemap=np.zeros((4, 8, 3), dtype=np.uint8),
        radius=1.0,
        n_lon=8,
        n_lat=4,
        tiles=tiles,
    )

    assert len(target.meshes) == tiles
    for mesh_kwargs in target.meshes:
        if expected_wrap is None:
            assert "texture_wrap" not in mesh_kwargs
        else:
            assert mesh_kwargs["texture_wrap"] == expected_wrap


def test_globe_ktx2_passes_rgb_tiles_to_the_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """KTX2 authoring receives pixels, never a pre-compressed image blob."""

    class Target:
        def __init__(self) -> None:
            self.meshes: list[dict[str, object]] = []

        def add_group(self, _name: str, **_kwargs: object) -> "Target":
            return self

        def add_mesh(self, _name: str, **kwargs: object) -> None:
            self.meshes.append(kwargs)

    monkeypatch.setattr(
        _globe_common,
        "encode_texture",
        lambda *_args, **_kwargs: pytest.fail("KTX2 must bypass bitmap encoding"),
    )
    target = Target()
    basemap = np.arange(4 * 8 * 3, dtype=np.uint8).reshape(4, 8, 3)

    add_textured_globe(
        target,
        "earth",
        basemap=basemap,
        radius=1.0,
        n_lon=8,
        n_lat=4,
        tiles=2,
        fmt="ktx2",
        quality=3,
    )

    assert len(target.meshes) == 2
    for mesh_kwargs in target.meshes:
        texture = mesh_kwargs["texture"]
        assert isinstance(texture, np.ndarray)
        assert texture.dtype == np.uint8
        assert texture.ndim == 3 and texture.shape[2] == 3
        assert mesh_kwargs["texture_width"] == texture.shape[1]
        assert mesh_kwargs["texture_height"] == texture.shape[0]
        assert mesh_kwargs["texture_encoding"] == "ktx2"
        assert mesh_kwargs["texture_ktx2_quality"] == 3


def test_globe_ktx2_rescales_float_pixels() -> None:
    """The KTX2 path preserves the helper's documented float [0, 1] input."""

    class Target:
        def __init__(self) -> None:
            self.meshes: list[dict[str, object]] = []

        def add_mesh(self, _name: str, **kwargs: object) -> None:
            self.meshes.append(kwargs)

    target = Target()
    add_textured_globe(
        target,
        "earth",
        basemap=np.full((2, 4, 3), 0.5, dtype=np.float32),
        radius=1.0,
        n_lon=4,
        n_lat=2,
        fmt="ktx2",
        quality=2,
    )

    texture = target.meshes[0]["texture"]
    assert isinstance(texture, np.ndarray)
    assert texture.dtype == np.uint8
    assert np.all(texture == 127)


def test_shared_earth_builder_defaults_to_ktx2() -> None:
    """Every shared Earth demo must emit GPU-compressed basemap textures."""
    import inspect

    signature = inspect.signature(_globe_common.build_earth)
    assert signature.parameters["fmt"].default == "ktx2"
    assert signature.parameters["quality"].default is None

    class Target:
        def __init__(self) -> None:
            self.meshes: list[dict[str, object]] = []

        def add_mesh(self, _name: str, **kwargs: object) -> None:
            self.meshes.append(kwargs)

    target = Target()
    build_earth(
        target,
        basemap=np.zeros((2, 4, 3), dtype=np.uint8),
        n_lon=4,
        n_lat=2,
    )
    assert target.meshes[0]["texture_encoding"] == "ktx2"
    assert target.meshes[0]["texture_ktx2_quality"] == 2

    earthquake_source = (
        Path(_globe_common.__file__).parent / "demo_earthquakes_3d.py"
    ).read_text()
    assert 'GLOBE_TEXTURE_FORMAT = "ktx2"' in earthquake_source
    assert "GLOBE_TEXTURE_QUALITY = 2" in earthquake_source


@pytest.mark.parametrize(("fmt", "expected_quality"), [("webp", 90), ("ktx2", 2)])
def test_globe_quality_default_follows_the_selected_format(
    monkeypatch: pytest.MonkeyPatch, fmt: str, expected_quality: int
) -> None:
    class Target:
        def __init__(self) -> None:
            self.meshes: list[dict[str, object]] = []

        def add_mesh(self, _name: str, **kwargs: object) -> None:
            self.meshes.append(kwargs)

    seen: dict[str, int] = {}

    def fake_encode(image: np.ndarray, **kwargs: object):
        seen["quality"] = int(kwargs["quality"])
        return np.array([1], dtype=np.uint8), "webp", image.shape[1], image.shape[0], 3

    monkeypatch.setattr(_globe_common, "encode_texture", fake_encode)
    target = Target()
    add_textured_globe(
        target,
        "earth",
        basemap=np.zeros((2, 4, 3), dtype=np.uint8),
        radius=1.0,
        n_lon=4,
        n_lat=2,
        fmt=fmt,
    )

    if fmt == "ktx2":
        assert target.meshes[0]["texture_ktx2_quality"] == expected_quality
    else:
        assert seen["quality"] == expected_quality


def test_non_integral_relief_resampling_area_averages_at_demo_resolution() -> None:
    """A source-cell spike is diluted, not selected whole or dropped."""
    target_width, target_height = 2049, 1025
    src = np.zeros((2 * target_height + 1, 2 * target_width + 1), dtype=np.float32)
    src[1, 1] = 8848.0

    sampled = resample_equirect_grid(src, target_width, target_height)

    assert sampled.dtype == np.float32
    assert sampled.shape == (target_height, target_width)
    assert sampled[0, 0] == pytest.approx(2212.0)
    assert sampled.max() < src.max()


@pytest.mark.parametrize(
    "place,lon,lat,should_be_dry",
    [
        # Low-lying coast: genuinely 1-6 m above the sea over tens of kilometres,
        # which is exactly what a ~20 km relief cell averages toward zero. These
        # are the places a sea surface at datum drowns.
        ("S Florida / Everglades", -80.9, 25.9, True),
        ("Nile delta", 31.0, 31.2, True),
        ("Bangladesh delta", 90.4, 22.8, True),
        ("Amazon mouth", -50.0, -0.5, True),
        # Real ocean, kilometres deep. If these came out dry the surface would be
        # below the sea floor and the layer would be pointless.
        ("Gulf of Mexico", -90.0, 25.0, False),
        ("mid-Atlantic", -40.0, 30.0, False),
    ],
)
def test_sea_level_leaves_low_coast_dry_and_ocean_wet(
    place: str, lon: float, lat: float, should_be_dry: bool
) -> None:
    """The water surface must sit below low-lying land and above the sea floor.

    This is the assertion a screenshot cannot make reliably and that two rounds of
    tuning got wrong. A positive offset of 2e-4 was 85 real metres under a 15x
    exaggeration and drowned every delta; taking it to zero still drowned them,
    because the relief is area-averaged onto ~20 km cells and anywhere 1-3 m above
    the sea averages to at or below datum there.

    Checked against ETOPO's own values at named places rather than against the
    render, so the answer is a number and not an impression. Skips when the ETOPO
    cache is absent, since this is a data-dependent property of the real grid.
    """
    from pathlib import Path

    cache = Path.home() / ".cache/luxar/global_rivers_earth/etopo_2022_60s.tif"
    if not cache.exists():
        pytest.skip("ETOPO cache not present")
    tifffile = pytest.importorskip("tifffile")

    from luxar.demos import demo_global_rivers_earth as demo
    from luxar.demos._globe_common import SeaLevel, resample_equirect_grid

    etopo = tifffile.imread(cache)
    grid = resample_equirect_grid(etopo, demo.GLOBE_LON + 1, demo.GLOBE_LAT + 1)
    relief = grid / demo.R_EARTH * demo.EXAGG
    water = SeaLevel().lift

    row = int(round((90.0 - lat) / 180.0 * demo.GLOBE_LAT))
    col = int(round((lon + 180.0) / 360.0 * demo.GLOBE_LON))
    is_dry = bool(relief[row, col] > water)
    assert is_dry is should_be_dry, (
        f"{place}: terrain {grid[row, col]:.1f} m ({relief[row, col]:+.2e}) vs "
        f"water {water:+.2e} — expected {'dry' if should_be_dry else 'submerged'}"
    )


def test_the_shared_globe_imagery_is_cached_once_per_machine() -> None:
    """The NASA masters must be keyed on the SOURCE, not on a demo or a width.

    Both halves of this were wrong in the first cut, and the comment above the URL
    already *claimed* the fixed behaviour ("the download is paid once per machine
    and every demo shares it") while the code did neither — so nothing but a
    measurement could catch it. On this machine the identical 28.5 MB master had
    accumulated five times: once per demo namespace, plus a leftover from raising
    one demo's texture width, since the width was part of the filename.

    Two invariants, because they fail independently:

    * the cache namespace is shared, so four demos are one download; and
    * no filename is passed, so ``cached_download`` derives it from the URL and a
      target-size change cannot invalidate the cache. The bytes on disk are the
      MASTER — the downscale happens in memory per build — so a size in the name
      describes something the file is not.

    Every globe demo must also declare the shared namespace in its ``caches``, or
    ``luxar demo cache list`` reports it as an orphan and ``clear --orphans``
    deletes a 28.5 MB download that takes ~5 minutes to replace.
    """
    source = Path(_globe_common.__file__).read_text()

    assert 'GLOBE_ASSET_CACHE = "blue_marble"' in source
    # Neither fetch may name a file: the URL basename is the width-independent key.
    for url_const in ("BLUE_MARBLE_HIRES_URL", "BLUE_MARBLE_CLOUDS_URL"):
        call = source.split(f"cached_download({url_const}")[1].split(")")[0]
        assert "GLOBE_ASSET_CACHE" in call, f"{url_const} must use the shared cache"
        assert '"' not in call and "f'" not in call, (
            f"{url_const} must not pass a filename — a width in the name re-downloads"
        )
    # Deliberately NOT a whole-file search for the old `blue_marble_{width}x{height}`
    # shape: the comment above GLOBE_ASSET_CACHE documents that shape on purpose, so
    # a text search matches the explanation and not the defect. The per-call check
    # above is stronger anyway — it forbids passing ANY filename, not one spelling.

    for module in (
        "demo_earthquakes_3d",
        "demo_ocean_currents_earth",
        "demo_global_rivers_earth",
        "demo_biodiversity_planetary_scale",
    ):
        demo_src = (Path(_globe_common.__file__).parent / f"{module}.py").read_text()
        caches = demo_src.split('"caches":')[1].split("]")[0]
        assert "blue_marble" in caches, f"{module} must claim the shared cache dir"
