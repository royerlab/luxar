"""Shared globe-surface construction for the Earth-based demos.

The earthquake and ocean-current demos build a textured planet out of Points
and kept their own copies of the same three primitives. The copies drifted: the
ocean demo learned to dither its lattice and to sample the texture vectorized,
and the earthquake demo did not, so the same planet came out visibly worse in
one of them. These are the shared versions.

The four primitives, and why each is shaped the way it is:

``fibonacci_sphere``
    A golden-angle lattice, because a regular lat/lon grid clusters points at
    the poles and wastes most of its budget there. **Dithered by default**: the
    bare lattice is a regular pattern, and once the rendered point radius
    approaches the point spacing it beats against any equirectangular texture
    into visible moire — long curved "worms" that look like a broken land mask.
    The dither trades that structure for unstructured noise, which the eye
    forgives.

``lonlat_to_xyz``
    Geographic degrees plus a fractional radial ``relief`` to Cartesian, with
    ``y`` as the north-pole axis and longitude increasing eastward. The ``-z``
    keeps the frame right-handed (East x North = outward) so the globe is not
    mirrored — get this wrong and every continent is its own mirror image,
    which is surprisingly easy to miss on a rotating sphere.

``sample_equirect``
    Bilinear equirectangular lookup, vectorized over all points. The
    per-point-in-a-Python-loop version this replaces cost about 90 us a point,
    which is tolerable at 120k points and impossible at 8M.

``surface_point_radius``
    The one piece of arithmetic that decides whether a Points globe reads as a
    surface or as a dot screen: the rendered radius has to be tied to the point
    SPACING, which shrinks as 1/sqrt(n). A radius pinned to a constant while the
    count changes is exactly how a globe ends up stippled.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
from arbol import aprint

__all__ = [
    "Clouds",
    "GLOBE_ASSET_CACHE",
    "SeaLevel",
    "add_cloud_shell",
    "add_textured_globe",
    "blue_marble_basemap",
    "blue_marble_clouds",
    "build_earth",
    "encode_globe_texture",
    "encode_texture",
    "fibonacci_sphere",
    "lonlat_to_xyz",
    "resample_equirect_grid",
    "sample_equirect",
    "surface_point_radius",
    "surface_vertex_normals",
    "uv_sphere",
]


def fibonacci_sphere(
    n: int, *, jitter: bool = True, seed: int = 1234
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(lon, lat)`` degrees for ``n`` points on a Fibonacci sphere.

    With ``jitter`` (the default) each point is dithered by up to half a mean
    angular spacing. The undithered lattice shows strong moire once the rendered
    point radius approaches the spacing; the dither trades that structure for
    unstructured noise, which is far less visible.

    Args:
        n: Number of points (must be >= 1).
        jitter: Dither the lattice by ~1 cell.
        seed: RNG seed for the dither (deterministic output).

    Returns:
        ``(lon, lat)`` float64 arrays of shape ``(n,)``, degrees.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    i = np.arange(n)
    golden = (1.0 + 5.0**0.5) / 2.0
    y = 1.0 - 2.0 * (i + 0.5) / n
    r_xy = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = 2.0 * np.pi * i / golden
    lat = np.degrees(np.arcsin(np.clip(y, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(r_xy * np.sin(theta), r_xy * np.cos(theta)))
    if jitter:
        rng = np.random.default_rng(seed)
        cell = np.degrees(np.sqrt(4.0 * np.pi / n))  # mean angular spacing
        lat = np.clip(lat + rng.uniform(-0.5, 0.5, n) * cell, -89.999, 89.999)
        # a degree of longitude shrinks with cos(lat), so scale the dither up
        lon = lon + rng.uniform(-0.5, 0.5, n) * cell / np.maximum(
            np.cos(np.radians(lat)), 1e-2
        )
    return lon, lat


def lonlat_to_xyz(
    lon: np.ndarray,
    lat: np.ndarray,
    relief: np.ndarray | float,
    radius: float,
) -> np.ndarray:
    """Map geographic degrees + fractional ``relief`` to sphere xyz.

    ``y`` is the north-pole axis and longitude increases eastward; the ``-z``
    keeps the frame right-handed (East x North = outward) so the globe is not
    mirrored.

    Args:
        lon: Longitudes in degrees.
        lat: Latitudes in degrees.
        relief: Fractional radial displacement (0 = on the sphere). Scalar or
            per-point.
        radius: Sphere radius in scene units.

    Returns:
        ``(n, 3)`` float32 positions.
    """
    la, lo = np.radians(lat), np.radians(lon)
    r = radius * (1.0 + np.asarray(relief))
    cl = np.cos(la)
    return np.column_stack(
        [r * cl * np.cos(lo), r * np.sin(la), -r * cl * np.sin(lo)]
    ).astype(np.float32)


def sample_equirect(tex: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Bilinearly sample an equirectangular RGB texture at ``lon``/``lat``.

    Vectorized over all points. Longitude wraps; latitude clamps.

    Args:
        tex: ``(h, w, 3)`` texture; row 0 is +90 deg latitude. Integer dtypes
            are treated as 0..255 and rescaled; float dtypes are assumed to be
            already normalized to [0, 1].
        lon: Longitudes in degrees (any range; wrapped).
        lat: Latitudes in degrees, -90..+90.

    Returns:
        ``(n, 3)`` float32 RGB in [0, 1].
    """
    h, w = tex.shape[:2]
    x = np.mod((lon + 180.0) / 360.0 * w, w)
    y = np.clip((90.0 - lat) / 180.0 * h, 0, h - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = (x0 + 1) % w
    y1 = np.minimum(y0 + 1, h - 1)
    wx = (x - x0)[:, None].astype(np.float32)
    wy = (y - y0)[:, None].astype(np.float32)
    t = tex.astype(np.float32)
    if np.issubdtype(tex.dtype, np.integer):
        t /= 255.0
    c0 = t[y0, x0] * (1.0 - wx) + t[y0, x1] * wx
    c1 = t[y1, x0] * (1.0 - wx) + t[y1, x1] * wx
    return np.clip(c0 * (1.0 - wy) + c1 * wy, 0.0, 1.0).astype(np.float32)


def surface_point_radius(n: int, radius: float, *, overlap: float = 0.75) -> float:
    """Point radius that makes ``n`` points on a sphere read as a solid surface.

    ``n`` points spread over a sphere of radius ``R`` each own a cap of area
    ``4*pi*R^2 / n``, so the mean centre-to-centre spacing is
    ``R * sqrt(4*pi/n)``. A point drawn with radius ``overlap`` times that
    spacing tiles with enough overlap to close the gaps without turning the
    surface into mush.

    The alternative — a constant radius — is what stipples a globe: it is tuned
    once at one point count and then silently becomes wrong when the count
    changes, in the direction of visible gaps if the count goes down and of a
    smeared surface if it goes up.

    Args:
        n: Number of surface points.
        radius: Sphere radius in scene units.
        overlap: Radius as a fraction of the mean spacing. 0.75 closes the gaps
            with a little margin; below ~0.55 the lattice starts to show
            through.

    Returns:
        Point radius in scene units.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    spacing = radius * float(np.sqrt(4.0 * np.pi / n))
    return spacing * overlap


def uv_sphere(
    n_lon: int,
    n_lat: int,
    radius: float,
    relief: np.ndarray | float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a textured UV sphere: vertices, faces, UVs and normals.

    A **UV** sphere, not the icosphere ``generate_test_data.py`` uses, and the
    difference is the whole point. An icosphere has excellent triangle uniformity
    and no natural equirectangular seam, so mapping a lon/lat image onto it needs
    a per-triangle decision about which side of the dateline each vertex belongs
    to. A lon/lat grid *is* the texture's own parameterization: UVs are the grid
    coordinates, and the seam is handled by duplicating the lambda = +/-180
    column so the two sides interpolate independently instead of wrapping the
    whole image backwards across one band of quads.

    Vertices follow :func:`lonlat_to_xyz` exactly — ``y`` polar, longitude
    eastward, ``-z`` keeping the frame right-handed. That is not a detail: every
    data overlay in these demos is registered through that same function, so a
    sphere built in a different frame would put earthquakes in the sea.

    ## Poles

    The pole rows are kept as degenerate quads rather than collapsed to a single
    vertex. A shared pole vertex needs one UV, but the triangles meeting it span
    the whole ``u`` range, so any single value smears one texel column across
    every one of them. Duplicated pole vertices each carry their own ``u`` and
    the distortion stays within a triangle, which is where an equirectangular
    projection puts it anyway.

    Args:
        n_lon: Longitude divisions. The returned grid has ``n_lon + 1`` columns —
            the extra one is the duplicated seam.
        n_lat: Latitude divisions, pole to pole.
        radius: Sphere radius in scene units.
        relief: Fractional radial displacement, scalar or ``(n_lat+1, n_lon+1)``.
            Applied exactly as :func:`lonlat_to_xyz` applies it.

    Returns:
        ``(vertices, faces, uvs, normals)`` — ``(V, 3)`` float32, ``(F, 3)``
        uint32, ``(V, 2)`` float32 in ``[0, 1]``, ``(V, 3)`` float32 unit
        outward normals.
    """
    if n_lon < 3 or n_lat < 2:
        raise ValueError(
            f"uv_sphere needs n_lon >= 3 and n_lat >= 2, got {n_lon}x{n_lat}"
        )

    # The seam column is a DUPLICATE of lon = -180 at lon = +180, so u runs the
    # full [0, 1] and the texture's last texel column meets its first without the
    # renderer having to wrap across a quad.
    lon = np.linspace(-180.0, 180.0, n_lon + 1)
    lat = np.linspace(90.0, -90.0, n_lat + 1)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    vertices = lonlat_to_xyz(
        lon_grid.ravel(), lat_grid.ravel(), np.asarray(relief).ravel(), radius
    )

    # UVs are the grid coordinates themselves, with **v running top-down**:
    # ``v = 0`` addresses the texture's FIRST row, which for an equirectangular
    # image is +90 latitude. That is the same convention
    # :func:`sample_equirect` applies, so the mesh and the point-sampled overlays
    # agree about which way up the world is, and it is the convention the viewer
    # pins by forcing ``flipY = false``.
    #
    # This was wrong in the first cut — ``v = (lat + 90) / 180``, i.e. bottom-up —
    # and the resulting globe rendered latitude-mirrored while every numeric check
    # passed, because those checks modelled the sampler from ``texture.flipY`` and
    # that flag is silently ignored for an ``ImageBitmap``. Caught by eye, from
    # earthquakes landing in the wrong ocean.
    u = (lon_grid.ravel() + 180.0) / 360.0
    v = (90.0 - lat_grid.ravel()) / 180.0
    uvs = np.column_stack([u, v]).astype(np.float32)

    # Outward unit normals, computed from the SPHERE rather than from the
    # displaced vertices. For a relief-displaced globe the true surface normal
    # tilts with the terrain, but the radial one is what makes the shading read
    # as a planet; a derived normal on a 45x-exaggerated relief grid produces
    # faceted noise, not shape. Callers wanting the geometric normal can pass
    # `shading="flat"` and let the shader derive it.
    normals = lonlat_to_xyz(lon_grid.ravel(), lat_grid.ravel(), 0.0, 1.0)

    # Faces, CCW as seen from OUTSIDE. `sorted(normal_dims)` for a plain 3D scene
    # is (0, 1, 2) = (x, y, z), and this winding is front-facing in that frame —
    # which is the contract MESH_NODE_SPEC section 3.2 states and what keeps a
    # single-sided globe from rendering inside-out.
    cols = n_lon + 1
    row = np.arange(n_lat)[:, None]
    col = np.arange(n_lon)[None, :]
    tl = (row * cols + col).ravel()
    tr = tl + 1
    bl = tl + cols
    br = bl + 1
    faces = np.concatenate(
        [
            np.column_stack([tl, bl, br]),
            np.column_stack([tl, br, tr]),
        ]
    ).astype(np.uint32)

    return vertices, faces, uvs, normals.astype(np.float32)


def encode_globe_texture(tex: np.ndarray, *, quality: int = 92) -> np.ndarray:
    """Re-encode an equirectangular RGB array as JPEG bytes for ``add_mesh``.

    Encoded rather than raw, and the ratio is why: the Blue Marble at 2048x1024
    is ~6.3 MB as a raw ``(h, w, 3)`` uint8 array and ~1.5 MB as JPEG. Blosc over
    the raw array does not close that gap — a photograph has little of the
    structure a general-purpose compressor exploits — and every demo store that
    embeds this is downloaded by users.

    JPEG specifically, not PNG: this is a photograph, so the lossy codec is both
    much smaller and visually indistinguishable at quality 92. A categorical or
    index-like texture would want PNG (and ``texture_filter="nearest"``), since
    JPEG's chroma subsampling invents intermediate values between classes.

    Args:
        tex: ``(h, w, 3)`` uint8 RGB, or float in [0, 1] (rescaled).
        quality: JPEG quality, 1-95.

    Returns:
        1-D uint8 array of JPEG bytes, ready for
        ``add_mesh(texture=..., texture_encoding="jpeg", ...)``.
    """
    from io import BytesIO

    from PIL import Image

    arr = np.asarray(tex)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected an (h, w, 3) RGB texture, got {arr.shape}")
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    buf = BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="JPEG", quality=quality)
    return np.frombuffer(buf.getvalue(), dtype=np.uint8)


#: NASA Blue Marble Next Generation, topography + bathymetry, December 2004.
#:
#: Cache namespace for the shared NASA globe imagery.
#:
#: SHARED across the four Earth demos rather than per-demo, and named after the
#: SOURCE rather than the target size, because these bytes depend on neither. The
#: first cut got both wrong: it cached under
#: ``<demo>/blue_marble_{width}x{height}.jpg``, so the identical 28.5 MB master was
#: fetched once per demo (4 copies on disk, ~20 minutes on a fresh machine) and
#: raising one demo's texture width re-downloaded the same file under a new name.
#: Measured, not reasoned: five copies of md5 ``0fb66aee...`` across four
#: namespaces.
#:
#: ``inventory_caches`` maps one directory to a LIST of claiming demos, so all
#: four declare this in their ``caches`` and it is not reported as an orphan.
GLOBE_ASSET_CACHE = "blue_marble"

#: The 21600x10800 master (30 MB). Downloaded once per machine into
#: :data:`GLOBE_ASSET_CACHE` and shared by every globe demo; the downscale to each
#: demo's texture width happens in memory on every build, since it is cheap next
#: to the transfer.
BLUE_MARBLE_HIRES_URL = (
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73909/"
    "world.topo.bathy.200412.3x21600x10800.jpg"
)

#: NASA Blue Marble cloud composite (2048x1024), a near-greyscale cloud
#: fraction map. Bright = cloud.
BLUE_MARBLE_CLOUDS_URL = (
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57747/"
    "cloud_combined_2048.jpg"
)

#: Per-axis ceiling for a mesh texture, mirroring ``MAX_MESH_TEXTURE_SIZE`` in
#: ``luxar/validation/base.py``. A globe basemap is the one place a demo can
#: realistically approach it, so the helpers below clamp rather than let the
#: writer refuse.
MAX_GLOBE_TEXTURE_WIDTH = 16384


def blue_marble_basemap(*, width: int = 8192) -> Tuple[np.ndarray, int, int]:
    """Fetch the Blue Marble basemap, downscaled to ``width`` pixels across.

    Defaults to 8192x4096 — **16x the pixels** of the 2048x1024 image these demos
    used while the globe was a point cloud. That ceiling was not a choice back
    then: a point cloud resolves a texture at roughly one sample per point, so
    detail beyond ~2k was invisible however large the image. A mesh samples per
    fragment, so the basemap is now the only thing limiting how sharp a coastline
    looks, and it is worth paying for.

    The source is the 21600x10800 master, so `draft` is used to let libjpeg do the
    first factor-of-two reduction in the DCT domain — decoding 233 megapixels at
    full size to then throw three quarters of it away costs about a gigabyte of
    RAM for nothing.

    Args:
        width: Target width; height follows the 2:1 equirectangular ratio.
            Clamped to :data:`MAX_GLOBE_TEXTURE_WIDTH`.

    Returns:
        ``(rgb, width, height)`` — ``(h, w, 3)`` uint8, row 0 at +90 latitude.
    """
    from PIL import Image

    from . import cached_download

    width = min(int(width), MAX_GLOBE_TEXTURE_WIDTH)
    height = width // 2
    # No filename: `cached_download` derives it from the URL, which is exactly the
    # width-independent name we want -- the cached bytes are the master, not the
    # resized result, so encoding a target size in the name only forces re-fetches.
    path = cached_download(BLUE_MARBLE_HIRES_URL, GLOBE_ASSET_CACHE)
    # PIL refuses anything over ~179 megapixels as a possible decompression bomb,
    # and the 21600x10800 master is 233. The guard is right in general and wrong
    # here: this is a pinned NASA URL with a known size, not user input. Raised
    # only around this decode and restored afterwards, so nothing else in the
    # process inherits a disabled safety check.
    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(path) as img:
            # Ask libjpeg for the smallest DCT-domain reduction that still covers
            # the target, then resample the rest. Without this the 21600-wide
            # master is decoded at full size only to discard three quarters of it.
            img.draft("RGB", (width, height))
            img = img.convert("RGB")
            if img.size != (width, height):
                img = img.resize((width, height), Image.Resampling.LANCZOS)
            return np.asarray(img), width, height
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit


def blue_marble_clouds(
    *,
    strength: float = 0.55,
    gamma: float = 1.8,
    width: int = 2048,
) -> Tuple[np.ndarray, int, int]:
    """Build an RGBA cloud texture from NASA's Blue Marble cloud composite.

    The cloud map is a near-greyscale cloud-fraction image, and it becomes a
    **texture whose alpha is the cloud cover**: RGB is flat white and the source
    luminance drives alpha. That is what makes the shell work — mesh multiplies
    texture alpha into coverage, so clear sky is genuinely transparent rather than
    black, and no separate mask channel or shader variant is needed.

    ``gamma`` above 1 is what makes the layer read as *weather* rather than as
    haze. The source has a broad low-luminance floor (thin cirrus, sensor
    background) which at linear alpha veils the whole planet and mutes the
    basemap's colours; raising it to a power pushes that floor toward zero while
    leaving the bright cores near their full value.

    PNG, necessarily: JPEG has no alpha channel at all, and the alpha IS the
    payload here. It compresses well regardless, since the three colour channels
    are constant.

    Args:
        strength: Peak alpha for a fully cloudy texel, in ``[0, 1]``.
        gamma: Exponent applied to the normalized luminance before scaling.
        width: Target width; height follows the 2:1 ratio.

    Returns:
        ``(rgba, width, height)`` — ``(h, w, 4)`` uint8, row 0 at +90 latitude.
    """
    from PIL import Image, ImageFilter

    from . import cached_download

    height = width // 2
    path = cached_download(BLUE_MARBLE_CLOUDS_URL, GLOBE_ASSET_CACHE)
    with Image.open(path) as img:
        img = img.convert("L")
        # De-block BEFORE anything else. The source is a JPEG, so it carries 8x8
        # DCT artefacts, and they are invisible at native size for a reason that
        # stops holding here: this layer is magnified (a 2048-wide cloud map over a
        # 16384-wide basemap) AND the gamma below is a contrast stretch on the low
        # end, which amplifies block noise precisely where cloud is faintest. The
        # result is visible square patches in the haze.
        #
        # A sub-pixel Gaussian removes the blocks while leaving real cloud edges —
        # weather has no structure at the 1-pixel scale of a 2048-wide global map,
        # so nothing of the data is lost. Applied before the resize and before the
        # gamma, since both would otherwise bake the artefact in.
        img = img.filter(ImageFilter.GaussianBlur(radius=0.9))
        if img.size != (width, height):
            # LANCZOS in BOTH directions, and the UPSAMPLING direction is the one
            # that matters here. The source is only 2048x1024 while the basemap is
            # now 16384x8192, so a cloud shell at native size is 8x coarser than
            # the terrain under it — and against a sharp coastline that coarseness
            # reads as square blocks of haze rather than as cloud.
            #
            # Bilinear would soften the blocks but leave the grid visible as
            # diamond-shaped facets; nearest would keep them hard. Lanczos gives a
            # genuinely smooth field, which is the right answer for a diffuse
            # quantity: no real information is invented either way, but the
            # ARTEFACT is what the eye picks up, and only Lanczos removes it.
            #
            # (Mipmaps and anisotropic filtering handle the MINIFYING direction at
            # render time; they cannot help when the texture is being magnified.)
            img = img.resize((width, height), Image.Resampling.LANCZOS)
        luminance = np.asarray(img, dtype=np.float32) / 255.0

    alpha = np.clip(luminance**gamma * float(strength), 0.0, 1.0)
    # DITHER before quantizing, which is the actual fix for the "blocky" cloud
    # layer. The artefact was not JPEG blocks and not bilinear facets: it was
    # 8-BIT BANDING in the alpha, and the gamma is what creates it. `x ** 1.7` has
    # a small derivative near zero, so a whole range of faint input luminances
    # collapses onto the same output byte — and faint is most of the sky, so the
    # result is broad terraces with hard edges, exactly the grey contour blobs
    # that were visible.
    #
    # A triangular +/- 1 LSB dither (the sum of two uniforms) converts those steps
    # into noise the eye integrates back to a smooth gradient. Standard practice
    # for quantizing a smooth gradient to 8 bits, and it costs nothing. Seeded so
    # a rebuild is reproducible.
    rng = np.random.default_rng(0xC10D)
    lsb = 1.0 / 255.0
    dither = (rng.random(alpha.shape) - rng.random(alpha.shape)) * lsb
    alpha = np.clip(alpha + dither, 0.0, 1.0)
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    # Flat white: cloud is white, and the shell is authored unlit so nothing
    # tints it. Colour variation in the source is sensor artefact, not weather.
    rgba[..., :3] = 255
    rgba[..., 3] = np.round(alpha * 255.0).astype(np.uint8)
    return rgba, width, height


def encode_globe_texture_png(rgba: np.ndarray) -> np.ndarray:
    """Encode an RGBA texture as PNG bytes for ``add_mesh``.

    The alpha-carrying sibling of :func:`encode_globe_texture`. JPEG cannot be
    used for anything with an alpha channel — it has none — and for a cloud mask
    the alpha is the entire payload.
    """
    from io import BytesIO

    from PIL import Image

    arr = np.asarray(rgba)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError(f"Expected an (h, w, 4) RGBA texture, got {arr.shape}")
    buf = BytesIO()
    Image.fromarray(arr.astype(np.uint8), mode="RGBA").save(
        buf, format="PNG", optimize=True
    )
    return np.frombuffer(buf.getvalue(), dtype=np.uint8)


def add_cloud_shell(
    scene: Any,
    name: str,
    *,
    radius: float,
    altitude: float = 0.012,
    strength: float = 0.55,
    gamma: float = 1.8,
    width: int = 2048,
    n_lon: int = 192,
    n_lat: int = 96,
    opacity: float = 1.0,
    intensity: float = 1.0,
    layer: bool = True,
    **extra: Any,
) -> None:
    """Add a translucent Blue Marble cloud shell just above a globe.

    The shell sits at ``radius * (1 + altitude)`` — a real altitude, not a
    coincident surface. Coplanar geometry z-fights: the clouds would strobe
    against the basemap as the camera moved, in a pattern that reads as a
    rendering glitch rather than as a missing offset. ``0.012`` of Earth's radius
    is ~76 km, which is above the troposphere and still small enough that the
    parallax at the limb looks like atmosphere rather than a detached bubble.

    ``luminous`` blending, and the reasoning here was wrong the first time round.
    The shell was ``normal`` on the argument that "clouds OCCLUDE the surface, and
    an additive layer would brighten it instead" — which conflates ``luminous``
    with ``additive``. They are different modes for exactly this reason:
    ``luminous`` is additive *and* ``depthTest: true``, so the far-side deck is
    still hidden by the opaque globe. Nothing is given up.

    And additive is the better physical model. A thin cloud at planetary scale does
    not replace the surface beneath it — it SCATTERS sunlight toward the viewer, so
    it adds light on top of whatever is there. Subtracting the surface is what a
    thick, opaque overcast does, which is not this layer.

    The decisive practical difference is order. ``normal`` puts the shell in the
    viewer's sorted transparent set, so its appearance depends on getting depth
    order right against every other translucent thing in the scene (the current
    ribbons, another shell). ``luminous`` is commutative — the result is
    independent of draw order — so an entire class of sorting artefact simply does
    not arise.

    Args:
        scene: The scene (or group) to add the node to.
        name: Node name.
        radius: The globe's radius in scene units.
        altitude: Shell height as a fraction of ``radius``.
        strength: Peak cloud alpha.
        gamma: Cloud-alpha exponent (>1 thins the haze).
        width: Cloud texture width.
        n_lon: Longitude divisions of the shell mesh.
        n_lat: Latitude divisions of the shell mesh.
        opacity: Node opacity, multiplying the texture's alpha.
        intensity: Linear brightness multiplier. Under `luminous` this is the
            knob that decides whether the deck reads at all — see
            :class:`Clouds`.
        layer: Expose in the Layers panel.
        **extra: Forwarded verbatim to ``add_mesh``. The reason this exists is
            ``dim_order`` / ``fill``: in an nD scene the shell has to be present
            in every non-displayed slot, exactly as the globe under it is, or the
            atmosphere would appear on one slice and vanish on the rest.
    """
    rgba, tex_w, tex_h = blue_marble_clouds(strength=strength, gamma=gamma, width=width)
    # WebP, not PNG, and this is where the codec choice pays most. The payload is
    # a smooth alpha field over three constant colour channels: PNG stores it
    # losslessly at several MB, while lossy WebP reaches a visually identical
    # result for a fraction of that — and unlike JPEG it can carry the alpha at
    # all, which here IS the image. Falls back to PNG above WebP's 16383 limit.
    cloud_fmt = "webp" if max(tex_w, tex_h) <= MAX_WEBP_DIMENSION else "png"
    payload, encoding, tex_w, tex_h, tex_c = encode_texture(
        rgba,
        fmt=cloud_fmt,
        quality=85,
        # LOSSLESS alpha (100), and the measurements are worth recording because
        # two independent artefacts were mistaken for each other here, and both
        # produced "blocky clouds".
        #
        # Plateau width along a scanline, against an authored signal whose mean is
        # 1.44 px:
        #
        #   alpha_q  size      plateau mean   max     error
        #   70       0.83 MB   6.66           191 px  2.45
        #   85       2.56 MB   1.64            31 px  0.22
        #   92       2.41 MB   1.44            31 px  0.00
        #   100      2.41 MB   1.44            31 px  0.00
        #
        # So (a) lossy alpha at 70 was inventing 191-pixel terraces — that was the
        # visible staircase — and (b) above ~92 it is byte-identical to lossless
        # AND SMALLER than 85, because the dithered signal defeats quantization, so
        # the encoder spends bytes fighting noise it cannot remove. There is no
        # point anywhere below lossless.
        #
        # The OTHER artefact was different and earlier: at the source's native
        # 2048 over a 16384-wide basemap the shell is magnified 8x, and bilinear
        # magnification of an 8-bit field shows its texel lattice. That one is
        # fixed by resolution, not by compression — see the caller's `width`.
        alpha_quality=100,
        channels=4,
    )
    vertices, faces, uvs, normals = uv_sphere(n_lon, n_lat, radius * (1.0 + altitude))
    scene.add_mesh(
        name,
        vertices=vertices,
        faces=faces,
        uvs=uvs,
        # Normals are supplied even though the shell is UNLIT and never reads
        # them, because `normal_dims` is what declares the authored winding frame
        # — and without a frame the viewer cannot decide projected winding, so it
        # falls back to `DoubleSide` regardless of `double_sided=False`. Measured,
        # not assumed: the first version of this omitted them and the shell came
        # back `side: DoubleSide`, doubling cloud density at the limb, which is
        # exactly what the `double_sided=False` below is here to prevent.
        normals=normals,
        normal_dims=[0, 1, 2],
        texture=payload,
        texture_encoding=encoding,
        texture_width=tex_w,
        texture_height=tex_h,
        texture_channels=tex_c,
        # Unlit: a cloud deck lit by a view-anchored key would slide its terminator
        # independently of the globe's underneath, which reads as two planets.
        shading="none",
        blending_mode="luminous",
        opacity=opacity,
        intensity=intensity,
        layer=layer,
        # Single-sided. Without it the shell's far interior draws over the near
        # clouds, doubling their density at the limb exactly where it is already
        # highest.
        double_sided=False,
        **extra,
    )


def resample_equirect_grid(src: np.ndarray, width: int, height: int) -> np.ndarray:
    """Area-average an equirectangular grid down to ``(height, width)``.

    Block-mean rather than nearest or bilinear, and for a topographic grid that
    matters: subsampling a 21600x10800 relief model picks one cell in every
    ~400 and the result is *noise* — a summit or a trench survives only if the
    sampling grid happens to land on it, so the same mountain appears and
    disappears as the target resolution changes. An area average keeps the
    hypsometric distribution honest at every scale.

    Non-integral ratios use variable-width bins whose edges cover every source
    cell exactly once, so the ETOPO-to-mesh path remains an area average too.
    Upsampling falls back to nearest-neighbour sampling because area bins would
    contain zero source cells.

    Args:
        src: ``(h, w)`` or ``(h, w, c)`` source grid, row 0 at +90 latitude.
        width: Target width.
        height: Target height.

    Returns:
        The resampled grid, float32, same trailing shape as ``src``.
    """
    arr = np.asarray(src, dtype=np.float32)
    sh, sw = arr.shape[:2]
    if (sh, sw) == (height, width):
        return arr
    if sh % height == 0 and sw % width == 0:
        fy, fx = sh // height, sw // width
        tail = arr.shape[2:]
        blocks = arr.reshape((height, fy, width, fx) + tail)
        return blocks.mean(axis=(1, 3), dtype=np.float32)
    if height <= sh and width <= sw:
        row_edges = np.linspace(0, sh, height + 1, dtype=np.int64)
        col_edges = np.linspace(0, sw, width + 1, dtype=np.int64)
        row_sums = np.add.reduceat(arr, row_edges[:-1], axis=0)
        sums = np.add.reduceat(row_sums, col_edges[:-1], axis=1)
        counts = np.outer(np.diff(row_edges), np.diff(col_edges)).astype(np.float32)
        counts = counts.reshape(counts.shape + (1,) * (arr.ndim - 2))
        return sums / counts
    rows = np.clip((np.arange(height) + 0.5) * sh / height, 0, sh - 1).astype(np.int64)
    cols = np.clip((np.arange(width) + 0.5) * sw / width, 0, sw - 1).astype(np.int64)
    return arr[np.ix_(rows, cols)]


#: Hard per-axis ceiling of the WebP bitstream: dimensions are 14-bit, so 16383
#: is the largest either axis can be.
#:
#: One pixel below the 16384 GPU limit, which is a genuinely awkward coincidence:
#: at the maximum texture size a GPU accepts, WebP is unavailable. Measured, not
#: read off a spec — PIL raises "encoding error 5: Image size exceeds WebP limit
#: of 16383 pixels" at 16384 wide. Ask for 16383 instead and it encodes.
MAX_WEBP_DIMENSION = 16383

#: Bytes-on-disk for the Blue Marble basemap, measured at two sizes so a caller
#: can choose knowing the cost rather than guessing:
#:
#: ===============  ==========  ==========  ==========
#: size             JPEG q92    JPEG q85    WebP q90
#: ===============  ==========  ==========  ==========
#: 8192 x 4096       6.0 MB      4.2 MB      4.3 MB
#: 16384 x 8192     20.4 MB     14.4 MB     n/a (over the WebP limit)
#: ===============  ==========  ==========  ==========
#:
#: WebP q90 is 28% smaller than JPEG q92 at the same visual quality, and WebP q82
#: is 55% smaller (2.7 MB) — worth it for a photographic basemap that every user
#: downloads. Encoding is ~15x slower (1.8 s vs 0.1 s at 8192), which is paid once
#: at authoring time and never by a viewer.
TEXTURE_FORMAT_NOTES = "see the table above"


def encode_texture(
    image: Any,
    *,
    fmt: str = "webp",
    quality: int = 90,
    alpha_quality: int = 70,
    channels: Optional[int] = None,
) -> Tuple[np.ndarray, str, int, int, int]:
    """Encode (or TRANSCODE) an image to a codec of your choosing.

    The input may be a numpy array, a path, or already-encoded bytes — and the
    output codec is chosen INDEPENDENTLY of it. That decoupling is the point:
    "here is a PNG, give me WebP at q88" is one call, so an author never has to
    pre-convert a source file to get the on-disk format they want.

    Returns everything ``add_mesh`` needs, in the order it needs it, because the
    declared dimensions are load-bearing on the read side (the viewer budgets a
    node from them before fetching a byte) and deriving them separately from the
    encode is how they drift apart.

    Format guidance, measured rather than assumed (see :data:`TEXTURE_FORMAT_NOTES`):

    * ``webp`` — the default, and the right choice for a photographic basemap:
      28% smaller than JPEG at matched quality. **Hard 16383-pixel per-axis
      limit** (:data:`MAX_WEBP_DIMENSION`), one below the GPU's 16384, so it is
      unavailable at the largest size a GPU will take.
    * ``jpeg`` — the fallback above 16383, and for a source that is already JPEG
      where a transcode would just add a second generation of loss.
    * ``png`` — lossless, and REQUIRED for anything with alpha (JPEG has no alpha
      channel at all) or a palette that JUMPS, like a hypsometric ramp at sea
      level: a lossy codec rings across that discontinuity and paints a fake
      coastline.

    Args:
        image: ``(h, w, c)`` array, a path to an image, or encoded bytes.
        fmt: ``webp`` | ``jpeg`` | ``png``.
        quality: 1-100 for the colour channels. Ignored for ``png``.
        alpha_quality: 1-100 for the ALPHA channel (WebP only), defaulting to 70.
            A separate knob because PIL's default is 100 — **lossless alpha** —
            and for a mask-carrying texture the alpha IS the payload, so `quality`
            alone changes nothing. Measured on the 8192x4096 cloud shell: 5.31 MB
            at alpha 100, 1.80 MB at 70, 1.40 MB at 50. A diffuse coverage field
            has no detail that survives to the screen anyway, so 70 is close to
            free.
        channels: Force a channel count (3 = RGB, 4 = RGBA). Inferred otherwise;
            ``png`` keeps alpha, the lossy codecs drop it.

    Returns:
        ``(payload, encoding, width, height, channels)``.
    """
    from io import BytesIO

    fmt = fmt.lower()
    if fmt not in ("webp", "jpeg", "png"):
        raise ValueError(f"Unsupported texture format {fmt!r}; use webp, jpeg or png")

    img = _open_any_image(image)

    want_alpha = channels == 4 or (channels is None and img.mode in ("RGBA", "LA", "P"))
    if fmt == "jpeg" and want_alpha:
        raise ValueError(
            "JPEG has no alpha channel; use png (lossless) or webp for an RGBA texture"
        )
    target_mode = "RGBA" if want_alpha else "RGB"
    if img.mode != target_mode:
        img = img.convert(target_mode)

    width, height = img.size
    if fmt == "webp" and max(width, height) > MAX_WEBP_DIMENSION:
        raise ValueError(
            f"{width}x{height} exceeds WebP's hard {MAX_WEBP_DIMENSION}-pixel "
            "per-axis limit. Use jpeg at this size, or resample to "
            f"{MAX_WEBP_DIMENSION} or below."
        )

    buf = BytesIO()
    if fmt == "png":
        img.save(buf, format="PNG", optimize=True)
    elif fmt == "webp":
        # method=4 is PIL's balance point: method=6 is ~3x slower for ~2% fewer
        # bytes, which is not a trade worth making on a 33-megapixel basemap.
        img.save(
            buf,
            format="WEBP",
            quality=int(quality),
            alpha_quality=int(alpha_quality),
            method=4,
        )
    else:
        img.save(buf, format="JPEG", quality=int(quality), optimize=True)

    payload = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    return payload, fmt, width, height, 4 if want_alpha else 3


def _prepare_globe_texture(
    image: np.ndarray, fmt: str, quality: int
) -> Tuple[np.ndarray, str, int, int, int, dict[str, int]]:
    if fmt.lower() == "ktx2":
        payload = np.asarray(image)
        if np.issubdtype(payload.dtype, np.floating):
            payload = np.clip(payload * 255.0, 0, 255).astype(np.uint8)
        elif payload.dtype != np.uint8:
            payload = payload.astype(np.uint8)
        height, width, channels = payload.shape
        return (
            payload,
            "ktx2",
            width,
            height,
            channels,
            {"texture_ktx2_mode": "uastc", "texture_ktx2_quality": quality},
        )

    payload, encoding, width, height, channels = encode_texture(
        image, fmt=fmt, quality=quality, channels=3
    )
    return payload, encoding, width, height, channels, {}


def _resolve_globe_texture_format(
    fmt: str,
    quality: Optional[int],
    *,
    height: int,
    width: int,
) -> tuple[str, Optional[int]]:
    if fmt.lower() != "ktx2" or shutil.which("toktx") is not None:
        return fmt, quality

    fallback_fmt = "webp" if max(height, width) <= MAX_WEBP_DIMENSION else "jpeg"
    fallback_name = "WebP" if fallback_fmt == "webp" else "JPEG"
    aprint(
        "KTX-Software `toktx` was not found; authoring the Earth basemap as "
        f"{fallback_name} quality 90 instead. Install KTX-Software to keep it "
        "GPU-compressed."
    )
    return fallback_fmt, None


def add_textured_globe(
    scene: Any,
    name: str,
    *,
    basemap: np.ndarray,
    radius: float,
    n_lon: int = 512,
    n_lat: int = 256,
    tiles: int = 1,
    fmt: str = "webp",
    quality: Optional[int] = None,
    shading: str = "smooth",
    relief: Any = 0.0,
    **mesh_kwargs: Any,
) -> int:
    """Add a textured globe, optionally SPLIT across several mesh nodes.

    Returns the number of nodes written.

    ## Why splitting exists, and why it is nodes rather than a partition

    A single texture is capped at 16384 pixels per axis (``MAX_MESH_TEXTURE_SIZE``,
    enforced on both the write and read sides because a GPU that is handed more
    *silently clamps*, rendering the wrong image with no diagnostic). That puts the
    largest single-texture globe at 16384x8192.

    The Blue Marble master is 21600x10800, so the full-resolution basemap does not
    fit in one texture at all. Splitting the sphere into longitude bands, each its
    own node with its own texture slice, lifts the ceiling to ``tiles`` x 16384 —
    and it is the ONLY route, because ``add_mesh`` refuses ``texture`` alongside
    ``partition``: a partition re-indexes vertices but the image is node-level, so
    every part would either duplicate the whole texture or need a shared-atlas
    mechanism that does not exist.

    Useful sizes for the portable bitmap path:

    * ``tiles=1`` at 8192 — 6.0 MB (JPEG) / 4.3 MB (WebP), the comfortable default.
    * ``tiles=1`` at 16384 — 4x the pixels, 20.4 MB JPEG. Over WebP's limit.
    * ``tiles=2`` at 8192 each — also 4x the pixels, but two 4.3 MB WebP payloads
      instead of one 20.4 MB JPEG, and each node is budgeted separately so neither
      approaches the per-node decode ceiling.
    * ``tiles=4`` at 5400 each — the native 21600x10800 master, exactly.

    The cost is real and worth stating: every tile is a separate draw call and a
    separate resident decoded surface. That is why this is a knob and not the
    default.

    With KTX2 UASTC including mipmaps, the resident figures are about
    43 MiB for one 8192x4096 tile, 171 MiB across two 8192x8192 tiles, and
    297 MiB across four 5400x10800 tiles, versus 128, 512 and 890 MiB as RGBA8.
    The 16384 per-axis device limit still applies; KTX2 removes CPU bitmap
    expansion and cuts resident GPU bytes, not the geometric split.

    ## The seam

    Each tile's UVs span its own sub-image, and adjacent tiles SHARE a vertex
    column at their boundary — the same duplicated-seam trick :func:`uv_sphere`
    uses at the dateline, applied at every tile edge. Slices are cut with a
    one-column overlap so the two sides interpolate to the same colour instead of
    clamping to different edge texels, which would draw a hairline meridian.

    Args:
        scene: Scene or group to add to.
        name: Node name. Multiple tiles create a group named ``name`` containing
            ``part_0``, ``part_1``, ... .
        basemap: ``(h, w, 3)`` equirectangular RGB, row 0 at +90 latitude.
        radius: Sphere radius in scene units.
        n_lon: Total longitude divisions across the whole globe.
        n_lat: Latitude divisions.
        tiles: Number of longitude bands. 1 = a single node.
        fmt: Texture codec. This generic low-level helper deliberately defaults
            to portable ``webp``; ``build_earth`` selects ``ktx2`` for the shared
            demos. KTX2 passes RGB tiles to the mesh writer for ``toktx``
            authoring. Without ``toktx`` it falls back to WebP, or JPEG when a
            tile exceeds WebP's 16383-pixel per-axis limit.
        quality: Codec quality. ``None`` selects 2 for KTX2/UASTC and 90 for
            bitmap codecs; the missing-``toktx`` fallback resets to bitmap
            quality 90. Explicit KTX2 values use the UASTC 0-4 scale.
        shading: ``smooth`` | ``flat`` | ``none``.
        relief: Fractional radial displacement, scalar or ``(n_lat+1, n_lon+1)``.
        **mesh_kwargs: Forwarded to ``add_mesh`` (blending_mode, opacity, ...).
    """
    if tiles < 1:
        raise ValueError(f"tiles must be >= 1, got {tiles}")
    src = np.asarray(basemap)
    if src.ndim != 3 or src.shape[2] != 3:
        raise ValueError(f"Expected an (h, w, 3) basemap, got {src.shape}")
    if n_lon % tiles != 0:
        raise ValueError(
            f"n_lon ({n_lon}) must divide evenly into {tiles} tiles so their "
            "boundaries land on vertex columns"
        )

    src_h, src_w = src.shape[:2]
    if src_w % tiles != 0:
        raise ValueError(
            f"basemap width ({src_w}) must divide evenly into {tiles} tiles"
        )
    tile_src_w = src_w // tiles
    tile_width = src_w if tiles == 1 else tile_src_w + 1
    fmt, quality = _resolve_globe_texture_format(
        fmt, quality, height=src_h, width=tile_width
    )
    lon_per_tile = 360.0 / tiles
    relief_grid = np.asarray(relief, dtype=np.float32)
    resolved_quality = (
        2
        if quality is None and fmt.lower() == "ktx2"
        else 90
        if quality is None
        else quality
    )

    # Displaced terrain needs TRUE surface normals, and they must be computed on
    # the WHOLE sphere before it is sliced. Computing them per band would leave
    # each band's edge vertices averaging only their own faces, so the shading
    # would step at every tile boundary — a lit seam running pole to pole, which
    # is far more visible than the texture seam the overlap column fixes.
    global_normals: Optional[np.ndarray] = None
    if relief_grid.ndim == 2:
        whole_v, whole_f, _uv, _n = uv_sphere(n_lon, n_lat, radius, relief=relief_grid)
        global_normals = surface_vertex_normals(whole_v, whole_f)
        del whole_v, whole_f

    # With more than one tile the nodes go inside a group, and the group
    # is what carries `layer=True`. Without it the Layers panel lists "Earth_0"
    # and "Earth_1" as separate entries — an implementation detail of how the
    # basemap was split, offered to the user as two independent things to toggle.
    # One globe should be one layer.
    #
    # A PLAIN group, not a `kind=partition` one, and that is a correction rather
    # than a preference. A partition group was the obvious choice — these nodes
    # genuinely are disjoint spatial parts of one object — but it is
    # coverage-selected: the viewer anchors a partition's framing on "one part
    # fills the screen", so auto-framing put the camera at 1.137 on a radius-1.118
    # globe, i.e. standing on the surface looking at the Amazon. Measured, not
    # guessed; the single-tile build framed correctly.
    #
    # A plain group carries the layer just as well and leaves framing alone. The
    # frustum culling a partition would have bought is not worth a scene that
    # opens inside the planet.
    layer_flag = bool(mesh_kwargs.pop("layer", False))
    if tiles > 1:
        parent = scene.add_group(name, layer=layer_flag)
        target: Any = parent
        child_layer = False
    else:
        target = scene
        child_layer = layer_flag

    for t in range(tiles):
        lon0 = -180.0 + t * lon_per_tile
        vertices, faces, uvs, normals = _uv_sphere_band(
            n_lon // tiles,
            n_lat,
            radius,
            lon0,
            lon0 + lon_per_tile,
            relief_grid,
            tiles,
            t,
            tile_src_w,
        )
        if global_normals is not None:
            # Slice the whole-sphere normals to this band's columns, sharing the
            # boundary column with the neighbour exactly as the vertices do.
            cols_all = n_lon + 1
            band_cols = n_lon // tiles
            c0 = t * band_cols
            idx = (
                np.arange(n_lat + 1)[:, None] * cols_all
                + (c0 + np.arange(band_cols + 1))[None, :]
            ).ravel()
            normals = global_normals[idx]
        # One column of overlap on the right, wrapping at the dateline, so the
        # shared vertex column samples the same colour from both sides.
        c0 = t * tile_src_w
        c1 = c0 + tile_src_w
        if tiles == 1:
            slice_rgb = src
        else:
            right = src[:, c1 % src_w : (c1 % src_w) + 1]
            slice_rgb = np.concatenate([src[:, c0:c1], right], axis=1)
        payload, encoding, tw, th, tc, texture_kwargs = _prepare_globe_texture(
            slice_rgb, fmt, resolved_quality
        )
        target.add_mesh(
            f"part_{t}" if tiles > 1 else name,
            vertices=vertices,
            faces=faces,
            uvs=uvs,
            texture=payload,
            texture_encoding=encoding,
            texture_width=tw,
            texture_height=th,
            texture_channels=tc,
            **texture_kwargs,
            normals=normals,
            normal_dims=[0, 1, 2],
            shading=shading,
            # Clamped, not repeated: a tile covers a longitude BAND, so wrapping
            # its u would fetch the far edge of its own slice — the opposite side
            # of the world — at the seam. A whole globe leaves this absent so the
            # viewer keeps its equirectangular repeat-u / clamp-v default.
            **({"texture_wrap": "clamp"} if tiles > 1 else {}),
            double_sided=False,
            layer=child_layer,
            **mesh_kwargs,
        )
    return tiles


def _uv_sphere_band(
    n_lon: int,
    n_lat: int,
    radius: float,
    lon_start: float,
    lon_end: float,
    relief: np.ndarray,
    tiles: int,
    tile_index: int,
    tile_src_w: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One longitude band of a UV sphere, with UVs spanning its own sub-texture.

    The band form of :func:`uv_sphere`. Its ``u`` runs 0..1 across the band rather
    than across the world, because each band carries its own texture slice — and
    the slice has one extra column of overlap, so ``u`` is scaled to land the
    band's last vertex on the FIRST column of the neighbour's content rather than
    on the overlap column's own edge.
    """
    lon = np.linspace(lon_start, lon_end, n_lon + 1)
    lat = np.linspace(90.0, -90.0, n_lat + 1)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    if relief.ndim == 2:
        # Slice the global relief grid to this band's columns, sharing the
        # boundary column with the neighbour so the surfaces meet exactly.
        cols = relief.shape[1] - 1
        step = cols // tiles
        c0 = tile_index * step
        band_relief = relief[:, c0 : c0 + n_lon + 1]
        relief_values = band_relief.ravel()
    else:
        relief_values = relief

    vertices = lonlat_to_xyz(lon_grid.ravel(), lat_grid.ravel(), relief_values, radius)
    normals = lonlat_to_xyz(lon_grid.ravel(), lat_grid.ravel(), 0.0, 1.0)

    # u is derived from TEXELS, not from the vertex count, and getting that wrong
    # was a visible bug: the first version scaled by `n_lon / (n_lon + 1)`, which
    # silently assumed the slice had one texel per vertex. It does not — a band of
    # 256 quads carries an 8193-texel slice — so the mapping drifted by a factor of
    # `(n_lon + 1) / n_lon` and accumulated ~32 texels of misregistration by the
    # seam. On screen: the coastline jumped where two tiles met.
    #
    # The right statement is in source columns. The slice spans columns
    # `[0, tile_src_w]` inclusive (`tile_src_w + 1` of them, the last being the
    # neighbour's first), vertex `j` sits at fractional column
    # `j * tile_src_w / n_lon`, and the half-texel offset puts a vertex on a texel
    # CENTRE rather than on the boundary between two — which is what stops linear
    # filtering blending across the seam.
    frac = (lon_grid.ravel() - lon_start) / (lon_end - lon_start)
    if tiles == 1:
        u = frac
    else:
        columns = frac * tile_src_w
        u = (columns + 0.5) / (tile_src_w + 1.0)
    v = (90.0 - lat_grid.ravel()) / 180.0
    uvs = np.column_stack([u, v]).astype(np.float32)

    cols = n_lon + 1
    row = np.arange(n_lat)[:, None]
    col = np.arange(n_lon)[None, :]
    tl = (row * cols + col).ravel()
    tr = tl + 1
    bl = tl + cols
    br = bl + 1
    faces = np.concatenate(
        [np.column_stack([tl, bl, br]), np.column_stack([tl, br, tr])]
    ).astype(np.uint32)
    return vertices, faces, uvs, normals.astype(np.float32)


def _open_any_image(image: Any) -> Any:
    """Open a path, encoded bytes, or an array as a PIL image.

    The transcode half of :func:`encode_texture`, extracted so that function stays
    under the complexity ratchet — and it reads better split, because past this
    point the input's own format is irrelevant to the encode.
    """
    from io import BytesIO
    from pathlib import Path as _Path

    from PIL import Image

    if isinstance(image, (str, _Path)):
        # The pixel guard is right in general and wrong for a caller who handed us
        # a specific file on purpose; raised only around this open and restored.
        previous_limit = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = None
        try:
            with Image.open(image) as opened:
                return opened.copy()
        finally:
            Image.MAX_IMAGE_PIXELS = previous_limit
    if isinstance(image, (bytes, bytearray, memoryview)):
        return Image.open(BytesIO(bytes(image)))

    arr = np.asarray(image)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"Expected an (h, w, 3|4) texture, got {arr.shape}")
    return Image.fromarray(arr, mode="RGBA" if arr.shape[2] == 4 else "RGB")


def surface_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted per-vertex normals of an actual triangle surface.

    For a relief-displaced globe this is what :func:`uv_sphere`'s radial normals
    are not: the TRUE surface normal, which tilts with the terrain. The difference
    decides whether the mesh looks like a planet or like a low-poly model.

    Neither of the two obvious alternatives works on displaced terrain:

    * radial normals (the sphere's) ignore the slope entirely, so the relief casts
      no light and a mountain range renders as a flat colour band;
    * ``shading="flat"`` derives a per-fragment normal from screen-space
      derivatives, which is geometrically correct but CONSTANT ACROSS EACH
      TRIANGLE — so every triangle shades as one facet and the tessellation
      becomes the dominant visual feature. That was the visible faceting.

    Averaging the adjacent face normals at each vertex gives a normal that varies
    continuously, so the interpolated shading follows the terrain instead of the
    mesh. Weighted by the un-normalized cross product, whose magnitude is twice
    the triangle area — so a large triangle contributes proportionally, which is
    what keeps a pole row's slivers from dominating their vertex.

    Args:
        vertices: ``(V, 3)`` positions.
        faces: ``(F, 3)`` vertex indices.

    Returns:
        ``(V, 3)`` float32 unit normals, outward for CCW-from-outside winding.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    tri = v[f]
    # NOT normalized: |cross| = 2 * area, which is exactly the weight wanted.
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])

    out = np.zeros_like(v)
    for corner in range(3):
        np.add.at(out, f[:, corner], face_normals)

    lengths = np.linalg.norm(out, axis=1, keepdims=True)
    # A degenerate vertex (the pole rows, whose quads collapse) gets no usable
    # normal from its faces. Fall back to the radial direction there rather than
    # dividing by zero — at a pole the two agree anyway.
    degenerate = lengths[:, 0] < 1e-12
    if degenerate.any():
        radial = v[degenerate]
        radial_len = np.linalg.norm(radial, axis=1, keepdims=True)
        out[degenerate] = np.divide(
            radial, radial_len, out=np.zeros_like(radial), where=radial_len > 0
        )
        lengths[degenerate] = 1.0
    return (out / lengths).astype(np.float32)


@dataclass(frozen=True)
class SeaLevel:
    """A translucent water surface at sea level, for a relief-displaced globe.

    Only meaningful with ``relief``: on a smooth sphere there is nothing for the
    water to be above or below, so :func:`build_earth` refuses the combination
    rather than drawing an inert shell.

    The ``specular`` default is not cosmetic. A translucent blue tint over Blue
    Marble's own dark-navy ocean is two of the same colour composited together and
    is nearly invisible; a sun-glint is a highlight the basemap has nowhere, so the
    eye reads a SURFACE. ``ambient`` is likewise well above the mesh default, so
    the veil does not fall to black around the limb.
    """

    #: RGBA. Alpha stays 1.0 — translucency is the node's ``opacity``, which the
    #: Layers panel can then recover, where a baked alpha could not be.
    color: Tuple[float, float, float, float] = (0.16, 0.42, 0.72, 1.0)
    opacity: float = 0.5
    specular: float = 0.65
    shininess: float = 48.0
    ambient: float = 0.55
    #: Radial offset as a fraction of radius. **NEGATIVE** — the water sits
    #: slightly BELOW datum — and the sign is the whole point.
    #:
    #: Two things pushed this, and only the second is subtle.
    #:
    #: A positive lift floods. A fraction of the radius is not a small number once
    #: relief is exaggerated: ``+2e-4`` under 15x is ``2e-4 * 6371 km / 15`` =
    #: **85 real metres** of sea-level rise, which drowned every delta.
    #:
    #: But taking it to zero still floods, and that is a DATA-RESOLUTION limit
    #: rather than a tuning one. The relief grid is area-averaged from ETOPO onto
    #: the mesh's own lon/lat grid, so at 2048 columns each cell spans ~20 km.
    #: South Florida, the Everglades, the Nile delta — anywhere genuinely 1-3 m
    #: above the sea over tens of kilometres — averages to at or below zero in a
    #: 20 km cell, so a surface at exactly datum submerges it. The giveaway is the
    #: SHAPE of the flooding: the shoreline follows large square steps, because it
    #: is tracing grid cells rather than the 16384-wide basemap's coastline.
    #:
    #: Dropping the water ~8 m below datum keeps that low-lying coast dry. The cost
    #: is a shoreline that has retreated by 8 m of elevation, which on a basemap
    #: whose texel is ~10 km is far below one pixel. It also removes the
    #: coplanarity outright — nothing is left to z-fight — which the original
    #: positive lift existed to avoid in the first place.
    lift: float = -2e-5
    #: Grid divisions, as a fraction of the terrain's.
    #:
    #: A UV sphere puts its VERTICES on the sphere, so its flat facets dip below it
    #: by the sagitta, ``R(1 - cos(pi/n))`` — 1.9e-5 of the radius at 512
    #: divisions, which was the same order as the old positive lift and cut notches
    #: in the coastline. With the offset now negative and an order larger than the
    #: sagitta at this divisor (4.7e-6), the facets no longer reach the terrain.
    grid_divisor: int = 2


@dataclass(frozen=True)
class Clouds:
    """A Blue Marble cloud deck above a globe. See :func:`add_cloud_shell`."""

    #: Peak alpha of a fully cloudy texel.
    #:
    #: Read this together with `luminous` blending, because the two interact and
    #: getting it wrong made the deck invisible. Under alpha-over, strength is a
    #: LERP toward white — 0.22 gets you 22% of the way there whatever is behind.
    #: Under additive it is an ADDITION of 0.22, which ACES then compresses along
    #: with everything else in the highlights, so on a mid-bright basemap it
    #: almost vanishes. Additive needs more amplitude than alpha-over for the same
    #: apparent density.
    strength: float = 0.55
    #: Node intensity, multiplying the deck's brightness independently of its
    #: alpha. The tuning knob to reach for first: it changes how BRIGHT the cloud
    #: is without changing which texels are cloud, and it stays live in the Layers
    #: panel where `strength` is baked into the texture.
    intensity: float = 1.0
    #: >1 thins the source's low-luminance haze floor.
    gamma: float = 1.8
    #: Shell height as a fraction of radius. ``None`` derives it from the relief's
    #: own peak — which is the only correct choice on an exaggerated globe, where a
    #: copied constant can sit below the mountains.
    altitude: Optional[float] = None
    #: Texture width. 4096 is the measured balance point: the source has 2048 of
    #: real detail, and past 4096 the bytes buy interpolation the GPU does anyway.
    width: int = 4096
    n_lon: int = 192
    n_lat: int = 96


def build_earth(
    scene: Any,
    name: str = "Earth",
    *,
    radius: float = 1.0,
    n_lon: int = 512,
    n_lat: int = 256,
    texture_width: int = 16384,
    tiles: int = 2,
    fmt: str = "ktx2",
    quality: Optional[int] = None,
    shading: str = "smooth",
    relief: Any = 0.0,
    basemap: Optional[np.ndarray] = None,
    clouds: Optional[Clouds] = None,
    sea_level: Optional[SeaLevel] = None,
    **mesh_kwargs: Any,
) -> int:
    """Build a complete textured Earth: basemap tiles, clouds, and water.

    The ONE entry point the Earth demos share, and the reason it exists is a bug
    rather than tidiness. Each demo used to assemble its own globe from the same
    pieces, and when the pieces gained an argument one demo missed it — the
    biodiversity globe shipped without ``texture_channels`` and failed at write
    time. Four copies of a twelve-line recipe is four chances to drift.

    Everything that genuinely differs between the four is an argument here:

    ==================  =========================================================
    demo                what it asks for
    ==================  =========================================================
    earthquakes         lit globe, full-strength clouds
    ocean currents      UNLIT (the basemap is a reference for the speed ramp, so a
                        view-anchored key would make the same ocean read
                        differently from different angles), thin clouds
    global rivers       relief, lit, sea level, thin clouds, derived altitude
    biodiversity        unlit, thin clouds, nD ``fill`` passthrough
    ==================  =========================================================

    Args:
        scene: Scene or group to add to.
        name: Node/group name for the globe.
        radius: Sphere radius in scene units.
        n_lon: Longitude divisions across the whole globe.
        n_lat: Latitude divisions.
        texture_width: Basemap width to fetch; halved per axis into ``tiles``.
        tiles: Longitude bands, each its own node. See :func:`add_textured_globe`.
        fmt: Basemap codec; defaults to GPU-compressed UASTC ``ktx2`` and falls
            back to WebP when ``toktx`` is unavailable, or JPEG when the tile
            exceeds WebP's 16383-pixel per-axis limit. Select a bitmap codec
            explicitly to require the portable path.
        quality: Basemap codec quality; defaults to 90 for bitmap codecs and
            UASTC level 2 for KTX2.
        shading: ``smooth`` | ``flat`` | ``none``.
        relief: Fractional radial displacement, scalar or ``(n_lat+1, n_lon+1)``.
        basemap: Supply the RGB array directly instead of fetching it — for a demo
            that has already loaded one, or for a test.
        clouds: A :class:`Clouds` to add a deck; ``None`` for none.
        sea_level: A :class:`SeaLevel` to add water; requires ``relief``.
        **mesh_kwargs: Forwarded to the globe's ``add_mesh`` (blending_mode,
            opacity, intensity, layer, dim_order, fill, ...).

    Returns:
        The number of globe tiles written.
    """
    relief_arr = np.asarray(relief, dtype=np.float32)
    if sea_level is not None and relief_arr.ndim == 0:
        raise ValueError(
            "sea_level needs `relief`: on a smooth sphere the water surface is "
            "coincident with the terrain everywhere, so there is nothing for it "
            "to be above or below"
        )

    if basemap is None:
        basemap, basemap_w, basemap_h = blue_marble_basemap(width=texture_width)
    else:
        basemap = np.asarray(basemap)
        basemap_h, basemap_w = basemap.shape[:2]
    # A small fallback image should not be split: the tiles would only add a seam.
    resolved_tiles = tiles if basemap_w > 4096 else 1

    n_written = add_textured_globe(
        scene,
        name,
        basemap=basemap,
        radius=radius,
        n_lon=n_lon,
        n_lat=n_lat,
        tiles=resolved_tiles,
        fmt=fmt,
        quality=quality,
        shading=shading,
        relief=relief_arr,
        **mesh_kwargs,
    )

    # nD placement has to reach the shells too, or the atmosphere and the sea
    # would appear in one slice and vanish from the rest.
    nd_kwargs = {k: mesh_kwargs[k] for k in ("dim_order", "fill") if k in mesh_kwargs}

    if sea_level is not None:
        water_v, water_f, _uv, water_n = uv_sphere(
            max(3, n_lon // sea_level.grid_divisor),
            max(2, n_lat // sea_level.grid_divisor),
            radius * (1.0 + sea_level.lift),
        )
        scene.add_mesh(
            f"{name} sea level",
            vertices=water_v,
            faces=water_f,
            normals=water_n,
            normal_dims=[0, 1, 2],
            # A uniform colour: the water carries no spatial information of its
            # own, so a texture and UVs would both be dead weight.
            colors=sea_level.color,
            # `luminous` — additive and depth-tested, the same rule the cloud deck
            # follows. Water SCATTERS light toward the viewer rather than replacing
            # what is beneath it, which is what additive composition means; the
            # depth test still hides the far hemisphere behind the opaque globe;
            # and the specular glint becomes an additive highlight, which is how a
            # sun-glint actually behaves.
            #
            # Order-independence is the practical win, as it is for the clouds:
            # `normal` would put the water in the sorted transparent set alongside
            # the cloud shell and every river ribbon.
            blending_mode="luminous",
            shading="smooth",
            specular=sea_level.specular,
            shininess=sea_level.shininess,
            ambient=sea_level.ambient,
            opacity=sea_level.opacity,
            double_sided=False,
            layer=True,
            **nd_kwargs,
        )

    if clouds is not None:
        altitude = clouds.altitude
        if altitude is None:
            # Derived from the relief's own peak. On the rivers globe Everest sits
            # at 2.1% of the radius under a 15x exaggeration, so the 1.2% the flat
            # globes use would put the atmosphere below the Himalaya.
            peak = float(np.max(relief_arr)) if relief_arr.size else 0.0
            altitude = max(0.012, peak * 1.35)
        add_cloud_shell(
            scene,
            f"{name} clouds",
            radius=radius,
            altitude=altitude,
            strength=clouds.strength,
            gamma=clouds.gamma,
            width=clouds.width,
            n_lon=clouds.n_lon,
            n_lat=clouds.n_lat,
            intensity=clouds.intensity,
            **nd_kwargs,
        )

    return n_written
