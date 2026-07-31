#!/usr/bin/env python3
"""Demo: Ocean Currents of Earth — a Blue Marble globe + surface-current streamlines.

The planet's surface circulation drawn the way NASA's "Perpetual Ocean" draws it:
tens of thousands of short, equal-length ribbons traced through a real
eddy-resolving ocean model, so the Gulf Stream, the Kuroshio, the Agulhas
retroflection and the Antarctic Circumpolar Current emerge from the flow itself
rather than from any hand-drawn arrows.

Two of Luxar's three geometry types, at global scale:

  * **Earth (Points)** — a **jittered Fibonacci sphere** sampled from the NASA
    Blue Marble *land_shallow_topo* texture. The jitter matters: a bare Fibonacci
    lattice is a *lattice*, and at radii ~ point spacing its spiral arms beat
    against themselves into visible moire "worms" over land and sea alike.
    Dithering each point by ~1 mean-spacing cell turns that structure into noise.
  * **Currents (Lines)** — HYCOM surface velocities, RK4-integrated into
    **connected polylines** draped just above the globe, coloured deep-blue ->
    white by current speed with the tail fading out (per-vertex RGBA), so each
    ribbon reads as a comet pointing downstream.

WHY FIXED ARC LENGTH, NOT FIXED TIME
------------------------------------
Streamlines are advected with a fixed **arc-length** step rather than a fixed
timestep. Under a fixed timestep, fast water draws long streaks and slow water
draws stubs, which collapses into a tangle in the boundary currents and empty
space in the gyre interiors. Fixing arc length makes every ribbon the same
on-screen length and hands the speed information to *colour* instead — this is
what produces the legible "Van Gogh" texture. Speed is sampled along the path and
carried separately for that purpose.

RENDERING NOTE — WHY `normal` AND NOT `additive`
------------------------------------------------
The lines use ``blending_mode="normal"``. Additive blending does not respect the
depth buffer, so with an additive line layer the currents on the **far side** of
the globe show straight through the near side and appear painted across the
continents. It looks exactly like a broken land mask and is not one.

KNOWN LIMITATION — NO RIBBONS POLEWARD OF 80 DEG
------------------------------------------------
The requested HYCOM subset spans 80S..80N, so there are no streamlines over the
Arctic or the high Southern Ocean. Because the rest of the ocean is covered in
bright ribbons, the uncovered caps read as darker water behind a fairly hard
edge at the cutoff latitude. That edge is the data domain, not a masking bug.

================================================================================
SELF-CONTAINED / REGENERATING (no LFS asset)
================================================================================
Ships only code. First run downloads ~72 MB of source data (both public, direct
download, **no account or API key**), builds the scene, and caches the sources
under ``~/.cache/luxar/ocean_currents_earth/``. Later runs load the built scene
instantly; ``--recompute`` rebuilds without re-fetching.

DATA SOURCES & CITATIONS
------------------------
HYCOM + NCODA Global 1/12 deg Analysis (GLBy0.08, expt_93.0), surface u/v.
    Chassignet et al. (2007), J. Mar. Syst. 65. Publicly available; funded by
    the US Navy and the National Ocean Partnership Program.
    https://www.hycom.org/dataserver/gofs-3pt1/analysis
NASA Blue Marble: Next Generation ("land_shallow_topo"), NASA Earth Observatory
    (Reto Stockli). Public domain. https://visibleearth.nasa.gov/

NOTE ON PROVENANCE: NASA's "Perpetual Ocean" film was rendered from **ECCO2**.
This demo uses **HYCOM**, an equivalent eddy-resolving ocean model that is
reachable without credentials. It is the same *kind* of visualization, not a
reproduction of that specific product.

USAGE
-----
    luxar demo run ocean_currents_earth
    python demo_ocean_currents_earth.py [--recompute] [--no-serve] [--serve-only]
"""

from __future__ import annotations

DEMO_META = {
    "key": "ocean_currents_earth",
    "title": "Ocean Currents of Earth",
    "description": "HYCOM surface-current streamlines (Lines) over a NASA Blue Marble globe (Points).",
    "category": "geoscience",
    "geometry": "points+lines",
    "requirements": {
        "download_mb": 72,
        "compute": "medium",
        "gpu": "none",
        "local_data": None,
    },
    "caches": ["ocean_currents_earth"],
    "outputs": ["ocean_currents_earth"],
}

from pathlib import Path
from typing import Final

import numpy as np
from arbol import Arbol, aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import CameraConfig, ViewerConfig
from luxar.demos import require_module
from luxar.encoding import EncodingMode
from luxar.utils.demos import cached_download, launch_viewer, parse_demo_flags
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

DEMO_NAME: Final = "ocean_currents_earth"

# HYCOM NCSS: one surface (vertCoord=0) timestep of the global 1/12 deg analysis.
# NOTE: HYCOM longitudes run 0..360, NOT -180..180 — passing west=-180 silently
# returns only half the globe (the server clips instead of erroring).
HYCOM_URL: Final = (
    "https://ncss.hycom.org/thredds/ncss/GLBy0.08/expt_93.0/uv3z"
    "?var=water_u&var=water_v"
    "&north=80&west=0&east=359.92&south=-80&horizStride=1"
    "&time_start=2024-01-01T00:00:00Z&time_end=2024-01-01T00:00:00Z"
    "&vertCoord=0&accept=netcdf"
)
BLUE_MARBLE_URL: Final = (
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57752/"
    "land_shallow_topo_2048.jpg"
)

R_EARTH_KM: Final = 6371.0
RADIUS: Final = 100.0  # globe radius in scene units

N_GLOBE: Final = 8_000_000  # jittered Fibonacci-sphere surface points
GLOBE_RADII: Final = 0.098  # ~1.4x mean point spacing -> a sealed shell
N_SEEDS: Final = 220_000  # streamlines
N_STEPS: Final = 52  # vertices per streamline
STEP_KM: Final = 14.0  # arc-length step -> ~730 km ribbons
FIELD_STRIDE: Final = 2  # subsample the 1/12 deg grid for advection
FLOW_LIFT: Final = 0.0015  # lift ribbons just clear of the globe shell
LINE_WIDTH: Final = 0.026
LINE_OPACITY: Final = 0.95
LINE_INTENSITY: Final = 1.0
SPEED_FULL_SCALE: Final = 1.0  # m/s mapped to the top of the colour ramp
MIN_SEED_SPEED: Final = 0.04  # skip near-still water when seeding
STALL_SPEED: Final = 0.02  # freeze a ribbon that runs out of current
LAT_LIMIT: Final = 79.9  # HYCOM's grid stops at +/-80

# A single Lines node cannot exceed 16,777,216 vertices: the viewer's lines
# loader collects unique vertex indices into a JS Set, and V8 caps Set at 2**24
# (royerlab/luxar#1049). Above it the node renders NOTHING, with only a console
# error — so this is a hard authoring budget, not a soft guideline.
MAX_LINE_VERTICES: Final = 16_777_216

FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]

CACHE_DIR: Final = Path.home() / ".cache" / "luxar" / DEMO_NAME

Arbol.max_depth = 5


# =============================================================================
# Pure helpers (unit-tested; no network / no IO)
# =============================================================================


def fibonacci_sphere(n: int, *, jitter: bool = True, seed: int = 1234) -> tuple:
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


def lonlat_to_xyz(lon: np.ndarray, lat: np.ndarray, relief: np.ndarray) -> np.ndarray:
    """Map geographic degrees + fractional ``relief`` to sphere xyz.

    ``y`` is the north-pole axis and longitude increases eastward; the ``-z``
    keeps the frame right-handed (East x North = outward) so the globe is not
    mirrored. Matches ``demo_global_rivers_earth``.
    """
    la, lo = np.radians(lat), np.radians(lon)
    r = RADIUS * (1.0 + relief)
    cl = np.cos(la)
    return np.column_stack(
        [r * cl * np.cos(lo), r * np.sin(la), -r * cl * np.sin(lo)]
    ).astype(np.float32)


def sample_equirect(tex: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Bilinearly sample an equirectangular RGB texture at ``lon``/``lat``.

    Vectorized over all points (``demo_earthquakes_3d`` samples one point per
    Python-loop iteration, which does not scale to millions). Longitude wraps;
    latitude clamps.

    Args:
        tex: ``(h, w, 3)`` uint8 or float texture; row 0 is +90 deg latitude.
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
    c0 = t[y0, x0] * (1.0 - wx) + t[y0, x1] * wx
    c1 = t[y1, x0] * (1.0 - wx) + t[y1, x1] * wx
    return np.clip((c0 * (1.0 - wy) + c1 * wy) / 255.0, 0.0, 1.0).astype(np.float32)


def build_lut(stops: list) -> np.ndarray:
    """Build a ``(256, 3)`` float32 LUT in [0, 1] by interpolating ``stops``.

    Args:
        stops: ``[(t, (r, g, b)), ...]`` with ``t`` ascending in [0, 1] and
            channels in 0..255.
    """
    xs = np.array([s[0] for s in stops], dtype=np.float64)
    cs = np.array([s[1] for s in stops], dtype=np.float64)
    t = np.linspace(0.0, 1.0, 256)
    lut = np.column_stack([np.interp(t, xs, cs[:, k]) for k in range(3)])
    return (lut / 255.0).astype(np.float32)


#: Deep blue -> cyan -> white, echoing the NASA "Perpetual Ocean" ribbons.
CURRENT_LUT: Final = build_lut(
    [
        (0.00, (20, 45, 120)),
        (0.30, (40, 120, 210)),
        (0.60, (150, 210, 245)),
        (1.00, (255, 255, 255)),
    ]
)


class LonLatField:
    """Surface-current field on a regular lon/lat grid, with a land mask.

    ``u``/``v`` are eastward/northward velocity in m/s with NaN on land. Land is
    kept as NaN for masking but interpolation runs on zero-filled copies, so a
    ribbon that reaches the coast stalls instead of producing NaN positions.
    """

    def __init__(
        self, u: np.ndarray, v: np.ndarray, lon: np.ndarray, lat: np.ndarray
    ) -> None:
        if u.shape != v.shape:
            raise ValueError(f"u/v shape mismatch: {u.shape} vs {v.shape}")
        if u.shape != (len(lat), len(lon)):
            raise ValueError(f"u {u.shape} does not match (lat, lon) grid")
        self.nlat, self.nlon = u.shape
        self.lon0 = float(lon[0])
        self.dlon = float(lon[1] - lon[0])
        self.lat0 = float(lat[0])
        self.dlat = float(lat[1] - lat[0])
        self.wet = np.isfinite(u) & np.isfinite(v)
        self._u = np.nan_to_num(u).astype(np.float32)
        self._v = np.nan_to_num(v).astype(np.float32)

    def sample(self, lon: np.ndarray, lat: np.ndarray) -> tuple:
        """Bilinearly sample ``(u, v)`` in m/s. Periodic in lon, clamped in lat."""
        fx = np.mod((lon - self.lon0) / self.dlon, self.nlon)
        fy = np.clip((lat - self.lat0) / self.dlat, 0, self.nlat - 1)
        x0 = np.floor(fx).astype(np.int64)
        y0 = np.floor(fy).astype(np.int64)
        x1 = (x0 + 1) % self.nlon
        y1 = np.minimum(y0 + 1, self.nlat - 1)
        wx = (fx - x0).astype(np.float32)
        wy = (fy - y0).astype(np.float32)

        def bilinear(a: np.ndarray) -> np.ndarray:
            top = a[y0, x0] * (1.0 - wx) + a[y0, x1] * wx
            bot = a[y1, x0] * (1.0 - wx) + a[y1, x1] * wx
            return top * (1.0 - wy) + bot * wy

        return bilinear(self._u), bilinear(self._v)

    def is_wet(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Nearest-cell ocean mask lookup."""
        ix = np.mod(np.rint((lon - self.lon0) / self.dlon).astype(np.int64), self.nlon)
        iy = np.clip(
            np.rint((lat - self.lat0) / self.dlat).astype(np.int64), 0, self.nlat - 1
        )
        return self.wet[iy, ix]


def advect_streamlines(
    field: LonLatField,
    seed_lon: np.ndarray,
    seed_lat: np.ndarray,
    n_steps: int,
    step_km: float,
) -> tuple:
    """RK4-advect seeds along the UNIT flow with a fixed arc-length step.

    Integrating the *normalized* field in arc length (not the raw field in time)
    gives every streamline the same length; ``speed`` is sampled alongside so the
    caller can colour by it. A ribbon that beaches or runs out of current is
    frozen in place, which collapses its remaining segments to zero length rather
    than letting it wander onto land.

    Args:
        field: The surface-current field.
        seed_lon: ``(m,)`` seed longitudes, degrees. Must be in wet cells.
        seed_lat: ``(m,)`` seed latitudes, degrees. Must be in wet cells.
        n_steps: Number of steps; each path has ``n_steps + 1`` vertices.
        step_km: Arc length per step, kilometres.

    Returns:
        ``(lon, lat, speed)``, each ``(m, n_steps + 1)`` float32.

    Raises:
        ValueError: any seed sits on land. Only the *next* position is masked
            each step, so a dry seed would be frozen at its start point and emit
            a whole ribbon lying on land — a silent, plausible-looking wrong
            render. Use :func:`seed_ocean_points`, which guarantees wet seeds.
    """
    dry = ~field.is_wet(seed_lon, seed_lat)
    if dry.any():
        raise ValueError(
            f"{int(dry.sum())} of {len(seed_lon)} seeds are on land; "
            "seed with seed_ocean_points()"
        )
    deg = np.degrees(step_km / R_EARTH_KM)  # angular step

    def unit_deriv(lo: np.ndarray, la: np.ndarray) -> tuple:
        """d(lon, lat)/ds in degrees per unit arc, from the unit flow direction."""
        u, v = field.sample(lo, la)
        speed = np.hypot(u, v)
        ok = speed > 1e-6
        safe = np.where(ok, speed, 1.0)
        # eastward degrees per unit arc grow as 1/cos(lat)
        cos_lat = np.maximum(np.cos(np.radians(la)), 1e-3)
        return np.where(ok, u / safe, 0.0) / cos_lat, np.where(ok, v / safe, 0.0)

    m = len(seed_lon)
    out_lon = np.empty((m, n_steps + 1), dtype=np.float32)
    out_lat = np.empty_like(out_lon)
    out_speed = np.empty_like(out_lon)

    lo = seed_lon.astype(np.float64).copy()
    la = seed_lat.astype(np.float64).copy()
    alive = np.ones(m, dtype=bool)

    for k in range(n_steps + 1):
        u, v = field.sample(lo, la)
        out_speed[:, k] = np.hypot(u, v)
        out_lon[:, k] = lo
        out_lat[:, k] = la
        if k == n_steps:
            break
        d1l, d1a = unit_deriv(lo, la)
        d2l, d2a = unit_deriv(lo + 0.5 * deg * d1l, la + 0.5 * deg * d1a)
        d3l, d3a = unit_deriv(lo + 0.5 * deg * d2l, la + 0.5 * deg * d2a)
        d4l, d4a = unit_deriv(lo + deg * d3l, la + deg * d3a)
        next_lon = lo + deg / 6.0 * (d1l + 2.0 * d2l + 2.0 * d3l + d4l)
        next_lat = np.clip(
            la + deg / 6.0 * (d1a + 2.0 * d2a + 2.0 * d3a + d4a), -LAT_LIMIT, LAT_LIMIT
        )
        alive = (
            alive & field.is_wet(next_lon, next_lat) & (out_speed[:, k] > STALL_SPEED)
        )
        lo = np.where(alive, next_lon, lo)
        la = np.where(alive, next_lat, la)

    return out_lon, out_lat, out_speed


def polyline_segment_indices(n_paths: int, n_vertices: int) -> np.ndarray:
    """Indices joining consecutive vertices WITHIN each path (never across).

    Sharing a vertex at each joint lets the line material render a seamless join
    instead of two overlapping end-caps; excluding the path boundaries stops the
    last vertex of one ribbon connecting to the first of the next.

    Returns:
        ``(2 * n_paths * (n_vertices - 1),)`` uint32, consecutive pairs.
    """
    if n_vertices < 2:
        raise ValueError(f"n_vertices must be >= 2, got {n_vertices}")
    base = (np.arange(n_paths, dtype=np.int64) * n_vertices)[:, None]
    starts = base + np.arange(n_vertices - 1, dtype=np.int64)[None, :]
    return np.stack([starts, starts + 1], axis=-1).reshape(-1).astype(np.uint32)


def seed_ocean_points(
    field: LonLatField, n: int, *, seed: int = 0, min_speed: float = MIN_SEED_SPEED
) -> tuple:
    """Rejection-sample ``n`` seeds uniformly over the *moving* ocean.

    Latitudes are drawn in ``sin(lat)`` so the samples are area-uniform on the
    sphere rather than clustered at the poles; cells that are land or nearly
    still are rejected so no ribbon starts where it cannot go anywhere.
    """
    rng = np.random.default_rng(seed)
    lo_parts: list = []
    la_parts: list = []
    got = 0
    sin_lo, sin_hi = np.sin(np.radians(-LAT_LIMIT)), np.sin(np.radians(LAT_LIMIT))
    while got < n:
        batch = int((n - got) * 1.8) + 1000
        lon = rng.uniform(0.0, 360.0, batch)
        lat = np.degrees(np.arcsin(rng.uniform(sin_lo, sin_hi, batch)))
        keep = field.is_wet(lon, lat)
        u, v = field.sample(lon, lat)
        keep &= np.hypot(u, v) > min_speed
        lo_parts.append(lon[keep])
        la_parts.append(lat[keep])
        got += int(keep.sum())
    return np.concatenate(lo_parts)[:n], np.concatenate(la_parts)[:n]


def globe_camera(lon: float, lat: float, *, distance: float = 2.05) -> CameraConfig:
    """Opening pose looking straight down at ``(lon, lat)`` on the globe."""
    la, lo = np.radians(lat), np.radians(lon)
    normal = np.array(
        [np.cos(la) * np.cos(lo), np.sin(la), -np.cos(la) * np.sin(lo)],
        dtype=np.float64,
    )
    return CameraConfig(
        position=tuple((normal * RADIUS * distance).tolist()),
        target=tuple((normal * RADIUS * 0.9).tolist()),
        up=(0.0, 1.0, 0.0),
        fov=42.0,
    )


# =============================================================================
# Data loading
# =============================================================================


def load_hycom_surface(path: Path, stride: int = FIELD_STRIDE) -> LonLatField:
    """Read HYCOM surface u/v from a netCDF3 file into a :class:`LonLatField`.

    ``accept=netcdf`` (classic netCDF3) is requested deliberately so
    ``scipy.io.netcdf_file`` can read it — no netCDF4/h5py/xarray needed. Values
    are ``int16 * scale_factor`` with land at ``_FillValue``.
    """
    netcdf_file = require_module("scipy.io").netcdf_file
    data = netcdf_file(str(path), "r", mmap=False)
    var_u = data.variables["water_u"]
    var_v = data.variables["water_v"]
    scale = float(getattr(var_u, "scale_factor", 1.0))
    fill = getattr(var_u, "_FillValue", None)

    raw_u = var_u.data[0, 0, ::stride, ::stride]
    raw_v = var_v.data[0, 0, ::stride, ::stride]
    lat = data.variables["lat"].data[::stride].astype(np.float64)
    lon = data.variables["lon"].data[::stride].astype(np.float64)

    u = np.where(raw_u == fill, np.nan, raw_u * scale).astype(np.float32)
    v = np.where(raw_v == fill, np.nan, raw_v * scale).astype(np.float32)
    aprint(
        f"HYCOM grid {u.shape[0]}x{u.shape[1]}, "
        f"{np.isfinite(u).mean() * 100:.1f}% ocean"
    )
    return LonLatField(u, v, lon, lat)


def download_sources() -> tuple:
    """Fetch (and cache) the HYCOM field and the Blue Marble texture."""
    with asection("Downloading source data (~72 MB, first run only)"):
        hycom = cached_download(HYCOM_URL, DEMO_NAME, "hycom_glby008_surface_uv.nc")
        marble = cached_download(
            BLUE_MARBLE_URL, DEMO_NAME, "land_shallow_topo_2048.jpg"
        )
    return hycom, marble


# =============================================================================
# Scene
# =============================================================================


def build_scene(hycom_path: Path, marble_path: Path, output_path: Path) -> Path:
    """Build the globe + current-streamline scene and write it to ``output_path``."""
    image_module = require_module("PIL.Image")

    with asection("Loading surface currents"):
        field = load_hycom_surface(hycom_path)

    with asection(f"Building globe ({N_GLOBE:,} points)"):
        texture = np.asarray(image_module.open(marble_path).convert("RGB"))
        glon, glat = fibonacci_sphere(N_GLOBE)
        gpos = lonlat_to_xyz(glon, glat, np.zeros(N_GLOBE, dtype=np.float64))
        gcolors = sample_equirect(texture, glon, glat)

    with asection(f"Advecting {N_SEEDS:,} streamlines x {N_STEPS} steps"):
        seed_lon, seed_lat = seed_ocean_points(field, N_SEEDS)
        path_lon, path_lat, path_speed = advect_streamlines(
            field, seed_lon, seed_lat, N_STEPS, STEP_KM
        )
        aprint(
            f"speed along paths: median {np.median(path_speed):.3f} m/s, "
            f"p99 {np.percentile(path_speed, 99):.3f} m/s"
        )

    with asection("Assembling ribbons"):
        n_paths, n_vertices = path_lon.shape
        total = n_paths * n_vertices
        if total > MAX_LINE_VERTICES:
            raise ValueError(
                f"{total:,} line vertices exceeds the viewer's {MAX_LINE_VERTICES:,} "
                "per-node ceiling (royerlab/luxar#1049); lower N_SEEDS or N_STEPS"
            )
        vertices = lonlat_to_xyz(
            path_lon.ravel(),
            path_lat.ravel(),
            np.full(total, FLOW_LIFT, dtype=np.float64),
        )
        norm = np.clip(path_speed.ravel() / SPEED_FULL_SCALE, 0.0, 1.0)
        rgb = CURRENT_LUT[np.clip((norm * 255).astype(np.int64), 0, 255)]
        # Per-vertex alpha ramps head-bright -> tail-faint, so each ribbon reads
        # as a comet and the direction of flow is unambiguous.
        taper = np.linspace(0.15, 1.0, n_vertices, dtype=np.float32) ** 1.5
        alpha = np.tile(taper, (n_paths, 1)).ravel()
        colors = np.column_stack([rgb, alpha]).astype(np.float32)
        indices = polyline_segment_indices(n_paths, n_vertices)
        aprint(
            f"{n_paths:,} ribbons, {len(indices) // 2:,} segments, {total:,} vertices"
        )

    with asection("Writing scene"):
        dims = Dimensions(
            [
                Dimension("x", unit="", display=True),
                Dimension("y", unit="", display=True),
                Dimension("z", unit="", display=True),
            ]
        )
        with LuxarZarrCompiler(output_path, encoding_mode=EncodingMode.PRECISION) as c:
            scene = c.create_scene(
                dimensions=dims,
                viewer_config=ViewerConfig(
                    # Neutral rather than ACES: the ramp is an encoding of speed,
                    # and ACES shifts hues away from the intended blue->white.
                    tone_mapping="Neutral",
                    camera=globe_camera(-84.0, 25.0),
                ),
            )
            scene.attrs["title"] = "Ocean Currents of Earth — HYCOM surface circulation"
            scene.add_points(
                "earth",
                positions=gpos,
                radii=GLOBE_RADII,
                colors=gcolors,
                blending_mode="normal",
                opacity=1.0,
                layer=True,
                # A geometric `stream:` ladder gives a fast first paint where a
                # stratified sampler would dump ~all 8M into one final commit.
                additive_lod=dict(counts="stream:20000", method="random", seed=0),
            )
            scene.add_lines(
                "currents",
                vertices=vertices,
                widths=LINE_WIDTH,
                colors=colors,
                indices=indices,
                line_type="indexed",
                # `normal`, NOT `additive` — see the module docstring: additive
                # ignores depth, so far-side currents bleed across the continents.
                blending_mode="normal",
                opacity=LINE_OPACITY,
                intensity=LINE_INTENSITY,
                layer=True,
            )
            scene.add_text(
                "Ocean Currents of Earth",
                position=(0.02, 0.02),
                font_size=0.045,
                anchor="top-left",
                color="rgba(255,255,255,0.75)",
                blend_mode="difference",
            )
            scene.add_text(
                "HYCOM GLBy0.08 surface velocities • NASA Blue Marble topography",
                position=(0.98, 0.97),
                font_size=0.015,
                anchor="bottom-right",
                color="rgba(200,200,220,0.5)",
            )
        aprint(f"Scene saved: {output_path}")
    return output_path


def load_or_build_scene(output_path: Path) -> Path:
    """Return the built scene, regenerating it on a fresh system."""
    if output_path.exists() and not RECOMPUTE:
        aprint(f"Using existing scene: {output_path}")
        return output_path
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    hycom_path, marble_path = download_sources()
    return build_scene(hycom_path, marble_path, output_path)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    aprint("=" * 70)
    aprint("Demo: Ocean Currents of Earth — HYCOM surface circulation")
    aprint("=" * 70)

    output_path = get_demos_output_dir() / f"{DEMO_NAME}.luxar.zarr"

    if SERVE_ONLY:
        if output_path.exists():
            launch_viewer(output_path)
        else:
            aprint("No scene found. Run without --serve-only first.")
        return

    scene_path = load_or_build_scene(output_path)

    if NO_SERVE:
        aprint(f"Scene ready at {scene_path}")
    else:
        aprint("Data credit: HYCOM (GOFS 3.1) + NASA Blue Marble (public domain)")
        launch_viewer(scene_path)


if __name__ == "__main__":
    main()
