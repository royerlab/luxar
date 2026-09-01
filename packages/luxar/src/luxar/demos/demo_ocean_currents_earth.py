#!/usr/bin/env python3
"""Demo: Ocean Currents of Earth — a Blue Marble globe + surface-current streamlines.

The planet's surface circulation drawn the way NASA's "Perpetual Ocean" draws it:
tens of thousands of short, equal-length ribbons traced through a real
eddy-resolving ocean model, so the Gulf Stream, the Kuroshio, the Agulhas
retroflection and the Antarctic Circumpolar Current emerge from the flow itself
rather than from any hand-drawn arrows.

Two of Luxar's four geometry types, at global scale:

  * **Earth (Mesh)** — a tiled UV sphere carrying the NASA Blue Marble
    *land_shallow_topo* texture at its full 16384-pixel master width. Per-fragment
    texture sampling keeps coastlines sharp without spending millions of points
    on the planet backdrop.
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

RENDERING NOTE — WHY `luminous`
-------------------------------
The lines use ``blending_mode="luminous"``, and the distinction that matters is
between ``luminous`` and plain ``additive`` — not between additive and
``normal``, which is how this note originally read.

``luminous`` is additive **and** depth-tested. So it keeps what the old ``normal``
choice was protecting (the far-side network stays hidden behind the opaque globe)
while gaining two things ``normal`` cannot give:

* overlapping ribbons ACCUMULATE, which is informative rather than incidental — a
  boundary current concentrates flow, so it gets brighter;
* the result is order-INDEPENDENT. ``normal`` puts every ribbon in the viewer's
  sorted transparent set, so what you see depends on getting depth order right
  across 11M segments and the cloud shell; additive composition is commutative, so
  that whole class of sorting artefact does not arise.

The original note below is kept because its warning is still true of plain
``additive``, which is a different mode:

Additive blending does not respect the
depth buffer, so with an additive line layer the currents on the **far side** of
the globe show straight through the near side and appear painted across the
continents. It looks exactly like a broken land mask and is not one.

RESIDENCY NOTE — WHY THE CURRENTS ARE PARTITION-OF-LOD
------------------------------------------------------
`partition=` bounds node size but not residency: every visible part is fetched
and drawn, and at the opening whole-globe pose every current tile is in view.
The current layer is therefore a `kind=partition` of per-tile `kind=lod`
ladders. Each coarse level keeps fewer **whole ribbons** and widens them
linearly, preserving the field's apparent ink without replacing the curves
with synthetic geometry.

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
instantly; ``--recompute`` rebuilds without re-fetching. A scene written by an
older demo or Luxar writer is rebuilt automatically (its ``builder_fingerprint``
no longer matches); pass ``--keep-stale`` to serve it anyway.

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
                                        [--keep-stale]
"""

from __future__ import annotations

DEMO_META = {
    "key": "ocean_currents_earth",
    "title": "Ocean Currents of Earth",
    "description": "HYCOM surface-current streamlines (Lines) over a NASA Blue Marble globe (textured Mesh).",
    "category": "geoscience",
    # "mixed": the globe is a textured Mesh and the currents are Lines. The
    # closed vocabulary in registry.py has no "mesh+lines".
    "geometry": "mixed",
    "requirements": {
        "download_mb": 72,
        "compute": "medium",
        "gpu": "none",
        "local_data": None,
    },
    "caches": ["ocean_currents_earth", "blue_marble"],
    "outputs": ["ocean_currents_earth"],
    "citation": {
        "short": "HYCOM GOFS 3.1 (Chassignet et al. 2007); NASA Blue Marble",
        "ref": "Chassignet et al. 2007 / NASA",
        "doi": "10.1016/j.jmarsys.2005.09.016",
    },
}

from pathlib import Path
from typing import Final

import numpy as np
from arbol import Arbol, aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.group.compositing import position_bounds_from_array
from luxar.core.group.lod.group import (
    coverage_fractions,
    level_additive_lod,
    partitioned_coverage_fractions,
)
from luxar.core.group.partition import bsp_leaf_parts, spatial_bsp_tree
from luxar.core.viewer_config import CameraConfig, ViewerConfig
from luxar.demos import (
    BUILDER_FINGERPRINT_ATTR,
    add_demo_caption,
    cached_download,
    demo_source_fingerprint,
    launch_viewer,
    parse_demo_flags,
    require_module,
    scene_is_current,
)
from luxar.demos._cinematic_camera import pull_in
from luxar.demos._globe_common import (
    Clouds,
    blue_marble_basemap,
    build_earth,
)
from luxar.demos._globe_common import lonlat_to_xyz as _lonlat_to_xyz
from luxar.encoding import EncodingMode
from luxar.typing_utils.constants import MAX_SEGMENTS_PER_LINES_NODE
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

# The globe is a TEXTURED MESH. It was 8M points, and it had to be: a point
# cloud resolves the Blue Marble at about one sample per point, so sealing a
# shell that could show the coastlines needed millions. A UV sphere samples the
# texture per FRAGMENT, so the vertex count only has to make the sphere read as
# round — and the 8M-element budget goes back to the streamlines.
GLOBE_LON: Final = 512
GLOBE_LAT: Final = 256
# 16384x8192 across two tiles, matching the earthquakes globe. One KTX2 node at
# that size would exceed the viewer's 512 MiB admission budget once its payload
# and resident mip chain are counted; two nodes retain comfortable headroom.
# They also keep the missing-`toktx` bitmap fallback below WebP's hard
# 16383-pixel per-axis limit.
GLOBE_TEXTURE_WIDTH: Final = 16384
GLOBE_TILES: Final = 2
N_SEEDS: Final = 220_000  # streamlines
N_STEPS: Final = 52  # advection steps per streamline (-> N_STEPS + 1 vertices)
STEP_KM: Final = 14.0  # arc-length step -> ~730 km ribbons
FIELD_STRIDE: Final = 2  # subsample the 1/12 deg grid for advection
FLOW_LIFT: Final = 0.0015  # lift ribbons just clear of the textured globe
LINE_WIDTH: Final = 0.026
LINE_OPACITY: Final = 0.77
LINE_INTENSITY: Final = 1.0
SPEED_FULL_SCALE: Final = 1.0  # m/s mapped to the top of the colour ramp
MIN_SEED_SPEED: Final = 0.04  # skip near-still water when seeding
STALL_SPEED: Final = 0.02  # freeze a ribbon that runs out of current
LAT_LIMIT: Final = 79.9  # HYCOM's grid stops at +/-80

# Per-tile substitutive ladders bound opening-view residency while preserving
# whole ribbons. A compression of 2 keeps the coarse field visually continuous;
# larger values save more memory but make the opening globe visibly sparse.
LOD_LEVELS: Final = 3
LOD_COMPRESSION: Final = 2
LOD_STREAM_CHUNK: Final = 20_000
CURRENT_TILE_RIBBONS: Final = 13_750


FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]
KEEP_STALE = FLAGS["keep_stale"]

#: Identifies the builder and Luxar writer that wrote a scene, so stale output
#: is rebuilt instead of served forever (#1957, #2037).
FINGERPRINT: Final = demo_source_fingerprint(__file__)

CACHE_DIR: Final = Path.home() / ".cache" / "luxar" / DEMO_NAME

Arbol.max_depth = 5


# =============================================================================
# Pure helpers (unit-tested; no network / no IO)
# =============================================================================


def lonlat_to_xyz(lon: np.ndarray, lat: np.ndarray, relief: np.ndarray) -> np.ndarray:
    """This demo's globe radius bound into the shared mapping (see `_globe_common`)."""
    return _lonlat_to_xyz(lon, lat, relief, RADIUS)


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
    than letting it wander onto land. Wetness is tested ALONG each segment at the
    mask's resolution (not only at the endpoint), so a step cannot bridge a
    narrow land feature and resume in open water on the far side.

    Args:
        field: The surface-current field.
        seed_lon: ``(m,)`` seed longitudes, degrees. Must be in wet cells.
        seed_lat: ``(m,)`` seed latitudes, degrees. Must be in wet cells.
        n_steps: Number of steps; each path has ``n_steps + 1`` vertices.
        step_km: Arc length per step, kilometres.

    Returns:
        ``(lon, lat, speed)``, each ``(m, n_steps + 1)`` float32.

    Raises:
        ValueError: any seed sits on land. Wetness is only tested on the
            segments stepped forward from a seed, never on the seed itself, so a
            dry seed would be frozen at its start point and emit a whole ribbon
            lying on land — a silent, plausible-looking wrong render. Use
            :func:`seed_ocean_points`, which guarantees wet seeds.
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

    # A single STEP_KM arc is wider than a mask cell (esp. poleward, where
    # longitude cells narrow with cos(lat)), so an endpoint-only wetness check can
    # hop clean over a one-cell dry band or a narrow island and resume in open
    # water — drawing a current across land. Sample each segment at mask resolution
    # and freeze on the first dry hit.
    lat_cell_km = abs(field.dlat) * np.radians(1.0) * R_EARTH_KM
    lon_cell_km = (
        abs(field.dlon) * np.radians(1.0) * R_EARTH_KM * np.cos(np.radians(LAT_LIMIT))
    )
    min_cell_km = max(min(lat_cell_km, lon_cell_km), 1e-3)
    # +1 keeps the sample spacing strictly under one cell even when an RK4 stage
    # is evaluated just past the latitude clamp (a slightly wider lon step); the
    # floor keeps at least the endpoint checked for any step size.
    n_sub = max(int(np.ceil(step_km / min_cell_km)) + 1, 1)
    sub_fracs = np.arange(1, n_sub + 1, dtype=np.float64) / n_sub

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
        wet_along = np.ones(m, dtype=bool)
        for frac in sub_fracs:
            wet_along &= field.is_wet(
                lo + frac * (next_lon - lo), la + frac * (next_lat - la)
            )
        alive = alive & wet_along & (out_speed[:, k] > STALL_SPEED)
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


def lod_counts(
    n: int, levels: int = LOD_LEVELS, compression: int = LOD_COMPRESSION
) -> list:
    """Return de-duplicated coarse-to-fine element counts ending at ``n``."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if levels < 1:
        raise ValueError(f"levels must be >= 1, got {levels}")
    if compression < 2:
        raise ValueError(f"compression must be >= 2, got {compression}")
    return sorted({max(1, n // compression**level) for level in range(levels)})


def level_subset(n: int, count: int, seed: int) -> np.ndarray:
    """Return a deterministic sorted random subset of ``count`` elements."""
    if count >= n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=count, replace=False))


def level_seed(layer: int, tile: int, level: int) -> int:
    """Return a collision-free seed for one layer/tile/level subset."""
    return 1_000_003 * layer + 1_009 * tile + level


def check_tile_budget(geometry: str, largest_tile: int, cap: int) -> None:
    """Reject a tile whose finest level would exceed the element-texture cap."""
    if largest_tile > cap:
        raise ValueError(
            f"largest {geometry} tile holds {largest_tile:,} elements, over the "
            f"{cap:,} a single node can render on a 4096-class GPU — the tail "
            f"would be silently clamped. Lower the tile size constant."
        )


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


def globe_camera(lon: float, lat: float, *, distance: float = 2.586) -> CameraConfig:
    """Opening pose looking straight down at ``(lon, lat)`` on the globe.

    The target is the ORIGIN — the centre of the Earth — not a point under the
    surface. The scene is a sphere centred on the origin, so that is the only
    target that makes the opening framing centred and makes orbiting pivot
    about the planet's axis. Aiming at ``normal * RADIUS * 0.9`` (just below the
    surface) instead put the pivot on the near face, so the first drag swung the
    globe about a surface point and threw it off-centre.

    ``distance=2.586`` reproduces the shipped ~1.62 R opening eye after the
    42-degree cinematic ``framing_scale`` of approximately 0.6264.
    """
    la, lo = np.radians(lat), np.radians(lon)
    normal = np.array(
        [np.cos(la) * np.cos(lo), np.sin(la), -np.cos(la) * np.sin(lo)],
        dtype=np.float64,
    )
    target = (0.0, 0.0, 0.0)
    return CameraConfig(
        position=pull_in(
            tuple((normal * RADIUS * distance).tolist()),
            target,
            from_fov_deg=42.0,
        ),
        target=target,
        up=(0.0, 1.0, 0.0),
    )


# =============================================================================
# Data loading
# =============================================================================


def load_hycom_surface(path: Path, stride: int = FIELD_STRIDE) -> LonLatField:
    """Read HYCOM surface u/v from a netCDF3 file into a :class:`LonLatField`.

    ``accept=netcdf`` (classic netCDF3) is requested deliberately so
    ``scipy.io.netcdf_file`` can read it — no netCDF4/h5py/xarray needed. Values
    are ``int16 * scale_factor + add_offset`` with land at ``_FillValue``.
    """
    netcdf_file = require_module("scipy.io").netcdf_file
    data = netcdf_file(str(path), "r", mmap=False)
    var_u = data.variables["water_u"]
    var_v = data.variables["water_v"]
    fill = getattr(var_u, "_FillValue", None)

    def unpack(var: object) -> np.ndarray:
        """CF-unpack ``raw * scale_factor + add_offset``, land -> NaN."""
        scale = float(getattr(var, "scale_factor", 1.0))
        offset = float(getattr(var, "add_offset", 0.0))
        raw = var.data[0, 0, ::stride, ::stride]  # type: ignore[attr-defined]
        return np.where(raw == fill, np.nan, raw * scale + offset).astype(np.float32)

    u = unpack(var_u)
    v = unpack(var_v)
    lat = data.variables["lat"].data[::stride].astype(np.float64)
    lon = data.variables["lon"].data[::stride].astype(np.float64)

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


def tile_coverage(counts: list, n_tiles: int) -> list:
    """Return LOD coverage thresholds for a tiled or whole-object anchor."""
    if n_tiles < 2:
        return coverage_fractions(counts)
    return partitioned_coverage_fractions(counts)


def write_current_parts(
    scene,
    vertices: np.ndarray,
    colors: np.ndarray,
    n_paths: int,
    n_vertices: int,
    tile_size: int = CURRENT_TILE_RIBBONS,
) -> tuple:
    """Write ribbons as a partition of per-tile substitutive LOD ladders."""
    blocks = vertices.reshape(n_paths, n_vertices, 3)
    centroids = blocks.mean(axis=1)
    tree = spatial_bsp_tree(centroids, tile_size, rule="median")
    parts = bsp_leaf_parts(tree)
    check_tile_budget(
        "lines",
        max(int(part.size) for part in parts) * (n_vertices - 1),
        MAX_SEGMENTS_PER_LINES_NODE,
    )
    wrapper = scene.add_partition_group(
        "currents",
        display_type="lines",
        max_elements=tile_size * (n_vertices - 1),
        blending_mode="luminous",
        opacity=LINE_OPACITY,
        intensity=LINE_INTENSITY,
        layer=True,
        position_bounds=position_bounds_from_array(vertices),
        bsp_tree=tree.to_serializable(),
    )
    coarsest = 0
    for tile, idx in enumerate(parts):
        counts = lod_counts(int(idx.size))
        lod = wrapper.add_lod_group(f"part_{tile}", selector="screen-area")
        coverage = tile_coverage(counts, len(parts))
        coarsest += counts[0] * (n_vertices - 1)
        for level, (count, cover) in enumerate(zip(counts, coverage)):
            sub = idx[level_subset(int(idx.size), count, level_seed(1, tile, level))]
            keep = (sub[:, None] * n_vertices + np.arange(n_vertices)[None, :]).ravel()
            lod.add_lines(
                f"child_{level}",
                vertices=vertices[keep],
                widths=LINE_WIDTH * (idx.size / count),
                colors=colors[keep],
                indices=polyline_segment_indices(int(sub.size), n_vertices),
                line_type="indexed",
                coverage_fraction=float(cover),
                additive_lod=level_additive_lod(
                    dict(counts=f"stream:{LOD_STREAM_CHUNK}", method="random", seed=0),
                    level_n=int(sub.size) * n_vertices,
                    compression_factor=LOD_COMPRESSION,
                    is_coarsest=(level == 0),
                ),
            )
    return len(parts), coarsest


def build_scene(hycom_path: Path, marble_path: Path, output_path: Path) -> Path:
    """Build the globe + current-streamline scene and write it to ``output_path``."""
    image_module = require_module("PIL.Image")

    with asection("Loading surface currents"):
        field = load_hycom_surface(hycom_path)

    with asection(f"Building globe mesh ({GLOBE_LON}x{GLOBE_LAT} quads)"):
        # The SHARED hi-res basemap, not `marble_path`'s 2048x1024 image. That
        # image was sized for the point cloud, which resolved roughly one texel per
        # point and so could not use more; a per-fragment mesh makes the basemap the
        # only limit on how sharp a coastline looks. `marble_path` is still
        # downloaded by `download_sources` (it is part of the demo's declared
        # provenance) and is used as the FALLBACK when the larger master is
        # unavailable.
        try:
            basemap, basemap_w, basemap_h = blue_marble_basemap(
                width=GLOBE_TEXTURE_WIDTH
            )
            tiles = GLOBE_TILES
        except Exception as error:
            aprint(f"⚠️  Hi-res basemap unavailable ({error}); using the 2048 image")
            basemap = np.asarray(image_module.open(marble_path).convert("RGB"))
            basemap_h, basemap_w = basemap.shape[:2]
            tiles = 1
        aprint(f"globe basemap: {basemap_w}x{basemap_h} across {tiles} tile(s)")

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
        aprint(
            f"{n_paths:,} ribbons, {n_paths * (n_vertices - 1):,} segments, "
            f"{total:,} vertices"
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
                citation=DEMO_META["citation"],
                dimensions=dims,
                viewer_config=ViewerConfig(
                    # Thin lines are the one geometry that genuinely rewards the pixels:
                    # at CSS resolution this scene's structure reads as mush. Cheap
                    # here, because line scenes are not fill-bound — see
                    # ViewerConfig.allow_high_dpr.
                    allow_high_dpr=True,
                    cinematic_mode=True,
                    # No tone mapping at all (#1459): the ramp is an encoding of
                    # speed, and every colour here — Blue Marble texture and
                    # blue->white LUT alike — already sits inside [0, 1], so a
                    # passthrough is exact. ACES shifts hues; Neutral subtracts
                    # its offset and compresses from peak 0.76 up, dulling the
                    # white the fastest currents are supposed to reach.
                    tone_mapping="None",
                    camera=globe_camera(-84.0, 25.0),
                    # Turntable about the SOUTH-NORTH axis. Orbit auto-rotation spins
                    # azimuthally about the controls' up vector, so pinning up to
                    # +y is what makes this a planetary rotation rather than a
                    # tumble: `lonlat_to_xyz` puts the north pole on +y.
                    auto_rotate=True,
                    auto_rotate_speed=0.35,
                ),
            )
            scene.attrs["title"] = "Ocean Currents of Earth — HYCOM surface circulation"
            scene.attrs[BUILDER_FINGERPRINT_ATTR] = FINGERPRINT
            # ONE call: basemap tiles + the thin cloud deck. Shared with the
            # other three Earth demos so they cannot drift apart.
            build_earth(
                scene,
                "earth",
                radius=RADIUS,
                n_lon=GLOBE_LON,
                n_lat=GLOBE_LAT,
                texture_width=GLOBE_TEXTURE_WIDTH,
                tiles=GLOBE_TILES,
                basemap=basemap,
                # UNLIT, unlike the earthquakes globe, and the difference is the
                # point of the arm existing. This basemap is a REFERENCE for the
                # current speeds drawn over it: a view-anchored diffuse key would
                # darken the limb as the camera moved, so the same ocean would read
                # as a different colour depending on where you were looking from.
                # tone_mapping="None" is pinned for the same reason (see
                # tests/test_demos_tone_mapping_policy.py) — the two only work as a
                # pair.
                shading="none",
                # THIN, because the ribbons are the data and they are drawn at
                # FLOW_LIFT (0.0015) while the shell sits at 0.012 — in front of
                # them. At this strength it reads as atmosphere over the map rather
                # than as an occluder of the currents.
                # A STRONGER deck than the other two data globes, because this
                # scene's foreground is by far the brightest: the current ribbons
                # are `luminous`, so overlapping streamlines ACCUMULATE and a
                # boundary current saturates. At the shared 0.45/1.6 the clouds
                # were there and unreadable. Cloud strength is not portable
                # between demos for the same reason exposure is not — it is
                # relative to whatever is drawn over it.
                clouds=Clouds(strength=0.65, gamma=1.9, intensity=2.4),
                # `opaque` is the mesh default and the right one here: the globe is
                # the BACKDROP. It is the only mode that leaves the viewer's sorted
                # transparent set and the only one that unconditionally depth-writes,
                # so it is the only one that reliably composites *under* the
                # translucent ribbons in front of it.
                blending_mode="opaque",
                opacity=1.0,
                layer=True,
            )
            n_current_tiles, current_coarsest = write_current_parts(
                scene, vertices, colors, n_paths, n_vertices
            )
            aprint(
                f"currents: {n_current_tiles} tiles x {LOD_LEVELS} levels; "
                f"{current_coarsest:,} segments resident at the coarsest levels, "
                f"down from {n_paths * (n_vertices - 1):,}"
            )
            scene.add_text(
                "Ocean Currents of Earth",
                position=(0.02, 0.02),
                font_size=0.045,
                anchor="top-left",
                color="rgba(255,255,255,0.75)",
                blend_mode="difference",
            )
            add_demo_caption(
                scene,
                "HYCOM GLBy0.08 surface velocities • NASA Blue Marble topography",
                DEMO_META.get("citation"),
            )
        aprint(f"Scene saved: {output_path}")
    return output_path


def load_or_build_scene(output_path: Path) -> Path:
    """Return the built scene, regenerating it on a fresh system."""
    if scene_is_current(
        output_path, FINGERPRINT, recompute=RECOMPUTE, keep_stale=KEEP_STALE
    ):
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
