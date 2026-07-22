#!/usr/bin/env python3
"""Demo: Rivers of Earth — a topographic globe + the planet's river networks.

A rotatable 3D globe built from two of Luxar's geometry types at once:

  * **Terrain (Points)** — the Earth's surface sampled on a **Fibonacci
    spiral-sphere** (uniform, no pole clustering), each point displaced radially
    by its **ETOPO 2022** elevation and colored by a hypsometric palette (deep
    abyssal navy -> ocean blue -> coastal cyan -> green lowland -> tan -> snow).
    Rendered near-transparent (opacity 0.05) as a subtle backdrop, with
    additive-LOD for a fast progressive first paint.
  * **Rivers (Lines)** — every HydroRIVERS reach (Strahler order >= 3), kept as
    **connected polylines** (so the line material renders seamless joints),
    draped just above the terrain and colored teal->white by Strahler order so
    minor tributaries read teal and major rivers white. Additive-LOD streams the
    biggest rivers first.

This exercises two of Luxar's three geometry types (Points + Lines) at global
scale with level-of-detail, in real geographic 3D.

================================================================================
SELF-CONTAINED / REGENERATING (no LFS asset)
================================================================================
The built scene is large (hundreds of MB), so instead of committing it we
regenerate + cache it. This demo ships ONLY code:

  * First run downloads ~1 GB of *source* data (both public, direct download,
    no API key): HydroRIVERS_v10 (~544 MB) + ETOPO 2022 60-arc-sec (~466 MB),
    parses + builds the globe scene (to the standard demos-output dir), and
    caches the source + parsed polylines under
    ``~/.cache/luxar/global_rivers_earth/``.
  * Subsequent runs load the built scene instantly; ``--recompute`` forces a
    rebuild (source/polylines stay cached, so it never re-fetches the ~1 GB).

Requires: ``pyshp`` (shapefile reader) and ``tifffile`` (GeoTIFF reader).

DATA SOURCES & CITATIONS
------------------------
HydroRIVERS v10 (HydroSHEDS) — Lehner, B., Grill G. (2013), Hydrological
    Processes 27(15). CC-BY-4.0. https://www.hydrosheds.org/products/hydrorivers
ETOPO 2022 Global Relief Model — NOAA NCEI (2022), doi:10.25921/fd45-gt74.
    Public domain. https://www.ncei.noaa.gov/products/etopo-global-relief-model

USAGE
-----
    python demo_global_rivers_earth.py [--recompute] [--no-serve] [--serve-only]
"""

DEMO_META = {
    "key": "global_rivers_earth",
    "title": "Rivers of Earth",
    "description": "A topographic ETOPO globe (Points) plus every HydroRIVERS reach (Lines) in geographic 3D.",
    "category": "geoscience",
    "geometry": "points+lines",
    "requirements": {
        "download_mb": 1000,
        "compute": "heavy",
        "gpu": "none",
        "local_data": None,
    },
    "caches": ["global_rivers_earth"],
    "outputs": ["global_rivers_earth"],
}

from pathlib import Path

import numpy as np
from arbol import Arbol, aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import ViewerConfig
from luxar.encoding import EncodingMode
from luxar.utils.demos import launch_viewer, parse_demo_flags
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

DEMO_NAME = "global_rivers_earth"
CACHE_DIR = Path.home() / ".cache" / "luxar" / DEMO_NAME

HYDRORIVERS_URL = "https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_shp.zip"
ETOPO_URL = (
    "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/60s/"
    "60s_surface_elev_gtif/ETOPO_2022_v1_60s_N90W180_surface.tif"
)

N_GLOBE = 8_000_000  # Fibonacci-sphere terrain points
MIN_ORDER = 3  # keep HydroRIVERS reaches with Strahler order >= this
DECIMATE_DEG = 0.06  # drop river vertices closer than this (~2-3x line width)
RADIUS = 100.0  # globe radius (scene units)
EXAGG = 45.0  # vertical exaggeration of elevation relief
POINT_RADII = 0.09  # terrain point size (8M points form a dense shell)
EARTH_OPACITY = 0.05  # near-transparent backdrop; lets the rivers dominate
RIVER_LIFT = 0.004  # lift rivers barely above the terrain surface
RIVER_WIDTH = 0.015
RIVER_INTENSITY = 1.6
R_EARTH = 6_371_000.0  # metres, for elevation -> relief fraction

FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]

Arbol.max_depth = 5


# =============================================================================
# Pure helpers (unit-tested; no network / no IO)
# =============================================================================


def fibonacci_sphere(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (lon, lat) in degrees for ``n`` points on a Fibonacci spiral sphere."""
    i = np.arange(n)
    golden = (1.0 + 5.0**0.5) / 2.0
    y = 1.0 - 2.0 * (i + 0.5) / n
    r_xy = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = 2.0 * np.pi * i / golden
    lat = np.degrees(np.arcsin(np.clip(y, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(r_xy * np.sin(theta), r_xy * np.cos(theta)))
    return lon.astype(np.float64), lat.astype(np.float64)


def lonlat_to_xyz(lon: np.ndarray, lat: np.ndarray, relief: np.ndarray) -> np.ndarray:
    """Map geographic (lon, lat) degrees + radial ``relief`` fraction to sphere xyz.

    ``y`` is the north pole axis; longitude increases eastward. The ``-z`` makes
    the mapping right-handed (East x North = outward) so the globe is NOT
    mirror-imaged when viewed from outside.
    """
    la, lo = np.radians(lat), np.radians(lon)
    rr = RADIUS * (1.0 + relief)
    x = rr * np.cos(la) * np.cos(lo)
    y = rr * np.sin(la)
    z = -rr * np.cos(la) * np.sin(lo)
    return np.column_stack([x, y, z]).astype(np.float32)


def hypsometric_scalars(elev: np.ndarray) -> np.ndarray:
    """Map elevation (metres, signed) to [0, 1] with sea level at the palette break.

    Ocean depths fill [0, 0.22] (blue) and land fills [0.221, 1] (green->snow), so
    the coastline lands on the ocean/land color boundary regardless of data range.
    """
    e = elev.astype(np.float32)
    s = np.empty_like(e)
    lo = float(e.min())
    hi = max(float(e.max()), 1.0)
    neg = e < 0
    s[neg] = 0.22 * (e[neg] - lo) / (0.0 - lo + 1e-6)
    s[~neg] = 0.221 + 0.779 * e[~neg] / hi
    return s.astype(np.float32)


def _lut_from(stops: list[tuple[float, tuple[int, int, int]]]) -> np.ndarray:
    """Build a (256, 3) uint8 colormap LUT by linear interpolation over ``stops``."""
    ts = np.array([s[0] for s in stops])
    cols = np.array([s[1] for s in stops], dtype=float)
    x = np.linspace(0.0, 1.0, 256)
    lut = np.stack([np.interp(x, ts, cols[:, k]) for k in range(3)], axis=1)
    return np.clip(lut, 0, 255).astype(np.uint8)


def decimate_polyline(pts: np.ndarray, min_len_deg: float) -> np.ndarray:
    """Drop intermediate vertices closer than ``min_len_deg`` (cos-lat weighted).

    Removes sub-line-width wiggles from a (k, 2) lon/lat polyline while keeping the
    first and last vertex, so no rendered segment is much shorter than the line.
    """
    if len(pts) < 3:
        return pts
    keep = [0]
    last = pts[0]
    for j in range(1, len(pts) - 1):
        clat = np.cos(np.radians(pts[j, 1]))
        d = np.hypot((pts[j, 0] - last[0]) * clat, pts[j, 1] - last[1])
        if d >= min_len_deg:
            keep.append(j)
            last = pts[j]
    keep.append(len(pts) - 1)
    return pts[keep]


# Hypsometric terrain palette and teal->white river palette (Strahler order).
EARTH_LUT = _lut_from(
    [
        (0.00, (2, 5, 22)),
        (0.08, (8, 20, 70)),
        (0.18, (18, 80, 140)),
        (0.215, (45, 150, 185)),
        (0.221, (28, 105, 55)),
        (0.30, (70, 135, 62)),
        (0.45, (120, 145, 72)),
        (0.60, (160, 135, 85)),
        (0.75, (140, 110, 82)),
        (0.88, (185, 175, 165)),
        (1.00, (255, 255, 255)),
    ]
)
RIVER_LUT = _lut_from(
    [
        (0.00, (30, 110, 145)),
        (0.50, (80, 200, 230)),
        (1.00, (245, 255, 255)),
    ]
)


# =============================================================================
# Data download + parse (network / IO; not unit-tested)
# =============================================================================


def _download_sources() -> tuple[Path, Path]:
    """Download HydroRIVERS + ETOPO into the cache (resumable). Returns their paths."""
    import zipfile

    from luxar.utils.download import robust_download

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    etopo = CACHE_DIR / "etopo_2022_60s.tif"
    rivers_zip = CACHE_DIR / "HydroRIVERS_v10_shp.zip"
    rivers_shp = CACHE_DIR / "HydroRIVERS_v10_shp" / "HydroRIVERS_v10.shp"

    with asection("Downloading source data (~1 GB, one time)"):
        robust_download(ETOPO_URL, etopo, max_retries=5, timeout=900)
        robust_download(HYDRORIVERS_URL, rivers_zip, max_retries=5, timeout=900)
        if not rivers_shp.exists():
            with asection("Extracting HydroRIVERS"):
                with zipfile.ZipFile(rivers_zip) as zf:
                    zf.extractall(CACHE_DIR)
    return etopo, rivers_shp


def _parse_river_polylines(shp_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse + decimate HydroRIVERS into connected polylines (cached as .npz).

    Returns ``(pts, offsets, order)``: ``pts`` (Nv, 2) lon/lat vertices of all
    reaches concatenated; ``offsets`` (Nr+1,) so reach r is
    ``pts[offsets[r]:offsets[r+1]]``; ``order`` (Nr,) per-reach Strahler order.
    Only order >= 2 is cached (the build filters higher). Keeping reaches as
    connected polylines (not independent segments) lets the line material render
    seamless joints instead of overlapping end-caps.
    """
    import shapefile  # pyshp

    cache = CACHE_DIR / "river_polylines.npz"
    if cache.exists():
        with asection("Loading cached river polylines"):
            d = np.load(cache)
            return d["pts"], d["offsets"], d["order"]

    with asection("Parsing + decimating HydroRIVERS reaches -> polylines"):
        r = shapefile.Reader(str(shp_path))
        flds = [f[0] for f in r.fields[1:]]
        i_ord = flds.index("ORD_STRA")
        pts_list: list[np.ndarray] = []
        offsets = [0]
        orders: list[int] = []
        off = 0
        for sr in r.iterShapeRecords():
            o = int(sr.record[i_ord])
            if o < 2:
                continue
            allpts = np.asarray(sr.shape.points, dtype=np.float32)
            parts = list(sr.shape.parts) + [len(allpts)]
            for a, b in zip(parts[:-1], parts[1:]):
                if b - a < 2:
                    continue
                p = decimate_polyline(allpts[a:b], DECIMATE_DEG)
                if len(p) < 2:
                    continue
                pts_list.append(p)
                off += len(p)
                offsets.append(off)
                orders.append(o)
        pts = np.concatenate(pts_list)
        offs = np.array(offsets, dtype=np.int64)
        order = np.array(orders, dtype=np.uint8)
        np.savez(cache, pts=pts, offsets=offs, order=order)
        aprint(f"Parsed {len(order):,} river polylines ({len(pts):,} vertices)")
        return pts, offs, order


# =============================================================================
# Scene build
# =============================================================================


def _sample_elevation(
    etopo: np.ndarray, lon: np.ndarray, lat: np.ndarray
) -> np.ndarray:
    """Sample the ETOPO grid (row 0 = 90N, col 0 = 180W) at geographic lon/lat."""
    h, w = etopo.shape
    col = np.mod(((lon + 180.0) / 360.0 * w).astype(np.int64), w)
    row = np.clip(((90.0 - lat) / 180.0 * h).astype(np.int64), 0, h - 1)
    return etopo[row, col].astype(np.float32)


def build_scene(etopo_path: Path, shp_path: Path, output_path: Path) -> Path:
    """Build the topographic globe + rivers scene and write it to ``output_path``."""
    import tifffile

    etopo = tifffile.imread(etopo_path)

    with asection(f"Building globe ({N_GLOBE:,} points)"):
        glon, glat = fibonacci_sphere(N_GLOBE)
        gelev = _sample_elevation(etopo, glon, glat)
        gpos = lonlat_to_xyz(glon, glat, gelev / R_EARTH * EXAGG)
        gscal = hypsometric_scalars(gelev)
        # pre-bake terrain colors from the LUT (identical across additive-LOD
        # levels; avoids serialising a LUT array on the additive-LOD path)
        gcolors = (
            EARTH_LUT[np.clip((gscal * 255).astype(np.int64), 0, 255)].astype(
                np.float32
            )
            / 255.0
        )

    with asection(f"Building rivers (order >= {MIN_ORDER})"):
        pts, offsets, order = _parse_river_polylines(shp_path)
        lengths = offsets[1:] - offsets[:-1]
        keep = order >= MIN_ORDER
        # gather the kept polylines' vertices (vectorized) and rebuild offsets
        vtx_keep = np.repeat(keep, lengths)
        rlon = pts[vtx_keep, 0]
        rlat = pts[vtx_keep, 1]
        kept_lengths = lengths[keep]
        new_off = np.concatenate([[0], np.cumsum(kept_lengths)])
        rrelief = np.maximum(_sample_elevation(etopo, rlon, rlat), 0.0)
        rpos = lonlat_to_xyz(rlon, rlat, rrelief / R_EARTH * EXAGG + RIVER_LIFT)
        # per-vertex colors by Strahler order (3..10 -> teal..white)
        overt = np.repeat(order[keep].astype(np.float32), kept_lengths)
        onorm = np.clip((overt - 3.0) / 7.0, 0.0, 1.0)
        rcolors = (
            RIVER_LUT[np.clip((onorm * 255).astype(np.int64), 0, 255)].astype(
                np.float32
            )
            / 255.0
        )
        # connected indices: consecutive pairs WITHIN each polyline (shared
        # vertices at joints -> the line material renders seamless joins, not
        # overlapping end-caps). Boundary pairs (crossing reaches) are excluded.
        i0 = np.arange(len(rlon) - 1)
        valid = i0[~np.isin(i0 + 1, new_off[1:-1])]
        rindices = np.column_stack([valid, valid + 1]).ravel().astype(np.uint32)
        aprint(f"{int(keep.sum()):,} river polylines, {len(valid):,} segments")

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
                dimensions=dims, viewer_config=ViewerConfig(tone_mapping="Neutral")
            )
            scene.attrs["title"] = "Rivers of Earth — global topography + HydroRIVERS"
            scene.add_points(
                "terrain",
                positions=gpos,
                radii=POINT_RADII,
                colors=gcolors,
                blending_mode="normal",
                opacity=EARTH_OPACITY,
                layer=True,
                additive_lod=dict(method="spatial-uniform", n_lods=5),
            )
            # Connected polylines (indexed) so the material renders seamless
            # joints. No additive-LOD here: LOD-ing connected lines requires an
            # O(N) union-find over ~2M polylines (minutes to build) for little
            # gain — the rivers stream via the spatial-chunk index instead.
            scene.add_lines(
                "rivers",
                vertices=rpos,
                widths=RIVER_WIDTH,
                colors=rcolors,
                indices=rindices,
                line_type="indexed",
                blending_mode="additive",
                opacity=0.95,
                intensity=RIVER_INTENSITY,
                layer=True,
            )
            scene.add_text(
                "Rivers of Earth",
                position=(0.02, 0.02),
                font_size=0.045,
                anchor="top-left",
                color="rgba(255,255,255,0.75)",
                blend_mode="difference",
            )
            scene.add_text(
                "ETOPO 2022 topography • HydroRIVERS (HydroSHEDS) river networks",
                position=(0.98, 0.97),
                font_size=0.015,
                anchor="bottom-right",
                color="rgba(200,200,220,0.5)",
            )
        aprint(f"Scene saved: {output_path}")
    return output_path


def load_or_build_scene(output_path: Path) -> Path:
    """Return the built scene path, regenerating on a fresh system.

    The scene is written to ``output_path`` (the standard demos-output location);
    the expensive source downloads + parsed polylines are cached under
    ``CACHE_DIR`` so a rebuild / ``--recompute`` never re-fetches the ~1 GB.
    """
    if output_path.exists() and not RECOMPUTE:
        aprint(f"Using existing scene: {output_path}")
        return output_path

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    etopo_path, shp_path = _download_sources()
    return build_scene(etopo_path, shp_path, output_path)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    aprint("=" * 70)
    aprint("Demo: Rivers of Earth — global topography + river networks")
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
        aprint("Data credit: HydroRIVERS (HydroSHEDS, CC-BY-4.0) + ETOPO 2022 (NOAA)")
        launch_viewer(scene_path)


if __name__ == "__main__":
    main()
