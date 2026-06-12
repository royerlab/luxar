#!/usr/bin/env python3
"""Self-Contained Demo: Cosmicflows-4 Laniakea Flow Field

This demo recreates the viral Cosmicflows-4 / Laniakea visualization by
Simone Conradi and Manlio De Domenico: galaxies in the local universe plus
colored streamlines tracing the reconstructed cosmic velocity field.

It downloads and caches the exact public inputs used by the reference
pipeline at https://github.com/manlius/laniakea:
- CF4 galaxy distance table from EDD
- CF4 64^3 velocity field
- CF4 128^3 basin-of-attraction grid

The full preset reproduces the reported counts:
- 55,486 galaxies inside the +/-500 Mpc CF4 cube
- 29,555 streamlines after RK4 integration startup validation

Usage:
    python demo_cosmicflows_laniakea.py [--preset full|preview]
    python demo_cosmicflows_laniakea.py --preset preview --no-serve

Controls:
    - Rotate / pan / zoom in the Luxar viewer
    - Toggle individual basin streamline layers in the Layers panel
    - Ctrl+C to stop and cleanup
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import CameraConfig, UIConfig, ViewerConfig
from luxar.demos import launch_viewer
from luxar.utils.paths import get_demos_output_dir

LANIAKEA_RAW_BASE: Final = "https://raw.githubusercontent.com/manlius/laniakea/main"
DATA_FILES: Final[dict[str, str]] = {
    "galaxies": (
        f"{LANIAKEA_RAW_BASE}/1_CF4_galaxies_table/EDDtable22Nov2025140156.txt"
    ),
    "velocity": (f"{LANIAKEA_RAW_BASE}/2_CF4_streamlines/CF4_new_64-z008_velocity.npy"),
    "basins": (f"{LANIAKEA_RAW_BASE}/2_CF4_streamlines/CF4_new_128-z008_BoA.npy"),
}

# Color palette from the reference pygfx configuration.
BASIN_COLORS: Final[dict[int, str]] = {
    0: "#ffffff",
    1: "#e6194b",  # Laniakea in the reference article
    2: "#3cb44b",
    3: "#ffe119",
    4: "#4363d8",
    5: "#f58231",
    6: "#fabed4",
    7: "#f032e6",
    8: "#42d4f4",
    9: "#dcbeff",
}

GALAXY_COLUMNS: Final[list[str]] = [
    "pgc",
    "group_pgc",
    "T17",
    "Vcmb",
    "DM",
    "eDM",
    "DMsnIa",
    "eDMsn1",
    "DMtf",
    "eDMtf",
    "DMfp",
    "eDMfp",
    "DMsbf",
    "eDMsbf",
    "DMsnII",
    "eDMsn2",
    "DMtrgb",
    "eDMt",
    "DMcep",
    "eDMcep",
    "DMmas",
    "eDMmas",
    "RA",
    "DE",
    "glon",
    "glat",
    "sgl",
    "sgb",
]


@dataclass(frozen=True)
class StreamlinePreset:
    """Numerical settings for seed selection and RK4 integration."""

    max_seeds_per_basin: int
    min_seeds_per_basin: int
    h_step: float
    nsteps: int
    line_width: float
    point_radius: float


PRESETS: Final[dict[str, StreamlinePreset]] = {
    # Matches the dense settings from manlius/laniakea and yields 29,555
    # streamlines that successfully take at least one integration step.
    "full": StreamlinePreset(
        max_seeds_per_basin=3000,
        min_seeds_per_basin=500,
        h_step=0.3,
        nsteps=400,
        line_width=0.55,
        point_radius=1.15,
    ),
    # Faster authoring preset for quick iteration.
    "preview": StreamlinePreset(
        max_seeds_per_basin=600,
        min_seeds_per_basin=100,
        h_step=0.35,
        nsteps=260,
        line_width=0.75,
        point_radius=1.45,
    ),
}


@dataclass(frozen=True)
class GalaxyData:
    """Galaxy positions and basin labels."""

    positions: np.ndarray
    basin_ids: np.ndarray
    radii: np.ndarray
    colors: np.ndarray


@dataclass(frozen=True)
class BasinLineData:
    """Indexed line geometry for one basin."""

    basin_id: int
    vertices: np.ndarray
    segments: np.ndarray
    streamline_count: int


# ---------------------------------------------------------------------------
# Downloading and input parsing
# ---------------------------------------------------------------------------


def get_cache_dir() -> Path:
    """Return the local data cache for this demo."""
    return Path.home() / ".cache" / "luxar" / "laniakea"


def download_file(url: str, destination: Path) -> None:
    """Download a URL atomically if it is not already cached."""
    if destination.exists() and destination.stat().st_size > 0:
        aprint(f"✓ Cached: {destination.name}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    aprint(f"Downloading {destination.name}...")
    with urllib.request.urlopen(url, timeout=120) as response:
        tmp_path.write_bytes(response.read())
    tmp_path.replace(destination)
    aprint(f"✓ Downloaded: {destination}")


def ensure_input_data(cache_dir: Path) -> tuple[Path, Path, Path]:
    """Ensure all public inputs are present in the local cache."""
    with asection("Preparing Cosmicflows-4 input data"):
        galaxy_path = cache_dir / "EDDtable22Nov2025140156.txt"
        velocity_path = cache_dir / "CF4_new_64-z008_velocity.npy"
        basins_path = cache_dir / "CF4_new_128-z008_BoA.npy"

        download_file(DATA_FILES["galaxies"], galaxy_path)
        download_file(DATA_FILES["velocity"], velocity_path)
        download_file(DATA_FILES["basins"], basins_path)

    return galaxy_path, velocity_path, basins_path


def hex_to_rgb(hex_color: str, intensity: float = 1.0) -> np.ndarray:
    """Convert '#rrggbb' to an RGB float32 vector, with optional HDR gain."""
    text = hex_color.lstrip("#")
    rgb = np.array(
        [int(text[i : i + 2], 16) / 255.0 for i in (0, 2, 4)],
        dtype=np.float32,
    )
    return rgb * np.float32(intensity)


def load_galaxies(galaxy_path: Path, basins: np.ndarray) -> GalaxyData:
    """Load CF4 galaxies, convert to SG Cartesian coordinates, assign basins."""
    with asection("Loading CF4 galaxies"):
        # The reference script starts data at file row 6 (zero-based), which
        # skips the first catalog row and gives the published 55,486 in-cube
        # count used by the visualization post.
        raw = np.genfromtxt(
            galaxy_path,
            delimiter=",",
            names=GALAXY_COLUMNS,
            skip_header=6,
            dtype=None,
            encoding=None,
            comments=None,
            filling_values=np.nan,
        )

        distance_mpc = 10.0 ** ((raw["DM"].astype(np.float64) - 25.0) / 5.0)
        sgl = np.radians(90.0 - raw["sgl"].astype(np.float64))
        sgb = np.radians(raw["sgb"].astype(np.float64))

        sgx = distance_mpc * np.cos(sgb) * np.cos(sgl)
        sgy = distance_mpc * np.cos(sgb) * np.sin(sgl)
        sgz = distance_mpc * np.sin(sgb)
        positions_all = np.column_stack([sgx, sgy, sgz]).astype(np.float32)

        n = basins.shape[0]
        delta = 1000.0 / n
        indices = np.floor((positions_all + 500.0) / delta).astype(np.int32)
        inside = np.all((indices >= 0) & (indices < n), axis=1)
        positions = positions_all[inside]
        indices_inside = indices[inside]

        basin_ids = np.zeros(len(positions), dtype=np.int16)
        ix = indices_inside[:, 0]
        iy = indices_inside[:, 1]
        iz = indices_inside[:, 2]
        basin_ids[:] = basins[iz, iy, ix].astype(np.int16)

        palette = np.zeros((10, 3), dtype=np.float32)
        for basin_id, color in BASIN_COLORS.items():
            palette[basin_id] = hex_to_rgb(color, intensity=0.95)

        # Keep undefined / exterior watershed points visible but subdued.
        palette[0] = np.array([0.65, 0.72, 0.82], dtype=np.float32)
        colors = palette[np.clip(basin_ids, 0, 9)]
        colors = np.clip(colors * 0.72 + 0.18, 0.0, None).astype(np.float32)

        radial = np.linalg.norm(positions, axis=1)
        radii = (1.35 - 0.75 * np.clip(radial / 500.0, 0.0, 1.0)).astype(np.float32)

        aprint(f"✓ Loaded {len(raw):,} CF4 rows")
        aprint(f"✓ Kept {len(positions):,} galaxies inside the ±500 Mpc cube")
        unique, counts = np.unique(basin_ids, return_counts=True)
        basin_summary = ", ".join(
            f"{int(b)}:{int(c):,}" for b, c in zip(unique, counts)
        )
        aprint(f"Basin counts: {basin_summary}")

        return GalaxyData(positions, basin_ids, radii, colors)


# ---------------------------------------------------------------------------
# Streamline generation (vectorized port of manlius/laniakea RK4 code)
# ---------------------------------------------------------------------------


def index128_to_sg(
    ix128: np.ndarray, iy128: np.ndarray, iz128: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert 128^3 basin-grid index coordinates to SG Mpc coordinates."""
    delta = 1000.0 / 128.0
    sgx = -500.0 + (ix128.astype(np.float32) + 0.5) * delta
    sgy = -500.0 + (iy128.astype(np.float32) + 0.5) * delta
    sgz = -500.0 + (iz128.astype(np.float32) + 0.5) * delta
    return sgx, sgy, sgz


def sg_to_index64(sgx: np.ndarray, sgy: np.ndarray, sgz: np.ndarray) -> np.ndarray:
    """Convert SG Mpc coordinates to 64^3 velocity-grid index coordinates."""
    delta = 1000.0 / 64.0
    ix64 = (sgx + 500.0) / delta - 0.5
    iy64 = (sgy + 500.0) / delta - 0.5
    iz64 = (sgz + 500.0) / delta - 0.5
    return np.column_stack([ix64, iy64, iz64]).astype(np.float32)


def index64_to_sg_output(points: np.ndarray) -> np.ndarray:
    """Convert 64^3 indices to SG Mpc, matching the reference CSV convention."""
    delta = 1000.0 / 64.0
    sgx = -500.0 + (points[..., 0] + 0.5) * delta
    sgy = -500.0 + (points[..., 1] + 0.5) * delta
    sgz = -500.0 + (points[..., 2] + 0.5) * delta
    # The source script assigns index_to_sg(...) as sgz, sgy, sgx before
    # writing CSV columns. Preserve that convention for visual parity.
    return np.stack([sgz, sgy, sgx], axis=-1).astype(np.float32)


def select_streamline_seeds(
    basins: np.ndarray, preset: StreamlinePreset
) -> tuple[np.ndarray, np.ndarray]:
    """Select basin-grid seeds exactly like the reference streamline script."""
    seed_points: list[np.ndarray] = []
    seed_basins: list[np.ndarray] = []

    for basin_id in sorted(int(b) for b in np.unique(basins) if b != 0):
        # np.argwhere returns (z, y, x). The reference script names these
        # (ix, iy, iz) and then swaps back on output; keep that behavior.
        voxels = np.argwhere(basins == basin_id)
        nvox = len(voxels)
        if nvox <= preset.min_seeds_per_basin:
            stride = 1
        else:
            stride = max(1, nvox // preset.max_seeds_per_basin)

        chosen = voxels[::stride]
        if len(chosen) < preset.min_seeds_per_basin <= nvox:
            stride = max(1, nvox // preset.min_seeds_per_basin)
            chosen = voxels[::stride]

        sgx, sgy, sgz = index128_to_sg(chosen[:, 0], chosen[:, 1], chosen[:, 2])
        seed_points.append(sg_to_index64(sgx, sgy, sgz))
        seed_basins.append(np.full(len(chosen), basin_id, dtype=np.int16))
        aprint(
            f"Basin {basin_id}: {nvox:,} voxels, stride {stride}, {len(chosen):,} seeds"
        )

    return np.vstack(seed_points), np.concatenate(seed_basins)


def trilinear_scalar_batch(field: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Trilinearly interpolate a scalar field at many 64^3 index points."""
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    nz, ny, nx = field.shape

    ix0 = np.floor(x).astype(np.int32)
    iy0 = np.floor(y).astype(np.int32)
    iz0 = np.floor(z).astype(np.int32)
    valid = (
        (ix0 >= 0)
        & (iy0 >= 0)
        & (iz0 >= 0)
        & (ix0 < nx - 1)
        & (iy0 < ny - 1)
        & (iz0 < nz - 1)
    )

    out = np.full(len(points), np.nan, dtype=np.float32)
    if not np.any(valid):
        return out

    ix = ix0[valid]
    iy = iy0[valid]
    iz = iz0[valid]
    dx = (x[valid] - ix).astype(np.float32)
    dy = (y[valid] - iy).astype(np.float32)
    dz = (z[valid] - iz).astype(np.float32)

    c000 = field[iz, iy, ix]
    c100 = field[iz, iy, ix + 1]
    c010 = field[iz, iy + 1, ix]
    c110 = field[iz, iy + 1, ix + 1]
    c001 = field[iz + 1, iy, ix]
    c101 = field[iz + 1, iy, ix + 1]
    c011 = field[iz + 1, iy + 1, ix]
    c111 = field[iz + 1, iy + 1, ix + 1]

    c00 = c000 * (1.0 - dx) + c100 * dx
    c01 = c001 * (1.0 - dx) + c101 * dx
    c10 = c010 * (1.0 - dx) + c110 * dx
    c11 = c011 * (1.0 - dx) + c111 * dx
    c0 = c00 * (1.0 - dy) + c10 * dy
    c1 = c01 * (1.0 - dy) + c11 * dy
    out[valid] = c0 * (1.0 - dz) + c1 * dz
    return out


def velocity_unit_batch(
    vx: np.ndarray, vy: np.ndarray, vz: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """Interpolate and normalize the velocity vector at many points."""
    vectors = np.column_stack(
        [
            trilinear_scalar_batch(vx, points),
            trilinear_scalar_batch(vy, points),
            trilinear_scalar_batch(vz, points),
        ]
    ).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1)
    valid = np.isfinite(norms) & (norms > 0.0)
    out = np.full_like(vectors, np.nan)
    out[valid] = vectors[valid] / norms[valid, None]
    return out


def rk4_step_batch(
    points: np.ndarray,
    h_step: float,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
) -> np.ndarray:
    """One vectorized RK4 step for dx/ds = unit_velocity(x)."""
    out = np.full_like(points, np.nan)

    k1 = velocity_unit_batch(vx, vy, vz, points)
    valid = np.isfinite(k1).all(axis=1)
    if not np.any(valid):
        return out

    source_idx = np.flatnonzero(valid)
    p = points[source_idx]
    kk1 = k1[source_idx]

    k2 = velocity_unit_batch(vx, vy, vz, p + 0.5 * h_step * kk1)
    valid = np.isfinite(k2).all(axis=1)
    if not np.any(valid):
        return out
    source_idx = source_idx[valid]
    p = p[valid]
    kk1 = kk1[valid]
    kk2 = k2[valid]

    k3 = velocity_unit_batch(vx, vy, vz, p + 0.5 * h_step * kk2)
    valid = np.isfinite(k3).all(axis=1)
    if not np.any(valid):
        return out
    source_idx = source_idx[valid]
    p = p[valid]
    kk1 = kk1[valid]
    kk2 = kk2[valid]
    kk3 = k3[valid]

    k4 = velocity_unit_batch(vx, vy, vz, p + h_step * kk3)
    valid = np.isfinite(k4).all(axis=1)
    if not np.any(valid):
        return out
    source_idx = source_idx[valid]
    p = p[valid]
    kk1 = kk1[valid]
    kk2 = kk2[valid]
    kk3 = kk3[valid]
    kk4 = k4[valid]

    out[source_idx] = p + (h_step / 6.0) * (kk1 + 2 * kk2 + 2 * kk3 + kk4)
    return out


def integrate_streamlines(
    velocity_path: Path, basins: np.ndarray, preset: StreamlinePreset
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate streamline vertex histories in SG coordinates."""
    with asection("Generating CF4 streamlines"):
        seeds, seed_basins = select_streamline_seeds(basins, preset)
        n_seeds = len(seeds)
        aprint(f"Total seeds: {n_seeds:,}")

        velocity = np.load(velocity_path)
        # Reference script component mapping for velocity array shape (3,64,64,64).
        vx = velocity[2].astype(np.float32)
        vy = velocity[1].astype(np.float32)
        vz = velocity[0].astype(np.float32)

        positions = np.full((preset.nsteps + 1, n_seeds, 3), np.nan, dtype=np.float32)
        valid = np.zeros((preset.nsteps + 1, n_seeds), dtype=bool)
        positions[0] = seeds
        valid[0] = True

        current = seeds.copy()
        active = np.ones(n_seeds, dtype=bool)

        for step in range(1, preset.nsteps + 1):
            active_idx = np.flatnonzero(active)
            if len(active_idx) == 0:
                break

            next_points = rk4_step_batch(current[active_idx], preset.h_step, vx, vy, vz)
            ok = (
                np.isfinite(next_points).all(axis=1)
                & (next_points[:, 0] >= 0.0)
                & (next_points[:, 0] <= 63.0)
                & (next_points[:, 1] >= 0.0)
                & (next_points[:, 1] <= 63.0)
                & (next_points[:, 2] >= 0.0)
                & (next_points[:, 2] <= 63.0)
            )

            good_idx = active_idx[ok]
            bad_idx = active_idx[~ok]
            current[good_idx] = next_points[ok]
            positions[step, good_idx] = current[good_idx]
            valid[step, good_idx] = True
            active[bad_idx] = False

            if step == 1 or step % 50 == 0 or step == preset.nsteps:
                aprint(
                    f"RK4 step {step:>3}/{preset.nsteps}: "
                    f"{int(active.sum()):,} active streamlines"
                )

        positions_sg = index64_to_sg_output(positions)
        positions_line_major = np.transpose(positions_sg, (1, 0, 2))
        valid_line_major = valid.T
        valid_counts = valid_line_major.sum(axis=1)
        streamline_count = int(np.count_nonzero(valid_counts > 1))
        aprint(f"✓ Generated {streamline_count:,} valid streamlines")

        return positions_line_major, valid_line_major, seed_basins, valid_counts


def build_basin_line_data(
    positions: np.ndarray,
    valid: np.ndarray,
    seed_basins: np.ndarray,
    valid_counts: np.ndarray,
) -> list[BasinLineData]:
    """Convert line-major streamline histories into indexed Luxar Lines."""
    basin_lines: list[BasinLineData] = []
    with asection("Building indexed line geometry"):
        for basin_id in sorted(int(b) for b in np.unique(seed_basins)):
            line_mask = (seed_basins == basin_id) & (valid_counts > 1)
            streamline_count = int(np.count_nonzero(line_mask))
            if streamline_count == 0:
                continue

            basin_positions = positions[line_mask]
            basin_valid = valid[line_mask]
            n_vertices = int(np.count_nonzero(basin_valid))

            index_map = np.full(basin_valid.shape, -1, dtype=np.int32)
            index_map[basin_valid] = np.arange(n_vertices, dtype=np.int32)
            start = index_map[:, :-1]
            end = index_map[:, 1:]
            segment_mask = (start >= 0) & (end >= 0)
            segments = np.column_stack([start[segment_mask], end[segment_mask]]).astype(
                np.uint32
            )
            vertices = basin_positions[basin_valid].astype(np.float32, copy=False)

            basin_lines.append(
                BasinLineData(
                    basin_id=basin_id,
                    vertices=vertices,
                    segments=segments,
                    streamline_count=streamline_count,
                )
            )
            aprint(
                f"Basin {basin_id}: {streamline_count:,} streamlines, "
                f"{len(vertices):,} vertices, {len(segments):,} segments"
            )

    return basin_lines


# ---------------------------------------------------------------------------
# Luxar scene writing
# ---------------------------------------------------------------------------


def add_reference_cube(scene: Any, half_extent: float = 500.0) -> None:
    """Add a faint +/-500 Mpc bounding cube."""
    h = half_extent
    corners = np.array(
        [
            [-h, -h, -h],
            [h, -h, -h],
            [h, h, -h],
            [-h, h, -h],
            [-h, -h, h],
            [h, -h, h],
            [h, h, h],
            [-h, h, h],
        ],
        dtype=np.float32,
    )
    edges = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ],
        dtype=np.uint32,
    )
    scene.add_lines(
        "plus-minus 500 Mpc reference cube",
        vertices=corners,
        widths=1.0,
        colors=(0.45, 0.55, 0.70),
        sharpness=0.8,
        indices=edges.ravel(),
        line_type="indexed",
        opacity=0.16,
        intensity=0.35,
        blending_mode="additive",
        layer=True,
        visible=False,
    )


def write_laniakea_scene(
    output_path: Path,
    galaxies: GalaxyData,
    basin_lines: list[BasinLineData],
    preset_name: str,
    preset: StreamlinePreset,
) -> None:
    """Write the generated Cosmicflows scene as a Luxar Zarr archive."""
    with asection("Writing Luxar scene"):
        dims = Dimensions(
            [
                Dimension("SGX", unit="Mpc", range=(-500, 500), display=True),
                Dimension("SGY", unit="Mpc", range=(-500, 500), display=True),
                Dimension("SGZ", unit="Mpc", range=(-500, 500), display=True),
            ]
        )
        viewer_config = ViewerConfig(
            camera=CameraConfig(
                position=(1050.0, -1500.0, 780.0),
                target=(0.0, 0.0, 0.0),
                up=(0.0, 0.0, 1.0),
                fov=42.0,
                near=0.1,
                far=5000.0,
            ),
            background_color="#1d252b",
            tone_mapping="ACES",
            exposure=-4.5,
            bloom_enabled=True,
            bloom_strength=0.65,
            bloom_radius=0.45,
            bloom_threshold=0.05,
            auto_rotate=True,
            auto_rotate_speed=0.15,
            dynamic_clipping_enabled=True,
            ui=UIConfig(show_layers=True),
            theme="dark",
        )

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=dims, viewer_config=viewer_config)

            scene.add_points(
                "CF4 galaxies (55,486 in plus-minus 500 Mpc cube)",
                galaxies.positions,
                colors=galaxies.colors,
                radii=galaxies.radii * preset.point_radius,
                sharpness=0.5,
                opacity=0.55,
                intensity=0.45,
                blending_mode="additive",
                layer=True,
            )

            for basin in basin_lines:
                color = hex_to_rgb(BASIN_COLORS[basin.basin_id], intensity=1.55)
                scene.add_lines(
                    f"Basin {basin.basin_id} streamlines",
                    vertices=basin.vertices,
                    widths=preset.line_width,
                    colors=tuple(float(c) for c in color),
                    sharpness=0.45,
                    indices=basin.segments.ravel(),
                    line_type="indexed",
                    opacity=0.36,
                    intensity=0.75,
                    blending_mode="additive",
                    layer=True,
                )

            add_reference_cube(scene)

            n_streamlines = sum(b.streamline_count for b in basin_lines)
            scene.add_text(
                "Cosmicflows-4 / Laniakea",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )
            scene.add_text(
                f"{len(galaxies.positions):,} galaxies • "
                f"{n_streamlines:,} streamlines • {preset_name} preset • "
                "Cosmicflows-4 / EDD",
                position=(0.98, 0.97),
                font_size=0.015,
                anchor="bottom-right",
                color="rgba(200,200,200,0.45)",
            )

        aprint(f"✓ Written to {output_path}")


def generate_cosmicflows_laniakea(
    output_path: Path,
    preset_name: str = "full",
    cache_dir: Path | None = None,
) -> None:
    """Generate the full Luxar demo dataset."""
    preset = PRESETS[preset_name]
    cache = cache_dir or get_cache_dir()

    galaxy_path, velocity_path, basins_path = ensure_input_data(cache)
    basins = np.load(basins_path).astype(np.int32)
    galaxies = load_galaxies(galaxy_path, basins)
    positions, valid, seed_basins, valid_counts = integrate_streamlines(
        velocity_path, basins, preset
    )
    basin_lines = build_basin_line_data(positions, valid, seed_basins, valid_counts)
    write_laniakea_scene(output_path, galaxies, basin_lines, preset_name, preset)


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a Luxar Cosmicflows-4 / Laniakea flow demo."
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="full",
        help="Streamline density preset. 'full' reproduces 29,555 streamlines.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output .zarr path. Defaults to datasets/demos for --no-serve.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Input data cache directory (default: ~/.cache/luxar/laniakea).",
    )
    parser.add_argument(
        "--no-serve",
        action="store_true",
        help="Generate the dataset without launching the viewer.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Main demo entry point."""
    args = parse_args(sys.argv[1:] if argv is None else argv)

    aprint("=" * 72)
    aprint("COSMICFLOWS-4 LANIAKEA FLOW DEMO")
    aprint("=" * 72)
    aprint(f"Preset: {args.preset}")
    if args.preset == "full":
        aprint("Full preset target: 55,486 galaxies and 29,555 streamlines")
    aprint("")

    if args.no_serve:
        output_path = args.output or (
            get_demos_output_dir() / f"cosmicflows_laniakea_{args.preset}.luxar.zarr"
        )
        generate_cosmicflows_laniakea(
            output_path, preset_name=args.preset, cache_dir=args.cache_dir
        )
        aprint(f"✓ Dataset generated at {output_path}")
        return

    with tempfile.TemporaryDirectory(prefix="luxar_demo_laniakea_") as tmpdir:
        output_path = args.output or Path(tmpdir) / "cosmicflows_laniakea.luxar.zarr"
        generate_cosmicflows_laniakea(
            output_path, preset_name=args.preset, cache_dir=args.cache_dir
        )

        aprint("")
        aprint("=" * 72)
        aprint("LAUNCHING VIEWER")
        aprint("=" * 72)
        aprint("The viewer will open in your browser automatically.")
        aprint("Press Ctrl+C when done to stop and cleanup.")
        aprint("")

        launch_viewer(output_path)

    aprint("✓ Cleanup complete - temporary files removed")


if __name__ == "__main__":
    main()
