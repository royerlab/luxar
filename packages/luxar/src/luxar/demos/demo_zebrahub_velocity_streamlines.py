#!/usr/bin/env python3
"""Self-Contained Demo: Zebrahub 3D RNA-velocity UMAP + streamlines.

This demo turns the Zebrahub VeloCyto AnnData (spliced/unspliced counts +
precomputed 3D RNA-velocity UMAP embedding) into a luminous Luxar scene:

1. Auto-download the .h5ad file from the shared Zebrahub Google Drive folder
   into ``~/.cache/luxar/zebrahub_velocity/`` (override with ``--h5ad``).
2. Load 3D UMAP positions + 3D RNA-velocity vectors from ``obsm``.
3. Bin per-cell velocity into a regularized cubic vector field (sum-of-vectors
   over count, with ε regularization and Gaussian smoothing in voxel units).
4. RK4-integrate streamlines forward through the smoothed velocity field
   starting from a stratified subsample of cells.
5. Render Cells as soft Points (categorical color by anatomy ontology) and
   streamlines as Lines (colored by their seed cell's anatomy class).

Usage:
    hatch run python packages/luxar/src/luxar/demos/demo_zebrahub_velocity_streamlines.py
    hatch run python packages/luxar/src/luxar/demos/demo_zebrahub_velocity_streamlines.py --preset preview
    hatch run python packages/luxar/src/luxar/demos/demo_zebrahub_velocity_streamlines.py --no-serve
    hatch run python packages/luxar/src/luxar/demos/demo_zebrahub_velocity_streamlines.py --h5ad /path/to/zebrahub_velocity.h5ad

Requirements:
    pip install 'luxar[demos]' anndata h5py gdown scipy
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import CameraConfig, UIConfig, ViewerConfig
from luxar.demos import launch_viewer
from luxar.utils._umap_utils import get_categorical_color
from luxar.utils.fields import (
    FlowField,
    add_reference_cube_to_scene,
    cubic_bounds,
    rk4_step,
)
from luxar.utils.paths import get_demos_output_dir

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

CACHE_DIR: Final = Path.home() / ".cache" / "luxar" / "zebrahub_velocity"
DRIVE_FOLDER_URL: Final = (
    "https://drive.google.com/drive/folders/1kWWqy38ZKU_-dpVPO8Bh5TjPcmly8qvW"
)
CACHE_VERSION: Final = "v1"

# obsm keys: required for the 3D RNA-velocity vectors and a 3D UMAP layout.
# The atlas h5ad uses ``velocity_umap`` for the 3D velocity; the position key
# is auto-detected at load time from a list of common candidates.
VELOCITY_OBSM_KEY: Final = "velocity_umap"
POSITION_OBSM_CANDIDATES: Final = (
    "X_umap_3d",
    "X_umap3d",
    "X_umap",
    "umap_3d",
    "X_umap_velocity",
    "umap",
)
ANATOMY_OBS_KEY: Final = "zebrafish_anatomy_ontology_class"
STAGE_OBS_KEY: Final = "developmental_stage"
TIMEPOINT_OBS_KEY: Final = "timepoint"

# Viewer default exposure is 0 EV — all dimming is baked into the per-layer
# intensities so the scene reads correctly without the user having to touch
# the exposure slider, and there is symmetric headroom in both directions.
VIEWER_EXPOSURE_EV: Final = 0.0
# Lines (velocity comets + streamlines) carry the directional information and
# get the dominant brightness budget.  Cells are 2× dimmer so the line field
# reads as the foreground.  The reference cube is a faint outline.
LINE_INTENSITY: Final = float(2.0**-7.0)  # ≈ 0.00781
CELL_INTENSITY: Final = LINE_INTENSITY * 0.5
REF_CUBE_INTENSITY: Final = LINE_INTENSITY * 0.35


@dataclass(frozen=True)
class StreamlinePreset:
    """Vector-field + streamline numerical settings.

    ``streamline_seeds = None`` means seed every cell.  Streamlines are kept
    short by design (small ``streamline_steps`` * ``step_voxels``) and
    truncated as soon as the head drifts more than
    ``cell_proximity_voxels`` voxels from the nearest cell.
    """

    name: str
    grid_size: int
    gaussian_sigma: float
    streamline_seeds: int | None
    streamline_steps: int
    step_voxels: float
    cell_proximity_voxels: float
    cell_radius: float
    streamline_width: float


PRESETS: Final[dict[str, StreamlinePreset]] = {
    "preview": StreamlinePreset(
        name="preview",
        grid_size=80,
        gaussian_sigma=1.0,
        streamline_seeds=None,
        streamline_steps=20,
        step_voxels=0.55,
        cell_proximity_voxels=2.5,
        cell_radius=0.030,
        streamline_width=0.0085,
    ),
    "standard": StreamlinePreset(
        name="standard",
        grid_size=128,
        gaussian_sigma=1.1,
        streamline_seeds=None,
        streamline_steps=30,
        step_voxels=0.55,
        cell_proximity_voxels=3.0,
        cell_radius=0.024,
        streamline_width=0.0066,
    ),
    "hifi": StreamlinePreset(
        name="hifi",
        grid_size=256,
        gaussian_sigma=1.3,
        streamline_seeds=None,
        streamline_steps=60,
        step_voxels=0.50,
        cell_proximity_voxels=3.0,
        cell_radius=0.020,
        streamline_width=0.0050,
    ),
}


@dataclass(frozen=True)
class ZebrahubData:
    """Per-cell positions, velocities, and categorical metadata."""

    positions: np.ndarray  # (N, 3) float32, stabilized
    velocities: np.ndarray  # (N, 3) float32, in stabilized coords
    anatomy_codes: np.ndarray  # (N,) int32
    anatomy_categories: list[str]
    stage_codes: np.ndarray  # (N,) int32 (or zeros if absent)
    stage_categories: list[str]


@dataclass(frozen=True)
class StreamlineGeometry:
    """Indexed Luxar Lines geometry for the integrated streamlines."""

    vertices: np.ndarray
    segments: np.ndarray
    colors: np.ndarray
    streamline_count: int


# -----------------------------------------------------------------------------
# Auto-install helpers (mirror the demo_tabula_sapiens approach)
# -----------------------------------------------------------------------------


def _require_module(module: str, pip_name: str | None = None) -> Any:
    """Import a module, raising a clear error with install instructions if missing.

    The earlier version of this helper silently ran ``pip install`` with
    stdout/stderr suppressed, mutating the user's Python environment
    without confirmation. That is hostile to constrained environments
    (HPC, locked-down CI, virtualenvs with pinned deps) and hides the
    install failure mode entirely. The PPI demo's import-error pattern is
    safer and more transparent.
    """
    try:
        return __import__(module)
    except ImportError as exc:
        pkg = pip_name or module
        aprint(f"❌ Missing dependency for the zebrahub demo: {module}")
        aprint(f"   Install with: pip install {pkg}")
        raise ImportError(
            f"{module} is required by demo_zebrahub_velocity_streamlines. "
            f"Install with `pip install {pkg}`."
        ) from exc


# -----------------------------------------------------------------------------
# Data resolution: download .h5ad from the shared Drive folder
# -----------------------------------------------------------------------------


def resolve_h5ad(cache_dir: Path, h5ad_override: Path | None) -> Path:
    """Return the path to the Zebrahub velocity .h5ad, downloading if missing.

    If ``h5ad_override`` is given, it takes precedence and is returned as-is.
    Otherwise the shared Google Drive folder is mirrored into ``cache_dir`` via
    ``gdown``; the first ``*.h5ad`` member is returned.
    """
    if h5ad_override is not None:
        path = h5ad_override.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"--h5ad path does not exist: {path}")
        if path.suffix != ".h5ad":
            raise ValueError(f"--h5ad must point to a .h5ad file, got {path.suffix}")
        aprint(f"Using local AnnData: {path}")
        return path

    cache_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(cache_dir.glob("*.h5ad"))
    if existing:
        path = existing[0]
        size_gb = path.stat().st_size / (1024**3)
        aprint(f"Using cached AnnData: {path.name} ({size_gb:.2f} GB)")
        return path

    gdown = _require_module("gdown")
    with asection("Downloading Zebrahub velocity AnnData from Google Drive"):
        aprint(f"  Folder: {DRIVE_FOLDER_URL}")
        aprint(f"  Target: {cache_dir}")
        aprint("  Note: ~4 GB; first download may take a while.")
        gdown.download_folder(
            url=DRIVE_FOLDER_URL,
            output=str(cache_dir),
            quiet=False,
            use_cookies=False,
            resume=True,
        )

    h5ad_files = sorted(cache_dir.rglob("*.h5ad"))
    if not h5ad_files:
        raise RuntimeError(
            f"No .h5ad file found under {cache_dir} after gdown download. "
            "If the Drive folder is restricted, download the file manually "
            "and pass it via --h5ad."
        )
    path = h5ad_files[0]
    aprint(f"  ✓ Resolved AnnData: {path}")
    return path


# -----------------------------------------------------------------------------
# AnnData loading (uses anndata in backed mode to avoid loading the count matrix)
# -----------------------------------------------------------------------------


def _categorical_obs(adata: Any, key: str) -> tuple[np.ndarray, list[str]]:
    """Return (codes, categories) for an obs column, treating strings as cats."""
    if key not in adata.obs.columns:
        return np.zeros(adata.n_obs, dtype=np.int32), []

    series = adata.obs[key]
    if hasattr(series, "cat") and hasattr(series.cat, "categories"):
        codes = series.cat.codes.to_numpy().astype(np.int32, copy=False)
        categories = [str(c) for c in series.cat.categories]
    else:
        # Plain string column; build a stable mapping.
        values = series.astype(str).to_numpy()
        categories_arr, inv = np.unique(values, return_inverse=True)
        codes = inv.astype(np.int32)
        categories = [str(c) for c in categories_arr]

    # Map -1 (NaN) to a synthetic "unknown" bin so downstream indexing is safe.
    if (codes < 0).any():
        unknown_idx = len(categories)
        codes = np.where(codes < 0, unknown_idx, codes).astype(np.int32)
        categories = list(categories) + ["unknown"]
    return codes, categories


def _stabilize_3d(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Center, PCA-align, sign-stabilize coords.  Return (R, mean) so the same
    transform can be applied to velocity vectors (which need rotation only).
    """
    coords = coords.astype(np.float32, copy=True)
    mean = coords.mean(axis=0, keepdims=True).astype(np.float32)
    centered = coords - mean

    if len(centered) >= 3:
        _, _, vt = np.linalg.svd(centered.astype(np.float64), full_matrices=False)
        rotation = vt.T.astype(np.float32)
    else:
        rotation = np.eye(3, dtype=np.float32)

    rotated = centered @ rotation
    sign = np.ones(3, dtype=np.float32)
    for axis in range(3):
        idx = int(np.argmax(np.abs(rotated[:, axis])))
        if rotated[idx, axis] < 0:
            sign[axis] = -1.0
    rotation = rotation * sign[None, :]
    rotated = centered @ rotation

    radius_95 = float(np.percentile(np.linalg.norm(rotated, axis=1), 95))
    if radius_95 > 0:
        scale = np.float32(10.0 / radius_95)
    else:
        scale = np.float32(1.0)
    transform = (rotation * scale).astype(np.float32)
    return transform, mean.astype(np.float32)


def load_zebrahub(h5ad_path: Path) -> ZebrahubData:
    """Load 3D UMAP positions, RNA-velocity vectors, and metadata."""
    ad = _require_module("anndata")

    with asection(f"Reading {h5ad_path.name}"):
        adata = ad.read_h5ad(h5ad_path, backed="r")
        aprint(f"  {adata.n_obs:,} cells × {adata.n_vars:,} genes")

        if VELOCITY_OBSM_KEY not in adata.obsm:
            available = list(adata.obsm.keys())
            raise KeyError(f"obsm['{VELOCITY_OBSM_KEY}'] missing.  Found: {available}")
        velocity_raw = np.asarray(adata.obsm[VELOCITY_OBSM_KEY]).astype(
            np.float32, copy=False
        )
        if velocity_raw.shape[1] < 3:
            raise ValueError(
                f"obsm['{VELOCITY_OBSM_KEY}'] has shape {velocity_raw.shape}; "
                "expected at least 3 components"
            )
        velocity_raw = velocity_raw[:, :3]

        position_key: str | None = None
        for candidate in POSITION_OBSM_CANDIDATES:
            if candidate in adata.obsm:
                arr = np.asarray(adata.obsm[candidate])
                if arr.ndim == 2 and arr.shape[1] >= 3:
                    position_key = candidate
                    break
        if position_key is None:
            available = list(adata.obsm.keys())
            raise KeyError(
                "Could not find a 3D UMAP position key in obsm.  "
                f"Tried {POSITION_OBSM_CANDIDATES}; available: {available}"
            )
        positions_raw = np.asarray(adata.obsm[position_key]).astype(
            np.float32, copy=False
        )[:, :3]
        aprint(f"  positions: obsm['{position_key}']  shape={positions_raw.shape}")
        aprint(f"  velocities: obsm['{VELOCITY_OBSM_KEY}']  shape={velocity_raw.shape}")

        anatomy_codes, anatomy_categories = _categorical_obs(adata, ANATOMY_OBS_KEY)
        if anatomy_categories:
            aprint(f"  anatomy ({ANATOMY_OBS_KEY}): {len(anatomy_categories)} classes")
        else:
            aprint(f"  anatomy column '{ANATOMY_OBS_KEY}' missing; using single class")
            anatomy_categories = ["unknown"]

        stage_codes, stage_categories = _categorical_obs(adata, STAGE_OBS_KEY)
        if not stage_categories:
            stage_codes, stage_categories = _categorical_obs(adata, TIMEPOINT_OBS_KEY)
        if not stage_categories:
            stage_categories = ["unknown"]

    transform, mean = _stabilize_3d(positions_raw)
    positions = ((positions_raw - mean) @ transform).astype(np.float32)
    velocities = (velocity_raw @ transform).astype(np.float32)

    aprint(
        f"  Stabilized layout extents: "
        f"x={float(np.ptp(positions[:, 0])):.2f}, "
        f"y={float(np.ptp(positions[:, 1])):.2f}, "
        f"z={float(np.ptp(positions[:, 2])):.2f}"
    )
    median_speed = float(np.median(np.linalg.norm(velocities, axis=1)))
    aprint(f"  Median |velocity|: {median_speed:.4f}")
    return ZebrahubData(
        positions=positions,
        velocities=velocities,
        anatomy_codes=anatomy_codes,
        anatomy_categories=anatomy_categories,
        stage_codes=stage_codes,
        stage_categories=stage_categories,
    )


# -----------------------------------------------------------------------------
# Vector-field construction (per-cell binning + Gaussian smoothing)
# -----------------------------------------------------------------------------


def _array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for arr in arrays:
        contig = np.ascontiguousarray(arr.astype(np.float32, copy=False))
        digest.update(contig.tobytes())
    return digest.hexdigest()[:16]


# Cubic-bounds, trilinear vector sampling, RK4 advection, and reference-cube
# rendering live in luxar.utils.fields. The imports at the top of this module
# bring in cubic_bounds, rk4_step, FlowField, and add_reference_cube_to_scene.


def compute_velocity_field(
    data: ZebrahubData,
    preset: StreamlinePreset,
    cache_path: Path,
    recompute: bool,
) -> FlowField:
    """Bin cell-level velocity into a smoothed cubic field, with cache."""
    layout_hash = _array_hash(data.positions, data.velocities)
    cache_key = (
        f"{CACHE_VERSION}:{layout_hash}:{preset.grid_size}:{preset.gaussian_sigma}"
    )

    if cache_path.exists() and not recompute:
        store = np.load(cache_path, allow_pickle=False)
        if "cache_key" in store and str(store["cache_key"]) == cache_key:
            with asection("Loading cached velocity field"):
                vectors = store["vectors"].astype(np.float32)
                grid_min = store["grid_min"].astype(np.float32)
                grid_max = store["grid_max"].astype(np.float32)
                spacing = float(store["spacing"])
                aprint(f"  {vectors.shape[0]}^3 field from {cache_path.name}")
                return FlowField(vectors, grid_min, grid_max, spacing, cache_key)

    grid_min, grid_max = cubic_bounds(data.positions, pad_fraction=0.06)
    n = preset.grid_size
    spacing = float((grid_max[0] - grid_min[0]) / max(n - 1, 1))

    with asection("Building velocity vector field"):
        aprint(
            f"  Grid: {n}^3 = {n**3:,} cells, spacing={spacing:.4f}, "
            f"raw size≈{n**3 * 3 * 4 / (1024 * 1024):.1f} MB"
        )

        idx = (data.positions - grid_min[None, :]) / np.float32(spacing)
        ix = np.floor(idx[:, 0]).astype(np.int32)
        iy = np.floor(idx[:, 1]).astype(np.int32)
        iz = np.floor(idx[:, 2]).astype(np.int32)
        valid = (ix >= 0) & (iy >= 0) & (iz >= 0) & (ix < n) & (iy < n) & (iz < n)
        ix = ix[valid]
        iy = iy[valid]
        iz = iz[valid]
        velocities = data.velocities[valid]
        flat = (ix * n + iy) * n + iz
        n_cells = n**3

        sums = np.zeros((n_cells, 3), dtype=np.float64)
        counts = np.zeros(n_cells, dtype=np.float64)
        np.add.at(sums, flat, velocities.astype(np.float64))
        np.add.at(counts, flat, 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(counts[:, None] > 0, sums / counts[:, None], 0.0)
        field = mean.astype(np.float32).reshape(n, n, n, 3)
        aprint(
            f"  Populated {int((counts > 0).sum()):,} / {n_cells:,} cells "
            f"({100.0 * (counts > 0).mean():.1f}%)"
        )

        if preset.gaussian_sigma > 0:
            from scipy.ndimage import gaussian_filter

            aprint(f"  Gaussian smoothing (sigma={preset.gaussian_sigma} voxels)")
            for component in range(3):
                field[..., component] = gaussian_filter(
                    field[..., component], sigma=preset.gaussian_sigma, mode="nearest"
                )

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            cache_key=np.array(cache_key),
            vectors=field,
            grid_min=grid_min,
            grid_max=grid_max,
            spacing=np.array(spacing, dtype=np.float32),
        )
        aprint(f"  ✓ Cached to {cache_path.name}")

    return FlowField(field, grid_min, grid_max, spacing, cache_key)


# RK4 streamline integration uses rk4_step from luxar.utils.fields
# (imported at the top of this module).


def _select_seeds(data: ZebrahubData, n_seeds: int | None) -> np.ndarray:
    """Stratified subsample of cell indices: evenly across anatomy classes.

    If ``n_seeds`` is ``None`` or ≥ total cells, every cell is seeded.
    """
    n_cells = len(data.positions)
    if n_seeds is None or n_seeds >= n_cells:
        return np.arange(n_cells, dtype=np.int32)

    rng = np.random.default_rng(42)
    n_classes = max(int(data.anatomy_codes.max()) + 1, 1)
    per_class = max(n_seeds // n_classes, 1)

    chosen: list[np.ndarray] = []
    for class_id in range(n_classes):
        members = np.flatnonzero(data.anatomy_codes == class_id)
        if len(members) == 0:
            continue
        take = min(per_class, len(members))
        chosen.append(rng.choice(members, size=take, replace=False))

    seeds = np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)
    if len(seeds) < n_seeds:
        remaining = np.setdiff1d(np.arange(n_cells, dtype=np.int64), seeds)
        if len(remaining) > 0:
            extra = rng.choice(
                remaining, size=min(n_seeds - len(seeds), len(remaining)), replace=False
            )
            seeds = np.concatenate([seeds, extra])
    if len(seeds) > n_seeds:
        seeds = rng.choice(seeds, size=n_seeds, replace=False)
    return np.sort(seeds.astype(np.int32))


def integrate_streamlines(
    data: ZebrahubData,
    flow: FlowField,
    preset: StreamlinePreset,
    cache_path: Path,
    recompute: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RK4-advect cells through the velocity field, with cache.

    Streamlines stop as soon as the head drifts farther than
    ``preset.cell_proximity_voxels`` voxels from the nearest real cell — long
    excursions through empty UMAP regions, where the smoothed velocity field
    is unsupported by data, are simply truncated rather than rendered.
    """
    from scipy.spatial import cKDTree

    proximity_threshold = float(preset.cell_proximity_voxels * flow.spacing)
    seed_token = (
        "all" if preset.streamline_seeds is None else str(preset.streamline_seeds)
    )
    cache_key = (
        f"{CACHE_VERSION}:{flow.cache_key}:{preset.streamline_steps}:"
        f"{preset.step_voxels}:seeds={seed_token}:"
        f"prox={preset.cell_proximity_voxels}"
    )

    if cache_path.exists() and not recompute:
        store = np.load(cache_path, allow_pickle=False)
        if "cache_key" in store and str(store["cache_key"]) == cache_key:
            with asection("Loading cached streamlines"):
                positions = store["positions"].astype(np.float32)
                valid = store["valid"].astype(bool)
                seed_indices = store["seed_indices"].astype(np.int32)
                aprint(f"  {len(seed_indices):,} streamlines from {cache_path.name}")
                return positions, valid, seed_indices

    with asection("Integrating streamlines (RK4)"):
        seed_indices = _select_seeds(data, preset.streamline_seeds)
        seeds = data.positions[seed_indices].astype(np.float32)
        n_seeds = len(seeds)
        n_steps = preset.streamline_steps
        step_size = float(preset.step_voxels * flow.spacing)
        aprint(
            f"  Seeds: {n_seeds:,} (stratified by anatomy); "
            f"steps: {n_steps}; step={step_size:.4f} ({preset.step_voxels} voxels)"
        )
        aprint(
            f"  Cell proximity cutoff: {preset.cell_proximity_voxels:.1f} voxels "
            f"(={proximity_threshold:.4f} in UMAP units)"
        )

        cell_tree = cKDTree(data.positions.astype(np.float32))

        positions = np.full((n_seeds, n_steps + 1, 3), np.nan, dtype=np.float32)
        valid = np.zeros((n_seeds, n_steps + 1), dtype=bool)
        positions[:, 0] = seeds
        valid[:, 0] = True
        current = seeds.copy()
        active = np.ones(n_seeds, dtype=bool)

        for step in range(1, n_steps + 1):
            active_idx = np.flatnonzero(active)
            if len(active_idx) == 0:
                break

            next_points = rk4_step(current[active_idx], step_size, flow)
            inside = np.all(
                (next_points >= flow.grid_min[None, :])
                & (next_points <= flow.grid_max[None, :]),
                axis=1,
            )
            finite = np.isfinite(next_points).all(axis=1)
            checkable = finite & inside

            near_cells = np.zeros(len(next_points), dtype=bool)
            if np.any(checkable):
                query_points = next_points[checkable]
                nearest_dist, _ = cell_tree.query(query_points, k=1, workers=-1)
                near_cells[checkable] = nearest_dist <= proximity_threshold

            ok = checkable & near_cells
            good = active_idx[ok]
            bad = active_idx[~ok]
            current[good] = next_points[ok]
            positions[good, step] = current[good]
            valid[good, step] = True
            active[bad] = False

            if step == 1 or step % 20 == 0 or step == n_steps:
                aprint(f"    step {step:>3}/{n_steps}: {int(active.sum()):,} active")

        n_valid = int(np.count_nonzero(valid.sum(axis=1) > 1))
        aprint(f"  ✓ {n_valid:,} streamlines with ≥2 vertices")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            cache_key=np.array(cache_key),
            positions=positions,
            valid=valid,
            seed_indices=seed_indices,
        )
        aprint(f"  Cached to {cache_path.name}")

    return positions, valid, seed_indices


def build_streamline_geometry(
    positions: np.ndarray,
    valid: np.ndarray,
    seed_indices: np.ndarray,
    anatomy_codes: np.ndarray,
    palette: np.ndarray,
) -> StreamlineGeometry:
    """Convert per-seed histories into indexed Luxar Lines geometry."""
    valid_counts = valid.sum(axis=1)
    line_mask = valid_counts > 1
    streamline_count = int(np.count_nonzero(line_mask))
    if streamline_count == 0:
        return StreamlineGeometry(
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 2), dtype=np.uint32),
            np.empty((0, 3), dtype=np.float32),
            0,
        )

    line_positions = positions[line_mask]
    line_valid = valid[line_mask]
    line_seeds = seed_indices[line_mask]
    n_vertices = int(np.count_nonzero(line_valid))

    index_map = np.full(line_valid.shape, -1, dtype=np.int32)
    index_map[line_valid] = np.arange(n_vertices, dtype=np.int32)
    start = index_map[:, :-1]
    end = index_map[:, 1:]
    seg_mask = (start >= 0) & (end >= 0)
    segments = np.column_stack([start[seg_mask], end[seg_mask]]).astype(np.uint32)
    vertices = line_positions[line_valid].astype(np.float32, copy=False)

    seed_anatomy = anatomy_codes[line_seeds]
    source_colors = palette[seed_anatomy].astype(np.float32)

    # Comet-style fade along each streamline: bright at the seed cell, fading
    # to ~0 at the trail end.  Step index is used as a proxy for advection
    # time, so all streamlines share the same fade rate.
    n_steps_total = line_valid.shape[1]
    if n_steps_total > 1:
        t_norm = np.linspace(0.0, 1.0, n_steps_total, dtype=np.float32)
    else:
        t_norm = np.zeros(1, dtype=np.float32)
    fade = (1.0 - t_norm).astype(np.float32)
    curve = (fade**1.5).astype(np.float32)
    head_brightness = np.float32(1.40)
    tail_brightness = np.float32(0.04)
    brightness = (head_brightness * curve + tail_brightness * (1.0 - curve)).astype(
        np.float32
    )

    expanded = (
        source_colors[:, None, :] * brightness[None, :, None]
    )  # (n_kept, n_steps, 3)
    colors = np.clip(expanded[line_valid], 0.0, 1.7).astype(np.float32)

    return StreamlineGeometry(vertices, segments, colors, streamline_count)


# -----------------------------------------------------------------------------
# Scene assembly
# -----------------------------------------------------------------------------


def build_velocity_comets(
    data: ZebrahubData,
    palette: np.ndarray,
    side: float,
    head_width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Build per-cell velocity comets as a segments-mode Lines geometry.

    The cell sits at the bright comet head; the tail extends *opposite* to
    velocity (P → P − scale·v) so the head leads in the velocity direction.
    Width tapers from ``head_width`` at the head to ~0 at the tail tip, and
    color tapers from a brightened anatomy color to a dim one — combined with
    additive blending this produces the vanishing-tail comet look.

    Returns ``(vertices, widths, colors, mean_length)``.
    """
    n = len(data.positions)
    speeds = np.linalg.norm(data.velocities, axis=1).astype(np.float32)
    nonzero = speeds > 0
    median_speed = float(np.median(speeds[nonzero])) if np.any(nonzero) else 1.0
    if median_speed <= 0:
        median_speed = 1.0

    target_length = 0.018 * float(side)
    max_length = 3.0 * target_length
    scale = target_length / median_speed

    scaled = data.velocities.astype(np.float32) * np.float32(scale)
    lengths = np.linalg.norm(scaled, axis=1)
    over = lengths > max_length
    if np.any(over):
        factor = np.where(over, max_length / np.maximum(lengths, 1e-9), 1.0).astype(
            np.float32
        )
        scaled = scaled * factor[:, None]

    heads = data.positions.astype(np.float32)
    tails = (heads - scaled).astype(np.float32)
    vertices = np.empty((2 * n, 3), dtype=np.float32)
    vertices[0::2] = heads
    vertices[1::2] = tails

    widths = np.empty(2 * n, dtype=np.float32)
    widths[0::2] = head_width
    widths[1::2] = max(head_width * 0.04, 1e-4)

    base = palette[data.anatomy_codes].astype(np.float32)
    head_colors = np.clip(base * 1.55, 0.0, 1.7).astype(np.float32)
    tail_colors = np.clip(base * 0.12, 0.0, 1.0).astype(np.float32)
    colors = np.empty((2 * n, 3), dtype=np.float32)
    colors[0::2] = head_colors
    colors[1::2] = tail_colors

    mean_length = float(np.mean(np.linalg.norm(scaled, axis=1)))
    return vertices, widths, colors, mean_length


def categorical_palette(n: int) -> np.ndarray:
    return (
        np.array(
            [get_categorical_color(i, max(n, 1)) for i in range(max(n, 1))],
            dtype=np.float32,
        )
        / 255.0
    )


def build_legend_html(
    anatomy_categories: list[str],
    palette: np.ndarray,
    counts: np.ndarray,
    top_n: int = 16,
) -> str:
    order = np.argsort(-counts)
    parts = [
        '<div style="font-size:1.25vh;line-height:1.45;'
        "background:rgba(0,0,0,0.58);padding:0.7vh 0.9vh;"
        'border-radius:5px;max-width:34vh">'
        '<div style="color:#ffcc44;font-weight:bold;margin-bottom:0.45vh">'
        "Zebrahub RNA-velocity field</div>"
        '<div style="color:#ddd;margin-bottom:0.6vh">'
        "Cells colored by anatomy ontology class.  Each cell carries a comet "
        "tail: <b>tip → head</b> = local RNA-velocity direction.  "
        "Streamlines briefly forward-advect every cell through the smoothed "
        "velocity field and fade out along advection time."
        "</div>"
        '<div style="color:#ffcc44;font-weight:bold;margin-bottom:0.25vh">'
        f"Top anatomy classes ({len(anatomy_categories)})</div>"
    ]
    for code in order[:top_n]:
        if counts[code] == 0:
            continue
        rgb = (palette[code] * 255.0).astype(int).clip(0, 255)
        parts.append(
            f'<div style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'
            f'<span style="color:rgb({rgb[0]},{rgb[1]},{rgb[2]})">█</span> '
            f"{anatomy_categories[code]} ({int(counts[code]):,})</div>"
        )
    if (counts > 0).sum() > top_n:
        remaining = int((counts > 0).sum() - top_n)
        parts.append(f'<div style="color:#888">… +{remaining} more</div>')
    parts.append("</div>")
    return "".join(parts)


def write_scene(
    output_path: Path,
    data: ZebrahubData,
    flow: FlowField,
    streamlines: StreamlineGeometry,
    preset: StreamlinePreset,
) -> None:
    """Write the final Luxar scene."""
    n_classes = len(data.anatomy_categories)
    palette = categorical_palette(n_classes)

    cell_colors = palette[data.anatomy_codes].astype(np.float32)
    cell_brightness = 0.78
    cell_colors = np.clip(cell_colors * cell_brightness, 0.0, 1.5).astype(np.float32)

    stage_lookup = data.stage_categories
    anatomy_lookup = data.anatomy_categories
    cell_labels = [
        f"{anatomy_lookup[int(a)] if int(a) < len(anatomy_lookup) else int(a)}\n"
        f"[stage {stage_lookup[int(s)] if int(s) < len(stage_lookup) else int(s)}]"
        for a, s in zip(data.anatomy_codes, data.stage_codes, strict=True)
    ]

    counts = np.zeros(n_classes, dtype=np.int64)
    np.add.at(counts, data.anatomy_codes, 1)

    center = (flow.grid_min + flow.grid_max) * 0.5
    side = float(flow.grid_max[0] - flow.grid_min[0])
    camera_position = (
        float(center[0] + 1.30 * side),
        float(center[1] - 1.65 * side),
        float(center[2] + 1.05 * side),
    )

    with asection("Writing Luxar scene"):
        dims = Dimensions(
            [
                Dimension(
                    "x",
                    unit="UMAP",
                    range=(float(flow.grid_min[0]), float(flow.grid_max[0])),
                    display=True,
                ),
                Dimension(
                    "y",
                    unit="UMAP",
                    range=(float(flow.grid_min[1]), float(flow.grid_max[1])),
                    display=True,
                ),
                Dimension(
                    "z",
                    unit="UMAP",
                    range=(float(flow.grid_min[2]), float(flow.grid_max[2])),
                    display=True,
                ),
            ]
        )
        viewer_config = ViewerConfig(
            camera=CameraConfig(
                position=camera_position,
                target=tuple(float(v) for v in center),
                up=(0.0, 0.0, 1.0),
                fov=42.0,
                near=0.01,
                far=side * 12.0,
            ),
            background_color="#0c1018",
            tone_mapping="ACES",
            exposure=VIEWER_EXPOSURE_EV,
            bloom_enabled=True,
            bloom_strength=0.74,
            bloom_radius=0.55,
            bloom_threshold=0.0,
            auto_rotate=True,
            auto_rotate_speed=0.16,
            dynamic_clipping_enabled=True,
            ui=UIConfig(show_layers=True),
            theme="dark",
        )

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=dims, viewer_config=viewer_config)

            scene.add_points(
                "Cells (anatomy color)",
                positions=data.positions.astype(np.float32),
                colors=cell_colors,
                radii=np.full(
                    len(data.positions), preset.cell_radius, dtype=np.float32
                ),
                sharpness=np.full(len(data.positions), 0.55, dtype=np.float32),
                opacity=0.92,
                intensity=CELL_INTENSITY,
                labels=cell_labels,
                layer=True,
            )

            head_width = max(preset.cell_radius * 0.55, 0.005)
            comet_vertices, comet_widths, comet_colors, comet_mean = (
                build_velocity_comets(data, palette, side, head_width)
            )
            aprint(
                f"  Velocity comets: {len(data.positions):,} tails, "
                f"mean length {comet_mean:.3f}, head width {head_width:.4f}"
            )
            scene.add_lines(
                "Velocity comets (tail → head = velocity direction)",
                vertices=comet_vertices,
                widths=comet_widths,
                colors=comet_colors,
                sharpness=0.85,
                line_type="segments",
                blending_mode="additive",
                opacity=0.95,
                intensity=LINE_INTENSITY,
                layer=True,
            )

            if len(streamlines.vertices) > 0:
                scene.add_lines(
                    "RNA-velocity streamlines",
                    vertices=streamlines.vertices,
                    widths=preset.streamline_width,
                    colors=streamlines.colors,
                    sharpness=0.55,
                    indices=streamlines.segments.ravel(),
                    line_type="indexed",
                    blending_mode="additive",
                    opacity=0.95,
                    intensity=LINE_INTENSITY,
                    layer=True,
                )

            add_reference_cube_to_scene(
                scene,
                flow.grid_min,
                flow.grid_max,
                name="Velocity field bounding cube",
                widths=0.0055,
                opacity=0.20,
                intensity=REF_CUBE_INTENSITY,
            )

            scene.add_text(
                "Zebrahub — 3D RNA-velocity UMAP",
                position=(0.02, 0.02),
                font_size=0.052,
                anchor="top-left",
                color="rgba(255,255,255,0.66)",
                blend_mode="difference",
            )
            scene.add_text(
                "{hover_label}",
                position=(0.02, 0.50),
                anchor="center-left",
                font_size=0.020,
                color="white",
                background="rgba(0,0,0,0.74)",
                padding=0.010,
                text_align="left",
                opacity=1.0,
                transition="fade",
                transition_duration=0.15,
                hover=True,
            )
            scene.add_html(
                build_legend_html(data.anatomy_categories, palette, counts),
                position=(0.98, 0.50),
                anchor="center-right",
                opacity=0.92,
            )
            scene.add_text(
                f"{len(data.positions):,} cells · {streamlines.streamline_count:,} streamlines · "
                f"{preset.grid_size}³ field · {preset.name} preset",
                position=(0.98, 0.97),
                font_size=0.0125,
                anchor="bottom-right",
                color="rgba(220,220,220,0.52)",
            )

        aprint(f"  ✓ Scene written to {output_path}")


# -----------------------------------------------------------------------------
# Pipeline orchestration
# -----------------------------------------------------------------------------


def generate_zebrahub_scene(
    output_path: Path,
    cache_dir: Path,
    preset: StreamlinePreset,
    h5ad_override: Path | None,
    recompute_field: bool,
    recompute_streamlines: bool,
) -> tuple[ZebrahubData, FlowField, StreamlineGeometry]:
    """Run the full pipeline and write the Luxar scene."""
    h5ad_path = resolve_h5ad(cache_dir, h5ad_override)
    data = load_zebrahub(h5ad_path)

    field_cache = (
        cache_dir
        / f"velocity_field_{preset.name}_{preset.grid_size}_{CACHE_VERSION}.npz"
    )
    flow = compute_velocity_field(data, preset, field_cache, recompute_field)

    seed_token = (
        "all" if preset.streamline_seeds is None else str(preset.streamline_seeds)
    )
    stream_cache = (
        cache_dir / f"streamlines_{preset.name}_{preset.grid_size}_"
        f"{preset.streamline_steps}_{seed_token}_{CACHE_VERSION}.npz"
    )
    positions, valid, seed_indices = integrate_streamlines(
        data, flow, preset, stream_cache, recompute_streamlines
    )
    palette = categorical_palette(len(data.anatomy_categories))
    streamlines = build_streamline_geometry(
        positions, valid, seed_indices, data.anatomy_codes, palette
    )

    write_scene(output_path, data, flow, streamlines, preset)
    return data, flow, streamlines


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Zebrahub 3D RNA-velocity UMAP + advected streamlines Luxar demo.")
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="standard",
        help="Numerical preset (preview / standard / hifi).",
    )
    parser.add_argument(
        "--no-serve",
        action="store_true",
        help="Write the .zarr dataset without launching the viewer.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .zarr path. Defaults to datasets/demos for --no-serve.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=CACHE_DIR,
        help="Cache directory for downloads, fields, and streamlines.",
    )
    parser.add_argument(
        "--h5ad",
        type=Path,
        default=None,
        help="Path to a local .h5ad file (skips Drive download).",
    )
    parser.add_argument("--recompute-field", action="store_true")
    parser.add_argument("--recompute-streamlines", action="store_true")
    parser.add_argument(
        "--recompute-all",
        action="store_true",
        help="Recompute the velocity field and streamlines.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=None,
        help="Override preset grid size.",
    )
    parser.add_argument(
        "--streamline-seeds",
        type=int,
        default=None,
        help="Override stratified streamline seed count.",
    )
    parser.add_argument(
        "--streamline-steps",
        type=int,
        default=None,
        help="Override RK4 step count.",
    )
    parser.add_argument(
        "--cell-proximity-voxels",
        type=float,
        default=None,
        help=(
            "Truncate streamlines once the head is more than this many voxels "
            "away from the nearest cell (default: preset value, 2.5–3.0)."
        ),
    )
    return parser.parse_args(argv)


def apply_overrides(
    preset: StreamlinePreset, args: argparse.Namespace
) -> StreamlinePreset:
    updates: dict[str, Any] = {}
    if args.grid_size is not None:
        if args.grid_size < 8:
            raise ValueError("--grid-size must be at least 8")
        updates["grid_size"] = int(args.grid_size)
    if args.streamline_seeds is not None:
        if args.streamline_seeds < 1:
            raise ValueError("--streamline-seeds must be positive")
        updates["streamline_seeds"] = int(args.streamline_seeds)
    if args.streamline_steps is not None:
        if args.streamline_steps < 1:
            raise ValueError("--streamline-steps must be positive")
        updates["streamline_steps"] = int(args.streamline_steps)
    if args.cell_proximity_voxels is not None:
        if args.cell_proximity_voxels <= 0:
            raise ValueError("--cell-proximity-voxels must be positive")
        updates["cell_proximity_voxels"] = float(args.cell_proximity_voxels)
    if not updates:
        return preset
    return replace(preset, **updates)


def main() -> None:
    args = parse_args(sys.argv[1:])
    try:
        preset = apply_overrides(PRESETS[args.preset], args)
    except ValueError as exc:
        aprint(f"❌ {exc}")
        raise SystemExit(2) from exc

    recompute_all = bool(args.recompute_all)
    recompute_field = bool(args.recompute_field or recompute_all)
    recompute_streamlines = bool(args.recompute_streamlines or recompute_all)

    aprint("=" * 72)
    aprint("Zebrahub RNA-velocity — 3D UMAP + advected streamlines")
    aprint("=" * 72)
    seed_label = (
        "all cells"
        if preset.streamline_seeds is None
        else f"{preset.streamline_seeds:,} seeds"
    )
    aprint(
        f"Preset: {preset.name} · grid {preset.grid_size}^3 · "
        f"{seed_label} · {preset.streamline_steps} RK4 steps"
    )
    aprint("Velocity source: obsm['velocity_umap'] (3D RNA-velocity embedding)")
    aprint("")

    cache_dir = args.cache_dir.expanduser()

    if args.output is not None:
        output_path = args.output.expanduser()
        data, _flow, streamlines = generate_zebrahub_scene(
            output_path,
            cache_dir,
            preset,
            args.h5ad,
            recompute_field,
            recompute_streamlines,
        )
        aprint(
            f"Generated {len(data.positions):,} cells, "
            f"{streamlines.streamline_count:,} streamlines → {output_path}"
        )
        if not args.no_serve:
            launch_viewer(output_path)
        return

    if args.no_serve:
        output_path = (
            get_demos_output_dir() / f"zebrahub_velocity_streamlines_{preset.name}.zarr"
        )
        data, _flow, streamlines = generate_zebrahub_scene(
            output_path,
            cache_dir,
            preset,
            args.h5ad,
            recompute_field,
            recompute_streamlines,
        )
        aprint(
            f"Dataset generated at {output_path} "
            f"({preset.grid_size}^3 field, {streamlines.streamline_count:,} streamlines)"
        )
        return

    with tempfile.TemporaryDirectory(prefix="luxar_zebrahub_velocity_") as tmpdir:
        output_path = Path(tmpdir) / f"zebrahub_velocity_streamlines_{preset.name}.zarr"
        data, _flow, streamlines = generate_zebrahub_scene(
            output_path,
            cache_dir,
            preset,
            args.h5ad,
            recompute_field,
            recompute_streamlines,
        )

        aprint("")
        aprint("=" * 72)
        aprint("NAVIGATION")
        aprint("=" * 72)
        aprint("  Cells: colored by zebrafish_anatomy_ontology_class")
        aprint("  Streamlines: forward RK4 advection through the velocity field")
        aprint("  Hover cells for anatomy / developmental stage labels")
        aprint(
            f"  {len(data.positions):,} cells · "
            f"{streamlines.streamline_count:,} streamlines · "
            f"{len(data.anatomy_categories)} anatomy classes"
        )
        aprint("")
        launch_viewer(output_path)

    aprint("Cleanup complete")


if __name__ == "__main__":
    main()
