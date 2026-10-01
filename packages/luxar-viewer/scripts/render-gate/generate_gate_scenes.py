"""Generate the small, deterministic scenes the render gate captures.

The render gate (``run-gate.mjs``, see ``docs/guides/developer/RENDER_GATE.md``)
compares a baseline build against a candidate build pixel by pixel. Seeded
synthetic scenes cover the bulk geometry; these stores cover what the synthetic
injector cannot: picking (injected nodes register no pick material), mixed
geometry types in one frame, the scene-captured environment, glass refraction,
a spatial partition, an LOD ladder under a rotated parent, and a nano-scale
orthographic line scene.

The perf-audit stores extend that: time- and space-partitioned gsplat trees,
LOD and laddered timelapses, ``array_ref`` dedup, ZIP_STORED copies, and a mesh
LOD beside a splat LOD. Their recommended poses (and the hidden time axis a
playback case steps) go to ``<out>/POSES.json``. A ``--heavy`` set of large
stores (thousands of parts, 10M splats, an h2afva-scale timelapse) is written
only on request, never by a default run.

Every scene is seeded, so two runs write the same data. Usage::

    python generate_gate_scenes.py --out <repo>/datasets/gate
    python generate_gate_scenes.py --out <repo>/datasets/gate --heavy
    python generate_gate_scenes.py --out <repo>/datasets/gate --only sp64 tp50
"""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.transforms import rotate_y
from luxar.core.viewer_config import CameraConfig, EnvironmentConfig, ViewerConfig
from luxar.encoding import EncodingMode
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.io.save_gsplats import write_gsplats_tree
from luxar.gsplats.lod.additive import make_additive_lod
from luxar.gsplats.lod.recipes import RecipeParams, build_recipe
from luxar.gsplats.tree import GSplatNode, GSplatPartition
from luxar.mesh.primitives import icosphere

#: The default set, in the order it is written. `gate-scenes.json` refers to
#: these. The ``zip_*`` copies come after the stores they pack.
SCENE_NAMES = (
    "mixed",
    "env_splats",
    "partition_normal",
    "glass",
    "lod_ladder",
    "tiny_units_ortho",
    "tp50",
    "sp64",
    "sp64_closeup_authored",
    "lod_timelapse",
    "pl_timelapse",
    "arrayref_4d",
    "zip_mixed",
    "zip_lod_ladder",
    "mesh_gsplat_lod",
)

#: Large, slow stores written only with ``--heavy`` (never by a default run).
HEAVY_SCENE_NAMES = (
    "tp2000",
    "sp500",
    "sp2000",
    "adaptive512",
    "normal_gsplats_10m",
    "timelapse51",
)

ALL_SCENE_NAMES = SCENE_NAMES + HEAVY_SCENE_NAMES


def _clustered(
    rng: np.random.Generator, count: int, spread: float, clusters: int = 12
) -> np.ndarray:
    """Points drawn from a few gaussian blobs inside ``[-spread, spread]^3``."""
    centers = rng.uniform(-spread, spread, size=(clusters, 3))
    which = rng.integers(0, clusters, size=count)
    jitter = rng.normal(scale=spread * 0.12, size=(count, 3))
    return (centers[which] + jitter).astype(np.float32)


def _splat_attrs(
    rng: np.random.Generator, count: int, sigma: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Amplitudes, packed lower-triangular Cholesky factors and colours."""
    amplitudes = rng.uniform(0.3, 1.0, size=count).astype(np.float32)
    diag = rng.uniform(0.5, 1.5, size=(count, 3)) * sigma
    off = rng.normal(scale=0.3 * sigma, size=(count, 3))
    # Packed lower triangle, row-major: L00, L10, L11, L20, L21, L22.
    chol = np.stack(
        [diag[:, 0], off[:, 0], diag[:, 1], off[:, 1], off[:, 2], diag[:, 2]], axis=1
    ).astype(np.float32)
    colors = rng.uniform(0.2, 1.0, size=(count, 3)).astype(np.float32)
    return amplitudes, chol, colors


def _polylines(
    rng: np.random.Generator, n_lines: int, n_vertices: int, spread: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenated random-walk polylines with their per-vertex widths and line indices."""
    steps = rng.normal(scale=spread * 0.04, size=(n_lines, n_vertices, 3))
    starts = rng.uniform(-spread, spread, size=(n_lines, 1, 3))
    vertices = (starts + np.cumsum(steps, axis=1)).reshape(-1, 3).astype(np.float32)
    widths = np.full(len(vertices), spread * 0.01, dtype=np.float32)
    indices = np.repeat(np.arange(n_lines, dtype=np.int32), n_vertices)
    return vertices, widths, indices


#: Picking only initialises for a scene where some node declares labels, label
#: ids, keys or interactions, so every pickable gate node carries a few.
_VOCAB = {i: f"class {i}" for i in range(8)}


def _labels(count: int) -> list[str]:
    return [_VOCAB[i % len(_VOCAB)] for i in range(count)]


def _label_ids(count: int) -> np.ndarray:
    return (np.arange(count) % len(_VOCAB)).astype(np.int32)


def _camera(position: tuple[float, float, float]) -> CameraConfig:
    return CameraConfig(position=position, target=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0))


def write_mixed(path: Path, *, small: bool = False) -> None:
    """Points, lines, splats and a mesh in one frame, all additive, all pickable."""
    rng = np.random.default_rng(1)
    vertices, faces, normals = icosphere(3, radius=1.5)
    with LuxarZarrCompiler(path, encoding_mode=EncodingMode.PRECISION) as compiler:
        scene = compiler.create_scene(
            dimensions=Dimensions.default_3d(),
            viewer_config=ViewerConfig(camera=_camera((14.0, 9.0, 16.0))),
        )
        positions = _clustered(rng, 20_000, 6.0)
        scene.add_points(
            "points",
            positions,
            colors=rng.uniform(0.2, 1.0, size=(len(positions), 3)).astype(np.float32),
            radii=rng.uniform(0.02, 0.12, size=len(positions)).astype(np.float32),
            labels=_labels(len(positions)),
            blending_mode="additive",
            layer=True,
        )
        line_vertices, widths, indices = _polylines(rng, 60, 80, 6.0)
        scene.add_lines(
            "lines",
            line_vertices,
            widths,
            indices=indices,
            labels=_labels(len(line_vertices)),
            blending_mode="additive",
            layer=True,
        )
        centers = _clustered(rng, 20_000, 6.0)
        amplitudes, chol, colors = _splat_attrs(rng, len(centers), 0.08)
        scene.add_gsplats(
            "splats",
            centers=centers,
            amplitudes=amplitudes,
            cholesky_factors=chol,
            colors=colors,
            label_ids=_label_ids(len(centers)),
            label_vocabulary=_VOCAB,
            blending_mode="additive",
            layer=True,
        )
        scene.add_mesh(
            "sphere",
            vertices,
            faces,
            normals=normals,
            normal_dims=[0, 1, 2],
            labels=_labels(len(vertices)),
            shading="smooth",
            layer=True,
        )


def write_env_splats(path: Path, *, small: bool = False) -> None:
    """Splats beside a chrome sphere lit by a capture of the scene itself.

    The reflection is where the pre-PR-1 cube capture mirrors splats: points
    project through three's (flipped) cube-face projection matrix, splats through
    a CPU-pushed focal length that carries no flip.
    """
    rng = np.random.default_rng(2)
    vertices, faces, normals = icosphere(4, radius=1.0)
    with LuxarZarrCompiler(path, encoding_mode=EncodingMode.PRECISION) as compiler:
        scene = compiler.create_scene(
            dimensions=Dimensions.default_3d(),
            viewer_config=ViewerConfig(
                environment=EnvironmentConfig(source="scene", probe="auto"),
                camera=_camera((0.0, 1.5, 6.5)),
            ),
        )
        # An asymmetric arrangement: one bright splat cluster on +x, one point
        # cluster on -x, so a mirrored reflection is unmistakable.
        centers = (rng.normal(scale=0.35, size=(3_000, 3)) + [3.5, 0.8, -1.0]).astype(
            np.float32
        )
        amplitudes, chol, colors = _splat_attrs(rng, len(centers), 0.06)
        scene.add_gsplats(
            "splats",
            centers=centers,
            amplitudes=amplitudes * 2.0,
            cholesky_factors=chol,
            colors=colors,
            blending_mode="additive",
            layer=True,
        )
        points = (rng.normal(scale=0.35, size=(3_000, 3)) + [-3.5, -0.6, -1.0]).astype(
            np.float32
        )
        scene.add_points(
            "points",
            points,
            colors=np.tile(
                np.array([[0.2, 0.8, 1.0]], dtype=np.float32), (len(points), 1)
            ),
            radii=np.full(len(points), 0.05, dtype=np.float32),
            blending_mode="additive",
            layer=True,
        )
        scene.add_mesh(
            "chrome",
            vertices,
            faces,
            normals=normals,
            normal_dims=[0, 1, 2],
            colors=np.tile(
                np.array([[0.95, 0.95, 0.97]], dtype=np.float32), (len(vertices), 1)
            ),
            shading="smooth",
            material="physical",
            metalness=1.0,
            roughness=0.05,
            layer=True,
        )


def write_partition_normal(path: Path, *, small: bool = False) -> None:
    """A spatially partitioned splat node in order-dependent ``normal`` blending."""
    rng = np.random.default_rng(3)
    centers = _clustered(rng, 60_000, 8.0, clusters=24)
    amplitudes, chol, colors = _splat_attrs(rng, len(centers), 0.15)
    with LuxarZarrCompiler(path, encoding_mode=EncodingMode.PRECISION) as compiler:
        scene = compiler.create_scene(
            dimensions=Dimensions.default_3d(),
            viewer_config=ViewerConfig(camera=_camera((18.0, 10.0, 20.0))),
        )
        scene.add_gsplats(
            "splats",
            centers=centers,
            amplitudes=amplitudes,
            cholesky_factors=chol,
            colors=colors,
            label_ids=_label_ids(len(centers)),
            label_vocabulary=_VOCAB,
            blending_mode="normal",
            partition={"max_elements": 8_000},
            layer=True,
        )


def write_glass(path: Path, *, small: bool = False) -> None:
    """A ``refract_data`` glass sphere in front of an additive point lattice."""
    vertices, faces, normals = icosphere(4, radius=1.2)
    grid = np.linspace(-4.0, 4.0, 17, dtype=np.float32)
    gx, gy, gz = np.meshgrid(grid, grid, grid - 3.0, indexing="ij")
    lattice = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)
    with LuxarZarrCompiler(path, encoding_mode=EncodingMode.PRECISION) as compiler:
        scene = compiler.create_scene(
            dimensions=Dimensions.default_3d(),
            viewer_config=ViewerConfig(camera=_camera((0.0, 1.0, 9.0))),
        )
        scene.add_points(
            "lattice",
            lattice,
            colors=np.tile(
                np.array([[1.0, 0.7, 0.3]], dtype=np.float32), (len(lattice), 1)
            ),
            radii=np.full(len(lattice), 0.06, dtype=np.float32),
            blending_mode="additive",
            layer=True,
        )
        scene.add_mesh(
            "lens",
            vertices + np.array([0.0, 0.0, 2.0], dtype=np.float32),
            faces,
            normals=normals,
            normal_dims=[0, 1, 2],
            colors=np.ones((len(vertices), 3), dtype=np.float32),
            shading="smooth",
            material="physical",
            roughness=0.02,
            transmission=1.0,
            ior=1.5,
            thickness=2.4,
            refract_data=True,
            layer=True,
        )


def write_lod_ladder(path: Path, *, small: bool = False) -> None:
    """A substitutive points LOD ladder, one copy under a 45-degree-rotated parent.

    The rotated copy is what PR 3's projected-corner LOD box changes: a re-boxed
    world AABB of a rotated node reads larger on screen than the node itself.
    """
    rng = np.random.default_rng(5)
    positions = (rng.uniform(-1.0, 1.0, size=(40_000, 3)) * [6.0, 1.0, 1.0]).astype(
        np.float32
    )
    colors = rng.uniform(0.3, 1.0, size=(len(positions), 3)).astype(np.float32)
    with LuxarZarrCompiler(path, encoding_mode=EncodingMode.PRECISION) as compiler:
        scene = compiler.create_scene(
            dimensions=Dimensions.default_3d(),
            viewer_config=ViewerConfig(camera=_camera((0.0, 6.0, 22.0))),
        )
        # The axis-aligned copy sits above the rotated one so both are in frame.
        lift = np.array([0.0, 4.0, 0.0], dtype=np.float32)
        copies = (
            (scene.add_group("axis_aligned"), positions + lift),
            (scene.add_group("rotated", transform=rotate_y(45.0)), positions),
        )
        for group, cloud in copies:
            group.add_points(
                "cloud",
                cloud,
                colors=colors,
                radii=np.full(len(positions), 0.04, dtype=np.float32),
                blending_mode="additive",
                substitutive_lod=dict(compression_factor=4, levels=3, device="cpu"),
            )


def write_tiny_units_ortho(path: Path, *, small: bool = False) -> None:
    """Nanometre-scale lines, to be viewed orthographically.

    Below a world frustum height of 1e-4 the pre-PR-1 ortho line scale clamps
    its frustum height, so widths come out too narrow; deriving the scale from
    the projection matrix has no such clamp.
    """
    rng = np.random.default_rng(6)
    vertices, widths, indices = _polylines(rng, 40, 60, 2e-5)
    with LuxarZarrCompiler(path, encoding_mode=EncodingMode.PRECISION) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_lines(
            "tiny_lines",
            vertices,
            widths,
            indices=indices,
            blending_mode="additive",
            layer=True,
        )


# ---------------------------------------------------------------------------
# Perf-audit stores: partitions, timelapses, LOD, dedup, zipped copies.
#
# Writers from here on return a POSES.json entry (or ``None``): the recommended
# camera poses, in the manifest's own pose shape, and the hidden time axis a
# playback / scrub case steps. ``main`` merges them into ``<out>/POSES.json``.
# ---------------------------------------------------------------------------

#: A POSES.json entry: ``{"store", "poses": {name: pose}, "time"?, "notes"?}``.
PoseEntry = dict[str, Any]

#: Splats per part in the partition stores (the perf audit's measure-trees shape).
_SPLATS_PER_PART = 256
#: Edge length of one spatial tile in the partition stores.
_CELL = 10.0
#: Spatial tile grids, keyed by tile count.
_GRIDS: dict[int, tuple[int, int, int]] = {
    4: (2, 2, 1),
    40: (5, 4, 2),
    64: (4, 4, 4),
    500: (10, 10, 5),
    2000: (20, 10, 10),
}
#: A pose inside the corner cell ``[0, CELL]^3`` looking -z: ~1 part in view.
_CLOSEUP = {
    "position": [3.0, 3.0, 8.0],
    "target": [3.0, 3.0, 0.0],
    "up": [0.0, 1.0, 0.0],
    "fov": 40.0,
}


def _pose(
    position: tuple[float, float, float] | np.ndarray,
    target: tuple[float, float, float] | np.ndarray,
) -> dict[str, Any]:
    """A pose in ``gate-scenes.json``'s shape."""
    return {
        "position": [round(float(v), 3) for v in position],
        "target": [round(float(v), 3) for v in target],
        "up": [0.0, 1.0, 0.0],
    }


def _viewer_config(pose: dict[str, Any]) -> ViewerConfig:
    """Author ``pose`` as the store's opening camera, with explicit ACES."""
    camera = CameraConfig(
        position=tuple(pose["position"]),
        target=tuple(pose["target"]),
        up=tuple(pose["up"]),
        fov=pose.get("fov"),
    )
    return ViewerConfig(camera=camera, tone_mapping="ACES")


def _full_view(extent: np.ndarray) -> dict[str, Any]:
    """A three-quarter pose framing the box ``[0, extent]`` whole."""
    center = extent / 2.0
    radius = float(np.linalg.norm(extent)) / 2.0
    return _pose(center + np.array([1.1, 0.8, 1.3]) * radius * 1.4, center)


def _time_axis(index: int, name: str, count: int) -> dict[str, Any]:
    """POSES.json's description of a hidden, discrete, step-1 time dimension."""
    return {
        "dimIndex": index,
        "name": name,
        "range": [0.0, float(count - 1)],
        "step": 1.0,
        "count": count,
        "discrete": True,
    }


def _time_dimension(name: str, count: int) -> Dimension:
    """A hidden time axis: discrete, step 1.0 (binary-exact), frames ``0..count-1``."""
    return Dimension(
        name,
        unit="frame",
        display=False,
        discrete=True,
        step=1.0,
        range=(0.0, float(count - 1)),
    )


def _spatial_dimensions(extent: np.ndarray) -> list[Dimension]:
    return [
        Dimension(axis, unit="um", display=True, range=(0.0, float(size)))
        for axis, size in zip(("X", "Y", "Z"), extent, strict=True)
    ]


def _part_splats(
    rng: np.random.Generator,
    lo: np.ndarray,
    hi: np.ndarray,
    count: int,
    t: float | None = None,
) -> GSplatData:
    """``count`` axis-aligned splats uniform in ``[lo, hi]``, optionally at time ``t``.

    With ``t`` the time is the LAST centre column (a barrier dim) and its packed
    Cholesky diagonal entry (index 9 of the 4-D lower triangle) is 0.3 frames.
    """
    centers = rng.uniform(lo, hi, size=(count, 3))
    sigma = rng.uniform(0.2, 0.5, size=(count, 3))
    zero = np.zeros(count)
    columns = [sigma[:, 0], zero, sigma[:, 1], zero, zero, sigma[:, 2]]
    if t is not None:
        centers = np.concatenate([centers, np.full((count, 1), float(t))], axis=1)
        columns += [zero, zero, zero, np.full(count, 0.3)]
    return GSplatData(
        centers=centers.astype(np.float32),
        amplitudes=rng.uniform(0.3, 1.0, size=count).astype(np.float32),
        cholesky_factors=np.stack(columns, axis=1).astype(np.float32),
        colors=rng.uniform(0.2, 1.0, size=(count, 3)).astype(np.float32),
    )


def _laddered(data: GSplatData) -> GSplatNode:
    """``data`` as a leaf with a 4-rung ``self_energy`` additive ladder."""
    return make_additive_lod(data, n_lods=4, method="self_energy").tree


def _tile_parts(
    rng: np.random.Generator,
    grid: tuple[int, int, int],
    per_part: int,
    times: list[float] | None = None,
) -> list[GSplatNode]:
    """One laddered part per (timepoint, tile), time-major."""
    parts: list[GSplatNode] = []
    for t in times if times is not None else [None]:
        for index in np.ndindex(*grid):
            lo = np.array(index, dtype=np.float64) * _CELL
            parts.append(_laddered(_part_splats(rng, lo, lo + _CELL, per_part, t=t)))
    return parts


def _write_tree_scene(
    path: Path,
    tree: GSplatNode,
    dims: list[Dimension],
    pose: dict[str, Any],
    *,
    barrier_dims: list[int] | None = None,
    blending_mode: str = "additive",
) -> None:
    """Graft the gsplat ``tree`` into a one-node scene at ``path``.

    The tree is staged as a ``.gsplats.zarr`` in a temporary directory that
    outlives the compile, since the graft may read it lazily.
    """
    with tempfile.TemporaryDirectory(prefix="gate-tree-") as staging:
        staged = Path(staging) / "tree.gsplats.zarr"
        write_gsplats_tree(staged, tree, barrier_dims=barrier_dims)
        with LuxarZarrCompiler(path) as compiler:
            scene = compiler.create_scene(
                dimensions=Dimensions(dims), viewer_config=_viewer_config(pose)
            )
            scene.add_gsplats_from_file(
                name="splats",
                path=str(staged),
                blending_mode=blending_mode,
                layer=True,
            )


def _write_partition_scene(
    path: Path,
    parts: list[GSplatNode],
    extent: np.ndarray,
    pose: dict[str, Any],
    *,
    t_count: int | None = None,
    max_elements: int = _SPLATS_PER_PART,
) -> None:
    """A ``kind=partition`` of ``parts``; with ``t_count`` the last column is time.

    The time column is then a hidden scene dimension AND a coarsening barrier.
    """
    dims = _spatial_dimensions(extent)
    if t_count is not None:
        dims.append(_time_dimension("Time", t_count))
    _write_tree_scene(
        path,
        GSplatPartition(children=parts, max_elements=max_elements),
        dims,
        pose,
        barrier_dims=[3] if t_count is not None else None,
    )


def _store_ref(name: str) -> str:
    return f"gate/{store_filename(name)}"


def _time_partition(
    path: Path, name: str, *, t_count: int, tiles: int, per_part: int, seed: int
) -> PoseEntry:
    """Shared body of ``tp50`` / ``tp2000``: ``t_count`` x ``tiles`` laddered parts."""
    grid = _GRIDS[tiles]
    extent = np.array(grid, dtype=np.float64) * _CELL
    times = [float(t) for t in range(t_count)]
    parts = _tile_parts(np.random.default_rng(seed), grid, per_part, times)
    full = _full_view(extent)
    _write_partition_scene(path, parts, extent, full, t_count=t_count)
    return {
        "store": _store_ref(name),
        "poses": {"full": full, "closeup": dict(_CLOSEUP)},
        "time": _time_axis(3, "Time", t_count),
        "notes": (
            f"{t_count} timepoints x {tiles} tiles = {len(parts)} parts of "
            f"{per_part} splats, each bounded to one timepoint, 4 additive rungs."
        ),
    }


def _spatial_partition(
    path: Path,
    name: str,
    tiles: int,
    small: bool,
    *,
    opening: dict[str, Any] | None = None,
) -> PoseEntry:
    """Shared body of ``sp64`` / ``sp500`` / ``sp2000``: one laddered part per tile.

    The data depends only on ``tiles`` (it seeds the generator), so a variant
    that changes ``opening`` (the authored opening camera, default ``full``)
    holds the same parts as its base store.
    """
    grid = _GRIDS[tiles]
    extent = np.array(grid, dtype=np.float64) * _CELL
    per_part = 32 if small else _SPLATS_PER_PART
    parts = _tile_parts(np.random.default_rng(tiles), grid, per_part)
    full = _full_view(extent)
    _write_partition_scene(path, parts, extent, opening or full)
    return {
        "store": _store_ref(name),
        "poses": {"full": full, "closeup": dict(_CLOSEUP)},
        "notes": (
            f"{len(parts)} parts of {per_part} splats on a {grid} grid of "
            f"{_CELL:g}-unit cells up to {extent.tolist()}; 'full' frames every "
            "part, 'closeup' sits inside cell [0,10]^3 looking -z (fov 40) so "
            "~1 part is in the frustum."
        ),
    }


def write_tp50(path: Path, *, small: bool = False) -> PoseEntry:
    """Time-partitioned gsplats: 50 timepoints x 4 tiles (200 parts x 256 splats).

    Every part holds ONE timepoint (the time column is a barrier dim, so a part's
    bounds span a single frame) and a 4-rung additive ladder. The hidden ``Time``
    axis is scene dimension 3. Gate cases: time-slice stepping / playback over a
    time-partitioned tree (only the parts at the current frame should load and
    draw) and per-part additive refinement under a slice change.
    """
    return _time_partition(
        path,
        "tp50",
        t_count=4 if small else 50,
        tiles=4,
        per_part=32 if small else _SPLATS_PER_PART,
        seed=50,
    )


def write_sp64(path: Path, *, small: bool = False) -> PoseEntry:
    """Spatial gsplat partition: 64 tiles (4x4x4 grid of 10-unit cells) x 256 splats.

    Each part carries a 4-rung additive ladder. POSES.json lists ``full`` (all 64
    parts framed, also the authored opening camera) and ``closeup`` (inside the
    corner cell looking -z, ~1 part in the frustum). Gate cases: partition
    frustum culling (closeup vs full request / draw counts) and partition
    refinement as the camera moves between the two poses.
    """
    return _spatial_partition(path, "sp64", 64, small)


def write_sp64_closeup_authored(path: Path, *, small: bool = False) -> PoseEntry:
    """``sp64``'s data with the CLOSE-UP as its authored opening camera.

    ``viewer_config.camera`` is the ``closeup`` pose (inside the corner cell
    looking -z, fov 40). Gate case: the opening camera is framed from the
    store BEFORE any node loads, so a cold load initialises and fetches the
    ~1 part in the frustum instead of all 64. ``cold_sp64_closeup`` cannot
    show that: its pose is applied by the harness after navigation, so its
    initial load still sees the default camera.
    """
    return _spatial_partition(
        path, "sp64_closeup_authored", 64, small, opening=dict(_CLOSEUP)
    )


def write_lod_timelapse(path: Path, *, small: bool = False) -> PoseEntry:
    """A 4D gsplat timelapse with a substitutive LOD group (40 frames x 20k splats).

    Drifting blobs, time as scene dimension 0 (hidden, discrete). The
    ``kind=lod`` group (compression 4, 2 coarse levels) coarsens only the
    displayed dims, grouped per frame, so the fine levels are LAZY: fetched only
    once the selector upgrades. Gate cases: playback over a substitutive LOD
    timelapse (lazy fine levels per frame; last-frame / cadence checks).
    """
    t_count, per_t = (4, 2_000) if small else (40, 20_000)
    rng = np.random.default_rng(40)
    base = rng.uniform(-40.0, 40.0, size=(per_t, 3))
    velocity = rng.normal(0.0, 0.3, size=(per_t, 3))
    centers = np.concatenate(
        [
            np.column_stack([np.full(per_t, float(t)), base + velocity * t])
            for t in range(t_count)
        ]
    ).astype(np.float32)
    total = len(centers)
    sigma = rng.uniform(0.4, 1.2, size=total).astype(np.float32)
    chol = np.zeros((total, 10), np.float32)
    # Packed lower triangle: (0,0) is time; (1,1), (2,2), (3,3) are x, y, z.
    chol[:, 0] = 0.15
    chol[:, 2] = sigma
    chol[:, 5] = sigma
    chol[:, 9] = sigma
    dims = Dimensions(
        [
            _time_dimension("t", t_count),
            Dimension("x", unit="um", range=(-60.0, 60.0)),
            Dimension("y", unit="um", range=(-60.0, 60.0)),
            Dimension("z", unit="um", range=(-60.0, 60.0)),
        ]
    )
    full = _pose((90.0, 65.0, 105.0), (0.0, 0.0, 0.0))
    with LuxarZarrCompiler(path) as compiler:
        scene = compiler.create_scene(
            dimensions=dims, viewer_config=_viewer_config(full)
        )
        scene.add_gsplats(
            "blobs",
            centers,
            rng.uniform(0.3, 1.0, size=total).astype(np.float32),
            chol,
            colors=rng.uniform(0.2, 1.0, size=(total, 3)).astype(np.float32),
            substitutive_lod=dict(compression_factor=4, levels=2, device="cpu"),
            blending_mode="additive",
            layer=True,
        )
    return {
        "store": _store_ref("lod_timelapse"),
        "poses": {"full": full},
        "time": _time_axis(0, "t", t_count),
        "notes": f"{t_count} frames x {per_t} splats; kind=lod, K=4, 2 coarse levels.",
    }


def write_pl_timelapse(path: Path, *, small: bool = False) -> PoseEntry:
    """Points + segment lines, 60 frames, per-frame counts varying ~10x, additive LOD 4.

    Per-frame point counts swing ~3k..30k on two sine periods (a tenth of the
    perf audit's 30k..300k, to keep the store small); lines are 8-vertex
    polylines stored as segments with a ``spatial-uniform`` ladder. ``frame`` is
    scene dimension 3. Gate cases: playback / scrub cadence and the chunk cache
    under strongly varying per-frame loads, over laddered points AND lines.
    """
    n_frames, peak = (4, 3_000) if small else (60, 27_000)
    rng = np.random.default_rng(60)
    phase = np.arange(n_frames) / n_frames * 4 * np.pi
    counts = (peak // 9 + peak * (0.5 + 0.5 * np.sin(phase))).astype(int)
    positions = np.concatenate(
        [
            np.column_stack([rng.normal(size=(c, 3)), np.full(c, float(f))])
            for f, c in enumerate(counts)
        ]
    ).astype(np.float32)
    segments = []
    for f, c in enumerate(counts):
        n_lines = max(50, c // 100)
        start = rng.normal(size=(n_lines, 1, 3))
        walk = start + np.cumsum(rng.normal(scale=0.05, size=(n_lines, 8, 3)), axis=1)
        pairs = np.stack([walk[:, :-1], walk[:, 1:]], axis=2).reshape(-1, 3)
        segments.append(np.column_stack([pairs, np.full(len(pairs), float(f))]))
    vertices = np.concatenate(segments).astype(np.float32)
    dims = Dimensions(
        [
            Dimension("x", range=(-4.0, 4.0), display=True),
            Dimension("y", range=(-4.0, 4.0), display=True),
            Dimension("z", range=(-4.0, 4.0), display=True),
            _time_dimension("frame", n_frames),
        ]
    )
    full = _pose((6.0, 4.5, 7.0), (0.0, 0.0, 0.0))
    with LuxarZarrCompiler(path) as compiler:
        scene = compiler.create_scene(
            dimensions=dims, viewer_config=_viewer_config(full)
        )
        scene.add_points(
            "pts",
            positions=positions,
            colors=rng.integers(0, 255, size=(len(positions), 3), dtype=np.uint8),
            radii=np.full(len(positions), 0.01, np.float32),
            additive_lod=dict(n_lods=4),
            layer=True,
        )
        scene.add_lines(
            "lns",
            vertices=vertices,
            widths=0.01,
            line_type="segments",
            additive_lod=dict(method="spatial-uniform", n_lods=4),
            layer=True,
        )
    return {
        "store": _store_ref("pl_timelapse"),
        "poses": {"full": full},
        "time": _time_axis(3, "frame", n_frames),
        "notes": (
            f"{len(positions)} points / {len(vertices)} segment vertices over "
            f"{n_frames} frames; per-frame points {counts.min()}..{counts.max()}."
        ),
    }


def write_arrayref_4d(path: Path, *, small: bool = False) -> PoseEntry:
    """Two 4D point nodes whose identical arrays dedupe to ``array_ref`` (20 x 50k).

    Nodes ``A`` and ``B`` carry the same positions and scalars, so the compiler
    writes each array once and points the second node at it. ``t`` is scene
    dimension 0. Gate cases: the chunk cache sharing one fetch between two
    nodes referencing the same array, under time scrubbing.
    """
    t_count, per_t = (4, 2_000) if small else (20, 50_000)
    rng = np.random.default_rng(20)
    total = t_count * per_t
    times = np.repeat(np.arange(t_count), per_t).astype(np.float32)
    positions = np.column_stack([times, rng.normal(size=(total, 3)) * 5]).astype(
        np.float32
    )
    scalars = rng.random(total).astype(np.float32)
    dims = Dimensions(
        [
            _time_dimension("t", t_count),
            Dimension("x", unit="units", range=(-20.0, 20.0)),
            Dimension("y", unit="units", range=(-20.0, 20.0)),
            Dimension("z", unit="units", range=(-20.0, 20.0)),
        ]
    )
    full = _pose((30.0, 22.0, 35.0), (0.0, 0.0, 0.0))
    with LuxarZarrCompiler(path) as compiler:
        scene = compiler.create_scene(
            dimensions=dims, viewer_config=_viewer_config(full)
        )
        for name in ("A", "B"):
            scene.add_points(
                name,
                positions.copy(),
                scalars=scalars,
                colormap="viridis",
                radii=0.05,
                layer=True,
            )
    return {
        "store": _store_ref("arrayref_4d"),
        "poses": {"full": full},
        "time": _time_axis(0, "t", t_count),
        "notes": f"{t_count} frames x {per_t} points; nodes A and B share arrays.",
    }


def _zip_store(source: Path, target: Path) -> None:
    """Pack the directory store ``source`` into a ZIP_STORED archive ``target``.

    Members are the store's relative POSIX paths in sorted order, uncompressed:
    the chunks are already compressed, and a STORED member is a plain byte range
    an HTTP reader fetches without inflating.
    """
    target.unlink(missing_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_STORED) as archive:
        for member in sorted(source.rglob("*")):
            if member.is_file():
                archive.write(member, member.relative_to(source).as_posix())


def _zip_copy(path: Path, source_name: str, small: bool) -> None:
    """Zip ``<out>/<source_name>.luxar.zarr``, writing the source first if absent."""
    source = path.parent / store_filename(source_name)
    if not source.exists():
        WRITERS[source_name](source, small=small)
    _zip_store(source, path)


def write_zip_mixed(path: Path, *, small: bool = False) -> None:
    """A ZIP_STORED copy of the ``mixed`` store (``zip_mixed.luxar.zarr.zip``).

    Gate case: the zipped-store read path (central-directory parse, range reads
    of stored members) must render the same frame as the directory ``mixed``;
    reuse ``mixed``'s poses. Zips the ``mixed`` store already in ``--out``
    (writing it first when absent), so regenerate both together.
    """
    _zip_copy(path, "mixed", small)


def write_zip_lod_ladder(path: Path, *, small: bool = False) -> None:
    """A ZIP_STORED copy of ``lod_ladder`` (``zip_lod_ladder.luxar.zarr.zip``).

    Gate case: LOD selection and lazy level fetches through the zipped read
    path, against the directory ``lod_ladder``; reuse ``lod_ladder``'s poses.
    """
    _zip_copy(path, "lod_ladder", small)


def write_mesh_gsplat_lod(path: Path, *, small: bool = False) -> PoseEntry:
    """A mesh substitutive-LOD node beside a gsplat substitutive-LOD node.

    The mesh (an icosphere decimated K=4 into 2 coarse levels) and the splats
    (K=4, 2 levels) are each a ``kind=lod`` group. Gate case: the GPU byte
    budget must count MESH level bytes as well as splat level bytes (the perf
    audit's "mesh levels unaccounted" finding); compare the budget / resident
    byte counters with the mesh in and out of view.
    """
    rng = np.random.default_rng(8)
    vertices, faces, normals = icosphere(3 if small else 5, radius=2.0)
    centers = _clustered(rng, 2_000 if small else 30_000, 2.5) + np.array(
        [3.0, 0.0, 0.0], np.float32
    )
    amplitudes, chol, colors = _splat_attrs(rng, len(centers), 0.05)
    full = _pose((0.0, 5.0, 14.0), (0.0, 0.0, 0.0))
    with LuxarZarrCompiler(path) as compiler:
        scene = compiler.create_scene(
            dimensions=Dimensions.default_3d(), viewer_config=_viewer_config(full)
        )
        scene.add_mesh(
            "surface",
            vertices - np.array([3.0, 0.0, 0.0], np.float32),
            faces,
            normals=normals,
            normal_dims=[0, 1, 2],
            colors=np.tile(np.array([[0.8, 0.6, 0.4]], np.float32), (len(vertices), 1)),
            shading="smooth",
            substitutive_lod=dict(compression_factor=4, levels=2),
            layer=True,
        )
        scene.add_gsplats(
            "splats",
            centers=centers,
            amplitudes=amplitudes,
            cholesky_factors=chol,
            colors=colors,
            blending_mode="additive",
            substitutive_lod=dict(compression_factor=4, levels=2, device="cpu"),
            layer=True,
        )
    return {
        "store": _store_ref("mesh_gsplat_lod"),
        "poses": {"full": full},
        "notes": (
            f"mesh {len(vertices)} vertices / {len(faces)} faces at x=-3 and "
            f"{len(centers)} splats at x=+3, each a K=4 substitutive ladder."
        ),
    }


# --- Heavy set (``--heavy`` only) ------------------------------------------


def write_tp2000(path: Path, *, small: bool = False) -> PoseEntry:
    """HEAVY. Time partition: 50 timepoints x 40 tiles = 2000 parts x 256 splats.

    Same construction as ``tp50`` (Time = scene dim 3, one frame per part),
    ~512k splats. Gate cases: per-frame part selection and slice-step latency
    on a tree of thousands of parts.
    """
    return _time_partition(
        path,
        "tp2000",
        t_count=4 if small else 50,
        tiles=40,
        per_part=32 if small else _SPLATS_PER_PART,
        seed=2000,
    )


def write_sp500(path: Path, *, small: bool = False) -> PoseEntry:
    """HEAVY. Spatial partition, 500 parts (10x10x5) x 256 splats, 4 rungs each.

    ~128k splats. Gate cases: the ``sp64`` culling / refinement cases at scale,
    and per-frame partition traversal cost.
    """
    return _spatial_partition(path, "sp500", 500, small)


def write_sp2000(path: Path, *, small: bool = False) -> PoseEntry:
    """HEAVY. Spatial partition, 2000 parts (20x10x10) x 256 splats, 4 rungs each.

    ~512k splats. Gate cases: partition frame cost and orbit refinement with
    thousands of parts.
    """
    return _spatial_partition(path, "sp2000", 2000, small)


def write_adaptive512(path: Path, *, small: bool = False) -> PoseEntry:
    """HEAVY. ``adaptive`` recipe: 512 BSP tiles, each its own 2-level lod group.

    512k uniform splats over 100x100x50, ``max_elements`` 1000 per tile, K=4, 2
    levels (``kmeans_lloyd``), 3-rung ladders. Gate cases: per-tile LOD selection
    churn and crossfades across many independent lod groups.
    """
    tiles, per_tile = (8, 200) if small else (512, 1_000)
    rng = np.random.default_rng(512)
    count = tiles * per_tile
    extent = np.array([100.0, 100.0, 50.0])
    data = _part_splats(rng, np.zeros(3), extent, count)
    params = RecipeParams(
        max_elements=per_tile,
        compression_factor=4,
        levels=2,
        substitutive_method="kmeans_lloyd",
        n_lods=3,
        device="cpu",
    )
    full = _full_view(extent)
    tree = build_recipe(data, "adaptive", params)
    _write_tree_scene(path, tree, _spatial_dimensions(extent), full)
    return {
        "store": _store_ref("adaptive512"),
        "poses": {"full": full},
        "notes": f"{count} splats, adaptive recipe, {per_tile} per tile, K=4, L=2.",
    }


def write_normal_gsplats_10m(path: Path, *, small: bool = False) -> PoseEntry:
    """HEAVY. 10M gsplats in ONE node, ``normal`` (order-dependent) blending.

    A single depth-sorted node (no partition) far above the sort density guard.
    Gate cases: the density-guard / sorted-node path, i.e. sort cost and the
    guard's behaviour on a 10M-splat normal-blended node. Several hundred MB.
    """
    rng = np.random.default_rng(10)
    centers = _clustered(rng, 20_000 if small else 10_000_000, 20.0, clusters=64)
    amplitudes, chol, colors = _splat_attrs(rng, len(centers), 0.05)
    full = _pose((45.0, 30.0, 52.0), (0.0, 0.0, 0.0))
    with LuxarZarrCompiler(path) as compiler:
        scene = compiler.create_scene(
            dimensions=Dimensions.default_3d(), viewer_config=_viewer_config(full)
        )
        scene.add_gsplats(
            "splats",
            centers=centers,
            amplitudes=amplitudes,
            cholesky_factors=chol,
            colors=colors,
            blending_mode="normal",
            layer=True,
        )
    return {
        "store": _store_ref("normal_gsplats_10m"),
        "poses": {"full": full},
        "notes": f"{len(centers)} splats, one node, normal blending.",
    }


def write_timelapse51(path: Path, *, small: bool = False) -> PoseEntry:
    """HEAVY. h2afva-like: 51 time parts x 100k splats, 4 additive rungs each.

    One part per timepoint (time = last centre column, a barrier; scene dim 3),
    every frame the same 100x100x50 region with fresh splats: ~5.1M splats.
    Gate cases: timelapse playback / scrub at h2afva scale (one ~100k ladder per
    frame). Estimated ~100-200 MB on disk (not measured: the small-part stores
    run ~290 B/splat of per-part overhead, which 100k-splat parts amortize).
    """
    t_count, per_t = (4, 2_000) if small else (51, 100_000)
    rng = np.random.default_rng(51)
    extent = np.array([100.0, 100.0, 50.0])
    parts = [
        _laddered(_part_splats(rng, np.zeros(3), extent, per_t, t=float(t)))
        for t in range(t_count)
    ]
    full = _full_view(extent)
    _write_partition_scene(
        path, parts, extent, full, t_count=t_count, max_elements=per_t
    )
    return {
        "store": _store_ref("timelapse51"),
        "poses": {"full": full},
        "time": _time_axis(3, "Time", t_count),
        "notes": f"{t_count} time parts x {per_t} splats, 4 additive rungs each.",
    }


Writer = Callable[..., "PoseEntry | None"]

WRITERS: dict[str, Writer] = {
    "mixed": write_mixed,
    "env_splats": write_env_splats,
    "partition_normal": write_partition_normal,
    "glass": write_glass,
    "lod_ladder": write_lod_ladder,
    "tiny_units_ortho": write_tiny_units_ortho,
    "tp50": write_tp50,
    "sp64": write_sp64,
    "sp64_closeup_authored": write_sp64_closeup_authored,
    "lod_timelapse": write_lod_timelapse,
    "pl_timelapse": write_pl_timelapse,
    "arrayref_4d": write_arrayref_4d,
    "zip_mixed": write_zip_mixed,
    "zip_lod_ladder": write_zip_lod_ladder,
    "mesh_gsplat_lod": write_mesh_gsplat_lod,
    "tp2000": write_tp2000,
    "sp500": write_sp500,
    "sp2000": write_sp2000,
    "adaptive512": write_adaptive512,
    "normal_gsplats_10m": write_normal_gsplats_10m,
    "timelapse51": write_timelapse51,
}

#: Scenes that ship as a ZIP_STORED archive rather than a directory store.
ZIP_SCENES = frozenset({"zip_mixed", "zip_lod_ladder"})

#: Name of the pose sidecar written next to the stores.
POSES_FILENAME = "POSES.json"


def store_filename(name: str) -> str:
    """The on-disk name of scene ``name`` under ``--out`` (and ``gate/`` in URLs)."""
    return f"{name}.luxar.zarr.zip" if name in ZIP_SCENES else f"{name}.luxar.zarr"


def _write_poses(out: Path, entries: dict[str, PoseEntry]) -> Path:
    """Merge ``entries`` into ``<out>/POSES.json``, keeping other scenes' entries."""
    target = out / POSES_FILENAME
    scenes: dict[str, Any] = {}
    if target.exists():
        scenes = json.loads(target.read_text()).get("scenes", {})
    scenes.update(entries)
    document = {
        "format": "luxar-render-gate-poses/1",
        "description": (
            "Recommended camera poses per gate store, in gate-scenes.json's pose "
            "shape ({position, target, up, fov?}); 'time' names a hidden discrete "
            "time dimension by its scene dimension index."
        ),
        "scenes": dict(sorted(scenes.items())),
    }
    target.write_text(json.dumps(document, indent=2) + "\n")
    return target


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument(
        "--only", nargs="*", choices=ALL_SCENE_NAMES, help="Write only these scenes."
    )
    parser.add_argument(
        "--heavy",
        action="store_true",
        help="Write the heavy set (large, slow stores) instead of the default set.",
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Shrink the perf-audit writers' element counts (tests only; the "
        "gate must use full-size stores).",
    )
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    names = args.only or (HEAVY_SCENE_NAMES if args.heavy else SCENE_NAMES)
    entries: dict[str, PoseEntry] = {}
    with asection(f"Writing render-gate scenes to {args.out}"):
        for name in names:
            target = args.out / store_filename(name)
            entry = WRITERS[name](target, small=args.small)
            if entry is not None:
                entries[name] = entry
            aprint(f"{name}: {target}")
        if entries:
            aprint(f"poses: {_write_poses(args.out, entries)}")


if __name__ == "__main__":
    main()
