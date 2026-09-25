"""Generate the small, deterministic scenes the render gate captures.

The render gate (``run-gate.mjs``, see ``docs/guides/developer/RENDER_GATE.md``)
compares a baseline build against a candidate build pixel by pixel. Seeded
synthetic scenes cover the bulk geometry; these stores cover what the synthetic
injector cannot: picking (injected nodes register no pick material), mixed
geometry types in one frame, the scene-captured environment, glass refraction,
a spatial partition, an LOD ladder under a rotated parent, and a nano-scale
orthographic line scene.

Every scene is seeded, so two runs write the same data. Usage::

    python generate_gate_scenes.py --out <repo>/datasets/gate
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimensions, LuxarZarrCompiler
from luxar.core.transforms import rotate_y
from luxar.core.viewer_config import CameraConfig, EnvironmentConfig, ViewerConfig
from luxar.encoding import EncodingMode
from luxar.mesh.primitives import icosphere

#: Scene names, in the order they are written. `gate-scenes.json` refers to them.
SCENE_NAMES = (
    "mixed",
    "env_splats",
    "partition_normal",
    "glass",
    "lod_ladder",
    "tiny_units_ortho",
)


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


def write_mixed(path: Path) -> None:
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


def write_env_splats(path: Path) -> None:
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


def write_partition_normal(path: Path) -> None:
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


def write_glass(path: Path) -> None:
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


def write_lod_ladder(path: Path) -> None:
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


def write_tiny_units_ortho(path: Path) -> None:
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


WRITERS = {
    "mixed": write_mixed,
    "env_splats": write_env_splats,
    "partition_normal": write_partition_normal,
    "glass": write_glass,
    "lod_ladder": write_lod_ladder,
    "tiny_units_ortho": write_tiny_units_ortho,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument(
        "--only", nargs="*", choices=SCENE_NAMES, help="Write only these scenes."
    )
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    with asection(f"Writing render-gate scenes to {args.out}"):
        for name in args.only or SCENE_NAMES:
            target = args.out / f"{name}.luxar.zarr"
            WRITERS[name](target)
            aprint(f"{name}: {target}")


if __name__ == "__main__":
    main()
