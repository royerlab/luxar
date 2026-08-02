#!/usr/bin/env python3
"""Data-Driven Demo: Human White-Matter Tractography (HCP-1065 dMRI atlas)

The wiring of the human brain, drawn as curves. Every one of the 87 named
white-matter tracts of the HCP-1065 population atlas is rendered as a bundle of
connected polylines in ICBM 2009a stereotaxic space, coloured by the standard
diffusion-MRI direction convention. The corpus callosum arches between the
hemispheres, the corticospinal tracts fan from the brainstem up into the motor
cortex, and the arcuate fasciculus hooks around the Sylvian fissure.

================================================================================
WHAT TRACTOGRAPHY IS
================================================================================
Diffusion MRI measures how water molecules diffuse in each voxel of the brain.
Inside a white-matter fibre bundle, water diffuses much more freely *along* the
axons than across them, so the diffusion profile points down the fibre. Chaining
those local orientations voxel to voxel — *tractography* — reconstructs the
long-range pathways that connect one region of cortex to another.

The result is not a picture of axons (an axon is ~1 um across; a voxel is ~1 mm).
It is a reconstruction of the dominant fibre geometry, at millimetre scale, and
it is the only non-invasive way to see the brain's structural wiring in a living
person.

THE DIRECTION COLOUR CONVENTION
-------------------------------
Every point is coloured by the absolute value of the local tangent direction,
the near-universal convention in the diffusion-imaging literature:

    RED   = left-right          (commissural fibres: the corpus callosum)
    GREEN = anterior-posterior  (association fibres: IFOF, ILF, arcuate)
    BLUE  = inferior-superior   (projection fibres: corticospinal tract)

So the colour is not decoration — the corpus callosum is red *because* it runs
left-right, and the corticospinal tract is blue *because* it runs up and down.
Reading the picture is reading the anatomy.

================================================================================
WHY THIS IS A LINES DEMO
================================================================================
Most volumetric science has to be *converted* into a Luxar geometry type. A
tractogram does not: a streamline is already a 3D polyline, so this is Luxar's
Lines geometry applied to data that is natively made of curves. Each of the 87
bundles becomes its own Lines node with ``layer=True``, so the Layers panel
(press **L**) gives per-tract visibility, display range, gamma and blending.

RENDERING NOTE — `additive`, AND WHY NOT `normal`
-------------------------------------------------
A tractogram is a *solid* object: ~6.8M segments packed into a 180 mm skull,
many hundreds deep along any view ray. That makes the choice of blending mode
the single biggest visual decision in this demo, and it is not the obvious one.

`normal` (alpha-over) at opacity 1.0 looks right in a still: depthWrite is on,
so each pixel shows the nearest fibre and the near/far hemispheres separate
cleanly. But alpha-blended nodes render in the *transparent* pass, which THREE
sorts back-to-front **per object**, every frame. With 87 mutually-overlapping
bundles whose centroids interleave, that sort order flips as the camera orbits
and whole tracts visibly pop in front of each other. The still is fine; the
interaction is not, and this scene is meant to be orbited.

`additive` has no such failure mode, because addition is commutative: the frame
is the same whatever order the 87 nodes draw in, so there is nothing to sort and
nothing to pop. The cost is that it ignores depth, which is why the naive
settings blow out — the accumulated sum clips to white long before the far side
of the brain has been drawn.

The fix is to make each sample contribute *little*: opacity 0.24 and a display
window of [0, 74.976] (a ~75x attenuation on the colour). Hundreds of faint
samples then integrate into a glowing, X-ray-like volume in which the internal
architecture — the callosal fan, the arcuate's hook, the cerebellar peduncles —
is visible *through* the surface fibres rather than hidden behind them.

If you ever do want hard occlusion back without the popping, the mode to reach
for is `opaque`, not `normal`: it is depth-tested and depth-written but never
enters the transparent pass, so it is also order-independent.

DATA SOURCE & CITATIONS:
========================
Dataset:
--------
Source:  HCP-1065 population-averaged tractography atlas
         https://brain.labsolver.org/hcp_trk_atlas.html
Archive: hcp1065_avg_tracts_trk.zip (588 MB, 87 bundles in TRK format)
Space:   ICBM 2009a Nonlinear Asymmetric, 1 mm isotropic
Built:   automatic + augmented fibre tracking over 1,065 Human Connectome
         Project young-adult subjects
License: Creative Commons Attribution-ShareAlike 4.0 International

Citation:
---------
Yeh, F-C. Population-based tract-to-region connectome of the human brain and
its hierarchical topology. Nature Communications 13, 4933 (2022).
https://doi.org/10.1038/s41467-022-32595-4

Data were provided in part by the Human Connectome Project, WU-Minn Consortium
(Principal Investigators: David Van Essen and Kamil Ugurbil; 1U54MH091657),
funded by the 16 NIH Institutes and Centers that support the NIH Blueprint for
Neuroscience Research; and by the McDonnell Center for Systems Neuroscience at
Washington University.

Because the source is ShareAlike, nothing derived from it is committed to this
repository — the atlas is downloaded and cached locally on first run.

Usage:
    python -m luxar.demos.demo_dmri_tractography
    python -m luxar.demos.demo_dmri_tractography --no-serve
    python -m luxar.demos.demo_dmri_tractography --recompute
    python -m luxar.demos.demo_dmri_tractography --per-bundle=3000 --points=20

Controls:
    - Press 'L' to open the Layers panel: one row per tract, 87 in total
    - Ctrl+C to stop and cleanup
"""

from __future__ import annotations

DEMO_META = {
    "key": "dmri_tractography",
    "title": "Human White-Matter Tractography (HCP-1065)",
    "description": (
        "The human brain's 87 named white-matter tracts as "
        "directionally-coloured streamlines."
    ),
    "category": "medical",
    "geometry": "lines",
    "requirements": {
        "download_mb": 588,
        "compute": "heavy",
        "gpu": "none",
        "local_data": None,
    },
    "caches": ["dmri_tractography"],
    "outputs": ["dmri_tractography"],
}

import gzip
import io
import zipfile
from pathlib import Path
from typing import Final

import numpy as np
from arbol import Arbol, aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import CameraConfig, ViewerConfig
from luxar.demos import require_module
from luxar.encoding import EncodingMode
from luxar.utils.demos import (
    cached_download,
    launch_viewer,
    parse_demo_flags,
    parse_int_arg,
)
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

DEMO_NAME: Final = "dmri_tractography"

ATLAS_URL: Final = (
    "https://github.com/data-others/atlas/releases/download/hcp1065/"
    "hcp1065_avg_tracts_trk.zip"
)
ATLAS_ZIP: Final = "hcp1065_avg_tracts_trk.zip"
ATLAS_ZIP_BYTES: Final = 587_869_457

#: The five anatomical divisions the atlas ships, in the order they are added to
#: the scene. These are the zip's top-level directories — note the space in
#: "cranial nerve", which is a real directory name and not a typo.
DIVISIONS: Final = (
    "association",
    "projection",
    "commissural",
    "cerebellum",
    "cranial nerve",
)

#: Points per streamline after arc-length resampling. The source is sampled at
#: ~0.4 mm (a ~10 cm tract carries ~270 points), far finer than any rendered
#: line width; 28 points keeps every tract's curvature while cutting the vertex
#: count ~10x.
DEFAULT_POINTS: Final = 28

#: Streamlines kept per bundle. Two independent viewer limits bound this, and
#: the tighter one wins:
#:
#:   * ``scripts/check_demo_ladders.py`` fails any un-laddered lines leaf above
#:     200,000 vertices, so a node must stay under 200_000 / POINTS streamlines.
#:   * The element data texture holds 6 texels per segment, capping a node at
#:     ``682 * maxTextureSize`` segments — 2,793,472 on a 4096-class GPU. Over
#:     that the viewer silently clamps and the tail never renders.
#:
#: At 6,000 x 28 a node peaks at 168,000 vertices / 162,000 segments, clearing
#: both with margin. Only ~16 of the 87 bundles are large enough to be capped.
DEFAULT_PER_BUNDLE: Final = 6_000

#: Deterministic subsample of the over-large bundles, so a rebuild is identical.
SUBSAMPLE_SEED: Final = 0

LINE_WIDTH: Final = 0.32  # mm; the brain is ~180 mm across
LINE_OPACITY: Final = 0.24
#: Gain on the per-vertex colour, i.e. a display window of ``[0, 74.976]``
#: (``intensity = 1 / (max - min)``, see the viewer's
#: ``rendering/display-range.ts``). A ~75x attenuation looks extreme written
#: down, but it is what additive compositing of a *solid* object needs: with
#: hundreds of segments along every view ray, each one may contribute only a
#: percent or so before the sum clips. Tuned in the Layers panel, then baked.
#:
#: NOTE: this must not be exactly 1.0. The line material compiles with a
#: ``LUXAR_NO_GOG`` define when gain/offset/gamma are all identity, which
#: strips the gain path out of the shader entirely — an authored 1.0 cannot
#: then be recovered at runtime.
LINE_INTENSITY: Final = 1.0 / 74.976

#: Substitutive LOD: each level replaces the finer one with fewer, larger
#: elements, so zooming out costs less instead of drawing every fibre. Coarse
#: levels are gsplat "beads" lifted off the segments (`luxar.gsplats.lift`);
#: the original Lines node stays as the finest level.
#:
#: ``compression_factor`` has to be MUCH larger than the K=4 default here, and
#: the reason is specific to thin lines. The lift spaces beads every
#: ``sigma_perp = 2w/T`` **along arc length**, so the bead count is
#: ``total_fibre_length / sigma_perp`` — it does not care how many segments the
#: fibre was cut into. At w=0.32 mm this bundle's 162K segments lift to ~3.8M
#: beads, so a K=4 "coarse" level is 912K splats: 5.6x *heavier* than the
#: 162K-segment level it is supposed to replace. Measured on AF_L:
#:
#:     K=4,   L=3  ->  65.0 MB   levels 54K / 221K / 912K   (worse than useless)
#:     K=64,  L=2  ->   5.3 MB   levels 930 / 59,457
#:     K=256, L=2  ->   3.2 MB   levels 59 / 14,865         <- chosen
#:
#: K=256 puts the first coarse level at ~9% of the fine level's element count,
#: which is what a substitutive level is for. Across all 87 nodes this is the
#: difference between a 2.1 GB scene and a ~200 MB one.
SUBSTITUTIVE_LOD: Final = dict(compression_factor=256, levels=2)

# NOTE — no `additive_lod` here, deliberately. An additive ladder composed
# under a substitutive one is REFUSED for `line_type="indexed"`:
#
#   UserWarning: the requested streaming ladder cannot be honoured
#   (line_type='indexed' edges are not preserved by the ladder);
#   levels will load all-at-once.
#
# The ladder rebuilds each connected component as a plain chain over its
# members, which for an arbitrary indexed edge list would invent edges that do
# not exist and drop ones that do (see `adders/lines.py`). Passing it anyway
# just warns on every build and changes nothing — every level still loads in
# one commit. It costs us little: each node is under the 200K-vertex threshold
# at which `check_demo_ladders.py` requires a ladder, and the 87 nodes already
# stream independently of one another.

FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]

POINTS_PER_STREAMLINE = parse_int_arg("points", DEFAULT_POINTS)
PER_BUNDLE = parse_int_arg("per-bundle", DEFAULT_PER_BUNDLE)

CACHE_DIR: Final = Path.home() / ".cache" / "luxar" / DEMO_NAME

Arbol.max_depth = 4


# =============================================================================
# Pure helpers (unit-tested; no network / no IO)
# =============================================================================


def resample_polyline(points: np.ndarray, n: int) -> np.ndarray:
    """Arc-length resample an ``(M, 3)`` streamline to exactly ``(n, 3)``.

    Interpolates each coordinate against the cumulative chord-length parameter,
    so the output points are evenly spaced *along the curve* rather than evenly
    spaced in the input index. That matters because tractography step sizes are
    not uniform across bundles: index-space resampling would bunch points where
    the tracker happened to take small steps.

    Endpoints are preserved exactly (the first and last parameter values are the
    ends of the interval), so a resampled tract still starts and stops where the
    original did.

    Args:
        points: ``(M, 3)`` streamline vertices, ``M >= 2``.
        n: Number of output points, ``>= 2``.

    Returns:
        ``(n, 3)`` float32 resampled streamline.

    Raises:
        ValueError: If ``points`` is not ``(M >= 2, 3)`` or ``n < 2``.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (M, 3), got {pts.shape}")
    if pts.shape[0] < 2:
        raise ValueError(f"points must have M >= 2, got {pts.shape[0]}")
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")

    step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    dist = np.concatenate([[0.0], np.cumsum(step)])
    total = float(dist[-1])
    if total <= 0.0:
        # A degenerate zero-length streamline (all vertices coincident): every
        # output point is that single location. np.interp would divide by zero.
        return np.repeat(pts[:1], n, axis=0).astype(np.float32)

    want = np.linspace(0.0, total, n)
    out = np.empty((n, 3), dtype=np.float32)
    for axis in range(3):
        out[:, axis] = np.interp(want, dist, pts[:, axis])
    return out


def direction_colors(paths: np.ndarray) -> np.ndarray:
    """Per-vertex RGB from the local tangent direction (the dMRI convention).

    ``R = |dx|`` (left-right), ``G = |dy|`` (anterior-posterior),
    ``B = |dz|`` (inferior-superior), from the normalized central-difference
    tangent. The absolute value is what makes the encoding orientation-only:
    a tract looks the same whichever end you traced it from.

    Args:
        paths: ``(P, V, 3)`` resampled streamlines in **anatomical RAS** order
            — the colour convention is defined on anatomical axes, so this must
            be called before any scene-space axis remap.

    Returns:
        ``(P, V, 3)`` uint8 RGB.

    Raises:
        ValueError: If ``paths`` is not ``(P, V >= 2, 3)``.
    """
    arr = np.asarray(paths, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"paths must be (P, V, 3), got {arr.shape}")
    if arr.shape[1] < 2:
        raise ValueError(f"paths must have V >= 2, got {arr.shape[1]}")

    # Central differences interior, one-sided at the two ends.
    tangent = np.empty_like(arr)
    tangent[:, 1:-1] = arr[:, 2:] - arr[:, :-2]
    tangent[:, 0] = arr[:, 1] - arr[:, 0]
    tangent[:, -1] = arr[:, -1] - arr[:, -2]

    norm = np.linalg.norm(tangent, axis=2, keepdims=True)
    # A coincident-vertex pair gives a zero tangent; leave it black rather than
    # dividing by zero and painting a NaN.
    unit = np.divide(tangent, norm, out=np.zeros_like(tangent), where=norm > 0.0)
    return np.clip(np.abs(unit) * 255.0, 0.0, 255.0).astype(np.uint8)


def ras_to_scene(ras: np.ndarray) -> np.ndarray:
    """Rotate anatomical RAS into the viewer's Y-up scene convention.

    RAS is ``(+x right, +y anterior, +z superior)``; the viewer orbits about a
    Y-up axis. Mapping ``(x, y, z) -> (x, z, -y)`` puts superior on +Y, so the
    default orbit behaves and the brain is upright without a custom camera up.

    Args:
        ras: ``(..., 3)`` coordinates in RAS millimetres.

    Returns:
        ``(..., 3)`` float32 in scene space.
    """
    arr = np.asarray(ras, dtype=np.float64)
    if arr.shape[-1] != 3:
        raise ValueError(f"last axis must be 3, got {arr.shape}")
    return np.stack([arr[..., 0], arr[..., 2], -arr[..., 1]], axis=-1).astype(
        np.float32
    )


def polyline_segment_indices(n_paths: int, n_vertices: int) -> np.ndarray:
    """Indices joining consecutive vertices WITHIN each path (never across).

    Sharing a vertex at each joint lets the line material render a seamless join
    instead of two overlapping end-caps; excluding the path boundaries stops the
    last vertex of one streamline connecting to the first of the next.

    Returns:
        ``(2 * n_paths * (n_vertices - 1),)`` uint32, consecutive pairs.
    """
    if n_vertices < 2:
        raise ValueError(f"n_vertices must be >= 2, got {n_vertices}")
    base = (np.arange(n_paths, dtype=np.int64) * n_vertices)[:, None]
    starts = base + np.arange(n_vertices - 1, dtype=np.int64)[None, :]
    return np.stack([starts, starts + 1], axis=-1).reshape(-1).astype(np.uint32)


def subsample_indices(n_available: int, n_keep: int, *, seed: int) -> np.ndarray:
    """Sorted indices of ``n_keep`` streamlines drawn without replacement.

    Sorted so the kept streamlines stay in file order (readable diffs, stable
    caches), and seeded so a rebuild reproduces the same tract exactly.
    """
    if n_keep >= n_available:
        return np.arange(n_available, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_available, size=n_keep, replace=False))


def brain_camera(radius: float) -> CameraConfig:
    """Left-lateral opening pose, the conventional view of a tractogram.

    Looks down the +X (right) axis at the left hemisphere, slightly raised and
    pulled forward, which is how these atlases are shown in the literature.
    """
    return CameraConfig(
        position=(-2.6 * radius, 0.55 * radius, 0.85 * radius),
        target=(0.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        fov=38.0,
    )


# =============================================================================
# Data loading
# =============================================================================


def _download_atlas() -> Path:
    """Fetch (once) the 588 MB TRK atlas archive into the demo cache."""
    return cached_download(
        ATLAS_URL,
        DEMO_NAME,
        ATLAS_ZIP,
        expected_size=ATLAS_ZIP_BYTES,
    )


def _bundle_members(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    """Return ``(division, member_path)`` for every bundle, in DIVISIONS order.

    Within a division the bundles are sorted by name so the Layers panel lists
    them predictably (AF_L before AF_R before C_FP_L ...).
    """
    members = [n for n in archive.namelist() if n.endswith(".trk.gz")]
    out: list[tuple[str, str]] = []
    for division in DIVISIONS:
        prefix = f"{division}/"
        out.extend(
            (division, name)
            for name in sorted(m for m in members if m.startswith(prefix))
        )
    missing = set(members) - {m for _, m in out}
    if missing:
        raise ValueError(
            f"{len(missing)} bundle(s) outside the known divisions {DIVISIONS}: "
            f"{sorted(missing)[:3]}"
        )
    return out


def _load_bundle(raw_gz: bytes, *, per_bundle: int, points: int) -> tuple:
    """Decode one ``.trk.gz`` into resampled paths + direction colours.

    Returns ``(scene_xyz, rgb, n_kept, n_total)`` where ``scene_xyz`` is
    ``(n_kept * points, 3)`` float32 in scene space and ``rgb`` is the matching
    ``(n_kept * points, 3)`` uint8.
    """
    # Imported at the point of use, not in a preflight: a warm-cache run never
    # needs nibabel at all. The submodule inherits nibabel's tabled spec, so the
    # error message still advertises `nibabel>=5.0.0` and the `demos` extra.
    nib_streamlines = require_module("nibabel.streamlines")

    trk = nib_streamlines.TrkFile.load(
        io.BytesIO(gzip.decompress(raw_gz)), lazy_load=False
    )
    streamlines = trk.tractogram.streamlines  # RAS+ millimetres
    n_total = len(streamlines)
    if n_total == 0:
        raise ValueError("bundle contains no streamlines")

    keep = subsample_indices(n_total, per_bundle, seed=SUBSAMPLE_SEED)
    paths = np.empty((len(keep), points, 3), dtype=np.float32)
    for row, idx in enumerate(keep):
        paths[row] = resample_polyline(np.asarray(streamlines[int(idx)]), points)

    rgb = direction_colors(paths).reshape(-1, 3)
    scene_xyz = ras_to_scene(paths).reshape(-1, 3)
    return scene_xyz, rgb, len(keep), n_total


def load_or_build_bundles(*, per_bundle: int, points: int) -> dict:
    """Return the assembled per-bundle arrays, using the ``.npz`` cache if fresh.

    The cache key includes the two sizing knobs, so ``--per-bundle`` /
    ``--points`` sweeps do not collide with each other.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_npz = CACHE_DIR / f"bundles_p{points}_n{per_bundle}.npz"

    if cache_npz.exists() and not RECOMPUTE:
        aprint(f"Using cached bundles: {cache_npz.name}")
        with np.load(cache_npz, allow_pickle=False) as data:
            names = [str(x) for x in data["names"]]
            return {
                "names": names,
                "divisions": [str(x) for x in data["divisions"]],
                "positions": [data[f"pos_{i}"] for i in range(len(names))],
                "colors": [data[f"rgb_{i}"] for i in range(len(names))],
            }

    zip_path = _download_atlas()
    names: list[str] = []
    divisions: list[str] = []
    positions: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    kept_total = 0
    source_total = 0

    with asection("Decoding 87 tract bundles"):
        with zipfile.ZipFile(zip_path) as archive:
            for division, member in _bundle_members(archive):
                bundle = Path(member).name.removesuffix(".trk.gz")
                xyz, rgb, n_kept, n_src = _load_bundle(
                    archive.read(member), per_bundle=per_bundle, points=points
                )
                names.append(bundle)
                divisions.append(division)
                positions.append(xyz)
                colors.append(rgb)
                kept_total += n_kept
                source_total += n_src
                capped = " (capped)" if n_kept < n_src else ""
                aprint(
                    f"{division}/{bundle}: {n_kept:,} of {n_src:,} streamlines{capped}"
                )

    aprint(
        f"{len(names)} bundles, {kept_total:,} streamlines kept of "
        f"{source_total:,}, {kept_total * points:,} vertices"
    )

    # Center on the atlas centroid so the brain sits at the origin.
    centroid = np.concatenate(positions).mean(axis=0)
    positions = [(p - centroid).astype(np.float32) for p in positions]

    payload = {f"pos_{i}": p for i, p in enumerate(positions)}
    payload.update({f"rgb_{i}": c for i, c in enumerate(colors)})
    payload["names"] = np.array(names)
    payload["divisions"] = np.array(divisions)
    np.savez_compressed(cache_npz, **payload)
    aprint(f"Cached bundles: {cache_npz.name}")

    return {
        "names": names,
        "divisions": divisions,
        "positions": positions,
        "colors": colors,
    }


# =============================================================================
# Scene
# =============================================================================


def build_scene(bundles: dict, output_path: Path, *, points: int) -> Path:
    """Write the 87-node tractography scene."""
    names = bundles["names"]
    divisions = bundles["divisions"]
    positions = bundles["positions"]
    colors = bundles["colors"]

    extent = float(np.abs(np.concatenate(positions)).max())

    with asection("Writing scene"):
        dims = Dimensions(
            [
                Dimension("x", unit="mm", display=True),
                Dimension("y", unit="mm", display=True),
                Dimension("z", unit="mm", display=True),
            ]
        )
        with LuxarZarrCompiler(output_path, encoding_mode=EncodingMode.PRECISION) as c:
            scene = c.create_scene(
                dimensions=dims,
                viewer_config=ViewerConfig(
                    tone_mapping="ACES",
                    camera=brain_camera(extent),
                ),
            )
            scene.attrs["title"] = (
                "Human White-Matter Tractography — HCP-1065 population atlas"
            )

            groups = {d: scene.add_group(d.replace(" ", "_")) for d in DIVISIONS}
            total_segments = 0

            for name, division, xyz, rgb in zip(names, divisions, positions, colors):
                n_paths = len(xyz) // points
                indices = polyline_segment_indices(n_paths, points)
                total_segments += len(indices) // 2
                groups[division].add_lines(
                    name,
                    vertices=xyz,
                    widths=LINE_WIDTH,
                    colors=rgb,
                    indices=indices,
                    # `indexed`, NOT `segments`: interior joints must share a
                    # vertex index or thick lines render as chains of beads.
                    line_type="indexed",
                    # `additive` — see the module docstring's RENDERING NOTE.
                    # Order-independent, so 87 mutually-overlapping nodes never
                    # pop as the camera moves.
                    blending_mode="additive",
                    opacity=LINE_OPACITY,
                    intensity=LINE_INTENSITY,
                    substitutive_lod=SUBSTITUTIVE_LOD,
                    layer=True,
                )

            scene.add_text(
                "Human White-Matter Tractography",
                position=(0.02, 0.02),
                font_size=0.042,
                anchor="top-left",
                color="rgba(255,255,255,0.75)",
                blend_mode="difference",
            )
            scene.add_text(
                "HCP-1065 atlas (Yeh 2022, CC BY-SA 4.0) — 87 tracts",
                position=(0.98, 0.97),
                font_size=0.015,
                anchor="bottom-right",
                color="rgba(200,200,220,0.5)",
            )

        aprint(f"{len(names)} lines nodes, {total_segments:,} segments")
        aprint(f"Scene saved: {output_path}")
    return output_path


def load_or_build_scene(output_path: Path) -> Path:
    """Return the built scene path, regenerating on a fresh system."""
    if output_path.exists() and not RECOMPUTE:
        aprint(f"Using existing scene: {output_path}")
        return output_path

    bundles = load_or_build_bundles(per_bundle=PER_BUNDLE, points=POINTS_PER_STREAMLINE)
    return build_scene(bundles, output_path, points=POINTS_PER_STREAMLINE)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    aprint("=" * 70)
    aprint("Demo: Human White-Matter Tractography — HCP-1065 dMRI atlas")
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
        aprint("Data credit: HCP-1065 tractography atlas (Yeh 2022, CC BY-SA 4.0)")
        aprint("Press 'L' in the viewer for per-tract visibility.")
        launch_viewer(scene_path)


if __name__ == "__main__":
    main()
