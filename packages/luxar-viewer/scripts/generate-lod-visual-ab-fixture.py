#!/usr/bin/env python3
"""Generate deterministic coarse-vs-finest fixtures for the LOD visual A/B."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from luxar import CameraConfig, Dimension, Dimensions, LuxarZarrCompiler, ViewerConfig
from luxar._zarr_compat import read_array_meta
from luxar.encoding import EncodingMode


def read_level_element_counts(
    output: Path, node_name: str = "points_additive", array_name: str = "positions"
) -> list[int]:
    level_element_counts: list[int] = []
    level = 0
    while True:
        array_metadata = read_array_meta(
            output / node_name / f"child_{level}" / array_name
        )
        if array_metadata is None:
            break
        shape = array_metadata["shape"]
        encoding = array_metadata.get("attributes", {}).get("encoding", {})
        if shape[0] == 0 and encoding.get("name") == "array_ref":
            shape = encoding["original_shape"]
        level_element_counts.append(int(shape[0]))
        level += 1
    if not level_element_counts:
        raise RuntimeError(
            f"generated fixture contains no readable LOD levels for "
            f"{node_name}/{array_name}: {output}"
        )
    return level_element_counts


def build_fixture(output: Path) -> None:
    if not output.name.endswith(".luxar.zarr"):
        raise ValueError(f"output must end in .luxar.zarr: {output}")
    if output.exists():
        shutil.rmtree(output)

    rng = np.random.default_rng(2779)
    count = 16_384
    radius = np.sqrt(rng.random(count)) * 3.6
    arm = rng.integers(0, 4, count)
    angle = radius * 1.65 + arm * (np.pi / 2) + rng.normal(0.0, 0.18, count)
    positions = np.column_stack(
        [
            radius * np.cos(angle),
            radius * np.sin(angle),
            rng.normal(0.0, 0.16 + 0.02 * radius, count),
        ]
    ).astype(np.float32)
    normalized_radius = (radius / radius.max()).astype(np.float32)
    colors = np.column_stack(
        [
            0.12 + 0.20 * (1.0 - normalized_radius),
            0.20 + 0.30 * (arm / 3.0),
            0.42 + 0.24 * normalized_radius,
        ]
    ).astype(np.float32)
    radii = np.full(count, 0.022, dtype=np.float32)

    dimensions = Dimensions(
        [
            Dimension("x", unit="units", display=True),
            Dimension("y", unit="units", display=True),
            Dimension("z", unit="units", display=True),
        ]
    )
    viewer_config = ViewerConfig(
        camera=CameraConfig(
            position=(0.0, 0.0, 12.0),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0),
            fov=40.0,
        ),
        background_color="#000000",
        tone_mapping="None",
        exposure=0.0,
        global_offset=0.0,
        global_gamma=1.0,
        bloom_enabled=False,
        vignette_enabled=False,
        detector_noise_enabled=False,
        fxaa_enabled=False,
        msaa_enabled=False,
        ssaa_enabled=False,
        chromatic_lens_distortion_enabled=False,
        adaptive_dpr_enabled=False,
        control_type="orbit",
        auto_rotate=False,
    )

    with LuxarZarrCompiler(
        output,
        encoding_mode=EncodingMode.PRECISION,
        compressor=None,
        float16_allowed=False,
    ) as compiler:
        scene = compiler.create_scene(
            dimensions=dimensions, viewer_config=viewer_config
        )
        scene.add_points(
            "points_additive",
            positions,
            colors=colors,
            radii=radii,
            sharpness=np.full(count, 0.5, dtype=np.float32),
            opacity=0.35,
            blending_mode="additive",
            substitutive_lod={
                "coarse": "points",
                "method": "merge",
                "compression_factor": 4,
                "levels": 1,
                "quality_stamps": False,
            },
            additive_lod=False,
        )

        gsplat_count = 4096
        gsplat_hue = np.arange(gsplat_count) % 4
        gsplat_sample = np.arange(gsplat_count) // 4
        gsplat_radius = 0.35 + 3.1 * np.sqrt(
            (gsplat_sample + 0.5) / (gsplat_count // 4)
        )
        gsplat_angle = gsplat_radius * 1.8
        strand_offset = (gsplat_hue - 1.5) * 0.055
        gsplat_centers = np.column_stack(
            [
                (gsplat_radius + strand_offset) * np.cos(gsplat_angle),
                (gsplat_radius + strand_offset) * np.sin(gsplat_angle),
                0.08 * np.sin(gsplat_angle * 3.0) + strand_offset,
            ]
        ).astype(np.float32)
        palette = 2.0 * np.array(
            [[1.0, 0.12, 0.08], [0.08, 0.35, 1.0], [0.12, 1.0, 0.24], [1.0, 0.2, 0.8]],
            dtype=np.float32,
        )
        gsplat_colors = palette[gsplat_hue]
        gsplat_cholesky = np.tile(
            np.array([0.045, 0.0, 0.045, 0.0, 0.0, 0.045], dtype=np.float32),
            (gsplat_count, 1),
        )
        scene.add_gsplats(
            "gsplats_chromatic",
            centers=gsplat_centers,
            amplitudes=np.full(gsplat_count, 0.65, dtype=np.float32),
            cholesky_factors=gsplat_cholesky,
            colors=gsplat_colors,
            opacity=0.8,
            blending_mode="normal",
            substitutive_lod={
                "compression_factor": 4,
                "levels": 1,
                "method": "kmeans_lloyd",
                "color_weight": 8.0,
                "quality_stamps": False,
            },
            additive_lod=False,
        )
        scene.add_gsplats(
            "gsplats_spatial",
            centers=gsplat_centers,
            amplitudes=np.full(gsplat_count, 0.65, dtype=np.float32),
            cholesky_factors=gsplat_cholesky,
            colors=gsplat_colors,
            opacity=0.8,
            blending_mode="normal",
            substitutive_lod={
                "compression_factor": 4,
                "levels": 1,
                "method": "kmeans_lloyd",
                "color_weight": 0.0,
                "quality_stamps": False,
            },
            additive_lod=False,
        )

    point_level_counts = read_level_element_counts(output)
    gsplat_level_counts = read_level_element_counts(
        output, "gsplats_chromatic", "centers"
    )
    spatial_gsplat_level_counts = read_level_element_counts(
        output, "gsplats_spatial", "centers"
    )

    metadata = {
        "schemaVersion": 1,
        "benches": [
            {
                "id": "points-additive-merge",
                "geometry": "points",
                "lodGroup": "/points_additive",
                "levelElementCounts": point_level_counts,
            },
            {
                "id": "gsplats-normal-chromatic",
                "geometry": "gsplats",
                "lodGroup": "/gsplats_chromatic",
                "levelElementCounts": gsplat_level_counts,
            },
            {
                "id": "gsplats-normal-spatial",
                "geometry": "gsplats",
                "lodGroup": "/gsplats_spatial",
                "levelElementCounts": spatial_gsplat_level_counts,
            },
        ],
    }
    (output.parent / "fixture-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build_fixture(args.output.resolve())


if __name__ == "__main__":
    main()
