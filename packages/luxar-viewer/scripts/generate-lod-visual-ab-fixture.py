#!/usr/bin/env python3
"""Generate the deterministic Points/additive fixture for the LOD visual A/B."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from luxar import CameraConfig, Dimension, Dimensions, LuxarZarrCompiler, ViewerConfig
from luxar._zarr_compat import read_array_meta
from luxar.encoding import EncodingMode


def read_level_element_counts(output: Path) -> list[int]:
    level_element_counts: list[int] = []
    level = 0
    while True:
        positions_metadata = read_array_meta(
            output / "points_additive" / f"child_{level}" / "positions"
        )
        if positions_metadata is None:
            break
        level_element_counts.append(int(positions_metadata["shape"][0]))
        level += 1
    if not level_element_counts:
        raise RuntimeError(
            f"generated fixture contains no readable LOD levels: {output}"
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

    level_element_counts = read_level_element_counts(output)

    metadata = {
        "schemaVersion": 1,
        "benches": [
            {
                "id": "points-additive-merge",
                "geometry": "points",
                "lodGroup": "/points_additive",
                "levelElementCounts": level_element_counts,
            }
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
