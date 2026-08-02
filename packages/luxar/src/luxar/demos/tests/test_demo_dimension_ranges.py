"""End-to-end guards for demo-authored dimension ranges.

These demos previously declared ranges narrower than the coordinates they wrote.
The compiler reports that mismatch as a warning, so ordinary smoke tests can stay
green while the generated scene is internally inconsistent. Build deliberately
small scenes, reject the warning, and compare the persisted declarations with the
stored scene extent.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import zarr

from luxar.demos.demo_network_performance import generate_performance_test_dataset
from luxar.demos.demo_particle_collision_animated import (
    generate_animated_detector_scene,
)


def _build_and_read_scene_attrs(
    output_path: Path, builder: Callable[[Path], object]
) -> dict[str, Any]:
    """Build one scene and return its root attrs, rejecting range warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        builder(output_path)

    range_warnings = [
        str(item.message)
        for item in caught
        if "outside declared range" in str(item.message)
    ]
    assert not range_warnings, "\n".join(range_warnings)

    return dict(zarr.open_group(output_path, mode="r").attrs)


def _dimensions_by_name(attrs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index the persisted scene-dimension dictionaries by dimension name."""
    dimensions = attrs["scene_dimensions"]["dimensions"]
    return {dimension["name"]: dimension for dimension in dimensions}


def test_network_performance_ranges_match_the_generated_xyz_extent(
    tmp_path: Path,
) -> None:
    """The Gaussian cluster tails must define x/y/z, not a fixed guess."""
    output = tmp_path / "network.luxar.zarr"
    attrs = _build_and_read_scene_attrs(
        output,
        lambda path: generate_performance_test_dataset(path, n_points=80, seed=42),
    )

    dimensions = _dimensions_by_name(attrs)
    bounds = attrs["position_bounds"]
    assert dimensions["w"]["range"] == [-50, 50]
    for axis, column in zip(("x", "y", "z"), (1, 2, 3), strict=True):
        assert dimensions[axis]["range"] == [
            bounds["min"][column],
            bounds["max"][column],
        ]


def test_animated_collision_time_range_covers_every_written_frame(
    tmp_path: Path,
) -> None:
    """Frame zero and the final frame must both lie in the declared time range."""
    output = tmp_path / "collision.luxar.zarr"
    attrs = _build_and_read_scene_attrs(
        output,
        lambda path: generate_animated_detector_scene(
            path,
            n_events=1,
            n_jets_per_event=1,
            n_frames=3,
        ),
    )

    dimensions = _dimensions_by_name(attrs)
    bounds = attrs["position_bounds"]
    assert dimensions["time"]["range"] == [bounds["min"][3], bounds["max"][3]]
    assert dimensions["time"]["range"] == [0.0, 50.0]
    assert dimensions["time"]["step"] == 25.0
