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

import numpy as np
import pytest
import zarr

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


def test_ppi_flow_field_persists_float32_grid_bounds(tmp_path: Path) -> None:
    """NumPy-derived field bounds must survive root-attribute serialization."""
    pd = pytest.importorskip("pandas")
    from luxar.demos import FlowField
    from luxar.demos.demo_ppi_flow_field import (
        PRESETS,
        OrientedEdges,
        StreamlineData,
        write_scene,
    )

    output = tmp_path / "ppi.luxar.zarr"
    grid_min = np.array([-1.25, -2.5, -3.75], dtype=np.float32)
    grid_max = np.array([4.5, 5.75, 6.25], dtype=np.float32)
    flow = FlowField(
        vectors=np.zeros((2, 2, 2, 3), dtype=np.float32),
        grid_min=grid_min,
        grid_max=grid_max,
        spacing=1.0,
        cache_key="test",
    )
    empty_int = np.empty(0, dtype=np.int32)
    empty_float = np.empty(0, dtype=np.float64)
    write_scene(
        output,
        nodes=["P1"],
        node_df=pd.DataFrame({"chromosome": ["1"]}),
        edges=pd.DataFrame({"sym_a": [], "sym_b": []}),
        coords=np.zeros((1, 3), dtype=np.float32),
        oriented=OrientedEdges(
            empty_int, empty_int, empty_float, empty_int.astype(bool)
        ),
        pagerank=np.array([1.0], dtype=np.float64),
        degrees=np.array([0], dtype=np.int32),
        communities=np.array([0], dtype=np.int32),
        flow=flow,
        streamline_data=StreamlineData(
            vertices=np.empty((0, 3), dtype=np.float32),
            segments=np.empty((0, 2), dtype=np.int32),
            colors=np.empty((0, 3), dtype=np.float32),
            streamline_count=0,
        ),
        preset=PRESETS["preview"],
    )

    dimensions = _dimensions_by_name(dict(zarr.open_group(output, mode="r").attrs))
    for axis, index in zip(("x", "y", "z"), range(3), strict=True):
        assert dimensions[axis]["range"] == [
            float(grid_min[index]),
            float(grid_max[index]),
        ]
