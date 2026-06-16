"""Direct unit tests for the finalize-time passes in luxar.io._compiler.finalize."""

from __future__ import annotations

import numpy as np
import zarr

from luxar.io._compiler.finalize.hashing import compute_content_hashes
from luxar.io._compiler.finalize.lod_backfill import (
    finalize_lod_display_types,
    finalize_lod_position_bounds,
)
from luxar.io._compiler.finalize.validation import validate_discrete_dimension_ranges


def _lod_tree() -> zarr.Group:
    """Root with a kind=lod wrapper holding two gsplat leaves (no parent bounds)."""
    root = zarr.group()
    lod = root.create_group("lodgrp")
    lod.attrs["kind"] = "lod"
    coarse = lod.create_group("coarse")
    coarse.attrs["type"] = "gsplats"
    coarse.attrs["position_bounds"] = {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}
    fine = lod.create_group("fine")
    fine.attrs["type"] = "gsplats"
    fine.attrs["position_bounds"] = {"min": [-1.0, 0.0, 0.0], "max": [1.0, 2.0, 1.0]}
    return root


def test_finalize_lod_position_bounds_backfills_union() -> None:
    root = _lod_tree()
    assert "position_bounds" not in dict(root["lodgrp"].attrs)
    finalize_lod_position_bounds(root)
    bounds = dict(root["lodgrp"].attrs)["position_bounds"]
    assert bounds["min"] == [-1.0, 0.0, 0.0]
    assert bounds["max"] == [1.0, 2.0, 1.0]


def test_finalize_lod_position_bounds_never_overwrites() -> None:
    root = _lod_tree()
    authored = {"min": [-9.0, -9.0, -9.0], "max": [9.0, 9.0, 9.0]}
    root["lodgrp"].attrs["position_bounds"] = authored
    finalize_lod_position_bounds(root)
    assert dict(root["lodgrp"].attrs)["position_bounds"] == authored


def _partition_tree() -> zarr.Group:
    """Root with a kind=partition wrapper holding two gsplat-leaf parts (no parent
    bounds) — the shape a grafted partition (add_gsplats_from_file) produces, since
    add_partition_group does not compute the children-union the standalone writer
    stamps at write time."""
    root = zarr.group()
    part = root.create_group("partgrp")
    part.attrs["kind"] = "partition"
    p0 = part.create_group("part_0")
    p0.attrs["type"] = "gsplats"
    p0.attrs["position_bounds"] = {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}
    p1 = part.create_group("part_1")
    p1.attrs["type"] = "gsplats"
    p1.attrs["position_bounds"] = {"min": [5.0, -2.0, 0.0], "max": [6.0, 1.0, 3.0]}
    return root


def test_finalize_position_bounds_backfills_partition_wrapper() -> None:
    """A grafted kind=partition wrapper missing position_bounds must be back-filled
    with the union of its parts (regression: graft dropped wrapper bounds, losing
    partition-unit culling / graft-vs-standalone parity)."""
    root = _partition_tree()
    assert "position_bounds" not in dict(root["partgrp"].attrs)
    finalize_lod_position_bounds(root)
    bounds = dict(root["partgrp"].attrs)["position_bounds"]
    assert bounds["min"] == [0.0, -2.0, 0.0]
    assert bounds["max"] == [6.0, 1.0, 3.0]


def test_finalize_position_bounds_partition_never_overwrites() -> None:
    root = _partition_tree()
    authored = {"min": [-9.0, -9.0, -9.0], "max": [9.0, 9.0, 9.0]}
    root["partgrp"].attrs["position_bounds"] = authored
    finalize_lod_position_bounds(root)
    assert dict(root["partgrp"].attrs)["position_bounds"] == authored


def test_finalize_lod_display_types_resolves_from_finest() -> None:
    root = _lod_tree()
    finalize_lod_display_types(root)
    assert dict(root["lodgrp"].attrs)["display_type"] == "gsplats"


def test_compute_content_hashes_is_deterministic_and_stamps_attrs() -> None:
    root = _lod_tree()
    h1 = compute_content_hashes(root)
    assert isinstance(h1, str) and len(h1) > 0
    assert dict(root.attrs)["content_hash"] == h1
    # Recompute on an identical fresh tree → same root hash (determinism).
    h2 = compute_content_hashes(_lod_tree())
    assert h1 == h2


def test_compute_content_hashes_changes_with_data() -> None:
    root = _lod_tree()
    root["lodgrp"]["fine"].create_dataset(
        "centers", data=np.ones((3, 3), dtype=np.float32)
    )
    h_with = compute_content_hashes(root)
    assert h_with != compute_content_hashes(_lod_tree())


def test_validate_discrete_dimension_ranges_noop_without_bounds() -> None:
    # No scene_bounds → returns immediately without raising.
    validate_discrete_dimension_ranges(zarr.group(), None)
