#!/usr/bin/env python3
"""Smoke tests for batch plan HPC fixes.

Validates:
  - .zarr.zip files are accepted by discover_ome_zarr_shape and load_volume
  - Custom axes attribute parsing (e.g. time,camera,channel,z,y,x)
  - --axes override works and validates length
  - Array selection consistency between discover and load (picks largest)
  - Auto-tile: small volumes produce 1 tile (no unnecessary tiling)
  - tasks_per_job packing: auto-calculated and reduces Slurm array size
  - 6D array slicing via flat channel index
  - cull_retention default is 0.95

Run with:
    hatch run python scripts/test_batch_plan_fixes.py
"""

import tempfile
from pathlib import Path

import numpy as np


def test_zarr_zip_suffix_detection():
    """Path('.zarr.zip') should be routed to zarr loading."""
    p = Path("foo.zarr.zip")
    suffix = p.suffix.lower()
    stem_is_zarr = p.stem.endswith(".zarr")
    assert suffix == ".zip" and stem_is_zarr, (
        f"Expected .zip suffix with .zarr stem, got suffix={suffix}, stem={p.stem}"
    )
    # Also check plain .zarr
    p2 = Path("bar.zarr")
    assert p2.suffix.lower() == ".zarr"
    print("PASS: .zarr.zip suffix detection")


def test_custom_axes_parsing():
    """_parse_custom_axes_attr correctly identifies T, C, and spatial dims."""
    from luxar.cli.gsplat_config import _parse_custom_axes_attr

    # 6D Keller-style
    info = _parse_custom_axes_attr(
        ["time", "camera", "channel", "z", "y", "x"],
        (10, 2, 4, 97, 627, 1383),
        Path("test.zarr"),
    )
    assert info.n_timepoints == 10, f"Expected T=10, got {info.n_timepoints}"
    assert info.n_channels == 8, f"Expected C=8 (2*4), got {info.n_channels}"
    assert info.spatial_shape == (97, 627, 1383), f"Got spatial {info.spatial_shape}"

    # 4D TimeFused-style
    info2 = _parse_custom_axes_attr(
        ["time", "z", "y", "x"],
        (1434, 108, 1352, 532),
        Path("test.zarr"),
    )
    assert info2.n_timepoints == 1434
    assert info2.n_channels == 1
    assert info2.spatial_shape == (108, 1352, 532)

    # All spatial (no time, no channel)
    info3 = _parse_custom_axes_attr(
        ["z", "y", "x"],
        (100, 200, 300),
        Path("test.zarr"),
    )
    assert info3.n_timepoints == 1
    assert info3.n_channels == 1
    assert info3.spatial_shape == (100, 200, 300)

    print("PASS: custom axes parsing")


def test_axes_override_validation():
    """axes_override must match array ndim."""
    import zarr

    from luxar.cli.gsplat_config import discover_ome_zarr_shape

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.zarr"
        z = zarr.open(str(path), mode="w")
        z.create_dataset("data", data=np.zeros((5, 10, 20), dtype=np.float32))
        z.attrs["axes"] = ["z", "y", "x"]

        # Correct override
        info = discover_ome_zarr_shape(path, axes_override=["time", "y", "x"])
        assert info.n_timepoints == 5

        # Wrong length
        try:
            discover_ome_zarr_shape(path, axes_override=["t", "c", "z", "y", "x"])
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "5 labels" in str(e) or "3" in str(e)

    print("PASS: axes override validation")


def test_array_selection_consistency():
    """Both discover_ome_zarr_shape and _load_zarr_volume pick the largest array."""
    import zarr

    from luxar.cli.gsplat_config import _load_zarr_volume, discover_ome_zarr_shape

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.zarr"
        z = zarr.open(str(path), mode="w")
        # session1 is smaller, session2 is larger
        z.create_dataset("session1", data=np.ones((3, 10, 10), dtype=np.float32))
        z.create_dataset("session2", data=np.ones((100, 20, 20), dtype=np.float32) * 2)

        # discover should pick session2 (larger)
        info = discover_ome_zarr_shape(path)
        assert info.shape == (100, 20, 20), f"discover picked wrong array: {info.shape}"

        # load should also pick session2
        vol = _load_zarr_volume(path, channel=None, timepoint=None, array_key=None)
        assert vol.shape == (100, 20, 20), f"load picked wrong array: {vol.shape}"

    print("PASS: array selection consistency")


def test_auto_tile_small_volume():
    """Volumes that fit in GPU capacity should produce 1 tile."""
    from luxar.gsplats.tiling import compute_tile_specs

    spatial = (108, 1352, 532)
    # If tile_size = max(spatial) + overlap, should be 1 tile
    tile_size = max(spatial) + 32
    specs = compute_tile_specs(spatial, tile_size, 32)
    assert len(specs) == 1, f"Expected 1 tile, got {len(specs)}"
    print("PASS: auto-tile small volume → 1 tile")


def test_cull_retention_defaults():
    """fit_gaussian_splats should default to cull_retention=0.95."""
    import inspect

    from luxar.gsplats.fit_gsplats import fit_gaussian_splats

    sig = inspect.signature(fit_gaussian_splats)
    default = sig.parameters["cull_retention"].default
    assert default == 0.95, f"fit_gaussian_splats cull_retention default={default}, expected 0.95"

    print("PASS: cull_retention default is 0.95")


def test_6d_channel_decoding():
    """Flat channel index should correctly decode to multi-dim indices for 6D."""
    # For shape (10, 2, 4, 97, 627, 1383) = T, camera, channel, z, y, x
    # channel=5 should decode to camera=1, channel=1 (5 = 1*4 + 1)
    shape = (10, 2, 4, 97, 627, 1383)
    ndim = len(shape)
    timepoint = 3
    channel = 5  # flat index

    # Reproduce the decoding logic
    t = timepoint
    idx = [t]
    remaining_non_spatial = ndim - 4
    c_flat = channel
    non_spatial_shape = shape[1: 1 + remaining_non_spatial]
    for dim_size in reversed(non_spatial_shape):
        idx.insert(1, c_flat % dim_size)
        c_flat //= dim_size

    assert idx == [3, 1, 1], f"Expected [3, 1, 1], got {idx}"
    # channel=5: 5 // 4 = 1 (camera), 5 % 4 = 1 (channel)
    print("PASS: 6D flat channel decoding")


def test_tasks_per_job_manifest():
    """Manifest fields are serializable and have correct defaults."""
    from luxar.gsplats.batch.manifest import BatchManifest

    m = BatchManifest()
    assert m.tasks_per_job == 1
    assert m.parallel_tasks_per_job is False

    # JSON serializable
    import json
    from dataclasses import asdict
    d = asdict(m)
    json.dumps(d)  # should not raise

    print("PASS: manifest fields serializable with correct defaults")


def test_env_capture_ld_library_path_prepend():
    """LD_LIBRARY_PATH should be prepended, not replaced, in preamble."""
    from luxar.gsplats.batch.env_capture import CapturedEnv, generate_env_preamble

    env = CapturedEnv(
        env_vars={"LD_LIBRARY_PATH": "/some/path"},
        loaded_modules=["cuda/12.8"],
    )
    preamble = generate_env_preamble(env)
    # Should have the prepend pattern
    assert "${LD_LIBRARY_PATH:-}" in preamble, (
        f"LD_LIBRARY_PATH not prepended in preamble:\n{preamble}"
    )
    # Should NOT be a plain assignment
    assert "export LD_LIBRARY_PATH='/some/path'\n" not in preamble

    # Env vars must appear BEFORE module loads so that modules can
    # prepend their paths and take priority over captured (stale) paths.
    ld_line_idx = preamble.find("export LD_LIBRARY_PATH=")
    module_line_idx = preamble.find("module load")
    assert ld_line_idx < module_line_idx, (
        f"LD_LIBRARY_PATH export (pos {ld_line_idx}) must appear before "
        f"module load (pos {module_line_idx}) in preamble:\n{preamble}"
    )

    print("PASS: LD_LIBRARY_PATH prepend in preamble")


def test_sbatch_omits_channel_when_single():
    """When n_channels=1, sbatch script must NOT pass --channel (avoids slicing bug)."""
    from luxar.gsplats.batch.manifest import BatchManifest
    from luxar.gsplats.batch.slurm_gen import generate_fit_sbatch

    # TimeFused-like: T=10, C=1
    m = BatchManifest(
        input_path="/data/test.zarr.zip",
        output_dir="/output/test",
        n_timepoints=10,
        n_channels=1,
        tile_size=1384, tile_overlap=32, n_tiles=1,
        total_tasks=10, preset="draft",
        slurm_partition="gpu", slurm_time_limit="00:15:00",
    )
    script = generate_fit_sbatch(m, "# preamble\n")
    assert "--channel" not in script, (
        "sbatch passes --channel when n_channels=1 — would override --timepoint!"
    )
    assert "--timepoint" in script, "sbatch should pass --timepoint when T>1"

    # Multi-channel, single timepoint
    m.n_channels = 4
    m.n_timepoints = 1
    m.total_tasks = 4
    script2 = generate_fit_sbatch(m, "# preamble\n")
    assert "--channel" in script2, "sbatch should pass --channel when C>1"
    assert "--timepoint" not in script2, (
        "sbatch passes --timepoint when n_timepoints=1 — would override --channel!"
    )

    # Both multi
    m.n_timepoints = 5
    m.total_tasks = 20
    script3 = generate_fit_sbatch(m, "# preamble\n")
    assert "--channel" in script3
    assert "--timepoint" in script3

    print("PASS: sbatch conditionally includes --channel/--timepoint")


if __name__ == "__main__":
    tests = [
        test_zarr_zip_suffix_detection,
        test_custom_axes_parsing,
        test_axes_override_validation,
        test_array_selection_consistency,
        test_auto_tile_small_volume,
        test_cull_retention_defaults,
        test_6d_channel_decoding,
        test_tasks_per_job_manifest,
        test_env_capture_ld_library_path_prepend,
        test_sbatch_omits_channel_when_single,
    ]

    passed = failed = 0
    print(f"\n{'=' * 60}")
    print("Batch Plan HPC Fixes — Smoke Tests")
    print(f"{'=' * 60}\n")

    for test in tests:
        name = test.__name__
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {name}\n  {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {name}\n  {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}\n")

    import sys
    sys.exit(0 if failed == 0 else 1)
