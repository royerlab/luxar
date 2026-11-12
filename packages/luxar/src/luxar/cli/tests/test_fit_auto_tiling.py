"""``--tiling auto``: the whole-volume budget follows the fit device's memory."""

from __future__ import annotations

import pytest

from luxar.cli.gsplat_ops.fitting import fit_utils
from luxar.cli.gsplat_ops.fitting.fit_utils import (
    auto_whole_volume_voxels,
    resolve_tiling,
)

GB = 1024**3


@pytest.mark.parametrize(
    ("memory", "expected_mvox"),
    [
        (None, 64),  # CPU / unknown device: the device-blind floor
        (8 * GB, 64),  # an 8 GB card stays at the floor
        (24 * GB, 192),  # (12 GB - 0.5 GB) / 64 B
        (128 * GB, 256),  # the ceiling
    ],
)
def test_budget_scales_with_device_memory(
    monkeypatch: pytest.MonkeyPatch, memory: int | None, expected_mvox: int
) -> None:
    monkeypatch.setattr(fit_utils, "_device_memory_bytes", lambda device: memory)
    got = auto_whole_volume_voxels("cuda")
    assert got == pytest.approx(expected_mvox * 1e6, rel=0.03)


def test_a_67_mvoxel_stack_fits_whole_on_a_large_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tester's 256x512x512 stack: whole on a 24 GB card, tiled on the CPU."""
    shape = (256, 512, 512)
    monkeypatch.setattr(fit_utils, "_device_memory_bytes", lambda device: 24 * GB)
    assert resolve_tiling("auto", shape, 256, False, device="cuda") == "none"
    monkeypatch.setattr(fit_utils, "_device_memory_bytes", lambda device: None)
    assert resolve_tiling("auto", shape, 256, False, device="cpu") == "uniform"
    assert resolve_tiling("auto", shape, 256, True, device="cpu") == "content"


def test_cpu_device_reports_no_memory() -> None:
    assert fit_utils._device_memory_bytes("cpu") is None
    assert fit_utils._device_memory_bytes(None) is None


def test_recipe_or_jobs_keep_tiling_where_the_floor_would_tile(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A big GPU fits 67 Mvoxel whole, but --recipe / -j need tiles: keep them."""
    shape = (256, 512, 512)
    monkeypatch.setattr(fit_utils, "_device_memory_bytes", lambda device: 128 * GB)
    assert resolve_tiling("auto", shape, 256, False, device="cuda") == "none"
    got = resolve_tiling("auto", shape, 256, False, device="cuda", wants_tiles=True)
    assert got == "uniform"
    assert "need tiles" in capsys.readouterr().out
    # below the device-blind floor nothing changes: a small stack stays whole
    small = (84, 580, 576)
    assert (
        resolve_tiling("auto", small, 256, False, device="cuda", wants_tiles=True)
        == "none"
    )
