"""Regression tests for CODEX pancreas cache compatibility."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_DEMO_PATH = Path(__file__).resolve().parents[1] / "demo_gsplats_2d_codex_pancreas.py"


def _load_demo_module():
    name = "_luxar_demo_codex_pancreas_for_tests"
    spec = importlib.util.spec_from_file_location(name, _DEMO_PATH)
    if spec is None or spec.loader is None:
        pytest.skip(f"Could not locate demo at {_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_demo = _load_demo_module()


def test_cache_version_excludes_pre_transpose_flat_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The old filename contains flat `(row, col)` fits and must never be reused."""
    monkeypatch.setattr(_demo, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(_demo, "RECOMPUTE", False)
    for channel_index in range(_demo.N_CHANNELS):
        (tmp_path / f"codex_ch{channel_index:02d}.gsplats.zarr.zip").touch()

    loaded_channels = []
    fitted_paths = []

    def fake_load_channel(tiff_dir, channel_config):
        loaded_channels.append((tiff_dir, channel_config))
        return object()

    def fake_fit_channel_tiled(image, channel_name, cache_file):
        fitted_paths.append(cache_file)
        return object()

    monkeypatch.setattr(_demo, "load_channel", fake_load_channel)
    monkeypatch.setattr(_demo, "fit_channel_tiled", fake_fit_channel_tiled)

    tiff_dir = tmp_path / "source"
    cache_paths, fitted = _demo.fit_all_channels(tiff_dir)

    expected_paths = [
        tmp_path / f"codex_ch{channel_index:02d}.v2.gsplats.zarr.zip"
        for channel_index in range(_demo.N_CHANNELS)
    ]
    assert cache_paths == expected_paths
    assert fitted_paths == expected_paths
    assert len(loaded_channels) == _demo.N_CHANNELS
    assert all(loaded_from == tiff_dir for loaded_from, _ in loaded_channels)
    assert all(result is not None for result in fitted)
