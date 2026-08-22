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
    old_cache = tmp_path / "codex_ch03.gsplats.zarr.zip"
    old_cache.touch()

    current = _demo.channel_cache_path(3)

    assert current == tmp_path / "codex_ch03.v2.gsplats.zarr.zip"
    assert current != old_cache
    assert not current.exists()
