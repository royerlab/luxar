"""Smoke tests for demo_gsplats_3d_flylight_mcfo_63x_brain.

Covers the scene builder's authored compositing and the serve call site — no
network, no git-LFS payload, no GPU fit. The demo is loaded by file path (see
test_demo_gsplats_3d_visible_human_head).

Why the serve test exists: the demo builds the scene *before* it serves, so a
bad argument to ``launch_viewer`` only surfaces after several minutes of work,
on the default (no-flag) path that ``--no-serve`` never exercises. Nothing else
in the tree covers a demo's serve call site.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import zarr

from luxar.gsplats.io.save_gsplats import save_gsplats

_DEMO_PATH = (
    Path(__file__).resolve().parents[1] / "demo_gsplats_3d_flylight_mcfo_63x_brain.py"
)


def _load_demo_module():
    name = "_luxar_demo_flylight_mcfo_for_tests"
    spec = importlib.util.spec_from_file_location(name, _DEMO_PATH)
    if spec is None or spec.loader is None:
        pytest.skip(f"Could not locate demo at {_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_demo = _load_demo_module()


def _tiny_store(path: Path, n: int = 16) -> Path:
    """A handful of coloured splats on disk — the shape the demo loads."""
    rng = np.random.default_rng(0)
    save_gsplats(
        path,
        centers=rng.uniform(-5.0, 5.0, (n, 3)).astype(np.float32),
        amplitudes=rng.uniform(0.2, 1.0, n).astype(np.float32),
        cholesky_factors=np.tile([1, 0, 1, 0, 0, 1], (n, 1)).astype(np.float32),
        colors=rng.uniform(0.1, 0.9, (n, 3)).astype(np.float32),
    )
    return path


class TestAuthoredCompositing:
    def test_scene_bakes_the_tuned_volumetric_window(self, tmp_path) -> None:
        """Pin the exposure story the module docstring explains.

        A silent revert to additive glow saturates every dense arbor to white
        and the MCFO hues — the whole point of the label — disappear. Storing
        the display maximum (1.101) as ``intensity`` instead of its reciprocal
        is the other easy slip, and renders the scene blown out.
        """
        src = _tiny_store(tmp_path / "tiny.gsplats.zarr")
        out = _demo.create_luxar_scene(src, tmp_path / "scene.luxar.zarr")

        attrs = dict(zarr.open_group(str(out), mode="r")["mcfo_neurons"].attrs)
        assert attrs["blending_mode"] == "volumetric"
        assert attrs["absorption"] == pytest.approx(0.10)
        assert attrs["opacity"] == pytest.approx(0.02)
        # intensity/offset are the stored form of the 0-1.101 display window:
        # intensity = 1/(hi-lo), offset = -lo/(hi-lo).
        assert attrs["intensity"] == pytest.approx(1.0 / 1.101)
        assert attrs["offset"] == pytest.approx(0.0)
        assert attrs["gamma"] == pytest.approx(1.0)


class TestServeCallSite:
    def test_default_path_serves_with_a_call_launch_viewer_accepts(
        self, tmp_path, monkeypatch
    ) -> None:
        """``autospec`` specs the mock from the real ``launch_viewer``.

        So a kwarg the barrel helper does not accept fails here, instead of
        as a TypeError at the very end of a full demo run.
        """
        src = _tiny_store(tmp_path / "tiny.gsplats.zarr")
        monkeypatch.setattr(_demo, "get_demos_output_dir", lambda: tmp_path)
        monkeypatch.setattr(_demo, "SCENE_NAME", "scene.luxar.zarr")
        monkeypatch.setattr(_demo, "resolve_data", lambda: src)
        monkeypatch.setattr(_demo, "NO_SERVE", False)
        monkeypatch.setattr(_demo, "SERVE_ONLY", False)

        with mock.patch.object(_demo, "launch_viewer", autospec=True) as spy:
            _demo.main()

        spy.assert_called_once_with(tmp_path / "scene.luxar.zarr")
        assert (tmp_path / "scene.luxar.zarr").exists()

    def test_no_serve_builds_without_serving(self, tmp_path, monkeypatch) -> None:
        src = _tiny_store(tmp_path / "tiny.gsplats.zarr")
        monkeypatch.setattr(_demo, "get_demos_output_dir", lambda: tmp_path)
        monkeypatch.setattr(_demo, "SCENE_NAME", "scene.luxar.zarr")
        monkeypatch.setattr(_demo, "resolve_data", lambda: src)
        monkeypatch.setattr(_demo, "NO_SERVE", True)
        monkeypatch.setattr(_demo, "SERVE_ONLY", False)

        with mock.patch.object(_demo, "launch_viewer", autospec=True) as spy:
            _demo.main()

        spy.assert_not_called()
        assert (tmp_path / "scene.luxar.zarr").exists()

    def test_serve_only_without_a_built_scene_does_not_serve(
        self, tmp_path, monkeypatch
    ) -> None:
        """``--serve-only`` on a missing scene must explain, not crash."""
        monkeypatch.setattr(_demo, "get_demos_output_dir", lambda: tmp_path)
        monkeypatch.setattr(_demo, "SCENE_NAME", "absent.luxar.zarr")
        monkeypatch.setattr(_demo, "NO_SERVE", False)
        monkeypatch.setattr(_demo, "SERVE_ONLY", True)

        def _fail() -> Path:
            raise AssertionError("--serve-only must not resolve the dataset")

        monkeypatch.setattr(_demo, "resolve_data", _fail)

        with mock.patch.object(_demo, "launch_viewer", autospec=True) as spy:
            _demo.main()

        spy.assert_not_called()
