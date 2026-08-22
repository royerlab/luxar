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
        the display maximum (2.723) as ``intensity`` instead of its reciprocal
        is the other easy slip, and renders the scene blown out.
        """
        src = _tiny_store(tmp_path / "tiny.gsplats.zarr")
        out = _demo.create_luxar_scene(src, tmp_path / "scene.luxar.zarr")

        attrs = dict(zarr.open_group(str(out), mode="r")["mcfo_neurons"].attrs)
        assert attrs["blending_mode"] == "volumetric"
        assert attrs["absorption"] == pytest.approx(0.81)
        assert attrs["opacity"] == pytest.approx(0.02)
        # intensity/offset are the stored form of the 0-2.723 display window:
        # intensity = 1/(hi-lo), offset = -lo/(hi-lo).
        assert attrs["intensity"] == pytest.approx(1.0 / 2.723)
        assert attrs["offset"] == pytest.approx(0.0)
        assert attrs["gamma"] == pytest.approx(1.0)


class TestCameraFramesTheBrain:
    """The viewer's default framing leaves the brain small; the demo bakes one.

    A camera that silently reverts to the default is invisible in every other
    assertion — the scene still loads, it just does not fill the canvas — so
    pin the geometry rather than merely the presence of a camera block.
    """

    def test_camera_is_baked_and_fills_the_canvas(self, tmp_path) -> None:
        import math

        src = _tiny_store(tmp_path / "tiny.gsplats.zarr")
        out = _demo.create_luxar_scene(src, tmp_path / "scene.luxar.zarr")

        root = zarr.open_group(str(out), mode="r")
        cam = dict(root.attrs)["viewer_config"]["camera"]
        assert "fov" not in cam
        assert tuple(cam["up"]) == (0.0, 1.0, 0.0)

        bounds = dict(root.attrs)["position_bounds"]
        bmin = np.asarray(bounds["min"], dtype=float)
        bmax = np.asarray(bounds["max"], dtype=float)
        centre = (bmin + bmax) / 2.0
        target = np.asarray(cam["target"], dtype=float)
        position = np.asarray(cam["position"], dtype=float)

        # Looks at the middle of the object, from straight down +Z (the thin
        # axis after the pipeline's rotate-y), so the brain is seen face-on.
        assert target == pytest.approx(centre, abs=1e-3)
        assert position[:2] == pytest.approx(centre[:2], abs=1e-3)
        assert position[2] > target[2]

        # ...and close enough that the object really does fill the frame at the
        # design aspect. A default-framed camera sits much further back.
        #
        # Measured at the NEAR FACE, where the frustum is narrowest — the
        # centre-plane fit this replaced put the near corners ~7% outside the
        # frame, and a centre-plane assertion could not see that.
        width, height = float(bmax[0] - bmin[0]), float(bmax[1] - bmin[1])
        half_depth = float(bmax[2] - bmin[2]) / 2.0
        near = float(position[2] - target[2]) - half_depth
        visible_h = 2.0 * near * math.tan(math.radians(_demo.CAMERA_FOV / 2.0))
        visible_w = visible_h * _demo.CAMERA_ASPECT
        assert max(width / visible_w, height / visible_h) == pytest.approx(
            _demo.CAMERA_FILL, rel=1e-6
        )

    def test_near_face_corners_are_inside_the_frame(self, tmp_path) -> None:
        """Project the actual bbox corners; none may fall outside the frustum.

        Independent of the distance formula — it re-derives nothing, it just
        asks whether the eight corners land in view. The pre-fix centre-plane
        camera fails this on the four near corners.
        """
        import math

        src = _tiny_store(tmp_path / "tiny.gsplats.zarr")
        out = _demo.create_luxar_scene(src, tmp_path / "scene.luxar.zarr")

        root = zarr.open_group(str(out), mode="r")
        cam = dict(root.attrs)["viewer_config"]["camera"]
        bounds = dict(root.attrs)["position_bounds"]
        bmin = np.asarray(bounds["min"], dtype=float)
        bmax = np.asarray(bounds["max"], dtype=float)
        eye = np.asarray(cam["position"], dtype=float)

        tan_half = math.tan(math.radians(_demo.CAMERA_FOV / 2.0))
        worst_x = worst_y = 0.0
        for xi in (bmin[0], bmax[0]):
            for yi in (bmin[1], bmax[1]):
                for zi in (bmin[2], bmax[2]):
                    depth = eye[2] - zi  # camera looks down -Z at the target
                    assert depth > 0, "a bbox corner is behind the camera"
                    half_h = depth * tan_half
                    worst_y = max(worst_y, abs(yi - eye[1]) / half_h)
                    worst_x = max(
                        worst_x, abs(xi - eye[0]) / (half_h * _demo.CAMERA_ASPECT)
                    )
        # <= 1.0 means inside; CAMERA_FILL is the headroom the framing asked for.
        assert max(worst_x, worst_y) <= 1.0
        assert max(worst_x, worst_y) == pytest.approx(_demo.CAMERA_FILL, rel=1e-6)

    def test_framing_is_calibrated_at_the_declared_aspect(self) -> None:
        """`CAMERA_ASPECT` is a declared calibration point, not a fudge factor.

        Pin the documented consequence: at exactly the declared aspect a
        width-bound object occupies `CAMERA_FILL` of the width; wider viewports
        leave margin. A change to either constant that silently broke that
        relationship would otherwise only show up on screen.
        """
        import math

        width, height, half_depth = 663.0, 303.0, 83.3
        d = _demo.camera_distance(width, height, half_depth)
        near = d - half_depth
        visible_h = 2.0 * near * math.tan(math.radians(_demo.CAMERA_FOV / 2.0))

        assert width / (visible_h * _demo.CAMERA_ASPECT) == pytest.approx(
            _demo.CAMERA_FILL
        )
        assert height / visible_h < _demo.CAMERA_FILL  # width is what binds
        # A wider window leaves margin rather than cropping.
        assert width / (visible_h * (_demo.CAMERA_ASPECT + 0.4)) < _demo.CAMERA_FILL


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
