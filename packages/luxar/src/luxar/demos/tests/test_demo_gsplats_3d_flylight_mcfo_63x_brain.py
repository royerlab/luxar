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
import json
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


def test_build_composite_writes_gain_balanced_uint16(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    volumes = {
        0: np.full((2, 3, 4), 10, dtype=np.uint8),
        1: np.full((2, 3, 4), 20, dtype=np.uint8),
        2: np.full((2, 3, 4), 5, dtype=np.uint8),
    }
    volumes[0][0, 0, 0] = 30
    volumes[1][0, 0, 0] = 40
    volumes[2][0, 0, 0] = 15
    monkeypatch.setattr(_demo, "signal_channel_indices", lambda _path: [0, 1, 2])
    monkeypatch.setattr(_demo, "reference_channel_index", lambda _path: 3)
    decoded = []

    def decode(_path: Path, channel: int) -> np.ndarray:
        decoded.append(channel)
        return volumes[channel]

    monkeypatch.setattr(_demo, "decode_h5j_channel", decode)
    monkeypatch.setattr(_demo, "BALANCE_PERCENTILE", 50.0)

    out = tmp_path / "composite.zarr"
    _demo.build_composite(tmp_path / "source.h5j", out)

    composite = np.asarray(zarr.open_group(str(out), mode="r")["composite"])
    assert composite.dtype == np.uint16
    assert composite[1, 1, 1] == 20
    assert composite[0, 0, 0] == 60
    assert decoded == [0, 1, 2, 0, 1, 2]


def test_colour_preserves_hue_metadata_and_off_grid_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fit = tmp_path / "fit.gsplats.zarr"
    centers = np.array([[0, 0, 0], [0, 0, 1], [-1, 0, 0]], dtype=np.float32)
    save_gsplats(
        fit,
        centers=centers,
        amplitudes=np.ones(3, dtype=np.float32),
        cholesky_factors=np.tile([1, 0, 1, 0, 0, 1], (3, 1)).astype(np.float32),
        fitting_info={"psnr_db": 41.5},
        fitting_config={"n_iters": 5000},
        provenance_info={"source_file": "source.h5j"},
        description="stamped fit",
        truncation_radius=2.5,
    )
    shape = (10, 10, 10)
    volumes = {channel: np.full(shape, 10, dtype=np.uint8) for channel in range(3)}
    for channel, (bright, dim) in enumerate(((250, 50), (50, 10), (20, 4))):
        volumes[channel][0, 0, 0] = bright
        volumes[channel][0, 0, 1] = dim
    monkeypatch.setattr(_demo, "signal_channel_indices", lambda _path: [0, 1, 2])
    monkeypatch.setattr(
        _demo, "decode_h5j_channel", lambda _path, channel: volumes[channel]
    )
    monkeypatch.setattr(_demo, "BALANCE_PERCENTILE", 50.0)

    out = tmp_path / "coloured.gsplats.zarr"
    _demo.colour_from_channels(tmp_path / "source.h5j", fit, out)

    loaded = _demo.load_gsplat_node(str(out))[0]
    colors = np.asarray(loaded.additive_sublods[0].colors)
    assert sorted(map(tuple, colors.tolist())) == [
        (0, 0, 0),
        (255, 51, 20),
        (255, 51, 20),
    ]
    root = zarr.open_group(str(out), mode="r")
    assert root["fitting"].attrs["psnr_db"] == pytest.approx(41.5)
    assert root["fitting/config"].attrs["n_iters"] == 5000
    assert root["provenance"].attrs["source_file"] == "source.h5j"
    assert root.attrs["description"] == "stamped fit"
    assert loaded.additive_sublods[0].truncation_radius == pytest.approx(2.5)


def test_colour_requires_exactly_three_signal_channels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fit = _tiny_store(tmp_path / "fit.gsplats.zarr", n=2)
    monkeypatch.setattr(_demo, "signal_channel_indices", lambda _path: [0, 1, 2, 3])
    with pytest.raises(ValueError, match="exactly 3 signal channels"):
        _demo.colour_from_channels(
            tmp_path / "source.h5j", fit, tmp_path / "coloured.gsplats.zarr"
        )


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


# ---------------------------------------------------------------------------
# Levelling angle
# ---------------------------------------------------------------------------
class _FakeSub:
    def __init__(self, centers, amplitudes):
        self.centers = centers
        self.amplitudes = amplitudes


class _FakeLeaf:
    def __init__(self, subs):
        self.additive_sublods = subs


def _tilted_cloud(angle_deg: float, n: int = 20000, seed: int = 0) -> _FakeLeaf:
    """An elongated cloud tilted by ``angle_deg`` in the (col 0, col 1) plane.

    Column 2 is the narrow axis, matching the shipped archive (extents
    663 x 303 x 167 um), so columns 0/1 are the view plane.
    """
    import math

    rng = np.random.default_rng(seed)
    long_ = rng.normal(0.0, 100.0, n)
    short = rng.normal(0.0, 20.0, n)
    thin = rng.normal(0.0, 5.0, n)
    t = math.radians(angle_deg)
    x = long_ * math.cos(t) - short * math.sin(t)
    y = long_ * math.sin(t) + short * math.cos(t)
    centers = np.stack([x, y, thin], axis=1).astype(np.float32)
    return _FakeLeaf([_FakeSub(centers, np.ones(n, dtype=np.float32))])


def _inplane_ratio(centers: np.ndarray) -> float:
    spread = centers.max(axis=0) - centers.min(axis=0)
    widest = np.sort(spread)[-2:]
    return float(widest[1] / widest[0])


@pytest.mark.parametrize("tilt", [0.0, 20.0, 48.84, -30.0, 70.0])
def test_levelling_angle_recovers_the_tilt_and_actually_levels(
    monkeypatch, tilt: float
) -> None:
    """The returned angle must undo the tilt for ANY tilt, including past 45 deg.

    Regression test for a real bug: the angle was originally derived in the plane
    of "the two widest axes". Past ~45 degrees the second axis becomes the wider
    one, the x/y roles swap, and the result is off by exactly 90 degrees — a
    70-degree tilt came back as -20, and levelling made the in-plane extent ratio
    WORSE (2.30 -> 1.15). `transform --rotate-z` acts on columns 0/1 regardless
    of which happens to be wider, so the plane must be fixed, not sorted.
    """
    import math

    import luxar.gsplats.tree as tree_mod

    leaf = _tilted_cloud(tilt)
    monkeypatch.setattr(tree_mod, "iter_leaves", lambda node: [node])

    angle = _demo.levelling_angle_deg(leaf)
    assert angle == pytest.approx(-tilt, abs=0.5), (
        f"expected about {-tilt} to undo a {tilt} deg tilt, got {angle}"
    )

    centers = np.asarray(leaf.additive_sublods[0].centers, dtype=np.float64)
    t = math.radians(angle)
    x, y = centers[:, 0], centers[:, 1]
    rotated = np.stack(
        [
            x * math.cos(t) - y * math.sin(t),
            x * math.sin(t) + y * math.cos(t),
            centers[:, 2],
        ],
        axis=1,
    )
    # Levelling must leave the cloud elongated in the view plane. The bound is
    # the load-bearing half of this test: the buggy version scored 1.15 here on a
    # 70-degree tilt. (Not asserted as a strict increase: for an already-level
    # cloud the empirical principal axis sits ~0.04 deg off the nominal one, so
    # rotating onto it can shave a hair off the axis-aligned extent.)
    assert _inplane_ratio(rotated) > 4.0
    assert _inplane_ratio(rotated) >= 0.99 * _inplane_ratio(centers)


def test_levelling_angle_has_the_cli_rotation_sign(tmp_path: Path, monkeypatch) -> None:
    import luxar.gsplats.tree as tree_mod

    leaf = _tilted_cloud(48.84, n=2000)
    monkeypatch.setattr(tree_mod, "iter_leaves", lambda node: [node])
    angle = _demo.levelling_angle_deg(leaf)
    centers = leaf.additive_sublods[0].centers
    source = tmp_path / "tilted.gsplats.zarr"
    output = tmp_path / "levelled.gsplats.zarr"
    save_gsplats(
        source,
        centers=centers,
        amplitudes=np.ones(len(centers), dtype=np.float32),
        cholesky_factors=np.tile([1, 0, 1, 0, 0, 1], (len(centers), 1)).astype(
            np.float32
        ),
    )

    _demo._luxar(
        "gsplat", "transform", str(source), str(output), "--rotate-z", f"{angle:.8f}"
    )

    levelled = _demo.load_gsplat_node(str(output))[0]
    transformed = np.asarray(levelled.additive_sublods[0].centers)
    assert _inplane_ratio(transformed) > 4.0
    assert _inplane_ratio(transformed) > 3.0 * _inplane_ratio(centers)


# ---------------------------------------------------------------------------
# The count the demo TELLS people must match the archive it ships.
#
# It was hardcoded in four places, one of them the on-screen caption, and the
# 5000-iteration refit changed the archive from 653,759 to 660,035 splats. Every
# gate stayed green -- they all checked the archive, and none compared it against
# the prose describing it. A peer session caught it by rendering the scene and
# reading the caption.
# ---------------------------------------------------------------------------


def test_the_hosted_splat_count_matches_the_measured_archive() -> None:
    """N_SPLATS must equal the sidecar's hosted-generation measurement.

    The sidecar is written by `gen_zenodo_records.py --refresh` from the
    archive's own stamps, so this ties the documented recompute recipe to the
    hosted bytes. Runtime strings use the loaded node's count because the
    bundled fallback is a different generation.
    """
    # Walk up to the sidecar rather than hardcoding a parent depth: this test
    # runs from a worktree as often as from the main checkout, and a fixed
    # `parents[N]` silently resolves to a different tree in one of them.
    here = Path(__file__).resolve()
    sidecar = next(
        (
            candidate
            for parent in here.parents
            if (
                candidate := parent / "scripts" / "demo_archive_characteristics.json"
            ).is_file()
        ),
        None,
    )
    assert sidecar is not None, f"no measurements sidecar found above {here}"
    chars = json.loads(sidecar.read_text())
    key = "gsplats_flylight_mcfo_63x/flylight_mcfo_63x.gsplats.zarr.zip"
    measured = chars["archives"][key]["n_splats"]

    assert measured is not None, (
        "the sidecar has no splat count for this archive, so this test cannot "
        "protect the caption -- re-run gen_zenodo_records.py --refresh"
    )
    assert _demo.N_SPLATS == measured, (
        f"the hosted recipe states {_demo.N_SPLATS:,} splats but the sidecar "
        f"measures {measured:,}"
    )


def test_no_stale_splat_count_literal_survives_in_a_user_visible_string() -> None:
    """The superseded count may appear only as labelled history.

    Four literals is how the desync happened, so a second literal creeping back
    into a caption or description is the regression worth blocking. The one
    permitted mention is the amplitude measurement recorded against the
    1000-iteration build.
    """
    source = Path(_demo.__file__).read_text()
    stale = [
        line.strip()
        for line in source.splitlines()
        if "653,759" in line
        and "bundled fallback" not in line
        and "1000-iteration build" not in line
        and "which this one replaces" not in line
    ]

    assert not stale, f"stale splat count outside its historical note: {stale}"
