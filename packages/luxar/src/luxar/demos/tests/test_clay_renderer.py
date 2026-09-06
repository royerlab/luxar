"""Tests for the GPU clay renderer.

The geometry helpers are pure numpy and always run. The rendering tests need a
GL context (moderngl standalone), which CI runners without a display or EGL do
not have: they are skipped there rather than failed, because the demo itself
treats "no GPU context" as "no turntables", not as an error.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from luxar.demos import _clay_renderer as cr


def test_load_obj_indexes_v_and_vn_corners(tmp_path: Path) -> None:
    obj = tmp_path / "m.obj"
    obj.write_text(
        "# two triangles sharing vertices, corners out of order\n"
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\n"
        "vn 0 0 1\nvn 0 0 -1\n"
        "f 1//1 2//1 3//1\n"
        "f 3//2 2//2 4//2\n"
    )
    pos, nrm = cr.load_obj(obj)
    assert pos.shape == (6, 3) and nrm.shape == (6, 3)
    assert pos.dtype == np.float32
    np.testing.assert_array_equal(pos[3], [0, 1, 0])  # face 2 corner 1 = v3
    np.testing.assert_array_equal(nrm[0], [0, 0, 1])
    np.testing.assert_array_equal(nrm[5], [0, 0, -1])


def test_load_obj_without_normals_computes_flat_ones(tmp_path: Path) -> None:
    obj = tmp_path / "m.obj"
    obj.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    pos, nrm = cr.load_obj(obj)
    assert pos.shape == (3, 3)
    np.testing.assert_allclose(nrm, [[0, 0, 1]] * 3)


def test_load_obj_rejects_empty_and_non_triangle_files(tmp_path: Path) -> None:
    empty = tmp_path / "e.obj"
    empty.write_text("# nothing\n")
    with pytest.raises(ValueError):
        cr.load_obj(empty)
    quad = tmp_path / "q.obj"
    quad.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n")
    with pytest.raises(ValueError):
        cr.load_obj(quad)


def test_principal_frame_stands_the_long_axis_up_and_is_right_handed() -> None:
    rng = np.random.default_rng(0)
    # An ellipsoid cloud with axes 10 (along a tilted direction), 3, 1.
    local = rng.normal(size=(5000, 3)) * np.array([10.0, 3.0, 1.0])
    tilt = np.array(
        [[0.6, -0.8, 0.0], [0.8, 0.6, 0.0], [0.0, 0.0, 1.0]]
    )  # rotate the long axis into the xy plane
    pts = local @ tilt.T + np.array([100.0, -5.0, 7.0])
    centre, rot = cr.principal_frame(pts)
    np.testing.assert_allclose(centre, [100.0, -5.0, 7.0], atol=0.5)
    assert np.linalg.det(rot) > 0  # right-handed
    np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-5)
    std = ((pts - centre) @ rot.T).std(axis=0)
    assert std[1] > std[0] > std[2]  # long axis -> y (vertical), then x, then z
    assert std[1] == pytest.approx(10.0, rel=0.1)


def test_perspective_and_rotation_conventions() -> None:
    p = cr.perspective(90.0, 1.0, 1.0, 3.0)
    assert p[0, 0] == pytest.approx(1.0) and p[3, 2] == -1.0
    # Positive rotation about +y moves the front (+z) point to the RIGHT (+x).
    front = cr.rotation_y(30.0) @ np.array([0.0, 0.0, 1.0, 1.0])
    assert front[0] > 0 and front[2] > 0


def test_ffmpeg_pipe_command_encodes_vp9_with_alpha_from_raw_rgba() -> None:
    cmd = cr.ffmpeg_pipe_command("/usr/bin/ffmpeg", 768, 30, Path("/out.webm"))
    assert cmd[0] == "/usr/bin/ffmpeg"
    assert cmd[cmd.index("-f") + 1] == "rawvideo"
    assert cmd[cmd.index("-s") + 1] == "768x768"
    assert cmd[cmd.index("-r") + 1] == "30"
    assert cmd[cmd.index("-vf") + 1] == "vflip"  # OpenGL rows are bottom-up
    assert cmd[cmd.index("-c:v") + 1] == "libvpx-vp9"
    assert cmd[cmd.index("-pix_fmt", cmd.index("-i")) + 1] == "yuva420p"
    assert cmd[cmd.index("-auto-alt-ref") + 1] == "0"  # keeps the alpha plane
    assert cmd[-1] == "/out.webm" and "-an" in cmd


def _gpu_renderer(**kwargs: object) -> cr.ClayRenderer:
    pytest.importorskip("moderngl")
    try:
        return cr.ClayRenderer(128, **kwargs)  # type: ignore[arg-type]
    except Exception as e:  # noqa: BLE001 — no GL context on this runner
        pytest.skip(f"no GPU context: {e}")


def _slotted_block() -> cr.Mesh:
    """A box with a deep narrow slot cut into its front face (AO must darken it)."""
    tris: list[list[float]] = []

    def quad(a, b, c, d, n):  # noqa: ANN001
        for p in (a, b, c, a, c, d):
            tris.append([*p, *n])

    # Front face split around a slot x in [-0.1, 0.1], depth 0.8 into the box.
    quad((-1, -1, 1), (-0.1, -1, 1), (-0.1, 1, 1), (-1, 1, 1), (0, 0, 1))
    quad((0.1, -1, 1), (1, -1, 1), (1, 1, 1), (0.1, 1, 1), (0, 0, 1))
    quad((-0.1, -1, 0.2), (0.1, -1, 0.2), (0.1, 1, 0.2), (-0.1, 1, 0.2), (0, 0, 1))
    quad((-0.1, -1, 1), (-0.1, -1, 0.2), (-0.1, 1, 0.2), (-0.1, 1, 1), (1, 0, 0))
    quad((0.1, -1, 0.2), (0.1, -1, 1), (0.1, 1, 1), (0.1, 1, 0.2), (-1, 0, 0))
    arr = np.array(tris, dtype=np.float32)
    return cr.Mesh(positions=arr[:, :3], normals=arr[:, 3:], color=(0.8, 0.8, 0.8))


def test_gpu_frame_has_straight_alpha_coverage_and_ambient_occlusion() -> None:
    r_ao = _gpu_renderer(ao_strength=1.0)
    try:
        r_ao.set_meshes([_slotted_block()])
        rgba = np.frombuffer(r_ao.render(0.0), np.uint8).reshape(128, 128, 4)
    finally:
        r_ao.release()
    r_flat = _gpu_renderer(ao_strength=0.0)
    try:
        r_flat.set_meshes([_slotted_block()])
        flat = np.frombuffer(r_flat.render(0.0), np.uint8).reshape(128, 128, 4)
    finally:
        r_flat.release()
    # The box fills roughly the middle 65% of the frame (bounding-sphere fit):
    # column 64 is the slot floor, column 40 the open front face beside it.
    slot_px, face_px = (64, 64), (64, 40)
    # Transparent background, opaque geometry, straight (not black-fringed) colour.
    assert rgba[0, 0, 3] == 0 and rgba[slot_px][3] == 255
    assert rgba[face_px][3] == 255 and rgba[face_px][:3].min() > 60
    # The slot floor is darker WITH ambient occlusion than the open face beside
    # it, and darker than the same pixel rendered without AO.
    slot, face = rgba[slot_px][:3].mean(), rgba[face_px][:3].mean()
    assert slot < face - 10
    assert slot < flat[slot_px][:3].mean() - 10
    # Without AO the slot floor and the face are lit alike (same normal).
    assert abs(float(flat[slot_px][:3].mean()) - float(flat[face_px][:3].mean())) < 6
