"""``luxar env bake`` / ``luxar env attach`` through the CLI.

The bake's browser half is replaced by an injected driver that writes a synthetic
container, so what is exercised here is the ORCHESTRATION — serve the store and the
viewer from one process, hand the driver a ``?bake-env`` URL against them, collect
the container, attach it — without a browser or a GPU.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import zarr
from typer.testing import CliRunner

from luxar import Dimensions, LuxarZarrCompiler
from luxar._zarr_compat import open_group
from luxar.cli.env_ops import bake as bake_module
from luxar.cli.env_ops.bake import bake_environment
from luxar.cli.main import app
from luxar.environment import ENVIRONMENT_FORMAT, FACE_ORDER, pack
from luxar.typing_utils.constants import ENVIRONMENT_GROUP

from ._testing import normalized_cli_output

runner = CliRunner()
RES = 16


def _scene(path: Path) -> str:
    with LuxarZarrCompiler(path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_points(
            "cloud", np.random.default_rng(1).random((200, 3)).astype(np.float32)
        )
    return str(dict(open_group(path, mode="r").attrs)["content_hash"])


def _container(scene_hash: str) -> bytes:
    faces = np.full((6, RES, RES, 4), 0x3C00, dtype=np.uint16)
    return pack(
        {
            "format": ENVIRONMENT_FORMAT,
            "face_order": list(FACE_ORDER),
            "coordinate_system": "webgl",
            "probe": {"spec": "auto", "position": [0, 0, 0]},
            "resolution": RES,
            "scene_content_hash": scene_hash,
        },
        faces,
    )


def test_env_attach_cli(tmp_path) -> None:
    store = tmp_path / "scene.luxar.zarr"
    scene_hash = _scene(store)
    blob = tmp_path / "scene.env.bin"
    blob.write_bytes(_container(scene_hash))

    result = runner.invoke(app, ["env", "attach", str(store), str(blob)])
    assert result.exit_code == 0, result.stdout
    out = normalized_cli_output(result)
    assert "attached: environment/faces-" in out
    root = zarr.open_group(str(store), mode="r")
    assert dict(root[ENVIRONMENT_GROUP].attrs)["scene_content_hash"] == scene_hash

    again = runner.invoke(app, ["env", "attach", str(store), str(blob)])
    assert again.exit_code == 0
    assert "unchanged" in normalized_cli_output(again)


def test_env_attach_cli_refuses_a_stale_bake(tmp_path) -> None:
    store = tmp_path / "scene.luxar.zarr"
    _scene(store)
    blob = tmp_path / "old.env.bin"
    blob.write_bytes(_container("0000000000000000"))
    result = runner.invoke(app, ["env", "attach", str(store), str(blob)])
    assert result.exit_code == 1
    assert "scene changed since the bake" in normalized_cli_output(result)
    forced = runner.invoke(app, ["env", "attach", str(store), str(blob), "--force"])
    assert forced.exit_code == 0, forced.stdout


@pytest.fixture
def built_viewer(monkeypatch, tmp_path) -> Path:
    """A minimal viewer dist so the served viewer mount has something to serve."""
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html><body>viewer</body></html>")
    monkeypatch.setattr("luxar.cli.serving.get_viewer_dist_path", lambda: dist)
    monkeypatch.setattr(
        "luxar.cli.env_ops.bake.ensure_viewer_built", lambda auto_build=True: True
    )
    return dist


def test_bake_environment_serves_drives_and_attaches(tmp_path, built_viewer) -> None:
    store = tmp_path / "scene.luxar.zarr"
    scene_hash = _scene(store)
    seen: dict[str, object] = {}

    def fake_driver(url: str, out_path: Path, timeout_s: float) -> None:
        seen["url"] = url
        seen["timeout"] = timeout_s
        # The URL the driver gets must point at a LIVE server: fetch the store's
        # root document and the viewer through it, as the browser would.
        src = url.split("?src=", 1)[1].split("&", 1)[0]
        with urllib.request.urlopen(f"{src}/zarr.json", timeout=5) as response:
            root_doc = json.loads(response.read())
        assert root_doc["attributes"]["content_hash"] == scene_hash
        viewer = url.split("?", 1)[0]
        with urllib.request.urlopen(viewer, timeout=5) as response:
            assert b"viewer" in response.read()
        out_path.write_bytes(_container(scene_hash))

    report = bake_environment(
        store, probe="node:cloud", resolution=RES, driver=fake_driver, timeout=42
    )
    url = str(seen["url"])
    assert "&debug&bake-env&probe=node:cloud&env-resolution=16" in url
    assert url.startswith("http://127.0.0.1:")
    assert seen["timeout"] == 42
    assert report.attach is not None and report.attach.status == "attached"
    assert report.resolution == RES
    # The temporary container was cleaned up after a successful attach…
    assert not report.container.exists()
    # …and the store carries the map.
    root = zarr.open_group(str(store), mode="r")
    assert dict(root[ENVIRONMENT_GROUP].attrs)["faces"] == report.attach.array_name
    assert dict(root.attrs)["content_hash"] == scene_hash


def test_bake_environment_keeps_the_container_and_can_skip_attach(
    tmp_path, built_viewer
) -> None:
    store = tmp_path / "scene.luxar.zarr"
    scene_hash = _scene(store)
    out = tmp_path / "kept.env.bin"

    def fake_driver(url: str, out_path: Path, timeout_s: float) -> None:
        out_path.write_bytes(_container(scene_hash))

    report = bake_environment(store, out=out, attach=False, driver=fake_driver)
    assert report.attach is None
    assert out.exists() and report.container == out
    assert ENVIRONMENT_GROUP not in zarr.open_group(str(store), mode="r")


def test_bake_environment_preserves_a_temporary_container_without_attach(
    tmp_path, built_viewer, monkeypatch
) -> None:
    store = tmp_path / "scene.luxar.zarr"
    scene_hash = _scene(store)
    temporary = tmp_path / "temporary.env.bin"

    def fake_mkstemp(**_kwargs) -> tuple[int, str]:
        temporary.touch()
        return os.open(temporary, os.O_WRONLY), str(temporary)

    def fake_driver(_url: str, out_path: Path, _timeout_s: float) -> None:
        out_path.write_bytes(_container(scene_hash))

    monkeypatch.setattr("luxar.cli.env_ops.bake.tempfile.mkstemp", fake_mkstemp)
    report = bake_environment(store, attach=False, driver=fake_driver)
    assert report.container == temporary
    assert temporary.exists()


def test_bake_environment_removes_a_temporary_container_after_driver_failure(
    tmp_path, built_viewer, monkeypatch
) -> None:
    store = tmp_path / "scene.luxar.zarr"
    _scene(store)
    temporary = tmp_path / "temporary.env.bin"

    def fake_mkstemp(**_kwargs) -> tuple[int, str]:
        temporary.touch()
        return os.open(temporary, os.O_WRONLY), str(temporary)

    def failing_driver(_url: str, _out_path: Path, _timeout_s: float) -> None:
        raise RuntimeError("driver failed")

    monkeypatch.setattr("luxar.cli.env_ops.bake.tempfile.mkstemp", fake_mkstemp)
    with pytest.raises(RuntimeError, match="driver failed"):
        bake_environment(store, driver=failing_driver)
    assert not temporary.exists()


def test_bake_environment_refuses_a_missing_viewer_or_bad_resolution(
    tmp_path, monkeypatch
) -> None:
    store = tmp_path / "scene.luxar.zarr"
    _scene(store)
    with pytest.raises(ValueError, match="--resolution"):
        bake_environment(store, resolution=4, driver=lambda *a: None)
    monkeypatch.setattr(
        bake_module, "ensure_viewer_built", lambda auto_build=True: False
    )
    with pytest.raises(RuntimeError, match="viewer is not built"):
        bake_environment(store, driver=lambda *a: None)


def test_bake_environment_refuses_a_compressed_store_before_viewer_setup(
    tmp_path, monkeypatch
) -> None:
    archive = tmp_path / "scene.luxar.zarr.zip"
    archive.write_bytes(b"not opened")

    def unexpected_viewer_check(*, auto_build: bool) -> bool:
        raise AssertionError("viewer setup must not run for an unwritable store")

    monkeypatch.setattr(bake_module, "ensure_viewer_built", unexpected_viewer_check)
    with pytest.raises(ValueError, match="uncompressed .zarr directory"):
        bake_environment(archive, driver=lambda *a: None)


def test_env_bake_cli_funnels_errors(tmp_path) -> None:
    result = runner.invoke(app, ["env", "bake", str(tmp_path / "missing.luxar.zarr")])
    assert result.exit_code == 1
    assert "Scene not found" in normalized_cli_output(result)


def test_env_attach_cli_funnels_a_missing_store(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "env",
            "attach",
            str(tmp_path / "missing.luxar.zarr"),
            str(tmp_path / "missing.env.bin"),
        ],
    )
    assert result.exit_code == 1
    assert "Scene not found" in normalized_cli_output(result)
