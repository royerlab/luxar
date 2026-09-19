"""The demo-side environment bake: gating, and that it never fails a build."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

import pytest

from luxar.demos._support.runtime.environment import bake_scene_environment


class _Attach:
    array_name = "faces-deadbeef"


class _Report:
    attach = _Attach()
    resolution = 128
    probe = "auto"


@contextlib.contextmanager
def _scene(
    tmp_path: Path, environment: dict | None, *, baked: bool = False
) -> Iterator[Path]:
    """A store whose `viewer_config.environment` reads back as `environment`."""
    store = tmp_path / "scene.luxar.zarr"
    store.mkdir(parents=True, exist_ok=True)
    if baked:
        (store / "environment").mkdir(exist_ok=True)
    attrs = {"viewer_config": {"environment": environment} if environment else {}}
    with patch(
        "luxar.demos._support.runtime.environment.read_node_attrs", return_value=attrs
    ):
        yield store


def test_bakes_a_live_capture_scene_with_its_own_probe_and_resolution(
    tmp_path: Path,
) -> None:
    """The baked map must be the capture the viewer WOULD have made.

    Using the scene's own probe and resolution is what makes baking a freeze
    rather than a different look.
    """
    cfg = {"source": "scene", "probe": "node:shell", "resolution": 256}
    with _scene(tmp_path, cfg) as store:
        with patch(
            "luxar.cli.env_ops.bake.bake_environment", return_value=_Report()
        ) as bake:
            assert bake_scene_environment(store) is True
    assert bake.call_count == 1
    assert bake.call_args.kwargs["probe"] == "node:shell"
    assert bake.call_args.kwargs["resolution"] == 256


@pytest.mark.parametrize(
    "environment",
    [None, {"source": "room"}, {"source": "hdri", "url": "x.hdr"}],
    ids=["no-environment", "room", "hdri"],
)
def test_does_nothing_for_a_scene_that_cannot_gain_from_a_bake(
    tmp_path: Path, environment: dict | None
) -> None:
    """`hdri` is already a fixed file and `room` is identical everywhere."""
    with _scene(tmp_path, environment) as store:
        with patch("luxar.cli.env_ops.bake.bake_environment") as bake:
            assert bake_scene_environment(store) is False
    assert bake.call_count == 0


def test_an_already_baked_store_is_left_alone(tmp_path: Path) -> None:
    with _scene(tmp_path, {"source": "scene"}, baked=True) as store:
        with patch("luxar.cli.env_ops.bake.bake_environment") as bake:
            assert bake_scene_environment(store) is False
    assert bake.call_count == 0


def test_a_failing_bake_never_fails_the_demo(tmp_path: Path) -> None:
    """No viewer dist, no Playwright, no browser: the demo is still finished.

    The scene falls back to the live capture it would have used anyway, so a
    missing development checkout must not turn a completed build into an error.
    """
    with _scene(tmp_path, {"source": "scene"}) as store:
        with patch(
            "luxar.cli.env_ops.bake.bake_environment",
            side_effect=RuntimeError("no viewer dist"),
        ):
            assert bake_scene_environment(store) is False
