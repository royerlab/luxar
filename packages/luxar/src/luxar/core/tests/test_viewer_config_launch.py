"""ViewerConfig.text_scale and the LaunchConfig machine-settings block."""

from __future__ import annotations

import pytest

from luxar.core.viewer_config import LaunchConfig, ViewerConfig


def test_text_scale_round_trips_and_is_range_checked() -> None:
    vc = ViewerConfig(text_scale=0.8)
    assert vc.to_dict()["text_scale"] == 0.8
    assert ViewerConfig.from_dict(vc.to_dict()).text_scale == 0.8
    for bad in (0.1, 5.0):
        with pytest.raises(ValueError, match="text_scale"):
            ViewerConfig(text_scale=bad)


def test_text_scale_is_not_a_per_waypoint_rendering_patch() -> None:
    assert "text_scale" not in ViewerConfig._RENDERING_FIELDS
    assert "text_scale" in ViewerConfig._SIMPLE_FIELDS


def test_launch_block_serializes_sparsely_and_round_trips() -> None:
    vc = ViewerConfig(launch=LaunchConfig(renderer="webgpu", workers=16, prefetch=12))
    assert vc.to_dict()["launch"] == {
        "prefetch": 12,
        "renderer": "webgpu",
        "workers": 16,
    }
    back = ViewerConfig.from_dict(vc.to_dict())
    assert back.launch == LaunchConfig(renderer="webgpu", workers=16, prefetch=12)
    assert "launch" not in ViewerConfig(launch=LaunchConfig()).to_dict()


def test_launch_query_is_the_viewer_url_fragment() -> None:
    assert (
        LaunchConfig(renderer="webgpu", workers=16, prefetch=12).query()
        == "&renderer=webgpu&workers=16&prefetch=12"
    )
    assert LaunchConfig(workers=0).query() == "&workers=0"
    assert LaunchConfig().query() == ""


@pytest.mark.parametrize(
    "kwargs",
    [
        {"renderer": "vulkan"},
        {"workers": 17},
        {"workers": -1},
        {"workers": True},
        {"prefetch": 0},
        {"prefetch": 13},
        {"prefetch": 4.0},
    ],
)
def test_launch_rejects_values_the_viewer_would_not_accept(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="launch"):
        LaunchConfig(**kwargs)


def test_launch_must_be_a_launch_config() -> None:
    with pytest.raises(ValueError, match="launch"):
        ViewerConfig(launch={"renderer": "webgpu"})  # type: ignore[arg-type]
