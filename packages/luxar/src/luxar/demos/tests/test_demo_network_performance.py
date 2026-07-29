"""Regression tests for the network-performance demo CLI boundary."""

from __future__ import annotations

import pytest

from luxar.demos import demo_network_performance as demo


def test_documented_space_separated_args_drive_generation_and_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def _generate(output_path, n_points=1_000_000, seed=42):
        seen["n_points"] = n_points

    def _launch(output_path, *, serve_args=None):
        seen["serve_args"] = serve_args

    monkeypatch.setattr(demo, "generate_performance_test_dataset", _generate)
    monkeypatch.setattr(demo, "launch_viewer", _launch)

    demo.main(["--points", "2000000", "--profile", "satellite"])

    assert seen == {
        "n_points": 2_000_000,
        "serve_args": ["--profile", "satellite"],
    }


def test_equals_form_and_boolean_flags_are_preserved() -> None:
    args = demo.parse_args(
        ["--points=42", "--profile=3g", "--no-serve", "--no-simulation"]
    )

    assert args.points == 42
    assert args.profile == "3g"
    assert args.no_serve is True
    assert args.no_simulation is True


def test_unknown_arguments_are_rejected() -> None:
    with pytest.raises(SystemExit) as exc_info:
        demo.parse_args(["--profil", "satellite"])

    assert exc_info.value.code == 2
