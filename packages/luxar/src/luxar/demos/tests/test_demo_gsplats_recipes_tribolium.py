"""Tests for the recipe-gallery command descriptions."""

from luxar.demos import demo_gsplats_recipes_tribolium as demo


def test_adaptive_command_matches_the_fixed_demo_depth() -> None:
    command = demo._cli_for("adaptive")
    assert f"--levels {demo.LEVELS}" in command
