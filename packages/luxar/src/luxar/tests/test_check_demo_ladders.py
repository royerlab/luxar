"""Tests for the built-demo additive-ladder structural gate."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest
import zarr

from luxar.utils import paths as luxar_paths

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _load_checker() -> ModuleType:
    script_path = Path(__file__).resolve().parents[5] / "scripts/check_demo_ladders.py"
    spec = importlib.util.spec_from_file_location("check_demo_ladders", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


def _make_leaf(
    path: Path,
    sizes: list[int] | None,
    *,
    declared_total: int | None = None,
) -> zarr.Group:
    root = zarr.open_group(path, mode="w")
    leaf = root.create_group("leaf")
    total = declared_total if declared_total is not None else sum(sizes or [])
    leaf.attrs.update(
        {
            "type": "points",
            "n_points": total,
            "n_additive_sublods": len(sizes) if sizes else 1,
        }
    )
    for index, size in enumerate(sizes or []):
        level = leaf.create_group(f"additive_{index}")
        level.attrs.update({"type": "points", "n_points": size})
    return leaf


def _check(leaf: zarr.Group, **overrides: int | float) -> tuple[str, str]:
    options: dict[str, int | float] = {
        "min_elements": 0,
        "max_share": 0.6,
        "max_level_elements": 10_000,
        "min_sublods": 3,
    }
    options.update(overrides)
    return checker.check_leaf(leaf, **options)


@pytest.mark.parametrize(
    ("sizes", "expected_status"),
    [
        ([20, 20, 60], "ok"),
        ([19, 20, 61], "fail"),
    ],
)
def test_relative_share_limit_is_strictly_greater_than_threshold(
    tmp_path: Path, sizes: list[int], expected_status: str
) -> None:
    """Exactly 60% passes; the first representable test case above it fails."""
    leaf = _make_leaf(tmp_path / f"share-{sizes[-1]}.zarr", sizes)
    status, message = _check(leaf)

    assert status == expected_status
    assert f"{sizes[-1]:.1f}%" in message
    if expected_status == "fail":
        assert "degenerate ladder" in message


def test_absolute_level_cap_catches_large_balanced_increment(tmp_path: Path) -> None:
    """A healthy relative split can still exceed the browser commit budget."""
    leaf = _make_leaf(tmp_path / "absolute.zarr", [500, 500])
    status, message = _check(leaf, max_level_elements=499)

    assert status == "fail"
    assert "500 elements" in message
    assert "absolute commit cap" in message


def test_large_unladdered_leaf_fails_but_small_leaf_skips(tmp_path: Path) -> None:
    large = _make_leaf(
        tmp_path / "large.zarr", None, declared_total=checker.DEFAULT_MIN_ELEMENTS + 1
    )
    small = _make_leaf(
        tmp_path / "small.zarr", None, declared_total=checker.DEFAULT_MIN_ELEMENTS
    )

    assert _check(large, min_elements=checker.DEFAULT_MIN_ELEMENTS)[0] == "fail"
    assert _check(small, min_elements=checker.DEFAULT_MIN_ELEMENTS)[0] == "skip"


def test_declared_total_and_level_structure_are_validated(tmp_path: Path) -> None:
    wrong_total = _make_leaf(
        tmp_path / "wrong-total.zarr", [10, 20, 30], declared_total=99
    )
    status, message = _check(wrong_total)
    assert status == "fail"
    assert "levels sum to 60" in message

    missing_level = _make_leaf(tmp_path / "missing.zarr", [10, 20, 30])
    del missing_level["additive_2"]
    status, message = _check(missing_level)
    assert status == "fail"
    assert "additive_2 is missing" in message


def test_scene_inventory_is_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A no-args quality check must not create ``datasets/demos``."""
    output_dir = tmp_path / "missing-demos"
    calls: list[bool] = []

    def fake_output_dir(*, create: bool = True) -> Path:
        calls.append(create)
        return output_dir

    monkeypatch.setattr(luxar_paths, "get_demos_output_dir", fake_output_dir)

    assert checker.scene_paths([]) == []
    assert calls == [False]
    assert not output_dir.exists()


def test_non_quiet_output_prints_scene_heading_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_path = tmp_path / "valid-scene.luxar.zarr"
    _make_leaf(scene_path, [20, 20, 60])

    exit_code = checker.main(
        [
            str(scene_path),
            "--min-elements",
            "0",
            "--max-level-elements",
            "1000",
        ]
    )
    output = _ANSI_ESCAPE.sub("", capsys.readouterr().out)

    assert exit_code == 0
    assert output.count(scene_path.name) == 1
    assert "1 ok, 0 warned, 0 failed" in output


def test_quiet_output_suppresses_healthy_scene_details(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_path = tmp_path / "valid-scene.luxar.zarr"
    _make_leaf(scene_path, [20, 20, 60])

    exit_code = checker.main(
        [
            str(scene_path),
            "--quiet",
            "--min-elements",
            "0",
            "--max-level-elements",
            "1000",
        ]
    )
    output = _ANSI_ESCAPE.sub("", capsys.readouterr().out)

    assert exit_code == 0
    assert scene_path.name not in output
    assert "1 ok, 0 warned, 0 failed" in output
