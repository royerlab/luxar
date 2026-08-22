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
    """Import ``scripts/check_demo_ladders.py`` as a module by file path."""
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
    """Write a minimal points leaf with the given additive-sublod sizes."""
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
    """Run ``checker.check_leaf`` with test-tuned defaults (``min_elements=0``
    disables the skip guard so every fixture is audited) plus per-call
    overrides."""
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
    """A leaf above the min-elements floor fails; one AT the floor skips
    (equality → skip)."""
    large = _make_leaf(
        tmp_path / "large.zarr", None, declared_total=checker.DEFAULT_MIN_ELEMENTS + 1
    )
    small = _make_leaf(
        tmp_path / "small.zarr", None, declared_total=checker.DEFAULT_MIN_ELEMENTS
    )

    assert _check(large, min_elements=checker.DEFAULT_MIN_ELEMENTS)[0] == "fail"
    assert _check(small, min_elements=checker.DEFAULT_MIN_ELEMENTS)[0] == "skip"


def test_declared_total_and_level_structure_are_validated(tmp_path: Path) -> None:
    """A wrong declared total and a missing sublod level both fail the check."""
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
        """Stand-in for ``get_demos_output_dir`` recording its ``create`` flag."""
        calls.append(create)
        return output_dir

    monkeypatch.setattr(luxar_paths, "get_demos_output_dir", fake_output_dir)

    assert checker.scene_paths([]) == []
    assert calls == [False]
    assert not output_dir.exists()


def test_non_quiet_output_prints_scene_heading_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Non-quiet output names a healthy scene exactly once with a summary line."""
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
    """Quiet mode omits a healthy scene's per-scene lines but keeps the summary."""
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


def test_default_invocation_does_not_run_the_lod_screen(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``hatch run check-demo-ladders`` must behave exactly as it did before.

    The opening-shot screen is a second, opt-in pass; a default run prints the
    ladder audit and nothing else, and its exit code is still the audit's.
    """
    scene_path = tmp_path / "valid-scene.luxar.zarr"
    _make_leaf(scene_path, [20, 20, 60])

    exit_code = checker.main(
        [str(scene_path), "--min-elements", "0", "--max-level-elements", "1000"]
    )
    output = _ANSI_ESCAPE.sub("", capsys.readouterr().out)

    assert exit_code == 0
    assert "1 ok, 0 warned, 0 failed" in output
    assert "LOD screen" not in output
    assert "opening-shot" not in output


def test_screen_flag_adds_the_pass_without_touching_the_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--screen`` appends the screen; a FAILING gate still exits 1, and only it can.

    The fixture leaf is a single 2M-element commit with no ladder — an outright
    gate failure — and it is not a scene, so the screen skips it. The exit code
    must come from the gate alone.
    """
    scene_path = tmp_path / "unladdered.luxar.zarr"
    _make_leaf(scene_path, None, declared_total=2_000_000)

    exit_code = checker.main([str(scene_path), "--screen"])
    output = _ANSI_ESCAPE.sub("", capsys.readouterr().out)

    assert exit_code == 1
    assert "0 ok, 0 warned, 1 failed" in output
    assert "LOD screen:" in output


def test_screen_only_skips_the_gate_and_never_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The same failing store exits 0 under ``--screen-only`` — a report cannot fail."""
    scene_path = tmp_path / "unladdered.luxar.zarr"
    _make_leaf(scene_path, None, declared_total=2_000_000)

    exit_code = checker.main([str(scene_path), "--screen-only"])
    output = _ANSI_ESCAPE.sub("", capsys.readouterr().out)

    assert exit_code == 0
    assert "0 ok, 0 warned" not in output  # the gate did not run at all
    assert "LOD screen:" in output


def test_screen_aspect_spec_accepts_labels_ratios_and_bare_numbers() -> None:
    """``--screen-aspect`` parsing, including the ``W:H`` spelling people write."""
    assert checker.parse_aspects("1.5") == [("1.5", 1.5)]
    assert checker.parse_aspects("4:3") == [("4:3", pytest.approx(4 / 3))]
    assert checker.parse_aspects("wide=2, 1:1") == [
        ("wide", 2.0),
        ("1:1", 1.0),
    ]
    with pytest.raises(ValueError):
        checker.parse_aspects("")
    with pytest.raises(ValueError, match="> 0"):
        checker.parse_aspects("0")
