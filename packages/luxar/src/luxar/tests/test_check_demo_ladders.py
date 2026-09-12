"""Tests for the built-demo additive-ladder structural gate."""

from __future__ import annotations

import importlib.util
import re
from collections import Counter
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import zarr

from luxar.utils import paths as luxar_paths

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_BOUND_HALF_WIDTH = 1e-3


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


def _add_slice_bounds(
    level: zarr.Group,
    *,
    chunk_size: int,
    coordinates: list[int | tuple[int, int]],
) -> None:
    """Stamp synthetic one-axis chunk bounds, including mixed boundary chunks."""
    level.attrs.update({"slice_dims": [3], "chunk_size": chunk_size})
    bounds = np.zeros((len(coordinates), 4, 2), dtype=np.float32)
    for index, coordinate in enumerate(coordinates):
        lo, hi = coordinate if isinstance(coordinate, tuple) else (coordinate,) * 2
        bounds[index, 3] = (lo - _BOUND_HALF_WIDTH, hi + _BOUND_HALF_WIDTH)
    level.create_array("chunk_bounds", data=bounds)


def _add_line_slice_bounds(
    level: zarr.Group, *, chunk_size: int, coordinates: list[int]
) -> None:
    """Stamp the nested Lines vertex-ordering equivalent of slice bounds."""
    level.attrs["vertex_ordering"] = {
        "slice_dims": [3],
        "chunk_size": chunk_size,
    }
    bounds = np.zeros((len(coordinates), 4, 2), dtype=np.float32)
    for index, coordinate in enumerate(coordinates):
        bounds[index, 3] = (
            coordinate - _BOUND_HALF_WIDTH,
            coordinate + _BOUND_HALF_WIDTH,
        )
    level.create_array("vertex_chunk_bounds", data=bounds)


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


def test_absolute_level_cap_uses_the_busiest_barrier_ordered_slice(
    tmp_path: Path,
) -> None:
    leaf = _make_leaf(tmp_path / "sliced.zarr", [120, 120, 120])
    for index in range(3):
        _add_slice_bounds(
            leaf[f"additive_{index}"],
            chunk_size=40,
            coordinates=[0, (0, 1), 1],
        )

    status, message = _check(leaf, max_level_elements=70)

    assert status == "fail"
    assert "largest coordinate fetch is 80 elements" in message


def test_absolute_level_cap_accepts_large_levels_when_each_slice_is_bounded(
    tmp_path: Path,
) -> None:
    leaf = _make_leaf(tmp_path / "sliced.zarr", [120, 120, 120])
    for index in range(3):
        _add_slice_bounds(
            leaf[f"additive_{index}"], chunk_size=30, coordinates=[0, 0, 1, 1]
        )

    status, message = _check(leaf, max_level_elements=70)

    assert status == "ok"
    assert "largest coordinate fetch 60 elements" in message


def test_absolute_level_cap_reads_lines_vertex_ordering_bounds(tmp_path: Path) -> None:
    leaf = _make_leaf(tmp_path / "lines.zarr", [120, 120, 120])
    leaf.attrs.update({"type": "lines", "n_vertices": 360})
    del leaf.attrs["n_points"]
    for index in range(3):
        level = leaf[f"additive_{index}"]
        level.attrs.update({"type": "lines", "n_vertices": 120})
        del level.attrs["n_points"]
        _add_line_slice_bounds(level, chunk_size=30, coordinates=[0, 0, 1, 1])

    status, message = _check(leaf, max_level_elements=70)

    assert status == "ok"
    assert "largest coordinate fetch 60 elements" in message


def test_absolute_level_cap_reads_gsplat_chunk_bounds(tmp_path: Path) -> None:
    leaf = _make_leaf(tmp_path / "gsplats.zarr", [120, 120, 120])
    leaf.attrs.update({"type": "gsplats", "n_splats": 360})
    del leaf.attrs["n_points"]
    for index in range(3):
        level = leaf[f"additive_{index}"]
        level.attrs.update({"type": "gsplats", "n_splats": 120})
        del level.attrs["n_points"]
        _add_slice_bounds(level, chunk_size=30, coordinates=[0, 0, 1, 1])

    status, message = _check(leaf, max_level_elements=70)

    assert status == "ok"
    assert "largest coordinate fetch 60 elements" in message


@pytest.mark.parametrize(
    "fallback",
    [
        "unsliced",
        "extend_to_all",
        "multiple_slice_dims",
        "malformed",
        "mixed_levels",
    ],
)
def test_absolute_level_cap_keeps_node_level_fallbacks(
    tmp_path: Path, fallback: str
) -> None:
    leaf = _make_leaf(tmp_path / f"{fallback}.zarr", [120, 120, 120])
    if fallback == "extend_to_all":
        leaf.attrs["extend_to_all"] = ["time"]
        for index in range(3):
            _add_slice_bounds(
                leaf[f"additive_{index}"],
                chunk_size=30,
                coordinates=[0, 0, 1, 1],
            )
    elif fallback in ("multiple_slice_dims", "malformed"):
        for index in range(3):
            level = leaf[f"additive_{index}"]
            _add_slice_bounds(level, chunk_size=30, coordinates=[0, 0, 1, 1])
            level.attrs["slice_dims"] = (
                [2, 3] if fallback == "multiple_slice_dims" else ["not-an-index"]
            )
    elif fallback == "mixed_levels":
        for index in range(2):
            _add_slice_bounds(
                leaf[f"additive_{index}"],
                chunk_size=30,
                coordinates=[0, 0, 1, 1],
            )

    status, message = _check(leaf, max_level_elements=70)

    assert status == "fail"
    assert "largest level has 120 elements" in message


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


def test_partitioned_slice_survey_reads_rung_zero_across_parts(tmp_path: Path) -> None:
    root = zarr.open_group(tmp_path / "partition.zarr", mode="w")
    root.attrs["kind"] = "partition"
    rows_by_part = [
        np.asarray([[0], [0], [0], [1]], dtype=np.uint16),
        np.asarray([[0], [1], [1], [2], [2]], dtype=np.uint16),
    ]
    for index, rows in enumerate(rows_by_part):
        leaf = root.create_group(f"part_{index}")
        leaf.attrs.update(
            {"type": "points", "n_points": len(rows), "n_additive_sublods": 2}
        )
        rung = leaf.create_group("additive_0")
        rung.attrs.update({"type": "points", "n_points": len(rows), "slice_dims": [0]})
        rung.create_array("positions", data=rows)
        fine = leaf.create_group("additive_1")
        fine.attrs.update({"type": "points", "n_points": 0, "slice_dims": [0]})

    # Parts SUM per coordinate: {0: 3+1, 1: 1+2, 2: 0+2} = {0: 4, 1: 3, 2: 2}.
    # The reported scalar is the SPARSEST coordinate (2), not the busiest (4).
    assert checker._node_slice_histogram(root, root) == Counter(
        {(0,): 4, (1,): 3, (2,): 2}
    )
    assert checker.sliced_first_rung_counts(root) == [("/", 2)]


def test_sliced_rung_with_no_centers_is_an_empty_measurement(tmp_path: Path) -> None:
    leaf = _make_leaf(tmp_path / "empty-survey.zarr", [1, 1])
    leaf["additive_0"].attrs["slice_dims"] = [0]

    assert checker.sliced_first_rung_counts(leaf) == [("/", 0)]


def test_substitutive_levels_are_alternatives_not_additive(tmp_path: Path) -> None:
    root = zarr.open_group(tmp_path / "lod.zarr", mode="w")
    root.attrs["kind"] = "lod"
    for index, count in enumerate((3, 5)):
        leaf = root.create_group(f"child_{index}")
        leaf.attrs.update(
            {"type": "points", "n_points": count, "n_additive_sublods": 2}
        )
        rung = leaf.create_group("additive_0")
        rung.attrs.update({"type": "points", "n_points": count, "slice_dims": [0]})
        rung.create_array("positions", data=np.zeros((count, 1), dtype=np.uint16))
        fine = leaf.create_group("additive_1")
        fine.attrs.update({"type": "points", "n_points": 0})

    assert checker.sliced_first_rung_counts(root) == [("/", 5)]


def test_partitioned_slice_survey_decodes_per_part_coordinate_grids(
    tmp_path: Path,
) -> None:
    root = zarr.open_group(tmp_path / "partition-encoded.zarr", mode="w")
    root.attrs["kind"] = "partition"
    for index, (low, rows) in enumerate(((0.0, [0, 1]), (2.0, [0, 1]))):
        leaf = root.create_group(f"part_{index}")
        leaf.attrs.update({"type": "points", "n_points": 2, "n_additive_sublods": 2})
        rung = leaf.create_group("additive_0")
        rung.attrs.update({"type": "points", "n_points": 2, "slice_dims": [0]})
        positions = rung.create_array(
            "positions", data=np.asarray(rows, dtype=np.uint16).reshape(-1, 1)
        )
        positions.attrs["encoding"] = {
            "name": "linear_perchannel_u16",
            "col_lo": [low],
            "col_hi": [low + 1.0],
            "bits": 16,
            "original_dtype": "float32",
        }
        leaf.create_group("additive_1").attrs.update(
            {"type": "points", "n_points": 0, "slice_dims": [0]}
        )

    assert checker.sliced_first_rung_counts(root) == [("/", 1)]


def test_slice_survey_decodes_lut_coordinate_array(tmp_path: Path) -> None:
    root = zarr.open_group(tmp_path / "lut-encoded.zarr", mode="w")
    leaf = root.create_group("leaf")
    leaf.attrs.update({"type": "points", "n_points": 4, "n_additive_sublods": 2})
    rung = leaf.create_group("additive_0")
    rung.attrs.update({"type": "points", "n_points": 4, "slice_dims": [3]})
    positions = rung.create_array(
        "positions",
        data=np.asarray(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]],
            dtype=np.uint8,
        ),
    )
    positions.attrs["encoding"] = {
        "name": "lut_uint8",
        "lut": [0.0, 7.0],
        "original_dtype": "float32",
    }
    leaf.create_group("additive_1").attrs.update(
        {"type": "points", "n_points": 0, "slice_dims": [3]}
    )

    assert checker.sliced_first_rung_counts(root) == [("/leaf", 2)]


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


def test_default_screen_aspect_spec_preserves_the_library_values() -> None:
    """The CLI default must measure the exact same ratios as ``screen_stores``."""
    assert checker.parse_aspects(checker.DEFAULT_SCREEN_ASPECT_SPEC) == list(
        checker.DEFAULT_ASPECTS
    )


def _make_lod_scene(path: Path) -> Path:
    """A real compiled scene with one whole-object ``kind=lod`` ladder.

    The bare ``_make_leaf`` fixtures above carry no ``type: "scene"`` root, so
    the screen takes its scene-skip branch on every one of them — which left the
    whole rendering path (`_print_group`, the ladder/metric formatters) and every
    ``--screen-*`` flag reachable only in theory. This one actually screens.

    ``spacer`` widens the scene root so the camera pulls back and the ladder's
    own blob occupies a modest share of the shot: past the stored legacy 0.5
    rung, short of the re-derived 0.25 one — a ``win``.
    """
    import numpy as np

    from luxar import Dimensions, LuxarZarrCompiler

    rng = np.random.default_rng(0)
    blob = rng.uniform(-30.0, 30.0, size=(4000, 3)).astype(np.float32)
    spacer = rng.uniform(-1.0, 1.0, size=(200, 3)).astype(np.float32) + np.array(
        [100.0, 0.0, 0.0], dtype=np.float32
    )
    with LuxarZarrCompiler(str(path)) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        lod = scene.add_lod_group("whole")
        lod.add_points("l0", blob[::16], radii=0.5, coverage_fraction=0.0)
        lod.add_points("l1", blob[::4], radii=0.5, coverage_fraction=0.5)
        lod.add_points("l2", blob, radii=0.5, coverage_fraction=1.0)
        scene.add_points("spacer", spacer, radii=0.5)
    return path


def test_screen_only_renders_a_real_group_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The per-group report, its ladders and the ``--screen-*`` flags, end to end."""
    scene_path = _make_lod_scene(tmp_path / "lod-scene.luxar.zarr")

    exit_code = checker.main(
        [
            str(scene_path),
            "--screen-only",
            "--screen-aspect",
            "1:1",
            "--screen-verdict",
            "win",
        ]
    )
    output = _ANSI_ESCAPE.sub("", capsys.readouterr().out)

    assert exit_code == 0
    assert "whole: win" in output
    assert "elements [250, 1,000, 4,000]" in output
    # Stored legacy ladder → the screen-area ladder restamp-lod would derive.
    assert "stored   [0, 0.5, 1]  →  re-derived [0, 0.25, 0.5]" in output
    assert "[coarser]" in output
    # `--screen-aspect 1:1` really replaced the three-aspect default.
    assert "1:1" in output
    assert "16:9" not in output
    assert "LOD screen: 1 win," in output


def test_the_screen_verdict_filter_hides_the_group_but_still_counts_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A filter that matches nothing must not read as "nothing to report"."""
    scene_path = _make_lod_scene(tmp_path / "lod-scene.luxar.zarr")

    exit_code = checker.main(
        [str(scene_path), "--screen-only", "--screen-verdict", "no-op"]
    )
    output = _ANSI_ESCAPE.sub("", capsys.readouterr().out)

    assert exit_code == 0
    assert "whole: win" not in output
    assert "[coarser]" not in output
    # The tally is always over everything, filtered or not.
    assert "LOD screen: 1 win," in output


@pytest.mark.parametrize("spec", ["abc", "1:0"])
def test_a_malformed_screen_aspect_exits_with_a_usage_error(
    spec: str, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """No raw traceback, and no PASS line printed before the process dies.

    ``abc`` raises ``ValueError`` and ``1:0`` raised ``ZeroDivisionError``, which
    the docstring never listed; both now leave ``parse_aspects`` as a plain
    ``ValueError`` and are routed through ``parser.error`` BEFORE the gate runs.
    """
    scene_path = tmp_path / "valid-scene.luxar.zarr"
    _make_leaf(scene_path, [20, 20, 60])

    with pytest.raises(SystemExit) as excinfo:
        checker.main([str(scene_path), "--screen", "--screen-aspect", spec])

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "--screen-aspect" in captured.err
    assert "ok, 0 warned" not in _ANSI_ESCAPE.sub("", captured.out)


def test_parse_aspects_reports_a_zero_height_as_a_value_error() -> None:
    """``1:0`` is a bad argument, not an arithmetic accident."""
    with pytest.raises(ValueError, match="zero height"):
        checker.parse_aspects("1:0")
    with pytest.raises(ValueError):
        checker.parse_aspects("abc")


def _sliced_node(
    path: Path,
    per_coordinate: dict[int, int],
    *,
    node_total: int | None = None,
    node_name: str = "leaf",
):
    """A points leaf whose rung 0 holds ``count`` elements at each coordinate."""
    root = zarr.open_group(path, mode="w")
    leaf = root.create_group(node_name)
    rows = np.asarray(
        [[coord] for coord, count in per_coordinate.items() for _ in range(count)],
        dtype=np.uint16,
    ).reshape(-1, 1)
    total = node_total if node_total is not None else len(rows)
    leaf.attrs.update({"type": "points", "n_points": total, "n_additive_sublods": 2})
    rung = leaf.create_group("additive_0")
    rung.attrs.update({"type": "points", "n_points": len(rows), "slice_dims": [0]})
    rung.create_array("positions", data=rows)
    leaf.create_group("additive_1").attrs.update(
        {"type": "points", "n_points": max(total - len(rows), 0), "slice_dims": [0]}
    )
    return root


def _sliced_verdicts(root, **overrides):
    results: list[tuple[str, str, str]] = []
    counts = {"ok": 0, "warn": 0, "fail": 0, "skip": 0}
    failures: list[str] = []
    options = {
        "scene_name": "scene",
        "min_first_rung": checker.DEFAULT_MIN_SLICE_FIRST_RUNG,
        "min_rung_share": checker.DEFAULT_MIN_SLICE_RUNG_SHARE,
    }
    options.update(overrides)
    checker._record_sliced_verdicts(
        root,
        options["scene_name"],
        options["min_first_rung"],
        results,
        counts,
        failures,
        options["min_rung_share"],
    )
    return results


def test_one_busy_coordinate_does_not_mask_starved_ones(tmp_path: Path) -> None:
    """The regression the percentile reduction replaces.

    Reducing with ``max()`` asked whether the BUSIEST slice was healthy, so a
    node with a single fat coordinate passed however many starved ones sat
    beside it. Measured spreads between p05 and max on real stores run from
    2.1x to 122x, and ``zebrafish_timelapse/endoderm`` passed the old reduction
    on a max of 6,648 while a twentieth of its timepoints held 330 or fewer.
    """
    per_coordinate = {0: 5_000} | {c: 5 for c in range(1, 40)}
    root = _sliced_node(tmp_path / "lopsided.zarr", per_coordinate)

    histogram = checker._node_slice_histogram(root["leaf"], root)
    assert max(histogram.values()) == 5_000  # the old reduction: comfortably passing
    assert checker.sparsest_slice_elements(histogram) == 5  # the new one: starved

    (path, status, message) = _sliced_verdicts(root)[0]
    assert (path, status) == ("/leaf", "fail")
    assert "sparsest rung-0 slices hold 5 elements" in message
    assert "covers 40 coordinates" in message


def test_a_uniformly_healthy_sliced_node_passes(tmp_path: Path) -> None:
    """The negative arm: a gate that never passes is indistinguishable from one
    that never fires, so pin that a good node produces NO verdict."""
    root = _sliced_node(tmp_path / "healthy.zarr", {c: 400 for c in range(30)})

    assert checker.sliced_first_rung_counts(root) == [("/leaf", 400)]
    assert _sliced_verdicts(root) == []


def test_share_arm_fires_when_every_slice_clears_the_absolute_floor(
    tmp_path: Path,
) -> None:
    """The two arms are independent: absolute counts can be comfortable while
    rung 0 is still a sliver of the node, which is the #2376 shape (a rung sized
    against a download budget, then divided by the slice count)."""
    # 30 coordinates x 400 = 12,000 in rung 0, against a 1,000,000-element node.
    root = _sliced_node(
        tmp_path / "thin-share.zarr", {c: 400 for c in range(30)}, node_total=1_000_000
    )

    assert checker.sliced_first_rung_counts(root) == [("/leaf", 400)]  # absolute: fine
    (path, status, message) = _sliced_verdicts(root)[0]
    assert (path, status) == ("/leaf", "fail")
    assert "rung 0 is 1.20% of the node" in message
    assert "share floor" in message


def test_share_denominator_skips_unladdered_partition_parts(tmp_path: Path) -> None:
    """Only parts contributing a sliced rung belong in its share denominator."""
    root = zarr.open_group(tmp_path / "mixed-partition.zarr", mode="w")
    root.attrs["kind"] = "partition"
    laddered = root.create_group("laddered")
    laddered.attrs.update(
        {"type": "points", "n_points": 100_000, "n_additive_sublods": 2}
    )
    rows = np.repeat(np.arange(40, dtype=np.uint16), 500).reshape(-1, 1)
    rung = laddered.create_group("additive_0")
    rung.attrs.update({"type": "points", "n_points": len(rows), "slice_dims": [0]})
    rung.create_array("positions", data=rows)
    laddered.create_group("additive_1").attrs.update(
        {"type": "points", "n_points": 80_000, "slice_dims": [0]}
    )
    root.create_group("unladdered").attrs.update(
        {"type": "points", "n_points": 900_000, "n_additive_sublods": 1}
    )

    assert _sliced_verdicts(root) == []


def test_substitutive_share_uses_one_level_not_keywise_maxima(tmp_path: Path) -> None:
    """Synthetic maxima from different LOD alternatives must not inflate share."""
    root = zarr.open_group(tmp_path / "lod-share.zarr", mode="w")
    root.attrs["kind"] = "lod"
    for index, coordinate in enumerate((0, 1)):
        leaf = root.create_group(f"level_{index}")
        leaf.attrs.update(
            {"type": "points", "n_points": 10_000, "n_additive_sublods": 2}
        )
        rung = leaf.create_group("additive_0")
        rung.attrs.update({"type": "points", "n_points": 600, "slice_dims": [0]})
        rung.create_array(
            "positions",
            data=np.full((600, 1), coordinate, dtype=np.uint16),
        )
        leaf.create_group("additive_1").attrs.update(
            {"type": "points", "n_points": 9_400, "slice_dims": [0]}
        )

    (_, status, message) = _sliced_verdicts(root)[0]
    assert status == "fail"
    assert "rung 0 is 6.00% of the node" in message


def test_share_arm_fires_when_an_exemption_is_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fires-proof for the allowlist, in both directions.

    An exemption that silently disabled the arm would be indistinguishable from
    a clean store, so pin that the SAME node fails the moment its entry goes
    away — this is what stops the allowlist from becoming a tolerance.
    """
    root = _sliced_node(
        tmp_path / "exempt.zarr", {c: 400 for c in range(30)}, node_total=192_000
    )
    key = "scene/leaf"

    monkeypatch.setattr(
        checker,
        "SHARE_ARM_EXEMPT",
        {key: (0.06, "measured: converges on keypress")},
    )
    (_, status, message) = _sliced_verdicts(root)[0]
    assert status == "warn"
    assert "[exempt: measured: converges on keypress]" in message

    monkeypatch.setattr(checker, "SHARE_ARM_EXEMPT", {})
    (_, status, _) = _sliced_verdicts(root)[0]
    assert status == "fail"


def test_share_exemption_stops_at_its_measured_floor(tmp_path: Path) -> None:
    """An exempt path goes red if a rebuild degrades below the allowed share."""
    scene = tmp_path / "esm3_protein_landscape.luxar.zarr"
    root = _sliced_node(
        scene,
        {0: 500, 1: 500},
        node_total=20_000,
        node_name="proteins",
    )

    (_, status, message) = _sliced_verdicts(
        root, scene_name="esm3_protein_landscape.luxar.zarr"
    )[0]
    assert status == "fail"
    assert "rung 0 is 5.00% of the node" in message


def test_every_share_arm_exemption_carries_a_measured_reason() -> None:
    """An exemption without a figure in it is a tolerance wearing a comment."""
    assert checker.SHARE_ARM_EXEMPT, "an empty allowlist should be deleted, not kept"
    for key, (_, reason) in checker.SHARE_ARM_EXEMPT.items():
        assert "%" in reason and "rung 0 =" in reason, (
            f"{key}: reason cites no measurement"
        )


def test_calibrated_sliced_defaults_and_cli_exemption_are_pinned(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Pin the measured defaults, CLI wiring, and exempt/fresh exit behavior."""
    assert checker.DEFAULT_MIN_SLICE_FIRST_RUNG == 250
    assert checker.DEFAULT_MIN_SLICE_RUNG_SHARE == 0.10
    exempt = tmp_path / "esm3_protein_landscape.luxar.zarr"
    _sliced_node(
        exempt,
        {0: 500, 1: 500},
        node_total=16_000,
        node_name="proteins",
    )

    assert checker.main([str(exempt)]) == 0
    exempt_output = _ANSI_ESCAPE.sub("", capsys.readouterr().out)
    assert "rung 0 is 6.25% of the node" in exempt_output
    assert "[exempt:" in exempt_output

    fresh = tmp_path / "fresh.luxar.zarr"
    _sliced_node(fresh, {0: 500, 1: 500}, node_total=16_000)
    assert checker.main([str(fresh)]) == 1
    fresh_output = _ANSI_ESCAPE.sub("", capsys.readouterr().out)
    assert "rung 0 is 6.25% of the node" in fresh_output
    assert "[exempt:" not in fresh_output

    assert checker.main([str(fresh), "--min-slice-rung-share", "0.005"]) == 0
    capsys.readouterr()

    healthy = tmp_path / "healthy.luxar.zarr"
    _sliced_node(healthy, {0: 500, 1: 500})
    assert checker.main([str(healthy), "--min-slice-first-rung", "100000"]) == 1
    healthy_output = _ANSI_ESCAPE.sub("", capsys.readouterr().out)
    assert "below the 100,000 absolute first-paint floor" in healthy_output


def test_starvation_percentile_is_the_minimum_for_a_handful_of_slices() -> None:
    """At 20 or fewer coordinates the index is 0, which is the right reading:
    with a handful of slices there is no tail to discount."""
    assert checker.sparsest_slice_elements(Counter({(0,): 9, (1,): 100})) == 9
    assert (
        checker.sparsest_slice_elements(
            Counter({(c,): 100 for c in range(19)} | {(19,): 3})
        )
        == 3
    )

    # At 100 coordinates the index is int(0.05 * 99) = 4, the FIFTH smallest.
    # So up to four starved slices are tolerated as a tail and the fifth is
    # reported — pin both sides of that boundary, since a percentile that
    # silently rounded to 0 or to the minimum would pass one of them.
    def hundred(starved: int) -> Counter:
        return Counter(
            {(c,): 1 for c in range(starved)} | {(c,): 900 for c in range(starved, 100)}
        )

    assert checker.sparsest_slice_elements(hundred(4)) == 900  # tail, discounted
    assert checker.sparsest_slice_elements(hundred(5)) == 1  # 5% starved, reported


def test_empty_histogram_is_a_zero_not_a_pass() -> None:
    """ "0 nodes measured" and "0 violations" render identically and mean the
    opposite, so an empty survey must fail rather than fall through."""
    assert checker.sparsest_slice_elements(Counter()) == 0


# ---------------------------------------------------------------------------
# Empty inventory (audit A9-04) — the same "0 measured reads as 0 violations"
# hazard as the test directly above, one level up: at the whole-run level.
# ---------------------------------------------------------------------------


def _empty_inventory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the default scene inventory at an empty directory."""

    def fake_output_dir(*, create: bool = True) -> Path:
        """Stand-in for ``get_demos_output_dir`` over an empty tree."""
        return tmp_path / "no-demos-here"

    monkeypatch.setattr(luxar_paths, "get_demos_output_dir", fake_output_dir)


def test_empty_inventory_passes_but_says_it_inspected_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Still exit 0 — but the message must not read like a result.

    The output directory is gitignored, so an empty inventory is the normal
    state of a fresh clone and of CI, and failing by default would make the gate
    unrunnable there. What was wrong was the wording: "No scenes found. Build a
    demo first" sat next to every other gate's pass line and was read as one.
    """
    _empty_inventory(tmp_path, monkeypatch)

    assert checker.main([]) == 0

    out = _ANSI_ESCAPE.sub("", capsys.readouterr().out)
    assert "INSPECTED NOTHING" in out
    assert "not a pass" in out


def test_require_scenes_turns_an_empty_inventory_into_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The opt-in for callers that KNOW scenes should be there.

    Gallery generation and the pre-upload audit run this AFTER building, where
    an empty inventory means the build produced nothing — the one situation in
    which silence is the bug rather than the normal case.
    """
    _empty_inventory(tmp_path, monkeypatch)

    assert checker.main(["--require-scenes"]) == 1

    out = _ANSI_ESCAPE.sub("", capsys.readouterr().out)
    assert "INSPECTED NOTHING" in out


def test_require_scenes_does_not_disturb_a_run_that_has_scenes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The flag must gate ONLY emptiness, not the verdict.

    Without this, `--require-scenes` could be implemented as a blanket "fail
    unless perfect" and every test above would still pass.
    """
    scene_path = tmp_path / "valid-scene.luxar.zarr"
    _make_leaf(scene_path, [20, 20, 60])

    exit_code = checker.main(
        [
            str(scene_path),
            "--require-scenes",
            "--min-elements",
            "0",
            "--max-level-elements",
            "1000",
        ]
    )

    assert exit_code == 0, _ANSI_ESCAPE.sub("", capsys.readouterr().out)
