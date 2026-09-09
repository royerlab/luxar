"""Tests for freshness-aware example fixture generation."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from luxar.conftest import viewer_source

_MOD_PATH = Path(__file__).resolve().parents[1] / "run_examples.py"
_spec = importlib.util.spec_from_file_location("run_examples", _MOD_PATH)
assert _spec is not None and _spec.loader is not None
run_examples = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_examples)


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    examples = repo / "packages/luxar/examples"
    production = repo / "packages/luxar/src/luxar/io"
    examples.mkdir(parents=True)
    production.mkdir(parents=True)
    (production.parent / "__init__.py").write_text("")
    (production / "__init__.py").write_text("")
    (repo / "pyproject.toml").write_text("[project]\nname = 'luxar'\n")
    (production / "writer.py").write_text("FORMAT = 2\n")
    return repo


def _write_example(repo: Path, name: str, body: str) -> Path:
    script = repo / "packages/luxar/examples" / f"{name}_example.py"
    script.write_text(body)
    return script


def _write_marker(
    repo: Path,
    output_dir: Path,
    outputs_by_producer: dict[str, list[str]] | None = None,
) -> None:
    scripts = {
        script.name: script
        for script in (repo / "packages/luxar/examples").glob("*_example.py")
    }
    if outputs_by_producer is None:
        assert len(scripts) == 1
        outputs_by_producer = {
            next(iter(scripts)): sorted(
                path.name for path in output_dir.glob("*.zarr") if path.is_dir()
            )
        }
    run_examples.write_marker(
        output_dir,
        examples={
            producer: {
                "fingerprint": run_examples.example_fingerprint(
                    repo, scripts[producer]
                ),
                "sources": [
                    path.relative_to(repo).as_posix()
                    for path in run_examples.example_source_files(
                        repo, scripts[producer]
                    )
                ],
                "outputs": outputs,
            }
            for producer, outputs in outputs_by_producer.items()
        },
    )


def _generating_example(
    repo: Path, name: str, sentinel: Path, imports: str = ""
) -> Path:
    output = repo / "datasets/examples" / f"{name}_example.luxar.zarr"
    return _write_example(
        repo,
        name,
        "from pathlib import Path\n" + imports + f"Path({str(sentinel)!r}).touch()\n"
        f"output = Path({str(output)!r})\n"
        "output.mkdir(parents=True, exist_ok=True)\n"
        "(output / 'zarr.json').write_text('fresh')\n",
    )


def test_example_fingerprint_covers_only_imported_production_code(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    example = _write_example(repo, "one", "from luxar.io import writer\n")
    writer = repo / "packages/luxar/src/luxar/io/writer.py"
    unrelated = repo / "packages/luxar/src/luxar/io/unrelated.py"
    unrelated.write_text("VALUE = 1\n")

    initial = run_examples.example_fingerprint(repo, example)
    unrelated.write_text("VALUE = 2\n")
    after_unrelated_change = run_examples.example_fingerprint(repo, example)
    (repo / "pyproject.toml").write_text("[tool.ruff]\nline-length = 100\n")
    after_project_change = run_examples.example_fingerprint(repo, example)
    example.write_text("print('changed')\n")
    after_builder_change = run_examples.example_fingerprint(repo, example)
    example.write_text("from luxar.io import writer\n")
    writer.write_text("FORMAT = 3\n")
    after_writer_change = run_examples.example_fingerprint(repo, example)

    assert after_unrelated_change == initial
    assert after_project_change == initial
    assert after_builder_change != initial
    assert after_writer_change != initial


def test_example_fingerprint_covers_writer_when_checkout_path_contains_tests(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path / "tests")
    example = _write_example(repo, "one", "from luxar.io import writer\n")
    writer = repo / "packages/luxar/src/luxar/io/writer.py"

    initial = run_examples.example_fingerprint(repo, example)
    writer.write_text("FORMAT = 3\n")

    assert run_examples.example_fingerprint(repo, example) != initial


def test_example_fingerprint_covers_imported_example_helper(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    helper = repo / "packages/luxar/examples/_helper.py"
    helper.write_text("VALUE = 1\n")
    example = _write_example(repo, "one", "from _helper import VALUE\n")

    initial = run_examples.example_fingerprint(repo, example)
    helper.write_text("VALUE = 2\n")

    assert run_examples.example_fingerprint(repo, example) != initial


def test_example_sources_honour_source_encoding_cookie(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    helper = repo / "packages/luxar/examples/_helper.py"
    helper.write_text("VALUE = 1\n")
    example = repo / "packages/luxar/examples/one_example.py"
    example.write_bytes(
        b'# -*- coding: latin-1 -*-\nfrom _helper import VALUE\nNAME = "caf\xe9"\n'
    )

    assert helper in run_examples.example_source_files(repo, example)


def test_current_marker_requires_every_recorded_output(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _write_example(repo, "one", "print('one')\n")
    output_dir = repo / "datasets/examples"
    output = output_dir / "one_example.luxar.zarr"
    output.mkdir(parents=True)
    marker = output_dir / run_examples.MARKER_NAME
    marker.write_text(
        json.dumps(
            {
                "version": run_examples.MARKER_VERSION,
                "environment": run_examples.build_environment(),
                "examples": {
                    "one_example.py": {
                        "fingerprint": run_examples.example_fingerprint(
                            repo,
                            repo / "packages/luxar/examples/one_example.py",
                        ),
                        "sources": [
                            "packages/luxar/examples/one_example.py",
                        ],
                        "outputs": [output.name],
                    }
                },
            }
        )
    )

    assert run_examples.fixtures_are_current(repo, output_dir)
    output.rmdir()
    assert not run_examples.fixtures_are_current(repo, output_dir)


def test_current_marker_ignores_an_unrecorded_extra_output(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _write_example(repo, "one", "print('one')\n")
    output_dir = repo / "datasets/examples"
    (output_dir / "one_example.luxar.zarr").mkdir(parents=True)
    _write_marker(repo, output_dir)

    (output_dir / "removed_example.luxar.zarr").mkdir()
    assert run_examples.fixtures_are_current(repo, output_dir)


def test_current_marker_rejects_environment_changes(
    tmp_path: Path, monkeypatch: object
) -> None:
    repo = _repo(tmp_path)
    _write_example(repo, "one", "print('one')\n")
    output_dir = repo / "datasets/examples"
    (output_dir / "one_example.luxar.zarr").mkdir(parents=True)
    monkeypatch.delenv("LUXAR_ZARR_FORMAT", raising=False)  # type: ignore[attr-defined]
    _write_marker(repo, output_dir)

    monkeypatch.setenv("LUXAR_ZARR_FORMAT", "2")  # type: ignore[attr-defined]
    assert not run_examples.fixtures_are_current(repo, output_dir)


def test_current_fixtures_skip_example_execution(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    sentinel = repo / "executed"
    _write_example(
        repo, "one", f"from pathlib import Path\nPath({str(sentinel)!r}).touch()\n"
    )
    output_dir = repo / "datasets/examples"
    output = output_dir / "one_example.luxar.zarr"
    output.mkdir(parents=True)
    _write_marker(repo, output_dir)

    assert run_examples.generate_examples(repo, output_dir, python=sys.executable) == 0
    assert not sentinel.exists()


def test_builder_change_rebuilds_only_its_example(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    output_dir = repo / "datasets/examples"
    first_ran = repo / "first-ran"
    second_ran = repo / "second-ran"
    first = _generating_example(repo, "one", first_ran)
    _generating_example(repo, "two", second_ran)
    assert run_examples.generate_examples(repo, output_dir, python=sys.executable) == 0
    first_ran.unlink()
    second_ran.unlink()

    first.write_text(first.read_text() + "# changed\n")
    assert run_examples.generate_examples(repo, output_dir, python=sys.executable) == 0

    assert first_ran.exists()
    assert not second_ran.exists()


def test_production_change_rebuilds_only_importing_examples(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    output_dir = repo / "datasets/examples"
    first_ran = repo / "first-ran"
    second_ran = repo / "second-ran"
    _generating_example(repo, "one", first_ran, imports="from luxar.io import writer\n")
    _generating_example(repo, "two", second_ran)
    assert run_examples.generate_examples(repo, output_dir, python=sys.executable) == 0
    first_ran.unlink()
    second_ran.unlink()

    writer = repo / "packages/luxar/src/luxar/io/writer.py"
    writer.write_text("FORMAT = 3\n")
    assert run_examples.generate_examples(repo, output_dir, python=sys.executable) == 0

    assert first_ran.exists()
    assert not second_ran.exists()


def test_missing_output_rebuilds_only_its_producer(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    output_dir = repo / "datasets/examples"
    first_ran = repo / "first-ran"
    second_ran = repo / "second-ran"
    _generating_example(repo, "one", first_ran)
    _generating_example(repo, "two", second_ran)
    assert run_examples.generate_examples(repo, output_dir, python=sys.executable) == 0
    first_ran.unlink()
    second_ran.unlink()
    shutil.rmtree(output_dir / "one_example.luxar.zarr")

    assert run_examples.generate_examples(repo, output_dir, python=sys.executable) == 0

    assert first_ran.exists()
    assert not second_ran.exists()


def test_removed_producer_prunes_its_output_without_rebuilding_others(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    output_dir = repo / "datasets/examples"
    first_ran = repo / "first-ran"
    second_ran = repo / "second-ran"
    _generating_example(repo, "one", first_ran)
    second = _generating_example(repo, "two", second_ran)
    assert run_examples.generate_examples(repo, output_dir, python=sys.executable) == 0
    first_ran.unlink()
    second_ran.unlink()
    second.unlink()

    assert run_examples.generate_examples(repo, output_dir, python=sys.executable) == 0

    assert not first_ran.exists()
    assert not second_ran.exists()
    assert not (output_dir / "two_example.luxar.zarr").exists()


def test_transferred_output_is_not_pruned_after_new_producer_stamps_it(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    output_dir = repo / "datasets/examples"
    shared = output_dir / "shared_example.luxar.zarr"
    first = _write_example(
        repo,
        "one",
        "from pathlib import Path\n"
        f"output = Path({str(shared)!r})\n"
        "output.mkdir(parents=True, exist_ok=True)\n"
        "(output / 'zarr.json').write_text('one')\n",
    )
    assert run_examples.generate_examples(repo, output_dir, python=sys.executable) == 0
    first.unlink()
    _write_example(
        repo,
        "two",
        "from pathlib import Path\n"
        f"output = Path({str(shared)!r})\n"
        "output.mkdir(parents=True, exist_ok=True)\n"
        "(output / 'zarr.json').write_text('two')\n",
    )

    assert run_examples.generate_examples(repo, output_dir, python=sys.executable) == 0

    assert shared.is_dir()
    assert (shared / "zarr.json").read_text() == "two"
    assert run_examples.fixtures_are_current(repo, output_dir)


def test_legacy_marker_rebuilds_all_examples_and_stamps_outputs(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    output_dir = repo / "datasets/examples"
    stale = output_dir / "removed_example.luxar.zarr"
    stale.mkdir(parents=True)
    unknown = output_dir / "handmade.luxar.zarr"
    unknown.mkdir()
    for name in ("one", "two"):
        output = output_dir / f"{name}_example.luxar.zarr"
        _write_example(
            repo,
            name,
            "from pathlib import Path\n"
            f"output = Path({str(output)!r})\n"
            "output.mkdir(parents=True, exist_ok=True)\n"
            "(output / 'zarr.json').write_text('fresh')\n",
        )
    (output_dir / run_examples.MARKER_NAME).write_text(
        json.dumps(
            {
                "version": 2,
                "fingerprint": "legacy",
                "environment": run_examples.build_environment(),
                "outputs": [stale.name],
            }
        )
    )

    assert run_examples.generate_examples(repo, output_dir, python=sys.executable) == 0

    assert not stale.exists()
    assert unknown.exists()
    fixtures = sorted(path.name for path in output_dir.glob("*.zarr"))
    assert fixtures == [
        "handmade.luxar.zarr",
        "one_example.luxar.zarr",
        "two_example.luxar.zarr",
    ]
    assert run_examples.fixtures_are_current(repo, output_dir)
    marker = json.loads((output_dir / run_examples.MARKER_NAME).read_text())
    assert sorted(marker["examples"]) == ["one_example.py", "two_example.py"]


def test_rebuild_progress_precedes_child_output_when_piped(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    output_dir = repo / "datasets/examples"
    output = output_dir / "one_example.luxar.zarr"
    _write_example(
        repo,
        "one",
        "from pathlib import Path\n"
        "print('CHILD OUTPUT')\n"
        f"output = Path({str(output)!r})\n"
        "output.mkdir(parents=True)\n"
        "(output / 'zarr.json').write_text('fresh')\n",
    )
    driver = (
        "import importlib.util, pathlib; "
        f"path = pathlib.Path({str(_MOD_PATH)!r}); "
        "spec = importlib.util.spec_from_file_location('run_examples', path); "
        "module = importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(module); "
        f"raise SystemExit(module.generate_examples(pathlib.Path({str(repo)!r}), "
        f"pathlib.Path({str(output_dir)!r}), python={sys.executable!r}))"
    )

    result = subprocess.run(
        [sys.executable, "-c", driver],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.index(
        "[1/1] 📊 Running one_example.py..."
    ) < result.stdout.index("CHILD OUTPUT")


def test_failed_rebuild_stamps_successes_for_selective_retry(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    output_dir = repo / "datasets/examples"
    reached = repo / "second-ran"
    _write_example(repo, "one", "raise RuntimeError('broken')\n")
    second_output = output_dir / "two_example.luxar.zarr"
    _write_example(
        repo,
        "two",
        "from pathlib import Path\n"
        f"Path({str(reached)!r}).touch()\n"
        f"output = Path({str(second_output)!r})\n"
        "output.mkdir(parents=True, exist_ok=True)\n"
        "(output / 'zarr.json').write_text('fresh')\n",
    )

    assert run_examples.generate_examples(repo, output_dir, python=sys.executable) == 1

    assert reached.exists()
    marker = json.loads((output_dir / run_examples.MARKER_NAME).read_text())
    assert list(marker["examples"]) == ["two_example.py"]
    reached.unlink()
    first_output = output_dir / "one_example.luxar.zarr"
    _write_example(
        repo,
        "one",
        "from pathlib import Path\n"
        f"output = Path({str(first_output)!r})\n"
        "output.mkdir(parents=True, exist_ok=True)\n"
        "(output / 'zarr.json').write_text('fresh')\n",
    )

    assert run_examples.generate_examples(repo, output_dir, python=sys.executable) == 0
    assert not reached.exists()


def test_stale_stamp_is_removed_before_producer_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path)
    output_dir = repo / "datasets/examples"
    script = _generating_example(repo, "one", repo / "ran")
    assert run_examples.generate_examples(repo, output_dir, python=sys.executable) == 0
    script.write_text(script.read_text() + "# stale\n")

    def interrupt(
        *_args: object, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        assert not (output_dir / run_examples.MARKER_NAME).exists()
        raise KeyboardInterrupt

    monkeypatch.setattr(run_examples.subprocess, "run", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_examples.generate_examples(repo, output_dir, python=sys.executable)

    assert not (output_dir / run_examples.MARKER_NAME).exists()


def test_successful_rebuild_without_outputs_preserves_previous_fixtures(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    output_dir = repo / "datasets/examples"
    previous = output_dir / "previous_example.luxar.zarr"
    previous.mkdir(parents=True)
    sentinel = previous / "zarr.json"
    sentinel.write_text("old")
    _write_example(repo, "one", "print('no output')\n")
    _write_marker(repo, output_dir, {"one_example.py": [previous.name]})

    assert (
        run_examples.generate_examples(
            repo, output_dir, python=sys.executable, force=True
        )
        == 0
    )

    assert sentinel.read_text() == "old"
    marker = json.loads((output_dir / run_examples.MARKER_NAME).read_text())
    assert marker["examples"]["one_example.py"]["outputs"] == [previous.name]
    assert run_examples.fixtures_are_current(repo, output_dir)


def test_successful_producer_without_outputs_is_stamped_current(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    output_dir = repo / "datasets/examples"
    _write_example(repo, "one", "print('no output')\n")

    assert run_examples.generate_examples(repo, output_dir, python=sys.executable) == 0

    marker = json.loads((output_dir / run_examples.MARKER_NAME).read_text())
    assert marker["examples"]["one_example.py"]["outputs"] == []
    assert run_examples.fixtures_are_current(repo, output_dir)


def test_generation_reuses_output_snapshot_between_producers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path)
    output_dir = repo / "datasets/examples"
    _generating_example(repo, "one", repo / "one-ran")
    _generating_example(repo, "two", repo / "two-ran")
    original = run_examples._output_signatures
    calls = 0

    def count_calls(path: Path) -> dict[str, tuple[tuple[str, int, int], ...]]:
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(run_examples, "_output_signatures", count_calls)

    assert run_examples.generate_examples(repo, output_dir, python=sys.executable) == 0
    assert calls == 3


def test_invalid_marker_entry_does_not_discard_valid_stamps(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    output_dir = repo / "datasets/examples"
    valid = _generating_example(repo, "one", repo / "ran")
    _write_example(repo, "bad", "print('bad')\n")
    output = output_dir / "one_example.luxar.zarr"
    output.mkdir(parents=True)
    (output / "zarr.json").write_text("fresh")
    marker = {
        "version": run_examples.MARKER_VERSION,
        "environment": run_examples.build_environment(),
        "examples": {
            "one_example.py": {
                "fingerprint": run_examples.example_fingerprint(repo, valid),
                "sources": ["packages/luxar/examples/one_example.py"],
                "outputs": [output.name],
            },
            "bad_example.py": {
                "fingerprint": "bad",
                "sources": [],
                "outputs": [],
            },
        },
    }
    (output_dir / run_examples.MARKER_NAME).write_text(json.dumps(marker))

    assert run_examples.stale_examples(repo, output_dir) == ["bad_example.py"]
    assert "Ignoring invalid fixture stamp: bad_example.py" in capsys.readouterr().err


def test_empty_marker_output_cannot_remove_the_output_directory(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    output_dir = repo / "datasets/examples"
    valid = _write_example(repo, "one", "print('one')\n")
    generated = output_dir / "one_example.luxar.zarr"
    generated.mkdir(parents=True)
    (generated / "zarr.json").write_text("fresh")
    handmade = output_dir / "handmade.luxar.zarr"
    handmade.mkdir()
    marker = {
        "version": run_examples.MARKER_VERSION,
        "environment": run_examples.build_environment(),
        "examples": {
            "one_example.py": {
                "fingerprint": run_examples.example_fingerprint(repo, valid),
                "sources": ["packages/luxar/examples/one_example.py"],
                "outputs": [generated.name],
            },
            "gone_example.py": {
                "fingerprint": "gone",
                "sources": ["packages/luxar/examples/gone_example.py"],
                "outputs": [""],
            },
        },
    }
    (output_dir / run_examples.MARKER_NAME).write_text(json.dumps(marker))

    assert run_examples.generate_examples(repo, output_dir, python=sys.executable) == 0

    assert generated.is_dir()
    assert handmade.is_dir()
    assert run_examples.fixtures_are_current(repo, output_dir)


def test_invalid_marker_warning_is_printed_once_per_generation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    output_dir = repo / "datasets/examples"
    valid = _write_example(repo, "one", "print('one')\n")
    _write_example(repo, "bad", "print('bad')\n")
    output = output_dir / "one_example.luxar.zarr"
    output.mkdir(parents=True)
    marker = {
        "version": run_examples.MARKER_VERSION,
        "environment": run_examples.build_environment(),
        "examples": {
            "one_example.py": {
                "fingerprint": run_examples.example_fingerprint(repo, valid),
                "sources": ["packages/luxar/examples/one_example.py"],
                "outputs": [output.name],
            },
            "bad_example.py": {
                "fingerprint": "bad",
                "sources": [],
                "outputs": [],
            },
        },
    }
    (output_dir / run_examples.MARKER_NAME).write_text(json.dumps(marker))

    assert run_examples.generate_examples(repo, output_dir, python=sys.executable) == 0

    assert (
        capsys.readouterr().err.count("Ignoring invalid fixture stamp: bad_example.py")
        == 1
    )


def test_all_failed_producers_report_the_failure_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    output_dir = repo / "datasets/examples"
    _write_example(repo, "one", "raise RuntimeError('one failed')\n")
    _write_example(repo, "two", "raise RuntimeError('two failed')\n")

    assert run_examples.generate_examples(repo, output_dir, python=sys.executable) == 1

    captured = capsys.readouterr()
    assert "Examples FAILED: one_example.py two_example.py" in captured.out
    assert "no .zarr datasets" not in captured.err


def test_write_marker_rejects_an_empty_success_set(tmp_path: Path) -> None:
    output_dir = tmp_path / "datasets/examples"

    with pytest.raises(RuntimeError, match="no example producer succeeded"):
        run_examples.write_marker(output_dir, examples={})


def test_marker_uses_prebuild_fingerprint(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    output_dir = repo / "datasets/examples"
    output = output_dir / "one_example.luxar.zarr"
    writer = repo / "packages/luxar/src/luxar/io/writer.py"
    _write_example(
        repo,
        "one",
        "from pathlib import Path\n"
        "from luxar.io import writer\n"
        f"output = Path({str(output)!r})\n"
        "output.mkdir(parents=True)\n"
        "(output / 'zarr.json').write_text('fresh')\n"
        f"Path({str(writer)!r}).write_text('FORMAT = 3\\n')\n",
    )
    script = repo / "packages/luxar/examples/one_example.py"
    prebuild_fingerprint = run_examples.example_fingerprint(repo, script)

    assert run_examples.generate_examples(repo, output_dir, python=sys.executable) == 0

    marker = json.loads((output_dir / run_examples.MARKER_NAME).read_text())
    assert marker["examples"][script.name]["fingerprint"] == prebuild_fingerprint
    assert (
        "packages/luxar/src/luxar/io/writer.py"
        in marker["examples"][script.name]["sources"]
    )
    assert not run_examples.fixtures_are_current(repo, output_dir)


def test_force_rebuilds_current_fixtures(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    output_dir = repo / "datasets/examples"
    output = output_dir / "one_example.luxar.zarr"
    sentinel = repo / "executed"
    _write_example(
        repo,
        "one",
        "from pathlib import Path\n"
        f"output = Path({str(output)!r})\n"
        "output.mkdir(parents=True, exist_ok=True)\n"
        "(output / 'zarr.json').write_text('fresh')\n"
        f"Path({str(sentinel)!r}).touch()\n",
    )
    output.mkdir(parents=True)
    (output / "zarr.json").write_text("old")
    _write_marker(repo, output_dir)

    assert (
        run_examples.generate_examples(
            repo, output_dir, python=sys.executable, force=True
        )
        == 0
    )
    assert sentinel.exists()


def test_check_mode_reports_missing_compatible_marker_without_writing(
    tmp_path: Path, monkeypatch: object, capsys: object
) -> None:
    repo = _repo(tmp_path)
    output_dir = repo / "datasets/examples"
    output_dir.mkdir(parents=True)
    _write_example(repo, "one", "print('one')\n")
    monkeypatch.setattr(run_examples, "REPO_ROOT", repo)  # type: ignore[attr-defined]
    monkeypatch.setattr(run_examples, "OUTPUT_DIR", output_dir)  # type: ignore[attr-defined]
    monkeypatch.setattr(  # type: ignore[attr-defined]
        run_examples, "luxar_is_from_repo", lambda _repo_root: True
    )

    assert run_examples.main(["--check"]) == run_examples.STALE_EXIT_CODE
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "no compatible fixture marker" in captured.err
    assert "one_example.py" not in captured.err
    assert list(output_dir.iterdir()) == []


def test_check_mode_names_genuinely_stale_producers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    output_dir = repo / "datasets/examples"
    script = _generating_example(repo, "one", repo / "ran")
    assert run_examples.generate_examples(repo, output_dir, python=sys.executable) == 0
    script.write_text(script.read_text() + "# stale\n")
    monkeypatch.setattr(run_examples, "REPO_ROOT", repo)
    monkeypatch.setattr(run_examples, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(run_examples, "luxar_is_from_repo", lambda _repo_root: True)

    assert run_examples.main(["--check"]) == run_examples.STALE_EXIT_CODE
    assert "stale example producers: one_example.py" in capsys.readouterr().err


def test_main_rejects_luxar_imported_from_another_checkout(
    tmp_path: Path, monkeypatch: object
) -> None:
    repo = _repo(tmp_path)
    monkeypatch.setattr(run_examples, "REPO_ROOT", repo)  # type: ignore[attr-defined]
    monkeypatch.setattr(  # type: ignore[attr-defined]
        run_examples.luxar, "__file__", str(tmp_path / "other/luxar.py")
    )

    assert run_examples.main(["--check"]) == 1


def test_make_e2e_targets_require_fresh_example_fixtures() -> None:
    makefile = (_MOD_PATH.parent.parent / "Makefile").read_text()

    for target in (
        "test-e2e",
        "test-e2e-browsers",
        "test-e2e-mobile",
        "test-e2e-smoke",
        "test-e2e-smoke-strict",
        "test-perf-e2e",
    ):
        assert f"{target}: run-examples " in makefile


def test_every_e2e_package_script_has_a_make_entry_point_or_reason() -> None:
    repo = _MOD_PATH.parent.parent
    package = json.loads(
        (repo / "packages/luxar-viewer/package.json").read_text(encoding="utf-8")
    )
    scripts = {
        name
        for name in package["scripts"]
        if name == "test:e2e" or name.startswith("test:e2e:") or name == "test:perf:e2e"
    }
    direct_only = {
        "test:e2e:ci",
        "test:e2e:debug",
        "test:e2e:report",
        "test:e2e:ui",
        "test:e2e:visual",
        "test:e2e:visual:update",
    }
    makefile = (repo / "Makefile").read_text(encoding="utf-8")
    make_scripts = set(
        re.findall(r"pnpm (test:e2e(?::[\w-]+)*|test:perf:e2e)", makefile)
    )

    assert scripts == make_scripts | direct_only


def test_typescript_checker_uses_python_stale_exit_code() -> None:
    checker = viewer_source("tools/example-fixture-freshness.ts").read_text()

    assert f"STALE_EXIT_CODE = {run_examples.STALE_EXIT_CODE}" in checker
