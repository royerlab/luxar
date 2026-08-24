"""Tests for freshness-aware example fixture generation."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

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
    (repo / "pyproject.toml").write_text("[project]\nname = 'luxar'\n")
    (production / "writer.py").write_text("FORMAT = 2\n")
    return repo


def _write_example(repo: Path, name: str, body: str) -> Path:
    script = repo / "packages/luxar/examples" / f"{name}_example.py"
    script.write_text(body)
    return script


def test_fingerprint_covers_builders_and_production_writer_code(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    example = _write_example(repo, "one", "print('one')\n")
    writer = repo / "packages/luxar/src/luxar/io/writer.py"

    initial = run_examples.source_fingerprint(repo)
    example.write_text("print('changed')\n")
    after_builder_change = run_examples.source_fingerprint(repo)
    writer.write_text("FORMAT = 3\n")
    after_writer_change = run_examples.source_fingerprint(repo)

    assert after_builder_change != initial
    assert after_writer_change != after_builder_change


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
                "fingerprint": run_examples.source_fingerprint(repo),
                "environment": run_examples.build_environment(),
                "outputs": [output.name],
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
    run_examples.write_marker(repo, output_dir)

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
    run_examples.write_marker(repo, output_dir)

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
    run_examples.write_marker(repo, output_dir)

    assert run_examples.generate_examples(repo, output_dir, python=sys.executable) == 0
    assert not sentinel.exists()


def test_stale_fixtures_rebuild_all_examples_and_stamp_outputs(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    output_dir = repo / "datasets/examples"
    stale = output_dir / "removed_example.luxar.zarr"
    stale.mkdir(parents=True)
    unknown = output_dir / "handmade.luxar.zarr"
    unknown.mkdir()
    run_examples.write_marker(repo, output_dir, outputs=[stale.name])
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
    assert marker["outputs"] == [
        "one_example.luxar.zarr",
        "two_example.luxar.zarr",
    ]


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


def test_failed_rebuild_attempts_every_example_and_leaves_no_marker(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    output_dir = repo / "datasets/examples"
    previous = output_dir / "previous_example.luxar.zarr"
    previous.mkdir(parents=True)
    (previous / "zarr.json").write_text("old")
    run_examples.write_marker(repo, output_dir)
    reached = repo / "second-ran"
    _write_example(repo, "one", "raise RuntimeError('broken')\n")
    _write_example(
        repo, "two", f"from pathlib import Path\nPath({str(reached)!r}).touch()\n"
    )

    assert run_examples.generate_examples(repo, output_dir, python=sys.executable) == 1

    assert reached.exists()
    assert previous.exists()
    assert (previous / "zarr.json").read_text() == "old"
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
    run_examples.write_marker(repo, output_dir, outputs=[previous.name])

    assert (
        run_examples.generate_examples(
            repo, output_dir, python=sys.executable, force=True
        )
        == 1
    )

    assert sentinel.read_text() == "old"
    assert not (output_dir / run_examples.MARKER_NAME).exists()


def test_marker_uses_prebuild_fingerprint(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    output_dir = repo / "datasets/examples"
    output = output_dir / "one_example.luxar.zarr"
    writer = repo / "packages/luxar/src/luxar/io/writer.py"
    _write_example(
        repo,
        "one",
        "from pathlib import Path\n"
        f"output = Path({str(output)!r})\n"
        "output.mkdir(parents=True)\n"
        "(output / 'zarr.json').write_text('fresh')\n"
        f"Path({str(writer)!r}).write_text('FORMAT = 3\\n')\n",
    )
    prebuild_fingerprint = run_examples.source_fingerprint(repo)

    assert run_examples.generate_examples(repo, output_dir, python=sys.executable) == 0

    marker = json.loads((output_dir / run_examples.MARKER_NAME).read_text())
    assert marker["fingerprint"] == prebuild_fingerprint
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
    run_examples.write_marker(repo, output_dir)

    assert (
        run_examples.generate_examples(
            repo, output_dir, python=sys.executable, force=True
        )
        == 0
    )
    assert sentinel.exists()


def test_check_mode_reports_stale_without_writing(
    tmp_path: Path, monkeypatch: object
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
    assert list(output_dir.iterdir()) == []


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

    for target in ("test-e2e", "test-e2e-smoke", "test-perf-e2e"):
        assert f"{target}: run-examples " in makefile
