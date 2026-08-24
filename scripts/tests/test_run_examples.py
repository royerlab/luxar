"""Tests for freshness-aware example fixture generation."""

from __future__ import annotations

import importlib.util
import json
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
                "outputs": [output.name],
            }
        )
    )

    assert run_examples.fixtures_are_current(repo, output_dir)
    output.rmdir()
    assert not run_examples.fixtures_are_current(repo, output_dir)


def test_current_marker_rejects_an_unrecorded_extra_output(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _write_example(repo, "one", "print('one')\n")
    output_dir = repo / "datasets/examples"
    (output_dir / "one_example.luxar.zarr").mkdir(parents=True)
    run_examples.write_marker(repo, output_dir)

    (output_dir / "removed_example.luxar.zarr").mkdir()
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
    for name in ("one", "two"):
        output = output_dir / f"{name}_example.luxar.zarr"
        _write_example(
            repo,
            name,
            f"from pathlib import Path\nPath({str(output)!r}).mkdir(parents=True)\n",
        )

    assert run_examples.generate_examples(repo, output_dir, python=sys.executable) == 0

    assert not stale.exists()
    fixtures = sorted(path.name for path in output_dir.glob("*.zarr"))
    assert fixtures
    assert fixtures == ["one_example.luxar.zarr", "two_example.luxar.zarr"]
    assert run_examples.fixtures_are_current(repo, output_dir)


def test_failed_rebuild_attempts_every_example_and_leaves_no_marker(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    output_dir = repo / "datasets/examples"
    reached = repo / "second-ran"
    _write_example(repo, "one", "raise RuntimeError('broken')\n")
    _write_example(
        repo, "two", f"from pathlib import Path\nPath({str(reached)!r}).touch()\n"
    )

    assert run_examples.generate_examples(repo, output_dir, python=sys.executable) == 1

    assert reached.exists()
    assert not (output_dir / run_examples.MARKER_NAME).exists()


def test_make_e2e_targets_require_fresh_example_fixtures() -> None:
    makefile = (_MOD_PATH.parent.parent / "Makefile").read_text()

    for target in ("test-e2e", "test-e2e-smoke", "test-perf-e2e"):
        assert f"{target}: run-examples " in makefile
