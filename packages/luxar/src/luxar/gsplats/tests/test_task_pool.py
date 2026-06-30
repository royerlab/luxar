# test_task_pool.py
"""Tests for the generic subprocess task pool (env pinning + resume-skip)."""

from __future__ import annotations

import sys
from pathlib import Path

from luxar.gsplats.batch.task_pool import run_task_pool


def test_env_builder_injects_per_task_env(tmp_path: Path) -> None:
    """Each worker writes its CUDA_VISIBLE_DEVICES to a marker — proving the
    per-task env injection pins each task to its assigned device."""
    assign = {0: "0", 1: "1", 2: "0"}

    def argv(key: int) -> list[str]:
        marker = tmp_path / f"marker_{key}.txt"
        code = (
            "import os,sys;"
            f"open({str(marker)!r},'w').write("
            "os.environ.get('CUDA_VISIBLE_DEVICES','UNSET'))"
        )
        return [sys.executable, "-c", code]

    def env(key: int) -> dict[str, str]:
        return {"CUDA_VISIBLE_DEVICES": assign[key]}

    results = run_task_pool(
        [0, 1, 2], max_workers=3, argv_builder=argv, env_builder=env
    )
    assert all(r.ok for r in results)
    assert (tmp_path / "marker_0.txt").read_text() == "0"
    assert (tmp_path / "marker_1.txt").read_text() == "1"
    assert (tmp_path / "marker_2.txt").read_text() == "0"


def test_no_env_builder_inherits_environment(tmp_path: Path) -> None:
    """Without env_builder the worker inherits os.environ (PATH present)."""
    marker = tmp_path / "path.txt"
    code = (
        f"import os;open({str(marker)!r},'w').write(str(bool(os.environ.get('PATH'))))"
    )
    results = run_task_pool(
        [0], max_workers=1, argv_builder=lambda k: [sys.executable, "-c", code]
    )
    assert results[0].ok
    assert marker.read_text() == "True"


def test_skip_if_skips_without_launching(tmp_path: Path) -> None:
    """A skipped task must not run its subprocess."""
    sentinel = tmp_path / "ran.txt"

    def argv(key: int) -> list[str]:
        code = f"open({str(sentinel)!r},'w').write('ran')"
        return [sys.executable, "-c", code]

    results = run_task_pool(
        [0], max_workers=1, argv_builder=argv, skip_if=lambda k: True
    )
    assert results[0].skipped is True
    assert results[0].ok
    assert not sentinel.exists()  # subprocess never launched


def test_failing_task_captured_not_raised() -> None:
    """A non-zero worker is reported, not raised; output captured."""

    def argv(key: int) -> list[str]:
        return [
            sys.executable,
            "-c",
            "import sys; sys.stderr.write('boom'); sys.exit(3)",
        ]

    results = run_task_pool([0], max_workers=1, argv_builder=argv)
    assert results[0].returncode == 3
    assert not results[0].ok
    assert "boom" in results[0].output


def test_build_failure_funneled_to_returncode_minus_one() -> None:
    def argv(key: int) -> list[str]:
        raise RuntimeError("bad builder")

    results = run_task_pool([0], max_workers=1, argv_builder=argv)
    assert results[0].returncode == -1
    assert "bad builder" in results[0].output


def test_on_done_progress_callback() -> None:
    seen: list[tuple[int, int]] = []
    run_task_pool(
        [0, 1],
        max_workers=2,
        argv_builder=lambda k: [sys.executable, "-c", "pass"],
        on_done=lambda res, n, total: seen.append((n, total)),
    )
    assert sorted(seen) == [(1, 2), (2, 2)]
