"""Tests for scripts/zenodo_migration_audit.py (the demo-data migration audit).

The script is a standalone repo script (not part of the installed ``luxar``
package), so it is loaded from ``REPO_ROOT/scripts/`` via importlib and the whole
module is skipped on a packaged install that ships no ``scripts/`` tree.

Two things are worth guarding: that the audit RUNS end to end (its sections are
wired together by hand, and a mismatch between them is invisible until the last
line prints), and that "ready to upload" means what it says — Zenodo publication
cannot be undone, so a dataset whose declared files are only partly here, or
whose in-tree copy is an unpulled git-LFS pointer, must not be reported ready.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[6]
_SCRIPT = REPO_ROOT / "scripts" / "zenodo_migration_audit.py"

pytestmark = pytest.mark.skipif(
    not _SCRIPT.exists(),
    reason="audit script not present (packaged install without repo scripts/)",
)

_LFS_POINTER = (
    b"version https://git-lfs.github.com/spec/v1\n"
    b"oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
    b"size 12345\n"
)


def _load(repo: Path, cache: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the audit against a throwaway repo tree and cache root.

    ``REPO`` comes from ``sys.argv[1]`` at import time, so the argv patch has to
    be in place before the module body runs; ``CACHE`` is home-relative and is
    redirected afterwards.
    """
    monkeypatch.setattr(sys, "argv", ["zenodo_migration_audit", str(repo)])
    spec = importlib.util.spec_from_file_location("zenodo_migration_audit", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "CACHE", cache)
    return module


def _write_manifest(repo: Path, datasets: dict) -> Path:
    """Write a minimal manifest into a fake repo tree; return its data dir."""
    demos = repo / "packages/luxar/src/luxar/demos"
    demos.mkdir(parents=True)
    (demos / "data_manifest.json").write_text(
        json.dumps(
            {
                "records": {
                    "cc-by": {
                        "license": "cc-by-4.0",
                        "zenodo_record": None,
                        "zenodo_doi": None,
                    }
                },
                "datasets": datasets,
            }
        )
    )
    data = demos / "data"
    data.mkdir()
    return data


def _two_file_dataset() -> dict:
    return {
        "ds": {
            "bucket": "zenodo",
            "record": "cc-by",
            "license": "cc-by-4.0",
            "dir": "ds",
            "files": [
                {"name": "a.zip", "sha256": "aa", "bytes": 1024},
                {"name": "b.zip", "sha256": "bb", "bytes": 1024},
            ],
        }
    }


def test_audit_runs_end_to_end_on_the_real_repo() -> None:
    """The whole audit prints its readiness summary and exits cleanly.

    Exit 1 is a legitimate result (undeclared files on disk); a traceback is
    not — the sections pass their results along by hand, and a signature that
    drifts from its call site only shows up here.
    """
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode in (0, 1), proc.stderr
    assert "Traceback" not in proc.stderr, proc.stderr
    assert "READINESS" in proc.stdout
    assert "datasets ready to upload now:" in proc.stdout


def test_complete_dataset_is_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    repo, cache = tmp_path / "repo", tmp_path / "cache"
    data = _write_manifest(repo, _two_file_dataset())
    (data / "ds").mkdir()
    (data / "ds" / "a.zip").write_bytes(b"real bytes")
    # The second file resolves from the cache: repo + cache together cover the
    # declared set, which is what "ready" means.
    (cache / "ds").mkdir(parents=True)
    (cache / "ds" / "b.zip").write_bytes(b"real bytes")

    audit = _load(repo, cache, monkeypatch)
    assert audit.main() == 0
    out = capsys.readouterr().out
    assert "datasets ready to upload now: 1" in out
    assert "incomplete here (NOT ready):  0" in out


def test_partial_dataset_is_not_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """One of two declared files present is INCOMPLETE, never ready."""
    repo, cache = tmp_path / "repo", tmp_path / "cache"
    data = _write_manifest(repo, _two_file_dataset())
    (data / "ds").mkdir()
    (data / "ds" / "a.zip").write_bytes(b"real bytes")

    audit = _load(repo, cache, monkeypatch)
    assert audit.main() == 0
    out = capsys.readouterr().out
    assert "INCOMPLETE: 1 of 2 files have bytes here" in out
    assert "datasets ready to upload now: 0" in out
    assert "incomplete here (NOT ready):  1" in out


def test_unpulled_lfs_pointer_counts_as_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """A pointer stub is a path without bytes — nothing to upload from here."""
    repo, cache = tmp_path / "repo", tmp_path / "cache"
    data = _write_manifest(repo, _two_file_dataset())
    (data / "ds").mkdir()
    (data / "ds" / "a.zip").write_bytes(_LFS_POINTER)
    (data / "ds" / "b.zip").write_bytes(_LFS_POINTER)

    audit = _load(repo, cache, monkeypatch)
    assert audit.main() == 0
    out = capsys.readouterr().out
    assert "BYTES NOT ON THIS MACHINE" in out
    assert "datasets ready to upload now: 0" in out
