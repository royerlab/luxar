"""The merge pass must check GitHub's complete changed-file list."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "check_pr_merge_policy.py"


def _run(tmp_path: Path, pages: list[list[dict[str, str]]], *, returncode: int = 0):
    gh = tmp_path / "gh"
    gh.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "assert sys.argv[1:] == ['api', '--paginate', '--slurp', "
        "'repos/royerlab/luxar/pulls/3268/files?per_page=100']\n"
        f"print({json.dumps(json.dumps(pages))})\n"
        f"sys.exit({returncode})\n"
    )
    gh.chmod(0o755)
    return subprocess.run(
        [sys.executable, str(SCRIPT), "3268"],
        env={"PATH": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        ([".github/workflows/ci.yml", "scripts/tests/test_ci.py"], 1),
        (["scripts/tests/test_ci.py", ".github/workflows/publish.yml"], 1),
        ([".github/workflows-old/ci.yml", "scripts/test.py"], 0),
    ],
)
def test_workflow_file_blocks_merge(tmp_path: Path, files: list[str], expected: int):
    result = _run(tmp_path, [[{"filename": file} for file in files]])
    assert result.returncode == expected, result.stderr


def test_checks_all_pages(tmp_path: Path):
    result = _run(
        tmp_path,
        [
            [{"filename": f"other/{n}.py"} for n in range(100)],
            [{"filename": ".github/workflows/ci.yml"}],
        ],
    )
    assert result.returncode == 1
    assert ".github/workflows/ci.yml" in result.stdout


def test_renamed_workflow_is_blocked_by_previous_filename(tmp_path: Path):
    result = _run(
        tmp_path,
        [
            [
                {
                    "filename": "archive/ci.yml",
                    "previous_filename": ".github/workflows/ci.yml",
                }
            ]
        ],
    )
    assert result.returncode == 1
    assert ".github/workflows/ci.yml" in result.stdout


def test_github_file_limit_fails_closed(tmp_path: Path):
    result = _run(tmp_path, [[{"filename": f"src/{n}.py"} for n in range(3000)]])
    assert result.returncode == 3


@pytest.mark.parametrize("pages", [[], [[]], [[{}]], {"files": []}])
def test_malformed_response_fails_closed(tmp_path: Path, pages):
    result = _run(tmp_path, pages)
    assert result.returncode == 3


def test_github_failure_fails_closed(tmp_path: Path):
    result = _run(tmp_path, [], returncode=1)
    assert result.returncode == 3
