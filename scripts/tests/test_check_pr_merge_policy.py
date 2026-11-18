"""The merge pass must check GitHub's complete changed-file list and fork status."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "check_pr_merge_policy.py"

FAKE_GH_SCRIPT = """\
import json
import sys
from pathlib import Path

base = Path(__file__).resolve().parent
argv = sys.argv[1:]

with (base / "calls.log").open("a") as handle:
    handle.write(json.dumps(argv) + "\\n")

if argv[:2] == ["pr", "view"]:
    returncode = int((base / "view_returncode.txt").read_text())
    if returncode != 0:
        sys.exit(returncode)
    responses = json.loads((base / "view_responses.json").read_text())
    counter_file = base / "view_calls"
    idx = int(counter_file.read_text())
    counter_file.write_text(str(idx + 1))
    print(json.dumps(responses[idx]))
    sys.exit(0)

if argv[:2] == ["api", "--paginate"]:
    returncode = int((base / "files_returncode.txt").read_text())
    print((base / "files_response.json").read_text())
    sys.exit(returncode)

sys.exit(2)
"""


def _write_gh(
    tmp_path: Path,
    *,
    view_responses: list[dict[str, object]] | None = None,
    view_returncode: int = 0,
    pages: list[list[dict[str, str]]] | None = None,
    files_returncode: int = 0,
) -> None:
    (tmp_path / "view_responses.json").write_text(
        json.dumps(view_responses if view_responses is not None else [])
    )
    (tmp_path / "view_returncode.txt").write_text(str(view_returncode))
    (tmp_path / "view_calls").write_text("0")
    (tmp_path / "files_response.json").write_text(
        json.dumps(pages if pages is not None else [])
    )
    (tmp_path / "files_returncode.txt").write_text(str(files_returncode))
    gh = tmp_path / "gh"
    gh.write_text(f"#!{sys.executable}\n{FAKE_GH_SCRIPT}")
    gh.chmod(0o755)


def _run(
    tmp_path: Path, args: list[str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *(args if args is not None else ["3268"])],
        env={**os.environ, "PATH": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )


def _calls(tmp_path: Path) -> list[list[str]]:
    log = tmp_path / "calls.log"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines()]


HEAD = "a" * 40
OTHER_HEAD = "b" * 40


def _views(head: str = HEAD, *, is_cross_repo: bool = False) -> list[dict[str, object]]:
    response = {"headRefOid": head, "isCrossRepository": is_cross_repo}
    return [response, response]


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        ([".github/workflows/ci.yml", "scripts/tests/test_ci.py"], 1),
        (["scripts/tests/test_ci.py", ".github/workflows/publish.yml"], 1),
        ([".github/workflows-old/ci.yml", "scripts/test.py"], 0),
    ],
)
def test_workflow_file_blocks_merge(
    tmp_path: Path, files: list[str], expected: int
) -> None:
    _write_gh(
        tmp_path,
        view_responses=_views(),
        pages=[[{"filename": file} for file in files]],
    )
    result = _run(tmp_path)
    assert result.returncode == expected, result.stderr


def test_checks_all_pages(tmp_path: Path) -> None:
    _write_gh(
        tmp_path,
        view_responses=_views(),
        pages=[
            [{"filename": f"other/{n}.py"} for n in range(100)],
            [{"filename": ".github/workflows/ci.yml"}],
        ],
    )
    result = _run(tmp_path)
    assert result.returncode == 1
    assert ".github/workflows/ci.yml" in result.stdout


def test_renamed_workflow_is_blocked_by_previous_filename(tmp_path: Path) -> None:
    _write_gh(
        tmp_path,
        view_responses=_views(),
        pages=[
            [
                {
                    "filename": "archive/ci.yml",
                    "previous_filename": ".github/workflows/ci.yml",
                }
            ]
        ],
    )
    result = _run(tmp_path)
    assert result.returncode == 1
    assert ".github/workflows/ci.yml" in result.stdout


def test_github_file_limit_fails_closed(tmp_path: Path) -> None:
    _write_gh(
        tmp_path,
        view_responses=_views(),
        pages=[[{"filename": f"src/{n}.py"} for n in range(3000)]],
    )
    result = _run(tmp_path)
    assert result.returncode == 3


@pytest.mark.parametrize("pages", [[], [[]], [[{}]], {"files": []}])
def test_malformed_response_fails_closed(tmp_path: Path, pages: Any) -> None:
    _write_gh(tmp_path, view_responses=_views(), pages=pages)
    result = _run(tmp_path)
    assert result.returncode == 3


def test_github_failure_fails_closed(tmp_path: Path) -> None:
    _write_gh(tmp_path, view_responses=_views(), pages=[], files_returncode=1)
    result = _run(tmp_path)
    assert result.returncode == 3


def test_prints_head_commit_on_success(tmp_path: Path) -> None:
    _write_gh(
        tmp_path,
        view_responses=_views(),
        pages=[[{"filename": "scripts/test.py"}]],
    )
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert HEAD in result.stdout


def test_fork_pr_requires_manual_merge_without_querying_files(tmp_path: Path) -> None:
    _write_gh(tmp_path, view_responses=_views(is_cross_repo=True))
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "fork" in result.stdout.lower()
    calls = _calls(tmp_path)
    assert len(calls) == 1, calls
    assert calls[0][:2] == ["pr", "view"]


def test_head_moved_during_check_fails_closed(tmp_path: Path) -> None:
    _write_gh(
        tmp_path,
        view_responses=[
            {"headRefOid": HEAD, "isCrossRepository": False},
            {"headRefOid": OTHER_HEAD, "isCrossRepository": False},
        ],
        pages=[[{"filename": "scripts/test.py"}]],
    )
    result = _run(tmp_path)
    assert result.returncode == 3
    assert HEAD in result.stderr
    assert OTHER_HEAD in result.stderr


def test_pr_view_failure_fails_closed(tmp_path: Path) -> None:
    _write_gh(tmp_path, view_returncode=1)
    result = _run(tmp_path)
    assert result.returncode == 3


@pytest.mark.parametrize(
    "response",
    [
        [],
        {"headRefOid": "", "isCrossRepository": False},
        {"isCrossRepository": False},
        {"headRefOid": HEAD},
        {"headRefOid": HEAD, "isCrossRepository": "false"},
    ],
)
def test_malformed_pr_view_response_fails_closed(tmp_path: Path, response: Any) -> None:
    _write_gh(tmp_path, view_responses=[response])
    result = _run(tmp_path)
    assert result.returncode == 3


def test_repo_flag_is_forwarded_to_every_gh_call(tmp_path: Path) -> None:
    _write_gh(
        tmp_path,
        view_responses=_views(),
        pages=[[{"filename": "scripts/test.py"}]],
    )
    result = _run(tmp_path, ["3268", "--repo", "someone/fork"])
    assert result.returncode == 0, result.stderr
    calls = _calls(tmp_path)
    assert calls[0] == [
        "pr",
        "view",
        "3268",
        "--repo",
        "someone/fork",
        "--json",
        "headRefOid,isCrossRepository",
    ]
    assert calls[1] == [
        "api",
        "--paginate",
        "--slurp",
        "repos/someone/fork/pulls/3268/files?per_page=100",
    ]
