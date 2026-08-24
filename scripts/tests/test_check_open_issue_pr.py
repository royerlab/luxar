"""Tests for the repository-wide open-PR issue guard."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import check_open_issue_pr as guard  # noqa: E402


def _pr(
    number: int,
    *closing_issues: int,
    title: str | None = None,
    body: str = "",
) -> guard.PullRequest:
    return guard.PullRequest(
        number=number,
        title=title or f"PR {number}",
        body=body,
        url=f"https://example.test/{number}",
        head_ref_name=f"branch-{number}",
        closing_issue_numbers=frozenset(closing_issues),
    )


def test_matching_open_pr_finds_the_incident_shape_before_ledger_recording() -> None:
    pull_requests = [
        _pr(2004, 2003),
        _pr(2005, 2003),
        _pr(2006),
    ]
    assert [
        pull_request.number
        for pull_request in guard.matching_pull_requests(2003, pull_requests)
    ] == [2004, 2005]


def test_current_pr_can_be_excluded_during_reconciliation() -> None:
    pull_requests = [_pr(2004, 2003), _pr(2005, 2003)]
    assert [
        pull_request.number
        for pull_request in guard.matching_pull_requests(
            2003, pull_requests, exclude_pr=2005
        )
    ] == [2004]


def test_path_comparison_never_treats_shared_files_as_equivalent() -> None:
    comparison = guard.compare_paths(
        {"viewer/element-id-map.ts", "viewer/init-picking.ts"},
        {"viewer/element-id-map.ts", "python/geometry_writers/README.md"},
    )
    assert comparison.duplicate_only == {"python/geometry_writers/README.md"}
    assert comparison.survivor_only == {"viewer/init-picking.ts"}
    assert comparison.shared == {"viewer/element-id-map.ts"}


def test_api_rows_keep_only_same_repository_closing_references() -> None:
    row = {
        "number": 2004,
        "title": "Docs",
        "url": "https://example.test/2004",
        "headRefName": "docs",
        "closingIssuesReferences": [
            {
                "number": 2003,
                "repository": {
                    "name": "luxar",
                    "owner": {"login": "royerlab"},
                },
            },
            {
                "number": 2003,
                "repository": {"nameWithOwner": "someone/else"},
            },
        ],
    }
    pull_request = guard.pull_request_from_row(row, "RoyerLab/Luxar")
    assert pull_request.closing_issue_numbers == {2003}


def test_pull_request_paths_flattens_every_api_page(monkeypatch) -> None:
    monkeypatch.setattr(
        guard,
        "_run_gh",
        lambda args: [
            [{"filename": "first.py"}, {"filename": "shared.ts"}],
            [{"filename": "last.md"}],
        ],
    )
    assert guard.pull_request_paths("royerlab/luxar", 2005) == {
        "first.py",
        "shared.ts",
        "last.md",
    }


@pytest.mark.parametrize(
    "outcome",
    [
        subprocess.CalledProcessError(
            4,
            ["gh", "pr", "list"],
            stderr="simulated gh failure\n",
        ),
        FileNotFoundError("gh not found"),
        subprocess.CompletedProcess(
            ["gh", "pr", "list"],
            returncode=0,
            stdout="not-json",
            stderr="",
        ),
    ],
)
def test_run_gh_failures_exit_three_without_traceback(
    monkeypatch, capsys, outcome
) -> None:
    def fake_run(*args, **kwargs):
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(guard.subprocess, "run", fake_run)
    with pytest.raises(SystemExit) as exc_info:
        guard._run_gh(["pr", "list"])
    assert exc_info.value.code == 3
    error = capsys.readouterr().err
    assert error.startswith("error:")
    assert "Traceback" not in error
    assert error.count("\n") == 1


def test_main_reports_claim_and_identifies_survivor_candidates(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        guard,
        "list_open_pull_requests",
        lambda repo: [_pr(2004, 2003), _pr(2010, 2010)],
    )
    assert guard.main(["2003"]) == 1
    output = capsys.readouterr().out
    assert "#2004 PR 2004" in output
    assert "#2010" not in output


def test_json_output_is_serializable_and_stable(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        guard,
        "list_open_pull_requests",
        lambda repo: [_pr(2004, 2011, 2003)],
    )
    assert guard.main(["2003", "--json"]) == 1
    row = json.loads(capsys.readouterr().out)[0]
    assert row["number"] == 2004
    assert row["closing_issue_numbers"] == [2003, 2011]


def test_compare_mode_reports_unique_and_shared_paths(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        guard,
        "list_open_pull_requests",
        lambda repo: [_pr(2004, 2003)],
    )
    paths = {
        2004: {"viewer/shared.ts", "viewer/survivor.ts"},
        2005: {"viewer/shared.ts", "python/unique.md"},
    }
    monkeypatch.setattr(guard, "pull_request_paths", lambda repo, number: paths[number])
    assert guard.main(["2003", "--exclude-pr", "2005", "--compare-pr", "2005"]) == 1
    output = capsys.readouterr().out
    assert "python/unique.md" in output
    assert "viewer/survivor.ts" in output
    assert "path overlap is not equivalence" in output


def test_compare_mode_reports_when_there_is_no_survivor(monkeypatch, capsys) -> None:
    monkeypatch.setattr(guard, "list_open_pull_requests", lambda repo: [])
    assert guard.main(["2011", "--exclude-pr", "2012", "--compare-pr", "2012"]) == 0
    assert (
        capsys.readouterr().out
        == "no other open PR declares it closes #2011; nothing to compare\n"
    )


def test_compare_mode_requires_excluding_the_duplicate_pr(monkeypatch) -> None:
    monkeypatch.setattr(guard, "list_open_pull_requests", lambda repo: [])
    with pytest.raises(SystemExit, match="2"):
        guard.main(["2003", "--compare-pr", "2005"])


def test_main_succeeds_when_issue_has_no_open_closing_pr(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        guard,
        "list_open_pull_requests",
        lambda repo: [_pr(2004)],
    )
    assert guard.main(["2003"]) == 0
    assert capsys.readouterr().out == ""


def test_loose_mode_advises_on_exact_non_closing_mentions(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        guard,
        "list_open_pull_requests",
        lambda repo: [
            _pr(2004, title="Part of #2003"),
            _pr(2005, body="Refs #20030"),
        ],
    )
    assert guard.main(["2003", "--loose"]) == 0
    output = capsys.readouterr().out
    assert "advisory" in output
    assert "#2004 Part of #2003" in output
    assert "#2005" not in output


def test_loose_mode_keeps_json_stdout_machine_readable(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        guard,
        "list_open_pull_requests",
        lambda repo: [_pr(2004, title="Part of #2003")],
    )
    assert guard.main(["2003", "--loose", "--json"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == []
    assert "advisory" in captured.err
