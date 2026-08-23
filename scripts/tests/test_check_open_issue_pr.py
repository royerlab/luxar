"""Tests for the repository-wide open-PR issue guard."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import check_open_issue_pr as guard  # noqa: E402


def _pr(number: int, *closing_issues: int) -> guard.PullRequest:
    return guard.PullRequest(
        number=number,
        title=f"PR {number}",
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
    pull_request = guard.pull_request_from_row(row, "royerlab/luxar")
    assert pull_request.closing_issue_numbers == {2003}


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


def test_main_succeeds_when_issue_has_no_open_closing_pr(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        guard,
        "list_open_pull_requests",
        lambda repo: [_pr(2004)],
    )
    assert guard.main(["2003"]) == 0
    assert capsys.readouterr().out == ""
