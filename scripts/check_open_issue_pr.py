#!/usr/bin/env python3
"""Guard issue work against an already-open pull request.

The private ownership ledger is intentionally not the source of truth here: a
pull request can exist before its creator records it, and another worker may be
running from a different host. GitHub's open pull requests are the durable,
repository-wide view.

Usage::

    hatch run python scripts/check_open_issue_pr.py 2011
    hatch run python scripts/check_open_issue_pr.py 2011 --exclude-pr 2020

Exit 0 means the issue is unclaimed. Exit 1 means at least one open pull request
declares that it closes the issue. Mentions such as ``Refs #2011`` do not count.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PullRequest:
    number: int
    title: str
    url: str
    head_ref_name: str
    closing_issue_numbers: frozenset[int]


def matching_pull_requests(
    issue: int,
    pull_requests: Iterable[PullRequest],
    exclude_pr: int | None = None,
) -> list[PullRequest]:
    """Open pull requests that declare they close ``issue``."""
    return [
        pull_request
        for pull_request in pull_requests
        if pull_request.number != exclude_pr
        and issue in pull_request.closing_issue_numbers
    ]


def _run_gh(args: Sequence[str]) -> Any:
    result = subprocess.run(
        ["gh", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _reference_repo(reference: dict[str, Any]) -> str | None:
    repository = reference.get("repository") or {}
    if name_with_owner := repository.get("nameWithOwner"):
        return str(name_with_owner)
    owner = (repository.get("owner") or {}).get("login")
    name = repository.get("name")
    return f"{owner}/{name}" if owner and name else None


def pull_request_from_row(row: dict[str, Any], repo: str) -> PullRequest:
    """Normalize one ``gh pr list`` row and keep same-repo closing issues."""
    return PullRequest(
        number=row["number"],
        title=row["title"],
        url=row["url"],
        head_ref_name=row["headRefName"],
        closing_issue_numbers=frozenset(
            reference["number"]
            for reference in row.get("closingIssuesReferences", [])
            if _reference_repo(reference) == repo
        ),
    )


def list_open_pull_requests(repo: str) -> list[PullRequest]:
    """Read all open pull requests from GitHub, following pagination."""
    rows = _run_gh(
        [
            "pr",
            "list",
            "--repo",
            repo,
            "--state",
            "open",
            "--limit",
            "1000",
            "--json",
            "number,title,url,headRefName,closingIssuesReferences",
        ]
    )
    return [pull_request_from_row(row, repo) for row in rows]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("issue", type=int)
    parser.add_argument("--repo", default="royerlab/luxar")
    parser.add_argument("--exclude-pr", type=int)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    matches = matching_pull_requests(
        args.issue,
        list_open_pull_requests(args.repo),
        args.exclude_pr,
    )
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "number": pull_request.number,
                        "title": pull_request.title,
                        "url": pull_request.url,
                        "head_ref_name": pull_request.head_ref_name,
                        "closing_issue_numbers": sorted(
                            pull_request.closing_issue_numbers
                        ),
                    }
                    for pull_request in matches
                ]
            )
        )
    elif matches:
        print(f"issue #{args.issue} already has {len(matches)} open pull request(s):")
        for pull_request in matches:
            print(
                f"  #{pull_request.number} {pull_request.title} "
                f"({pull_request.head_ref_name}) {pull_request.url}"
            )
    return 1 if matches else 0


if __name__ == "__main__":
    sys.exit(main())
