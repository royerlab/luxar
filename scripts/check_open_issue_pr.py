#!/usr/bin/env python3
"""Guard issue work against an already-open pull request.

A local branch list is intentionally not the source of truth here: a pull
request may have been created from another checkout or host. GitHub's open pull
requests are the durable, repository-wide view.

Usage::

    hatch run python scripts/check_open_issue_pr.py 2011
    hatch run python scripts/check_open_issue_pr.py 2011 --exclude-pr 2020
    hatch run python scripts/check_open_issue_pr.py 2011 --loose
    hatch run python scripts/check_open_issue_pr.py 2011 --json
    hatch run python scripts/check_open_issue_pr.py 2011 \
        --exclude-pr 2020 --compare-pr 2020

Exit 0 means the issue is unclaimed. Exit 1 means at least one open pull request
declares that it closes the issue. Exit 3 means the GitHub query failed.
Mentions such as ``Refs #2011`` do not count unless ``--loose`` is supplied;
loose matches are advisory and do not change the exit status. ``--compare-pr``
inventories unique and shared paths before a duplicate PR is closed; shared
paths still require patch review and are never called equivalent.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, TextIO


@dataclass(frozen=True)
class PullRequest:
    number: int
    title: str
    body: str
    url: str
    head_ref_name: str
    closing_issue_numbers: frozenset[int]


@dataclass(frozen=True)
class PathComparison:
    survivor_only: frozenset[str]
    duplicate_only: frozenset[str]
    shared: frozenset[str]


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


def loosely_matching_pull_requests(
    issue: int,
    pull_requests: Iterable[PullRequest],
    exclude_pr: int | None = None,
) -> list[PullRequest]:
    """Open pull requests that mention ``issue`` without closing it."""
    mention = re.compile(rf"(?<!\d)#{issue}(?!\d)")
    return [
        pull_request
        for pull_request in pull_requests
        if pull_request.number != exclude_pr
        and issue not in pull_request.closing_issue_numbers
        and mention.search(f"{pull_request.title}\n{pull_request.body}")
    ]


def _run_gh(args: Sequence[str]) -> Any:
    try:
        result = subprocess.run(
            ["gh", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print("error: GitHub CLI 'gh' was not found", file=sys.stderr)
        raise SystemExit(3) from None
    except subprocess.CalledProcessError as error:
        stderr_lines = [line.strip() for line in (error.stderr or "").splitlines()]
        detail = next(
            (line for line in reversed(stderr_lines) if line),
            f"exit status {error.returncode}",
        )
        print(f"error: gh command failed: {detail}", file=sys.stderr)
        raise SystemExit(3) from None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print("error: gh returned invalid JSON", file=sys.stderr)
        raise SystemExit(3) from None


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
        body=str(row.get("body") or ""),
        url=row["url"],
        head_ref_name=row["headRefName"],
        closing_issue_numbers=frozenset(
            reference["number"]
            for reference in row.get("closingIssuesReferences", [])
            if (_reference_repo(reference) or "").lower() == repo.lower()
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
            "number,title,body,url,headRefName,closingIssuesReferences",
        ]
    )
    return [pull_request_from_row(row, repo) for row in rows]


def pull_request_paths(repo: str, number: int) -> frozenset[str]:
    """Changed paths for an open or closed pull request."""
    pages = _run_gh(
        [
            "api",
            "--paginate",
            "--slurp",
            f"repos/{repo}/pulls/{number}/files?per_page=100",
        ]
    )
    return frozenset(file["filename"] for page in pages for file in page)


def compare_paths(
    survivor_paths: Iterable[str], duplicate_paths: Iterable[str]
) -> PathComparison:
    """Partition two PR path sets without claiming shared patches are equal."""
    survivor = frozenset(survivor_paths)
    duplicate = frozenset(duplicate_paths)
    return PathComparison(
        survivor_only=survivor - duplicate,
        duplicate_only=duplicate - survivor,
        shared=survivor & duplicate,
    )


def _print_paths(label: str, paths: Iterable[str]) -> None:
    ordered = sorted(paths)
    print(f"{label} ({len(ordered)}):")
    for path in ordered:
        print(f"  {path}")


def print_comparison(repo: str, survivor: PullRequest, duplicate_pr: int) -> None:
    comparison = compare_paths(
        pull_request_paths(repo, survivor.number),
        pull_request_paths(repo, duplicate_pr),
    )
    print(f"compare survivor #{survivor.number} with duplicate #{duplicate_pr}")
    _print_paths(
        "duplicate-only paths (must be transferred or explained)",
        comparison.duplicate_only,
    )
    _print_paths("survivor-only paths", comparison.survivor_only)
    _print_paths(
        "shared paths (inspect both patches; path overlap is not equivalence)",
        comparison.shared,
    )


def print_loose_matches(
    issue: int, matches: Iterable[PullRequest], stream: TextIO | None = None
) -> None:
    if stream is None:
        stream = sys.stdout
    ordered = list(matches)
    print(
        f"advisory: issue #{issue} is mentioned by {len(ordered)} open "
        "pull request(s) without a closing reference:",
        file=stream,
    )
    for pull_request in ordered:
        print(
            f"  #{pull_request.number} {pull_request.title} "
            f"({pull_request.head_ref_name}) {pull_request.url}",
            file=stream,
        )


def _print_json_results(
    issue: int,
    matches: Iterable[PullRequest],
    loose_matches: Iterable[PullRequest],
) -> None:
    print(
        json.dumps(
            [
                {
                    "number": pull_request.number,
                    "title": pull_request.title,
                    "url": pull_request.url,
                    "head_ref_name": pull_request.head_ref_name,
                    "closing_issue_numbers": sorted(pull_request.closing_issue_numbers),
                }
                for pull_request in matches
            ]
        )
    )
    loose_matches = list(loose_matches)
    if loose_matches:
        print_loose_matches(issue, loose_matches, sys.stderr)


def _print_human_results(
    issue: int,
    repo: str,
    matches: Sequence[PullRequest],
    compare_pr: int | None,
    loose_matches: Iterable[PullRequest],
) -> None:
    if matches:
        print(f"issue #{issue} already has {len(matches)} open pull request(s):")
        for pull_request in matches:
            print(
                f"  #{pull_request.number} {pull_request.title} "
                f"({pull_request.head_ref_name}) {pull_request.url}"
            )
        if compare_pr is not None:
            for pull_request in matches:
                print_comparison(repo, pull_request, compare_pr)
    elif compare_pr is not None:
        print(f"no other open PR declares it closes #{issue}; nothing to compare")
    loose_matches = list(loose_matches)
    if loose_matches:
        print_loose_matches(issue, loose_matches)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("issue", type=int)
    parser.add_argument("--repo", default="royerlab/luxar")
    parser.add_argument("--exclude-pr", type=int)
    parser.add_argument("--compare-pr", type=int)
    parser.add_argument("--loose", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.compare_pr is not None and args.exclude_pr != args.compare_pr:
        parser.error("--compare-pr requires the same PR number in --exclude-pr")
    if args.compare_pr is not None and args.json:
        parser.error("--compare-pr cannot be combined with --json")

    pull_requests = list_open_pull_requests(args.repo)
    matches = matching_pull_requests(
        args.issue,
        pull_requests,
        args.exclude_pr,
    )
    loose_matches = (
        loosely_matching_pull_requests(args.issue, pull_requests, args.exclude_pr)
        if args.loose
        else []
    )
    if args.json:
        _print_json_results(args.issue, matches, loose_matches)
    else:
        _print_human_results(
            args.issue,
            args.repo,
            matches,
            args.compare_pr,
            loose_matches,
        )
    return 1 if matches else 0


if __name__ == "__main__":
    sys.exit(main())
