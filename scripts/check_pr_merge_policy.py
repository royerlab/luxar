#!/usr/bin/env python3
"""Refuse an automated merge when a PR edits a workflow file or is a fork.

Run immediately before ``gh pr merge`` in every merge path::

    hatch run python scripts/check_pr_merge_policy.py 3268

On success (exit 0) the script prints the exact head commit it checked; pass
that value to ``gh pr merge --match-head-commit <sha>`` so a push landing
after this check ran cannot slip an unreviewed workflow edit (or a fork head
swap) past it.

Exit 0 permits the workflow-file and fork-PR parts of the merge policy; it
does not replace the other merge gates. Exit 1 means the PR requires a manual
maintainer merge — either it edits a workflow file or it is a fork pull
request (``isCrossRepository``). Exit 3 means a GitHub query failed, returned
unusable data, or the PR's head commit moved while this script was checking
it. The file check uses the complete paginated PR file list, including
deleted and renamed files.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any

WORKFLOW_PREFIX = ".github/workflows/"
MAX_GITHUB_FILES = 3000


def pr_view(number: int, repo: str) -> dict[str, Any]:
    """Read the PR's current head commit and fork status, failing closed."""
    try:
        result = subprocess.run(
            [
                "gh",
                "pr",
                "view",
                str(number),
                "--repo",
                repo,
                "--json",
                "headRefOid,isCrossRepository",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        data: Any = json.loads(result.stdout)
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ) as error:
        raise ValueError("could not query GitHub PR head commit") from error

    if not isinstance(data, dict):
        raise ValueError("GitHub returned an invalid PR view response")
    head_sha = data.get("headRefOid")
    is_cross_repo = data.get("isCrossRepository")
    if not isinstance(head_sha, str) or not head_sha:
        raise ValueError("GitHub returned an invalid PR head commit")
    if not isinstance(is_cross_repo, bool):
        raise ValueError("GitHub returned an invalid PR fork status")
    return {"head_sha": head_sha, "is_cross_repo": is_cross_repo}


def changed_paths(number: int, repo: str) -> list[str]:
    """Read every changed filename from GitHub, failing closed on bad results."""
    try:
        result = subprocess.run(
            [
                "gh",
                "api",
                "--paginate",
                "--slurp",
                f"repos/{repo}/pulls/{number}/files?per_page=100",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        pages: Any = json.loads(result.stdout)
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ) as error:
        raise ValueError("could not query GitHub PR files") from error

    if not isinstance(pages, list) or not pages:
        raise ValueError("GitHub returned an invalid PR file list")
    paths: list[str] = []
    file_count = 0
    for page in pages:
        if not isinstance(page, list):
            raise ValueError("GitHub returned an invalid PR file page")
        for file in page:
            if (
                not isinstance(file, dict)
                or not isinstance(file.get("filename"), str)
                or not file["filename"]
            ):
                raise ValueError("GitHub returned an invalid PR filename")
            file_count += 1
            paths.append(file["filename"])
            previous = file.get("previous_filename")
            if previous is not None:
                if not isinstance(previous, str) or not previous:
                    raise ValueError("GitHub returned an invalid previous PR filename")
                paths.append(previous)
    if not file_count or file_count >= MAX_GITHUB_FILES:
        raise ValueError(
            "GitHub returned an empty or potentially truncated PR file list"
        )
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("number", type=int, help="pull request number")
    parser.add_argument("--repo", default="royerlab/luxar")
    args = parser.parse_args(argv)
    if args.number < 1:
        parser.error("pull request number must be positive")

    try:
        before = pr_view(args.number, args.repo)
        if before["is_cross_repo"]:
            print(
                f"#{args.number} is a fork pull request and requires a manual "
                "maintainer merge"
            )
            return 1
        paths = changed_paths(args.number, args.repo)
        after = pr_view(args.number, args.repo)
    except ValueError as error:
        print(f"error: {error}; refusing merge", file=sys.stderr)
        return 3

    if after["head_sha"] != before["head_sha"]:
        print(
            f"error: #{args.number} head moved from {before['head_sha']} to "
            f"{after['head_sha']} while checking; refusing merge",
            file=sys.stderr,
        )
        return 3

    workflow_paths = [path for path in paths if path.startswith(WORKFLOW_PREFIX)]
    if workflow_paths:
        print(f"#{args.number} requires a manual maintainer merge:")
        for path in workflow_paths:
            print(f"  {path}")
        return 1

    print(f"#{args.number}: no workflow files changed (head {after['head_sha']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
