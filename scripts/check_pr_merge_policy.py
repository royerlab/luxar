#!/usr/bin/env python3
"""Refuse an automated merge when a PR edits a workflow file.

Run immediately before ``gh pr merge`` in every merge path::

    hatch run python scripts/check_pr_merge_policy.py 3268

Exit 0 permits the workflow-file part of the merge policy; it does not replace
the other merge gates. Exit 1 requires a manual maintainer merge. Exit 3 means
the GitHub file query failed or returned unusable data, so the merge must stop.
The query uses the complete paginated PR file list, including deleted files.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any

REPO = "royerlab/luxar"
WORKFLOW_PREFIX = ".github/workflows/"
MAX_GITHUB_FILES = 3000


def changed_paths(number: int) -> list[str]:
    """Read every changed filename from GitHub, failing closed on bad results."""
    try:
        result = subprocess.run(
            [
                "gh",
                "api",
                "--paginate",
                "--slurp",
                f"repos/{REPO}/pulls/{number}/files?per_page=100",
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
    args = parser.parse_args(argv)
    if args.number < 1:
        parser.error("pull request number must be positive")

    try:
        paths = changed_paths(args.number)
    except ValueError as error:
        print(f"error: {error}; refusing merge", file=sys.stderr)
        return 3

    workflow_paths = [path for path in paths if path.startswith(WORKFLOW_PREFIX)]
    if workflow_paths:
        print(f"#{args.number} requires a manual maintainer merge:")
        for path in workflow_paths:
            print(f"  {path}")
        return 1
    print(f"#{args.number}: no workflow files changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
