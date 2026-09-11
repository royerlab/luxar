"""Fail when a repository-owned workflow cadence stops succeeding."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode

API_ROOT = "https://api.github.com"
REQUEST_TIMEOUT_SECONDS = 30
REPO_ROOT = Path(__file__).resolve().parents[1]
OpenUrl = Callable[..., Any]


@dataclass(frozen=True)
class Cadence:
    workflow_name: str
    workflow_path: str
    branch: str
    event: str
    max_age: timedelta
    bootstrap_grace: timedelta


@dataclass(frozen=True)
class Result:
    cadence: Cadence
    level: str
    detail: str


CADENCES = (
    Cadence(
        workflow_name="CUDA native cadence",
        workflow_path=".github/workflows/cuda-native.yml",
        branch="dev",
        event="workflow_dispatch",
        max_age=timedelta(days=3),
        bootstrap_grace=timedelta(days=3),
    ),
)


def _read_json(url: str, opener: OpenUrl, token: str) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "luxar-cadence-watchdog",
            "Authorization": f"Bearer {token}",
        },
    )
    with opener(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError("GitHub returned a non-object response")
    return payload


def _parse_time(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("GitHub response omitted a timestamp")
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _age_text(age: timedelta) -> str:
    total_hours = max(0, int(age.total_seconds() // 3600))
    days, hours = divmod(total_hours, 24)
    if days:
        suffix = "day" if days == 1 else "days"
        return f"{days} {suffix}, {hours} hours"
    return f"{hours} hours"


def _matching_workflow(
    payload: dict[str, object], name: str
) -> dict[str, object] | None:
    workflows = payload.get("workflows")
    if not isinstance(workflows, list):
        raise ValueError("GitHub response omitted workflows")
    matches = [
        item
        for item in workflows
        if isinstance(item, dict) and item.get("name") == name
    ]
    if len(matches) > 1:
        raise ValueError(f"multiple workflows are named {name!r}")
    return matches[0] if matches else None


def check_cadence(
    cadence: Cadence,
    *,
    repository: str,
    now: datetime,
    token: str,
    opener: OpenUrl = urllib.request.urlopen,
    expected_present: bool | None = None,
) -> Result:
    workflows = _read_json(
        f"{API_ROOT}/repos/{repository}/actions/workflows?per_page=100", opener, token
    )
    workflow = _matching_workflow(workflows, cadence.workflow_name)
    if workflow is None:
        present = (
            (REPO_ROOT / cadence.workflow_path).is_file()
            if expected_present is None
            else expected_present
        )
        level = "CONFIG" if present else "HUMAN"
        detail = (
            "workflow is present in this checkout but not registered on the default branch"
            if present
            else "workflow is not registered on the default branch yet"
        )
        return Result(cadence, level, detail)

    state = workflow.get("state")
    if state != "active":
        return Result(cadence, "CONFIG", f"workflow state is {state!r}, not 'active'")
    workflow_id = workflow.get("id")
    if not isinstance(workflow_id, int):
        raise ValueError("GitHub response omitted the workflow id")
    runs = _read_json(
        f"{API_ROOT}/repos/{repository}/actions/workflows/{workflow_id}/runs"
        f"?{urlencode({'branch': cadence.branch, 'event': cadence.event, 'status': 'success', 'per_page': 1})}",
        opener,
        token,
    )
    workflow_runs = runs.get("workflow_runs")
    if not isinstance(workflow_runs, list):
        raise ValueError("GitHub response omitted workflow runs")
    if not workflow_runs:
        age = now - _parse_time(workflow.get("created_at"))
        if age <= cadence.bootstrap_grace:
            return Result(
                cadence,
                "HUMAN",
                "no successful run yet; "
                f"{_age_text(cadence.bootstrap_grace - age)} of bootstrap grace remains",
            )
        return Result(
            cadence,
            "STALE",
            f"no successful run after {_age_text(age)}",
        )

    latest = workflow_runs[0]
    if not isinstance(latest, dict) or latest.get("conclusion") != "success":
        raise ValueError("GitHub success query returned an invalid run")
    age = now - _parse_time(latest.get("updated_at"))
    detail = (
        f"last success was {_age_text(age)} ago; maximum {cadence.max_age.days} days"
    )
    return Result(cadence, "OK" if age <= cadence.max_age else "STALE", detail)


def check_all(
    *,
    repository: str,
    now: datetime,
    token: str,
    opener: OpenUrl = urllib.request.urlopen,
) -> list[Result]:
    results = []
    for cadence in CADENCES:
        try:
            result = check_cadence(
                cadence,
                repository=repository,
                now=now,
                token=token,
                opener=opener,
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            result = Result(
                cadence, "CONFIG", f"could not query GitHub Actions: {error}"
            )
        results.append(result)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository", default=os.environ.get("GITHUB_REPOSITORY", "royerlab/luxar")
    )
    args = parser.parse_args(argv)
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        print("[CONFIG] GitHub Actions: GITHUB_TOKEN with actions:read is required")
        return 1
    results = check_all(repository=args.repository, now=datetime.now(UTC), token=token)
    for result in results:
        print(f"[{result.level}] {result.cadence.workflow_name}: {result.detail}")
    return int(any(result.level in {"STALE", "CONFIG"} for result in results))


if __name__ == "__main__":
    sys.exit(main())
