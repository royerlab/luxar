#!/usr/bin/env python3
"""Classify obsidian jobs and scan repository workflow runs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Sequence, cast


class ApiError(RuntimeError):
    """GitHub API data could not be read."""


@dataclass
class RunScan:
    run_id: int
    queued: list[str] = field(default_factory=list)
    running: list[str] = field(default_factory=list)


@dataclass
class ScanResult:
    visible: list[str] = field(default_factory=list)
    queued: list[str] = field(default_factory=list)
    running: list[str] = field(default_factory=list)
    runs: list[RunScan] = field(default_factory=list)
    scanned_runs: int = 0
    truncated: bool = False
    stopped: bool = False
    error: dict[str, str] | None = None

    def extend(self, other: ScanResult) -> None:
        self.visible.extend(other.visible)
        self.queued.extend(other.queued)
        self.running.extend(other.running)


def parse_timestamp(value: str) -> float:
    """Parse the ISO-8601 timestamps returned by the Actions API."""
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _object_list(payload: object, key: str) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get(key), list):
        raise ValueError(f"payload must contain a {key!r} list")
    values = cast(list[object], payload[key])
    if not all(isinstance(value, dict) for value in values):
        raise ValueError(f"payload {key!r} entries must be objects")
    return cast(list[dict[str, Any]], values)


def _job_state(
    job: dict[str, Any], queued_before: float | None
) -> tuple[str, str, bool] | None:
    labels = job.get("labels")
    if not isinstance(labels, list) or not all(
        isinstance(label, str) for label in labels
    ):
        raise ValueError("job labels must be a string list")
    if "obsidian" not in labels:
        return None
    name = job.get("name")
    status = job.get("status")
    if not isinstance(name, str) or not isinstance(status, str):
        raise ValueError("job name and status must be strings")
    queued = status == "queued"
    if queued and queued_before is not None:
        created_at = job.get("created_at")
        if not isinstance(created_at, str):
            raise ValueError("queued job created_at must be a string")
        try:
            queued = parse_timestamp(created_at) <= queued_before
        except ValueError as error:
            raise ValueError("queued job created_at is not ISO-8601") from error
    return name, status, queued


def classify_jobs(payload: object, *, queued_before: float | None = None) -> ScanResult:
    """Partition obsidian-labelled jobs into visible, queued, and running names."""
    result = ScanResult()
    for job in _object_list(payload, "jobs"):
        state = _job_state(job, queued_before)
        if state is None:
            continue
        name, status, queued = state
        result.visible.append(name)
        if status == "in_progress":
            result.running.append(name)
        if queued:
            result.queued.append(name)
    return result


def read_api(endpoint: str) -> object:
    """Read one GitHub API endpoint through the authenticated gh CLI."""
    try:
        completed = subprocess.run(
            ["gh", "api", endpoint],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except FileNotFoundError as error:
        raise ApiError("gh was not found") from error
    except subprocess.TimeoutExpired as error:
        raise ApiError("gh API request timed out") from error
    if completed.returncode != 0:
        detail = next(
            (line.strip() for line in completed.stderr.splitlines() if line.strip()),
            f"gh exited {completed.returncode}",
        )
        raise ApiError(detail)
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ApiError("gh returned invalid JSON") from error


def _eligible_run_id(
    run: dict[str, Any],
    *,
    status: str,
    queued_before: float | None,
    exclude_run: int | None,
) -> int | None:
    run_id = run.get("id")
    if not isinstance(run_id, int):
        raise ValueError("run id must be an integer")
    if run_id == exclude_run:
        return None
    if status != "queued" or queued_before is None:
        return run_id
    created_at = run.get("created_at")
    if not isinstance(created_at, str):
        raise ValueError("queued run created_at must be a string")
    try:
        return run_id if parse_timestamp(created_at) <= queued_before else None
    except ValueError as error:
        raise ValueError("queued run created_at is not ISO-8601") from error


def _limit_reached(
    result: ScanResult,
    *,
    stop_after_queued: int | None,
    stop_after_running: int | None,
) -> bool:
    return (
        stop_after_queued is not None and len(result.queued) >= stop_after_queued
    ) or (stop_after_running is not None and len(result.running) >= stop_after_running)


def scan_repository(
    repository: str,
    *,
    statuses: Sequence[str],
    queued_before: float | None,
    max_runs: int,
    exclude_run: int | None = None,
    stop_after_queued: int | None = None,
    stop_after_running: int | None = None,
) -> ScanResult:
    """Fan out from workflow runs to jobs and aggregate obsidian classifications."""
    result = ScanResult()
    # Split the total budget per status: which run-level status hosts the live
    # job is unknowable in advance, so no pass may consume another's share.
    status_budget = max(1, max_runs // (len(statuses) or 1))
    for status in statuses:
        endpoint = f"repos/{repository}/actions/runs?status={status}&per_page=100"
        try:
            runs_payload = read_api(endpoint)
            runs = _object_list(runs_payload, "workflow_runs")
            result.truncated = result.truncated or len(runs) == 100
        except (ApiError, ValueError) as error:
            result.error = {"scope": "runs", "detail": str(error)}
            return result
        scanned_for_status = 0
        for run in runs:
            try:
                run_id = _eligible_run_id(
                    run,
                    status=status,
                    queued_before=queued_before,
                    exclude_run=exclude_run,
                )
            except ValueError as error:
                result.error = {"scope": "runs", "detail": str(error)}
                return result
            if run_id is None:
                continue
            if scanned_for_status >= status_budget:
                result.truncated = True
                break
            result.scanned_runs += 1
            scanned_for_status += 1
            try:
                jobs_payload = read_api(
                    f"repos/{repository}/actions/runs/{run_id}/jobs?per_page=100"
                )
                classified = classify_jobs(jobs_payload, queued_before=queued_before)
                result.extend(classified)
                if classified.queued or classified.running:
                    result.runs.append(
                        RunScan(
                            run_id=run_id,
                            queued=classified.queued,
                            running=classified.running,
                        )
                    )
            except (ApiError, ValueError) as error:
                result.error = {"scope": "jobs", "detail": str(error)}
                return result
            if _limit_reached(
                result,
                stop_after_queued=stop_after_queued,
                stop_after_running=stop_after_running,
            ):
                result.stopped = True
                return result
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    classify = subparsers.add_parser("classify")
    classify.add_argument("--queued-before", type=float)
    scan = subparsers.add_parser("scan")
    scan.add_argument("--repository", required=True)
    scan.add_argument("--status", action="append", dest="statuses", required=True)
    scan.add_argument("--queued-before", type=float)
    scan.add_argument("--max-runs", type=int, required=True)
    scan.add_argument("--exclude-run", type=int)
    scan.add_argument("--stop-after-queued", type=int)
    scan.add_argument("--stop-after-running", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "classify":
        try:
            payload = json.loads(sys.stdin.read())
            result = classify_jobs(payload, queued_before=args.queued_before)
        except (json.JSONDecodeError, ValueError) as error:
            result = ScanResult(error={"scope": "jobs", "detail": str(error)})
    else:
        result = scan_repository(
            args.repository,
            statuses=args.statuses,
            queued_before=args.queued_before,
            max_runs=args.max_runs,
            exclude_run=args.exclude_run,
            stop_after_queued=args.stop_after_queued,
            stop_after_running=args.stop_after_running,
        )
    print(json.dumps(asdict(result), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
