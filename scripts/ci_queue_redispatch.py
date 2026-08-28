#!/usr/bin/env python3
"""Re-dispatch one CI run stranded behind a saturated obsidian queue."""

from __future__ import annotations

import argparse
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence, cast

import ci_queue_scan

WORKFLOW = "ci-queue-redispatch.yml"


@dataclass(frozen=True)
class Candidate:
    run_id: int
    queued: tuple[str, ...]


def select_candidate(result: ci_queue_scan.ScanResult) -> Candidate | None:
    """Select one wholly queued run only when complete saturation evidence exists."""
    if result.error is not None or result.truncated or not result.running:
        return None
    for run in result.runs:
        if run.queued and not run.running:
            return Candidate(run.run_id, tuple(run.queued))
    return None


def _object(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ci_queue_scan.ApiError("run payload must be an object")
    return cast(dict[str, Any], payload)


def write_api(endpoint: str, fields: Mapping[str, str] | None = None) -> None:
    command = ["gh", "api", "-X", "POST", endpoint]
    for key, value in (fields or {}).items():
        command.extend(["-f", f"{key}={value}"])
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except FileNotFoundError as error:
        raise ci_queue_scan.ApiError("gh was not found") from error
    except subprocess.TimeoutExpired as error:
        raise ci_queue_scan.ApiError("gh API request timed out") from error
    if completed.returncode != 0:
        detail = next(
            (line.strip() for line in completed.stderr.splitlines() if line.strip()),
            f"gh exited {completed.returncode}",
        )
        raise ci_queue_scan.ApiError(detail)


def hand_off_and_cancel(
    repository: str,
    ref: str,
    candidate: Candidate,
    *,
    read: Callable[[str], object] = ci_queue_scan.read_api,
    write: Callable[[str, Mapping[str, str] | None], None] = write_api,
) -> bool:
    """Start a durable recovery run before cancelling an eligible first attempt."""
    run_endpoint = f"repos/{repository}/actions/runs/{candidate.run_id}"
    run = _object(read(run_endpoint))
    if run.get("status") not in {"queued", "in_progress"}:
        return False
    if run.get("run_attempt") != 1:
        return False
    write(
        f"repos/{repository}/actions/workflows/{WORKFLOW}/dispatches",
        {"ref": ref, "inputs[target_run_id]": str(candidate.run_id)},
    )
    write(f"{run_endpoint}/cancel", None)
    return True


def finish_redispatch(
    repository: str,
    run_id: int,
    *,
    read: Callable[[str], object] = ci_queue_scan.read_api,
    write: Callable[[str, Mapping[str, str] | None], None] = write_api,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Wait for the target cancellation, then rerun the complete original workflow."""
    run_endpoint = f"repos/{repository}/actions/runs/{run_id}"
    for _ in range(24):
        try:
            run = _object(read(run_endpoint))
        except ci_queue_scan.ApiError:
            sleep(5)
            continue
        if run.get("status") == "completed":
            if run.get("conclusion") != "cancelled" or run.get("run_attempt") != 1:
                return False
            write(f"{run_endpoint}/rerun", None)
            return True
        sleep(5)
    return False


def scan_and_handoff(
    repository: str, *, ref: str, queued_before: float, max_runs: int
) -> int:
    result = ci_queue_scan.scan_repository(
        repository,
        statuses=["in_progress", "queued"],
        queued_before=queued_before,
        max_runs=max_runs,
    )
    if result.error is not None:
        print(f"queue scan unreadable ({result.error['scope']}); leaving runs alone")
        return 0
    if result.truncated:
        print("queue scan truncated; leaving runs alone")
        return 0
    candidate = select_candidate(result)
    if candidate is None:
        print("no wholly queued obsidian run behind active obsidian work")
        return 0
    names = ", ".join(candidate.queued)
    print(f"run {candidate.run_id} has aged queued obsidian jobs ({names})")
    try:
        changed = hand_off_and_cancel(repository, ref, candidate)
    except ci_queue_scan.ApiError as error:
        print(f"redispatch handoff failed safely: {error}")
        return 0
    if changed:
        print(f"handed off recovery and cancelled run {candidate.run_id}")
    else:
        print(f"run {candidate.run_id} is no longer eligible for automatic redispatch")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    scan = subparsers.add_parser("scan")
    scan.add_argument("--repository", required=True)
    scan.add_argument("--ref", required=True)
    scan.add_argument("--queued-before", type=float, required=True)
    scan.add_argument("--max-runs", type=int, default=100)
    finish = subparsers.add_parser("finish")
    finish.add_argument("--repository", required=True)
    finish.add_argument("--run-id", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "scan":
        return scan_and_handoff(
            args.repository,
            ref=args.ref,
            queued_before=args.queued_before,
            max_runs=args.max_runs,
        )
    try:
        changed = finish_redispatch(args.repository, args.run_id)
    except ci_queue_scan.ApiError as error:
        print(f"redispatch failed safely: {error}")
        return 0
    if changed:
        print(f"re-dispatched run {args.run_id}")
    else:
        print(f"run {args.run_id} did not reach the eligible cancelled state")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
