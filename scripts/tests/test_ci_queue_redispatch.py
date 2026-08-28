from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[1]
sys.path.insert(0, str(SCRIPTS))
SCRIPT_PATH = SCRIPTS / "ci_queue_redispatch.py"
SPEC = importlib.util.spec_from_file_location("ci_queue_redispatch", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
ci_queue_redispatch = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ci_queue_redispatch
SPEC.loader.exec_module(ci_queue_redispatch)
ci_queue_scan = ci_queue_redispatch.ci_queue_scan


def test_select_candidate_requires_complete_saturation_evidence() -> None:
    candidate = ci_queue_redispatch.select_candidate(
        ci_queue_scan.ScanResult(
            running=["other run"],
            runs=[
                ci_queue_scan.RunScan(10, queued=["python-tests (3.12)"]),
                ci_queue_scan.RunScan(
                    11,
                    queued=["python-tests (3.14)"],
                    running=["python-tests (3.12)"],
                ),
            ],
        )
    )
    assert candidate == ci_queue_redispatch.Candidate(10, ("python-tests (3.12)",))

    unsafe_results = [
        ci_queue_scan.ScanResult(
            running=["other run"],
            truncated=True,
            runs=[ci_queue_scan.RunScan(10, queued=["queued"])],
        ),
        ci_queue_scan.ScanResult(runs=[ci_queue_scan.RunScan(10, queued=["queued"])]),
        ci_queue_scan.ScanResult(
            running=["same run"],
            runs=[ci_queue_scan.RunScan(10, queued=["queued"], running=["same run"])],
        ),
        ci_queue_scan.ScanResult(
            running=["other run"],
            runs=[ci_queue_scan.RunScan(10, queued=["python-tests (3.14)"])],
        ),
    ]
    assert all(
        ci_queue_redispatch.select_candidate(result) is None
        for result in unsafe_results
    )


def test_handoff_precedes_cancellation() -> None:
    writes: list[tuple[str, object]] = []

    changed = ci_queue_redispatch.hand_off_and_cancel(
        "royerlab/luxar",
        "dev",
        ci_queue_redispatch.Candidate(42, ("typescript-tests",)),
        read=lambda endpoint: {"status": "in_progress", "run_attempt": 1},
        write=lambda endpoint, fields: writes.append((endpoint, fields)),
    )

    assert changed is True
    assert writes == [
        (
            "repos/royerlab/luxar/actions/workflows/ci-queue-redispatch.yml/dispatches",
            {"ref": "dev", "inputs[target_run_id]": "42"},
        ),
        ("repos/royerlab/luxar/actions/runs/42/cancel", None),
    ]


@pytest.mark.parametrize(
    "run",
    [
        {"status": "completed", "conclusion": "success", "run_attempt": 1},
        {"status": "in_progress", "run_attempt": 2},
    ],
)
def test_handoff_does_not_loop_or_touch_ineligible_runs(run: object) -> None:
    writes: list[tuple[str, object]] = []

    changed = ci_queue_redispatch.hand_off_and_cancel(
        "royerlab/luxar",
        "dev",
        ci_queue_redispatch.Candidate(42, ("typescript-tests",)),
        read=lambda endpoint: run,
        write=lambda endpoint, fields: writes.append((endpoint, fields)),
    )

    assert changed is False
    assert writes == []


def test_finish_waits_through_api_error_then_reruns_cancelled_attempt() -> None:
    reads = iter(
        [
            ci_queue_scan.ApiError("temporary"),
            {"status": "in_progress", "run_attempt": 1},
            {"status": "completed", "conclusion": "cancelled", "run_attempt": 1},
        ]
    )
    writes: list[tuple[str, object]] = []

    def read(endpoint: str) -> object:
        value = next(reads)
        if isinstance(value, Exception):
            raise value
        return value

    changed = ci_queue_redispatch.finish_redispatch(
        "royerlab/luxar",
        42,
        read=read,
        write=lambda endpoint, fields: writes.append((endpoint, fields)),
        sleep=lambda seconds: None,
    )

    assert changed is True
    assert writes == [("repos/royerlab/luxar/actions/runs/42/rerun", None)]


def test_finish_retries_rejected_rerun_request() -> None:
    attempts = 0

    def write(endpoint: str, fields: object) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ci_queue_scan.ApiError("not settled")

    changed = ci_queue_redispatch.finish_redispatch(
        "royerlab/luxar",
        42,
        read=lambda endpoint: {
            "status": "completed",
            "conclusion": "cancelled",
            "run_attempt": 1,
        },
        write=write,
        sleep=lambda seconds: None,
    )

    assert changed is True
    assert attempts == 2
