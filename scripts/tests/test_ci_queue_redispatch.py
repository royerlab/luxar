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


def test_select_candidates_require_complete_saturation_evidence() -> None:
    candidates = ci_queue_redispatch.select_candidates(
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
    assert candidates == (ci_queue_redispatch.Candidate(10, ("python-tests (3.12)",)),)

    unsafe_results = [
        ci_queue_scan.ScanResult(
            running=["other run"],
            error={"scope": "jobs", "detail": "boom"},
            runs=[ci_queue_scan.RunScan(10, queued=["python-tests (3.12)"])],
        ),
        ci_queue_scan.ScanResult(
            running=["other run"],
            truncated=True,
            runs=[ci_queue_scan.RunScan(10, queued=["python-tests (3.12)"])],
        ),
        ci_queue_scan.ScanResult(
            runs=[ci_queue_scan.RunScan(10, queued=["python-tests (3.12)"])]
        ),
        ci_queue_scan.ScanResult(
            running=["same run"],
            runs=[
                ci_queue_scan.RunScan(
                    10,
                    queued=["python-tests (3.12)"],
                    running=["same run"],
                )
            ],
        ),
        ci_queue_scan.ScanResult(
            running=["other run"],
            runs=[ci_queue_scan.RunScan(10, queued=["python-tests (3.14)"])],
        ),
    ]
    assert all(
        ci_queue_redispatch.select_candidates(result) == () for result in unsafe_results
    )


def test_scan_skips_ineligible_candidate_and_hands_off_next(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        ci_queue_scan,
        "scan_repository",
        lambda *args, **kwargs: ci_queue_scan.ScanResult(
            running=["live obsidian work"],
            runs=[
                ci_queue_scan.RunScan(99, queued=["typescript-tests"]),
                ci_queue_scan.RunScan(42, queued=["python-tests (3.12)"]),
                ci_queue_scan.RunScan(43, queued=["typescript-tests"]),
                ci_queue_scan.RunScan(7, running=["python-tests (3.12)"]),
            ],
        ),
    )
    reads: list[int] = []
    writes: list[tuple[str, object]] = []

    def read(endpoint: str) -> object:
        run_id = int(endpoint.rsplit("/", 1)[-1])
        reads.append(run_id)
        return {"status": "in_progress", "run_attempt": 2 if run_id == 99 else 1}

    assert (
        ci_queue_redispatch.scan_and_handoff(
            "royerlab/luxar",
            ref="dev",
            queued_before=1_800_000_000,
            max_runs=100,
            read=read,
            write=lambda endpoint, fields: writes.append((endpoint, fields)),
        )
        == 0
    )

    assert reads == [99, 42]
    assert writes == [
        (
            "repos/royerlab/luxar/actions/workflows/ci-queue-redispatch.yml/dispatches",
            {"ref": "dev", "inputs[target_run_id]": "42"},
        ),
        ("repos/royerlab/luxar/actions/runs/42/cancel", None),
    ]
    output = capsys.readouterr().out
    assert "run 99 is no longer eligible for automatic redispatch" in output
    assert "handed off recovery and cancelled run 42" in output


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


def test_handoff_distinguishes_cancel_rejection_after_dispatch() -> None:
    writes: list[tuple[str, object]] = []

    def write(endpoint: str, fields: object) -> None:
        writes.append((endpoint, fields))
        if endpoint.endswith("/cancel"):
            raise ci_queue_scan.ApiError("conflict")

    with pytest.raises(ci_queue_redispatch.CancellationRejected, match="conflict"):
        ci_queue_redispatch.hand_off_and_cancel(
            "royerlab/luxar",
            "dev",
            ci_queue_redispatch.Candidate(42, ("typescript-tests",)),
            read=lambda endpoint: {"status": "in_progress", "run_attempt": 1},
            write=write,
        )

    assert writes == [
        (
            "repos/royerlab/luxar/actions/workflows/ci-queue-redispatch.yml/dispatches",
            {"ref": "dev", "inputs[target_run_id]": "42"},
        ),
        ("repos/royerlab/luxar/actions/runs/42/cancel", None),
    ]


def test_scan_reports_post_dispatch_cancel_rejection(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def read(endpoint: str) -> object:
        if endpoint.endswith("actions/runs?status=in_progress&per_page=100"):
            return {
                "workflow_runs": [
                    {
                        "id": 7,
                        "name": "CI",
                        "status": "in_progress",
                        "created_at": "2026-08-28T20:00:00Z",
                    },
                    {
                        "id": 42,
                        "name": "CI",
                        "status": "in_progress",
                        "created_at": "2026-08-28T20:00:00Z",
                    },
                ]
            }
        if endpoint.endswith("actions/runs?status=queued&per_page=100"):
            return {"workflow_runs": []}
        if endpoint.endswith("actions/runs/7/jobs?per_page=100"):
            return {
                "jobs": [
                    {
                        "name": "python-tests (3.13)",
                        "status": "in_progress",
                        "labels": ["obsidian"],
                    }
                ]
            }
        if endpoint.endswith("actions/runs/42/jobs?per_page=100"):
            return {
                "jobs": [
                    {
                        "name": "python-tests (3.12)",
                        "status": "queued",
                        "created_at": "2026-08-28T20:00:00Z",
                        "labels": ["obsidian"],
                    }
                ]
            }
        if endpoint.endswith("actions/runs/42"):
            return {"status": "in_progress", "run_attempt": 1}
        raise AssertionError(endpoint)

    writes: list[str] = []

    def write(endpoint: str, fields: object) -> None:
        writes.append(endpoint)
        if endpoint.endswith("/cancel"):
            raise ci_queue_scan.ApiError("conflict")

    monkeypatch.setattr(ci_queue_scan, "read_api", read)
    assert (
        ci_queue_redispatch.scan_and_handoff(
            "royerlab/luxar",
            ref="dev",
            queued_before=1_800_000_000,
            max_runs=100,
            read=read,
            write=write,
        )
        == 0
    )

    assert writes[-1].endswith("/cancel")
    output = capsys.readouterr().out
    assert "recovery run dispatched but cancellation was rejected: conflict" in output
    assert "failed safely" not in output


def test_scan_leaves_runs_alone_when_repository_read_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def read(endpoint: str) -> object:
        raise ci_queue_scan.ApiError("boom")

    monkeypatch.setattr(ci_queue_scan, "read_api", read)
    assert (
        ci_queue_redispatch.scan_and_handoff(
            "royerlab/luxar",
            ref="dev",
            queued_before=1_800_000_000,
            max_runs=100,
            write=lambda endpoint, fields: pytest.fail("must not write"),
        )
        == 0
    )
    assert "queue scan unreadable (runs); leaving runs alone" in capsys.readouterr().out


def test_scan_leaves_runs_alone_when_result_is_truncated(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        ci_queue_scan,
        "read_api",
        lambda endpoint: {"workflow_runs": [{"id": run_id} for run_id in range(100)]},
    )
    assert (
        ci_queue_redispatch.scan_and_handoff(
            "royerlab/luxar",
            ref="dev",
            queued_before=1_800_000_000,
            max_runs=0,
            write=lambda endpoint, fields: pytest.fail("must not write"),
        )
        == 0
    )
    assert "queue scan truncated; leaving runs alone" in capsys.readouterr().out


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


def test_finish_refuses_run_that_completed_successfully() -> None:
    writes: list[tuple[str, object]] = []

    changed = ci_queue_redispatch.finish_redispatch(
        "royerlab/luxar",
        42,
        read=lambda endpoint: {
            "status": "completed",
            "conclusion": "success",
            "run_attempt": 1,
        },
        write=lambda endpoint, fields: writes.append((endpoint, fields)),
        sleep=lambda seconds: None,
    )

    assert changed is False
    assert writes == []


def test_finish_refuses_run_taken_over_by_another_repair() -> None:
    writes: list[tuple[str, object]] = []

    changed = ci_queue_redispatch.finish_redispatch(
        "royerlab/luxar",
        42,
        read=lambda endpoint: {"status": "in_progress", "run_attempt": 2},
        write=lambda endpoint, fields: writes.append((endpoint, fields)),
        sleep=lambda seconds: None,
    )

    assert changed is False
    assert writes == []


def test_finish_returns_false_when_poll_budget_expires() -> None:
    now = 0.0
    writes: list[tuple[str, object]] = []

    def clock() -> float:
        return now

    def sleep(seconds: float) -> None:
        nonlocal now
        now += seconds

    changed = ci_queue_redispatch.finish_redispatch(
        "royerlab/luxar",
        42,
        read=lambda endpoint: {"status": "in_progress", "run_attempt": 1},
        write=lambda endpoint, fields: writes.append((endpoint, fields)),
        sleep=sleep,
        clock=clock,
        timeout_seconds=20,
    )

    assert changed is False
    assert now == 20
    assert writes == []


def test_finish_command_fails_when_first_attempt_may_be_stranded(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(ci_queue_redispatch, "finish_redispatch", lambda *args: False)
    monkeypatch.setattr(
        ci_queue_scan,
        "read_api",
        lambda endpoint: {"status": "in_progress", "run_attempt": 1},
    )

    assert (
        ci_queue_redispatch.main(
            ["finish", "--repository", "royerlab/luxar", "--run-id", "42"]
        )
        == 1
    )
    assert "::error::" in capsys.readouterr().out
