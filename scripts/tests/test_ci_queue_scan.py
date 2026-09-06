from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "ci_queue_scan.py"
SPEC = importlib.util.spec_from_file_location("ci_queue_scan", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
ci_queue_scan = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ci_queue_scan
SPEC.loader.exec_module(ci_queue_scan)


def _job(
    name: str,
    status: str,
    *,
    label: str = "obsidian",
    created_at: str = "2026-08-27T00:00:00Z",
) -> dict[str, object]:
    return {
        "name": name,
        "status": status,
        "labels": [label],
        "created_at": created_at,
    }


def test_classify_jobs_partitions_only_obsidian_work() -> None:
    result = ci_queue_scan.classify_jobs(
        {
            "jobs": [
                _job("old queued", "queued"),
                _job(
                    "young queued",
                    "queued",
                    created_at="2026-08-27T00:09:59Z",
                ),
                _job("running", "in_progress"),
                _job("hosted", "queued", label="ubuntu-latest"),
                _job("done", "completed"),
            ]
        },
        queued_before=ci_queue_scan.parse_timestamp("2026-08-27T00:05:00Z"),
    )

    assert result.visible == ["old queued", "young queued", "running", "done"]
    assert result.queued == ["old queued"]
    assert result.running == ["running"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"jobs": None},
        {"jobs": [{"name": "queued", "status": "queued", "labels": None}]},
        {
            "jobs": [
                {
                    "name": "queued",
                    "status": "queued",
                    "labels": ["obsidian"],
                    "created_at": "not-a-date",
                }
            ]
        },
    ],
)
def test_classify_jobs_rejects_unreadable_payloads(payload: object) -> None:
    with pytest.raises(ValueError):
        ci_queue_scan.classify_jobs(payload, queued_before=0)


def test_scan_repository_filters_runs_and_stops_at_queue_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    responses = {
        "repos/royerlab/luxar/actions/runs?status=in_progress&per_page=100": {
            "workflow_runs": [
                {"id": 10, "created_at": "2026-08-27T00:09:59Z"},
            ]
        },
        "repos/royerlab/luxar/actions/runs/10/jobs?per_page=100": {
            "jobs": [_job("running", "in_progress"), _job("queued one", "queued")]
        },
        "repos/royerlab/luxar/actions/runs?status=queued&per_page=100": {
            "workflow_runs": [
                {"id": 20, "created_at": "2026-08-27T00:01:00Z"},
                {"id": 21, "created_at": "2026-08-27T00:09:00Z"},
            ]
        },
        "repos/royerlab/luxar/actions/runs/20/jobs?per_page=100": {
            "jobs": [_job("queued two", "queued")]
        },
    }

    def fake_api(endpoint: str) -> object:
        calls.append(endpoint)
        return responses[endpoint]

    monkeypatch.setattr(ci_queue_scan, "read_api", fake_api)
    result = ci_queue_scan.scan_repository(
        "royerlab/luxar",
        statuses=["in_progress", "queued"],
        queued_before=ci_queue_scan.parse_timestamp("2026-08-27T00:05:00Z"),
        max_runs=10,
        stop_after_queued=2,
    )

    assert result.queued == ["queued one", "queued two"]
    assert result.running == ["running"]
    assert result.runs == [
        ci_queue_scan.RunScan(10, queued=["queued one"], running=["running"]),
        ci_queue_scan.RunScan(20, queued=["queued two"]),
    ]
    assert result.scanned_runs == 2
    assert result.stopped is True
    assert not any("runs/21/jobs" in call for call in calls)


def test_scan_repository_reports_partial_results_on_jobs_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            {
                "workflow_runs": [
                    {"id": 10, "created_at": "2026-08-27T00:00:00Z"},
                    {"id": 11, "created_at": "2026-08-27T00:00:00Z"},
                ]
            },
            {"jobs": [_job("queued", "queued")]},
            ci_queue_scan.ApiError("jobs unavailable"),
        ]
    )

    def fake_api(endpoint: str) -> object:
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(ci_queue_scan, "read_api", fake_api)
    result = ci_queue_scan.scan_repository(
        "royerlab/luxar",
        statuses=["in_progress"],
        queued_before=None,
        max_runs=10,
    )

    assert result.queued == ["queued"]
    assert result.error == {"scope": "jobs", "detail": "jobs unavailable"}


def test_scan_repository_marks_truncation_without_querying_extra_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_api(endpoint: str) -> object:
        calls.append(endpoint)
        if "?status=" in endpoint:
            return {
                "workflow_runs": [
                    {"id": run_id, "created_at": "2026-08-27T00:00:00Z"}
                    for run_id in range(3)
                ]
            }
        return {"jobs": []}

    monkeypatch.setattr(ci_queue_scan, "read_api", fake_api)
    result = ci_queue_scan.scan_repository(
        "royerlab/luxar",
        statuses=["queued"],
        queued_before=ci_queue_scan.parse_timestamp("2026-08-27T00:05:00Z"),
        max_runs=2,
    )

    assert result.scanned_runs == 2
    assert result.truncated is True
    assert len([call for call in calls if "/jobs?" in call]) == 2


def test_scan_repository_marks_full_api_page_as_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_api(endpoint: str) -> object:
        if "?status=" in endpoint:
            return {
                "workflow_runs": [
                    {"id": run_id, "created_at": "2026-08-27T00:00:00Z"}
                    for run_id in range(100)
                ]
            }
        return {"jobs": []}

    monkeypatch.setattr(ci_queue_scan, "read_api", fake_api)
    result = ci_queue_scan.scan_repository(
        "royerlab/luxar",
        statuses=["queued"],
        queued_before=ci_queue_scan.parse_timestamp("2026-08-27T00:05:00Z"),
        max_runs=100,
    )

    assert result.scanned_runs == 100
    assert result.truncated is True


def test_scan_repository_short_circuits_on_the_first_running_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_api(endpoint: str) -> object:
        calls.append(endpoint)
        if "?status=queued" in endpoint:
            return {
                "workflow_runs": [
                    {"id": run_id, "created_at": "2026-08-27T00:00:00Z"}
                    for run_id in (30, 31)
                ]
            }
        if "?status=in_progress" in endpoint:
            return {"workflow_runs": []}
        if "runs/30/jobs" in endpoint:
            return {"jobs": [_job("running", "in_progress")]}
        return {"jobs": []}

    monkeypatch.setattr(ci_queue_scan, "read_api", fake_api)
    result = ci_queue_scan.scan_repository(
        "royerlab/luxar",
        statuses=["queued", "in_progress"],
        queued_before=None,
        max_runs=100,
        stop_after_running=1,
    )

    assert result.running == ["running"]
    assert result.stopped is True
    assert not any("?status=in_progress" in call for call in calls)
    assert len([call for call in calls if "/jobs?" in call]) == 1


def test_scan_repository_scans_every_status_when_the_first_hits_its_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_api(endpoint: str) -> object:
        calls.append(endpoint)
        if "?status=queued" in endpoint:
            return {
                "workflow_runs": [
                    {"id": 40 + offset, "created_at": "2026-08-27T00:00:00Z"}
                    for offset in range(5)
                ]
            }
        if "?status=in_progress" in endpoint:
            return {"workflow_runs": [{"id": 50, "created_at": "2026-08-27T00:00:00Z"}]}
        if "runs/50/jobs" in endpoint:
            return {"jobs": [_job("running", "in_progress")]}
        return {"jobs": []}

    monkeypatch.setattr(ci_queue_scan, "read_api", fake_api)
    result = ci_queue_scan.scan_repository(
        "royerlab/luxar",
        statuses=["queued", "in_progress"],
        queued_before=None,
        max_runs=4,
        stop_after_running=1,
    )

    assert result.truncated is True
    assert any("?status=in_progress" in call for call in calls)
    assert result.running == ["running"]
    assert result.scanned_runs == 4


def test_scan_repository_never_caps_a_status_below_the_remaining_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_api(endpoint: str) -> object:
        if "?status=queued" in endpoint:
            return {
                "workflow_runs": [
                    {"id": 100 + offset, "created_at": "2026-08-27T00:00:00Z"}
                    for offset in range(60)
                ]
            }
        if "?status=in_progress" in endpoint:
            return {"workflow_runs": []}
        if "runs/154/jobs" in endpoint:
            return {"jobs": [_job("running", "in_progress")]}
        return {"jobs": []}

    monkeypatch.setattr(ci_queue_scan, "read_api", fake_api)
    result = ci_queue_scan.scan_repository(
        "royerlab/luxar",
        statuses=["queued", "in_progress"],
        queued_before=None,
        max_runs=100,
        stop_after_running=1,
    )

    assert result.running == ["running"]
    assert result.stopped is True
    assert result.scanned_runs == 55


def test_scan_repository_reads_every_status_when_the_budget_is_below_the_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_api(endpoint: str) -> object:
        calls.append(endpoint)
        if "?status=queued" in endpoint:
            return {"workflow_runs": [{"id": 60, "created_at": "2026-08-27T00:00:00Z"}]}
        if "?status=in_progress" in endpoint:
            return {"workflow_runs": [{"id": 61, "created_at": "2026-08-27T00:00:00Z"}]}
        if "runs/60/jobs" in endpoint:
            return {"jobs": [_job("running in the queued run", "in_progress")]}
        return {"jobs": [_job("running in the in_progress run", "in_progress")]}

    monkeypatch.setattr(ci_queue_scan, "read_api", fake_api)
    result = ci_queue_scan.scan_repository(
        "royerlab/luxar",
        statuses=["queued", "in_progress"],
        queued_before=None,
        max_runs=1,
    )

    assert result.running == [
        "running in the queued run",
        "running in the in_progress run",
    ]
    assert len([call for call in calls if "/jobs?" in call]) == 2


def test_cli_classify_emits_structured_error_for_malformed_json(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sys.stdin.read", lambda: "not-json")

    assert ci_queue_scan.main(["classify"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["error"]["scope"] == "jobs"


def test_read_api_reports_gh_failure_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 1, stdout="", stderr="rate limited\n"
        ),
    )

    with pytest.raises(ci_queue_scan.ApiError, match="rate limited"):
        ci_queue_scan.read_api("repos/royerlab/luxar/actions/runs")


def test_read_api_times_out_instead_of_hanging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timeout(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert kwargs["timeout"] == 30
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timeout)

    with pytest.raises(ci_queue_scan.ApiError, match="timed out"):
        ci_queue_scan.read_api("repos/royerlab/luxar/actions/runs")
