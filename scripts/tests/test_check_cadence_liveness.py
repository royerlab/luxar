from __future__ import annotations

import importlib.util
import io
import json
import sys
import urllib.error
from datetime import UTC, datetime
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "check_cadence_liveness.py"
SPEC = importlib.util.spec_from_file_location("check_cadence_liveness", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)

NOW = datetime(2026, 9, 11, 20, tzinfo=UTC)


def _response(payload: object) -> io.BytesIO:
    return io.BytesIO(json.dumps(payload).encode())


def _opener(workflows: object, runs: object):
    def open_request(request, timeout):
        assert timeout == module.REQUEST_TIMEOUT_SECONDS
        assert request.headers["User-agent"] == "luxar-cadence-watchdog"
        if request.full_url.endswith("/actions/workflows?per_page=100"):
            return _response(workflows)
        if "/actions/workflows/42/runs?" in request.full_url:
            return _response(runs)
        raise AssertionError(request.full_url)

    return open_request


def test_recent_success_is_current() -> None:
    result = module.check_cadence(
        module.CADENCES[0],
        repository="royerlab/luxar",
        now=NOW,
        opener=_opener(
            {
                "workflows": [
                    {
                        "id": 42,
                        "name": "CUDA native cadence",
                        "state": "active",
                        "created_at": "2026-09-01T00:00:00Z",
                    }
                ]
            },
            {
                "workflow_runs": [
                    {"updated_at": "2026-09-10T03:00:00Z", "conclusion": "success"}
                ]
            },
        ),
    )

    assert result.level == "OK"
    assert "1 day" in result.detail


def test_old_success_is_stale() -> None:
    result = module.check_cadence(
        module.CADENCES[0],
        repository="royerlab/luxar",
        now=NOW,
        opener=_opener(
            {
                "workflows": [
                    {
                        "id": 42,
                        "name": "CUDA native cadence",
                        "state": "active",
                        "created_at": "2026-08-01T00:00:00Z",
                    }
                ]
            },
            {
                "workflow_runs": [
                    {"updated_at": "2026-09-01T00:00:00Z", "conclusion": "success"}
                ]
            },
        ),
    )

    assert result.level == "STALE"
    assert "10 days" in result.detail
    assert "maximum 3 days" in result.detail


def test_missing_workflow_is_bootstrap_not_failure() -> None:
    result = module.check_cadence(
        module.CADENCES[0],
        repository="royerlab/luxar",
        now=NOW,
        opener=_opener({"workflows": []}, {}),
    )

    assert result.level == "HUMAN"
    assert "not registered on the default branch yet" in result.detail


def test_new_workflow_without_success_uses_bootstrap_grace() -> None:
    result = module.check_cadence(
        module.CADENCES[0],
        repository="royerlab/luxar",
        now=NOW,
        opener=_opener(
            {
                "workflows": [
                    {
                        "id": 42,
                        "name": "CUDA native cadence",
                        "state": "active",
                        "created_at": "2026-09-10T00:00:00Z",
                    }
                ]
            },
            {"workflow_runs": []},
        ),
    )

    assert result.level == "HUMAN"
    assert "bootstrap grace" in result.detail


def test_old_workflow_without_success_fails_closed() -> None:
    result = module.check_cadence(
        module.CADENCES[0],
        repository="royerlab/luxar",
        now=NOW,
        opener=_opener(
            {
                "workflows": [
                    {
                        "id": 42,
                        "name": "CUDA native cadence",
                        "state": "active",
                        "created_at": "2026-09-01T00:00:00Z",
                    }
                ]
            },
            {"workflow_runs": []},
        ),
    )

    assert result.level == "STALE"
    assert "no successful run" in result.detail


def test_disabled_workflow_and_api_failure_are_configuration_errors() -> None:
    disabled = module.check_cadence(
        module.CADENCES[0],
        repository="royerlab/luxar",
        now=NOW,
        opener=_opener(
            {
                "workflows": [
                    {
                        "id": 42,
                        "name": "CUDA native cadence",
                        "state": "disabled_manually",
                        "created_at": "2026-09-01T00:00:00Z",
                    }
                ]
            },
            {},
        ),
    )

    def failing_opener(request, timeout):
        raise urllib.error.URLError("offline")

    failed = module.check_all(
        repository="royerlab/luxar", now=NOW, opener=failing_opener
    )

    assert disabled.level == "CONFIG"
    assert "disabled_manually" in disabled.detail
    assert failed[0].level == "CONFIG"
    assert "could not query GitHub Actions" in failed[0].detail


@pytest.mark.parametrize(
    ("levels", "expected"),
    [(["OK"], 0), (["HUMAN"], 0), (["STALE"], 1), (["CONFIG"], 1)],
)
def test_main_only_fails_for_stale_or_broken_cadences(
    monkeypatch, capsys, levels: list[str], expected: int
) -> None:
    monkeypatch.setattr(
        module,
        "check_all",
        lambda **kwargs: [
            module.Result(module.CADENCES[0], level, "detail") for level in levels
        ],
    )

    assert module.main([]) == expected
    assert f"[{levels[0]}] CUDA native cadence: detail" in capsys.readouterr().out
