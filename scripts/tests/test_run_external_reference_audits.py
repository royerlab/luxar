from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import yaml

SCRIPT = Path(__file__).parents[1] / "run_external_reference_audits.py"
SPEC = importlib.util.spec_from_file_location("run_external_reference_audits", SCRIPT)
assert SPEC and SPEC.loader
audit_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit_module
SPEC.loader.exec_module(audit_module)


def test_all_external_reference_audits_are_declared() -> None:
    assert [audit.name for audit in audit_module.AUDITS] == [
        "Documentation links",
        "Demo click-throughs",
        "Zenodo manifest pins",
    ]
    assert audit_module.AUDITS[2].required_env == ("ZENODO_TOKEN",)


def test_demo_levels_are_normalized_to_the_worst_finding(monkeypatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, "[OK] host-a\n[HUMAN] host-b\n[STALE] host-c\n", ""
        ),
    )

    result = audit_module.run_audit(audit_module.AUDITS[1], env={})

    assert result.level is audit_module.Level.WARNING
    assert result.detail == "completed with findings"


def test_demo_audit_cannot_silently_report_no_destinations(monkeypatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "", ""),
    )

    result = audit_module.run_audit(audit_module.AUDITS[1], env={})

    assert result.level is audit_module.Level.ERROR


def test_missing_configuration_is_an_error_without_running(monkeypatch) -> None:
    def unexpected_run(*args, **kwargs):
        raise AssertionError("command must not run without its required secret")

    monkeypatch.setattr(subprocess, "run", unexpected_run)

    result = audit_module.run_audit(audit_module.AUDITS[2], env={})

    assert result.level is audit_module.Level.ERROR
    assert result.detail == "missing required environment: ZENODO_TOKEN"


def test_secret_is_only_exposed_to_the_audit_that_requires_it(monkeypatch) -> None:
    seen_env = {}

    def run(*args, **kwargs):
        seen_env.update(kwargs["env"])
        return subprocess.CompletedProcess(args[0], 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)

    audit_module.run_audit(
        audit_module.AUDITS[0], env={"PATH": "/bin", "ZENODO_TOKEN": "secret"}
    )

    assert seen_env == {"PATH": "/bin"}


def test_nonzero_audit_is_reported_without_stopping_later_audits(monkeypatch) -> None:
    completed = iter(
        (
            (2, "audit output"),
            (0, "[OK] host-a"),
            (1, "audit output"),
        )
    )

    def run(*args, **kwargs):
        returncode, output = next(completed)
        return subprocess.CompletedProcess(args[0], returncode, output, "")

    monkeypatch.setattr(subprocess, "run", run)
    audits = tuple(
        audit_module.Audit(audit.name, audit.command, parse_levels=audit.parse_levels)
        for audit in audit_module.AUDITS
    )

    results = [audit_module.run_audit(audit, env={}) for audit in audits]

    assert [result.level for result in results] == [
        audit_module.Level.WARNING,
        audit_module.Level.PASS,
        audit_module.Level.WARNING,
    ]


def test_main_writes_the_same_visible_summary_and_always_exits_zero(
    monkeypatch, tmp_path, capsys
) -> None:
    results = [
        audit_module.Result(audit, audit_module.Level.WARNING, "exited 1", "broken")
        for audit in audit_module.AUDITS
    ]
    monkeypatch.setattr(audit_module, "run_audit", lambda audit: results.pop(0))
    summary_path = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_path))

    assert audit_module.main() == 0

    stdout = capsys.readouterr().out
    assert summary_path.read_text() == stdout
    assert stdout.count("**WARNING**") == 3
    assert "report-only and never gate merges" in stdout


def test_summary_escapes_and_bounds_command_output() -> None:
    dangerous = "x" * (audit_module.MAX_SUMMARY_OUTPUT_CHARS + 1) + "</pre>"
    result = audit_module.Result(
        audit_module.AUDITS[0], audit_module.Level.WARNING, "finding", dangerous
    )

    summary = audit_module.render_summary([result])

    assert "earlier characters omitted" in summary
    assert "&lt;/pre&gt;" in summary
    assert dangerous not in summary


def test_weekly_workflow_is_manual_read_only_and_uses_the_aggregator() -> None:
    workflow_path = (
        SCRIPT.parents[1] / ".github/workflows/external-reference-audits.yml"
    )
    workflow = yaml.load(workflow_path.read_text(), Loader=yaml.BaseLoader)

    assert workflow["on"]["workflow_dispatch"] == ""
    assert workflow["on"]["schedule"] == [{"cron": "43 10 * * 1"}]
    assert workflow["permissions"] == {"contents": "read"}
    steps = workflow["jobs"]["audit"]["steps"]
    run_steps = [step["run"] for step in steps if "run" in step]
    assert "python scripts/run_external_reference_audits.py" in run_steps
    audit_step = next(
        step for step in steps if step.get("name") == "Run report-only audits"
    )
    assert audit_step["env"] == {"ZENODO_TOKEN": "${{ secrets.ZENODO_TOKEN }}"}
