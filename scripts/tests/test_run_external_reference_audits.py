from __future__ import annotations

import ast
import http.client
import importlib.util
import io
import subprocess
import sys
import urllib.error
from pathlib import Path

import pytest
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
        "Hosted gallery media",
    ]
    assert audit_module.AUDITS[2].required_env == ("ZENODO_TOKEN",)
    assert audit_module.AUDITS[2].command == ("make", "check-zenodo-live")
    assert audit_module.AUDITS[3].command == ("make", "check-gallery-media")


def test_make_targets_use_the_intended_python_environments() -> None:
    makefile = (SCRIPT.parents[1] / "Makefile").read_text()

    assert (
        "check-external-references:  ## Run all network-backed reference audits"
        " (report-only)\n\t$(HATCH) run python scripts/run_external_reference_audits.py"
        in makefile
    )
    assert (
        "check-zenodo-live:  ## Opt-in live Zenodo manifest-pin audit"
        " (not a required CI gate)\n\tpython3 scripts/zenodo_migration_audit.py --live"
        in makefile
    )
    assert (
        "check-gallery-media:  ## Verify hosted root-README media against its manifest"
        " (opt-in)\n\t$(HATCH) run python scripts/gallery/verify_media.py" in makefile
    )


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


def test_demo_configuration_findings_are_not_reported_as_parse_failures(
    monkeypatch,
) -> None:
    outputs = iter(("[CONFIG] manifest missing\n", ""))
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, next(outputs), ""
        ),
    )

    configuration = audit_module.run_audit(audit_module.AUDITS[1], env={})
    invalid_report = audit_module.run_audit(audit_module.AUDITS[1], env={})

    assert configuration.level is audit_module.Level.ERROR
    assert configuration.detail == "completed with configuration findings"
    assert invalid_report.level is audit_module.Level.ERROR
    assert invalid_report.detail == "completed with invalid report levels"


def test_unknown_demo_marker_does_not_override_known_levels() -> None:
    output = "[INFO] nothing wrong here\n[OK] fine\n"

    assert audit_module._level_from_output(output) is audit_module.Level.PASS


def test_demo_level_contract_matches_the_producer() -> None:
    producer = SCRIPT.with_name("check_demo_links.py")
    tree = ast.parse(producer.read_text())
    producer_levels = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.isupper()
        and node.value.isalpha()
    }

    assert producer_levels == set(audit_module._DEMO_LEVELS)


def test_missing_configuration_is_a_notice_without_running(monkeypatch) -> None:
    def unexpected_run(*args, **kwargs):
        raise AssertionError("command must not run without its required secret")

    monkeypatch.setattr(subprocess, "run", unexpected_run)

    result = audit_module.run_audit(audit_module.AUDITS[2], env={})

    assert result.level is audit_module.Level.NOTICE
    assert result.detail == "not configured; leg skipped: ZENODO_TOKEN"


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
            (0, "gallery media verified"),
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
        audit_module.Level.PASS,
    ]


def test_command_traceback_is_a_configuration_error(monkeypatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 1, "", "Traceback (most recent call last):\nValueError"
        ),
    )

    result = audit_module.run_audit(audit_module.AUDITS[0], env={})

    assert result.level is audit_module.Level.ERROR


def test_missing_command_interpreter_is_a_configuration_error(monkeypatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            2,
            "",
            "bash: /tmp/empty/.local/bin/hatch: No such file or directory\n"
            "make: *** [Makefile:882: check-docs-external-links] Error 127",
        ),
    )

    result = audit_module.run_audit(audit_module.AUDITS[0], env={})

    assert result.level is audit_module.Level.ERROR


@pytest.mark.parametrize(
    ("status", "reason"), [(401, "UNAUTHORIZED"), (403, "FORBIDDEN")]
)
def test_rejected_required_credential_is_a_configuration_error(
    monkeypatch, status: int, reason: str
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            2,
            "",
            f"Zenodo returned HTTP {status} {reason} for deposition 21912280.\n"
            "make: *** [Makefile:885: check-zenodo-live] Error 1",
        ),
    )

    result = audit_module.run_audit(
        audit_module.AUDITS[2], env={"ZENODO_TOKEN": "rejected-token"}
    )

    assert result.level is audit_module.Level.ERROR


@pytest.mark.parametrize(
    ("status", "reason"), [(401, "UNAUTHORIZED"), (403, "FORBIDDEN")]
)
def test_rejected_credential_markers_match_the_zenodo_producer(
    monkeypatch, status: int, reason: str
) -> None:
    producer = SCRIPT.with_name("zenodo_migration_audit.py")
    spec = importlib.util.spec_from_file_location("zenodo_migration_audit", producer)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setattr(sys, "argv", [str(producer)])
    spec.loader.exec_module(module)
    error = urllib.error.HTTPError("https://example.invalid", status, reason, {}, None)
    monkeypatch.setattr(
        module.urllib.request,
        "urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(SystemExit) as exc_info:
        module.fetch_deposition("21912280", "rejected-token")

    assert str(exc_info.value).startswith(audit_module._REJECTED_CREDENTIAL_MARKERS)


@pytest.mark.parametrize(
    "failure", ["url", "timeout", "json", "reset", "http", "bad-utf8"]
)
def test_zenodo_transport_failures_remain_warnings(monkeypatch, failure: str) -> None:
    producer = SCRIPT.with_name("zenodo_migration_audit.py")
    spec = importlib.util.spec_from_file_location("zenodo_migration_audit", producer)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setattr(sys, "argv", [str(producer)])
    spec.loader.exec_module(module)

    if failure in {"json", "bad-utf8"}:
        monkeypatch.setattr(
            module.urllib.request,
            "urlopen",
            lambda *args, **kwargs: io.BytesIO(
                b"not json" if failure == "json" else b"\x80abc"
            ),
        )
    elif failure in {"reset", "http"}:
        error = (
            ConnectionResetError("connection reset by peer")
            if failure == "reset"
            else http.client.IncompleteRead(b"partial response")
        )

        class BrokenResponse(io.BytesIO):
            def read(self, *args, **kwargs):
                raise error

        monkeypatch.setattr(
            module.urllib.request,
            "urlopen",
            lambda *args, **kwargs: BrokenResponse(),
        )
    else:
        error = (
            urllib.error.URLError("temporary DNS failure")
            if failure == "url"
            else TimeoutError("read timed out")
        )
        monkeypatch.setattr(
            module.urllib.request,
            "urlopen",
            lambda *args, **kwargs: (_ for _ in ()).throw(error),
        )

    with pytest.raises(SystemExit) as exc_info:
        module.fetch_deposition("21912280", "configured-token")

    output = str(exc_info.value)
    assert output.startswith("Zenodo request failed for deposition 21912280:")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, "", output),
    )

    result = audit_module.run_audit(
        audit_module.AUDITS[2], env={"ZENODO_TOKEN": "configured-token"}
    )

    assert result.level is audit_module.Level.WARNING
    assert result.detail == "exited 1"


def test_auth_failure_text_without_required_env_remains_a_warning(monkeypatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 1, "", "Zenodo returned HTTP 403 FORBIDDEN"
        ),
    )
    audit = audit_module.Audit("Public endpoint", ("check-public-endpoint",))

    result = audit_module.run_audit(audit, env={})

    assert result.level is audit_module.Level.WARNING


def test_missing_demo_make_target_is_a_configuration_error(monkeypatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 2, "", "make: *** No rule to make target 'check-demo-links'. Stop."
        ),
    )

    result = audit_module.run_audit(audit_module.AUDITS[1], env={})

    assert result.level is audit_module.Level.ERROR
    assert result.detail == "exited 2"


def test_missing_established_make_target_is_a_configuration_error(monkeypatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            2,
            "",
            "make: *** No rule to make target 'check-zenodo-live'. Stop.",
        ),
    )
    audit = audit_module.Audit("Zenodo", ("make", "check-zenodo-live"))

    result = audit_module.run_audit(audit, env={})

    assert result.level is audit_module.Level.ERROR


def test_timeout_is_warning_with_partial_output(monkeypatch) -> None:
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            args[0], kwargs["timeout"], output=b"partial-progress\n", stderr=None
        )

    monkeypatch.setattr(subprocess, "run", timeout)

    result = audit_module.run_audit(audit_module.AUDITS[0], env={}, timeout_seconds=1)

    assert result.level is audit_module.Level.WARNING
    assert result.detail == "timed out after 1 seconds"
    assert result.output == "partial-progress"


def test_audits_run_from_the_repository_root(monkeypatch) -> None:
    seen_cwd = None

    def run(*args, **kwargs):
        nonlocal seen_cwd
        seen_cwd = kwargs["cwd"]
        return subprocess.CompletedProcess(args[0], 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)

    audit_module.run_audit(audit_module.AUDITS[0], env={})

    assert seen_cwd == SCRIPT.parents[1]


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
    summary = summary_path.read_text()
    assert stdout.endswith(summary)
    assert stdout.count("::warning title=External reference audit::") == len(
        audit_module.AUDITS
    )
    assert "**Worst level: WARNING**" in summary
    assert stdout.count("**WARNING**") == len(audit_module.AUDITS)
    assert "report-only and never gate merges" in stdout


def test_annotations_surface_non_pass_levels_and_escape_commands() -> None:
    results = [
        audit_module.Result(
            audit_module.AUDITS[0], audit_module.Level.PASS, "clean", ""
        ),
        audit_module.Result(
            audit_module.AUDITS[0],
            audit_module.Level.NOTICE,
            "skipped%\nnot configured",
            "",
        ),
        audit_module.Result(
            audit_module.AUDITS[1], audit_module.Level.WARNING, "stale", ""
        ),
        audit_module.Result(
            audit_module.AUDITS[2], audit_module.Level.ERROR, "broken", ""
        ),
    ]

    annotations = audit_module.render_annotations(results)

    assert annotations.count("::warning title=External reference audit::") == 2
    assert annotations.count("::error title=External reference audit::") == 1
    assert "skipped%25%0Anot configured" in annotations
    assert len(annotations.splitlines()) == 3
    assert "PASS" not in annotations


def test_summary_escapes_and_bounds_command_output() -> None:
    dangerous = "x" * (audit_module.MAX_SUMMARY_OUTPUT_CHARS + 1) + "</pre>"
    result = audit_module.Result(
        audit_module.AUDITS[0], audit_module.Level.WARNING, "finding", dangerous
    )

    summary = audit_module.render_summary([result])

    assert "earlier characters omitted" in summary
    assert "&lt;/pre&gt;" in summary
    assert dangerous not in summary


def test_summary_escapes_table_delimiters_and_newlines() -> None:
    result = audit_module.Result(
        audit_module.AUDITS[0], audit_module.Level.ERROR, "bad | config\nline", ""
    )

    summary = audit_module.render_summary([result])

    assert "bad \\| config line" in summary


def test_unwritable_summary_does_not_make_the_audit_gate(monkeypatch, tmp_path) -> None:
    results = [
        audit_module.Result(audit, audit_module.Level.PASS, "clean", "")
        for audit in audit_module.AUDITS
    ]
    monkeypatch.setattr(audit_module, "run_audit", lambda audit: results.pop(0))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path))

    assert audit_module.main() == 0


def test_weekly_workflow_is_manual_read_only_and_uses_the_aggregator() -> None:
    workflow_path = (
        SCRIPT.parents[1] / ".github/workflows/external-reference-audits.yml"
    )
    workflow = yaml.load(workflow_path.read_text(), Loader=yaml.BaseLoader)

    assert workflow["on"]["workflow_dispatch"] == ""
    assert workflow["on"]["schedule"] == [{"cron": "43 10 * * 1"}]
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"] == {
        "group": "external-reference-audits",
        "cancel-in-progress": "false",
    }
    audit_job = workflow["jobs"]["audit"]
    assert audit_job["timeout-minutes"] == "65"
    steps = audit_job["steps"]
    free_disk = next(step for step in steps if step.get("name") == "Free disk space")
    assert free_disk["if"] == "runner.environment == 'github-hosted'"
    assert "/usr/local/lib/android" in free_disk["run"]
    run_steps = [step["run"] for step in steps if "run" in step]
    assert "python scripts/run_external_reference_audits.py" in run_steps
    audit_step = next(
        step for step in steps if step.get("name") == "Run report-only audits"
    )
    assert audit_step["env"] == {"ZENODO_TOKEN": "${{ secrets.ZENODO_TOKEN }}"}
    setup_python = next(
        step
        for step in steps
        if step.get("uses", "").startswith("actions/setup-python@")
    )
    assert setup_python["id"] == "setup-python"
    install_hatch = next(step for step in steps if step.get("name") == "Install Hatch")
    assert install_hatch["run"] == (
        'pipx install hatch --python "${{ steps.setup-python.outputs.python-path }}"'
    )
