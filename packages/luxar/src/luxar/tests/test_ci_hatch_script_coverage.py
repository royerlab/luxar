"""Keep the declared Python static-check aggregate wired into CI."""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from luxar.tests.test_ci_diff_classifier import (
    _ALL_DOMAINS_BLOCK_RE,
    _classifies,
    _domain_patterns,
)
from luxar.tests.test_ci_diff_classifier import (
    WORKFLOW as CI_WORKFLOW,
)

PROJECT_ROOT = Path(__file__).resolve().parents[5]
WORKFLOWS_ROOT = PROJECT_ROOT / ".github" / "workflows"

CHECK_STATIC_EXEMPTIONS = {
    "format": (
        "check-static's format step rewrites files; CI reaches the same formatter "
        "as a read-only check through hatch run lint"
    ),
    "check-demo-ladders": (
        "generated demo scenes are gitignored and absent from checkout-only CI; "
        "the gallery generator gates generated stores and the pre-upload runbook "
        "exposes the corpus-wide inventory with --require-scenes"
    ),
    "check-scene-credits": (
        "generated demo scenes are gitignored and absent from checkout-only CI; "
        "the gallery generator gates generated stores and the pre-upload runbook "
        "checks the full inventory with --require-scenes"
    ),
}


def _default_scripts(pyproject_text: str) -> Mapping[str, str | list[str]]:
    pyproject = tomllib.loads(pyproject_text)
    return pyproject["tool"]["hatch"]["envs"]["default"]["scripts"]


def _workflow_run_commands(workflow: Mapping[str, object]) -> list[str]:
    jobs = workflow.get("jobs")
    assert isinstance(jobs, Mapping)
    commands = []
    for job in jobs.values():
        if not isinstance(job, Mapping):
            continue
        if job.get("continue-on-error") or job.get("if") is False:
            continue
        steps = job.get("steps", [])
        assert isinstance(steps, list)
        commands.extend(
            step["run"]
            for step in steps
            if isinstance(step, Mapping)
            and isinstance(step.get("run"), str)
            and not step.get("continue-on-error")
            and step.get("if") is not False
        )
    return commands


def _invoked_hatch_scripts(commands: Sequence[str]) -> set[str]:
    return {
        match.group(1)
        for command in commands
        for match in re.finditer(r"\bhatch run ([A-Za-z0-9_-]+)", command)
    }


def test_non_blocking_steps_do_not_count_as_ci_coverage() -> None:
    workflow = {
        "jobs": {
            "non-blocking-job": {
                "continue-on-error": True,
                "steps": [{"run": "hatch run check-contract"}],
            },
            "disabled-job": {
                "if": False,
                "steps": [{"run": "hatch run check-data-manifest"}],
            },
            "python-tests": {
                "steps": [
                    {"run": "hatch run security", "continue-on-error": True},
                    {"run": "hatch run check-versions", "if": False},
                    {"run": "hatch run check-imports"},
                ]
            },
        }
    }

    assert _invoked_hatch_scripts(_workflow_run_commands(workflow)) == {"check-imports"}


def _uncovered_check_static_members(
    members: Sequence[str], commands: Sequence[str]
) -> set[str]:
    invoked = _invoked_hatch_scripts(commands)
    return set(members) - invoked - CHECK_STATIC_EXEMPTIONS.keys()


def _classified_python_workflow_paths() -> set[Path]:
    classifier = CI_WORKFLOW.read_text(encoding="utf-8")
    python_pattern = _domain_patterns(classifier)["py"]
    all_domains_patterns = [
        pattern for pattern, _body in _ALL_DOMAINS_BLOCK_RE.findall(classifier)
    ]
    return {
        path.relative_to(PROJECT_ROOT)
        for path in WORKFLOWS_ROOT.glob("*.yml")
        if _classifies(python_pattern, str(path.relative_to(PROJECT_ROOT)))
        or any(
            _classifies(pattern, str(path.relative_to(PROJECT_ROOT)))
            for pattern in all_domains_patterns
        )
    }


def _classified_python_workflows() -> dict[Path, Mapping[str, object]]:
    workflows = {}
    for relative_path in _classified_python_workflow_paths():
        workflow = yaml.safe_load(
            (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        )
        assert isinstance(workflow, Mapping)
        workflows[relative_path] = workflow
    return workflows


def _triggers_pull_requests(workflow: Mapping[str, object]) -> bool:
    triggers = workflow.get(True, workflow.get("on"))
    if isinstance(triggers, str):
        return triggers == "pull_request"
    if isinstance(triggers, Sequence):
        return "pull_request" in triggers
    return isinstance(triggers, Mapping) and "pull_request" in triggers


def _pull_request_workflow_commands() -> list[str]:
    return [
        command
        for workflow in _classified_python_workflows().values()
        if _triggers_pull_requests(workflow)
        for command in _workflow_run_commands(workflow)
    ]


def test_workflow_scan_is_classified_and_only_credits_pull_request_gates() -> None:
    workflows = _classified_python_workflows()
    credited = {
        path
        for path, workflow in workflows.items()
        if _triggers_pull_requests(workflow)
    }
    scheduled_only = {
        Path(".github/workflows/docs.yml"),
        Path(".github/workflows/external-reference-audits.yml"),
    }

    assert set(workflows) == _classified_python_workflow_paths()
    assert scheduled_only <= set(workflows)
    assert Path(".github/workflows/ci.yml") in credited
    assert scheduled_only.isdisjoint(credited)


def test_every_check_static_member_is_reached_or_explicitly_exempted() -> None:
    """A new local static check must reach a workflow that gates pull requests."""
    pyproject_text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    members = _default_scripts(pyproject_text)["check-static"]
    assert isinstance(members, list)
    uncovered = _uncovered_check_static_members(
        members, _pull_request_workflow_commands()
    )

    assert not uncovered, (
        "check-static member(s) are not invoked by a workflow that gates pull "
        "requests and have no "
        f"documented exemption: {sorted(uncovered)}"
    )


def test_check_static_exemptions_are_live_and_still_unwired() -> None:
    """Remove exemptions that leave the aggregate or reach pull-request CI."""
    pyproject_text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    scripts = _default_scripts(pyproject_text)
    members = scripts["check-static"]
    assert isinstance(members, list)
    invoked = _invoked_hatch_scripts(_pull_request_workflow_commands())

    assert set(CHECK_STATIC_EXEMPTIONS) <= set(members)
    assert set(CHECK_STATIC_EXEMPTIONS).isdisjoint(invoked)
    assert all(reason.strip() for reason in CHECK_STATIC_EXEMPTIONS.values())


def test_an_unwired_new_member_fails_the_coverage_check() -> None:
    """Control arm: prove the meta-gate catches the regression it describes."""
    pyproject_text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    members = _default_scripts(pyproject_text)["check-static"]
    assert isinstance(members, list)

    assert _uncovered_check_static_members(
        [*members, "forgotten-check"], _pull_request_workflow_commands()
    ) == {"forgotten-check"}
