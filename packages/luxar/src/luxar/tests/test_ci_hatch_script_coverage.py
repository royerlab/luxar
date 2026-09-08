"""Keep the declared Python static-check aggregate wired into CI."""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[5]

CHECK_STATIC_EXEMPTIONS = {
    "format": (
        "check-static's format step rewrites files; CI reaches the same formatter "
        "as a read-only check through hatch run lint"
    ),
    "check-demo-ladders": (
        "generated demo scenes are gitignored and absent from checkout-only CI; "
        "artifact builders must use --require-scenes"
    ),
    "check-scene-credits": (
        "generated demo scenes are gitignored and absent from checkout-only CI; "
        "artifact builders must use --require-scenes"
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
            "python-tests": {
                "steps": [
                    {"run": "hatch run security", "continue-on-error": True},
                    {"run": "hatch run check-versions", "if": False},
                    {"run": "hatch run check-imports"},
                ]
            }
        }
    }

    assert _invoked_hatch_scripts(_workflow_run_commands(workflow)) == {"check-imports"}


def _uncovered_check_static_members(
    members: Sequence[str], commands: Sequence[str]
) -> set[str]:
    invoked = _invoked_hatch_scripts(commands)
    return set(members) - invoked - CHECK_STATIC_EXEMPTIONS.keys()


def _ci_workflow_commands() -> list[str]:
    workflow_path = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    assert isinstance(workflow, Mapping)
    return _workflow_run_commands(workflow)


def test_every_check_static_member_is_reached_or_explicitly_exempted() -> None:
    """A new local static check must not silently miss every CI workflow."""
    pyproject_text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    members = _default_scripts(pyproject_text)["check-static"]
    assert isinstance(members, list)
    uncovered = _uncovered_check_static_members(members, _ci_workflow_commands())

    assert not uncovered, (
        "check-static member(s) are not invoked by any workflow and have no "
        f"documented exemption: {sorted(uncovered)}"
    )


def test_check_static_exemptions_are_live_and_still_unwired() -> None:
    """Remove an exemption when its artifact check leaves the aggregate or gains CI."""
    pyproject_text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    scripts = _default_scripts(pyproject_text)
    members = scripts["check-static"]
    assert isinstance(members, list)
    invoked = _invoked_hatch_scripts(_ci_workflow_commands())

    assert set(CHECK_STATIC_EXEMPTIONS) <= set(members)
    assert set(CHECK_STATIC_EXEMPTIONS).isdisjoint(invoked)
    assert all(reason.strip() for reason in CHECK_STATIC_EXEMPTIONS.values())


def test_an_unwired_new_member_fails_the_coverage_check() -> None:
    """Control arm: prove the meta-gate catches the regression it describes."""
    pyproject_text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    members = _default_scripts(pyproject_text)["check-static"]
    assert isinstance(members, list)

    assert _uncovered_check_static_members(
        [*members, "forgotten-check"], _ci_workflow_commands()
    ) == {"forgotten-check"}
