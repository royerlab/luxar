"""Keep the declared Python static-check aggregate wired into CI."""

from __future__ import annotations

import re
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

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


def _run_commands(value: Any) -> Iterator[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "run" and isinstance(child, str):
                yield child
            else:
                yield from _run_commands(child)
    elif isinstance(value, list):
        for child in value:
            yield from _run_commands(child)


def _invoked_hatch_scripts(commands: Sequence[str]) -> set[str]:
    return {
        match.group(1)
        for command in commands
        for match in re.finditer(r"\bhatch run ([A-Za-z0-9_-]+)", command)
    }


def _uncovered_check_static_members(
    pyproject_text: str, commands: Sequence[str]
) -> set[str]:
    scripts = _default_scripts(pyproject_text)
    members = scripts["check-static"]
    assert isinstance(members, list)
    invoked = _invoked_hatch_scripts(commands)
    return set(members) - invoked - CHECK_STATIC_EXEMPTIONS.keys()


def _repository_workflow_commands() -> list[str]:
    workflow_dir = PROJECT_ROOT / ".github" / "workflows"
    workflows = [
        yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in sorted(workflow_dir.iterdir())
        if path.suffix in {".yml", ".yaml"}
    ]
    return list(_run_commands(workflows))


def test_every_check_static_member_is_reached_or_explicitly_exempted() -> None:
    """A new local static check must not silently miss every CI workflow."""
    pyproject_text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    uncovered = _uncovered_check_static_members(
        pyproject_text, _repository_workflow_commands()
    )

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
    invoked = _invoked_hatch_scripts(_repository_workflow_commands())

    assert set(CHECK_STATIC_EXEMPTIONS) <= set(members)
    assert set(CHECK_STATIC_EXEMPTIONS).isdisjoint(invoked)
    assert all(reason.strip() for reason in CHECK_STATIC_EXEMPTIONS.values())


def test_an_unwired_new_member_fails_the_coverage_check() -> None:
    """Control arm: prove the meta-gate catches the regression it describes."""
    pyproject_text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    mutant = pyproject_text.replace(
        '    "security"\n]', '    "security",\n    "forgotten-check"\n]', 1
    )

    assert _uncovered_check_static_members(mutant, _repository_workflow_commands()) == {
        "forgotten-check"
    }
