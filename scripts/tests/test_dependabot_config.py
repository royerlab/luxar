"""Regression tests for lockstep Dependabot dependency groups."""

from __future__ import annotations

import json
from fnmatch import fnmatchcase
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).parents[2]
DEPENDABOT_CONFIG = REPO / ".github" / "dependabot.yml"
VIEWER_PACKAGE = REPO / "packages" / "luxar-viewer" / "package.json"
DEPENDENCY_SECTIONS = (
    "dependencies",
    "devDependencies",
    "peerDependencies",
    "optionalDependencies",
)


@pytest.mark.parametrize(
    ("family", "unscoped_name", "scoped_prefix"),
    [
        ("TypeScript ESLint", "typescript-eslint", "@typescript-eslint/"),
        ("Vitest", "vitest", "@vitest/"),
    ],
)
@pytest.mark.parametrize("applies_to", ["version-updates", "security-updates"])
def test_viewer_lockstep_dependencies_share_dependabot_group(
    family: str,
    unscoped_name: str,
    scoped_prefix: str,
    applies_to: str,
) -> None:
    config = yaml.safe_load(DEPENDABOT_CONFIG.read_text())
    viewer_update = next(
        (
            update
            for update in config["updates"]
            if update["package-ecosystem"] == "npm"
            and update["directory"] == "/packages/luxar-viewer"
        ),
        None,
    )
    assert viewer_update is not None, "Dependabot has no npm update for the viewer"

    package = json.loads(VIEWER_PACKAGE.read_text())
    dependency_names = set().union(
        *(package.get(section, {}) for section in DEPENDENCY_SECTIONS)
    )
    family_members = sorted(
        name
        for name in dependency_names
        if name == unscoped_name or name.startswith(scoped_prefix)
    )
    assert family_members, f"package.json has no {family} dependencies"

    matching_groups = []
    for group_name, group in viewer_update.get("groups", {}).items():
        if group.get("applies-to", "version-updates") != applies_to:
            continue
        patterns = group["patterns"]
        if all(
            any(fnmatchcase(member, pattern) for pattern in patterns)
            for member in family_members
        ):
            matching_groups.append(group_name)

    assert matching_groups, (
        f"{family} dependencies {family_members} do not share a Dependabot "
        f"{applies_to} group"
    )
