from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).parents[2]
DEPENDABOT_CONFIG = REPO / ".github" / "dependabot.yml"


@pytest.mark.parametrize(
    ("group", "patterns"),
    [
        ("typescript-eslint", ["@typescript-eslint/*"]),
        ("vitest", ["vitest", "@vitest/*"]),
    ],
)
def test_viewer_lockstep_dependencies_are_grouped(
    group: str, patterns: list[str]
) -> None:
    config = yaml.safe_load(DEPENDABOT_CONFIG.read_text())
    viewer_update = next(
        update
        for update in config["updates"]
        if update["package-ecosystem"] == "npm"
        and update["directory"] == "/packages/luxar-viewer"
    )

    assert viewer_update["groups"][group]["patterns"] == patterns
