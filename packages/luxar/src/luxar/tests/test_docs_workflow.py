"""Regression guards for the GitHub Pages deployment workflow."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[5]
WORKFLOW = REPO / ".github/workflows/docs.yml"
DOCS_LFS_ASSETS = {
    "docs/images/docs/basic-3d-pointcloud.png",
    "docs/images/docs/gsplats-scene.png",
    "docs/images/docs/nd-navigation-sliders.png",
    "docs/images/docs/viewer-ui-overview.png",
}


def _workflow() -> dict[str, Any]:
    """Load the Pages workflow as structured YAML."""
    return yaml.safe_load(WORKFLOW.read_text())


def test_pages_checkout_does_not_smudge_the_whole_lfs_repository() -> None:
    """Keep the checkout from downloading every repository LFS object."""
    steps = _workflow()["jobs"]["build"]["steps"]
    checkout = next(
        step for step in steps if step.get("uses", "").startswith("actions/checkout@")
    )

    assert checkout.get("with", {}).get("lfs") is not True


def test_pages_fetches_only_the_lfs_assets_published_by_sphinx() -> None:
    """Fetch the four published screenshots without pulling unrelated payloads."""
    steps = _workflow()["jobs"]["build"]["steps"]
    fetch = next(
        step for step in steps if step.get("name") == "Fetch documentation LFS assets"
    )
    match = re.search(r'git lfs pull --include="([^"]+)" --exclude=""', fetch["run"])

    assert match is not None
    assert set(match.group(1).split(",")) == DOCS_LFS_ASSETS


def test_pages_cancels_deployments_superseded_by_newer_main_promotions() -> None:
    """Do not serialize stale Pages builds behind newer main promotions."""
    concurrency = _workflow()["concurrency"]

    assert concurrency == {"group": "pages", "cancel-in-progress": True}
