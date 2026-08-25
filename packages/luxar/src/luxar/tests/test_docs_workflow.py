"""Regression guards for the GitHub Pages deployment workflow."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[5]
DOCS = REPO / "docs"
WORKFLOW = REPO / ".github/workflows/docs.yml"


def _workflow() -> dict[str, Any]:
    """Load the Pages workflow as structured YAML."""
    return yaml.safe_load(WORKFLOW.read_text())


def _published_lfs_assets() -> set[str]:
    """Find LFS-tracked screenshots referenced by Sphinx source files."""
    sources = [
        source
        for source in DOCS.rglob("*")
        if source.suffix in {".md", ".rst"} and "_build" not in source.parts
    ]
    published = set()
    for asset in (DOCS / "images/docs").glob("*.png"):
        docs_relative = asset.relative_to(DOCS).as_posix()
        if any(docs_relative in source.read_text() for source in sources):
            published.add(asset.relative_to(REPO).as_posix())
    return published


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
    assert set(match.group(1).split(",")) == _published_lfs_assets()


def test_pages_cancels_deployments_superseded_by_newer_main_promotions() -> None:
    """Do not serialize stale Pages builds behind newer main promotions."""
    concurrency = _workflow()["concurrency"]

    assert concurrency == {"group": "pages", "cancel-in-progress": True}
