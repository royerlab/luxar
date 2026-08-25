"""Regression guards for the GitHub Pages deployment workflow."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[5]
DOCS = REPO / "docs"
WORKFLOW = REPO / ".github/workflows/docs.yml"
GITATTRIBUTES = REPO / ".gitattributes"


def _workflow() -> dict[str, Any]:
    """Load the Pages workflow as structured YAML."""
    return yaml.safe_load(WORKFLOW.read_text())


def _published_lfs_assets() -> set[str]:
    """Find LFS-tracked assets referenced by Sphinx source files."""
    sources = [
        source
        for source in DOCS.rglob("*")
        if source.suffix in {".md", ".rst"} and "_build" not in source.parts
    ]
    published = set()
    lfs_patterns = {
        line.split()[0]
        for line in GITATTRIBUTES.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#") and "filter=lfs" in line
    }
    for pattern in lfs_patterns:
        if not pattern.startswith("docs/"):
            continue
        for asset in REPO.glob(pattern):
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

    assert str(checkout.get("with", {}).get("lfs", "")).lower() != "true"


def test_pages_fetches_only_the_lfs_assets_published_by_sphinx() -> None:
    """Fetch the four published screenshots without pulling unrelated payloads."""
    steps = _workflow()["jobs"]["build"]["steps"]
    fetch = next(
        step for step in steps if step.get("name") == "Fetch documentation LFS assets"
    )
    fetch_index = steps.index(fetch)
    build_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("name") == "Build Sphinx documentation"
    )
    match = re.search(r'git lfs pull --include="([^"]+)" --exclude=""', fetch["run"])

    assert match is not None
    assert set(match.group(1).split(",")) == _published_lfs_assets()
    assert fetch_index < build_index


def test_pages_cancels_deployments_superseded_by_newer_main_promotions() -> None:
    """Cancel stale builds without interrupting an in-flight deployment."""
    workflow = _workflow()
    jobs = workflow["jobs"]

    assert "concurrency" not in workflow
    assert jobs["build"]["concurrency"] == {
        "group": "pages-build",
        "cancel-in-progress": True,
    }
    assert jobs["deploy"]["concurrency"] == {
        "group": "pages-deploy",
        "cancel-in-progress": False,
    }


def test_pages_rejects_lfs_pointers_before_upload(tmp_path: Path) -> None:
    """Fail closed if an LFS pointer reaches Sphinx's published asset trees."""
    steps = _workflow()["jobs"]["build"]["steps"]
    check_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("name") == "Reject unresolved LFS pointers"
    )
    build_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("name") == "Build Sphinx documentation"
    )
    upload_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("name") == "Upload Pages artifact"
    )
    run = steps[check_index]["run"]

    assert build_index < check_index < upload_index
    assert "docs/_build/html/_images" in run
    assert "docs/_build/html/_downloads" in run
    assert "version https://git-lfs.github.com/spec/v1" in run
    assert "::error file=" in run

    images = tmp_path / "docs/_build/html/_images"
    images.mkdir(parents=True)
    (images / "valid.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    clean = subprocess.run(
        ["bash", "-c", run],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert clean.returncode == 0, clean.stdout + clean.stderr

    (images / "bad.png").write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:0123456789abcdef\n"
        "size 8\n"
    )
    rejected = subprocess.run(
        ["bash", "-c", run],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert rejected.returncode == 1, rejected.stdout + rejected.stderr
    assert (
        "::error file=docs/_build/html/_images/bad.png::"
        "Git LFS pointer would be published"
    ) in rejected.stdout
