"""Regression guards for the GitHub Pages deployment workflow."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[5]
DOCS = REPO / "docs"
WORKFLOW = REPO / ".github/workflows/docs.yml"
GITATTRIBUTES = REPO / ".gitattributes"
DEMO_SITE_RUNBOOK = DOCS / "guides/developer/DEMO_SITE_RUNBOOK.md"
DEMO_DATA_MANIFEST = REPO / "packages/luxar/src/luxar/demos/data_manifest.json"


def _workflow() -> dict[str, Any]:
    """Load the Pages workflow as structured YAML."""
    return yaml.safe_load(WORKFLOW.read_text())


def _workflow_triggers() -> dict[str, Any]:
    """Load workflow triggers without YAML 1.1 coercing ``on`` to true."""
    workflow = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)
    return workflow["on"]


def _published_lfs_assets() -> set[str]:
    """Find assets referenced by this checkout's Sphinx source files."""
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


def test_runbook_archive_digest_prefixes_match_active_manifest_pins() -> None:
    """The §3.10–3.12 worked examples must name the active generations."""
    text = DEMO_SITE_RUNBOOK.read_text()
    section = text.split("### 3.10", 1)[1].split("### 3.13", 1)[0]
    quoted_prefixes = set(re.findall(r"\b([0-9a-f]{8})\.\.\.", section))
    assert quoted_prefixes, "§3.10–3.12 no longer quotes any archive digest prefixes"

    manifest = json.loads(DEMO_DATA_MANIFEST.read_text())
    active_digests: set[str] = set()

    def collect(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"sha256", "hosted_sha256"} and isinstance(child, str):
                    active_digests.add(child)
                else:
                    collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(manifest["datasets"])
    stale = sorted(
        prefix
        for prefix in quoted_prefixes
        if not any(digest.startswith(prefix) for digest in active_digests)
    )
    assert not stale, (
        "DEMO_SITE_RUNBOOK §3.10–3.12 quotes archive generations that are no "
        f"longer active manifest pins: {stale}"
    )


def test_pages_publishes_daily_or_on_demand_not_on_main_push() -> None:
    """Keep publication off pushes; scheduling is active only on default dev."""
    triggers = _workflow_triggers()

    assert "push" not in triggers
    assert triggers.get("schedule")
    assert "workflow_dispatch" in triggers


def test_pages_publishes_promoted_main_content() -> None:
    """Use dev's workflow steps to publish only promoted main content."""
    steps = _workflow()["jobs"]["build"]["steps"]
    checkout = next(
        step for step in steps if step.get("uses", "").startswith("actions/checkout@")
    )

    assert checkout.get("with", {}).get("ref") == "main"


def test_pages_checkout_does_not_smudge_the_whole_lfs_repository() -> None:
    """Keep the checkout from downloading every repository LFS object."""
    steps = _workflow()["jobs"]["build"]["steps"]
    checkout = next(
        step for step in steps if step.get("uses", "").startswith("actions/checkout@")
    )

    assert str(checkout.get("with", {}).get("lfs", "")).lower() != "true"


def test_pages_fetches_only_the_lfs_assets_published_by_sphinx() -> None:
    """Keep dev's asset oracle compatible with main until it is promoted."""
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


def test_pages_cancels_builds_superseded_by_newer_publication_runs() -> None:
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
