"""Regression guards for docs deployment and the demo-site runbook."""

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


def test_runbook_section_three_headings_are_sequential() -> None:
    """Keep independently landed runbook additions in contiguous order."""
    text = DEMO_SITE_RUNBOOK.read_text()
    section_numbers = [
        int(number) for number in re.findall(r"^### 3\.(\d+)\b", text, re.MULTILINE)
    ]

    assert section_numbers, "no '### 3.x' headings found in the runbook"
    assert section_numbers == list(range(1, len(section_numbers) + 1))


def test_runbook_audits_built_scenes_before_upload() -> None:
    """Keep both fail-loud artifact checks in the publishing wave."""
    text = DEMO_SITE_RUNBOOK.read_text()
    section = text.split("## 2. Publishing a wave", 1)[1].split(
        "### 2.1 Always publish to a new dated prefix", 1
    )[0]

    assert "hatch run check-demo-ladders --require-scenes" in section
    assert "hatch run check-scene-credits --require-scenes" in section
    assert section.index("audit the complete local scene inventory") < section.index(
        "upload only what changed"
    )


def test_runbook_archive_digest_prefixes_match_active_manifest_pins() -> None:
    """Runbook archive examples must name active record generations."""
    text = DEMO_SITE_RUNBOOK.read_text()
    quoted_prefixes = set(re.findall(r"\b([0-9a-f]{6,64})(?:\.\.\.|…)", text))
    assert quoted_prefixes, "runbook no longer quotes any archive digest prefixes"

    manifest = json.loads(DEMO_DATA_MANIFEST.read_text())
    active_digests: set[str] = set()

    def collect(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "sha256" and isinstance(child, str):
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
        "DEMO_SITE_RUNBOOK quotes archive generations that are no longer active "
        f"manifest pins: {stale}"
    )

    cmu1_ch0 = next(
        file
        for file in manifest["datasets"]["gsplats_cmu1_pathology"]["files"]
        if file["name"] == "cmu1_ch0.gsplats.zarr.zip"
    )
    labelled_pin = re.search(
        r"^[ \t]+cmu1_ch0[ \t]+sha256[ \t]+"
        r"([0-9a-f]{6,64})(?:\.\.\.|…)[ \t]+bytes[ \t]+([\d,]+)"
        r"[ \t]+<-[ \t]+current record pin$",
        text,
        re.MULTILINE,
    )
    assert labelled_pin is not None, "runbook current cmu1_ch0 record pin not found"
    sha256_prefix, _ = labelled_pin.groups()
    assert cmu1_ch0["sha256"].startswith(sha256_prefix), (
        "runbook cmu1_ch0 sha256 prefix does not match the active manifest pin"
    )


def test_runbook_archive_size_example_matches_active_manifest_pin() -> None:
    """Runbook archive size examples must match the active record pin."""
    text = DEMO_SITE_RUNBOOK.read_text()
    labelled_pin = re.search(
        r"^[ \t]+cmu1_ch0[ \t]+sha256[ \t]+"
        r"([0-9a-f]{6,64})(?:\.\.\.|…)[ \t]+bytes[ \t]+([\d,]+)"
        r"[ \t]+<-[ \t]+current record pin$",
        text,
        re.MULTILINE,
    )
    assert labelled_pin is not None, "runbook current cmu1_ch0 record pin not found"

    manifest = json.loads(DEMO_DATA_MANIFEST.read_text())
    cmu1_ch0 = next(
        file
        for file in manifest["datasets"]["gsplats_cmu1_pathology"]["files"]
        if file["name"] == "cmu1_ch0.gsplats.zarr.zip"
    )
    _, byte_text = labelled_pin.groups()
    assert int(byte_text.replace(",", "")) == cmu1_ch0["bytes"]


def test_runbook_publish_constraint_is_recorded_as_discharged() -> None:
    """Keep the historical hazard closed after the records were published."""
    text = DEMO_SITE_RUNBOOK.read_text()
    section = text.split(
        "#### The partial re-pin left one archive on its deliberate previous generation",
        1,
    )[1].split("### 3.21", 1)[0]
    prose = " ".join(section.split())

    assert "**seven of the eight**" in prose
    assert "Only `milkyway_dust` remains on its previous pin" in prose
    assert "publish-order constraint was discharged before #2354" in prose
    assert "10,647,985-byte pin" in prose
    assert "published together on 2026-09-02" in prose
    assert "never publish a record whose bytes disagree with the committed pin" in prose
    assert (
        "Roll the draft file back to the pinned contract before publishing" not in prose
    )
    assert "drafts hold restructured files for all three" not in prose
    assert "land the paired sidecar migration first" not in prose


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
