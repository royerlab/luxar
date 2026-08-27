"""Tests for the report-only gallery tile staleness checker."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import check_tile_staleness as stale  # noqa: E402


def _stamp(day: int) -> stale.CommitStamp:
    return stale.CommitStamp(
        sha=f"commit-{day}",
        committed_at=datetime(2026, 8, day, tzinfo=timezone.utc),
    )


def _git(repo: Path, *args: str, day: int | None = None) -> str:
    env = os.environ.copy()
    if day is not None:
        date = f"2026-08-{day:02d}T12:00:00+00:00"
        env.update(GIT_AUTHOR_DATE=date, GIT_COMMITTER_DATE=date)
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit(repo: Path, message: str, day: int) -> None:
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", message, day=day)


def _write_manifest(repo: Path, title_b: str = "B", include_c: bool = False) -> None:
    demos = [
        {
            "id": "a",
            "title": "A",
            "script": "demo_a.py",
            "dataset": "datasets/demos/a.luxar.zarr",
        },
        {
            "id": "b",
            "title": title_b,
            "script": "demo_b.py",
            "dataset": "datasets/demos/b.luxar.zarr",
        },
    ]
    if include_c:
        demos.append(
            {
                "id": "c",
                "title": "C",
                "script": "demo_c.py",
                "dataset": "datasets/demos/c.luxar.zarr",
            }
        )
    manifest = {
        "$comment": "synthetic gallery manifest",
        "demos": demos,
    }
    path = repo / "scripts/gallery/manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def _repo(tmp_path: Path) -> Path:
    for relative in (
        "scripts/gallery",
        "packages/luxar/src/luxar/demos",
        "packages/luxar/src/luxar/shading",
        "packages/luxar/src/luxar/shading/tests",
        "packages/luxar-viewer/src/tests/screenshots",
        "docs/images/readme/gallery",
    ):
        (tmp_path / relative).mkdir(parents=True, exist_ok=True)
    _write_manifest(tmp_path)
    (tmp_path / "scripts/gallery/generate_gallery_datasets.py").write_text(
        "# generator\n"
    )
    for filename in (
        "generate-gallery.spec.ts",
        "exposure-policy.ts",
        "crop-policy.ts",
    ):
        (
            tmp_path / f"packages/luxar-viewer/src/tests/screenshots/{filename}"
        ).write_text(f"// {filename}\n")
    for demo_id in ("a", "b"):
        (tmp_path / f"packages/luxar/src/luxar/demos/demo_{demo_id}.py").write_text(
            "from luxar.shading import bake_ambient_occlusion\n"
            if demo_id == "a"
            else "# demo b\n"
        )
        for suffix in ("webp", "webm"):
            (tmp_path / f"docs/images/readme/gallery/{demo_id}.{suffix}").write_text(
                f"tile {demo_id}.{suffix}\n"
            )
    (tmp_path / "packages/luxar/src/luxar/shading/occlusion.py").write_text(
        "# shading\n"
    )

    _git(tmp_path, "init", "-b", "dev")
    _git(tmp_path, "config", "user.name", "Gallery Test")
    _git(tmp_path, "config", "user.email", "gallery@example.com")
    _commit(tmp_path, "initial", 20)

    for demo_id in ("a", "b"):
        for suffix in ("webp", "webm"):
            (tmp_path / f"docs/images/readme/gallery/{demo_id}.{suffix}").write_text(
                f"fresh tile {demo_id}.{suffix}\n"
            )
    _commit(tmp_path, "capture tiles", 21)
    return tmp_path


def test_stale_inputs_require_a_strictly_newer_commit() -> None:
    inputs = {"older": _stamp(19), "same commit": _stamp(20), "newer": _stamp(21)}

    assert stale.stale_inputs(_stamp(20), inputs) == ["newer"]


def test_manifest_line_ranges_isolate_each_demo_entry() -> None:
    text = json.dumps(
        {
            "$comment": "a brace in a string does not end an entry: }",
            "demos": [
                {"id": "a", "nested": {"value": 1}},
                {"id": "b", "items": [1, 2, 3]},
            ],
        },
        indent=2,
    )

    ranges = stale.manifest_entry_line_ranges(text)

    lines = text.splitlines()
    assert '"id": "a"' in "\n".join(lines[ranges["a"].start - 1 : ranges["a"].stop])
    assert '"id": "b"' not in "\n".join(lines[ranges["a"].start - 1 : ranges["a"].stop])
    assert '"id": "b"' in "\n".join(lines[ranges["b"].start - 1 : ranges["b"].stop])


def test_manifest_history_is_entry_specific_and_shading_only_affects_importers(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    history = stale.GalleryHistory(repo)

    assert all(not status.stale_inputs for status in history.tile_statuses())

    _write_manifest(repo, title_b="B revised")
    _commit(repo, "revise b framing", 22)

    by_id = {status.demo_id: status for status in history.tile_statuses()}
    assert by_id["a"].stale_inputs == ()
    assert by_id["b"].stale_inputs == ("manifest entry",)

    (repo / "packages/luxar/src/luxar/shading/occlusion.py").write_text(
        "# revised shading\n"
    )
    _commit(repo, "revise shading", 23)

    by_id = {status.demo_id: status for status in history.tile_statuses()}
    assert by_id["a"].stale_inputs == ("luxar.shading",)
    assert by_id["b"].stale_inputs == ("manifest entry",)


def test_uncommitted_manifest_edit_does_not_affect_committed_report(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    _write_manifest(repo, title_b="uncommitted edit")

    by_id = {
        status.demo_id: status for status in stale.GalleryHistory(repo).tile_statuses()
    }

    assert by_id["a"].stale_inputs == ()
    assert by_id["b"].stale_inputs == ()


def test_staged_uncommitted_media_does_not_affect_committed_report(
    tmp_path: Path, capsys
) -> None:
    repo = _repo(tmp_path)
    (repo / "docs/images/readme/gallery/zzz.webp").write_text("staged still\n")
    (repo / "docs/images/readme/gallery/zzz.webm").write_text("staged video\n")
    _git(repo, "add", "docs/images/readme/gallery")

    assert stale.main(["--repo-root", str(repo)]) == 0
    output = capsys.readouterr().out
    assert "UNKNOWN zzz" not in output
    assert "0 stale, 2 current, 0 unknown" in output


def test_tile_uses_older_commit_from_still_and_video_pair(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "docs/images/readme/gallery/a.webp").write_text("new still only\n")
    _commit(repo, "replace still only", 22)

    by_id = {
        status.demo_id: status for status in stale.GalleryHistory(repo).tile_statuses()
    }

    assert by_id["a"].tile.committed_at.day == 21


def test_demo_script_edit_stales_only_its_tile(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "packages/luxar/src/luxar/demos/demo_b.py").write_text("# revised b\n")
    _commit(repo, "revise demo b", 22)

    by_id = {
        status.demo_id: status for status in stale.GalleryHistory(repo).tile_statuses()
    }

    assert by_id["a"].stale_inputs == ()
    assert by_id["b"].stale_inputs == ("demo generator",)


def test_gallery_capture_policy_is_a_global_input(tmp_path: Path, capsys) -> None:
    repo = _repo(tmp_path)
    crop_policy = repo / "packages/luxar-viewer/src/tests/screenshots/crop-policy.ts"
    crop_policy.write_text("// revised crop policy\n")
    _commit(repo, "revise crop policy", 22)

    assert stale.main(["--repo-root", str(repo)]) == 0
    output = capsys.readouterr().out
    assert "global inputs:" in output
    assert "crop policy" in output
    assert "STALE a" in output
    assert "STALE b" in output


def test_appending_a_manifest_entry_does_not_stale_the_previous_last_entry(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    _write_manifest(repo, include_c=True)
    (repo / "packages/luxar/src/luxar/demos/demo_c.py").write_text("# demo c\n")
    _commit(repo, "add c", 22)

    by_id = {
        status.demo_id: status for status in stale.GalleryHistory(repo).tile_statuses()
    }
    assert by_id["a"].stale_inputs == ()
    assert by_id["b"].stale_inputs == ()


def test_shading_docs_and_tests_do_not_mark_tiles_stale(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "packages/luxar/src/luxar/shading/README.md").write_text("docs only\n")
    (repo / "packages/luxar/src/luxar/shading/tests/test_occlusion.py").write_text(
        "# tests only\n"
    )
    _commit(repo, "document shading", 22)

    assert all(
        not status.stale_inputs for status in stale.GalleryHistory(repo).tile_statuses()
    )

    (repo / "packages/luxar/src/luxar/shading/occlusion.py").write_text(
        "# revised shading\n"
    )
    _commit(repo, "revise shading", 23)

    by_id = {
        status.demo_id: status for status in stale.GalleryHistory(repo).tile_statuses()
    }
    assert by_id["a"].stale_inputs == ("luxar.shading",)
    assert by_id["b"].stale_inputs == ()


def test_stale_findings_are_report_only(tmp_path: Path, capsys) -> None:
    repo = _repo(tmp_path)
    crop_policy = repo / "packages/luxar-viewer/src/tests/screenshots/crop-policy.ts"
    crop_policy.write_text("// revised crop policy\n")
    _commit(repo, "revise crop policy", 22)

    assert stale.main(["--repo-root", str(repo)]) == 0
    output = capsys.readouterr().out
    assert "STALE a" in output
    assert "STALE b" in output
    assert "2 stale, 0 current" in output


def test_bad_rows_are_reported_unknown_without_hiding_other_tiles(
    tmp_path: Path, capsys
) -> None:
    repo = _repo(tmp_path)
    (repo / "docs/images/readme/gallery/zzz.webp").write_text("unknown still\n")
    (repo / "docs/images/readme/gallery/zzz.webm").write_text("unknown video\n")
    _commit(repo, "add unmatched media", 22)

    assert stale.main(["--repo-root", str(repo)]) == 0
    output = capsys.readouterr().out
    assert "UNKNOWN zzz: committed tile 'zzz' has no manifest entry" in output
    assert "CURRENT a" in output
    assert "CURRENT b" in output
    assert "0 stale, 2 current, 1 unknown" in output


def test_uncommitted_demo_script_is_an_unknown_row(tmp_path: Path, capsys) -> None:
    repo = _repo(tmp_path)
    _write_manifest(repo, include_c=True)
    for suffix in ("webp", "webm"):
        (repo / f"docs/images/readme/gallery/c.{suffix}").write_text(
            f"tile c.{suffix}\n"
        )
    _commit(repo, "add c manifest and media", 22)
    (repo / "packages/luxar/src/luxar/demos/demo_c.py").write_text("# uncommitted\n")

    assert stale.main(["--repo-root", str(repo)]) == 0
    output = capsys.readouterr().out
    assert "UNKNOWN c: no commit history for" in output
    assert "CURRENT a" in output
    assert "CURRENT b" in output


def test_shallow_history_is_rejected_instead_of_misreported(tmp_path: Path) -> None:
    source = _repo(tmp_path / "source")
    shallow = tmp_path / "shallow"
    _git(tmp_path, "clone", "--depth", "1", source.as_uri(), str(shallow))

    with pytest.raises(stale.StalenessError, match="requires full Git history"):
        stale.GalleryHistory(shallow).tile_statuses()
