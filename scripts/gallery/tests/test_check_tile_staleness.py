"""Tests for the report-only gallery tile staleness checker."""

from __future__ import annotations

import json
import os
import re
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


def _lfs_pointer(size_bytes: int) -> str:
    return (
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{'0' * 64}\n"
        f"size {size_bytes}\n"
    )


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
        "packages/luxar/src/luxar/demos/_support",
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
        "gallery-timelapse-settle.ts",
        "orbit-axis.ts",
        "exposure-policy.ts",
        "crop-policy.ts",
    ):
        (
            tmp_path / f"packages/luxar-viewer/src/tests/screenshots/{filename}"
        ).write_text(f"// {filename}\n")
    (tmp_path / "packages/luxar-viewer/playwright.gallery.config.ts").write_text(
        "// gallery config\n"
    )
    for demo_id in ("a", "b"):
        (tmp_path / f"packages/luxar/src/luxar/demos/demo_{demo_id}.py").write_text(
            "from luxar.demos._cinematic_camera import pull_in\n"
            "from luxar.shading import bake_ambient_occlusion\n"
            if demo_id == "a"
            else "from luxar.demos._support._umap_utils import colors\n"
        )
        for suffix in ("webp", "webm"):
            (tmp_path / f"docs/images/readme/gallery/{demo_id}.{suffix}").write_text(
                f"tile {demo_id}.{suffix}\n"
            )
    (tmp_path / "packages/luxar/src/luxar/shading/occlusion.py").write_text(
        "# shading\n"
    )
    (tmp_path / "packages/luxar/src/luxar/demos/_cinematic_camera.py").write_text(
        "# camera helper\n"
    )
    (tmp_path / "packages/luxar/src/luxar/demos/_support/_umap_utils.py").write_text(
        "# umap helper\n"
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


def test_media_flags_use_inclusive_warning_and_limit_boundaries() -> None:
    path = Path("tile.webm")

    assert (
        stale._format_media_size(stale.GALLERY_MEDIA_WARNING_BYTES - 1) == "19.99 MiB"
    )
    assert (
        stale._media_flag(
            stale.GalleryMedia(path, stale.GALLERY_MEDIA_WARNING_BYTES - 1)
        )
        == ""
    )
    assert (
        stale._media_flag(stale.GalleryMedia(path, stale.GALLERY_MEDIA_WARNING_BYTES))
        == " [WARNING]"
    )
    assert (
        stale._media_flag(stale.GalleryMedia(path, stale.GALLERY_MEDIA_LIMIT_BYTES - 1))
        == " [WARNING]"
    )
    assert stale._format_media_size(stale.GALLERY_MEDIA_LIMIT_BYTES - 1) == "24.99 MiB"
    assert (
        stale._media_flag(stale.GalleryMedia(path, stale.GALLERY_MEDIA_LIMIT_BYTES))
        == " [OVER LIMIT]"
    )


def test_missing_media_blob_is_reported_as_staleness_error(
    tmp_path: Path, capsys
) -> None:
    repo = _repo(tmp_path)
    media_path = Path("docs/images/readme/gallery/a.webm")
    object_id = _git(repo, "rev-parse", f"HEAD:{media_path.as_posix()}")
    (repo / ".git/objects" / object_id[:2] / object_id[2:]).unlink()

    message = f"Git object is unavailable: {object_id} ({media_path})"
    with pytest.raises(stale.StalenessError, match=re.escape(message)):
        stale.GalleryHistory(repo).report()

    assert stale.main(["--repo-root", str(repo)]) == 2
    assert capsys.readouterr().err.strip() == f"ERROR: {message}"


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


def test_demo_script_edit_stales_only_its_tile(tmp_path: Path, capsys) -> None:
    repo = _repo(tmp_path)
    (repo / "packages/luxar/src/luxar/demos/demo_b.py").write_text("# revised b\n")
    _commit(repo, "revise demo b", 22)

    by_id = {
        status.demo_id: status for status in stale.GalleryHistory(repo).tile_statuses()
    }

    assert by_id["a"].stale_inputs == ()
    assert by_id["b"].stale_inputs == ("demo generator",)

    assert stale.main(["--repo-root", str(repo)]) == 0
    output = capsys.readouterr().out
    assert "STALE b:" in output
    assert "newer per-tile inputs: demo generator" in output


@pytest.mark.parametrize(
    ("relative_path", "expected_demo", "expected_label"),
    [
        (
            "packages/luxar/src/luxar/demos/_cinematic_camera.py",
            "a",
            "demo helper _cinematic_camera",
        ),
        (
            "packages/luxar/src/luxar/demos/_support/_umap_utils.py",
            "b",
            "demo helper _support._umap_utils",
        ),
    ],
)
def test_demo_helper_edit_stales_only_its_importers(
    tmp_path: Path,
    relative_path: str,
    expected_demo: str,
    expected_label: str,
) -> None:
    repo = _repo(tmp_path)
    (repo / relative_path).write_text("# revised helper\n")
    _commit(repo, "revise demo helper", 22)

    by_id = {
        status.demo_id: status for status in stale.GalleryHistory(repo).tile_statuses()
    }

    other_demo = "b" if expected_demo == "a" else "a"
    assert by_id[expected_demo].stale_inputs == (expected_label,)
    assert by_id[other_demo].stale_inputs == ()


def test_missing_demo_helper_is_an_unknown_row(tmp_path: Path, capsys) -> None:
    repo = _repo(tmp_path)
    (repo / "packages/luxar/src/luxar/demos/demo_b.py").write_text(
        "from luxar.demos._missing import helper\n"
    )
    _commit(repo, "reference missing helper", 22)

    assert stale.main(["--repo-root", str(repo)]) == 0
    output = capsys.readouterr().out
    assert "CURRENT a" in output
    assert "UNKNOWN b: demo helper '_missing' is not tracked at HEAD" in output


def test_gallery_capture_policy_is_a_global_input(tmp_path: Path, capsys) -> None:
    repo = _repo(tmp_path)
    crop_policy = repo / "packages/luxar-viewer/src/tests/screenshots/crop-policy.ts"
    crop_policy.write_text("// revised crop policy\n")
    _commit(repo, "revise crop policy", 22)

    assert stale.main(["--repo-root", str(repo)]) == 0
    output = capsys.readouterr().out
    global_header = output.splitlines()[0]
    assert "crop policy" in global_header
    assert output.count("2026-08-22T12:00:00Z") == 1
    assert output.count("newer global inputs: crop policy") == 2
    assert "STALE a" in output
    assert "STALE b" in output


@pytest.mark.parametrize(
    "relative_path",
    [
        "packages/luxar-viewer/src/tests/screenshots/gallery-timelapse-settle.ts",
        "packages/luxar-viewer/src/tests/screenshots/orbit-axis.ts",
        "packages/luxar-viewer/playwright.gallery.config.ts",
    ],
)
def test_gallery_capture_tracks_all_render_configuration(
    tmp_path: Path, relative_path: str
) -> None:
    repo = _repo(tmp_path)
    (repo / relative_path).write_text("// revised capture configuration\n")
    _commit(repo, "revise capture configuration", 22)

    statuses = stale.GalleryHistory(repo).tile_statuses()

    assert all(status.stale_inputs == ("gallery capture",) for status in statuses)


@pytest.mark.parametrize(
    ("label", "relative_path"),
    [
        (
            "gallery capture",
            "packages/luxar-viewer/src/tests/screenshots/orbit-axis.ts",
        ),
        (
            "crop policy",
            "packages/luxar-viewer/src/tests/screenshots/crop-policy.ts",
        ),
    ],
)
def test_renamed_configured_input_is_rejected(
    tmp_path: Path,
    label: str,
    relative_path: str,
) -> None:
    repo = _repo(tmp_path)
    original = Path(relative_path)
    renamed = original.with_name(f"{original.stem}-renamed{original.suffix}")
    _git(repo, "mv", original.as_posix(), renamed.as_posix())
    _commit(repo, "rename configured input", 22)

    with pytest.raises(
        stale.StalenessError,
        match=rf"{label}.*no tracked files at HEAD.*{re.escape(relative_path)}",
    ):
        stale.GalleryHistory(repo).report()


def test_shading_guard_requires_production_file(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _git(
        repo,
        "mv",
        "packages/luxar/src/luxar/shading/occlusion.py",
        "packages/luxar/src/luxar/shading/tests/occlusion.py",
    )
    _commit(repo, "move shading implementation under tests", 22)

    with pytest.raises(
        stale.StalenessError,
        match=r"luxar\.shading.*no tracked files at HEAD",
    ):
        stale.GalleryHistory(repo).report()


def test_staged_configured_input_rename_does_not_affect_committed_report(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    _git(
        repo,
        "mv",
        "packages/luxar-viewer/src/tests/screenshots/crop-policy.ts",
        "packages/luxar-viewer/src/tests/screenshots/crop-rules.ts",
    )

    assert all(
        not status.stale_inputs for status in stale.GalleryHistory(repo).tile_statuses()
    )


def test_staged_replacement_does_not_hide_committed_input_rename(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    original = Path("packages/luxar-viewer/src/tests/screenshots/orbit-axis.ts")
    renamed = original.with_name("orbit-axis-renamed.ts")
    _git(repo, "mv", original.as_posix(), renamed.as_posix())
    _commit(repo, "rename configured input", 22)
    (repo / original).write_text("// staged replacement\n")
    _git(repo, "add", original.as_posix())

    with pytest.raises(
        stale.StalenessError,
        match=rf"gallery capture.*no tracked files at HEAD.*{re.escape(original.as_posix())}",
    ):
        stale.GalleryHistory(repo).report()


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


def test_shading_docs_and_tests_do_not_mark_tiles_stale(tmp_path: Path, capsys) -> None:
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

    assert stale.main(["--repo-root", str(repo)]) == 0
    output = capsys.readouterr().out
    assert "luxar.shading" not in output.splitlines()[0]
    assert "newer per-tile inputs: luxar.shading" in output


def test_stale_findings_are_report_only(tmp_path: Path, capsys) -> None:
    repo = _repo(tmp_path)
    crop_policy = repo / "packages/luxar-viewer/src/tests/screenshots/crop-policy.ts"
    crop_policy.write_text("// revised crop policy\n")
    _commit(repo, "revise crop policy", 22)

    assert stale.main(["--repo-root", str(repo)]) == 0
    output = capsys.readouterr().out
    assert "STALE a" in output
    assert "media: a.webm 0.00 MiB, a.webp 0.00 MiB" in output
    assert "STALE b" in output
    assert "2 stale, 0 current" in output


def test_media_sizes_use_committed_lfs_metadata_and_remain_report_only(
    tmp_path: Path, capsys
) -> None:
    repo = _repo(tmp_path)
    warning_path = repo / "docs/images/readme/gallery/a.webm"
    over_limit_path = repo / "docs/images/readme/gallery/b.webm"
    regular_path = repo / "docs/images/readme/gallery/a.webp"
    warning_path.write_text(_lfs_pointer(stale.GALLERY_MEDIA_WARNING_BYTES))
    over_limit_path.write_text(_lfs_pointer(stale.GALLERY_MEDIA_LIMIT_BYTES))
    regular_path.write_bytes(b"x" * 2048)
    _commit(repo, "grow gallery media", 22)

    warning_path.write_text(_lfs_pointer(1))

    report = stale.GalleryHistory(repo).report()
    sizes = {media.path.name: media.size_bytes for media in report.media}
    assert sizes["a.webm"] == stale.GALLERY_MEDIA_WARNING_BYTES
    assert sizes["a.webp"] == 2048
    assert sizes["b.webm"] == stale.GALLERY_MEDIA_LIMIT_BYTES

    assert stale.main(["--repo-root", str(repo)]) == 0
    output = capsys.readouterr().out
    assert "a.webm 20.00 MiB (20,971,520 bytes) [WARNING]" in output
    assert "b.webm 25.00 MiB (26,214,400 bytes) [OVER LIMIT]" in output
    assert "a.webp 0.00 MiB (2,048 bytes)" not in output
    assert output.index(
        "25.00 MiB (26,214,400 bytes)  b.webm [OVER LIMIT]"
    ) < output.index("20.00 MiB (20,971,520 bytes)  a.webm")
    assert "Gallery media: 45.00 MiB total across 4 files" in output
    assert (
        "Gallery media limits: 1 warning (20.00 MiB <= size < 25.00 MiB), "
        "1 over-limit file (>= 25.00 MiB)" in output
    )
    assert "Report only" in output


def test_bad_rows_are_reported_unknown_without_hiding_other_tiles(
    tmp_path: Path, capsys
) -> None:
    repo = _repo(tmp_path)
    (repo / "docs/images/readme/gallery/zzz.webp").write_text("unknown still\n")
    (repo / "docs/images/readme/gallery/zzz.webm").write_text("unknown video\n")
    _commit(repo, "add unmatched media", 22)

    assert stale.main(["--repo-root", str(repo)]) == 0
    output = capsys.readouterr().out
    assert (
        "UNKNOWN zzz: committed tile 'zzz' has no manifest entry; "
        "media: zzz.webm 0.00 MiB, zzz.webp 0.00 MiB" in output
    )
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
