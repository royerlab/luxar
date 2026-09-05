"""Tests for scripts/changelog_build.py (the changelog-fragment assembler)."""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

_MOD_PATH = Path(__file__).resolve().parents[2] / "scripts" / "changelog_build.py"
_spec = importlib.util.spec_from_file_location("changelog_build", _MOD_PATH)
cb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cb)


CHANGELOG_TEMPLATE = """\
# Changelog

All notable changes to Luxar are documented in this file.

## [Unreleased]

### August 2026

#### An existing entry

Existing prose.

## Earlier History
"""


def test_fold_into_existing_month_prepends_newest_first():
    out = cb._fold(CHANGELOG_TEMPLATE, "August 2026", ["#### New entry\n\nNew prose."])
    # New entry appears under the August heading, ABOVE the existing one.
    aug = out.index("### August 2026")
    new = out.index("#### New entry")
    existing = out.index("#### An existing entry")
    assert aug < new < existing
    # Existing content is preserved.
    assert "Existing prose." in out
    assert "## Earlier History" in out


def test_fold_reuses_an_existing_month_that_is_not_the_newest():
    # Every month sits under [Unreleased], so a pinned `--month` may name a section
    # that is not the first one. It must be reused, not duplicated above the newer.
    changelog = CHANGELOG_TEMPLATE.replace(
        "### August 2026",
        "### September 2026\n\n#### A September entry\n\nSeptember prose.\n\n### August 2026",
    )
    out = cb._fold(changelog, "August 2026", ["#### Backfilled\n\nBackfilled prose."])
    assert out.count("### August 2026") == 1
    sep = out.index("### September 2026")
    aug = out.index("### August 2026")
    new = out.index("#### Backfilled")
    existing = out.index("#### An existing entry")
    assert sep < aug < new < existing


def test_fold_creates_missing_month_under_unreleased():
    out = cb._fold(
        CHANGELOG_TEMPLATE, "September 2026", ["#### Sep entry\n\nSep prose."]
    )
    unreleased = out.index("## [Unreleased]")
    sep = out.index("### September 2026")
    aug = out.index("### August 2026")
    # New month is created between Unreleased and the older August section.
    assert unreleased < sep < aug


def test_fold_requires_unreleased_header():
    with pytest.raises(SystemExit):
        cb._fold("# Changelog\n\nno unreleased here\n", "August 2026", ["#### x\n\ny"])


def test_read_block_rejects_non_heading(tmp_path):
    frag = tmp_path / "1.md"
    frag.write_text("just prose, no heading\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        cb._read_block(frag)


def test_read_block_accepts_heading(tmp_path):
    frag = tmp_path / "1.md"
    frag.write_text("#### Title\n\nBody.\n", encoding="utf-8")
    assert cb._read_block(frag).startswith("#### Title")


def test_natural_key_orders_numeric_before_slug_ascending(tmp_path):
    p9 = tmp_path / "9.md"
    p100 = tmp_path / "100.md"
    pslug = tmp_path / "aardvark.md"
    ordered = sorted([pslug, p100, p9], key=cb._natural_key)
    assert [p.name for p in ordered] == ["9.md", "100.md", "aardvark.md"]


def test_end_to_end_folds_and_deletes_fragments(tmp_path, monkeypatch):
    frag_dir = tmp_path / "changelog.d"
    frag_dir.mkdir()
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(CHANGELOG_TEMPLATE, encoding="utf-8")
    (frag_dir / "README.md").write_text("readme, not an entry\n", encoding="utf-8")
    (frag_dir / "42.md").write_text(
        "#### Folded entry\n\nFolded prose.\n", encoding="utf-8"
    )

    monkeypatch.setattr(cb, "FRAG_DIR", frag_dir)
    monkeypatch.setattr(cb, "CHANGELOG", changelog)
    monkeypatch.setattr("sys.argv", ["changelog_build.py", "--month", "August 2026"])
    assert cb.main() == 0

    text = changelog.read_text(encoding="utf-8")
    assert "#### Folded entry" in text
    # The fragment is consumed; the README is left alone.
    assert not (frag_dir / "42.md").exists()
    assert (frag_dir / "README.md").exists()


def test_draft_changes_nothing(tmp_path, monkeypatch, capsys):
    frag_dir = tmp_path / "changelog.d"
    frag_dir.mkdir()
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(CHANGELOG_TEMPLATE, encoding="utf-8")
    (frag_dir / "7.md").write_text(
        "#### Draft entry\n\nDraft prose.\n", encoding="utf-8"
    )

    monkeypatch.setattr(cb, "FRAG_DIR", frag_dir)
    monkeypatch.setattr(cb, "CHANGELOG", changelog)
    # `--month` is required here because the fragment lives in a tmp dir with
    # no git history: since the fold began filing entries under the month they
    # were WRITTEN, an undatable fragment fails rather than being mis-filed.
    # This test's subject is that a draft mutates nothing, not month derivation.
    monkeypatch.setattr(
        "sys.argv", ["changelog_build.py", "--draft", "--month", "August 2026"]
    )
    assert cb.main() == 0

    # Draft mutates nothing.
    assert (frag_dir / "7.md").exists()
    assert changelog.read_text(encoding="utf-8") == CHANGELOG_TEMPLATE
    assert "Draft entry" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Filing entries under the month they were WRITTEN, not the month of the fold.
# --------------------------------------------------------------------------

_GIT_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
}


def _repo_with_fragments(tmp_path: Path, dated: dict[str, str]) -> Path:
    """A throwaway git repo whose fragments were committed on the given dates."""
    env = {**os.environ, **_GIT_ENV, "HOME": str(tmp_path)}

    def git(*args: str, when: str | None = None) -> None:
        e = dict(env)
        if when:
            e["GIT_AUTHOR_DATE"] = e["GIT_COMMITTER_DATE"] = when
        subprocess.run(
            ["git", *args], cwd=tmp_path, check=True, capture_output=True, env=e
        )

    git("init", "-q")
    frag_dir = tmp_path / "changelog.d"
    frag_dir.mkdir()
    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG_TEMPLATE, encoding="utf-8")
    for name, when in dated.items():
        (frag_dir / name).write_text(f"#### Entry {name}\n\nProse.\n", encoding="utf-8")
        git("add", f"changelog.d/{name}")
        git("commit", "-q", "-m", name, when=when)
    return frag_dir


def test_fragments_are_filed_under_the_month_they_were_written(tmp_path, monkeypatch):
    """The defect this closes: every fragment landed under the CURRENT month.

    440 pending fragments spanned two months, and all 440 would have gone under
    whichever month the fold happened to run in — mis-dating 408 of them into a
    single unnavigable heading.
    """
    frag_dir = _repo_with_fragments(
        tmp_path,
        {
            "10.md": "2026-08-10T12:00:00",
            "20.md": "2026-08-20T12:00:00",
            "30.md": "2026-09-02T12:00:00",
        },
    )
    monkeypatch.setattr(cb, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cb, "FRAG_DIR", frag_dir)

    groups = cb._group_by_month(cb._fragments(), None)

    assert list(groups) == ["August 2026", "September 2026"], (
        "months must come out oldest-first, so the newest ends up on top"
    )
    assert [p.name for p in groups["August 2026"]] == ["10.md", "20.md"]
    assert [p.name for p in groups["September 2026"]] == ["30.md"]


def test_restored_fragment_keeps_its_original_authoring_month(tmp_path, monkeypatch):
    frag_dir = _repo_with_fragments(tmp_path, {"10.md": "2026-08-10T12:00:00"})
    env = {**os.environ, **_GIT_ENV, "HOME": str(tmp_path)}
    september_env = {
        **env,
        "GIT_AUTHOR_DATE": "2026-09-01T12:00:00",
        "GIT_COMMITTER_DATE": "2026-09-01T12:00:00",
    }
    subprocess.run(
        ["git", "rm", "-q", "changelog.d/10.md"],
        cwd=tmp_path,
        check=True,
        env=env,
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "remove"],
        cwd=tmp_path,
        check=True,
        env=september_env,
    )
    frag_dir.mkdir()
    (frag_dir / "10.md").write_text("#### Restored\n\nProse.\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "changelog.d/10.md"], cwd=tmp_path, check=True, env=env
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "restore"],
        cwd=tmp_path,
        check=True,
        env=september_env,
    )

    monkeypatch.setattr(cb, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cb, "FRAG_DIR", frag_dir)

    groups = cb._group_by_month(cb._fragments(), None)

    assert list(groups) == ["August 2026"]
    assert [p.name for p in groups["August 2026"]] == ["10.md"]


def test_an_undatable_fragment_fails_rather_than_being_mis_filed(tmp_path, monkeypatch):
    """Fail closed: guessing the month is the bug, so refuse to guess."""
    frag_dir = _repo_with_fragments(tmp_path, {"10.md": "2026-08-10T12:00:00"})
    # Never committed, so git cannot date it.
    (frag_dir / "99.md").write_text("#### Uncommitted\n\nProse.\n", encoding="utf-8")
    monkeypatch.setattr(cb, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cb, "FRAG_DIR", frag_dir)

    with pytest.raises(SystemExit, match="99.md"):
        cb._group_by_month(cb._fragments(), None)


def test_pinning_a_month_overrides_the_git_dates(tmp_path, monkeypatch):
    """``--month`` stays the explicit escape hatch, undatable files included."""
    frag_dir = _repo_with_fragments(tmp_path, {"10.md": "2026-08-10T12:00:00"})
    (frag_dir / "99.md").write_text("#### Uncommitted\n\nProse.\n", encoding="utf-8")
    monkeypatch.setattr(cb, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cb, "FRAG_DIR", frag_dir)

    groups = cb._group_by_month(cb._fragments(), "July 2026")
    assert list(groups) == ["July 2026"]
    assert len(groups["July 2026"]) == 2


def test_main_folds_every_authored_month(tmp_path, monkeypatch):
    frag_dir = _repo_with_fragments(
        tmp_path,
        {
            "10.md": "2026-08-10T12:00:00",
            "20.md": "2026-09-02T12:00:00",
        },
    )
    changelog = tmp_path / "CHANGELOG.md"
    monkeypatch.setattr(cb, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cb, "FRAG_DIR", frag_dir)
    monkeypatch.setattr(cb, "CHANGELOG", changelog)
    monkeypatch.setattr("sys.argv", ["changelog_build.py"])

    assert cb.main() == 0

    text = changelog.read_text(encoding="utf-8")
    assert text.index("### September 2026") < text.index("### August 2026")
    assert "#### Entry 10.md" in text
    assert "#### Entry 20.md" in text


# --------------------------------------------------------------------------
# Cutting [Unreleased] into a version section.
# --------------------------------------------------------------------------


def test_release_cut_renames_unreleased_and_opens_a_fresh_one():
    out = cb._cut_release(CHANGELOG_TEMPLATE, "2026.09.15", "2026-09-15")

    assert "## [2026.09.15] - 2026-09-15" in out
    # A fresh empty [Unreleased] ABOVE it, so the next PR has somewhere to land.
    assert out.index("## [Unreleased]") < out.index("## [2026.09.15]")
    assert "#### An existing entry" in out
    assert "### August 2026" in out
    assert "## Earlier History" in out


def test_release_cut_loses_no_line():
    """The cut is a rename plus an insert — never a deletion.

    Worth asserting directly: this runs once per release against a file that is
    hundreds of kilobytes of the project's only narrative history, and a silent
    truncation would be noticed long after the tag.
    """
    out = cb._cut_release(CHANGELOG_TEMPLATE, "2026.09.15", "2026-09-15")
    before = [ln for ln in CHANGELOG_TEMPLATE.splitlines() if ln.strip()]
    after = [ln for ln in out.splitlines() if ln.strip()]
    assert len(after) == len(before) + 1  # the re-added [Unreleased]
    for line in before:
        assert line in after, f"cut dropped: {line!r}"


def test_release_cut_requires_the_unreleased_header():
    with pytest.raises(SystemExit, match="Unreleased"):
        cb._cut_release("# Changelog\n\n## [1.0] - 2020-01-01\n", "2.0", "2026-09-15")


def test_release_refuses_while_fragments_are_pending(tmp_path, monkeypatch):
    frag_dir = tmp_path / "changelog.d"
    frag_dir.mkdir()
    (frag_dir / "7.md").write_text("#### Pending\n\nProse.\n", encoding="utf-8")
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(CHANGELOG_TEMPLATE, encoding="utf-8")
    init = tmp_path / "__init__.py"
    init.write_text('__version__ = "2026.09.15"\n', encoding="utf-8")
    monkeypatch.setattr(cb, "FRAG_DIR", frag_dir)
    monkeypatch.setattr(cb, "CHANGELOG", changelog)
    monkeypatch.setattr(cb, "INIT", init)
    monkeypatch.setattr("sys.argv", ["changelog_build.py", "--release"])

    with pytest.raises(SystemExit, match="pending"):
        cb.main()


def test_release_refuses_an_already_cut_version(tmp_path, monkeypatch):
    frag_dir = tmp_path / "changelog.d"
    frag_dir.mkdir()
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(
        CHANGELOG_TEMPLATE + "\n## [2026.09.15] - 2026-09-15\n", encoding="utf-8"
    )
    init = tmp_path / "__init__.py"
    init.write_text('__version__ = "2026.09.15"\n', encoding="utf-8")
    monkeypatch.setattr(cb, "FRAG_DIR", frag_dir)
    monkeypatch.setattr(cb, "CHANGELOG", changelog)
    monkeypatch.setattr(cb, "INIT", init)
    monkeypatch.setattr("sys.argv", ["changelog_build.py", "--release"])

    with pytest.raises(SystemExit, match="already has"):
        cb.main()


def test_release_draft_changes_nothing(tmp_path, monkeypatch):
    frag_dir = tmp_path / "changelog.d"
    frag_dir.mkdir()
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(CHANGELOG_TEMPLATE, encoding="utf-8")
    before = changelog.read_bytes()
    init = tmp_path / "__init__.py"
    init.write_text('__version__ = "2026.09.15"\n', encoding="utf-8")
    monkeypatch.setattr(cb, "FRAG_DIR", frag_dir)
    monkeypatch.setattr(cb, "CHANGELOG", changelog)
    monkeypatch.setattr(cb, "INIT", init)
    monkeypatch.setattr("sys.argv", ["changelog_build.py", "--release", "--draft"])

    assert cb.main() == 0
    assert changelog.read_bytes() == before


def test_current_version_reads_the_single_source_of_truth(tmp_path, monkeypatch):
    init = tmp_path / "__init__.py"
    init.write_text('x = 1\n__version__ = "2026.09.15"\ny = 2\n', encoding="utf-8")
    monkeypatch.setattr(cb, "INIT", init)
    assert cb._current_version() == "2026.09.15"


def test_current_version_fails_loudly_when_absent(tmp_path, monkeypatch):
    init = tmp_path / "__init__.py"
    init.write_text("# no version here\n", encoding="utf-8")
    monkeypatch.setattr(cb, "INIT", init)
    with pytest.raises(SystemExit, match="__version__"):
        cb._current_version()


def test_release_date_is_derived_from_a_calver_version():
    """The version IS the date, so the header must not contradict itself.

    The bump lands in a PR and the cut follows review, so "today" is usually a
    different day from the version — which would publish
    `## [2026.09.15] - 2026-09-04`.
    """
    assert cb._release_date("2026.09.15") == "2026-09-15"
    # PEP 440 strips leading zeros on install, so both spellings must work.
    assert cb._release_date("2026.9.5") == "2026-09-05"


def test_release_date_falls_back_to_today_for_a_non_date_version():
    import datetime

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    assert cb._release_date("1.2.3rc1") == today
    assert cb._release_date("main") == today
