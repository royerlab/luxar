"""Tests for scripts/changelog_build.py (the changelog-fragment assembler)."""

from __future__ import annotations

import importlib.util
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
    monkeypatch.setattr("sys.argv", ["changelog_build.py", "--draft"])
    assert cb.main() == 0

    # Draft mutates nothing.
    assert (frag_dir / "7.md").exists()
    assert changelog.read_text(encoding="utf-8") == CHANGELOG_TEMPLATE
    assert "Draft entry" in capsys.readouterr().out
