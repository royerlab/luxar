"""Tests for the README live-demo count audit."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "audit_readme_demo_count.py"
SPEC = importlib.util.spec_from_file_location("audit_readme_demo_count", SCRIPT)
assert SPEC and SPEC.loader
audit_mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit_mod
SPEC.loader.exec_module(audit_mod)


PAGE = "\n".join(f'<a href="/viewer/index.html?src=x{i}">t</a>' for i in range(86))
README_OK = "Try it — 86 live demos, no install.\n| gallery | 86 demos as interactive scenes |\n"


def test_counts_tiles_by_viewer_link():
    assert audit_mod.count_tiles(PAGE) == 86
    assert audit_mod.count_tiles("<p>no tiles here</p>") == 0


def test_matching_counts_report_no_staleness():
    tiles, claims, stale = audit_mod.audit(PAGE, README_OK)
    assert tiles == 86
    assert len(claims) == 2
    assert stale == []


def test_partial_fix_is_caught():
    # the case that actually happened: two spots, one updated, one left behind
    readme = "Try it — 84 live demos, no install.\n| gallery | 86 demos as interactive scenes |\n"
    _, claims, stale = audit_mod.audit(PAGE, readme)
    assert len(claims) == 2
    assert stale == [(84, "intro banner")]


def test_claims_found_by_wording_not_line_number():
    reorganised = (
        "| gallery | 86 demos as interactive scenes |\n\n...\n\nlater: 86 live demos\n"
    )
    _, claims, stale = audit_mod.audit(PAGE, reorganised)
    assert {where for _, where in claims} == {"intro banner", "docs table row"}
    assert stale == []


def test_reworded_readme_reports_unverified_rather_than_passing(capsys, tmp_path):
    page = tmp_path / "index.html"
    page.write_text(PAGE)
    readme = tmp_path / "README.md"
    readme.write_text("The gallery hosts lots of demos.\n")
    assert audit_mod.main(["--page", str(page), "--readme", str(readme)]) == 0
    assert "NOT verified" in capsys.readouterr().out


def test_always_exits_zero_even_when_stale(capsys, tmp_path):
    page = tmp_path / "index.html"
    page.write_text(PAGE)
    readme = tmp_path / "README.md"
    readme.write_text("Try it — 12 live demos.\n")
    # report-only: a stale README must never block a deploy of correct data
    assert audit_mod.main(["--page", str(page), "--readme", str(readme)]) == 0
    assert "WARNING" in capsys.readouterr().out


def test_missing_page_is_skipped_not_fatal(capsys, tmp_path):
    assert audit_mod.main(["--page", str(tmp_path / "nope.html")]) == 0
    assert "skipped" in capsys.readouterr().out
