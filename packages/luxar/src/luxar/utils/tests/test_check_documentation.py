"""Tests for scripts/check_documentation.py (the docs-quality ratchet).

Exercises the pure ratchet helpers, a small real scan over a temporary project
tree, and the ``main()`` CLI paths (via subprocess). The module under test is a
standalone repo script (not part of the installed ``luxar`` package), so it is
loaded from ``REPO_ROOT/scripts/check_documentation.py`` via importlib and the
whole module is skipped on a packaged install that ships no ``scripts/`` tree.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[6]
_SCRIPT = REPO_ROOT / "scripts" / "check_documentation.py"
_CHANGELOG_SCRIPT = REPO_ROOT / "scripts" / "changelog_build.py"

pytestmark = pytest.mark.skipif(
    not _SCRIPT.exists(),
    reason="generator script not present (packaged install without repo scripts/)",
)


def _load_module() -> ModuleType:
    """Load check_documentation.py from the repo scripts/ directory."""
    spec = importlib.util.spec_from_file_location("check_documentation", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cd = _load_module() if _SCRIPT.exists() else None


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_result_key_is_check_and_relative_posix_path(tmp_path: Path) -> None:
    root = tmp_path
    fp = root / "packages" / "luxar" / "src" / "luxar" / "core" / "foo.py"
    result = cd.CheckResult(
        passed=False,
        file_path=str(fp),
        check_name="Module docstring",
        message="Missing module docstring: core/foo.py",
    )
    key = cd.result_key(result, root)
    assert key == "Module docstring::packages/luxar/src/luxar/core/foo.py"


def test_result_key_independent_of_message(tmp_path: Path) -> None:
    root = tmp_path
    fp = root / "a" / "b.py"
    r1 = cd.CheckResult(False, str(fp), "Docstring coverage", "Low coverage (72%)")
    r2 = cd.CheckResult(False, str(fp), "Docstring coverage", "Low coverage (11%)")
    assert cd.result_key(r1, root) == cd.result_key(r2, root)


def test_result_key_includes_detail_when_set(tmp_path: Path) -> None:
    root = tmp_path
    fp = root / "packages" / "pkg" / "README.md"
    r1 = cd.CheckResult(
        False, str(fp), "Markdown path reference", "m", key_detail="src/a.ts"
    )
    r2 = cd.CheckResult(
        False, str(fp), "Markdown path reference", "m", key_detail="src/b.ts"
    )
    assert cd.result_key(r1, root) == (
        "Markdown path reference::packages/pkg/README.md::src/a.ts"
    )
    assert cd.result_key(r1, root) != cd.result_key(r2, root)


def test_failure_keys_only_failed(tmp_path: Path) -> None:
    root = tmp_path
    results = [
        cd.CheckResult(True, str(root / "ok.py"), "Module docstring", "ok"),
        cd.CheckResult(False, str(root / "bad.py"), "Module docstring", "bad"),
    ]
    keys = cd.failure_keys(results, root)
    assert keys == {"Module docstring::bad.py"}


def test_evaluate_ratchet_splits_correctly() -> None:
    current = {"a", "b", "c"}
    baseline = {"b", "c", "d"}
    new, fixed, still = cd.evaluate_ratchet(current, baseline)
    assert new == {"a"}
    assert fixed == {"d"}
    assert still == {"b", "c"}


# ---------------------------------------------------------------------------
# Baseline load / save
# ---------------------------------------------------------------------------


def test_load_baseline_missing_file_returns_empty(tmp_path: Path) -> None:
    assert cd.load_baseline(tmp_path / "does_not_exist.json") == set()


def test_save_baseline_is_sorted_and_valid(tmp_path: Path) -> None:
    path = tmp_path / "docs_baseline.json"
    keys = {"z::c.py", "a::b.py", "m::n.py"}
    cd.save_baseline(path, keys)

    text = path.read_text()
    assert text.endswith("\n")

    data = json.loads(text)
    assert "_comment" in data
    assert data["failures"] == sorted(keys)
    # Round-trips back to the same set.
    assert cd.load_baseline(path) == keys


def test_load_baseline_malformed_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{ not json ")
    with pytest.raises(ValueError):
        cd.load_baseline(path)

    path.write_text(json.dumps({"nope": []}))
    with pytest.raises(ValueError):
        cd.load_baseline(path)


# ---------------------------------------------------------------------------
# Integration: real scan over a tiny temp project tree
# ---------------------------------------------------------------------------


def _make_project(tmp_path: Path, *, with_defect: bool) -> Path:
    """Create a minimal project tree with one python package.

    When ``with_defect`` is True the module lacks a docstring (a NEW finding).
    """
    pkg = tmp_path / "packages" / "luxar" / "src" / "luxar" / "widgets"
    pkg.mkdir(parents=True)

    # A substantial, compliant README so the README checks pass.
    readme = pkg / "README.md"
    readme.write_text(
        "# Widgets\n\n"
        "## Quick Start\n\n"
        "```python\nimport widgets\n```\n\n" + ("Detailed prose about widgets. " * 40)
    )

    module = pkg / "gadget.py"
    if with_defect:
        # No module docstring -> a "Module docstring" finding.
        module.write_text("x = 1\n")
    else:
        module.write_text('"""Gadget module."""\n\nx = 1\n')

    return tmp_path


def _scan(module_root: Path):
    checker = cd.DocumentationChecker(module_root, verbose=False)
    checker.check_all()
    return checker


DEFECT_KEY = "Module docstring::packages/luxar/src/luxar/widgets/gadget.py"


def test_ratchet_new_finding_without_baseline(tmp_path: Path) -> None:
    root = _make_project(tmp_path, with_defect=True)
    checker = _scan(root)
    current = cd.failure_keys(checker.results, root)
    new, _, _ = cd.evaluate_ratchet(current, set())  # empty baseline
    assert DEFECT_KEY in current
    assert DEFECT_KEY in new


def test_ratchet_baselined_finding_tolerated(tmp_path: Path) -> None:
    root = _make_project(tmp_path, with_defect=True)
    checker = _scan(root)
    current = cd.failure_keys(checker.results, root)
    baseline = set(current)  # baseline everything currently failing
    new, _, still = cd.evaluate_ratchet(current, baseline)
    assert new == set()
    assert still == current


def test_ratchet_fixed_finding_detected(tmp_path: Path) -> None:
    # Baseline captured while the defect existed.
    root = _make_project(tmp_path, with_defect=True)
    baseline = cd.failure_keys(_scan(root).results, root)

    # Now fix the defect (add the module docstring).
    module = root / "packages" / "luxar" / "src" / "luxar" / "widgets" / "gadget.py"
    module.write_text('"""Gadget module."""\n\nx = 1\n')

    current = cd.failure_keys(_scan(root).results, root)
    new, fixed, _ = cd.evaluate_ratchet(current, baseline)
    assert new == set()
    assert DEFECT_KEY in fixed


# ---------------------------------------------------------------------------
# AST-based docstring checks (regression: text-heuristic false positives)
# ---------------------------------------------------------------------------


def _make_pkg(tmp_path: Path) -> Path:
    """Create an empty python package tree and return its directory."""
    pkg = tmp_path / "packages" / "luxar" / "src" / "luxar" / "widgets"
    pkg.mkdir(parents=True)
    return pkg


def test_shebang_before_module_docstring_not_flagged(tmp_path: Path) -> None:
    # A leading shebang used to defeat the ``.startswith('"""')`` heuristic even
    # though a real module docstring follows it.
    pkg = _make_pkg(tmp_path)
    module = pkg / "gadget.py"
    module.write_text(
        "#!/usr/bin/env python3\n"
        '"""Real docstring."""\n\n'
        "def alpha():\n"
        '    """Alpha."""\n'
        "    return 1\n\n\n"
        "def beta():\n"
        '    """Beta."""\n'
        "    return 2\n"
    )
    checker = _scan(tmp_path)
    keys = cd.failure_keys(checker.results, tmp_path)
    rel = "packages/luxar/src/luxar/widgets/gadget.py"
    assert f"Module docstring::{rel}" not in keys


def test_multiline_signature_with_docstring_counts_as_documented(
    tmp_path: Path,
) -> None:
    # A multi-line signature put the docstring beyond the 2-line lookahead, so
    # a fully-documented function was miscounted as undocumented.
    pkg = _make_pkg(tmp_path)
    module = pkg / "gadget.py"
    module.write_text(
        '"""Module."""\n\n'
        "def wide(\n"
        "    a,\n"
        "    b,\n"
        "    c,\n"
        "):\n"
        '    """Wide docstring."""\n'
        "    return a + b + c\n\n\n"
        "def narrow():\n"
        '    """Narrow docstring."""\n'
        "    return 0\n"
    )
    checker = _scan(tmp_path)
    keys = cd.failure_keys(checker.results, tmp_path)
    rel = "packages/luxar/src/luxar/widgets/gadget.py"
    assert f"Docstring coverage::{rel}" not in keys


def test_long_jsdoc_directly_above_counts_as_documented() -> None:
    # The TypeScript twin of the multi-line-signature bug above: a JSDoc block
    # longer than the 10-line lookback window put its `/**` opener out of range, so
    # a fully-documented export was miscounted. The metric then penalised thorough
    # docs — the cheapest way to raise a file's coverage was to write SHORTER ones.
    lines = (
        ["/**"]
        + [f" * line {i}" for i in range(20)]
        + [" */", "export type T = string;"]
    )
    assert cd.DocumentationChecker._has_jsdoc_above(lines, len(lines) - 1)


def test_blank_lines_between_jsdoc_and_export_are_skipped() -> None:
    lines = ["/**", " * Doc.", " */", "", "", "export interface I {}"]
    assert cd.DocumentationChecker._has_jsdoc_above(lines, len(lines) - 1)


def test_ten_line_window_still_accepted_so_the_rule_stays_a_superset() -> None:
    # The lookback window is kept OR-ed with the directly-above check, not replaced.
    # Requiring the immediately-preceding line to close a block comment alone is
    # stricter in the other direction and rejects three shapes the window accepts —
    # an export beneath the imports (crediting the MODULE docstring), a member
    # beneath a previous member's closing brace, and a `//`-style comment. Those are
    # pre-existing leniency in this metric; narrowing them is a separate decision,
    # so the rule must only ever REMOVE findings.
    brace = [
        "/**",
        " * Doc for something else.",
        " */",
        "  method() {}",
        "}",
        "export const c = 1;",
    ]
    assert cd.DocumentationChecker._has_jsdoc_above(brace, len(brace) - 1)

    line_comment = [
        "/**",
        " * Doc.",
        " */",
        "// Export type definitions",
        "export type T = 1;",
    ]
    assert cd.DocumentationChecker._has_jsdoc_above(line_comment, len(line_comment) - 1)


def test_a_plain_block_comment_does_not_count_as_jsdoc() -> None:
    # Condition 1 walks back from the closing */ to the block's opener and
    # requires /** — an ordinary /* ... */ implementation comment above an export
    # is not documentation, and the historical 10-line window never credited it
    # either (it greps for the JSDoc opener specifically). So rejecting it keeps
    # the rule a strict superset of the old one while closing the loophole where
    # any block comment would count.
    lines = ["/* not really jsdoc */", "export type T = string;"]
    assert not cd.DocumentationChecker._has_jsdoc_above(lines, len(lines) - 1)

    multiline = ["/*", " * plain block, not jsdoc", " */", "export type T = string;"]
    assert not cd.DocumentationChecker._has_jsdoc_above(multiline, len(multiline) - 1)

    # A trailing block comment on a CODE line opens on itself and fails the
    # JSDoc-opener requirement too.
    trailing = ["const x = 1; /* c */", "export type T = string;"]
    assert not cd.DocumentationChecker._has_jsdoc_above(trailing, len(trailing) - 1)

    # A one-line JSDoc still counts: the closer's own line is the opener.
    oneliner = ["/** Doc. */", "export type T = string;"]
    assert cd.DocumentationChecker._has_jsdoc_above(oneliner, len(oneliner) - 1)


def test_undocumented_export_is_still_undocumented() -> None:
    # Anti-vacuity: the two accepting conditions must not accept everything.
    lines = (
        ["import { x } from './x';"]
        + ["const y = 1;"] * 12
        + ["export type T = string;"]
    )
    assert not cd.DocumentationChecker._has_jsdoc_above(lines, len(lines) - 1)


def test_low_coverage_below_threshold_is_flagged(tmp_path: Path) -> None:
    # Module docstring present so ONLY the coverage check can fire; three
    # undocumented functions sit at 0% -> well below the 70% floor.
    pkg = _make_pkg(tmp_path)
    module = pkg / "gadget.py"
    module.write_text(
        '"""Module."""\n\n'
        "def alpha():\n    return 1\n\n\n"
        "def beta():\n    return 2\n\n\n"
        "def gamma():\n    return 3\n"
    )
    checker = _scan(tmp_path)
    keys = cd.failure_keys(checker.results, tmp_path)
    rel = "packages/luxar/src/luxar/widgets/gadget.py"
    assert f"Docstring coverage::{rel}" in keys


def test_overload_signatures_do_not_lower_docstring_coverage(tmp_path: Path) -> None:
    """Only the documented implementation counts, not its typing signatures."""
    pkg = _make_pkg(tmp_path)
    module = pkg / "gadget.py"
    module.write_text(
        '"""Module."""\n\n'
        "from typing import overload\n\n"
        "@overload\n"
        "def parse(value: int) -> int: ...\n\n"
        "@overload\n"
        "def parse(value: str) -> str: ...\n\n"
        "def parse(value: int | str) -> int | str:\n"
        '    """Return the parsed value."""\n'
        "    return value\n"
    )
    checker = _scan(tmp_path)
    keys = cd.failure_keys(checker.results, tmp_path)
    rel = "packages/luxar/src/luxar/widgets/gadget.py"
    assert f"Docstring coverage::{rel}" not in keys


def test_declared_non_utf8_source_is_parsed_not_reported(tmp_path: Path) -> None:
    # PEP 263: a source file declaring a non-UTF-8 encoding is valid Python.
    # Decoding as UTF-8 before parsing turned it into a bogus syntax finding.
    pkg = _make_pkg(tmp_path)
    module = pkg / "gadget.py"
    module.write_bytes(
        b"# coding: latin-1\n"
        b'"""Module d\xe9j\xe0 vu."""\n\n'
        b"def alpha():\n"
        b'    """Alpha."""\n'
        b"    return 1\n"
    )
    checker = _scan(tmp_path)
    keys = cd.failure_keys(checker.results, tmp_path)
    rel = "packages/luxar/src/luxar/widgets/gadget.py"
    assert f"Python syntax::{rel}" not in keys
    assert f"Module docstring::{rel}" not in keys


def test_syntax_error_file_reported_once_and_does_not_raise(
    tmp_path: Path,
) -> None:
    pkg = _make_pkg(tmp_path)
    module = pkg / "broken.py"
    module.write_text('"""Module."""\n\ndef (:\n    pass\n')

    checker = _scan(tmp_path)  # must not raise
    rel = "packages/luxar/src/luxar/widgets/broken.py"
    syntax = [
        r
        for r in checker.results
        if r.check_name == "Python syntax"
        and cd.result_key(r, tmp_path) == f"Python syntax::{rel}"
        and not r.passed
    ]
    assert len(syntax) == 1


# ---------------------------------------------------------------------------
# build_json_report structure (with and without a baseline)
# ---------------------------------------------------------------------------


def test_build_json_report_summary_and_sorted_findings(tmp_path: Path) -> None:
    root = _make_project(tmp_path, with_defect=True)
    checker = _scan(root)

    report = cd.build_json_report(checker.results, root)
    summary = report["summary"]
    assert summary["passed"] + summary["failed"] == summary["total"]
    assert summary["total"] == len(checker.results)
    assert summary["failed"] >= 1

    keys = [f["key"] for f in report["findings"]]
    assert keys == sorted(keys)
    # No baseline -> ratchet omitted (null).
    assert report["ratchet"] is None


def test_build_json_report_ratchet_classification(tmp_path: Path) -> None:
    root = _make_project(tmp_path, with_defect=True)
    checker = _scan(root)
    current = cd.failure_keys(checker.results, root)

    # Baseline that tolerates every current finding EXCEPT the defect, plus a
    # stale key that is no longer present (should classify as fixed).
    stale = "Module docstring::packages/luxar/src/luxar/gone/removed.py"
    baseline = (current - {DEFECT_KEY}) | {stale}

    report = cd.build_json_report(checker.results, root, baseline=baseline)
    ratchet = report["ratchet"]
    assert ratchet is not None
    assert DEFECT_KEY in ratchet["new"]
    assert stale in ratchet["fixed"]
    assert DEFECT_KEY not in ratchet["still_present"]
    assert stale not in ratchet["still_present"]
    # Each list is sorted.
    for field in ("new", "fixed", "still_present"):
        assert ratchet[field] == sorted(ratchet[field])


# ---------------------------------------------------------------------------
# Markdown path-reference check (real scan over a tiny temp GIT repo)
# ---------------------------------------------------------------------------


def _make_git_project(tmp_path: Path, files: dict[str, str]) -> Path:
    """Create ``files`` under ``tmp_path`` and ``git init + add`` them all."""
    for rel, content in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    return tmp_path


def _markdown_failures(root: Path) -> list[str]:
    checker = cd.DocumentationChecker(root, verbose=False)
    checker._check_markdown_path_references()
    return [
        r.message
        for r in checker.results
        if r.check_name == "Markdown path reference" and not r.passed
    ]


needs_git = pytest.mark.skipif(
    shutil.which("git") is None,
    reason="git not available",
)


@needs_git
def test_markdown_valid_references_pass(tmp_path: Path) -> None:
    root = _make_git_project(
        tmp_path,
        {
            "packages/pkg/README.md": (
                "See `./src/keep.ts`, `src/keep.ts`, and `../pkg/src/keep.ts`.\n"
            ),
            "packages/pkg/src/keep.ts": "export const x = 1;\n",
        },
    )
    assert _markdown_failures(root) == []


@needs_git
def test_markdown_broken_relative_reference_flagged(tmp_path: Path) -> None:
    root = _make_git_project(
        tmp_path,
        {
            "packages/pkg/README.md": "See `./src/gone.ts`.\n",
            "packages/pkg/src/keep.ts": "export const x = 1;\n",
        },
    )
    assert _markdown_failures(root) == [
        "README cites path that does not resolve: `./src/gone.ts`"
    ]


@needs_git
def test_markdown_repo_path_claim_with_deleted_basename_flagged(
    tmp_path: Path,
) -> None:
    # `src/...` is anchored in a tracked directory, so the stale reference is
    # flagged even though no file named removed-unique.ts exists anywhere.
    root = _make_git_project(
        tmp_path,
        {
            "packages/pkg/README.md": "See `src/data/removed-unique.ts`.\n",
            "packages/pkg/src/data/keep.ts": "export const x = 1;\n",
        },
    )
    assert _markdown_failures(root) == [
        "README cites path that does not resolve: `src/data/removed-unique.ts`"
    ]


@needs_git
def test_markdown_external_path_with_colliding_basename_not_flagged(
    tmp_path: Path,
) -> None:
    # `three/...` is not anchored in any tracked directory and only the
    # basename matches a tracked file — an external (npm dependency) path.
    root = _make_git_project(
        tmp_path,
        {
            "packages/pkg/README.md": "See `three/src/math/Vector3.ts`.\n",
            "packages/pkg/src/math/Vector3.ts": "export const x = 1;\n",
        },
    )
    assert _markdown_failures(root) == []


@needs_git
def test_markdown_npm_package_specifier_not_flagged(tmp_path: Path) -> None:
    # `mypkg/styles.css` is npm import syntax for the published package, even
    # though `packages/mypkg` is a tracked directory reachable from the anchor
    # walk.
    root = _make_git_project(
        tmp_path,
        {
            "packages/mypkg/examples/embed/README.md": (
                "Import `mypkg/styles.css` in the host page.\n"
            ),
            "packages/mypkg/src/keep.ts": "export const x = 1;\n",
        },
    )
    assert _markdown_failures(root) == []


@needs_git
def test_markdown_cross_package_shorthand_flagged(tmp_path: Path) -> None:
    # A shorthand that suffix-matches a file in ANOTHER package must be
    # spelled out fully, so it is flagged.
    root = _make_git_project(
        tmp_path,
        {
            "packages/pkg/README.md": "Padded on the write side (`io/ordering.py`).\n",
            "packages/other/src/other/io/ordering.py": "x = 1\n",
        },
    )
    assert _markdown_failures(root) == [
        "README cites path that does not resolve: `io/ordering.py`"
    ]


@needs_git
def test_markdown_new_broken_reference_in_baselined_readme_is_new(
    tmp_path: Path,
) -> None:
    # One broken reference is baselined; a SECOND broken reference added to the
    # SAME README must still classify as a NEW finding (per-token keys — a
    # baselined file must not become a blind spot).
    readme = "packages/pkg/README.md"
    root = _make_git_project(
        tmp_path,
        {
            readme: "See `./src/gone.ts`.\n",
            "packages/pkg/src/keep.ts": "export const x = 1;\n",
        },
    )
    checker = cd.DocumentationChecker(root, verbose=False)
    checker._check_markdown_path_references()
    baseline = cd.failure_keys(checker.results, root)
    assert len(baseline) == 1

    (root / readme).write_text("See `./src/gone.ts` and `./src/also-gone.ts`.\n")
    checker = cd.DocumentationChecker(root, verbose=False)
    checker._check_markdown_path_references()
    current = cd.failure_keys(checker.results, root)

    new, _, still = cd.evaluate_ratchet(current, baseline)
    assert still == baseline
    assert new == {f"Markdown path reference::{readme}::./src/also-gone.ts"}


@needs_git
def test_markdown_readme_deleted_from_worktree_skipped(tmp_path: Path) -> None:
    # Tracked in the index but deleted from the working tree: the scan must
    # skip it rather than crash.
    root = _make_git_project(
        tmp_path,
        {
            "packages/pkg/README.md": "See `./src/gone.ts`.\n",
            "packages/pkg/src/keep.ts": "export const x = 1;\n",
        },
    )
    (root / "packages/pkg/README.md").unlink()
    assert _markdown_failures(root) == []


@needs_git
def test_markdown_same_package_shorthand_resolves(tmp_path: Path) -> None:
    # The suffix fallback accepts shorthand refs within the README's own
    # package.
    root = _make_git_project(
        tmp_path,
        {
            "packages/pkg/README.md": "See `loaders/loader.ts`.\n",
            "packages/pkg/src/data/loaders/loader.ts": "export const x = 1;\n",
        },
    )
    assert _markdown_failures(root) == []


# ---------------------------------------------------------------------------
# CLI (subprocess): the main() exit-code + output contracts
# ---------------------------------------------------------------------------


def _install_script(root: Path) -> tuple[Path, Path]:
    """Copy the script into ``root/scripts`` so project_root resolves to root.

    Returns ``(script_copy, baseline_path)``.
    """
    scripts_dir = root / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    script_copy = scripts_dir / "check_documentation.py"
    script_copy.write_text(_SCRIPT.read_text())
    (scripts_dir / "changelog_build.py").write_text(_CHANGELOG_SCRIPT.read_text())
    return script_copy, scripts_dir / "docs_baseline.json"


def _run(script: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
    )


def test_docs_gate_rejects_every_invalid_changelog_fragment(tmp_path: Path) -> None:
    root = _make_project(tmp_path, with_defect=False)
    fragment_dir = root / "changelog.d"
    fragment_dir.mkdir()
    (fragment_dir / "1.md").write_text("### Fixed\n\n- Bullet.\n")
    (fragment_dir / "2.md").write_text("\n")
    script_copy, _ = _install_script(root)

    proc = _run(script_copy)

    assert proc.returncode == 1
    assert "2 invalid changelog fragments" in proc.stderr
    assert "fragment 1.md must start" in proc.stderr
    assert "fragment 2.md is empty" in proc.stderr

    json_proc = _run(script_copy, "--json")
    assert json_proc.returncode == 1
    report = json.loads(json_proc.stdout)
    failures = [item for item in report["findings"] if not item["passed"]]
    changelog_failure = next(
        item for item in failures if item["check_name"] == "Changelog fragment format"
    )
    assert "fragment 1.md must start" in changelog_failure["message"]
    assert "fragment 2.md is empty" in changelog_failure["message"]


def test_update_baseline_cli(tmp_path: Path) -> None:
    root = _make_project(tmp_path, with_defect=True)
    script_copy, baseline_path = _install_script(root)

    proc = _run(script_copy, "--update-baseline")
    assert proc.returncode == 0, proc.stderr
    assert baseline_path.exists()
    # The confirmation goes to stderr, not stdout (so `--json --update-baseline`
    # keeps a clean machine-readable stdout stream).
    assert "Wrote baseline" in proc.stderr
    assert "Wrote baseline" not in proc.stdout

    written = cd.load_baseline(baseline_path)
    expected = cd.failure_keys(_scan(root).results, root)
    assert written == expected

    data = json.loads(baseline_path.read_text())
    assert data["failures"] == sorted(data["failures"])
    assert DEFECT_KEY in written

    # `--json --update-baseline` must leave stdout empty (no confirmation, no
    # banner) so the stream stays machine-readable.
    proc = _run(script_copy, "--json", "--update-baseline")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""
    assert "Wrote baseline" in proc.stderr


def test_ratchet_mode_new_finding_fails_then_baselined_passes(tmp_path: Path) -> None:
    root = _make_project(tmp_path, with_defect=True)
    script_copy, _ = _install_script(root)

    # No baseline yet -> the defect is a NEW finding -> exit 1.
    proc = _run(script_copy)
    assert proc.returncode == 1, proc.stdout + proc.stderr

    # Baseline it, then the same finding is tolerated -> exit 0.
    assert _run(script_copy, "--update-baseline").returncode == 0
    proc = _run(script_copy)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_no_baseline_strict_mode_fails_on_any_finding(tmp_path: Path) -> None:
    root = _make_project(tmp_path, with_defect=True)
    script_copy, _ = _install_script(root)
    # Even with everything baselined, --no-baseline ignores the baseline.
    assert _run(script_copy, "--update-baseline").returncode == 0

    proc = _run(script_copy, "--no-baseline")
    assert proc.returncode == 1, proc.stdout + proc.stderr


def test_json_new_finding_no_baseline_exits_1_and_is_valid_json(tmp_path: Path) -> None:
    root = _make_project(tmp_path, with_defect=True)
    script_copy, _ = _install_script(root)

    proc = _run(script_copy, "--json")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    report = json.loads(proc.stdout)  # must parse
    assert report["ratchet"] is not None
    assert DEFECT_KEY in report["ratchet"]["new"]


def test_json_everything_baselined_exits_0(tmp_path: Path) -> None:
    root = _make_project(tmp_path, with_defect=True)
    script_copy, _ = _install_script(root)
    assert _run(script_copy, "--update-baseline").returncode == 0

    proc = _run(script_copy, "--json")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(proc.stdout)
    assert report["ratchet"]["new"] == []
    assert DEFECT_KEY in report["ratchet"]["still_present"]


def test_malformed_baseline_exits_2_cleanly(tmp_path: Path) -> None:
    root = _make_project(tmp_path, with_defect=True)
    script_copy, baseline_path = _install_script(root)
    baseline_path.write_text("{ not valid json <<<<<< merge conflict")

    # Human mode.
    proc = _run(script_copy)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "Traceback" not in proc.stderr

    # --json mode: same clean exit code, no traceback.
    proc = _run(script_copy, "--json")
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "Traceback" not in proc.stderr


# ---------------------------------------------------------------------------
# Scan depth (audit A11-01)
# ---------------------------------------------------------------------------
#
# The gate used a one-level `glob("*.py")` / `glob("*.ts")`, so it saw 219 of
# 1,012 Python modules and 146 of 819 TypeScript files — 78% and 82% of the
# source was never scanned, and the resulting 11-entry baseline recorded the
# absence of LOOKING rather than the absence of debt. These tests exist so that
# cannot silently come back.


def _make_nested_project(tmp_path: Path) -> Path:
    """A project whose only undocumented module is THREE directories deep."""
    pkg = tmp_path / "packages" / "luxar" / "src" / "luxar" / "widgets"
    deep = pkg / "fitting" / "dynamic_ops"
    deep.mkdir(parents=True)

    (pkg / "README.md").write_text(
        "# Widgets\n\n## Quick Start\n\n```python\nimport widgets\n```\n\n"
        + ("Detailed prose about widgets. " * 40)
    )
    # Compliant at the top level, so ONLY the nested file can produce a finding.
    (pkg / "gadget.py").write_text('"""Gadget module."""\n\nx = 1\n')
    (deep / "buried.py").write_text("x = 1\n")
    return tmp_path


NESTED_KEY = (
    "Module docstring::packages/luxar/src/luxar/widgets/fitting/dynamic_ops/buried.py"
)


def test_scan_recurses_into_nested_python_packages(tmp_path: Path) -> None:
    """A module three directories down must be scanned.

    This is the regression the audit found: with `glob` this file was invisible
    and the whole run reported zero findings.
    """
    checker = _scan(_make_nested_project(tmp_path))

    assert NESTED_KEY in cd.failure_keys(checker.results, tmp_path)


def test_nested_scan_is_a_strict_superset_of_the_one_level_scan(
    tmp_path: Path,
) -> None:
    """Recursing must ADD coverage, never move it.

    Guards the plausible mistake of "fixing" the depth by rewriting the loop in
    a way that changes which top-level files are visited.
    """
    root = _make_nested_project(tmp_path)
    # Make the TOP-level module defective too, so both depths yield a finding.
    top = root / "packages" / "luxar" / "src" / "luxar" / "widgets" / "gadget.py"
    top.write_text("x = 1\n")

    keys = cd.failure_keys(_scan(root).results, root)

    assert "Module docstring::packages/luxar/src/luxar/widgets/gadget.py" in keys
    assert NESTED_KEY in keys


def test_nested_typescript_is_scanned(tmp_path: Path) -> None:
    """The viewer half of the same fix."""
    deep = (
        tmp_path / "packages" / "luxar-viewer" / "src" / "data" / "loaders" / "spatial"
    )
    deep.mkdir(parents=True)
    (tmp_path / "packages" / "luxar-viewer" / "src" / "data" / "README.md").write_text(
        "# Data\n"
    )
    # Enough undocumented declarations to land under the 70% JSDoc threshold.
    (deep / "buried.ts").write_text(
        "\n".join(f"export function undocumented{i}(): void {{}}" for i in range(5))
        + "\n"
    )

    keys = cd.failure_keys(_scan(tmp_path).results, tmp_path)

    assert any("data/loaders/spatial/buried.ts" in key for key in keys)


def test_test_directories_are_excluded_from_coverage(tmp_path: Path) -> None:
    """A `tests/` module is not held to the API-documentation threshold."""
    root = _make_nested_project(tmp_path)
    pkg = root / "packages" / "luxar" / "src" / "luxar" / "widgets"
    tests_dir = pkg / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_gadget.py").write_text("x = 1\n")
    top_level_tests = root / "packages" / "luxar" / "src" / "luxar" / "tests"
    top_level_tests.mkdir()
    (top_level_tests / "test_bad.py").write_text("x = 1\n")
    viewer_tests = root / "packages" / "luxar-viewer" / "src" / "tests"
    viewer_tests.mkdir(parents=True)
    (viewer_tests / "bad.ts").write_text(
        "\n".join(f"export function undocumented{i}(): void {{}}" for i in range(5))
        + "\n"
    )

    keys = cd.failure_keys(_scan(root).results, root)

    assert not [key for key in keys if "/widgets/tests/" in key]
    assert not [
        key
        for key in keys
        if key.startswith("Module docstring::") and "/luxar/tests/" in key
    ]
    assert not [
        key
        for key in keys
        if key.startswith("JSDoc coverage::") and "/src/tests/" in key
    ]
    # ...and the exclusion is narrow: the nested non-test file still fails.
    assert NESTED_KEY in keys


def test_docs_gate_exclusions_stay_narrow() -> None:
    """Pin the exclusion set exactly.

    An exclusion is the cheapest possible way to reintroduce the blindness this
    change removed — adding one directory name here can silently un-scan a whole
    subtree — so widening it must be a deliberate, reviewed edit to this list.
    """
    assert cd.UNDOCUMENTED_DIR_NAMES == frozenset({"tests", "__pycache__"})
    assert cd.UNDOCUMENTED_TS_SUFFIXES == (".test.ts", ".spec.ts", ".d.ts")


@pytest.mark.parametrize(
    ("relpath", "excluded"),
    [
        ("tests/test_a.py", True),
        ("sub/tests/test_a.py", True),
        ("__pycache__/a.py", True),
        # Whole path COMPONENTS only. A name that merely contains "tests", or a
        # test-looking FILE outside a tests directory, stays in scope — matching
        # on substrings would quietly exempt real source.
        ("latests/a.py", False),
        ("tests_helpers/a.py", False),
        ("sub/test_helpers.py", False),
        ("sub/a.py", False),
    ],
)
def test_is_excluded_path_matches_whole_components(
    relpath: str, excluded: bool
) -> None:
    """The exclusion is by directory component, never by substring."""
    assert cd.is_excluded_path(Path(relpath)) is excluded


def test_committed_baseline_reflects_a_recursive_scan() -> None:
    """The real baseline must contain findings only a recursive scan can produce.

    Reverting to `glob` would drop every nested key. Those would be reported as
    "fixed" — ADVISORY, exit 0 — so the ratchet itself would not go red. This
    asserts the depth directly rather than trusting that path.
    """
    baseline = cd.load_baseline(REPO_ROOT / cd.DEFAULT_BASELINE_RELPATH)
    assert baseline, "the committed documentation baseline is missing or empty"

    def depth_below_package(key: str) -> int:
        """Directory levels between a package root and the file."""
        path = key.split("::")[1]
        for root in ("src/luxar/", "luxar-viewer/src/"):
            if root in path:
                return path.split(root, 1)[1].count("/")
        return 0

    assert any(depth_below_package(key) >= 2 for key in baseline), (
        "No baselined finding lies more than one directory below a package root, "
        "which is what a non-recursive scan produces. The gate has regressed to "
        "`glob`, or the baseline was regenerated by a build that had."
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
