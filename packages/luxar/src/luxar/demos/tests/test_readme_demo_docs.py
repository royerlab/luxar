"""Guard the README's ``luxar demo`` docs against the live registry / CLI.

Issue #718: the README claimed ``luxar demo run 1`` equaled ``luxar demo run
lorenz`` (index 1 is actually ``arxiv_papers``), carried stale sample-table
rows, and described ``run-all`` skips inaccurately. These checks fail if the
cited demo keys, the illustrative sample-table rows, or the ``run-all``
download-cap wording drift from what the registry / CLI report.
"""

from __future__ import annotations

import inspect
import re

import pytest

from luxar.cli import demo_commands
from luxar.cli.demo_commands import _needs_glyphs
from luxar.demos.registry import iter_demos
from luxar.utils.paths import get_project_root


def _readme_text() -> str:
    try:
        readme = get_project_root() / "README.md"
    except RuntimeError:  # pragma: no cover - only in an installed wheel
        pytest.skip("project root unavailable (installed wheel)")
    if not readme.exists():  # pragma: no cover
        pytest.skip("README.md not found at project root")
    return readme.read_text(encoding="utf-8")


def _sample_rows(text: str) -> list[tuple[int, str, str, str, str]]:
    """Parse the illustrative ``luxar demo`` sample table.

    The fenced block runs from the ``🎬 [Luxar]`` banner to the ``Run one:``
    footer; each data row is ``index key geom category needs [status]``.
    """
    block = re.search(r"🎬 \[Luxar\].*?\n(.*?)\nRun one:", text, re.DOTALL)
    assert block, "illustrative `luxar demo` sample table not found in README"
    rows: list[tuple[int, str, str, str, str]] = []
    for line in block.group(1).splitlines():
        m = re.match(r"\s*(\d+)\s+([A-Za-z0-9_]+)\s+(\S+)\s+(\S+)\s+(.*)$", line)
        if m:
            idx, key, geom, cat, tail = m.groups()
            rows.append((int(idx), key, geom, cat, tail))
    return rows


# STATUS words that may trail the NEEDS column in the illustrative table.
_STATUS_WORDS = ("output ✓", "cached", "")


def _needs_from_tail(tail: str) -> str:
    """Strip the trailing STATUS word to isolate the NEEDS column of a row."""
    stripped = tail.strip()
    for status in _STATUS_WORDS:
        if status and stripped.endswith(status):
            return stripped[: len(stripped) - len(status)].strip()
    return stripped


def test_sample_table_rows_match_registry() -> None:
    by_key = {r.key: r for r in iter_demos()}
    rows = _sample_rows(_readme_text())
    # The illustrative table shows five numbered rows; a dropped/corrupted row
    # (bad index, hyphenated key) fails the row regex and shrinks this count.
    assert len(rows) >= 5, f"expected >=5 sample rows, parsed {len(rows)}"
    for idx, key, geom, cat, tail in rows:
        assert key in by_key, f"README sample row cites unknown demo key {key!r}"
        r = by_key[key]
        assert idx == r.index, f"{key}: README index {idx} != registry {r.index}"
        assert geom == r.geometry, f"{key}: geometry {geom!r} != {r.geometry!r}"
        assert cat == r.category, f"{key}: category {cat!r} != {r.category!r}"
        # Exact NEEDS match (not a prefix) so a row cannot over- or under-claim
        # requirements and stay green.
        assert _needs_from_tail(tail) == _needs_glyphs(r), (
            f"{key}: README NEEDS {_needs_from_tail(tail)!r} != {_needs_glyphs(r)!r}"
        )


def test_cited_demo_keys_exist() -> None:
    text = _readme_text()
    keys = {r.key for r in iter_demos()}
    # Include '-' in the capture so a malformed citation like `run lorenz-bad`
    # is captured whole and flagged as unknown, rather than matching only its
    # valid `lorenz` prefix and slipping through.
    cited = set(re.findall(r"luxar demo (?:run|info) ([a-z][a-z0-9_-]*)", text))
    assert cited, "expected at least one cited demo key in the README"
    unknown = sorted(k for k in cited if k not in keys)
    assert not unknown, f"README cites demo keys absent from the registry: {unknown}"
    # Indices shift as demos are added, so a hard-coded numeric `run <N>` example
    # would drift (issue #718: "run 1" was wrongly equated with "run lorenz").
    assert not re.search(r"luxar demo (?:run|info) \d", text), (
        "README must not hard-code a numeric demo index in a run/info example"
    )


def test_demo_count_matches_registry() -> None:
    # The banner and the quick-start blurb both hard-code the demo count; adding
    # a demo whose key sorts late leaves the sampled indices intact, so guard the
    # count explicitly (the same staleness class as issue #718).
    text = _readme_text()
    n = len(iter_demos())
    assert f"[Luxar] {n} demos" in text, f"README sample banner should read {n} demos"
    assert f"{n} bundled demos" in text, (
        f"README quick-start should say {n} bundled demos"
    )


def test_run_all_wording_matches_cli() -> None:
    text = _readme_text()
    # Scope the checks to the `run-all` command-table row so unrelated prose
    # elsewhere can neither satisfy nor break them.
    row = next(
        (ln for ln in text.splitlines() if "`luxar demo run-all`" in ln),
        None,
    )
    assert row, "README command table is missing the `luxar demo run-all` row"
    sig = inspect.signature(demo_commands.demo_run_all)
    opt = sig.parameters["max_download_mb"].default
    cap = getattr(opt, "default", opt)
    assert f"{cap} MB" in row, (
        f"run-all row should cite the {cap} MB default download cap"
    )
    # Any other run-all mention (e.g. the gallery "Reproducing these locally"
    # block) that hard-codes a MB figure must cite the same default.
    for ln in text.splitlines():
        if "run-all" not in ln:
            continue
        for mb in re.findall(r"(\d+)\s*MB", ln):
            assert int(mb) == cap, (
                f"run-all mention cites {mb} MB but the CLI default is {cap} MB: {ln!r}"
            )
    # Manual/Kaggle data is the UNCONDITIONAL skip — it must be the class tied to
    # "always skipped" (issue #718: the old wording wrongly implied it was lifted
    # "unless told otherwise", and a later mis-edit could invert the two classes).
    assert re.search(r"[Mm]anual/Kaggle[^.]*always skipped", row), (
        "run-all row should state manual/Kaggle data is always skipped"
    )
    assert "unless told otherwise" not in row
    # The configurable skips are lifted by these CLI flags — all three must be
    # documented, and each must still be a real flag on the command (derived
    # from the Typer declarations, so a CLI rename fails this test instead of
    # leaving the README stale).
    cli_flags = {
        flag
        for p in sig.parameters.values()
        for decl in getattr(p.default, "param_decls", ()) or ()
        for flag in decl.split("/")
    }
    for flag in ("--include-gpu", "--max-download-mb", "--force"):
        assert flag in cli_flags, f"{flag} is no longer a run-all CLI flag"
        assert flag in row, f"run-all row should document the {flag} override"
