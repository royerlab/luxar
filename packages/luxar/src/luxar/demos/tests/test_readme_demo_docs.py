"""Guard the README's ``luxar demo`` docs against the live registry / CLI.

Issue #718: the README claimed ``luxar demo run 1`` equaled ``luxar demo run
lorenz`` (index 1 is actually ``arxiv_papers_kaggle``), carried stale sample-table
rows, and described ``run-all`` skips inaccurately. These checks fail if the
cited demo keys, the illustrative sample-table rows, or the ``run-all``
download-cap wording drift from what the registry / CLI report.
"""

from __future__ import annotations

import difflib
import inspect
import re

import pytest
from rich.console import Console

from luxar.cli import demo_commands
from luxar.cli.demo_commands import _starter_key
from luxar.cli.demo_render import STATUS_BUILT, STATUS_CACHED, render_catalogue
from luxar.demos import registry
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


def _demo_catalogue_text() -> str:
    try:
        readme = get_project_root() / "packages/luxar/src/luxar/demos/README.md"
    except RuntimeError:  # pragma: no cover - only in an installed wheel
        pytest.skip("project root unavailable (installed wheel)")
    if not readme.exists():  # pragma: no cover
        pytest.skip("demo catalogue README.md not found at project root")
    return readme.read_text(encoding="utf-8")


def _demo_catalogue_sections() -> dict[str, str]:
    text = _demo_catalogue_text()
    demo_keys = {demo.path.stem: demo.key for demo in iter_demos()}
    return {
        demo_keys[match.group(1)]: match.group(0)
        for match in re.finditer(
            r"^#### (demo_[a-z0-9_]+)\.py\b.*?(?=^#### |^## |\Z)",
            text,
            re.MULTILINE | re.DOTALL,
        )
    }


def _sample_block(text: str) -> list[str]:
    """The fenced sample listing, from the banner to the last footer line."""
    block = re.search(
        r"```\n(\U0001F3AC \d+ Luxar demos.*?\n Stop  [^\n]*)\n```", text, re.DOTALL
    )
    assert block, "illustrative `luxar demo` sample table not found in README"
    return block.group(1).splitlines()


def _renderer_lines() -> set[str]:
    """Every line the real catalogue can emit, over all three rail states.

    The rail is per-machine, so a README row is accepted if it matches under
    ANY uniform status assignment. Everything else — section rules and their
    counts, column alignment, requirement words, legend, footer — is identical
    across the three, so this stays an exact check on all of it.
    """
    demos = iter_demos()
    lines: set[str] = set()
    for status in (STATUS_BUILT, STATUS_CACHED, ""):
        console = Console(width=200, no_color=True, soft_wrap=True)
        with console.capture() as capture:
            render_catalogue(
                console,
                demos,
                {d.key: status for d in demos},
                example_key=_starter_key(demos),
            )
        lines |= {
            re.sub(r"\x1b\[[0-9;]*m", "", ln) for ln in capture.get().splitlines()
        }
    return lines


def test_sample_block_is_verbatim_renderer_output() -> None:
    """Every sample line must be a line the renderer actually produces.

    Stronger than parsing the rows field by field, and it catches the classes
    that field checks structurally cannot: a section rule citing an invented
    demo count, a demo filed under a category it is not in (the CATEGORY column
    is gone, so nothing else checks that association any more), and footer
    padding that drifts when the layout constants change.
    """
    real = _renderer_lines()
    checked = 0
    for line in _sample_block(_readme_text()):
        # The summary counts how much of the catalogue THIS machine has built,
        # so it is illustrative; `test_sample_summary_is_self_consistent` covers
        # it instead. " ..." marks the elided middle of the listing.
        if not line.strip() or line.startswith("\U0001f3ac") or line.strip() == "...":
            continue
        checked += 1
        assert line in real, (
            "README sample line is not renderer output:\n"
            f"  README: |{line}|\n"
            + "\n".join(
                f"  near  : |{n}|"
                for n in difflib.get_close_matches(line, real, n=1, cutoff=0.5)
            )
        )
    assert checked >= 10, f"only {checked} sample lines checked; block looks truncated"


def test_sample_summary_is_self_consistent() -> None:
    """The illustrative counts must at least add up to the real demo count."""
    summary = next(
        ln for ln in _sample_block(_readme_text()) if ln.startswith("\U0001f3ac")
    )
    total = int(re.search(r"(\d+) Luxar demos", summary).group(1))
    assert total == len(iter_demos()), f"sample banner says {total} demos"
    parts = [int(n) for n in re.findall(r"\u00b7\s+(\d+) ", summary)]
    assert sum(parts) == total, f"{parts} do not sum to {total}: {summary!r}"


def test_sample_table_shows_every_rail_state() -> None:
    """The sample must illustrate all three rail states, or the legend is moot."""
    rails = {
        ln[1]
        for ln in _sample_block(_readme_text())
        if re.match(r"^ [\u2713\u2022 ] *\d+  ", ln)
    }
    assert rails == {"\u2713", "\u2022", " "}, f"sample rails {rails} miss a state"


def test_documented_filter_vocabularies_match_the_schema() -> None:
    """`demo list -c/-g` advertises exactly the values the schema allows.

    These are hand-transcribed lists in a table, so they rot the moment a new
    category or geometry lands — `mesh` was already missing from the geometry
    row when this check was written.
    """
    text = _readme_text()
    for label, allowed in (
        ("Filter by category", registry.CATEGORY_VALUES),
        ("Filter by geometry", registry.GEOMETRY_VALUES),
    ):
        row = next(ln for ln in text.splitlines() if label in ln)
        listed = set(re.findall(r"`([a-z+]+)`", row.split(label, 1)[1]))
        assert listed == set(allowed), (
            f"{label}: README lists {sorted(listed)}, schema allows {sorted(allowed)}"
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


def test_catalogue_lfs_provisioning_matches_demo_metadata() -> None:
    """Only demos with a local Git-LFS input may advertise LFS provisioning."""
    sections = _demo_catalogue_sections()
    demos = {demo.key: demo for demo in iter_demos()}
    provisioning_claim = re.compile(
        r"(?:ships?|uses)[^.\n]*Git LFS|Git LFS data \(default\)|"
        r"git lfs pull|LFS asset",
        re.IGNORECASE,
    )
    documented = {
        key for key, section in sections.items() if provisioning_claim.search(section)
    }
    declared = {key for key, demo in demos.items() if demo.local_data == "git-lfs"}
    assert documented == declared, (
        "catalogue LFS provisioning claims disagree with DEMO_META.local_data: "
        f"documented={sorted(documented)}, declared={sorted(declared)}"
    )


def test_catalogue_sections_use_registry_keys_and_cover_every_entry() -> None:
    text = _demo_catalogue_text()
    demo_keys = {demo.path.stem: demo.key for demo in iter_demos()}
    expected = {
        demo_keys[match.group(1)]
        for match in re.finditer(r"^#### (demo_[a-z0-9_]+)\.py\b", text, re.MULTILINE)
    }
    assert set(_demo_catalogue_sections()) == expected


def test_gallery_tiles_match_registry_credits() -> None:
    gallery = _readme_text().split("## Gallery", 1)[1].split("\n---\n", 1)[0]
    tiles: dict[str, bool] = {}
    for line in gallery.splitlines():
        if "https://demos.luxarviewer.dev/d/" not in line:
            continue
        for cell in line.strip("|").split(" | "):
            match = re.search(
                r"https://demos\.luxarviewer\.dev/d/([a-z][a-z0-9_-]*)", cell
            )
            if match is None:
                continue
            key = match.group(1)
            assert key not in tiles, f"gallery repeats demo key {key}"
            tiles[key] = "<sub>" in cell

    assert len(tiles) == gallery.count("[!["), (
        "every gallery preview must have one linked demo title"
    )
    demos = {demo.key: demo for demo in iter_demos()}
    unknown = sorted(set(tiles) - set(demos))
    assert not unknown, f"gallery links demo keys absent from the registry: {unknown}"
    for key, has_credit in tiles.items():
        assert has_credit == (demos[key].citation is not None), (
            f"gallery credit for {key} does not match DEMO_META citation"
        )


def test_published_demo_count_agrees_across_readmes() -> None:
    root = get_project_root()
    readmes = (
        root / "README.md",
        root / "packages/luxar/README.md",
        root / "packages/luxar-viewer/README.md",
    )
    counts = {}
    for readme in readmes:
        matches = re.findall(r"— (\d+) (?:live )?demos", readme.read_text())
        assert len(matches) == 1, f"expected one published demo count in {readme}"
        counts[readme] = int(matches[0])
    assert len(set(counts.values())) == 1, f"published demo counts disagree: {counts}"


def test_demo_count_matches_registry() -> None:
    # The banner and the quick-start blurb both hard-code the demo count; adding
    # a demo whose key sorts late leaves the sampled indices intact, so guard the
    # count explicitly (the same staleness class as issue #718).
    text = _readme_text()
    n = len(iter_demos())
    assert f"🎬 {n} Luxar demos" in text, (
        "README sample banner demo count is stale — run `hatch run sync-demo-counts`"
    )
    assert f"{n} bundled demos" in text, (
        "README quick-start demo count is stale — run `hatch run sync-demo-counts`"
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
