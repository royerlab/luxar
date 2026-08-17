"""Tests for the `luxar demo` catalogue rendering (`luxar.cli.demo_render`)."""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import pytest
from rich.console import Console
from rich.text import Text

from luxar.cli import demo_render
from luxar.cli.demo_render import (
    STATUS_BUILT,
    STATUS_CACHED,
    format_download,
    needs_text,
    render_caches,
    render_catalogue,
    render_dependencies,
    render_detail,
)
from luxar.cli.utils import format_memory_size as _size
from luxar.demos import registry
from luxar.demos._dependencies import DependencySpec, DependencyStatus
from luxar.demos.registry import CacheEntry, DemoInfo, iter_demos

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _demo(
    key: str = "demo_key",
    *,
    index: int = 1,
    geometry: str = "points",
    download_mb: int = 0,
    gpu: str = "none",
    local_data: str | None = None,
    category: str = "synthetic",
) -> DemoInfo:
    return DemoInfo(
        key=key,
        index=index,
        module=f"luxar.demos.demo_{key}",
        path=Path(f"demo_{key}.py"),
        title=key.replace("_", " ").title(),
        description="A demo.",
        category=category,
        geometry=geometry,
        requirements={
            "compute": "light",
            "download_mb": download_mb,
            "gpu": gpu,
            "local_data": local_data,
        },
        caches=(),
        outputs=(key,),
    )


def _render(
    demos, statuses, width: int = 200, example_key: str | None = None
) -> list[str]:
    """Render a catalogue to plain lines, at a width that never soft-wraps."""
    console = Console(width=width, force_terminal=False, no_color=True, soft_wrap=True)
    with console.capture() as capture:
        render_catalogue(console, demos, statuses, example_key=example_key)
    return [_ANSI.sub("", line) for line in capture.get().splitlines()]


class TestFormatDownload:
    @pytest.mark.parametrize(
        ("megabytes", "expected"),
        [
            (0, ""),
            (-1, ""),
            (5, "5 MB"),
            (999, "999 MB"),
            # Past 1000 MB the reader wants gigabytes: 30000 MB is the raw
            # metadata for a ~29 GB dataset and is unreadable as printed.
            # 1000 is the LIVE boundary — `global_rivers_earth` records exactly
            # that — and it is the case that pins the cut-off at a round 1000
            # rather than at the 1024 the division uses.
            (1000, "1.0 GB"),
            (1024, "1.0 GB"),
            (1536, "1.5 GB"),
            (30000, "29 GB"),
        ],
    )
    def test_scales_to_a_readable_unit(self, megabytes: int, expected: str) -> None:
        assert format_download(megabytes) == expected

    def test_the_live_thousand_megabyte_demo_reads_as_gigabytes(self) -> None:
        """Pin the boundary to the demo that sits on it, not just to a number.

        A strict-1024 cut-off would print "1000 MB" here, which is exactly the
        unreadable form this function exists to translate. If that demo's size
        ever changes this test goes quiet rather than false — the parametrized
        1000 case above still guards the boundary itself.
        """
        on_the_boundary = [d for d in iter_demos() if d.download_mb == 1000]
        for demo in on_the_boundary:
            assert needs_text(demo).startswith("1.0 GB"), demo.key

    def test_one_decimal_only_below_ten_gigabytes(self) -> None:
        # A 182 GB download does not need a tenth of a gigabyte of precision,
        # and the extra characters come straight out of the row's width budget.
        assert format_download(186000) == "182 GB"


class TestNeedsText:
    def test_offline_cpu_demo_needs_nothing(self) -> None:
        assert needs_text(_demo()) == ""

    def test_orders_download_then_gpu_then_local_data(self) -> None:
        info = _demo(download_mb=150, gpu="optional", local_data="git-lfs")
        assert needs_text(info) == "150 MB GPU? git-lfs"

    @pytest.mark.parametrize(
        ("gpu", "expected"), [("required", "GPU"), ("optional", "GPU?")]
    )
    def test_gpu_requirement_is_distinguishable(self, gpu, expected) -> None:
        assert needs_text(_demo(gpu=gpu)) == expected

    @pytest.mark.parametrize(
        ("local", "expected"),
        [("git-lfs", "git-lfs"), ("kaggle-auth", "kaggle"), ("manual-file", "manual")],
    )
    def test_every_local_data_mode_has_a_word(self, local, expected) -> None:
        assert needs_text(_demo(local_data=local)) == expected

    def test_unknown_local_data_mode_is_dropped_not_crashed(self) -> None:
        # Metadata validation owns the vocabulary; the renderer must not be the
        # thing that raises if a new mode lands before it learns the word.
        assert needs_text(_demo(local_data="carrier-pigeon")) == ""


class TestCatalogue:
    def test_groups_rows_under_their_category(self) -> None:
        demos = [
            _demo("alpha", index=1, category="astronomy"),
            _demo("beta", index=2, category="synthetic"),
            _demo("gamma", index=3, category="astronomy"),
        ]
        lines = _render(demos, {})
        text = "\n".join(lines)
        astronomy = text.index("ASTRONOMY")
        synthetic = text.index("SYNTHETIC")
        # Categories are alphabetical, and each demo sits under its own.
        assert astronomy < text.index("alpha") < text.index("gamma") < synthetic
        assert synthetic < text.index("beta")

    def test_section_rule_states_the_group_size(self) -> None:
        demos = [_demo(f"d{i}", index=i, category="astronomy") for i in range(3)]
        rule = next(ln for ln in _render(demos, {}) if "ASTRONOMY" in ln)
        assert rule.endswith("3 demos")

    def test_single_demo_section_is_singular(self) -> None:
        rule = next(ln for ln in _render([_demo("solo")], {}) if "SYNTHETIC" in ln)
        assert rule.endswith("1 demo")

    def test_status_rail_marks_built_and_cached_rows(self) -> None:
        demos = [
            _demo("built_one", index=1),
            _demo("cached_one", index=2),
            _demo("fresh_one", index=3),
        ]
        statuses = {"built_one": STATUS_BUILT, "cached_one": STATUS_CACHED}
        rows = {
            key: next(ln for ln in _render(demos, statuses) if key in ln)
            for key in ("built_one", "cached_one", "fresh_one")
        }
        assert rows["built_one"].lstrip().startswith("✓")
        assert rows["cached_one"].lstrip().startswith("•")
        # Nothing built and nothing cached: the rail stays blank, so the row
        # still aligns with its neighbours.
        assert rows["fresh_one"].startswith("   ")
        assert "✓" not in rows["fresh_one"] and "•" not in rows["fresh_one"]

    def test_summary_omits_zero_counts(self) -> None:
        summary = _render([_demo("a")], {"a": STATUS_BUILT})[0]
        assert "1 built" in summary
        assert "0 cached" not in summary
        assert "0 not generated" not in summary

    def test_summary_counts_each_state(self) -> None:
        demos = [_demo(f"d{i}", index=i) for i in range(4)]
        statuses = {"d0": STATUS_BUILT, "d1": STATUS_BUILT, "d2": STATUS_CACHED}
        summary = _render(demos, statuses)[0]
        assert "4 Luxar demos" in summary
        assert "2 built" in summary and "1 cached" in summary
        assert "1 not generated yet" in summary

    def test_rows_carry_no_trailing_whitespace(self) -> None:
        # A demo with no requirements ends in the geometry column's padding
        # unless the row is stripped — invisible on screen, but real once the
        # listing is piped into a file or a diff.
        lines = _render([_demo("bare"), _demo("heavy", index=2, download_mb=5)], {})
        assert all(line == line.rstrip() for line in lines)

    def test_example_key_appears_in_the_run_hint(self) -> None:
        console = Console(width=200, no_color=True, soft_wrap=True)
        with console.capture() as capture:
            render_catalogue(console, [_demo("lorenz")], {}, example_key="lorenz")
        assert "e.g. luxar demo run lorenz" in capture.get()

    def test_footer_names_every_demo_entry_point(self) -> None:
        text = "\n".join(_render([_demo("a")], {}))
        for command in ("run", "info", "list", "deps"):
            assert f"luxar demo {command}" in text, f"footer omits `demo {command}`"

    def test_footer_keeps_the_actionable_half_of_every_pair(self) -> None:
        """Naming the report without naming the fix is the loss to guard against.

        The pre-redesign footer carried both halves of each pair; a tidier
        four-line replacement kept only the read-only commands, so nothing told
        you how to install a missing dependency, reclaim the gigabytes `cache
        list` had just measured, or why `stop` exists.
        """
        text = "\n".join(_render([_demo("a")], {}))
        for fragment in ("--install", "cache clear", "ports"):
            assert fragment in text, f"footer no longer mentions {fragment!r}"

    def test_legend_defines_every_glyph_the_rows_can_show(self) -> None:
        demos = [
            _demo("a", index=1, gpu="required"),
            _demo("b", index=2, gpu="optional"),
            _demo("c", index=3, local_data="git-lfs"),
        ]
        text = "\n".join(_render(demos, {"a": STATUS_BUILT, "b": STATUS_CACHED}))
        for term in ("✓ built", "• inputs cached", "not generated yet"):
            assert term in text
        for term in ("GPU", "GPU?", "git-lfs", "kaggle", "manual"):
            assert term in text


class TestKeysAreNeverTruncated:
    """Every demo key must appear IN FULL, whatever it costs in width.

    The counterpart to `test_no_row_exceeds_eighty_columns`, and deliberately
    in tension with it: the cheapest way to satisfy the width budget is to
    truncate the key column, and that silently breaks the one thing a row is
    for — the key is what you paste into `luxar demo run`. Fence from #637.
    """

    def test_every_bundled_key_appears_verbatim(self) -> None:
        demos = iter_demos()
        # An empty registry makes `missing` empty too, so this must assert that
        # it actually had keys to look for.
        assert len(demos) > 1, f"registry gave only {len(demos)} demos"
        rendered = "\n".join(
            _render(demos, {d.key: STATUS_BUILT for d in demos}, width=300)
        )
        missing = [d.key for d in demos if d.key not in rendered]
        assert not missing, f"keys truncated or dropped from the catalogue: {missing}"

    def test_an_absurdly_long_key_is_still_printed_whole(self) -> None:
        key = "a_demo_key_far_longer_than_any_terminal_would_ever_care_to_show"
        lines = _render([_demo(key, index=1)], {}, width=60)
        assert any(key in ln for ln in lines), "long key was truncated to fit"


class TestRealCatalogueFitsATerminal:
    def test_no_row_exceeds_eighty_columns(self) -> None:
        """The whole bundled catalogue must fit an 80-column terminal.

        The key column is sized to the longest demo key, so one long new key
        silently pushes every row past 80 and wraps the table. Rendering wide
        (so nothing soft-wraps) and measuring is the only way to catch that.

        Measured with ``Text.cell_len``, not ``len()``: the terminal wraps on
        display cells, and the summary line's 🎬 is one character but two cells,
        so ``len()`` under-measures exactly the line that could grow. The
        catalogue currently peaks at 80 with ZERO headroom, which is what makes
        the distinction worth getting right rather than academic.

        The footer's ``run`` hint carries a demo key too, so it is rendered here
        with the real suggestion rather than none: that line is
        ``48 + len("e.g. luxar demo run ") + len(key)`` wide, so an offline demo
        key of 12 characters or more sorting ahead of today's ``cloud`` overruns
        80 just as surely as a long key in the table does.
        """
        from luxar.cli.demo_commands import _starter_key

        demos = iter_demos()
        lines = _render(
            demos,
            {d.key: STATUS_BUILT for d in demos},
            width=300,
            example_key=_starter_key(demos),
        )
        # Guard the guard. `len(lines) > len(demos)` was NOT enough: with an
        # empty registry it degenerates to `lines > 0` and the whole check
        # passes on a catalogue that rendered nothing. Tie it to the rows.
        rows = [ln for ln in lines if re.match(r"^ [\u2713\u2022 ] *\d+  ", ln)]
        assert len(demos) > 1 and len(rows) == len(demos), (
            f"registry gave {len(demos)} demos, catalogue rendered {len(rows)} rows"
        )
        too_wide = [ln for ln in lines if Text(ln).cell_len > 80]
        assert not too_wide, (
            "these catalogue lines exceed 80 columns and will wrap: "
            + "; ".join(
                f"{Text(ln).cell_len}: {ln.strip()[:60]}" for ln in too_wide[:5]
            )
        )


class TestDetail:
    def _detail(self, info: DemoInfo, status: str = "") -> str:
        console = Console(width=200, no_color=True, soft_wrap=True)
        with console.capture() as capture:
            render_detail(console, info, status)
        return capture.get()

    def test_shows_the_run_command_for_the_demo(self) -> None:
        assert "luxar demo run lorenz" in self._detail(_demo("lorenz"))

    def test_offline_demo_says_so_rather_than_zero_megabytes(self) -> None:
        assert "none (offline)" in self._detail(_demo())

    def test_download_is_humanized(self) -> None:
        assert "29 GB" in self._detail(_demo(download_mb=30000))

    def test_status_is_spelled_out(self) -> None:
        assert "✓ built" in self._detail(_demo(), STATUS_BUILT)
        assert "• inputs cached" in self._detail(_demo(), STATUS_CACHED)
        assert "not generated yet" in self._detail(_demo(), "")

    def test_local_data_line_only_when_there_is_local_data(self) -> None:
        assert "Local data" not in self._detail(_demo())
        assert "Local data" in self._detail(_demo(local_data="kaggle-auth"))

    def test_outputs_line_only_when_the_demo_declares_outputs(self) -> None:
        """The empty-outputs branch: the default fixture always had one."""
        info = _demo()
        with_outputs = self._detail(info)
        assert "Outputs" in with_outputs

        bare = DemoInfo(**{**info.__dict__, "outputs": (), "caches": ()})
        text = self._detail(bare)
        assert "Outputs" not in text
        assert "Caches" not in text
        # ...and the rest of the record is still complete.
        for field in ("Category", "Geometry", "Status", "Run", "Module"):
            assert field in text

    def test_caches_line_names_every_cache_the_demo_claims(self) -> None:
        """The other side of the branch above; no fixture here declares a cache.

        Without it the caches line is only ever reached from
        `test_demo_commands.py` (via `demo info` on a demo that happens to have
        one), so this module's own suite would leave it unrendered.
        """
        info = DemoInfo(**{**_demo().__dict__, "caches": ("first", "second")})
        assert "Caches" in self._detail(info)
        assert "first, second" in self._detail(info)


class TestDependencies:
    def _render(self, rows) -> list[str]:
        console = Console(width=200, no_color=True, soft_wrap=True)
        with console.capture() as capture:
            render_dependencies(console, rows)
        return [_ANSI.sub("", ln) for ln in capture.get().splitlines()]

    @staticmethod
    def _row(module: str, spec: str, extra: str, installed: bool, satisfied: bool):
        return DependencyStatus(
            module, DependencySpec(spec, extra), installed, satisfied
        )

    def test_three_way_status_is_distinguishable(self) -> None:
        rows = [
            self._row("fine", "fine>=1", "demos", True, True),
            self._row("old", "old>=2", "demos", True, False),
            self._row("gone", "gone>=3", "demos", False, False),
        ]
        lines = self._render(rows)
        assert next(ln for ln in lines if "fine" in ln).endswith("ok")
        # Installed but below its pin is NOT the same problem as absent, and the
        # words must keep them apart — both need an install, only one is a
        # surprise on a machine that "already has it".
        assert next(ln for ln in lines if " old" in ln).endswith("OUTDATED")
        assert next(ln for ln in lines if "gone" in ln).endswith("MISSING")

    def test_specs_in_no_extra_show_a_dash(self) -> None:
        lines = self._render([self._row("gdown", "gdown", "", False, False)])
        assert "—" in next(ln for ln in lines if "gdown" in ln)

    @pytest.mark.parametrize(
        ("installed", "satisfied", "expected"),
        [
            (True, True, "ok"),
            (True, False, "OUTDATED"),
            # survey() computes `satisfied = installed and version_ok`, so this
            # fourth combination cannot arise there — but the verdict must stay
            # TOTAL anyway. A lookup table missing it turned a caller that
            # violated the invariant into a KeyError traceback out of the very
            # command you run when the environment is already suspect.
            (False, True, "ok"),
            (False, False, "MISSING"),
        ],
    )
    def test_verdict_is_total_over_every_flag_combination(
        self, installed: bool, satisfied: bool, expected: str
    ) -> None:
        rows = [self._row("mod", "mod>=1", "demos", installed, satisfied)]
        assert self._render(rows)[-1].endswith(expected)

    def test_columns_align_between_header_and_every_row(self) -> None:
        """A module shorter than "MODULE" must not shift its row left."""
        lines = self._render(
            [
                self._row("ab", "ab>=1", "demos", True, True),
                self._row("a_long_module", "a-long-spec>=1.2.3", "io", True, True),
            ]
        )
        header = next(ln for ln in lines if "MODULE" in ln)
        for extra in ("demos", "io"):
            row = next(ln for ln in lines if ln.rstrip().endswith("ok") and extra in ln)
            assert header.index("EXTRA") == row.index(extra), f"{header!r} vs {row!r}"
            assert header.index("STATUS") == row.index("ok"), f"{header!r} vs {row!r}"

    def test_columns_align_in_cells_not_code_points(self) -> None:
        """A wide character in a row must not shift the columns after it.

        The widths are measured in cells, so the cells have to be PADDED in
        cells too — `f"{value:<{width}}"` counts code points and puts the rest
        of the row one column out per wide character.
        """
        lines = self._render(
            [
                self._row("モジュ", "mod>=1", "demos", True, True),
                self._row("plain_module", "other>=1", "demos", True, True),
            ]
        )
        rows = [ln for ln in lines if ln.rstrip().endswith("ok")]
        assert len(rows) == 2
        offsets = {Text(r[: r.rindex("ok")]).cell_len for r in rows}
        assert len(offsets) == 1, f"STATUS column ragged at {sorted(offsets)}"

    def test_singular_heading_for_a_one_row_report(self) -> None:
        heading = self._render([self._row("ab", "ab>=1", "demos", True, True)])[0]
        assert heading.endswith("1 dependency")


class TestCaches:
    def _render(self, entries) -> list[str]:
        console = Console(width=200, no_color=True, soft_wrap=True)
        with console.capture() as capture:
            render_caches(console, entries, Path("/cache/root"), _size)
        return [_ANSI.sub("", ln) for ln in capture.get().splitlines()]

    @staticmethod
    def _entry(name: str, size: int, keys=(), protected: bool = False) -> CacheEntry:
        return CacheEntry(
            path=Path("/cache/root") / name,
            size_bytes=size,
            demo_keys=tuple(keys),
            protected=protected,
        )

    def test_heading_totals_every_entry(self) -> None:
        heading = self._render([self._entry("a", 100), self._entry("b", 50)])[0]
        assert "150.0 B in 2 dirs" in heading

    def test_single_entry_heading_is_singular(self) -> None:
        assert self._render([self._entry("a", 1)])[0].endswith("in 1 dir")

    def test_unclaimed_directory_is_flagged_orphan(self) -> None:
        lines = self._render([self._entry("stray", 10)])
        assert "ORPHAN" in next(ln for ln in lines if "stray" in ln)

    def test_protected_directory_is_never_called_an_orphan(self) -> None:
        # A hand-placed input has no download to get it back, so labelling it an
        # ORPHAN would invite exactly the `clear --orphans` that must not run.
        lines = self._render([self._entry("input", 10, protected=True)])
        row = next(ln for ln in lines if "input" in ln)
        assert "hand-placed input" in row
        assert "ORPHAN" not in row

    def test_claimed_directory_names_its_demos(self) -> None:
        lines = self._render([self._entry("shared", 10, keys=("one", "two"))])
        row = next(ln for ln in lines if "shared" in ln)
        assert "one, two" in row and "ORPHAN" not in row

    def test_cache_root_is_printed_once(self) -> None:
        lines = self._render([self._entry("a", 1)])
        assert sum(1 for ln in lines if ln.strip() == "/cache/root") == 1


class TestRuleBracketsItsTable:
    """Every listing's heading rule must span exactly its widest body line.

    Measured in TERMINAL CELLS, not code points. `🔒 hand-placed input` is 19
    code points and 20 columns, and the cache inventory originally sized its
    owner column from `demo_keys` alone — ignoring that the protected branch
    appends the lock label on top of them. The rule came out 13 columns short
    and stopped inside the table it is supposed to bracket, which is invisible
    on any row that happens to have no emoji in it.
    """

    @staticmethod
    def _check(render, args, is_body) -> None:
        """Assert the rule spans exactly the widest line ``is_body`` selects.

        Each listing needs its own body selector — the catalogue's legend and
        footer are not table rows, and measuring against them measures the
        wrong thing. The tables here are deliberately wider than their own
        heading text, so the fill is measured rather than clamped to its
        one-glyph minimum (which is correct behaviour, just not this test's
        subject).
        """
        console = Console(width=300, no_color=True, soft_wrap=True)
        with console.capture() as capture:
            render(console, *args)
        lines = [_ANSI.sub("", ln) for ln in capture.get().splitlines()]
        rule = next(ln for ln in lines if "\u2500" in ln)
        body = [ln for ln in lines if is_body(ln)]
        assert body, "body selector matched nothing - the check would be vacuous"
        assert rule.count("\u2500") > 1, "fill was clamped; widen the fixture table"
        # Text().cell_len, not len(): that difference IS the bug under test.
        widest = max(Text(b).cell_len for b in body)
        assert Text(rule).cell_len == widest, "rule %d != widest row %d\n%s" % (
            Text(rule).cell_len,
            widest,
            "\n".join("  |%s|" % ln for ln in lines),
        )

    def test_catalogue(self) -> None:
        demos = [
            _demo(
                "a_key_long_enough_to_outrun_the_heading",
                index=1,
                category="astronomy",
                download_mb=150,
                gpu="optional",
                local_data="git-lfs",
            ),
            _demo("short", index=22, category="astronomy", geometry="points+lines"),
        ]
        self._check(
            render_catalogue,
            (demos, {"short": STATUS_BUILT}),
            lambda ln: re.match(r"^ [\u2713\u2022 ] *\d+  ", ln) is not None,
        )

    def test_dependencies(self) -> None:
        rows = [
            DependencyStatus(
                "a_module",
                DependencySpec("a-spec>=1.2.3,<2,!=1.9,!=1.8,!=1.7", "demos"),
                True,
                True,
            )
        ]
        self._check(
            render_dependencies,
            (rows,),
            lambda ln: "MODULE" in ln or ln.rstrip().endswith(("ok", "OUTDATED")),
        )

    def test_caches_including_the_protected_row(self) -> None:
        entries = [
            CacheEntry(Path("/c/orphan_dir"), 1, (), False),
            # The widest row: keys AND the lock label - and the lock emoji makes
            # its cell width one greater than its code-point length.
            CacheEntry(Path("/c/locked"), 2, ("a_claimed_demo_key",), True),
            CacheEntry(Path("/c/claimed"), 3, ("a", "b"), False),
        ]
        self._check(
            render_caches,
            (entries, Path("/c"), _size),
            lambda ln: " B  " in ln,
        )

    def test_rule_matches_the_widest_row_over_many_column_shapes(self) -> None:
        """Sweep column shapes, not one hand-picked table.

        A single fixture cannot cover the case that actually broke: `_row` is
        rstripped, so a demo needing NOTHING loses its trailing gutter and its
        geometry padding, and a rule derived from column arithmetic overshot
        exactly those rows. It only shows up when the needs-less demo is also
        the one with the longest key.
        """
        geoms = ["points", "points+lines", "gsplats"]
        needs = [
            dict(),
            dict(download_mb=5),
            dict(download_mb=300000, gpu="required", local_data="git-lfs"),
        ]
        for gi, geom in enumerate(geoms):
            for ni, extra in enumerate(needs):
                demos = [
                    # The long-keyed demo is the one carrying `extra`, so each
                    # arm puts a different row in the "widest" position.
                    _demo("k" * 30, index=1, geometry=geom, **extra),
                    _demo("short", index=2, geometry="points", download_mb=7),
                ]
                self._check(
                    render_catalogue,
                    (demos, {"short": STATUS_BUILT}),
                    lambda ln: re.match(r"^ [\u2713\u2022 ] *\d+  ", ln) is not None,
                )

    #: Cache-inventory shapes whose widths are computed differently. `_KEY` is
    #: long enough that the table always outgrows its own heading, so the rule
    #: fill is measured rather than clamped.
    _KEY = "a_reasonably_long_demo_key_here"

    @pytest.mark.parametrize(
        ("entries", "size"),
        [
            # The ordinary case: NO orphan. Reserving room for the "⚠️  ORPHAN"
            # placeholder here put 9 columns of rule past the end of the table.
            pytest.param(
                [(("a_reasonably_long_demo_key_here",), False)],
                "1.0 KB",
                id="no-orphan",
            ),
            # The case that actually BINDS the phantom floor: a SHORT owner,
            # so reserving 10 cells for an ⚠️  ORPHAN that is not there
            # inflates the rule past the table. The long directory name keeps
            # the table wider than its own heading, so the fill stays measured.
            pytest.param([(("k",), False)], "1.0 KB", id="short-owner-no-orphan"),
            pytest.param(
                [((), False), (("a_reasonably_long_demo_key_here",), False)],
                "1.0 KB",
                id="with-an-orphan",
            ),
            pytest.param(
                [(("a_reasonably_long_demo_key_here",), True)],
                "1.0 KB",
                id="protected-lock-label",
            ),
            # `format_size` is caller-supplied and has no length contract, so a
            # hardcoded size-column width desynchronises row and rule.
            pytest.param(
                [(("a_reasonably_long_demo_key_here",), False)],
                "12345678 PB",
                id="size-longer-than-column",
            ),
            pytest.param(
                [(("a_reasonably_long_demo_key_here",), False)],
                "1125899906842624.0 PB",
                id="absurdly-long-size",
            ),
        ],
    )
    def test_caches_across_width_shapes(self, entries, size) -> None:
        made = [
            CacheEntry(Path(f"/c/some_cache_dir_name_{i}"), i + 1, keys, prot)
            for i, (keys, prot) in enumerate(entries)
        ]
        self._check(
            render_caches,
            (made, Path("/c"), lambda _n: size),
            lambda ln: "some_cache_dir_name" in ln,
        )

    @pytest.mark.parametrize("width", [40, 80, 120])
    def test_rule_never_exceeds_the_terminal(self, width: int) -> None:
        """A rule wider than the terminal wraps onto a second line of dashes.

        Rows may legitimately overrun (`cache list` reaches 151 columns when one
        directory is claimed by four demos, which is content, not layout). The
        RULE must not: it is decoration, and decoration that wraps is just
        noise. `min(width, console.width)` is what enforces that.
        """
        entries = [
            CacheEntry(
                Path("/c/a_cache_directory"),
                1,
                tuple(f"a_demo_key_{i}" for i in range(10)),
                False,
            )
        ]
        console = Console(width=width, no_color=True, soft_wrap=True)
        with console.capture() as capture:
            render_caches(console, entries, Path("/c"), _size)
        lines = [_ANSI.sub("", ln) for ln in capture.get().splitlines()]
        rule = next(ln for ln in lines if "\u2500" in ln)
        assert Text(rule).cell_len <= width, (
            f"rule is {Text(rule).cell_len} cells in a {width}-column terminal"
        )
        # ...and the row really does overrun, so this is not vacuous.
        assert max(Text(ln).cell_len for ln in lines) > width


class TestUnknownTerminalWidth:
    def test_columns_zero_does_not_erase_the_catalogue(self, monkeypatch) -> None:
        """`COLUMNS=0` means "unknown", not "zero columns wide".

        Rich reads it literally and renders every line to nothing, so the whole
        listing disappeared while the command still exited 0 — silent total
        loss for anything consuming the output.
        """
        monkeypatch.setenv("COLUMNS", "0")
        console = demo_render.demo_console()
        assert console.width > 0, "a zero-width console renders nothing at all"

    @pytest.mark.parametrize("columns", ["1", "5", "40", "200"])
    def test_a_real_width_is_still_honoured(self, monkeypatch, columns) -> None:
        """Only 0 is overridden; an absurd-but-real width is the user's call."""
        monkeypatch.setenv("COLUMNS", columns)
        assert demo_render.demo_console().width == int(columns)


class TestUnvalidatedTextStillAligns:
    """Cache directory names come off the FILESYSTEM, validated by nothing.

    `~/.cache/luxar/` can hold any name, and an orphan directory there is
    exactly what `cache list` exists to surface. Demo keys look like the safe
    case and are not quite: the schema's slug rule is `c.islower() or
    c.isdigit()`, which is Unicode-wide, so a fullwidth key passes validation
    and measures two cells per character.
    """

    @pytest.mark.parametrize(
        "exotic",
        [
            "\u6f14\u793a\u30c7\u30e2",  # wide CJK: 4 code points, 8 cells
            "caf\u00e9_dir",  # precomposed accent: 1:1, the control
            "a\u200bb\u200bc",  # zero-width spaces: 5 code points, 3 cells
        ],
    )
    def test_owner_column_aligns_whatever_the_directory_is_called(
        self, exotic: str
    ) -> None:
        entries = [
            CacheEntry(Path("/c") / exotic, 1, ("a_key",), False),
            CacheEntry(Path("/c/plain_cache_dir"), 2, ("b_key",), False),
        ]
        console = Console(width=300, no_color=True, soft_wrap=True)
        with console.capture() as capture:
            render_caches(console, entries, Path("/c"), _size)
        rows = [_ANSI.sub("", ln) for ln in capture.get().splitlines() if "_key" in ln]
        assert len(rows) == 2
        # Measured in cells: that is the whole point.
        offsets = {Text(r[: r.index("_key") - 1]).cell_len for r in rows}
        assert len(offsets) == 1, f"owner column ragged at {sorted(offsets)}"

    def test_catalogue_columns_align_for_a_wide_key(self) -> None:
        """A fullwidth key passes the slug rule, so the row must survive it."""
        wide = "ｃｌｏｕｄ"  # fullwidth "cloud": 5 chars, 10 cells
        assert all(c.islower() for c in wide), "fixture is not a valid demo key"
        demos = [_demo(wide, index=1), _demo("plain_key", index=2)]
        rows = [
            ln
            for ln in _render(demos, {}, width=300)
            if re.match(r"^ [✓• ] *\d+  ", ln)
        ]
        assert len(rows) == 2
        offsets = {Text(r[: r.index("points")]).cell_len for r in rows}
        assert len(offsets) == 1, f"geometry column ragged at {sorted(offsets)}"


class TestLegendMatchesTheRail:
    """The legend must teach the marks the rows actually print.

    It advertised "–" for the third state while the rail printed a space, so a
    reader scanning for a dash found none among 85 rows.
    """

    def test_legend_teaches_no_mark_the_rows_do_not_print(self) -> None:
        demos = [
            _demo("built_one", index=1),
            _demo("cached_one", index=2),
            _demo("fresh_one", index=3),
        ]
        statuses = {"built_one": STATUS_BUILT, "cached_one": STATUS_CACHED}
        lines = _render(demos, statuses)
        # Not "built and cached" — the SUMMARY line matches that too.
        legend = next(ln for ln in lines if "(blank)" in ln)
        rails = {ln[1] for ln in lines if re.match(r"^ [\u2713\u2022 ] *\d+  ", ln)}
        # Every non-blank mark the legend shows must be a mark some row emits.
        for mark in ("\u2713", "\u2022"):
            assert mark in legend and mark in rails
        # ...and the blank state is NAMED, not given an invented glyph.
        assert "(blank)" in legend
        assert "\u2013" not in legend, "legend shows a dash the rail never prints"

    def test_detail_view_may_still_use_a_dash(self) -> None:
        """There is no rail in the detail view, so a bare indent would be worse."""
        console = Console(width=120, no_color=True, soft_wrap=True)
        with console.capture() as capture:
            render_detail(console, _demo(), "")
        assert "\u2013 not generated yet" in capture.get()

    def test_the_three_states_are_named_identically_everywhere(self) -> None:
        """Legend, detail view and summary must not invent separate wordings."""
        for status, label in demo_render._STATUS_LABEL.items():
            assert label in demo_render._STATUS_WORD[status][0]


class TestDataIsNeverTreatedAsMarkup:
    """Demo metadata is DATA. Rich would read `[red]...[/]` in it as markup.

    `Text(...)` does not parse markup — only `Text.from_markup` and a bare
    `console.print("str")` do — so the safety here is structural. It is also
    one refactor away from being lost, which is what this pins.
    """

    def test_a_hostile_key_stays_literal_and_emits_no_escape(self) -> None:
        console = Console(width=200, force_terminal=True, soft_wrap=True)
        with console.capture() as capture:
            render_catalogue(console, [_demo("[red]evil")], {})
        out = capture.get()
        assert "[red]evil" in out, "markup was consumed instead of shown"
        # Ours are truecolor (38;2;...); a bare 31m could only come from the data.
        assert "\x1b[31m" not in out

    def test_no_render_path_prints_a_bare_string(self) -> None:
        """Every `console.print` argument must be a Text (or nothing)."""
        import ast
        import inspect

        source = inspect.getsource(demo_render)
        offenders = []
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "print"):
                continue
            for arg in node.args:
                if isinstance(arg, (ast.Constant, ast.JoinedStr)):
                    offenders.append(ast.unparse(node))
        assert not offenders, (
            "console.print given a raw string - Rich will parse markup in it: "
            + "; ".join(offenders)
        )


class TestNarrowTerminal:
    """No row content may disappear, however narrow the terminal gets.

    The layout is sized for 80 columns; below that the rows overrun. What must
    NOT happen is content vanishing. (Rich would word-wrap rather than crop
    even without `soft_wrap`; `soft_wrap` is what keeps a row on one line so
    the columns stay aligned. Both are checked here so a future change to
    `demo_console` cannot quietly trade one for the other.)
    """

    @staticmethod
    def _row_text(width: int) -> str:
        demos = [
            _demo(
                "a" * 38,
                index=1,
                download_mb=300000,
                gpu="required",
                local_data="git-lfs",
            ),
            _demo("b", index=85, geometry="points+lines", download_mb=500),
        ]
        console = Console(width=width, no_color=True, soft_wrap=True)
        with console.capture() as capture:
            render_catalogue(console, demos, {d.key: STATUS_BUILT for d in demos})
        plain = _ANSI.sub("", capture.get())
        # Rule fill is decorative and is deliberately capped at the terminal
        # width, so it must be excluded or it reads as content loss.
        body = "\n".join(ln for ln in plain.splitlines() if "\u2500" not in ln)
        return re.sub(r"\s+", " ", body).strip()

    @pytest.mark.parametrize("width", [20, 40, 60, 79, 80, 200])
    def test_no_content_is_lost_at_any_width(self, width: int) -> None:
        assert self._row_text(width) == self._row_text(400)

    def test_a_row_stays_on_one_line(self) -> None:
        """Soft wrap: one printed line per row, whatever the width."""
        demos = [_demo("k" * 38, index=1, download_mb=500, gpu="required")]
        console = Console(width=40, no_color=True, soft_wrap=True)
        with console.capture() as capture:
            render_catalogue(console, demos, {})
        rows = [
            ln for ln in _ANSI.sub("", capture.get()).splitlines() if "k" * 38 in ln
        ]
        assert len(rows) == 1, f"row was re-flowed onto {len(rows)} lines"


class TestVocabularyCoverage:
    """The renderer must know EVERY value DEMO_META is allowed to carry.

    `_LOCAL_LABEL.get()` degrades to a blank cell for an unknown mode, which is
    the right runtime behaviour but a silent one: a schema that grows a fourth
    provisioning mode would just stop mentioning it in the catalogue. These
    tests turn that silence into a failure.
    """

    def test_every_local_data_mode_has_a_label(self) -> None:
        schema = {v for v in registry.LOCAL_DATA_VALUES if v is not None}
        assert set(demo_render._LOCAL_LABEL) == schema, (
            "renderer's local-data labels drifted from the DEMO_META schema"
        )

    def test_every_gpu_value_renders_distinctly(self) -> None:
        rendered = {gpu: needs_text(_demo(gpu=gpu)) for gpu in registry.GPU_VALUES}
        assert len(set(rendered.values())) == len(rendered), (
            f"two GPU requirements render the same: {rendered}"
        )

    def test_every_status_has_both_a_glyph_and_a_word(self) -> None:
        assert set(demo_render._STATUS_GLYPH) == set(demo_render._STATUS_WORD)


def test_console_is_built_per_call() -> None:
    """Two calls must not share a console bound to a stale ``sys.stdout``.

    ``CliRunner`` swaps ``sys.stdout`` for the duration of a command; a console
    captured at import time would write past that capture and every CLI test
    asserting on demo output would see an empty string.

    The second assertion is the one that actually catches that: a fresh console
    is worthless if it pinned a file handle. ``Console.file`` resolves
    ``sys.stdout`` at access time when none was passed, so redirecting stdout
    must move the console's target with it.
    """
    assert demo_render.demo_console() is not demo_render.demo_console()

    original, replacement = sys.stdout, io.StringIO()
    assert demo_render.demo_console().file is original
    sys.stdout = replacement
    try:
        assert demo_render.demo_console().file is replacement
    finally:
        sys.stdout = original
