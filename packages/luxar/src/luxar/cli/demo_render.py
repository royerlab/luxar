"""Presentation for the ``luxar demo`` catalogue views.

Pure rendering: everything here takes already-gathered facts and turns them into
terminal output. No filesystem probing, no registry lookups — those live in
:mod:`luxar.cli.demo_commands`, which passes the results in.

Two deliberate departures from the rest of the Python CLI:

* **Rich, not hand-built f-strings.** The catalogue is a real table with five
  columns and 85+ rows; Rich gives it colour, terminal-width awareness, and
  automatic plain-text degradation when the output is piped or ``NO_COLOR`` is
  set. Column widths are measured across the WHOLE list up front, so every
  category section lines up with every other one — a per-section Rich ``Table``
  would size its columns independently and stagger the sections.
* **Plain stdout, not** ``aprint``. Arbol prefixes every line with its tree
  glyph (``├``), which is right for nested progress output and wrong for a
  catalogue: it pushes the columns off the left margin and rides along when a
  row is copy-pasted.

The split is by what the output IS, not by which command prints it: a
**standalone inventory you read** — the catalogue, one demo's detail record,
the dependency report, the cache listing — renders here, while anything
**interleaved with an action** stays on arbol, where the tree glyph is doing
real work showing nesting. That is why ``demo deps`` is deliberately mixed (a
rich table, then arbol advice about installing), and why ``demo stop`` keeps its
running-demo list on arbol even though it is tabular: that list is the preamble
to killing something, not a page you sit and read.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.text import Text

from ..demos._dependencies import DependencyStatus
from ..demos.registry import CacheEntry, DemoInfo

#: ``_status`` results, also the status column's display order of preference.
STATUS_BUILT = "built"
STATUS_CACHED = "cached"

#: Gap between columns, in spaces.
_GUTTER = 2

#: Left margin, and the narrower gap after the one-character status rail. Both
#: are 1 rather than :data:`_GUTTER` to buy back the columns that keep the
#: widest row inside an 80-column terminal (the longest demo key is 38
#: characters, which is most of the budget on its own). A future key longer
#: than today's costs only a terminal wrap on the rows that overrun; no row is
#: ever truncated, and `test_no_row_exceeds_eighty_columns` says when it lands.
_INDENT = 1
_RAIL_GAP = 1


def _cells(text: str) -> int:
    """Width of ``text`` in terminal columns, which is NOT ``len(text)``.

    Every emoji the listings use (``🔒`` ``📦`` ``💾`` ``🎬``) is one code point
    and two columns, so a width computed with ``len`` lands short by one per
    emoji and the rule stops inside the table it is meant to bracket. Box and
    bullet glyphs (``─ · – ✓ •``) happen to be 1:1, and ``⚠️`` is 2:2 — which is
    exactly why the bug is easy to miss by eye on the rows that don't have an
    emoji in them.
    """
    return Text(text).cell_len


def _pad(text: str, width: int, *, right: bool = False) -> str:
    """Pad ``text`` to ``width`` TERMINAL COLUMNS, not code points.

    ``f"{text:<{width}}"`` counts code points, so it disagrees with
    :func:`_cells` on anything wide or combining. Cache directory names come off
    the filesystem and are validated by nothing, and a CJK directory there
    shifted the owner column four cells out of line with its neighbours. Demo
    keys look immune and are not quite: the schema spells its slug rule as
    ``c.islower() or c.isdigit()``, which is Unicode-wide, so a fullwidth
    ``ｄ`` passes validation and takes two cells. Every column therefore pads
    through here, not through a format spec.
    """
    fill = " " * max(0, width - _cells(text))
    return fill + text if right else text + fill


#: Local-data provisioning mode → the word shown in the NEEDS column.
_LOCAL_LABEL = {
    "git-lfs": "git-lfs",
    "kaggle-auth": "kaggle",
    "manual-file": "manual",
}

#: Status → the glyph in the left rail of a catalogue row. A one-character rail
#: rather than a trailing ``✓ built`` column: the words cost ten columns at the
#: far end of every row (enough to push the table past 80 and wrap it), while
#: the rail collects "what do I already have" into a single scannable stripe.
#: :data:`_STATUS_WORD` spells the same thing out where there is room for it.
_STATUS_GLYPH = {
    STATUS_BUILT: ("✓", "green"),
    STATUS_CACHED: ("•", "blue"),
    "": (" ", "dim"),
}

#: What each state is called. One source of truth for the legend, the detail
#: view and the summary, so the three cannot describe the same state differently.
_STATUS_LABEL = {
    STATUS_BUILT: "built",
    STATUS_CACHED: "inputs cached",
    "": "not generated yet",
}

#: Marker for the DETAIL view, which has no rail to be consistent with: there,
#: the blank state would render as a bare indent, so it takes a dash. The
#: catalogue's legend must NOT borrow this dash — it teaches the rail, and the
#: rail prints a space. Advertising a "–" the rows never emit sends the reader
#: hunting for a glyph that does not exist.
_DETAIL_MARK = {STATUS_BUILT: "✓", STATUS_CACHED: "•", "": "–"}

_STATUS_WORD = {
    status: (f"{_DETAIL_MARK[status]} {label}", _STATUS_GLYPH[status][1])
    for status, label in _STATUS_LABEL.items()
}


#: Width assumed when the environment reports none (``COLUMNS=0``). Matches
#: Rich's own default for a non-terminal stream.
_FALLBACK_WIDTH = 80


def demo_console() -> Console:
    """Console for catalogue output.

    ``soft_wrap`` emits each row as ONE line and leaves any overflow to the
    terminal. Without it Rich word-wraps an over-long row into a re-flowed
    block, which loses no characters but does break the column alignment the
    whole layout exists for: at 60 columns the trailing ``500 MB`` migrates to
    a continuation line and no longer sits under its neighbours' requirements.
    (Rich does not *crop* by default — that needs ``overflow="crop"`` — so the
    choice here is between wrapping styles, not between wrapping and data
    loss.) ``highlight=False`` turns off Rich's automatic number/path
    colouring, which would otherwise recolour parts of demo keys and fight the
    column styles.

    Built per call rather than at import: ``typer.testing.CliRunner`` swaps
    ``sys.stdout`` for the duration of a command, and a console bound at import
    time would write past the capture.
    """
    console = Console(soft_wrap=True, highlight=False)
    if console.width <= 0:
        # ``COLUMNS=0`` conventionally means "width unknown" — some CI runners
        # and terminal-less environments export exactly that. Rich takes it
        # literally and renders every line to nothing, so the whole catalogue
        # vanishes while the command still exits 0: silent total loss, the worst
        # outcome for anything reading the listing. Any width >= 1 is honoured
        # as a real (if absurd) terminal; only 0 is treated as "unknown".
        console = Console(soft_wrap=True, highlight=False, width=_FALLBACK_WIDTH)
    return console


def format_download(megabytes: int) -> str:
    """Human-readable download size, or ``""`` when the demo downloads nothing.

    Demo metadata records whole megabytes, so the raw value reads as ``30000MB``
    for a 29 GB dataset — a number nobody can size up at a glance.

    The two constants below are deliberately different and it is not an
    oversight: **1000 is a readability cut-off, 1024 is the unit**. The metadata
    means mebibytes, so that is what the division uses; the switch happens at a
    round 1000 because ``global_rivers_earth`` records exactly 1000 MB and
    "1.0 GB" is the reading that helps there, where a strict 1024 cut-off would
    print "1000 MB" — the very number this function exists to translate.
    """
    if megabytes <= 0:
        return ""
    if megabytes < 1000:
        return f"{megabytes} MB"
    gigabytes = megabytes / 1024
    return f"{gigabytes:.0f} GB" if gigabytes >= 10 else f"{gigabytes:.1f} GB"


def _needs_tokens(info: DemoInfo) -> list[tuple[str, str]]:
    """``(text, style)`` pairs summarising what a demo costs you to run."""
    tokens: list[tuple[str, str]] = []
    download = format_download(info.download_mb)
    if download:
        tokens.append((download, "cyan"))
    if info.gpu == "required":
        tokens.append(("GPU", "magenta"))
    elif info.gpu == "optional":
        tokens.append(("GPU?", "magenta dim"))
    label = _LOCAL_LABEL.get(info.local_data or "")
    if label:
        tokens.append((label, "yellow"))
    return tokens


def needs_text(info: DemoInfo) -> str:
    """The requirements cell as plain text: what a demo costs you to run.

    The plain-text twin of :func:`_needs_tokens` — same content, no styling —
    used to measure the column and asserted on directly by the render tests,
    where comparing a string beats reconstructing one from styled spans.
    """
    return " ".join(text for text, _ in _needs_tokens(info))


@dataclass(frozen=True)
class _Widths:
    """Column widths measured across the whole listing, so sections align.

    NEEDS is deliberately absent: it is the last column and is never padded, so
    a row is only as wide as its own requirements. Padding it to the widest
    demo's would add its full width to *every* row and push the table past 80
    columns for the sake of trailing blanks.
    """

    index: int
    key: int
    geometry: int


def _measure(demos: Sequence[DemoInfo]) -> _Widths:
    return _Widths(
        index=max((_cells(str(d.index)) for d in demos), default=1),
        key=max((_cells(d.key) for d in demos), default=3),
        geometry=max((_cells(d.geometry) for d in demos), default=8),
    )


def _row(info: DemoInfo, status: str, widths: _Widths) -> Text:
    """One demo, as a styled row aligned to ``widths``."""
    glyph, glyph_style = _STATUS_GLYPH.get(status, _STATUS_GLYPH[""])
    row = Text(" " * _INDENT)
    row.append(glyph, style=glyph_style)
    row.append(" " * _RAIL_GAP)
    row.append(f"{info.index:>{widths.index}}", style="dim")
    row.append(" " * _GUTTER)
    # The key is what the user types into `demo run`, so it gets the one
    # attention-grabbing style in the row.
    row.append(_pad(info.key, widths.key), style="bold cyan")
    row.append(" " * _GUTTER)
    # No explicit colour: forcing "white" is a literal ANSI white that
    # vanishes on a light-background terminal, where unstyled text uses the
    # reader's own foreground and is always legible.
    row.append(_pad(info.geometry, widths.geometry))
    row.append(" " * _GUTTER)
    for index, (text, style) in enumerate(_needs_tokens(info)):
        if index:
            row.append(" ")
        row.append(text, style=style)
    # A demo that needs nothing would otherwise end in the geometry column's
    # padding — invisible here, but real trailing whitespace once the row is
    # piped into a file or a diff.
    row.rstrip()
    return row


def _summary(demos: Sequence[DemoInfo], statuses: Mapping[str, str]) -> Text:
    """Headline: how many demos there are and how many are ready to open."""
    built = sum(1 for d in demos if statuses.get(d.key) == STATUS_BUILT)
    cached = sum(1 for d in demos if statuses.get(d.key) == STATUS_CACHED)
    noun = "demo" if len(demos) == 1 else "demos"
    line = Text("🎬 ")
    line.append(f"{len(demos)} Luxar {noun}", style="bold")
    # Zero counts are omitted: "0 not generated yet" is noise on a fully built
    # checkout, and reads as a warning that isn't there.
    for count, label, style in (
        (built, "built", "green"),
        (cached, "cached", "blue"),
        (len(demos) - built - cached, "not generated yet", "dim"),
    ):
        if count:
            line.append(f"  ·  {count} {label}", style=style)
    return line


def _rule(heading: str, tail: str, width: int) -> Text:
    """``HEADING ─────────────── tail``, filling to ``width``.

    The one section-header shape every ``luxar demo`` listing uses, so the
    catalogue, the dependency report, and the cache inventory read as pages of
    the same document rather than three unrelated tables.
    """
    head = Text(" " * _INDENT)
    head.append(heading, style="bold")
    fill = max(1, width - head.cell_len - _cells(tail) - 2)
    head.append(" " + "─" * fill + " ", style="dim")
    head.append(tail, style="dim")
    return head


def _section_rule(category: str, count: int, width: int) -> Text:
    """``CATEGORY ───────────── N demos``, filling to ``width``."""
    noun = "demo" if count == 1 else "demos"
    return _rule(category.upper(), f"{count} {noun}", width)


#: Minimum widths for the cache inventory's size and name columns, so a listing
#: of tiny caches still reads as a table rather than a ragged left edge. Both
#: are floors, not fixed widths — the columns grow to fit their content.
_SIZE_COLUMN = 10
_NAME_COLUMN = 8

#: Footer column widths. Derived from ``_INDENT`` rather than hard-coded: the
#: literals here were once ``11 + 38``, which silently baked in a left margin of
#: 2 and kept meaning "39" after the margin narrowed to 1.
_HINT_LABEL = 9
_HINT_COMMAND = 38

#: Gap between the legend's three rail entries.
_LEGEND_GAP = 3

#: The detail view is a RECORD, not a table: its fields sit indented under
#: the title rather than at the catalogue's left margin, so it deliberately
#: does not use ``_INDENT``. Named so that the difference reads as a decision
#: rather than as drift — the label column is 13 so the longest label,
#: "Network sim", still gets a clear gap before its value.
_DETAIL_INDENT = 3
_DETAIL_LABEL = 13


def _hint(label: str, command: str, note: str = "") -> Text:
    """One footer line: an aligned action label, its command, and a comment."""
    line = Text(" " * _INDENT)
    line.append(f"{label:<{_HINT_LABEL}}", style="bold")
    line.append(command, style="cyan")
    if note:
        # At least one space, so an over-long command never abuts its note.
        note_column = _INDENT + _HINT_LABEL + _HINT_COMMAND
        line.pad_right(max(1, note_column - line.cell_len))
        line.append(note, style="dim")
    return line


def _legend() -> list[Text]:
    """Decode the row rail and the NEEDS words, in the colours they appear in.

    Two lines rather than one: a single line runs past 80 columns and wraps in
    the middle of a term it is trying to define.
    """
    rail = Text(" " * _INDENT)
    for index, status in enumerate((STATUS_BUILT, STATUS_CACHED, "")):
        glyph, style = _STATUS_GLYPH[status]
        if index:
            rail.append(" " * _LEGEND_GAP)
        # Built from the rail's OWN glyph, so the legend cannot describe a mark
        # the rows do not print. The third state has no mark at all, and saying
        # so is the only honest way to teach it.
        rail.append(glyph if glyph.strip() else "(blank)", style=style)
        rail.append(f" {_STATUS_LABEL[status]}", style=style)
    needs = Text(" " * _INDENT)
    needs.append("GPU", style="magenta")
    needs.append("/", style="dim")
    needs.append("GPU?", style="magenta dim")
    needs.append(" = required/optional     ", style="dim")
    needs.append("git-lfs kaggle manual", style="yellow")
    needs.append(" = data you supply", style="dim")
    return [rail, needs]


def _dep_verdict(row: DependencyStatus) -> tuple[str, str]:
    """A dependency row's word and colour: ok / OUTDATED / MISSING.

    Branches rather than a ``(satisfied, installed)`` dict lookup so it is
    TOTAL. ``survey`` computes ``satisfied = installed and version_ok``, so
    satisfied-but-not-importable cannot arise there — but that invariant is
    nowhere enforced, and a lookup table missing the fourth combination turns a
    caller that violates it into a ``KeyError`` traceback out of ``demo deps``,
    which is precisely the command you run when the environment is already
    suspect. "Unmet" still splits in two, because installed-but-below-its-pin
    and absent-entirely are different surprises even though both need the same
    (re)install.
    """
    if row.satisfied:
        return ("ok", "green")
    return ("OUTDATED", "yellow") if row.installed else ("MISSING", "red")


def render_dependencies(console: Console, rows: Sequence[DependencyStatus]) -> None:
    """Print the optional-dependency report, in the catalogue's visual key."""
    modules = [r.module for r in rows]
    specs = [r.spec.spec for r in rows]
    extras = [r.spec.extra or "—" for r in rows]
    # Never let a column be narrower than its own header: a one-row report
    # (`--only scipy`) would otherwise print a ragged table.
    mw = max([_cells(m) for m in modules] + [_cells("MODULE")])
    sw = max([_cells(s) for s in specs] + [_cells("REQUIREMENT")])
    ew = max([_cells(e) for e in extras] + [_cells("EXTRA")])
    stw = max([_cells(_dep_verdict(r)[0]) for r in rows] + [_cells("STATUS")])
    plural = "dependency" if len(rows) == 1 else "dependencies"
    # Computing the rule width from the columns is exact HERE, unlike in the
    # catalogue, and only because of one property: nothing in this table is
    # rstripped except the header, whose own rstrip removes exactly the one
    # trailing gutter. So the header line always measures `width` on the nose
    # and the rule always brackets the table. The catalogue used the same
    # arithmetic and was wrong, because rstripping its rows silently dropped a
    # gutter and a column of padding. If you ever rstrip these rows, switch to
    # measuring the built lines the way `render_catalogue` now does.
    width = _INDENT + mw + sw + ew + stw + 3 * _GUTTER
    console.print(
        _rule(
            "📦 OPTIONAL DEMO DEPENDENCIES",
            f"{len(rows)} {plural}",
            min(width, console.width),
        )
    )
    header = Text(" " * _INDENT)
    for label, column in (
        ("MODULE", mw),
        ("REQUIREMENT", sw),
        ("EXTRA", ew),
        ("STATUS", stw),
    ):
        header.append(f"{label:<{column}}", style="dim")
        header.append(" " * _GUTTER)
    header.rstrip()
    console.print(header)
    for row, module, spec, extra in zip(rows, modules, specs, extras):
        label, style = _dep_verdict(row)
        line = Text(" " * _INDENT)
        # Padded through `_pad` like every other data column: these come from
        # the dependency table rather than from this module, and the widths
        # above are measured in cells, so a format spec could disagree with them.
        line.append(_pad(module, mw), style="bold cyan")
        line.append(" " * _GUTTER)
        line.append(_pad(spec, sw))
        line.append(" " * _GUTTER)
        line.append(_pad(extra, ew), style="dim")
        line.append(" " * _GUTTER)
        line.append(label, style=style)
        console.print(line)


def _cache_owner(entry: CacheEntry) -> Text:
    """Who claims this cache directory — the inventory's trailing cell.

    A protected dir holds a hand-placed input, so it is claimed whatever
    DEMO_META says — calling it an ORPHAN would invite the very
    ``clear --orphans`` that must never touch it.
    """
    owner = Text()
    if entry.protected:
        if entry.demo_keys:
            owner.append(", ".join(entry.demo_keys) + "  ", style="dim")
        owner.append("🔒 hand-placed input", style="yellow")
    elif entry.demo_keys:
        owner.append(", ".join(entry.demo_keys), style="dim")
    else:
        owner.append("⚠️  ORPHAN", style="yellow")
    return owner


def render_caches(
    console: Console,
    entries: Sequence[CacheEntry],
    root: Path,
    format_size: Callable[[float], str],
) -> None:
    """Print the demo cache inventory, in the catalogue's visual key.

    ``format_size`` is injected rather than imported so this module stays a leaf
    of the CLI package and cannot pull the rest of it into a rendering test.
    """
    names = [e.path.name for e in entries]
    sizes = [format_size(e.size_bytes) for e in entries]
    # Both columns are measured from what will actually be printed. A hardcoded
    # size width has to be repeated in the row format AND the rule arithmetic,
    # and `format_size` is a caller-supplied function with no length contract —
    # a petabyte total overruns any literal and desynchronises the two.
    sw = max([_cells(s) for s in sizes] + [_SIZE_COLUMN])
    nw = max([_cells(n) for n in names] + [_NAME_COLUMN])
    # Measure the owner column from the cells actually rendered, not from
    # ``demo_keys`` alone: the protected branch appends "🔒 hand-placed input"
    # on top of the keys, so a width derived from the keys stops the rule 13
    # columns inside the widest row. NO placeholder floor here — the owner cell
    # is last and unpadded, so reserving room for "⚠️  ORPHAN" on a table that
    # contains no orphan is 9 columns of rule past the end of the table.
    owner_cells = [_cache_owner(e) for e in entries]
    owners = max([o.cell_len for o in owner_cells] + [0])
    total = sum(e.size_bytes for e in entries)
    noun = "dir" if len(entries) == 1 else "dirs"
    width = _INDENT + sw + _GUTTER + nw + _GUTTER + owners
    console.print(
        _rule(
            "💾 DEMO CACHES",
            f"{format_size(total)} in {len(entries)} {noun}",
            min(width, console.width),
        )
    )
    console.print(Text(f"{' ' * _INDENT}{root}", style="dim"))
    for size, name, owner in zip(sizes, names, owner_cells):
        line = Text(" " * _INDENT)
        line.append(_pad(size, sw, right=True), style="cyan")
        line.append(" " * _GUTTER)
        line.append(_pad(name, nw), style="bold cyan")
        line.append(" " * _GUTTER)
        line.append_text(owner)
        console.print(line)


def _field(label: str, value: str, style: str = "") -> Text:
    """One ``label   value`` line of the detail view."""
    line = Text(" " * _DETAIL_INDENT)
    # 13 wide, so the longest label ("Network sim") still gets a clear gap
    # before its value rather than butting straight up against it.
    line.append(f"{label:<{_DETAIL_LABEL}}", style="dim")
    line.append(value, style=style)
    return line


def render_detail(console: Console, info: DemoInfo, status: str) -> None:
    """Print everything known about one demo, in the catalogue's visual key."""
    req = info.requirements
    title = Text("🎬 ")
    title.append(info.title, style="bold")
    title.append("  ·  ")
    title.append(info.key, style="bold cyan")
    title.append(f"  ·  #{info.index}", style="dim")
    console.print(title)
    console.print(Text(f"   {info.description}", style="dim"))
    console.print()
    console.print(_field("Category", info.category))
    console.print(_field("Geometry", info.geometry))
    console.print(_field("Compute", str(req["compute"])))
    download = format_download(info.download_mb)
    console.print(
        _field("Download", download, "cyan")
        if download
        else _field("Download", "none (offline)", "dim")
    )
    console.print(_field("GPU", str(req["gpu"])))
    if req["local_data"]:
        console.print(_field("Local data", str(req["local_data"]), "yellow"))
    label, style = _STATUS_WORD.get(status, _STATUS_WORD[""])
    console.print(_field("Status", label, style))
    if info.caches:
        console.print(_field("Caches", ", ".join(info.caches)))
    if info.outputs:
        console.print(
            _field("Outputs", ", ".join(f"{o}.luxar.zarr" for o in info.outputs))
        )
    console.print()
    console.print(_field("Run", f"luxar demo run {info.key}", "cyan"))
    console.print(_field("Module", f"python -m {info.module}", "cyan"))
    console.print(
        _field(
            "Network sim",
            "luxar serve <scene> --viewer --profile 3g",
            "cyan",
        )
    )


def render_catalogue(
    console: Console,
    demos: Sequence[DemoInfo],
    statuses: Mapping[str, str],
    *,
    example_key: Optional[str] = None,
) -> None:
    """Print the demo catalogue: a summary, category sections, and next steps.

    ``statuses`` maps demo key → ``STATUS_BUILT`` / ``STATUS_CACHED`` / ``""``;
    ``example_key`` names a demo to show in the ``run`` hint (a cheap offline
    one, when there is one to suggest).
    """
    widths = _measure(demos)
    # Build every row up front and MEASURE it, rather than re-deriving its width
    # from the column arithmetic. The two would have to agree forever, and they
    # already didn't: `_row` is rstripped, so a demo that needs nothing loses
    # both its trailing gutter AND the geometry column's padding, and a computed
    # width overshot such a row by up to `_GUTTER + geometry` — the section rule
    # then ran past the right edge of the table it is meant to bracket.
    rendered = [
        (demo, _row(demo, statuses.get(demo.key, ""), widths))
        for demo in sorted(demos, key=lambda d: (d.category, d.key))
    ]
    rule_width = min(
        max((row.cell_len for _, row in rendered), default=40), console.width
    )
    console.print(_summary(demos, statuses))
    for category, pairs in groupby(rendered, key=lambda pair: pair[0].category):
        group = list(pairs)
        console.print()
        console.print(_section_rule(category, len(group), rule_width))
        for _, row in group:
            console.print(row)
    console.print()
    for line in _legend():
        console.print(line)
    console.print()
    example = f"e.g. luxar demo run {example_key}" if example_key else ""
    console.print(_hint("Run", "luxar demo run <key|#>", example))
    console.print(_hint("Details", "luxar demo info <key|#>"))
    console.print(_hint("Filter", "luxar demo list -c astronomy -g gsplats"))
    # One line per management command, each naming the ACTION as well as the
    # report. Folding these into a single "deps · cache list · stop" line lost
    # the actionable half of each pair — `--install` (the fix for what `deps`
    # reports), `cache clear` (how you reclaim what `cache list` measures), and
    # why anyone runs `stop` at all. Those were all in the previous footer.
    console.print(_hint("Deps", "luxar demo deps", "--install fixes what it reports"))
    console.print(
        _hint("Caches", "luxar demo cache list", "cache clear <key> to reclaim")
    )
    console.print(_hint("Stop", "luxar demo stop", "frees ports of forgotten runs"))
