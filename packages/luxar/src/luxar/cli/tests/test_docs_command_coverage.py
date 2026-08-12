"""Drift guard: the docs CLI reference and the live Typer app must match.

Walks the live Typer application (``luxar.cli.main.app``) to enumerate the full
set of public (non-hidden) leaf-command paths, extracts every ``luxar ...``
invocation from the fenced code blocks of ``docs/guides/user/CLI_REFERENCE.md``,
and compares the two sets in BOTH directions:

* a live public command missing from the doc's code blocks fails (new command
  added without documenting it);
* a documented invocation that no longer resolves to a live public command or
  command group fails (command removed, renamed, or hidden without pruning the
  doc).

That first guard truncates each documented invocation at its first ``-`` token,
so it only ever validates command PATHS, never OPTION SPELLINGS — a doc that
told readers to pass an option the command doesn't declare (e.g. ``-o`` on a
command whose output is a positional argument) would sail through it. The
second guard below closes part of that gap: for every ``### `luxar <path>` ``
-style per-command section heading, it collects the option-looking tokens
named in that section's prose — but only from INLINE-CODE spans (single
backtick); a fenced ```bash catalog block inside the same section is not
scanned, so a stale flag hiding inside one of those example lines sails
through undetected. A heading that resolves to a LEAF command is checked
strictly, against that command's own declared spellings. A heading that
resolves to a command GROUP is checked more weakly, against the UNION of
every one of its leaf descendants' spellings — a group's prose legitimately
names its subcommands' flags, so a union check can only prove a flag still
exists SOMEWHERE under the group, which is enough to catch a rename or
removal but not enough to prove the flag belongs to the subcommand named
beside it. Prose that sits OUTSIDE any per-command heading — this doc's
``## Top-level commands`` and ``## Staying in sync`` sections in particular —
is covered by neither guard.
"""

from __future__ import annotations

import re
from pathlib import Path

import click
import pytest
import typer

from luxar.cli.main import app

_DOC_RELPATH = Path("docs/guides/user/CLI_REFERENCE.md")

# Matches a per-command section heading such as ``### `luxar mesh lod` `` at
# heading levels 2-4 (the levels actually used in the doc today).
_SECTION_HEADING_RE = re.compile(r"^(#{2,4})\s+`luxar\s+([^`]+)`\s*$")

# Any Markdown heading, used to find where a matched section's body ends.
_ANY_HEADING_RE = re.compile(r"^(#{1,6})\s")

# An option-shaped token: a single-letter short flag or a `--long-name` (with
# internal hyphens allowed, e.g. `--compression-factor`). Deliberately does
# NOT match a bare negative number (`-1`), a range (`0:50`), or a path-like
# span (`<stem>.luxar.zarr`) — those are common in this doc's prose and must
# not be mistaken for an option spelling.
_OPTION_TOKEN_RE = re.compile(r"^(--[A-Za-z][A-Za-z0-9-]*|-[A-Za-z])$")


def _command_name(command: typer.models.CommandInfo) -> str:
    """Resolve the invocation name of a registered command."""
    if command.name:
        return command.name
    assert command.callback is not None
    return command.callback.__name__.replace("_", "-")


def _leaf_paths(app_instance: typer.Typer, prefix: tuple[str, ...] = ()) -> list[str]:
    """Return space-joined paths of all non-hidden leaf commands under ``app``."""
    paths: list[str] = []
    for command in app_instance.registered_commands:
        if getattr(command, "hidden", False):
            continue
        paths.append(" ".join((*prefix, _command_name(command))))
    for group in app_instance.registered_groups:
        assert group.typer_instance is not None
        if getattr(group, "hidden", False) or getattr(
            group.typer_instance.info, "hidden", False
        ):
            continue
        group_name = group.name or group.typer_instance.info.name
        assert group_name is not None
        paths.extend(_leaf_paths(group.typer_instance, (*prefix, group_name)))
    return paths


def _documented_paths(doc_text: str) -> set[str]:
    """Extract ``luxar ...`` invocation paths from the doc's fenced code blocks.

    An invocation path is the whitespace-joined tokens after ``luxar`` up to the
    first option token (``-...``) or trailing ``#`` comment, so both catalog
    lines (``luxar gsplat fit   # ...``) and help examples
    (``luxar gsplat fit --help``) resolve to ``gsplat fit``. A bare ``luxar``
    (e.g. ``luxar --help``) yields the empty path and is dropped.
    """
    paths: set[str] = set()
    in_block = False
    for line in doc_text.splitlines():
        if line.lstrip().startswith("```"):
            in_block = not in_block
            continue
        tokens = line.split()
        if not in_block or not tokens or tokens[0] != "luxar":
            continue
        path: list[str] = []
        for token in tokens[1:]:
            if token.startswith("-") or token.startswith("#"):
                break
            path.append(token)
        if path:
            paths.add(" ".join(path))
    return paths


def _find_doc() -> Path | None:
    """Locate the CLI reference doc by walking up from this file."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / _DOC_RELPATH
        if candidate.is_file():
            return candidate
    return None


def _resolve_click_command(path: str) -> object | None:
    """Resolve a space-separated ``luxar`` command path against the live app.

    Walks ``click.Group.get_command`` from the root, exactly like Click's own
    dispatch does at invocation time, so it can never disagree with what
    running ``luxar <path>`` actually resolves to. Returns ``None`` if any
    segment fails to resolve (a stale heading — ``test_docs_and_cli_command_sets_match``
    above already fails that case; this helper just needs to not raise).
    """
    node: object = typer.main.get_command(app)
    ctx = click.Context(node)  # type: ignore[arg-type]
    for part in path.split():
        if not _is_group(node):
            return None
        found = node.get_command(ctx, part)  # type: ignore[attr-defined]
        if found is None:
            return None
        ctx = click.Context(found, parent=ctx)
        node = found
    return node


def _is_group(command: object) -> bool:
    """True if ``command`` is a command GROUP (has subcommands), not a leaf.

    ``isinstance(command, click.Group)`` looks like the obvious check and is
    WRONG here: Typer vendors its own click shim (``typer._click.core``), so a
    live ``TyperGroup``'s MRO is ``(TyperGroup, typer._click.core.Command, ABC,
    object)`` and never includes ``click.Group`` — the isinstance check is
    False for EVERY group, verified against typer 0.27.0 / click 8.4.2 in this
    repo's environment. That bug is invisible from the test's outcome: it just
    makes every group look like a leaf, so the section below it gets skipped
    and the assertion never fires — a silent false negative, not an error. Duck
    typing on ``get_command`` (the method Click's own dispatch calls) can't
    drift the same way, so resist "cleaning it up" to an isinstance check.
    """
    return hasattr(command, "get_command")


def _option_spellings(command: object) -> set[str]:
    """Every option spelling declared on a leaf command.

    Union of ``opts`` AND ``secondary_opts`` — not ``opts`` alone — because a
    ``--flag/--no-flag`` boolean pair only exposes the "on" spelling via
    ``opts``, and a retired ``hidden=True`` alias (e.g. mesh lod's old
    ``--method``/``-m``) is exactly the kind of spelling a doc may legitimately
    still name, to tell the reader it is gone.

    Filtered to spellings that start with ``-``: a Click ``Argument``'s
    ``opts`` is its bare name (e.g. ``input_path``), not a flag, and iterating
    ``params`` without discriminating by param type would otherwise leak those
    bare names into the "Real spellings" list in the assertion message below —
    exactly the positional-vs-option confusion this test exists to catch. Safe
    to filter unconditionally: a bare name never matches ``_OPTION_TOKEN_RE``,
    so it could never have affected which tokens pass or fail, only how
    trustworthy the message looks when one fails.
    """
    spellings: set[str] = set()
    for param in getattr(command, "params", []):
        spellings.update(getattr(param, "opts", ()))
        spellings.update(getattr(param, "secondary_opts", ()))
    return {spelling for spelling in spellings if spelling.startswith("-")}


def _option_tokens_in_span(span: str) -> list[str]:
    """Pull option-shaped tokens out of one inline-code span's text.

    Spans in this doc take several shapes: an alias pair (``-L/--levels``), a
    usage example (``-o out``, ``--node surf/child_0``), a bare flag
    (``--overwrite``), or something that is not an option at all (a path like
    ``<stem>.luxar.zarr``, a slice range like ``0:50``, a negative number). This
    splits on whitespace and ``/`` to cover the first two shapes, strips
    leading AND trailing bracket/quote punctuation (so ``[--overwrite]`` and
    ``(--node)`` are not silently dropped the way an unstripped leading ``[``/
    ``(`` would otherwise make them) and any ``=value`` suffix from each piece,
    and keeps only the pieces that are option-shaped — everything else
    (including a bare word like ``out`` from ``-o out``) is silently dropped
    rather than misread as a flag.

    Known false NEGATIVE, not a silent one: the span regex this feeds
    (``re.findall(r"`([^`\\n]+)`", ...)``) forbids an embedded newline, so an
    inline-code span broken across a hand-wrapped line — this doc is
    hand-wrapped, not prose-reflowed — is missed entirely rather than
    misparsed. A reflow that happens to split a span mid-token can hide a
    token from this check.
    """
    tokens: list[str] = []
    for piece in re.split(r"[\s/]+", span.strip()):
        piece = piece.lstrip("([{<\"'")
        piece = piece.rstrip(".,;:)]}>\"'")
        piece = piece.split("=", 1)[0]
        if _OPTION_TOKEN_RE.match(piece):
            tokens.append(piece)
    return tokens


def _section_body(lines: list[str], line_no: int, level: int) -> str:
    """Return the body text of the section heading at ``lines[line_no]``.

    The body runs from the line after the heading to the next heading that is
    either (a) at the same or shallower level — normal Markdown nesting — or
    (b) a per-command heading (matches ``_SECTION_HEADING_RE``) at ANY deeper
    level, because that nested command section is checked on its own, more
    strictly, and must not be swallowed into an ancestor group's body. This is
    what carves ``### `luxar mesh lod` `` out of ``## `luxar mesh` ``'s body,
    while leaving ``## `luxar gsplat` ``'s own non-command ``###`` subsections
    ("Fitting & calibration", "Level-of-detail", …) inside it, since those
    don't match the per-command heading pattern.

    Tracks the same fenced-code-block toggle ``_documented_paths`` uses, so a
    shell ``# comment`` line inside a ```bash block is never mistaken for a
    Markdown heading by ``_ANY_HEADING_RE`` (which has no fence awareness of
    its own) and doesn't truncate the body early, silently dropping the rest
    of the section from the token scan.
    """
    end = len(lines)
    in_block = False
    for j in range(line_no + 1, len(lines)):
        if lines[j].lstrip().startswith("```"):
            in_block = not in_block
            continue
        if in_block:
            continue
        hm = _ANY_HEADING_RE.match(lines[j])
        if not hm:
            continue
        if len(hm.group(1)) <= level or _SECTION_HEADING_RE.match(lines[j]):
            end = j
            break
    return "\n".join(lines[line_no + 1 : end])


def _group_option_spellings(path: str, leaf_paths: list[str]) -> set[str]:
    """Union of declared option spellings over every leaf descendant of ``path``.

    Deliberately weaker than the leaf check in
    ``test_docs_option_spellings_are_declared``: a GROUP section's prose
    legitimately names its subcommands' flags (``--no-weld`` inside
    ``## `luxar mesh` `` belongs to ``mesh import``, not ``mesh lod``), so all
    a group check can prove is that the named flag still exists SOMEWHERE
    under the group — enough to catch a rename or removal (the exact bug
    class ``--no-weld`` demonstrated: renaming it left the leaf check
    skipping the whole group section, and only a union check like this one
    would notice).
    """
    prefix = path + " "
    spellings: set[str] = set()
    for leaf in leaf_paths:
        if leaf.startswith(prefix):
            command = _resolve_click_command(leaf)
            if command is not None:
                spellings.update(_option_spellings(command))
    return spellings


def test_docs_option_spellings_are_declared() -> None:
    """Every option named in a per-command doc section must be real.

    ``test_docs_and_cli_command_sets_match`` guards command PATHS but stops at
    the first ``-`` token, so it never checks that an option a doc names is one
    the command actually accepts. This walks every ``### `luxar <path>` ``-style
    section heading (levels 2-4) and checks the option-looking tokens in that
    section's INLINE-CODE spans only (see ``_section_body`` for exactly where a
    section's prose is taken to end) — a fenced example block inside the same
    section is not scanned, so a stale or invented flag hiding inside one of
    its catalog lines is not caught here:

    * a section that resolves to a LEAF command is checked strictly, against
      that command's own declared spellings (union of ``opts`` and
      ``secondary_opts``, so a retired ``hidden=True`` alias a doc may
      legitimately still name — e.g. mesh lod's old ``--method``/``-m`` — counts
      as declared too);
    * a section that resolves to a command GROUP is checked more weakly,
      against the UNION of every one of its leaf descendants' spellings (see
      ``_group_option_spellings``) — enough to catch a rename or removal, not
      enough to prove the flag belongs to the subcommand actually named beside
      it.

    NOT covered: prose that sits outside any per-command heading — this doc's
    ``## Top-level commands`` and ``## Staying in sync`` sections in
    particular — since there is no single resolved command to check those
    tokens against; and, even inside a covered heading, any option token that
    only appears in a fenced code block rather than inline code.

    Known false-positive shape: a section may legitimately name ANOTHER
    command's flag as a cross-reference (`mesh lod`'s prose naming
    `gsplat lod`'s ``-m``, to explain why mesh's own ``-m`` is retired). That
    passes today only because the SECTION'S OWN command still declares a
    ``hidden=True`` migration shim under that spelling (the referenced command is
    never consulted for the token); if such a shim is
    ever deleted, this assertion will fire on otherwise-correct prose. See the
    assertion message below for what to do when that happens.
    """
    doc = _find_doc()
    if doc is None:
        pytest.skip("docs tree not available")

    lines = doc.read_text(encoding="utf-8").splitlines()
    headings = [
        (i, len(m.group(1)), m.group(2).strip())
        for i, line in enumerate(lines)
        for m in [_SECTION_HEADING_RE.match(line)]
        if m
    ]
    assert headings, "no per-command section headings found in the CLI reference"

    leaf_paths = _leaf_paths(app)

    checked_tokens = 0
    for line_no, level, path in headings:
        command = _resolve_click_command(path)
        if command is None:
            continue  # stale heading; caught by test_docs_and_cli_command_sets_match

        is_group = _is_group(command)
        # `--help` is a universal Click option that never appears in a
        # command's own `params`, so it is always valid to name.
        if is_group:
            # Union in the group's OWN callback options too, not just its
            # descendants': no group in this app declares any today, but the
            # root app does (`--version`, `--install-completion`,
            # `--show-completion`), so a group with its own `@app.callback()`
            # option is one edit away and must not spuriously fail here.
            declared = (
                _group_option_spellings(path, leaf_paths)
                | _option_spellings(command)
                | {"--help"}
            )
        else:
            declared = _option_spellings(command) | {"--help"}

        body = _section_body(lines, line_no, level)
        for span in re.findall(r"`([^`\n]+)`", body):
            for token in _option_tokens_in_span(span):
                checked_tokens += 1
                not_declared = (
                    f"`luxar {path}` nor any command under it declares"
                    if is_group
                    else f"`luxar {path}` does not declare"
                )
                assert token in declared, (
                    f"CLI reference section '{'#' * level} `luxar {path}`' "
                    f"names option {token!r}, which {not_declared}. Real "
                    f"spellings: {sorted(declared)}. If this is deliberately "
                    "naming ANOTHER command's flag as a cross-reference, spell "
                    "the token outside inline code (or rephrase the sentence) "
                    "instead of inventing an option on this command."
                )

    assert checked_tokens, (
        "no option-looking inline-code tokens were found in any per-command "
        "section of the CLI reference — the heading pattern or extraction "
        "regex likely stopped matching anything"
    )


def test_docs_and_cli_command_sets_match() -> None:
    """The doc catalog and the live public command tree must agree both ways."""
    doc = _find_doc()
    if doc is None:
        pytest.skip("docs tree not available")

    leaves = set(_leaf_paths(app))
    assert leaves, "no commands discovered on the live Typer app"
    # Group invocations like `luxar gsplat --help` are valid doc content: every
    # proper prefix of a public leaf path names a live command group.
    groups = {
        " ".join(parts[:i])
        for path in leaves
        for parts in [path.split()]
        for i in range(1, len(parts))
    }

    documented = _documented_paths(doc.read_text(encoding="utf-8"))
    assert documented, "no luxar invocations extracted from the CLI reference"

    missing = sorted(leaves - documented)
    assert not missing, (
        "CLI reference "
        f"({_DOC_RELPATH}) is missing these public commands:\n"
        + "\n".join(f"  - luxar {path}" for path in missing)
    )

    stale = sorted(documented - leaves - groups)
    assert not stale, (
        "CLI reference "
        f"({_DOC_RELPATH}) documents commands that no longer exist publicly:\n"
        + "\n".join(f"  - luxar {path}" for path in stale)
    )
