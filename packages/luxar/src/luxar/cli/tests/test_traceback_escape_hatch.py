"""A failing CLI command must be able to hand back its traceback.

Seventeen `except Exception` blocks across the CLI printed one line and raised
`typer.Exit(1)`. That is the right default — a stack trace is noise when the
cause is "file not found" — but there was no way to opt out, so a genuine bug
inside the library surfaced as one line with nowhere to go next. Six raised a
bare exit, and two used `from None`, so eight discarded the exception chain
(audit finding `A9-02`).

Three things are tested, and the third is the one that keeps working:

1. `exit_with_error` behaves both ways, and `traceback_requested` reads the
   environment the way the docstring says (including the falsey spellings).
2. The chain survives even on the quiet path, so `--show-locals`-style tooling
   and `raise ... from` consumers still see the cause.
3. No CLI source discards a caught exception in an `except Exception` block
   any more. That is a source scan, and it is what catches the eighteenth site
   somebody adds next month.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import typer

from luxar.cli import _traceback
from luxar.cli._traceback import (
    TRACEBACK_ENV_VAR,
    exit_with_error,
    traceback_requested,
)

CLI_ROOT = Path(_traceback.__file__).parent


class TestTracebackRequested:
    """Reading the environment."""

    def test_unset_is_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(TRACEBACK_ENV_VAR, raising=False)
        assert traceback_requested() is False

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on", "anything"])
    def test_truthy_spellings_are_on(
        self, value: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(TRACEBACK_ENV_VAR, value)
        assert traceback_requested() is True

    @pytest.mark.parametrize("value", ["", "0", "false", "FALSE", "no", "off", "  0  "])
    def test_falsey_spellings_are_off(
        self, value: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`LUXAR_TRACEBACK=0` in a shell profile must mean off.

        The failure mode this guards is "set, therefore on", which would make
        the variable impossible to turn off once exported.
        """
        monkeypatch.setenv(TRACEBACK_ENV_VAR, value)
        assert traceback_requested() is False


class TestExitWithError:
    """Both branches, and the chain."""

    def test_the_quiet_path_exits_1_and_prints_the_message(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv(TRACEBACK_ENV_VAR, raising=False)
        cause = ValueError("the underlying problem")
        with pytest.raises(typer.Exit) as caught:
            exit_with_error("❌ Error doing the thing: boom", cause)
        assert caught.value.exit_code == 1
        output = capsys.readouterr().out
        assert "❌ Error doing the thing: boom" in output
        assert TRACEBACK_ENV_VAR in output, (
            "the message must say how to get the traceback — a hint nobody "
            "reads in the docs is a hint nobody has"
        )

    def test_the_quiet_path_still_chains_the_cause(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`__cause__` survives even when the traceback is suppressed.

        Six of the routed sites raised a bare `typer.Exit(1)` and two used
        `from None`, so the cause was gone on the default path too.
        """
        monkeypatch.delenv(TRACEBACK_ENV_VAR, raising=False)
        cause = ValueError("the underlying problem")
        with pytest.raises(typer.Exit) as caught:
            exit_with_error("❌ nope", cause)
        assert caught.value.__cause__ is cause

    def test_the_loud_path_reraises_the_original_unchanged(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv(TRACEBACK_ENV_VAR, "1")
        cause = ValueError("the underlying problem")
        with pytest.raises(ValueError) as caught:
            exit_with_error("❌ nope", cause)
        assert caught.value is cause, "must re-raise the SAME exception object"
        assert capsys.readouterr().out == "", (
            "the loud path must not also print the one-liner — the traceback "
            "already carries the message"
        )

    def test_it_reraises_a_bare_exception_too(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The helper does not narrow the `BaseException` object it receives."""
        monkeypatch.setenv(TRACEBACK_ENV_VAR, "1")
        cause = KeyboardInterrupt()
        with pytest.raises(KeyboardInterrupt):
            exit_with_error("❌ interrupted", cause)


def _cli_sources() -> list[Path]:
    """Every production CLI source file."""
    return sorted(
        p
        for p in CLI_ROOT.rglob("*.py")
        if "__pycache__" not in p.parts and "tests" not in p.parts
    )


def test_the_cli_source_scan_finds_files() -> None:
    """Fail closed: an empty scan would make the gate below pass vacuously."""
    files = _cli_sources()
    assert len(files) > 40, f"only found {len(files)} CLI sources under {CLI_ROOT}"
    assert any(p.name == "main.py" for p in files)


def _discarding_exits(
    tree: ast.Module,
) -> list[tuple[ast.ExceptHandler, ast.Raise, str]]:
    offenders: list[tuple[ast.ExceptHandler, ast.Raise, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        # Only the broad handlers: a narrow `except FileNotFoundError` that
        # exits quietly is reporting an expected condition, not swallowing
        # a bug.
        caught = ast.unparse(node.type) if node.type else "BaseException"
        if caught not in {"Exception", "BaseException"}:
            continue
        body = "\n".join(ast.unparse(s) for s in node.body)
        prints_traceback = "print_exc" in body or "format_exc" in body
        for stmt in ast.walk(node):
            if not isinstance(stmt, ast.Raise) or stmt.exc is None:
                continue
            if not re.search(r"\btyper\.Exit\b", ast.unparse(stmt.exc)):
                continue
            chained = (
                node.name is not None
                and stmt.cause is not None
                and ast.unparse(stmt.cause) == node.name
            )
            if not chained and not prints_traceback:
                offenders.append((node, stmt, caught))
    return offenders


def test_no_cli_handler_discards_a_caught_exception() -> None:
    """A broad handler that exits must not throw the cause away entirely.

    Three ways to satisfy it, and the bar is deliberately the weakest of them:

    1. Route through `exit_with_error`, which offers the traceback on request
       and chains the cause either way. Preferred.
    2. Chain it yourself: `raise typer.Exit(1) from err`.
    3. Print the traceback in the handler (`traceback.print_exc()`).

    Option 3 is included because roughly two dozen CLI handlers already do it,
    and they are not the defect this file is about: they surface the traceback,
    they just surface it *unconditionally*, so a plain "file not found" arrives
    with a stack trace attached. Routing those through `exit_with_error` too
    would make the traceback opt-in everywhere — a real improvement, and a
    behaviour change for two dozen commands, so it belongs in its own change
    rather than being smuggled in behind a gate. #2553 tracks that follow-up.
    Until then this test pins the honest invariant (nothing is thrown away)
    rather than the one we would like (everything is consistent).

    Parsed with `ast` rather than grepped, so a `raise typer.Exit` in a string
    or comment cannot trip it and one inside a nested function cannot hide from
    it.
    """
    offenders: list[str] = []
    for path in _cli_sources():
        tree = ast.parse(path.read_text())
        rel = path.relative_to(CLI_ROOT.parent)
        for node, stmt, caught in _discarding_exits(tree):
            binding = f" as {node.name}" if node.name is not None else ""
            offenders.append(
                f"{rel}:{stmt.lineno}: except {caught}{binding} -> {ast.unparse(stmt)}"
            )
    assert not offenders, (
        "these handlers discard the exception they caught. Use "
        "`exit_with_error(message, err)` or `raise typer.Exit(1) from err`:\n  "
        + "\n  ".join(offenders)
    )


def test_the_scan_would_notice_a_discarding_handler() -> None:
    """The gate above passes; prove it is not passing vacuously.

    Runs the same AST rule over a synthetic module holding one compliant
    handler plus bound and unnamed discarding handlers, and asserts it
    separates them. Without this, an `ast` walk that quietly stopped matching
    `ExceptHandler` (a Python-version change, a refactor of the helper) would
    report a clean CLI forever.
    """
    source = (
        "import typer\n"
        "import traceback\n"
        "def good():\n"
        "    try:\n"
        "        pass\n"
        "    except Exception as e:\n"
        "        raise typer.Exit(1) from e\n"
        "def bad():\n"
        "    try:\n"
        "        pass\n"
        "    except Exception as e:\n"
        "        raise typer.Exit(1)\n"
        "def unnamed_bad():\n"
        "    try:\n"
        "        pass\n"
        "    except Exception:\n"
        "        raise typer.Exit(1)\n"
        "def narrow_good():\n"
        "    try:\n"
        "        pass\n"
        "    except ValueError as e:\n"
        "        raise typer.Exit(1)\n"
        "def printing_good():\n"
        "    try:\n"
        "        pass\n"
        "    except BaseException:\n"
        "        traceback.print_exc()\n"
        "        raise typer.Exit(1)\n"
    )
    found = [stmt.lineno for _, stmt, _ in _discarding_exits(ast.parse(source))]
    assert found == [12, 17], f"expected both discarding handlers, got {found}"


class TestItWorksThroughTheRealCli:
    """The helper's unit tests do not prove a command is wired to it.

    Everything above exercises `exit_with_error` directly. These invoke the
    actual `luxar info` command in a subprocess against a directory that exists
    but is not a zarr store — which reaches the routed handler in
    `info_command.py` — and check both branches.

    A subprocess because the environment variable is read at failure time and
    because the loud path re-raises, which Typer's own runner turns into a real
    process-level traceback; neither is observable in-process.
    """

    @staticmethod
    def _run(store: Path, *, traceback: bool) -> subprocess.CompletedProcess[str]:
        env = {**os.environ}
        if traceback:
            env[TRACEBACK_ENV_VAR] = "1"
        else:
            env.pop(TRACEBACK_ENV_VAR, None)
        return subprocess.run(
            [sys.executable, "-m", "luxar", "info", str(store)],
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )

    @pytest.fixture
    def not_a_store(self, tmp_path: Path) -> Path:
        """A directory that exists, so the pre-flight passes and the read fails."""
        store = tmp_path / "notastore.luxar.zarr"
        store.mkdir()
        return store

    def test_the_default_prints_one_line_and_the_hint(self, not_a_store: Path) -> None:
        result = self._run(not_a_store, traceback=False)
        assert result.returncode == 1, result.stdout + result.stderr
        combined = result.stdout + result.stderr
        assert "Error reading info for" in combined
        assert TRACEBACK_ENV_VAR in combined, "the hint must be discoverable"
        assert "Traceback" not in combined, (
            "the default path must not print a traceback:\n" + combined
        )

    def test_setting_the_variable_produces_a_real_traceback(
        self, not_a_store: Path
    ) -> None:
        result = self._run(not_a_store, traceback=True)
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        # The exception type from deep inside zarr, i.e. the actual origin —
        # not the CLI's one-line paraphrase of it.
        assert "GroupNotFoundError" in combined, (
            "expected the underlying exception, got:\n" + combined
        )
