"""A failing CLI command must be able to hand back its traceback.

CLI handlers report one line and raise `typer.Exit(1)` by default, while
`LUXAR_TRACEBACK=1` re-raises fatal exceptions and prints recoverable ones.
The first seventeen routed sites included eight that discarded the exception
chain (audit finding `A9-02`); #2553 routes the remaining interactive handlers
that used to print a traceback unconditionally.

Three things are tested, and the third is the one that keeps working:

1. `exit_with_error` behaves both ways, and `traceback_requested` reads the
   environment the way the docstring says (including the falsey spellings).
2. The chain survives even on the quiet path, so `--show-locals`-style tooling
   and `raise ... from` consumers still see the cause.
3. No interactive CLI source discards a caught exception in an
   `except Exception` block, and the four unattended fit/batch exceptions stay
   exact. That source scan catches the next inconsistent handler.
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
from arbol import Arbol

import luxar
from luxar.cli import _traceback
from luxar.cli._traceback import (
    TRACEBACK_ENV_VAR,
    exit_with_error,
    report_error,
    traceback_requested,
)

CLI_ROOT = Path(_traceback.__file__).parent
UNATTENDED_TRACEBACK_HANDLERS = {
    "cli/gsplat_ops/batch/merge_command.py::run_batch_merge_cmd",
    "cli/gsplat_ops/batch/run.py::run_batch_run",
    "cli/gsplat_ops/batch/submit.py::run_batch_submit",
    "cli/gsplat_ops/fitting/fit.py::run_fit_volume",
}


def test_the_package_fixture_clears_this_exact_variable() -> None:
    """The package conftest hard-codes the key to avoid importing the CLI."""
    conftest = Path(luxar.__file__).parent / "conftest.py"
    assert f'delenv("{TRACEBACK_ENV_VAR}"' in conftest.read_text()


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

    def test_the_quiet_path_survives_silent_arbol(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A fatal report must not disappear with ordinary narration."""
        monkeypatch.delenv(TRACEBACK_ENV_VAR, raising=False)
        monkeypatch.setattr(Arbol, "enable_output", False)
        cause = ValueError("the underlying problem")

        with pytest.raises(typer.Exit):
            exit_with_error("❌ nope", cause)

        captured = capsys.readouterr()
        assert captured.out == ""
        assert "❌ nope" in captured.err
        assert TRACEBACK_ENV_VAR in captured.err

    def test_the_quiet_path_survives_arbol_depth_truncation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A nested fatal report must outlive the narration depth cap."""
        monkeypatch.delenv(TRACEBACK_ENV_VAR, raising=False)
        monkeypatch.setattr(Arbol, "passthrough", False)
        monkeypatch.setattr(Arbol, "enable_output", True)
        monkeypatch.setattr(Arbol, "max_depth", 0)
        monkeypatch.setattr(Arbol, "_depth", 1)
        monkeypatch.setattr(Arbol._thread_local, "captured", False, raising=False)

        with pytest.raises(typer.Exit):
            exit_with_error("❌ nested nope", ValueError("cause"))

        captured = capsys.readouterr()
        assert captured.out == ""
        assert "❌ nested nope" in captured.err
        assert TRACEBACK_ENV_VAR in captured.err

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


class TestReportError:
    """Recoverable failures keep control flow unchanged in both modes."""

    def test_the_quiet_path_prints_the_message_and_hint(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv(TRACEBACK_ENV_VAR, raising=False)

        report_error("❌ recoverable", ValueError("cause"))

        output = capsys.readouterr().out
        assert "❌ recoverable" in output
        assert TRACEBACK_ENV_VAR in output

    def test_the_loud_path_prints_the_traceback_and_returns(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv(TRACEBACK_ENV_VAR, "1")

        report_error("❌ recoverable", ValueError("the underlying problem"))

        captured = capsys.readouterr()
        assert captured.out == ""
        assert "ValueError: the underlying problem" in captured.err


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
            if not chained:
                offenders.append((node, stmt, caught))
    return offenders


def _enclosing_function(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    owner = node
    while owner in parents and not isinstance(
        parents[owner], (ast.FunctionDef, ast.AsyncFunctionDef)
    ):
        owner = parents[owner]
    function = parents.get(owner)
    assert isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
    return function


def test_no_interactive_cli_handler_discards_a_caught_exception() -> None:
    """A broad interactive handler must preserve the cause it exits from.

    Two ways to satisfy it:

    1. Route through `exit_with_error`, which offers the traceback on request
       and chains the cause either way. Preferred.
    2. Chain it yourself: `raise typer.Exit(1) from err`.

    Parsed with `ast` rather than grepped, so a `raise typer.Exit` in a string
    or comment cannot trip it and one inside a nested function cannot hide from
    it.
    """
    offenders: list[str] = []
    for path in _cli_sources():
        tree = ast.parse(path.read_text())
        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        rel = path.relative_to(CLI_ROOT.parent)
        for node, stmt, caught in _discarding_exits(tree):
            key = f"{rel}::{_enclosing_function(node, parents).name}"
            if key in UNATTENDED_TRACEBACK_HANDLERS:
                continue
            binding = f" as {node.name}" if node.name is not None else ""
            offenders.append(
                f"{rel}:{stmt.lineno}: except {caught}{binding} -> {ast.unparse(stmt)}"
            )
    assert not offenders, (
        "these handlers discard the exception they caught. Use "
        "`exit_with_error(message, err)` or `raise typer.Exit(1) from err`:\n  "
        + "\n  ".join(offenders)
    )


def test_only_unattended_handlers_print_tracebacks_unconditionally() -> None:
    """Long-running fit/batch entry points keep tracebacks in unattended logs."""
    observed: set[str] = set()
    for path in _cli_sources():
        tree = ast.parse(path.read_text())
        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in {"print_exc", "format_exc"}
            ):
                continue
            function = _enclosing_function(node, parents)
            observed.add(f"{path.relative_to(CLI_ROOT.parent)}::{function.name}")

    assert observed == UNATTENDED_TRACEBACK_HANDLERS


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
    )
    found = [stmt.lineno for _, stmt, _ in _discarding_exits(ast.parse(source))]
    assert found == [11, 16], f"expected both discarding handlers, got {found}"


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
    def _run(
        command: list[str], *, traceback: bool
    ) -> subprocess.CompletedProcess[str]:
        env = {**os.environ}
        if traceback:
            env[TRACEBACK_ENV_VAR] = "1"
        else:
            env.pop(TRACEBACK_ENV_VAR, None)
        return subprocess.run(
            [sys.executable, "-m", "luxar", *command],
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

    @pytest.fixture
    def unreadable_gsplat_store(self, tmp_path: Path) -> Path:
        """A recognized gsplat root whose required arrays are absent."""
        from luxar._zarr_compat import open_group

        store = tmp_path / "unreadable.gsplats.zarr"
        root = open_group(str(store), mode="w")
        root.attrs["format_type"] = "gsplats_zarr"
        root.attrs["format_version"] = "3.4"
        return store

    def test_the_default_prints_one_line_and_the_hint(self, not_a_store: Path) -> None:
        result = self._run(["info", str(not_a_store)], traceback=False)
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
        result = self._run(["info", str(not_a_store)], traceback=True)
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        # The exception type from deep inside zarr, i.e. the actual origin —
        # not the CLI's one-line paraphrase of it.
        assert "GroupNotFoundError" in combined, (
            "expected the underlying exception, got:\n" + combined
        )

    def test_gsplat_info_uses_the_same_opt_in_path(
        self, unreadable_gsplat_store: Path
    ) -> None:
        command = ["gsplat", "info", str(unreadable_gsplat_store)]

        quiet = self._run(command, traceback=False)
        quiet_output = quiet.stdout + quiet.stderr
        assert quiet.returncode == 1, quiet_output
        assert "Error: 'centers'" in quiet_output
        assert TRACEBACK_ENV_VAR in quiet_output
        assert "Traceback" not in quiet_output

        loud = self._run(command, traceback=True)
        loud_output = loud.stdout + loud.stderr
        assert loud.returncode != 0
        assert "Traceback" in loud_output
        assert "KeyError: 'centers'" in loud_output
