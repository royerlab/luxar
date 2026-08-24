"""Tests for shared CLI test helpers."""

from __future__ import annotations

from dataclasses import dataclass

from luxar.cli.tests._testing import normalized_cli_output


@dataclass
class _SplitResult:
    stdout: str
    stderr: str


class _StdoutOnlyResult:
    stdout = "ordinary output"

    @property
    def stderr(self) -> str:
        raise ValueError("stderr was not separately captured")


def test_normalized_cli_output_combines_and_flattens_rich_streams() -> None:
    result = _SplitResult(
        stdout="\x1b[31m--compression\x1b[0m-factor",
        stderr="╭─ Error ─╮\n│ --refine\n│ volume cannot be combined │\n╰─────────╯",
    )

    output = normalized_cli_output(result)

    assert "--compression-factor" in output
    assert "--refine volume cannot be combined" in output
    assert "\x1b" not in output
    assert "│" not in output


def test_normalized_cli_output_tolerates_unavailable_stderr() -> None:
    assert normalized_cli_output(_StdoutOnlyResult()) == "ordinary output"
