"""Shared helpers for CLI tests."""

from __future__ import annotations

import re
from typing import Protocol


class CliResult(Protocol):
    """The output surface shared by Click and Typer test results."""

    stdout: str

    @property
    def stderr(self) -> str: ...


_ANSI_CSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_RICH_BOX_DRAWING = re.compile(r"[\u2500-\u257f]")


def normalized_cli_output(result: CliResult) -> str:
    """Return assertion-friendly stdout and stderr from a CLI invocation.

    Click versions differ on whether ``result.output`` includes stderr. Typer
    writes usage errors there, and Rich may colour and wrap their text inside a
    box. Read both streams explicitly, strip presentation glyphs, and collapse
    whitespace so message assertions remain stable across those versions.
    """
    streams = [result.stdout or ""]
    try:
        stderr = result.stderr or ""
    except (AttributeError, ValueError):
        stderr = ""
    if stderr:
        streams.append(stderr)

    text = _ANSI_CSI.sub("", "\n".join(streams))
    text = _RICH_BOX_DRAWING.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()
