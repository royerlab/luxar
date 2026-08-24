"""Shared helpers for CLI tests."""

from __future__ import annotations

import re
from typing import Protocol


class CliResult(Protocol):
    """The output surface shared by Click and Typer test results."""

    @property
    def stdout(self) -> str: ...

    @property
    def stderr(self) -> str: ...


_ANSI_CSI = re.compile(r"\x1b\[[0-9;:?]*[ -/]*[@-~]")
_RICH_BOX_DRAWING = re.compile(r"[\u2500-\u257f]")


def normalized_cli_output(result: CliResult) -> str:
    """Return assertion-friendly stdout and stderr from a CLI invocation.

    Rich may colour and hard-wrap Typer usage errors inside a panel, splitting
    message text with presentation codes and line breaks. Click versions also
    differ on whether stderr is separately available. Read both streams, strip
    presentation glyphs, and collapse whitespace so assertions remain stable.
    """
    streams = [result.stdout or ""]
    try:
        stderr = result.stderr or ""
    except (AttributeError, ValueError):
        stderr = ""
    if stderr:
        streams.append(stderr)

    # Joining before whitespace collapse can make a phrase span the stream boundary;
    # that is preferable to losing an error when Click captures stderr separately.
    text = _ANSI_CSI.sub("", "\n".join(streams))
    text = _RICH_BOX_DRAWING.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()
