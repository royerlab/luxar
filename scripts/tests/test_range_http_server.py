"""Tests for the viewer's Range-capable static fixture server."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "packages"
    / "luxar-viewer"
    / "tools"
    / "range-http-server.py"
)
_SPEC = importlib.util.spec_from_file_location("range_http_server", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_suffix_range_is_unsatisfiable_for_an_empty_file() -> None:
    assert _MODULE.parse_byte_range("bytes=-100", 0) is None
