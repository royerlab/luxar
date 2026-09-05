"""Shared isolation for CLI tests."""

from __future__ import annotations

import pytest

from luxar.cli._traceback import TRACEBACK_ENV_VAR


@pytest.fixture(autouse=True)
def _traceback_opt_in_is_test_local(monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not let a developer's shell preference change CLI test semantics."""
    monkeypatch.delenv(TRACEBACK_ENV_VAR, raising=False)
