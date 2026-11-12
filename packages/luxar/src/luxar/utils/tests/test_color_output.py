"""arbol colour follows NO_COLOR / FORCE_COLOR and stays off in redirected logs."""

from __future__ import annotations

import io
import os
import subprocess
import sys
from typing import Iterator

import pytest
from arbol import Arbol

from luxar.utils.color_output import apply_arbol_color_policy, color_env_override

_SNIPPET = (
    "import luxar\n"
    "from arbol import aprint, asection\n"
    "with asection('section'):\n"
    "    aprint('line')\n"
)


@pytest.fixture(autouse=True)
def _restore_colorful(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    saved = Arbol.colorful
    yield
    Arbol.colorful = saved


class _Tty(io.TextIOWrapper):
    def isatty(self) -> bool:
        return True


def _tty() -> io.TextIOWrapper:
    return _Tty(io.BytesIO())


def test_redirected_stream_turns_color_off() -> None:
    Arbol.colorful = True
    apply_arbol_color_policy(io.TextIOWrapper(io.BytesIO()))
    assert Arbol.colorful is False


def test_terminal_keeps_color() -> None:
    Arbol.colorful = True
    apply_arbol_color_policy(_tty())
    assert Arbol.colorful is True


def test_notebook_style_stream_is_left_alone() -> None:
    """A non-TextIOWrapper stream (Jupyter's OutStream) renders ANSI itself."""
    Arbol.colorful = True
    apply_arbol_color_policy(io.StringIO())
    assert Arbol.colorful is True


def test_no_color_turns_color_off_on_a_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    Arbol.colorful = True
    apply_arbol_color_policy(_tty())
    assert Arbol.colorful is False


def test_force_color_wins_over_redirect_and_no_color(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("FORCE_COLOR", "1")
    Arbol.colorful = False
    apply_arbol_color_policy(io.TextIOWrapper(io.BytesIO()))
    assert Arbol.colorful is True


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, None),
        ({"NO_COLOR": ""}, None),
        ({"FORCE_COLOR": "0"}, None),
        ({"NO_COLOR": "1"}, False),
        ({"FORCE_COLOR": "true"}, True),
    ],
)
def test_env_override(
    monkeypatch: pytest.MonkeyPatch, env: dict, expected: object
) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert color_env_override() is expected


def _run(extra_env: dict) -> str:
    env = {k: v for k, v in os.environ.items() if k not in ("NO_COLOR", "FORCE_COLOR")}
    env.update(extra_env)
    out = subprocess.run(  # nosec B603 - fixed argv, no shell
        [sys.executable, "-c", _SNIPPET],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return out.stdout


def test_piped_output_has_no_ansi_codes() -> None:
    assert "line" in (stdout := _run({}))
    assert "\x1b[" not in stdout


def test_force_color_restores_ansi_codes_in_a_pipe() -> None:
    assert "\x1b[" in _run({"FORCE_COLOR": "1"})
