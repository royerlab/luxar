"""Tests for shared demo command-line flag parsers."""

import re
import sys
from pathlib import Path

from luxar.demos._support.runtime import flags as flag_utils


def _plain(captured: str) -> str:
    """Captured arbol output with ANSI colour and line wrapping removed.

    Both matter. Arbol writes SGR escapes around every line, and it wraps long
    ones — so a phrase can be split across a newline mid-sentence. A raw
    `in` check is therefore unreliable in the positive direction and, worse,
    passes VACUOUSLY in the negative direction.
    """
    without_colour = re.sub(r"\x1b\[[0-9;]*m", "", captured)
    return " ".join(without_colour.split())


def test_parse_int_arg_equals_space_and_default():
    assert flag_utils.parse_int_arg("points", 100, ["--points=4000"]) == 4000
    assert flag_utils.parse_int_arg("points", 100, ["--points", "2000"]) == 2000
    assert flag_utils.parse_int_arg("points", 100, ["--other=1"]) == 100
    assert flag_utils.parse_int_arg("points", 100, ["--points=bad"]) == 100


def test_parse_int_arg_none_default_and_parse_path_arg(capsys):
    # A None default returns None when the flag is absent, and still parses.
    assert flag_utils.parse_int_arg("max", None, ["--other=1"]) is None
    assert flag_utils.parse_int_arg("max", None, ["--max=5"]) == 5

    # A malformed value warns and falls back to the default — for a None default
    # (so the caller's "unset" sentinel survives) and for an int one.
    assert flag_utils.parse_int_arg("max", None, ["--max=abc"]) is None
    out = capsys.readouterr().out
    assert "--max" in out and "abc" in out and "None" in out

    assert flag_utils.parse_int_arg("points", 100, ["--points=bad"]) == 100
    out = capsys.readouterr().out
    assert "--points" in out and "bad" in out and "100" in out

    # parse_path_arg: absent → None; both flag forms parse (and expand ~).
    assert flag_utils.parse_path_arg("cache-dir", ["--x"]) is None
    assert flag_utils.parse_path_arg("cache-dir", ["--cache-dir=/tmp/foo"]) == Path(
        "/tmp/foo"
    )
    assert flag_utils.parse_path_arg("cache-dir", ["--cache-dir", "/tmp/foo"]) == Path(
        "/tmp/foo"
    )
    assert flag_utils.parse_path_arg("cache-dir", ["--cache-dir=~/foo"]) == (
        Path.home() / "foo"
    )


def test_parse_args_tolerate_pre_written_dashes():
    """A name passed as ``--points`` must not silently search for ``----points``."""
    assert flag_utils.parse_int_arg("--points", 100, ["--points=4000"]) == 4000
    assert flag_utils.parse_path_arg("--data", ["--data", "/tmp/foo"]) == Path(
        "/tmp/foo"
    )


def test_parse_path_arg_empty_value_reads_as_absent():
    """``--data=`` must not resolve to the current directory."""
    assert flag_utils.parse_path_arg("data", ["--data="]) is None
    assert flag_utils.parse_path_arg("data", ["--data", ""]) is None
    # An empty occurrence is skipped, not a hit: the scan keeps going.
    assert flag_utils.parse_path_arg("data", ["--data=", "--data=/tmp/foo"]) == Path(
        "/tmp/foo"
    )


def test_parse_path_arg_rejects_a_following_option_as_the_value(capsys):
    """``--data --no-tsp`` is a missing value, not a path called ``--no-tsp``."""
    assert flag_utils.parse_path_arg("data", ["--data", "--no-tsp"]) is None
    out = capsys.readouterr().out
    assert "--data" in out and "--no-tsp" in out

    # The scan continues past the value-less occurrence.
    assert flag_utils.parse_path_arg(
        "data", ["--data", "--no-tsp", "--data", "/tmp/foo"]
    ) == Path("/tmp/foo")

    # The explicit ``=`` form stays literal — that really is the path asked for.
    assert flag_utils.parse_path_arg("data", ["--data=--odd"]) == Path("--odd")


def test_parse_path_arg_reads_sys_argv_by_default(monkeypatch):
    """``argv=None`` scans the real ``sys.argv``."""
    monkeypatch.setattr(sys, "argv", ["prog", "--data=/tmp/bar"])
    assert flag_utils.parse_path_arg("data") == Path("/tmp/bar")
    assert flag_utils.parse_path_arg("cache-dir") is None


def test_parse_str_arg_equals_space_and_default(capsys):
    assert (
        flag_utils.parse_str_arg("control-token", ["--control-token=s3cret"])
        == "s3cret"
    )
    assert (
        flag_utils.parse_str_arg("control-token", ["--control-token", "s3cret"])
        == "s3cret"
    )
    assert flag_utils.parse_str_arg("control-token", ["--other=1"]) is None
    # An empty value is not a hit, so a later real one still wins.
    assert (
        flag_utils.parse_str_arg("host", ["--host=", "--host", "0.0.0.0"]) == "0.0.0.0"
    )
    # In the space form the next token must not itself look like an option.
    assert (
        flag_utils.parse_str_arg("control-token", ["--control-token", "--host"]) is None
    )
    assert "not a value" in capsys.readouterr().out


class TestControlServeArgs:
    """Remote control is OPT-IN. That is the property worth a test."""

    def test_off_without_the_flag(self):
        # The default must add nothing: no demo should ever start listening for
        # remote control because a user ran it the ordinary way.
        assert flag_utils.control_serve_args([]) == []
        assert flag_utils.control_serve_args(["--no-audio", "--no-turntables"]) == []

    def test_a_token_alone_does_not_enable_it(self):
        # `--control-token` is a modifier, not a switch. Treating it as one
        # would turn a half-typed command into an exposed display.
        assert flag_utils.control_serve_args(["--control-token", "s3cret"]) == []

    def test_the_flag_enables_it(self):
        assert flag_utils.control_serve_args(["--control"]) == ["--control"]

    def test_token_and_host_are_forwarded(self):
        assert flag_utils.control_serve_args(
            ["--control", "--control-token=s3cret", "--host=0.0.0.0"]
        ) == ["--control", "--control-token", "s3cret", "--host", "0.0.0.0"]

    def test_a_network_bind_without_a_token_warns(self, capsys):
        # The one dangerous combination: reachable from the network, with
        # nothing asked of whoever connects.
        args = flag_utils.control_serve_args(["--control", "--host", "0.0.0.0"])
        assert args == ["--control", "--host", "0.0.0.0"]
        warning = _plain(capsys.readouterr().out)
        assert "can drive the display" in warning
        assert "Pass --control-token to restrict it" in warning

    def test_a_network_bind_with_a_token_is_quiet(self, capsys):
        flag_utils.control_serve_args(
            ["--control", "--host", "0.0.0.0", "--control-token", "s3cret"]
        )
        assert "can drive the display" not in _plain(capsys.readouterr().out)

    def test_loopback_is_not_warned_about(self, capsys):
        flag_utils.control_serve_args(["--control", "--host", "127.0.0.1"])
        assert "can drive the display" not in _plain(capsys.readouterr().out)

    def test_it_reads_sys_argv_by_default(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog", "--control"])
        assert flag_utils.control_serve_args() == ["--control"]
        monkeypatch.setattr(sys, "argv", ["prog"])
        assert flag_utils.control_serve_args() == []
