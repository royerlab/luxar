"""Tests for shared demo command-line flag parsers."""

import sys
from pathlib import Path

from luxar.demos._support.runtime import flags as flag_utils


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
