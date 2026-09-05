"""Tests for the console-output seam (:mod:`luxar.verbosity`).

Two things are worth testing here and they are different in kind.

The first is the API contract: names resolve, the context manager restores, bad
input is refused. Ordinary unit tests.

The second is that the documented *effects* are real. The module's docstring
makes specific claims — ``"silent"`` prints nothing, ``max_depth=0`` is not
silence because arbol emits truncation notices — and those are claims about
another library's behaviour, which is exactly the kind of thing that is true
when written and false after an upgrade. So they are measured against captured
stdout rather than asserted against the switch values.
"""

from __future__ import annotations

import contextlib
import io
import math

import pytest
from arbol import Arbol, aprint, asection

import luxar
from luxar.verbosity import get_verbosity, set_verbosity, verbosity


@pytest.fixture(autouse=True)
def _restore_arbol():
    """Every test here writes process-global arbol state; put it back.

    Without this a failing test leaves the whole session muted, which would
    show up as unrelated tests losing their output rather than as this file
    failing.
    """
    saved = (Arbol.enable_output, Arbol.max_depth, Arbol.colorful)
    yield
    Arbol.enable_output, Arbol.max_depth, Arbol.colorful = saved


def _emit_a_three_deep_tree() -> str:
    """Print a known tree and return what reached stdout."""
    buffer = io.StringIO()
    Arbol.colorful = False  # ANSI codes would obscure the line count
    with contextlib.redirect_stdout(buffer):
        aprint("top-level line")
        with asection("section A"):
            aprint("inside A")
            with asection("section B"):
                aprint("inside B")
    return buffer.getvalue()


class TestTheApiContract:
    """Names, round-trips and refusals."""

    def test_the_default_is_full_and_matches_arbols_own_defaults(self) -> None:
        """`"full"` must be exactly what a caller who never asks gets.

        This is the one level that is a compatibility claim rather than a
        feature: if `"full"` ever stopped meaning arbol's defaults, importing
        Luxar would change the output of code that never called this API.
        """
        set_verbosity("full")
        assert Arbol.enable_output is True
        assert Arbol.max_depth == math.inf
        assert get_verbosity() == "full"

    @pytest.mark.parametrize("level", ["silent", "summary", "normal", "full"])
    def test_every_named_level_round_trips(self, level: str) -> None:
        set_verbosity(level)
        assert get_verbosity() == level

    def test_an_int_sets_an_explicit_depth(self) -> None:
        set_verbosity(7)
        assert Arbol.max_depth == 7
        assert Arbol.enable_output is True
        assert get_verbosity() == 7

    def test_silent_wins_over_depth_when_reporting(self) -> None:
        """A muted tree reports "silent", not "full".

        `"silent"` leaves `max_depth` at infinity, so a naive lookup that
        checked depth first would report an entirely muted process as `"full"`.
        """
        set_verbosity("silent")
        assert Arbol.max_depth == math.inf
        assert get_verbosity() == "silent"

    def test_a_hand_set_depth_is_reported_as_that_depth(self) -> None:
        """Roughly twenty bundled demos set `Arbol.max_depth` directly."""
        Arbol.enable_output = True
        Arbol.max_depth = 4
        assert get_verbosity() == 4

    def test_an_unknown_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Unknown verbosity level"):
            set_verbosity("loud")  # type: ignore[arg-type]

    def test_a_negative_depth_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be >= 0"):
            set_verbosity(-1)

    def test_a_bool_is_refused(self) -> None:
        """`set_verbosity(False)` reads as "quiet" but resolves to depth 0.

        Depth 0 is not silence (see the effect test below), so accepting a bool
        would do the surprising thing quietly. `bool` is an `int` subclass, so
        this needs an explicit check rather than falling out of the types.
        """
        with pytest.raises(TypeError, match="not a bool"):
            set_verbosity(False)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="not a bool"):
            set_verbosity(True)  # type: ignore[arg-type]


class TestTheContextManager:
    """Scoping and restoration."""

    def test_it_restores_the_previous_level(self) -> None:
        set_verbosity("normal")
        with verbosity("silent"):
            assert get_verbosity() == "silent"
        assert get_verbosity() == "normal"

    def test_it_restores_a_hand_set_depth_not_a_level_name(self) -> None:
        """Restore the exact previous pair, not the level it resolves to.

        A caller who set `Arbol.max_depth = 4` by hand (as the demos do) must
        get 4 back, not whichever named level happens to be nearest.
        """
        Arbol.enable_output = True
        Arbol.max_depth = 4
        with verbosity("silent"):
            pass
        assert Arbol.max_depth == 4
        assert Arbol.enable_output is True

    def test_it_restores_after_an_exception(self) -> None:
        set_verbosity("full")
        with pytest.raises(RuntimeError, match="boom"), verbosity("silent"):
            raise RuntimeError("boom")
        assert get_verbosity() == "full"

    def test_a_bad_level_does_not_change_anything(self) -> None:
        """The resolve happens before the write, so a refusal is a no-op."""
        set_verbosity("normal")
        with pytest.raises(ValueError):
            set_verbosity("nonsense")  # type: ignore[arg-type]
        assert get_verbosity() == "normal"


class TestTheDocumentedEffectsAreReal:
    """Claims about arbol's behaviour, measured against captured stdout."""

    def test_silent_prints_nothing_at_all(self) -> None:
        set_verbosity("silent")
        assert _emit_a_three_deep_tree() == ""

    def test_full_prints_the_whole_tree(self) -> None:
        set_verbosity("full")
        output = _emit_a_three_deep_tree()
        for expected in (
            "top-level line",
            "section A",
            "inside A",
            "section B",
            "inside B",
        ):
            assert expected in output, f"{expected!r} missing from full output"

    def test_summary_truncates_the_nested_sections(self) -> None:
        set_verbosity("summary")
        output = _emit_a_three_deep_tree()
        assert "top-level line" in output
        assert "section A" in output
        assert "inside B" not in output, (
            "depth 1 should not reach the second nesting level"
        )

    def test_each_step_down_the_ladder_prints_no_more_than_the_last(self) -> None:
        """Monotonic, which is the only thing that makes it a ladder."""
        counts = {}
        for level in ("silent", "summary", "normal", "full"):
            set_verbosity(level)
            counts[level] = len(
                [line for line in _emit_a_three_deep_tree().split("\n") if line.strip()]
            )
        assert counts["silent"] == 0
        assert (
            counts["silent"] <= counts["summary"] <= counts["normal"] <= counts["full"]
        )
        assert counts["summary"] < counts["full"], (
            f"summary and full print the same amount ({counts}) — the depth cap "
            f"is not taking effect, so the ladder is decorative"
        )

    def test_depth_zero_is_not_silence(self) -> None:
        """The reason `"silent"` uses `enable_output` and not `max_depth=0`.

        arbol at depth 0 still prints depth-0 lines AND a "log tree truncated
        here" notice per suppressed section — so a caller asking for quiet would
        get truncation noise instead. If a future arbol makes depth 0 silent,
        this fails and the module docstring's explanation should be revisited.
        """
        set_verbosity(0)
        output = _emit_a_three_deep_tree()
        assert output != "", "depth 0 became silent — revisit the docstring"
        assert "top-level line" in output


def test_the_seam_is_reachable_from_the_package_root() -> None:
    """The whole point is discoverability: `luxar.set_verbosity(...)`."""
    for name in ("set_verbosity", "get_verbosity", "verbosity"):
        assert hasattr(luxar, name), f"luxar.{name} is not exported"
        assert name in luxar.__all__, f"luxar.{name} is missing from __all__"


class TestItSilencesRealLuxarWork:
    """End-to-end: the finding was about Luxar's own output, not arbol's.

    Every test above drives `aprint`/`asection` directly, which proves the
    switches work but not that Luxar's 547 library-layer calls actually go
    through them. These write a real scene and read stdout — the thing a
    notebook user is complaining about.
    """

    @staticmethod
    def _write_a_scene(tmp_path, name: str) -> str:
        import numpy as np

        from luxar.core.dimensions import Dimensions
        from luxar.io.compiler import LuxarZarrCompiler

        buffer = io.StringIO()
        Arbol.colorful = False
        with contextlib.redirect_stdout(buffer):
            with LuxarZarrCompiler(tmp_path / f"{name}.luxar.zarr") as compiler:
                compiler.create_scene(dimensions=Dimensions.default_3d())
                compiler.write_points(
                    "P", positions=np.zeros((32, 3), dtype=np.float32)
                )
        return buffer.getvalue()

    def test_a_compile_is_chatty_by_default(self, tmp_path) -> None:
        """The control arm for the test below: there IS output to silence."""
        set_verbosity("full")
        output = self._write_a_scene(tmp_path, "chatty")
        assert output.strip(), (
            "a default compile printed nothing — the silent test below would "
            "pass vacuously"
        )
        assert "points" in output.lower()

    def test_the_same_compile_prints_nothing_under_silent(self, tmp_path) -> None:
        set_verbosity("silent")
        assert self._write_a_scene(tmp_path, "quiet") == ""

    def test_the_context_manager_scopes_it_to_one_call(self, tmp_path) -> None:
        """What a notebook user actually writes."""
        set_verbosity("full")
        with verbosity("silent"):
            assert self._write_a_scene(tmp_path, "scoped_quiet") == ""
        assert self._write_a_scene(tmp_path, "scoped_loud").strip()
