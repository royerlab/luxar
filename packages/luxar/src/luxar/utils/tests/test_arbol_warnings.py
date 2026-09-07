"""Tests for arbol-routed warning display."""

from __future__ import annotations

import sys
import warnings
from contextlib import contextmanager
from typing import Iterator

import pytest
from arbol import Arbol

from luxar._zarr_compat import memory_group
from luxar.utils.arbol_warnings import (
    _arbol_showwarning,
    _default_display_active,
    arbol_warnings,
    install_arbol_warnings,
)


@contextmanager
def _default_display_sandbox() -> Iterator[None]:
    """Simulate a production process (Python's stock warning display).

    Under pytest, every test runs inside ``catch_warnings(record=True)``
    (pytest's per-item capture), so ``warnings._showwarnmsg_impl`` is a
    recorder and the override correctly refuses to engage. To test the
    engaged path we restore default-looking display inside a
    ``catch_warnings`` sandbox that puts everything back on exit.
    """

    def _impl(msg: object) -> None:  # stand-in for the stock displayer
        text = warnings._formatwarnmsg(msg)  # type: ignore[attr-defined]
        print(text, file=getattr(msg, "file", None) or sys.stderr, end="")

    _impl.__module__ = "warnings"
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = warnings._showwarning_orig  # type: ignore[attr-defined]
        warnings._showwarnmsg_impl = _impl  # type: ignore[attr-defined]
        yield


class TestArbolShowwarning:
    def test_formats_via_aprint(self, capsys: pytest.CaptureFixture) -> None:
        with _default_display_sandbox():
            _arbol_showwarning("splat drifted", UserWarning, "/a/b/encoder.py", 42)
        captured = capsys.readouterr()
        assert "⚠️" in captured.out
        assert "UserWarning" in captured.out
        assert "splat drifted" in captured.out
        assert "encoder.py:42" in captured.out
        assert "/a/b/" not in captured.out  # basename only, no raw stderr-style path
        assert captured.err == ""

    @pytest.mark.parametrize(
        ("enable_output", "depth", "max_depth"),
        [(False, 0, float("inf")), (True, 2, 1)],
    )
    def test_falls_back_to_stock_display_when_arbol_would_hide_warning(
        self,
        enable_output: bool,
        depth: int,
        max_depth: float,
        capsys: pytest.CaptureFixture,
    ) -> None:
        saved = (Arbol.enable_output, Arbol._depth, Arbol.max_depth)
        try:
            Arbol.enable_output = enable_output
            Arbol._depth = depth
            Arbol.max_depth = max_depth
            with _default_display_sandbox():
                _arbol_showwarning("still visible", UserWarning, "/a/b/encoder.py", 42)
        finally:
            Arbol.enable_output, Arbol._depth, Arbol.max_depth = saved

        captured = capsys.readouterr()
        assert captured.out == ""
        assert "UserWarning: still visible" in captured.err
        assert "/a/b/encoder.py:42" in captured.err


class TestArbolWarningsContext:
    def test_engages_when_default_display_active(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        with _default_display_sandbox():
            assert _default_display_active()
            with arbol_warnings():
                warnings.warn("engage-me", UserWarning, stacklevel=1)
        captured = capsys.readouterr()
        assert "⚠️" in captured.out
        assert "engage-me" in captured.out
        assert captured.err == ""

    def test_restores_previous_handler(self) -> None:
        with _default_display_sandbox():
            before = warnings.showwarning
            with arbol_warnings():
                assert warnings.showwarning is _arbol_showwarning
            assert warnings.showwarning is before

    def test_steps_aside_for_pytest_warns(self, capsys: pytest.CaptureFixture) -> None:
        # pytest.warns must still catch warnings raised inside the block —
        # the override must not steal them from the recorder.
        with pytest.warns(UserWarning, match="boom"):
            with arbol_warnings():
                warnings.warn("boom", UserWarning, stacklevel=1)
        assert "⚠️" not in capsys.readouterr().out

    def test_steps_aside_for_custom_hook(self) -> None:
        seen: list[str] = []

        def hook(message, category, filename, lineno, file=None, line=None):  # type: ignore[no-untyped-def]
            seen.append(str(message))

        with _default_display_sandbox():
            warnings.showwarning = hook
            with arbol_warnings():
                warnings.warn("custom-hook", UserWarning, stacklevel=1)
            assert warnings.showwarning is hook
        assert seen == ["custom-hook"]

    def test_reentrant(self, capsys: pytest.CaptureFixture) -> None:
        with _default_display_sandbox():
            with arbol_warnings():
                with arbol_warnings():
                    warnings.warn("nested", UserWarning, stacklevel=1)
                assert warnings.showwarning is _arbol_showwarning
        assert "nested" in capsys.readouterr().out

    def test_usable_as_decorator(self, capsys: pytest.CaptureFixture) -> None:
        @arbol_warnings()
        def noisy() -> str:
            warnings.warn("decorated", UserWarning, stacklevel=1)
            return "ok"

        with _default_display_sandbox():
            assert noisy() == "ok"
        assert "decorated" in capsys.readouterr().out


class TestInstall:
    def test_installs_when_default_display_active(self) -> None:
        with _default_display_sandbox():
            install_arbol_warnings()
            assert warnings.showwarning is _arbol_showwarning

    def test_skips_when_recorder_active(self) -> None:
        # With a recorder active, the process-wide install must refuse to
        # divert warnings from it. An explicit catch_warnings(record=True)
        # guarantees the recorder regardless of pytest's warnings plugin
        # (and restores state, so a wrongly-engaging install can't leak).
        with warnings.catch_warnings(record=True):
            before = warnings.showwarning
            install_arbol_warnings()
            assert warnings.showwarning is before


class TestPytestWarnsIntegration:
    def test_cholesky_escalation_still_catchable(self) -> None:
        """The compiler-path warning that motivated this module stays
        catchable through the decorated encoder path."""
        import numpy as np

        from luxar.encoding.encoder import ArrayEncoder

        rng = np.random.default_rng(0)
        n, ndim = 4000, 3
        # A few huge-sigma outliers stretch the per-column log range so the
        # uint8 certificate fails and escalates (mirrors
        # test_cholesky_split_quant.py::test_auto_escalates_to_u16_...).
        diag = rng.uniform(0.4, 5.0, size=(n, ndim)).astype(np.float32)
        diag[:10] = 1e8
        offdiag = (rng.standard_normal((n, ndim)) * 0.3).astype(np.float32)
        group = memory_group()
        with pytest.warns(UserWarning, match="escalating to uint16"):
            with arbol_warnings():
                ArrayEncoder().encode_cholesky_split(
                    zarr_group=group,
                    diag=diag,
                    offdiag=offdiag,
                    ndim=ndim,
                )
