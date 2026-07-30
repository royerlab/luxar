"""Tests for arbol-routed warning display."""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Iterator

import pytest

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
        pass

    _impl.__module__ = "warnings"
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = warnings._showwarning_orig  # type: ignore[attr-defined]
        warnings._showwarnmsg_impl = _impl  # type: ignore[attr-defined]
        yield


class TestArbolShowwarning:
    def test_formats_via_aprint(self, capsys: pytest.CaptureFixture) -> None:
        _arbol_showwarning("splat drifted", UserWarning, "/a/b/encoder.py", 42)
        out = capsys.readouterr().out
        assert "⚠️" in out
        assert "UserWarning" in out
        assert "splat drifted" in out
        assert "encoder.py:42" in out
        assert "/a/b/" not in out  # basename only, no raw stderr-style path


class TestArbolWarningsContext:
    def test_engages_when_default_display_active(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        with _default_display_sandbox():
            assert _default_display_active()
            with arbol_warnings():
                warnings.warn("engage-me", UserWarning, stacklevel=1)
        out = capsys.readouterr().out
        assert "⚠️" in out
        assert "engage-me" in out

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
        # Under pytest's per-item capture a recorder is active, so the
        # process-wide install must refuse to divert warnings from it.
        before = warnings.showwarning
        install_arbol_warnings()
        assert warnings.showwarning is before


class TestPytestWarnsIntegration:
    def test_cholesky_escalation_still_catchable(self) -> None:
        """The compiler-path warning that motivated this module stays
        catchable through the decorated encoder path."""
        import numpy as np
        import zarr

        from luxar.encoding.encoder import ArrayEncoder

        rng = np.random.default_rng(0)
        n, ndim = 4000, 3
        # A few huge-sigma outliers stretch the per-column log range so the
        # uint8 certificate fails and escalates (mirrors
        # test_cholesky_split_quant.py::test_auto_escalates_to_u16_...).
        diag = rng.uniform(0.4, 5.0, size=(n, ndim)).astype(np.float32)
        diag[:10] = 1e8
        offdiag = (rng.standard_normal((n, ndim)) * 0.3).astype(np.float32)
        group = zarr.group(store=zarr.MemoryStore())
        with pytest.warns(UserWarning, match="escalating to uint16"):
            with arbol_warnings():
                ArrayEncoder().encode_cholesky_split(
                    zarr_group=group,
                    diag=diag,
                    offdiag=offdiag,
                    ndim=ndim,
                )
