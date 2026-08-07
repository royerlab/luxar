"""Unit tests for the shared ``--show-roundtrip`` figure builder.

Exercises ``luxar.demos._roundtrip_common.show_roundtrip_comparison`` on real
(tiny) ``GSplatData`` objects rendered on the CPU — no GPU, no network. Only
``matplotlib.pyplot`` is faked, because it is an external dependency and CI has
no display; the fake records every panel's image array, colormap and display
range so the assertions can pin what the five demos used to build inline:

* one subplot ROW per entry in ``volumes``, three columns, ``squeeze=False``
* per-row titles built from the *passed-in* ``channel_names`` (the parameter
  that replaced the demos' module-level ``CHANNELS`` global); extra names are
  unused, while a ``gsplats_list`` or ``channel_names`` shorter than
  ``volumes`` is a ``ValueError`` rather than a half-drawn figure
* the panel CONTENTS: original / reconstruction mid-z slices in that order,
  and a genuine absolute difference, each with its pinned ``vmin``/``vmax``
* the PSNR/MSE arithmetic, against the reference formula recomputed here
* the ``device`` argument reaching ``render_to_volume`` (was ``DEVICE``)
* quiet, non-raising early returns for no volumes and for missing matplotlib
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pytest
from arbol import Arbol

from luxar.demos import _roundtrip_common
from luxar.demos._dependencies import MissingDependencyError
from luxar.demos._roundtrip_common import show_roundtrip_comparison
from luxar.gsplats.gsplat_data import GSplatData

# Deliberately NON-cubic and with distinct extents, so a transposed or
# reversed ``shape`` argument cannot pass unnoticed and ``shape[0] // 2``
# is distinguishable from the other axes' midpoints.
VOLUME_SHAPE = (6, 8, 10)
MID_Z = VOLUME_SHAPE[0] // 2  # 3


@pytest.fixture(autouse=True)
def _pin_arbol_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin Arbol's depth limit so console assertions don't depend on import order.

    ``Arbol.max_depth`` is global and 36 demo modules set it at import time, so
    whichever ran last wins. The helper prints its PSNR/MSE line two ``asection``
    levels deep, which needs a depth of at least 2; the lowest value any demo
    sets is 3, leaving a margin of a single level. Pin it rather than bank on
    that margin surviving the next demo.
    """
    monkeypatch.setattr(Arbol, "max_depth", 100)


# =============================================================================
# Fake matplotlib.pyplot (external dependency; no display in CI)
# =============================================================================


class _Recorder:
    """Collects everything the helper draws."""

    def __init__(self) -> None:
        self.subplots_calls: list[dict[str, Any]] = []
        self.images: dict[tuple[int, int], np.ndarray] = {}
        self.cmaps: dict[tuple[int, int], Optional[str]] = {}
        self.ranges: dict[tuple[int, int], tuple[Any, Any]] = {}
        self.titles: dict[tuple[int, int], str] = {}
        self.suptitle: Optional[str] = None
        self.colorbars = 0
        self.tight_layout_calls = 0
        self.show_calls = 0


class _FakeAxis:
    def __init__(self, rec: _Recorder, row: int, col: int) -> None:
        self._rec, self._row, self._col = rec, row, col

    def imshow(self, image: np.ndarray, **kwargs: Any) -> object:
        assert image.ndim == 2, "each panel must show a 2-D slice"
        key = (self._row, self._col)
        self._rec.images[key] = np.array(image, copy=True)
        self._rec.cmaps[key] = kwargs.get("cmap")
        self._rec.ranges[key] = (kwargs.get("vmin"), kwargs.get("vmax"))
        return object()  # stands in for the mappable passed to colorbar()

    def set_title(self, title: str) -> None:
        self._rec.titles[(self._row, self._col)] = title

    def axis(self, *_args: Any) -> None:
        pass


class _FakeFigure:
    def __init__(self, rec: _Recorder) -> None:
        self._rec = rec

    def colorbar(self, _mappable: object, **_kwargs: Any) -> None:
        self._rec.colorbars += 1

    def suptitle(self, text: str, **_kwargs: Any) -> None:
        self._rec.suptitle = text


class _FakePyplot:
    def __init__(self, rec: _Recorder) -> None:
        self._rec = rec

    def subplots(
        self, nrows: int, ncols: int, **kwargs: Any
    ) -> tuple[_FakeFigure, np.ndarray]:
        self._rec.subplots_calls.append({"nrows": nrows, "ncols": ncols, **kwargs})
        axes = np.empty((nrows, ncols), dtype=object)
        for r in range(nrows):
            for c in range(ncols):
                axes[r, c] = _FakeAxis(self._rec, r, c)
        return _FakeFigure(self._rec), axes

    def tight_layout(self) -> None:
        self._rec.tight_layout_calls += 1

    def show(self) -> None:
        self._rec.show_calls += 1


@pytest.fixture()
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    """Patch ``require_module`` so the helper gets the recording pyplot."""
    rec = _Recorder()
    fake = _FakePyplot(rec)

    def _require(module: str, **_kwargs: Any) -> object:
        assert module == "matplotlib.pyplot"
        return fake

    monkeypatch.setattr(_roundtrip_common, "require_module", _require)
    return rec


# =============================================================================
# Tiny real gsplats (no mocking of Luxar's own code)
# =============================================================================


def _make_gsplats(n: int = 3, *, seed: int = 0) -> GSplatData:
    """A handful of unit-ish 3-D Gaussians well inside ``VOLUME_SHAPE``."""
    rng = np.random.default_rng(seed)
    low = np.full(3, 1.5)
    high = np.asarray(VOLUME_SHAPE, dtype=float) - 1.5
    centers = rng.uniform(low, high, (n, 3)).astype(np.float32)
    # Pin one splat to the mid-z plane so that slice always carries real
    # signal: it is the slice every panel assertion reads, and an all-zero
    # mid-z slice would make several of them vacuously true.
    centers[0, 0] = float(MID_Z)
    amplitudes = rng.uniform(0.4, 1.0, n).astype(np.float32)
    # Lower-triangular Cholesky factor of a 3x3 (6 entries), isotropic.
    cholesky = np.tile(
        np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32), (n, 1)
    )
    return GSplatData(centers=centers, amplitudes=amplitudes, cholesky_factors=cholesky)


def _render(gsplats: GSplatData) -> np.ndarray:
    """Render on the CPU exactly as the helper will, and refuse a blank volume."""
    volume = gsplats.render_to_volume(shape=VOLUME_SHAPE, device="cpu")
    assert volume.shape == VOLUME_SHAPE
    assert volume.max() > 0, "synthetic splats fell outside the volume box"
    assert volume[MID_Z].max() > 0, "the mid-z slice must carry signal"
    return volume


def _make_exact_pair(seed: int) -> tuple[np.ndarray, GSplatData]:
    """``(volume, gsplats)`` where the volume IS the splats' own render.

    The round trip is therefore lossless: MSE 0, PSNR inf.
    """
    gsplats = _make_gsplats(seed=seed)
    return _render(gsplats), gsplats


def _make_mismatched_pair(seed: int) -> tuple[np.ndarray, GSplatData, np.ndarray]:
    """``(volume, gsplats, recon)`` where the volume differs from the render.

    Needed to pin panel CONTENTS and the metrics: with an exact round trip the
    original and reconstruction panels are identical and the difference panel
    is all zeros, so neither a swapped pair nor a broken difference would show.

    The perturbation is SIGNED and the mid-z slice is forced to straddle the
    render in both directions. With strictly positive noise ``volume >= recon``
    everywhere, ``np.abs`` is a no-op, and the difference panel could be
    computed as a plain ``volume - recon`` without any test noticing.
    """
    gsplats = _make_gsplats(seed=seed)
    recon = _render(gsplats)
    rng = np.random.default_rng(seed + 1000)
    delta = rng.uniform(0.08, 0.30, VOLUME_SHAPE)
    signs = rng.choice([-1.0, 1.0], VOLUME_SHAPE)
    volume = np.clip(recon + signs * delta, 0.0, 1.0).astype(np.float32)
    # Clipping at 0 turns most negative perturbations into equality (the
    # background is 0), so pull the brightest mid-z voxel down explicitly to
    # guarantee at least one strictly-negative signed difference there.
    peak = np.unravel_index(int(np.argmax(recon[MID_Z])), recon[MID_Z].shape)
    volume[(MID_Z, *peak)] = np.float32(recon[(MID_Z, *peak)] * 0.5)

    signed = volume[MID_Z] - recon[MID_Z]
    assert (signed < 0).any(), "np.abs must matter: no voxel is below the render"
    assert (signed > 0).any(), "no voxel is above the render"
    # The mid-z assertions are only meaningful if the z-slices differ.
    assert not np.array_equal(volume[0], volume[MID_Z])
    assert not np.array_equal(volume, recon)
    return volume, gsplats, recon


def _reference_metrics(volume: np.ndarray, recon: np.ndarray) -> tuple[float, float]:
    """The helper's PSNR/MSE formula, recomputed independently."""
    mse = float(np.mean((volume - recon) ** 2))
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
    return psnr, mse


class _DeviceSpy:
    """Wraps a real ``GSplatData``, recording the device it is rendered on.

    Not a mock: ``render_to_volume`` still runs the real renderer.
    """

    def __init__(self, inner: GSplatData) -> None:
        self._inner = inner
        self.devices: list[Optional[str]] = []

    @property
    def amplitudes(self) -> np.ndarray:
        return self._inner.amplitudes

    def render_to_volume(
        self, shape: tuple[int, ...], device: Optional[str] = None
    ) -> np.ndarray:
        self.devices.append(device)
        return self._inner.render_to_volume(shape=shape, device="cpu")


# =============================================================================
# Tests
# =============================================================================


class TestLayout:
    def test_single_channel_builds_one_row_of_three_panels(
        self, recorder: _Recorder
    ) -> None:
        volume, gsplats = _make_exact_pair(seed=1)

        show_roundtrip_comparison([volume], [gsplats], ["Nuclei"], device="cpu")

        assert len(recorder.subplots_calls) == 1
        call = recorder.subplots_calls[0]
        assert (call["nrows"], call["ncols"]) == (1, 3)
        # squeeze=False is load-bearing: the helper indexes axes[i, j].
        assert call["squeeze"] is False
        assert recorder.titles[(0, 0)] == "Original — Nuclei"
        assert recorder.titles[(0, 2)] == "|Difference|"
        assert recorder.colorbars == 1
        assert recorder.tight_layout_calls == 1
        assert recorder.show_calls == 1

    def test_row_count_and_titles_follow_the_channels(
        self, recorder: _Recorder
    ) -> None:
        # The multi-channel path: two rows, each titled from channel_names.
        channels = [_make_exact_pair(seed=s) for s in (2, 3)]
        volumes = [v for v, _ in channels]
        gsplats_list = [g for _, g in channels]

        show_roundtrip_comparison(volumes, gsplats_list, ["WGA", "Actin"], device="cpu")

        call = recorder.subplots_calls[0]
        assert (call["nrows"], call["ncols"]) == (2, 3)
        assert call["figsize"] == (14, 9.0)  # 4.5 per channel row
        assert recorder.titles[(0, 0)] == "Original — WGA"
        assert recorder.titles[(1, 0)] == "Original — Actin"
        assert recorder.colorbars == 2

    def test_extra_channel_names_are_unused(self, recorder: _Recorder) -> None:
        # The row count follows ``volumes``, not ``channel_names``, so a caller
        # may hand over its whole channel table even when fewer volumes were
        # loaded (the demos all pass ``[c["name"] for c in CHANNELS]``).
        channels = [_make_exact_pair(seed=s) for s in (4, 5)]
        volumes = [v for v, _ in channels]
        gsplats_list = [g for _, g in channels]

        show_roundtrip_comparison(
            volumes, gsplats_list, ["Nuclei", "WGA", "Actin"], device="cpu"
        )

        call = recorder.subplots_calls[0]
        assert (call["nrows"], call["ncols"]) == (2, 3)
        assert recorder.titles[(0, 0)] == "Original — Nuclei"
        assert recorder.titles[(1, 0)] == "Original — WGA"
        assert (2, 0) not in recorder.titles

    def test_suptitle_reports_slice_and_total_splat_count(
        self, recorder: _Recorder
    ) -> None:
        channels = [_make_exact_pair(seed=s) for s in (6, 7)]
        volumes = [v for v, _ in channels]
        gsplats_list = [g for _, g in channels]

        show_roundtrip_comparison(volumes, gsplats_list, ["a", "b"], device="cpu")

        total = sum(len(g.amplitudes) for g in gsplats_list)
        assert recorder.suptitle is not None
        assert f"z-slice {MID_Z}" in recorder.suptitle
        assert f"{total:,} total splats" in recorder.suptitle

    def test_extra_gsplats_are_not_counted_in_the_total(
        self, recorder: _Recorder
    ) -> None:
        # The total must describe the rows actually drawn. Surplus entries are
        # ignored the same way surplus channel_names are.
        volume, gsplats = _make_exact_pair(seed=27)
        spare = _make_gsplats(n=7, seed=28)

        show_roundtrip_comparison(
            [volume], [gsplats, spare], ["drawn", "unused"], device="cpu"
        )

        assert recorder.subplots_calls[0]["nrows"] == 1
        assert recorder.suptitle is not None
        assert f"({len(gsplats.amplitudes):,} total splats)" in recorder.suptitle

    def test_rows_of_different_depth_each_use_their_own_mid_slice(
        self, recorder: _Recorder
    ) -> None:
        # Each row slices its own volume, so the suptitle must name every slice
        # drawn rather than whichever one the last row left behind.
        tall, gsplats = _make_exact_pair(seed=29)
        short = tall[:4]  # depth 4 -> mid-z 2, against the tall volume's 3

        show_roundtrip_comparison(
            [tall, short], [gsplats, gsplats], ["tall", "short"], device="cpu"
        )

        assert np.array_equal(recorder.images[(0, 0)], tall[3])
        assert np.array_equal(recorder.images[(1, 0)], short[2])
        assert recorder.suptitle is not None
        assert "z-slice 3/2" in recorder.suptitle


class TestPanelContents:
    def test_columns_show_original_then_reconstruction_at_mid_z(
        self, recorder: _Recorder
    ) -> None:
        volume, gsplats, recon = _make_mismatched_pair(seed=20)

        show_roundtrip_comparison([volume], [gsplats], ["ch"], device="cpu")

        # Column 0 is the ORIGINAL and column 1 the RECONSTRUCTION, in that
        # order, each at the mid-z slice the suptitle advertises.
        assert np.array_equal(recorder.images[(0, 0)], volume[MID_Z])
        assert np.array_equal(recorder.images[(0, 1)], recon[MID_Z])
        assert recorder.cmaps[(0, 0)] == "gray"
        assert recorder.cmaps[(0, 1)] == "gray"

    def test_difference_panel_is_the_absolute_difference(
        self, recorder: _Recorder
    ) -> None:
        volume, gsplats, recon = _make_mismatched_pair(seed=21)

        show_roundtrip_comparison([volume], [gsplats], ["ch"], device="cpu")

        expected = np.abs(volume[MID_Z] - recon[MID_Z])
        assert np.allclose(recorder.images[(0, 2)], expected)
        # The fixture straddles the render in both directions, so a signed
        # difference would show up as negative values here.
        assert recorder.images[(0, 2)].min() >= 0
        assert (volume[MID_Z] - recon[MID_Z]).min() < 0, "fixture lost its straddle"
        assert recorder.cmaps[(0, 2)] == "inferno"

    def test_display_ranges_are_pinned_per_column(self, recorder: _Recorder) -> None:
        # Without an explicit range matplotlib autoscales each panel, so the
        # original and reconstruction would no longer be visually comparable
        # and the difference map would exaggerate noise.
        volume, gsplats, _ = _make_mismatched_pair(seed=22)

        show_roundtrip_comparison([volume], [gsplats], ["ch"], device="cpu")

        assert recorder.ranges[(0, 0)] == (0, 1)
        assert recorder.ranges[(0, 1)] == (0, 1)
        assert recorder.ranges[(0, 2)] == (0, 0.3)

    def test_each_row_shows_its_own_channel(self, recorder: _Recorder) -> None:
        a_volume, a_gsplats, a_recon = _make_mismatched_pair(seed=23)
        b_volume, b_gsplats, b_recon = _make_mismatched_pair(seed=24)

        show_roundtrip_comparison(
            [a_volume, b_volume], [a_gsplats, b_gsplats], ["a", "b"], device="cpu"
        )

        assert np.array_equal(recorder.images[(0, 0)], a_volume[MID_Z])
        assert np.array_equal(recorder.images[(0, 1)], a_recon[MID_Z])
        assert np.array_equal(recorder.images[(1, 0)], b_volume[MID_Z])
        assert np.array_equal(recorder.images[(1, 1)], b_recon[MID_Z])


class TestMetrics:
    def test_exact_reconstruction_reports_infinite_psnr(
        self, recorder: _Recorder, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # volume == render(gsplats), so MSE is 0 and PSNR is inf by the
        # helper's own ``mse > 0`` guard.
        volume, gsplats = _make_exact_pair(seed=8)

        show_roundtrip_comparison([volume], [gsplats], ["ch"], device="cpu")

        assert "PSNR: inf dB, MSE: 0" in capsys.readouterr().out
        assert recorder.titles[(0, 1)] == "Reconstructed (PSNR inf dB)"

    def test_psnr_and_mse_match_the_reference_formula(
        self, recorder: _Recorder, capsys: pytest.CaptureFixture[str]
    ) -> None:
        volume, gsplats, recon = _make_mismatched_pair(seed=9)
        expected_psnr, expected_mse = _reference_metrics(volume, recon)
        assert 0.0 < expected_psnr < 60.0, "fixture should be a lossy round trip"

        show_roundtrip_comparison([volume], [gsplats], ["ch"], device="cpu")

        # Console line: 2 decimals for PSNR, %.6g for MSE — both pinned, so a
        # dropped MSE or a changed formula/base cannot pass.
        assert (
            f"PSNR: {expected_psnr:.2f} dB, MSE: {expected_mse:.6g}"
            in capsys.readouterr().out
        )
        # Panel title: 1 decimal.
        title = recorder.titles[(0, 1)]
        assert title == f"Reconstructed (PSNR {expected_psnr:.1f} dB)"
        parsed = float(title.removeprefix("Reconstructed (PSNR ").split(" dB")[0])
        assert parsed == pytest.approx(expected_psnr, abs=0.05)

    def test_metrics_are_computed_per_channel(
        self, recorder: _Recorder, capsys: pytest.CaptureFixture[str]
    ) -> None:
        a_volume, a_gsplats, a_recon = _make_mismatched_pair(seed=25)
        b_volume, b_gsplats, b_recon = _make_mismatched_pair(seed=26)
        a_psnr, _ = _reference_metrics(a_volume, a_recon)
        b_psnr, _ = _reference_metrics(b_volume, b_recon)

        show_roundtrip_comparison(
            [a_volume, b_volume], [a_gsplats, b_gsplats], ["a", "b"], device="cpu"
        )

        out = capsys.readouterr().out
        assert f"PSNR: {a_psnr:.2f} dB" in out
        assert f"PSNR: {b_psnr:.2f} dB" in out
        assert recorder.titles[(0, 1)] == f"Reconstructed (PSNR {a_psnr:.1f} dB)"
        assert recorder.titles[(1, 1)] == f"Reconstructed (PSNR {b_psnr:.1f} dB)"


class TestDeviceForwarding:
    def test_device_argument_reaches_render_to_volume(
        self, recorder: _Recorder
    ) -> None:
        volume, gsplats = _make_exact_pair(seed=10)
        spy = _DeviceSpy(gsplats)

        show_roundtrip_comparison(
            [volume],
            [spy],  # type: ignore[list-item]
            ["ch"],
            device="cpu",
        )

        assert spy.devices == ["cpu"]

    def test_device_defaults_to_none_for_auto_detection(
        self, recorder: _Recorder
    ) -> None:
        volume, gsplats = _make_exact_pair(seed=11)
        spy = _DeviceSpy(gsplats)

        show_roundtrip_comparison([volume], [spy], ["ch"])  # type: ignore[list-item]

        assert spy.devices == [None]


class TestGuards:
    def test_too_few_channel_names_raises(self, recorder: _Recorder) -> None:
        # Promoting the names from a module global to a parameter made an
        # under-length list possible; half-drawing the figure (a reserved but
        # blank row, no warning) would be the worst outcome.
        channels = [_make_exact_pair(seed=s) for s in (13, 14)]
        volumes = [v for v, _ in channels]
        gsplats_list = [g for _, g in channels]

        with pytest.raises(ValueError, match=r"channel_names has 1"):
            show_roundtrip_comparison(volumes, gsplats_list, ["only-one"])

        assert recorder.subplots_calls == []

    def test_too_few_gsplats_raises(self, recorder: _Recorder) -> None:
        # ``gsplats_list`` is consumed by the same two ``zip``s, so a short one
        # truncates just as silently — same blank row, plus a suptitle total
        # that understates the splat count.
        channels = [_make_exact_pair(seed=s) for s in (15, 16)]
        volumes = [v for v, _ in channels]

        with pytest.raises(ValueError, match=r"gsplats_list has 1"):
            show_roundtrip_comparison(
                volumes, [channels[0][1]], ["a", "b"], device="cpu"
            )

        assert recorder.subplots_calls == []

    def test_both_short_lists_are_named_together(self, recorder: _Recorder) -> None:
        channels = [_make_exact_pair(seed=s) for s in (17, 18)]
        volumes = [v for v, _ in channels]

        with pytest.raises(ValueError) as excinfo:
            show_roundtrip_comparison(volumes, [], [], device="cpu")

        message = str(excinfo.value)
        assert "gsplats_list has 0" in message
        assert "channel_names has 0" in message
        assert "2 volumes" in message

    def test_no_volumes_returns_without_drawing(
        self, recorder: _Recorder, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Used to die with UnboundLocalError on the suptitle's ``mid_z``.
        assert show_roundtrip_comparison([], [], []) is None

        assert recorder.subplots_calls == []
        assert "Skipping --show-roundtrip" in capsys.readouterr().out


class TestMissingMatplotlib:
    def test_returns_quietly_without_drawing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        def _raise(module: str, **_kwargs: Any) -> object:
            raise MissingDependencyError(f"{module} is not installed")

        monkeypatch.setattr(_roundtrip_common, "require_module", _raise)
        volume, gsplats = _make_exact_pair(seed=12)

        # Must not raise, and must not attempt any rendering.
        assert show_roundtrip_comparison([volume], [gsplats], ["ch"]) is None
        out = capsys.readouterr().out
        assert "Skipping --show-roundtrip" in out
        assert "PSNR" not in out
