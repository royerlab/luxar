"""Background-floor resolution on the DENOISED basis (issue #1178).

``--tiling none`` denoises the whole volume and then estimates ``--floor`` from
that denoised array, while the tiled paths denoise each tile and subtract a
level resolved from the RAW volume. Denoising collapses the noise tail and
shifts the histogram mode, so the two removed measurably different pedestals
from the same input. :func:`resolve_volume_floor_denoised` keeps the
whole-volume basis (one global level, #1174) and corrects it onto the denoised
basis with a bounded probe.
"""

import numpy as np
import pytest

# `luxar.gsplats.fitting.preprocessing` imports torch at module level (torch
# ships in the `gsplats` extra), so a core install must SKIP rather than ERROR.
try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")

if HAS_TORCH:
    from luxar.gsplats.fitting import preprocessing as pp
    from luxar.gsplats.fitting.preprocessing import (
        DENOISE_PROBE_BUDGET_VOXELS,
        _resolve_floor,
        _sample_blocks_for_denoise_probe,
        resolve_volume_floor,
        resolve_volume_floor_denoised,
    )

# Denoise settings shared by every test here: small, CPU-pinned and explicit so
# the probe pass costs ~0.1 s and the same numbers come out on any machine.
DENOISE_H = 0.15
DENOISE_PARAMS = {
    "patch_size": 3,
    "search_distance": 2,
    "backend": "pytorch",
    "device": "cpu",
    "use_2d": False,
}


def _skewed_background_volume(shape=(12, 28, 28)) -> np.ndarray:
    """A right-SKEWED noise pedestal (~100) plus one bright blob.

    Skew is what makes this test bite: the raw histogram mode of a gamma tail
    sits well below its mean, and denoising pulls the background toward that
    mean, so the mode moves by several intensity units. Symmetric Gaussian
    noise would barely move it and the bug would be invisible.
    """
    rng = np.random.default_rng(1178)
    vol = 100.0 + rng.gamma(2.0, 3.0, size=shape)
    grids = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    centre = [s / 2.0 for s in shape]
    r2 = sum((g - c) ** 2 for g, c in zip(grids, centre))
    vol = vol + 180.0 * np.exp(-r2 / (2 * 3.0**2))
    return vol.astype(np.float32)


def _denoise_whole(volume: np.ndarray) -> np.ndarray:
    """What the NON-TILED path does: denoise the entire volume up front."""
    from luxar.gsplats.preprocessing.denoise_pipeline import denoise_volume_array

    return denoise_volume_array(volume, h=DENOISE_H, **DENOISE_PARAMS)


class _CountingArray:
    """Lazy-array shim that records every ``__getitem__`` region it serves."""

    def __init__(self, data: np.ndarray) -> None:
        self._data = data
        self.regions: list = []

    @property
    def shape(self):  # noqa: D102 - array protocol
        return self._data.shape

    @property
    def ndim(self):  # noqa: D102 - array protocol
        return self._data.ndim

    def __getitem__(self, key):  # noqa: D105 - array protocol
        self.regions.append(key)
        return self._data[key]


class TestDenoisedBasisResolution:
    def test_level_is_the_denoised_estimate_not_the_raw_one(self) -> None:
        """``auto`` resolves to the DENOISED whole-volume mode, not the raw one."""
        volume = _skewed_background_volume()

        raw_level = resolve_volume_floor(volume, "auto")
        level = resolve_volume_floor_denoised(
            volume,
            "auto",
            denoise_h=DENOISE_H,
            denoise_params=DENOISE_PARAMS,
        )
        assert raw_level is not None and level is not None

        # The whole volume fits the probe budget, so the correction reproduces
        # the denoised whole-volume estimate.
        assert volume.size < DENOISE_PROBE_BUDGET_VOXELS
        expected = _resolve_floor(_denoise_whole(volume).ravel(), "auto")
        assert expected is not None
        # Bit-exact, not merely close: the correction is grouped so that a
        # whole-volume probe contributes a hard 0.0 offset.
        assert level == expected

        # ... and that is not what the raw basis says (the whole point of #1178).
        assert abs(level - raw_level) > 1.0

    def test_parity_between_tiled_and_non_tiled_bases(self) -> None:
        """The two ``--denoise`` orderings resolve ONE level (the issue's ask).

        Non-tiled: denoise the whole volume, then estimate (what
        ``_normalize_data`` sees). Tiled: estimate on the raw volume with the
        denoise correction. Before the fix the tiled side was plain
        ``resolve_volume_floor`` on the raw volume, which the assertion at the
        bottom shows is a different number.
        """
        volume = _skewed_background_volume()

        non_tiled = _resolve_floor(_denoise_whole(volume).ravel(), "auto")
        tiled = resolve_volume_floor_denoised(
            volume,
            "auto",
            denoise_h=DENOISE_H,
            denoise_params=DENOISE_PARAMS,
            guard_numeric=True,
        )
        assert non_tiled is not None and tiled is not None
        assert tiled == non_tiled  # tight: the same float, not merely close

        # Pre-fix behaviour, for the record: the raw basis disagrees.
        pre_fix = resolve_volume_floor(volume, "auto", guard_numeric=True)
        assert pre_fix is not None
        assert pre_fix != pytest.approx(non_tiled, rel=1e-3)

    def test_percentile_spec_is_corrected_with_the_percentile_estimator(self) -> None:
        """``pNN`` is corrected by the shift in that PERCENTILE, not the mode."""
        volume = _skewed_background_volume()

        level = resolve_volume_floor_denoised(
            volume,
            "p10",
            denoise_h=DENOISE_H,
            denoise_params=DENOISE_PARAMS,
        )
        expected = _resolve_floor(_denoise_whole(volume).ravel(), "p10")
        assert level is not None and expected is not None
        assert level == expected

        # Distinct from both the raw p10 and the denoised MODE, so this really
        # is the percentile estimator on the denoised basis.
        raw_p10 = resolve_volume_floor(volume, "p10")
        assert raw_p10 is not None and abs(level - raw_p10) > 1.0
        mode_level = resolve_volume_floor_denoised(
            volume,
            "auto",
            denoise_h=DENOISE_H,
            denoise_params=DENOISE_PARAMS,
        )
        assert mode_level is not None and level != pytest.approx(mode_level, rel=1e-6)

    def test_whole_volume_probe_is_never_declined(self, capsys) -> None:
        """A whole-volume probe disagrees with the volume by exactly 0.0.

        The basis-agreement check that declines an untrustworthy shift (see
        :class:`TestCroppedProbeRegime`) must never fire in the exact regime.
        """
        volume = _skewed_background_volume()
        assert volume.size < DENOISE_PROBE_BUDGET_VOXELS

        level = resolve_volume_floor_denoised(
            volume,
            "auto",
            denoise_h=DENOISE_H,
            denoise_params=DENOISE_PARAMS,
        )
        expected = _resolve_floor(_denoise_whole(volume).ravel(), "auto")
        assert level is not None and expected is not None
        assert level == expected  # still bit-exact
        assert "not transferable" not in capsys.readouterr().out

    def test_workers_agree_on_the_corrected_level(self) -> None:
        """Two independent resolutions of one volume reach the same level.

        This is what lets a ``--tile k/M`` subprocess and its parent subtract
        the same pedestal without coordinating: both the floor sample and the
        denoise probe are pure functions of ``volume.shape``.
        """
        volume = _skewed_background_volume()
        kwargs = {"denoise_h": DENOISE_H, "denoise_params": DENOISE_PARAMS}
        parent = resolve_volume_floor_denoised(volume, "auto", **kwargs)
        worker = resolve_volume_floor_denoised(_CountingArray(volume), "auto", **kwargs)
        assert parent is not None
        assert worker == parent


class TestDenoiseOffAndAbsoluteSpecs:
    """Nothing is corrected unless denoising is actually active on a spec."""

    @pytest.mark.parametrize("spec", ["auto", "p10", 105.0, "105.0", "none", None])
    def test_no_denoise_params_is_the_old_behaviour(self, spec) -> None:
        volume = _skewed_background_volume()
        assert resolve_volume_floor_denoised(
            volume, spec, guard_numeric=True
        ) == resolve_volume_floor(volume, spec, guard_numeric=True)
        # h without params (and params without h) must not half-correct either.
        assert resolve_volume_floor_denoised(
            volume, spec, denoise_h=DENOISE_H, guard_numeric=True
        ) == resolve_volume_floor(volume, spec, guard_numeric=True)
        assert resolve_volume_floor_denoised(
            volume, spec, denoise_params=DENOISE_PARAMS, guard_numeric=True
        ) == resolve_volume_floor(volume, spec, guard_numeric=True)

    @pytest.mark.parametrize("spec", [105.0, "105.0", "none", None, 0.0])
    def test_absolute_spec_is_never_probed(self, monkeypatch, spec) -> None:
        """A user absolute costs no denoise pass and (bar the guard) no read."""
        volume = _skewed_background_volume()

        def _boom(*_args, **_kwargs):
            raise AssertionError("an absolute --floor must not be probed")

        monkeypatch.setattr(
            "luxar.gsplats.preprocessing.denoise_pipeline.denoise_volume_array",
            _boom,
        )
        monkeypatch.setattr(pp, "_sample_blocks_for_denoise_probe", _boom)

        shim = _CountingArray(volume)
        level = resolve_volume_floor_denoised(
            shim,
            spec,
            denoise_h=DENOISE_H,
            denoise_params=DENOISE_PARAMS,
        )
        assert level == resolve_volume_floor(volume, spec)
        # guard_numeric defaults to False, so a numeric spec reads nothing.
        assert shim.regions == []


class TestGracefulDegradation:
    def test_failing_probe_denoise_falls_back_to_the_raw_level(
        self, monkeypatch, capsys
    ) -> None:
        """No torch / a dead backend must not break a fit — it degrades."""
        volume = _skewed_background_volume()

        def _boom(*_args, **_kwargs):
            raise RuntimeError("no NLM backend here")

        monkeypatch.setattr(
            "luxar.gsplats.preprocessing.denoise_pipeline.denoise_volume_array",
            _boom,
        )
        level = resolve_volume_floor_denoised(
            volume,
            "auto",
            denoise_h=DENOISE_H,
            denoise_params=DENOISE_PARAMS,
        )
        assert level == resolve_volume_floor(volume, "auto")
        out = capsys.readouterr().out
        assert "denoise floor probe failed" in out
        assert "RuntimeError" in out

    def test_empty_volume_has_no_level_to_correct(self) -> None:
        """An empty volume has no RAW level either, so nothing is probed.

        This exits at the ``level_raw is None`` early return (the floor sample of
        an empty array is ``None``), NOT at the degenerate-probe branch — the
        probe is never reached. It is here to pin that the wrapper adds no new
        crash on the empty input the raw resolver already handles.
        """
        empty = np.zeros((0, 4, 4), dtype=np.float32)
        assert (
            resolve_volume_floor_denoised(
                empty,
                "auto",
                denoise_h=DENOISE_H,
                denoise_params=DENOISE_PARAMS,
            )
            is None
        )

    def test_level_above_the_raw_sampled_max_is_refused(
        self, monkeypatch, capsys
    ) -> None:
        """The 'would erase all signal' guard, re-applied on the RAW sample's max.

        A stand-in "denoiser" that BRIGHTENS its input 4x makes the probe propose
        a huge upward shift, so the corrected level lands above the volume's
        sampled maximum and must be refused exactly as :func:`resolve_volume_floor`
        refuses an over-high raw level. The guard is deliberately NOT on the
        probe's own denoised max (see
        ``TestCroppedProbeRegime.test_signal_free_probe_does_not_veto_a_valid_level``),
        which here would be 4x the volume max and would wave this level through.
        """
        volume = _skewed_background_volume()

        def _brighten(block, **_kwargs):
            return np.asarray(block, dtype=np.float32) * 4.0

        monkeypatch.setattr(
            "luxar.gsplats.preprocessing.denoise_pipeline.denoise_volume_array",
            _brighten,
        )
        assert (
            resolve_volume_floor_denoised(
                volume,
                "auto",
                denoise_h=DENOISE_H,
                denoise_params=DENOISE_PARAMS,
            )
            is None
        )
        out = capsys.readouterr().out
        assert "would erase all signal" in out
        # Named against the raw sample's max (~the volume max), not 4x it.
        assert "sampled volume max" in out
        assert f"{float(volume.max()):.6g}" in out


class TestDenoiseProbeSampling:
    def test_volume_within_budget_is_one_whole_block(self) -> None:
        volume = _skewed_background_volume()
        blocks = _sample_blocks_for_denoise_probe(volume, DENOISE_PROBE_BUDGET_VOXELS)
        assert blocks is not None and len(blocks) == 1
        np.testing.assert_array_equal(blocks[0], volume)

    def test_oversized_volume_is_bounded_shape_preserving_and_deterministic(
        self,
    ) -> None:
        # A lazy array of 4M voxels against a 1M budget: nothing is read whole.
        data = np.arange(64 * 256 * 256, dtype=np.float32).reshape(64, 256, 256)
        shim = _CountingArray(data)
        blocks = _sample_blocks_for_denoise_probe(shim, 1_000_000)
        assert blocks is not None
        assert 1 < len(blocks) <= 3
        assert sum(b.size for b in blocks) <= 1_000_000
        # Blocks stay nD (3D NLM needs context along every axis) and are read as
        # contiguous slices, never strided.
        for block in blocks:
            assert block.ndim == data.ndim
            assert all(s > 1 for s in block.shape)
        assert all(
            all(isinstance(s, slice) and (s.step is None) for s in region)
            for region in shim.regions
        )
        # Pure function of shape+budget: a second pass reads the same regions.
        again = _sample_blocks_for_denoise_probe(_CountingArray(data), 1_000_000)
        assert again is not None
        for a, b in zip(blocks, again, strict=True):
            np.testing.assert_array_equal(a, b)

    def test_empty_volume_has_no_probe(self) -> None:
        assert (
            _sample_blocks_for_denoise_probe(
                np.zeros((0, 3, 3), dtype=np.float32), 1000
            )
            is None
        )


# Every test above probes the WHOLE volume as one block — the trivially exact
# regime. A real light-sheet stack is far above the 2M-voxel budget, so the probe
# is a handful of cubic CROPS from the middle of the volume, and the correction it
# proposes is only as good as the crops' agreement with the whole volume. The
# tests below live in that regime, reached cheaply by shrinking the budget instead
# of growing the volume.
_SMALL_PROBE_BUDGET = 6_000
_CROP_SHAPE = (24, 48, 48)


def _grids(shape=_CROP_SHAPE):
    return np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")


def _uniform_background_volume() -> np.ndarray:
    """Skewed pedestal, SPATIALLY UNIFORM, plus one centred blob.

    A centre crop of this volume sees the same background population the whole
    volume does, so the shift it measures transfers.
    """
    rng = np.random.default_rng(7)
    zz, yy, xx = _grids()
    vol = 100.0 + rng.gamma(2.0, 3.0, size=_CROP_SHAPE)
    vol = vol + 180.0 * np.exp(
        -(((zz - 12) ** 2 + (yy - 24) ** 2 + (xx - 24) ** 2) / (2 * 5.0**2))
    )
    return vol.astype(np.float32)


def _vignetted_volume() -> np.ndarray:
    """A VIGNETTED background (100 in the middle, 115 at the edges) + an off-centre blob.

    The background is not one population: a centre crop measures the bottom of
    the vignette while the whole-volume slab sample averages over all of it, so
    the two estimators disagree about the RAW level and the crop's denoise shift
    says nothing about the whole volume's.
    """
    rng = np.random.default_rng(1178)
    zz, yy, xx = _grids()
    r = np.sqrt(((yy - 23.5) / 23.5) ** 2 + ((xx - 23.5) / 23.5) ** 2)
    vol = 100.0 + 15.0 * r + rng.normal(0.0, 3.0, size=_CROP_SHAPE)
    vol = vol + 200.0 * np.exp(
        -(((zz - 6) ** 2 + (yy - 10) ** 2 + (xx - 10) ** 2) / (2 * 4.0**2))
    )
    return vol.astype(np.float32)


class TestCroppedProbeRegime:
    """Above the budget the probe is a CROP, which may not speak for the volume."""

    @staticmethod
    def _crop_the_probe(monkeypatch) -> None:
        monkeypatch.setattr(pp, "DENOISE_PROBE_BUDGET_VOXELS", _SMALL_PROBE_BUDGET)

    def test_correction_lands_closer_than_the_raw_level(self, monkeypatch) -> None:
        """On a uniform background the cropped probe still improves the level.

        The reference is what ``--tiling none`` computes: estimate on the whole
        DENOISED volume. The corrected level has to be measurably closer to it
        than the raw-basis level is — that is the whole point of the correction,
        and no test outside this regime shows it (inside the budget the corrected
        level simply IS the reference).
        """
        self._crop_the_probe(monkeypatch)
        volume = _uniform_background_volume()
        assert volume.size > _SMALL_PROBE_BUDGET

        reference = _resolve_floor(_denoise_whole(volume).ravel(), "auto")
        raw = resolve_volume_floor(volume, "auto")
        level = resolve_volume_floor_denoised(
            volume,
            "auto",
            denoise_h=DENOISE_H,
            denoise_params=DENOISE_PARAMS,
        )
        assert reference is not None and raw is not None and level is not None
        # Approximate, not exact (a crop is not the volume) — but closer.
        assert level != raw
        assert abs(level - reference) < abs(raw - reference)

    def test_declined_when_the_probe_disagrees_on_the_raw_level(
        self, monkeypatch, capsys
    ) -> None:
        """A probe that cannot reproduce the raw level may not move it.

        On a vignetted background the centre crop reads a lower raw level than the
        whole volume does, and that basis gap (~3 units here) dwarfs the denoise
        shift it measures (~0.6) — applying the shift would move the level FURTHER
        from the non-tiled reference. So it is declined, the raw-basis level is
        kept (exactly the pre-#1178 behaviour), and the decision is said out loud.
        """
        self._crop_the_probe(monkeypatch)
        volume = _vignetted_volume()

        reference = _resolve_floor(_denoise_whole(volume).ravel(), "auto")
        raw = resolve_volume_floor(volume, "auto")
        level = resolve_volume_floor_denoised(
            volume,
            "auto",
            denoise_h=DENOISE_H,
            denoise_params=DENOISE_PARAMS,
        )
        assert reference is not None and raw is not None
        assert level == raw

        out = capsys.readouterr().out
        assert "not transferable" in out
        assert f"{raw:.6g}" in out  # the note names both levels and the shift

        # The premise: the shift this probe proposed really would have hurt.
        blocks = _sample_blocks_for_denoise_probe(volume, _SMALL_PROBE_BUDGET)
        assert blocks is not None and len(blocks) > 1
        probe_raw = _resolve_floor(np.concatenate([b.ravel() for b in blocks]), "auto")
        probe_denoised = _resolve_floor(
            np.concatenate(
                [
                    np.asarray(_denoise_whole(b), dtype=np.float32).ravel()
                    for b in blocks
                ]
            ),
            "auto",
        )
        assert probe_raw is not None and probe_denoised is not None
        would_have_been = probe_denoised + (raw - probe_raw)
        assert abs(would_have_been - reference) > abs(raw - reference)

    def test_two_workers_agree_in_the_cropped_regime(self, monkeypatch) -> None:
        """Independent workers reach the same level from a cropped probe too.

        The determinism claim is about the SAMPLER, so it has to hold where the
        sampler actually chooses: same shape, same ``h``, same params, no
        coordination — one level.
        """
        self._crop_the_probe(monkeypatch)
        volume = _uniform_background_volume()
        kwargs = {"denoise_h": DENOISE_H, "denoise_params": dict(DENOISE_PARAMS)}

        first = resolve_volume_floor_denoised(volume, "auto", **kwargs)
        second = resolve_volume_floor_denoised(_CountingArray(volume), "auto", **kwargs)
        assert first is not None
        assert second == first

    def test_signal_free_probe_does_not_veto_a_valid_level(
        self, monkeypatch, capsys
    ) -> None:
        """A probe that sees no signal must not refuse the level the raw path applies.

        The probe crops the MIDDLE of the volume; on masked / zero-padded data
        (light-sheet fusion, a cropped FOV) that middle can be entirely empty
        while the rest of the volume carries the pedestal and the signal. Guarding
        the corrected level against the probe's own denoised max refuses the level
        outright in that case — no floor is subtracted for the whole run, which is
        worse than not correcting at all, and invisible under ``-j N``. The guard
        is on the RAW sample's max for exactly this reason.
        """
        self._crop_the_probe(monkeypatch)
        volume = _uniform_background_volume()
        # Blank exactly the regions the probe reads — asked of the sampler itself
        # rather than recomputed here, so the test cannot drift from it.
        shim = _CountingArray(volume)
        assert _sample_blocks_for_denoise_probe(shim, _SMALL_PROBE_BUDGET) is not None
        for region in shim.regions:
            volume[region] = 0.0
        probed = _sample_blocks_for_denoise_probe(volume, _SMALL_PROBE_BUDGET)
        assert probed is not None and all(not np.any(b) for b in probed)

        raw = resolve_volume_floor(volume, "auto")
        assert raw is not None and raw > 0.0
        level = resolve_volume_floor_denoised(
            volume,
            "auto",
            denoise_h=DENOISE_H,
            denoise_params=DENOISE_PARAMS,
        )
        assert level == raw
        assert "would erase all signal" not in capsys.readouterr().out


class TestVerboseReporting:
    def test_verbose_reports_the_accepted_correction(self, capsys) -> None:
        """The corrected level is logged with its provenance when asked for.

        Without this the only lines a run ever prints about the denoised basis
        are the failure notes, so an accepted correction — the normal case — is
        invisible in a log.
        """
        volume = _skewed_background_volume()
        raw = resolve_volume_floor(volume, "auto")
        level = resolve_volume_floor_denoised(
            volume,
            "auto",
            denoise_h=DENOISE_H,
            denoise_params=DENOISE_PARAMS,
            verbose=True,
        )
        assert raw is not None and level is not None

        out = capsys.readouterr().out
        assert "DENOISED basis" in out
        assert f"{level:.6g}" in out
        assert f"raw {raw:.6g}" in out  # the level it was corrected FROM
