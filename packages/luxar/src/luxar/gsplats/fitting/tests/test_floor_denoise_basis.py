"""Background-floor resolution on the DENOISED basis (issue #1178).

``--tiling none`` denoises the whole volume and then estimates ``--floor`` from
that denoised array, while the tiled paths denoise each tile and subtract a
level resolved from the RAW volume. Denoising collapses the noise tail and
shifts the histogram mode, so the two removed measurably different pedestals
from the same input. :func:`resolve_volume_floor_denoised` keeps the
whole-volume basis (one global level, #1174) and corrects it onto the denoised
basis with a bounded probe — for any volume-derived spec while the probe covers
the whole volume, and above that budget for a ``pNN`` spec only, the default
``auto`` keeping its raw basis because a crop-measured histogram-mode shift is
dominated by noise. Every test here uses PRODUCTION-shaped denoise params
(:func:`_prod_params`, ``norm_range`` included); omitting ``norm_range`` measures
a configuration the pipeline never runs.
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
_BASE_DENOISE_PARAMS = {
    "patch_size": 3,
    "search_distance": 2,
    "backend": "pytorch",
    "device": "cpu",
    "use_2d": False,
}


def _prod_params(volume: np.ndarray) -> dict:
    """PRODUCTION-shaped denoise params for ``volume``.

    ``norm_range`` is not optional decoration: ``resolve_denoise_h`` always
    records the WHOLE-volume ``(min, max)`` and ``assemble_fit_config`` forwards
    it, because a fixed ``h`` is not scale-invariant. Omitting it lets each probe
    block normalize against its OWN extremes, which changes the smoothing every
    block receives and can flip the SIGN of the measured shift — so no test here
    may omit it, or it measures a configuration production never runs.
    """
    return {
        **_BASE_DENOISE_PARAMS,
        "norm_range": (float(volume.min()), float(volume.max())),
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

    return denoise_volume_array(volume, h=DENOISE_H, **_prod_params(volume))


def _denoise_like_tiles(block: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Denoise one probe BLOCK the way the probe does — with the WHOLE volume's
    ``norm_range``, not the block's own extremes (see :func:`_prod_params`)."""
    from luxar.gsplats.preprocessing.denoise_pipeline import denoise_volume_array

    return denoise_volume_array(block, h=DENOISE_H, **_prod_params(volume))


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


def test_raw_basis_fallback_honors_sample_budget() -> None:
    data = np.arange(64**3, dtype=np.float32).reshape(64, 64, 64)
    volume = _CountingArray(data)
    budget = 1_000

    level = resolve_volume_floor_denoised(
        volume,
        "p10",
        denoise_h=None,
        denoise_params=_prod_params(data),
        sample_budget=budget,
    )

    assert level is not None
    voxels_read = sum(
        np.empty(volume.shape, dtype=np.uint8)[region].size for region in volume.regions
    )
    assert voxels_read <= budget


class TestDenoisedBasisResolution:
    def test_level_is_the_denoised_estimate_not_the_raw_one(self) -> None:
        """``auto`` resolves to the DENOISED whole-volume mode, not the raw one."""
        volume = _skewed_background_volume()

        raw_level = resolve_volume_floor(volume, "auto")
        level = resolve_volume_floor_denoised(
            volume,
            "auto",
            denoise_h=DENOISE_H,
            denoise_params=_prod_params(volume),
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
            denoise_params=_prod_params(volume),
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
            denoise_params=_prod_params(volume),
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
            denoise_params=_prod_params(volume),
        )
        assert mode_level is not None and level != pytest.approx(mode_level, rel=1e-6)

    @pytest.mark.parametrize("spec", ["auto", "p10"])
    def test_within_budget_is_exact_and_never_skipped(self, capsys, spec) -> None:
        """Regime 1: a whole-volume probe is applied for EVERY volume-derived spec.

        The regime rule that keeps ``auto`` on the raw basis above the budget
        (see :class:`TestCroppedProbeRegime`) must never fire here: within the
        budget the probe IS the volume, so the corrected level is the
        denoised-whole-volume estimate itself and there is nothing to be careful
        about.
        """
        volume = _skewed_background_volume()
        assert volume.size < DENOISE_PROBE_BUDGET_VOXELS

        level = resolve_volume_floor_denoised(
            volume,
            spec,
            denoise_h=DENOISE_H,
            denoise_params=_prod_params(volume),
        )
        expected = _resolve_floor(_denoise_whole(volume).ravel(), spec)
        assert level is not None and expected is not None
        assert level == expected  # bit-exact, not merely close
        assert "RAW-basis level" not in capsys.readouterr().out

    @pytest.mark.parametrize("spec", ["auto", "p10"])
    def test_a_volume_of_exactly_the_budget_is_still_the_exact_regime(
        self, monkeypatch, spec
    ) -> None:
        """``total <= budget``, not ``<``, matching the sampler exactly.

        ``_sample_blocks_for_denoise_probe`` returns a volume of exactly ``budget``
        voxels as ONE whole block, so the correction measured on it is exact. A
        ``<`` here would send that volume down the cropped-probe path and throw an
        exact ``auto`` correction away.
        """
        volume = _skewed_background_volume()
        monkeypatch.setattr(pp, "DENOISE_PROBE_BUDGET_VOXELS", volume.size)

        level = resolve_volume_floor_denoised(
            volume,
            spec,
            denoise_h=DENOISE_H,
            denoise_params=_prod_params(volume),
        )
        expected = _resolve_floor(_denoise_whole(volume).ravel(), spec)
        assert level is not None and expected is not None
        assert level == expected

    def test_an_unmoved_estimator_returns_the_raw_level_untouched(
        self, monkeypatch, capsys
    ) -> None:
        """A zero shift takes the early return, so the level stays BIT-exact.

        Reconstructing ``probe_denoised + (level_raw - probe_raw)`` when
        ``probe_denoised == probe_raw`` is only the identity up to rounding, and
        an "accepted correction" of ``+0`` is not news either. Pinned by an
        identity stand-in denoiser: the level must come back untouched and the
        verbose correction line must NOT be printed.
        """
        volume = _skewed_background_volume()
        monkeypatch.setattr(
            "luxar.gsplats.preprocessing.denoise_pipeline.denoise_volume_array",
            lambda block, **_kwargs: np.asarray(block, dtype=np.float32),
        )
        raw = resolve_volume_floor(volume, "auto")
        capsys.readouterr()
        level = resolve_volume_floor_denoised(
            volume,
            "auto",
            denoise_h=DENOISE_H,
            denoise_params=_prod_params(volume),
            verbose=True,
        )
        assert raw is not None
        assert level == raw
        assert "DENOISED basis" not in capsys.readouterr().out

    def test_workers_agree_on_the_corrected_level(self) -> None:
        """Two independent resolutions of one volume reach the same level.

        This is what lets a ``--tile k/M`` subprocess and its parent subtract
        the same pedestal without coordinating: both the floor sample and the
        denoise probe are pure functions of ``volume.shape``.
        """
        volume = _skewed_background_volume()
        kwargs = {"denoise_h": DENOISE_H, "denoise_params": _prod_params(volume)}
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
            volume, spec, denoise_params=_prod_params(volume), guard_numeric=True
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
            denoise_params=_prod_params(volume),
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
            denoise_params=_prod_params(volume),
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
                denoise_params=_BASE_DENOISE_PARAMS,
            )
            is None
        )

    @pytest.mark.parametrize(
        ("spec", "bad"),
        [
            ("auto", np.nan),
            ("auto", np.inf),
            ("auto", -np.inf),
            ("p10", np.nan),
            ("p10", -np.inf),
            # ("p10", +inf) is deliberately absent: a low percentile of an array
            # whose HIGH tail is infinite is still an ordinary number, so there is
            # no poisoned level to insure against in that combination.
        ],
    )
    def test_a_non_finite_probe_keeps_the_raw_level(
        self, monkeypatch, capsys, spec, bad
    ) -> None:
        """Non-finite denoiser output must not poison the level.

        Today's kernels emit none, so this is insurance — but the failure modes
        are ugly enough to insure against, and each is silent: a ``nan`` ``p10``
        level makes every tile all-NaN and the fit dies later blaming the user's
        clean data; a ``nan`` under ``auto`` collapses the level to ~0, disabling
        floor suppression; a ``-inf`` raises out of ``np.histogram``. All must
        degrade to the raw-basis level with a note. Note the bare ``>=`` re-guard
        cannot do this on its own: a NaN passes every comparison.
        """
        volume = _skewed_background_volume()

        def _poison(block, **_kwargs):
            out = np.asarray(block, dtype=np.float32).copy()
            # A fifth of the voxels, so a p10 sees it too, not just the mode.
            out.flat[: max(1, out.size // 5)] = bad
            return out

        monkeypatch.setattr(
            "luxar.gsplats.preprocessing.denoise_pipeline.denoise_volume_array",
            _poison,
        )
        raw = resolve_volume_floor(volume, spec)
        capsys.readouterr()
        level = resolve_volume_floor_denoised(
            volume,
            spec,
            denoise_h=DENOISE_H,
            denoise_params=_prod_params(volume),
        )
        assert raw is not None
        assert level == raw
        out = capsys.readouterr().out
        assert "keeping the raw-basis background level" in out

    def test_a_failing_probe_READ_keeps_the_raw_level(
        self, monkeypatch, capsys
    ) -> None:
        """The probe READ is inside the try, not only the denoise call.

        A lazy store that raises on the probe's slice (a dropped connection, a
        corrupt chunk) must degrade like every other probe failure — the floor
        sample it already read succeeded, so there IS a level to fall back to.
        """
        volume = _skewed_background_volume()
        raw = resolve_volume_floor(volume, "auto")

        def _boom(*_args, **_kwargs):
            raise OSError("chunk unreadable")

        monkeypatch.setattr(pp, "_sample_blocks_for_denoise_probe", _boom)
        level = resolve_volume_floor_denoised(
            volume,
            "auto",
            denoise_h=DENOISE_H,
            denoise_params=_prod_params(volume),
        )
        assert raw is not None and level == raw
        out = capsys.readouterr().out
        assert "denoise floor probe failed" in out
        assert "OSError" in out

    def test_level_above_the_raw_sampled_max_is_refused(
        self, monkeypatch, capsys
    ) -> None:
        """The 'would erase all signal' guard, re-applied on the RAW sample's max.

        A stand-in "denoiser" that BRIGHTENS its input 4x makes the probe propose
        a huge upward shift, so the corrected level lands above the volume's
        sampled maximum and must be refused exactly as :func:`resolve_volume_floor`
        refuses an over-high raw level. The guard is deliberately NOT on the
        probe's own denoised max (see
        ``TestCroppedProbeRegime.test_a_dim_probe_does_not_veto_a_valid_level``),
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
                denoise_params=_prod_params(volume),
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
#
# What was MEASURED there decides the rule these tests pin (see
# `resolve_volume_floor_denoised`'s Notes for the full table): a `pNN` shift
# transfers and improves with probe size, a histogram-MODE shift does neither, so
# above the budget only `pNN` is corrected.
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


# A volume whose PROBE REGIONS carry a dim, noisy floor while the pedestal and
# all the signal live outside them. `(32, 64, 64)` against a 6k budget puts the
# probe at ~2% of the volume, so the whole-volume percentile still lands on the
# pedestal rather than on the dim regions.
_DIM_PROBE_SHAPE = (32, 64, 64)


def _dim_probe_volume() -> np.ndarray:
    """Bright pedestal + blob, DIM and noisy exactly where the probe reads.

    The masked/zero-padded case (light-sheet fusion, a cropped FOV), with two
    deliberate choices. The dim regions are taken from the SAMPLER itself rather
    than recomputed here, so the test cannot drift from where the probe actually
    looks; and they carry low-level NOISE rather than a constant, because a
    constant probe makes the raw and denoised estimates identical, which exits at
    the "nothing to correct" early return and never reaches the guard the
    scenario is about.
    """
    rng = np.random.default_rng(31178)
    zz, yy, xx = np.meshgrid(*[np.arange(s) for s in _DIM_PROBE_SHAPE], indexing="ij")
    vol = 100.0 + rng.gamma(2.0, 3.0, size=_DIM_PROBE_SHAPE)
    vol = vol + 200.0 * np.exp(
        -(((zz - 8) ** 2 + (yy - 16) ** 2 + (xx - 16) ** 2) / (2 * 5.0**2))
    )
    vol = vol.astype(np.float32)

    shim = _CountingArray(vol)
    assert _sample_blocks_for_denoise_probe(shim, _SMALL_PROBE_BUDGET) is not None
    for region in shim.regions:
        vol[region] = (5.0 + rng.normal(0.0, 0.6, size=vol[region].shape)).astype(
            np.float32
        )
    return vol


class TestCroppedProbeRegime:
    """Above the budget the probe is a CROP: ``pNN`` is corrected, ``auto`` is not."""

    @staticmethod
    def _crop_the_probe(monkeypatch) -> None:
        monkeypatch.setattr(pp, "DENOISE_PROBE_BUDGET_VOXELS", _SMALL_PROBE_BUDGET)

    def test_auto_above_the_budget_keeps_the_raw_level_without_probing(
        self, monkeypatch, capsys
    ) -> None:
        """The measured rule: no crop-measured mode shift, and no wasted NLM pass.

        A histogram-mode shift measured on a bounded crop carries ~1 unit of noise
        and does not converge as the probe grows (0.91 / 0.75 / 0.72 / 0.90 mean
        error at 2.3 / 4.7 / 18.8 / 37.5% of the volume), so applying it is a coin
        flip. The level therefore stays on the raw basis — and because the regime
        is decided from ``volume.shape`` alone, nothing is read or denoised for it.
        """
        self._crop_the_probe(monkeypatch)
        volume = _uniform_background_volume()
        assert volume.size > _SMALL_PROBE_BUDGET

        def _boom(*_args, **_kwargs):
            raise AssertionError("`auto` above the budget must not probe at all")

        monkeypatch.setattr(
            "luxar.gsplats.preprocessing.denoise_pipeline.denoise_volume_array",
            _boom,
        )
        monkeypatch.setattr(pp, "_sample_blocks_for_denoise_probe", _boom)

        raw = resolve_volume_floor(volume, "auto")
        capsys.readouterr()
        level = resolve_volume_floor_denoised(
            volume,
            "auto",
            denoise_h=DENOISE_H,
            denoise_params=_prod_params(volume),
        )
        assert raw is not None and level == raw

        # And it says so, honestly, pointing at what does work.
        out = capsys.readouterr().out
        assert "RAW-basis level" in out
        assert f"{raw:.6g}" in out
        assert "--floor pNN" in out

    @pytest.mark.parametrize(
        "make_volume", [_uniform_background_volume, _vignetted_volume]
    )
    def test_percentile_above_the_budget_lands_closer_than_the_raw_level(
        self, monkeypatch, make_volume
    ) -> None:
        """Regime 2 for ``pNN``: the crop-measured percentile shift transfers.

        The reference is what ``--tiling none`` computes: estimate on the whole
        DENOISED volume. Measured over 36 volumes the corrected ``p10`` level is
        closer to it on 28 of them (mean error 4.4 -> 1.4 units), and it improves
        monotonically with probe size — so it is applied. Both a spatially uniform
        and a vignetted background are checked, because a vignette is exactly the
        case the old basis-agreement gate refused (see the next test).
        """
        self._crop_the_probe(monkeypatch)
        volume = make_volume()
        assert volume.size > _SMALL_PROBE_BUDGET

        reference = _resolve_floor(_denoise_whole(volume).ravel(), "p10")
        raw = resolve_volume_floor(volume, "p10")
        level = resolve_volume_floor_denoised(
            volume,
            "p10",
            denoise_h=DENOISE_H,
            denoise_params=_prod_params(volume),
        )
        assert reference is not None and raw is not None and level is not None
        # Approximate, not exact (a crop is not the volume) — but closer.
        assert level != raw
        assert abs(level - reference) < abs(raw - reference)

    def test_a_probe_that_disagrees_on_the_raw_level_is_still_applied(
        self, monkeypatch, capsys
    ) -> None:
        """Regression: agreement on the RAW level is NOT the test of a good shift.

        The first cut of #1178 declined a shift whenever the probe's own raw level
        differed from the whole volume's by more than the shift proposed — which is
        backwards twice over. It grows more permissive as the proposed shift grows,
        and on a vignetted background it refuses a shift that is a large, correct
        improvement: here the basis gap is several times the shift, yet applying
        the shift takes the error from ~1.2 units to ~0.05. So it is applied, and
        no "not transferable" verdict is printed.
        """
        self._crop_the_probe(monkeypatch)
        volume = _vignetted_volume()

        reference = _resolve_floor(_denoise_whole(volume).ravel(), "p10")
        raw = resolve_volume_floor(volume, "p10")
        level = resolve_volume_floor_denoised(
            volume,
            "p10",
            denoise_h=DENOISE_H,
            denoise_params=_prod_params(volume),
        )
        assert reference is not None and raw is not None and level is not None

        # The premise: the old gate really would have fired on this volume.
        blocks = _sample_blocks_for_denoise_probe(volume, _SMALL_PROBE_BUDGET)
        assert blocks is not None and len(blocks) > 1
        probe_raw = _resolve_floor(np.concatenate([b.ravel() for b in blocks]), "p10")
        assert probe_raw is not None
        basis_gap = abs(raw - probe_raw)
        delta = level - raw
        assert basis_gap > abs(delta)

        # ... and it would have thrown away a large improvement.
        assert abs(level - reference) < abs(raw - reference)
        assert "not transferable" not in capsys.readouterr().out

    def test_two_workers_agree_in_the_cropped_regime(self, monkeypatch) -> None:
        """Independent workers reach the same level from a cropped probe too.

        The determinism claim is about the SAMPLER, so it has to hold where the
        sampler actually chooses: same shape, same ``h``, same params, no
        coordination — one level. Uses ``p10``, the spec that is corrected in this
        regime and therefore the one that exercises the sampler at all.
        """
        self._crop_the_probe(monkeypatch)
        volume = _uniform_background_volume()
        kwargs = {"denoise_h": DENOISE_H, "denoise_params": _prod_params(volume)}

        first = resolve_volume_floor_denoised(volume, "p10", **kwargs)
        second = resolve_volume_floor_denoised(_CountingArray(volume), "p10", **kwargs)
        assert first is not None
        assert second == first

    def test_a_dim_probe_does_not_veto_a_valid_level(self, monkeypatch, capsys) -> None:
        """The re-guard is on the RAW sample's max, and that choice is load-bearing.

        The probe crops the MIDDLE of the volume; on masked / zero-padded data
        (light-sheet fusion, a cropped FOV) that middle can be far dimmer than the
        rest, or empty. Guarding the corrected level against the probe's own
        DENOISED max — a strictly tighter bound, since NLM shrinks the range —
        would refuse the level outright in that case: no floor subtracted for the
        whole run, worse than not correcting at all, and invisible under ``-j N``.

        The probe regions carry low-level NOISE rather than a constant on purpose:
        a constant probe makes the raw and denoised estimates identical, which
        exits at the "nothing to correct" early return and never reaches the guard
        this test is about. The three assertions on ``probe_max < level <
        raw_max`` are what make the scenario discriminating.
        """
        self._crop_the_probe(monkeypatch)
        volume = _dim_probe_volume()
        assert volume.size > _SMALL_PROBE_BUDGET

        raw = resolve_volume_floor(volume, "p10")
        assert raw is not None and raw > 50.0  # the pedestal, not the dim regions
        level = resolve_volume_floor_denoised(
            volume,
            "p10",
            denoise_h=DENOISE_H,
            denoise_params=_prod_params(volume),
        )
        assert level is not None
        # The correction really ran (the early return was not taken).
        assert level != raw

        # The discriminating geometry: the level sits ABOVE everything the probe
        # can see and below the volume's own max, so the two candidate guard bases
        # give opposite answers — and the one that keeps the level is chosen.
        blocks = _sample_blocks_for_denoise_probe(volume, _SMALL_PROBE_BUDGET)
        assert blocks is not None
        probe_denoised_max = max(
            float(np.asarray(_denoise_like_tiles(b, volume), dtype=np.float32).max())
            for b in blocks
        )
        assert probe_denoised_max < level < float(volume.max())
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
            denoise_params=_prod_params(volume),
            verbose=True,
        )
        assert raw is not None and level is not None

        out = capsys.readouterr().out
        assert "DENOISED basis" in out
        assert f"{level:.6g}" in out
        assert f"raw {raw:.6g}" in out  # the level it was corrected FROM

    def test_a_read_free_numeric_level_is_reported_silently(self, capsys) -> None:
        """:func:`resolve_volume_floor` says nothing when it MEASURED nothing.

        The numeric short-circuit returns the user's constant without touching the
        volume, and it used to ``return`` before the verbose line. Extracting the
        shared body dropped that early return, which added a
        ``Resolved whole-volume background floor: 110`` line to every caller that
        forwards ``verbose`` with an absolute ``--floor``. Both directions are
        pinned here: silent when nothing was sampled, spoken when something was.
        """
        volume = _skewed_background_volume()

        assert resolve_volume_floor(volume, "110", verbose=True) == 110.0
        assert resolve_volume_floor(volume, 110.0, verbose=True) == 110.0
        assert capsys.readouterr().out == ""

        # A level measured against the volume IS reported...
        assert resolve_volume_floor(volume, "auto", verbose=True) is not None
        assert "Resolved whole-volume background floor" in capsys.readouterr().out
        # ...and so is a constant the volume had to be sampled to GUARD, which is
        # what this function did before #1178 too.
        assert (
            resolve_volume_floor(volume, "110", guard_numeric=True, verbose=True)
            == 110.0
        )
        assert "Resolved whole-volume background floor" in capsys.readouterr().out
