"""The background-relative basis conversion (#1173, #1177).

A fit reconstructs ``V - image_min``, never ``V``. These pin the one conversion
every consumer of a fitted dataset needs, and the "the store does not say" case,
which is what the affected call sites were silently getting wrong.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from luxar.gsplats.fit_basis import (
    MISSING_BASIS_HINT,
    fit_image_min,
    reference_on_fit_basis,
    rescale_intensity_basis,
    source_intensity_span,
)


class TestFitImageMin:
    def test_prefers_image_min_over_floor(self) -> None:
        """``image_min`` is the level actually subtracted; ``floor`` is only the
        request. They coincide under an active floor and diverge without one, so
        reversing the shift must use ``image_min``."""
        assert fit_image_min({"image_min": 0.25, "floor": 0.10}) == pytest.approx(0.25)

    def test_falls_back_to_floor_for_incomplete_metadata(self) -> None:
        assert fit_image_min({"floor": 0.042}) == pytest.approx(0.042)

    @pytest.mark.parametrize("stats", [None, {}, {"psnr_db": 31.0}])
    def test_absent_is_none_not_zero(self, stats) -> None:
        """``None`` and ``0.0`` are different facts: "did not say" versus
        "removed nothing". Only the first should make a caller warn."""
        assert fit_image_min(stats) is None

    def test_zero_is_reported_as_zero(self) -> None:
        assert fit_image_min({"image_min": 0.0}) == 0.0

    @pytest.mark.parametrize(
        "bad", [float("nan"), float("inf"), -1.0, "not-a-number", None]
    )
    def test_unusable_values_are_treated_as_absent(self, bad) -> None:
        """A NaN/inf/negative level would corrupt every shifted voxel, and no fit
        legitimately applies one — so it must read as "not recorded" rather than
        be propagated."""
        assert fit_image_min({"image_min": bad}) is None

    def test_a_bad_image_min_still_falls_through_to_floor(self) -> None:
        assert fit_image_min(
            {"image_min": float("nan"), "floor": 0.3}
        ) == pytest.approx(0.3)

    def test_a_boolean_is_not_read_as_a_level(self) -> None:
        """``bool`` is an ``int`` subclass, so ``float(True)`` is 1.0.

        On a normalised store, where levels are ~0.04, a level of 1.0 clips the
        ENTIRE reference to zero — a silently meaningless score rather than an
        error. No fit writes a boolean here, so it must read as not-recorded.
        """
        assert fit_image_min({"image_min": True}) is None
        assert fit_image_min({"image_min": False}) is None
        # ...and it must not shadow a usable `floor` sitting behind it.
        assert fit_image_min({"image_min": True, "floor": 0.25}) == pytest.approx(0.25)

    def test_hint_names_the_stat_and_the_consequence(self) -> None:
        """The three call sites share this string; it has to say what is wrong and
        what the reader should distrust."""
        assert "image_min" in MISSING_BASIS_HINT
        assert "RAW" in MISSING_BASIS_HINT


def test_repeated_intensity_scaling_preserves_raw_levels() -> None:
    stats = {"image_min": 112.0, "image_max": 4012.0, "intensity_range": 3900.0}
    rescale_intensity_basis(stats, 0.5)
    rescale_intensity_basis(stats, 0.5)
    assert stats == {
        "image_min": 112.0,
        "image_max": 4012.0,
        "intensity_range": 975.0,
        "intensity_scale": 0.25,
    }
    rescale_intensity_basis(stats, 4.0)
    assert stats == {"image_min": 112.0, "image_max": 4012.0, "intensity_range": 3900.0}


def test_scaled_span_never_falls_back_to_raw_bounds() -> None:
    stats = {"image_min": 112.0, "image_max": 4012.0, "intensity_scale": 0.5}
    assert source_intensity_span(stats) is None
    stats["intensity_range"] = 1950.0
    assert source_intensity_span(stats) == 1950.0


class TestReferenceOnFitBasis:
    def test_is_the_exact_inverse_of_the_normalisation_shift(self) -> None:
        """`_normalize_data` computes ``clip((V - image_min) / range, 0, 1)``;
        undoing the scale must leave exactly this."""
        from luxar.gsplats.fitting.preprocessing import _normalize_data

        rng = np.random.default_rng(0)
        volume = (rng.gamma(2.0, 300.0, (6, 7, 8)) + 200.0).astype(np.float32)
        volume.flat[0] = 0.0
        level = 675.0
        vmax = float(volume.max())

        normalised, image_min, _image_max, intensity_range, _floor = _normalize_data(
            (volume / vmax).astype(np.float32),
            norm_percentile=0.0,
            verbose=False,
            floor=level / vmax,
        )
        shifted = reference_on_fit_basis(volume, level) / (vmax * intensity_range)
        np.testing.assert_allclose(shifted, np.asarray(normalised), atol=1e-6)
        assert image_min == pytest.approx(level / vmax)

    def test_clips_sub_floor_voxels_to_zero(self) -> None:
        """Sub-floor voxels were clipped going in, so leaving them negative here
        would score the fit for failing to reproduce values it never saw."""
        v = np.array([0.0, 100.0, 675.0, 676.0, 2000.0], dtype=np.float32)
        out = reference_on_fit_basis(v, 675.0)
        assert out.min() == 0.0
        np.testing.assert_allclose(out, [0.0, 0.0, 0.0, 1.0, 1325.0], atol=1e-5)

    @pytest.mark.parametrize("level", [None, 0.0])
    def test_no_level_passes_through_unchanged(self, level) -> None:
        """So a caller can pass whatever it resolved without branching."""
        v = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        np.testing.assert_array_equal(reference_on_fit_basis(v, level), v)

    @pytest.mark.parametrize(
        ("dtype", "expected"),
        [
            (np.float32, np.float32),
            (np.float64, np.float64),
            (np.float16, np.float16),
            (np.uint16, np.float32),
            (np.int16, np.float32),
            (np.int32, np.float32),
        ],
    )
    def test_floating_dtype_is_kept_and_integers_become_float32(
        self, dtype, expected
    ) -> None:
        """Not cosmetic. An integer array minus a python float promotes to
        float64 under NEP 50 — four times the memory of a ``uint16`` reference,
        and a float64 result makes ``compute_quality_metrics`` raise "expected
        scalar type Double but found Float" against a float32 render.
        """
        v = np.full((3, 3), 1000, dtype=dtype)
        assert reference_on_fit_basis(v, 675.0).dtype == expected

    def test_the_no_shift_path_also_normalises_integer_dtype(self) -> None:
        """Otherwise a caller's dtype would depend on whether a floor happened to
        be active, which is a surprise waiting for the first integer reference."""
        v = np.full((3, 3), 1000, dtype=np.uint16)
        assert reference_on_fit_basis(v, None).dtype == np.float32
        assert reference_on_fit_basis(v, 0.0).dtype == np.float32

    def test_does_not_mutate_the_caller_s_array(self) -> None:
        v = np.array([700.0, 800.0], dtype=np.float32)
        before = v.copy()
        reference_on_fit_basis(v, 675.0)
        np.testing.assert_array_equal(v, before)

    def test_a_raw_reference_penalises_a_floored_fit(self) -> None:
        """The defect these fixes exist for, stated as a test.

        A perfect reconstruction of the floored signal scores worse against a raw
        reference than against the basis it actually reconstructs — and the error
        is exactly the pedestal, so it grows with the floor. That inversion is
        what made a better tribolium fit report 21 dB worse than a hazier one.
        """
        level = 675.0
        volume = np.full((4, 4, 4), 1675.0, dtype=np.float32)
        perfect_render = reference_on_fit_basis(volume, level)  # == 1000 everywhere

        mse_raw = float(np.mean((perfect_render - volume) ** 2))
        mse_basis = float(np.mean((perfect_render - perfect_render) ** 2))

        assert mse_basis == 0.0
        assert mse_raw == pytest.approx(level**2)


class TestIntensityEditsKeepTheBasisHonest:
    """The Python ``GSplatData`` intensity edits rebase the basis like the CLI."""

    BASIS: ClassVar[dict[str, float]] = {
        "image_min": 100.0,
        "image_max": 4100.0,
        "intensity_range": 4000.0,
    }

    def _data(self):
        from luxar.gsplats.gsplat_data import GSplatData

        rng = np.random.default_rng(1)
        n = 40
        return GSplatData(
            centers=rng.uniform(0, 10, (n, 3)).astype(np.float32),
            amplitudes=rng.uniform(10, 1000, n).astype(np.float32),
            cholesky_factors=np.tile(np.array([1, 0, 1, 0, 0, 1], np.float32), (n, 1)),
            stats=dict(self.BASIS),
        )

    def test_scale_scales_the_range_and_records_the_scale(self) -> None:
        data = self._data()
        out = data.scale_intensity(0.5)
        assert out.stats["intensity_range"] == 2000.0
        assert out.stats["intensity_scale"] == 0.5
        assert out.stats["image_min"] == 100.0
        assert data.stats == self.BASIS  # the source is untouched

    def test_scale_by_one_is_a_no_op(self) -> None:
        assert self._data().scale_intensity(1.0).stats == self.BASIS

    def test_affine_without_offset_is_a_scale(self) -> None:
        assert self._data().affine_intensity(2.0).stats["intensity_range"] == 8000.0

    @pytest.mark.parametrize(
        "edit",
        [
            lambda d: d.normalize_intensity(1.0),
            lambda d: d.affine_intensity(1.0, 5.0),
            lambda d: d.clamp_intensity(max=500.0),
            lambda d: d.reweight_amplitude(np.linspace(0.5, 1.0, d.n_splats)),
            lambda d: d.soft_scale_filter(highpass=0.5),
        ],
    )
    def test_non_global_edits_drop_the_amplitude_basis(self, edit) -> None:
        out = edit(self._data())
        assert "intensity_range" not in out.stats
        assert "image_max" not in out.stats
        assert "intensity_scale" not in out.stats
        assert out.stats["image_min"] == 100.0
