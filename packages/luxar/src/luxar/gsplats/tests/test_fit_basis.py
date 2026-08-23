"""The background-relative basis conversion (#1173, #1177).

A fit reconstructs ``V - image_min``, never ``V``. These pin the one conversion
every consumer of a fitted dataset needs, and the "the store does not say" case,
which is what the affected call sites were silently getting wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from luxar.gsplats.fit_basis import (
    MISSING_BASIS_HINT,
    fit_image_min,
    reference_on_fit_basis,
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

    def test_hint_names_the_stat_and_the_consequence(self) -> None:
        """The three call sites share this string; it has to say what is wrong and
        what the reader should distrust."""
        assert "image_min" in MISSING_BASIS_HINT
        assert "RAW" in MISSING_BASIS_HINT


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
