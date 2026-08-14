"""Unit tests for :mod:`luxar.gsplats.lod.volume_regions`.

The module's job is to hand one volume re-fit exactly the sub-volume its splats
are responsible for, so the tests are about *exactness*: the slice must be the
right slice with its axes in the right order, the projection must be the true
marginal, the lift back must be bit-faithful on the dims it did not touch, and
the aggregated stats must describe what was actually stored.
"""

from __future__ import annotations

import numpy as np
import pytest

from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.lod.volume_regions import (
    finalize_volume_refit_stats,
    merge_volume_refit_stats,
    project_to_dims,
    restore_dims,
    select_sub_volume,
)
from luxar.gsplats.utils.trils import pack_tril, unpack_tril


def _splats(n: int, ndim: int, *, barrier_value: float | None = None, seed: int = 0):
    """``n`` splats in ``ndim`` dims; the last dim is a zero-extent barrier."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(2.0, 8.0, (n, ndim)).astype(np.float32)
    if barrier_value is not None:
        centers[:, -1] = barrier_value
    factors = np.zeros((n, ndim, ndim), np.float32)
    for d in range(ndim - 1):
        factors[:, d, d] = rng.uniform(0.7, 1.5, n)
    factors[:, ndim - 1, ndim - 1] = 1e-4
    factors[:, 1, 0] = rng.uniform(-0.3, 0.3, n)  # a real spatial cross-term
    return GSplatData(
        centers=centers,
        amplitudes=rng.uniform(0.3, 1.0, n).astype(np.float32),
        cholesky_factors=pack_tril(factors),
    )


class TestSelectSubVolume:
    def test_barrier_slice_with_non_identity_axis_map(self) -> None:
        """The real Luxar layout: splats are ``(z, y, x, t)`` while the source
        array is ``(t, z, y, x)``. The identity map would slice the wrong axis,
        so the mapping has to be applied AND the survivors transposed back into
        center-dim order."""
        vol = np.random.default_rng(0).random((3, 4, 5, 6)).astype(np.float32)
        for t in range(3):
            sub = select_sub_volume(
                vol,
                ndim=4,
                barrier_dims=(3,),
                barrier_coords=(t,),
                volume_axes=(1, 2, 3, 0),
            )
            np.testing.assert_array_equal(sub.array, vol[t])
            assert sub.dims == (0, 1, 2)
            np.testing.assert_array_equal(sub.origin, np.zeros(3))

    def test_box_composes_with_the_barrier_slice(self) -> None:
        """A tiled timelapse needs both at once: a box within a slice."""
        vol = np.random.default_rng(1).random((3, 8, 8, 8)).astype(np.float32)
        sub = select_sub_volume(
            vol,
            ndim=4,
            barrier_dims=(3,),
            barrier_coords=(2,),
            box=[(2.0, 5.0), (0.0, 7.0), (3.0, 4.0)],
            volume_axes=(1, 2, 3, 0),
        )
        np.testing.assert_array_equal(sub.array, vol[2][2:6, 0:8, 3:5])
        np.testing.assert_array_equal(sub.origin, np.array([2.0, 0.0, 3.0]))

    def test_infinite_box_faces_clamp_to_the_volume(self) -> None:
        """A BSP cell records CUTS, so its outer faces come back infinite. The
        volume's own extent is the missing bound — without this the crop maths
        raises ``OverflowError`` on the first outermost tile."""
        vol = np.zeros((6, 7, 8), np.float32)
        sub = select_sub_volume(
            vol,
            ndim=3,
            box=[(-np.inf, np.inf), (2.0, np.inf), (-np.inf, 3.0)],
        )
        assert sub.array.shape == (6, 5, 4)
        np.testing.assert_array_equal(sub.origin, np.array([0.0, 2.0, 0.0]))

    def test_degenerate_box_falls_back_to_the_full_extent(self) -> None:
        """An empty crop would give the fit nothing to see at all."""
        vol = np.zeros((5, 5, 5), np.float32)
        sub = select_sub_volume(vol, ndim=3, box=[(4.0, 1.0), (0, 4), (0, 4)])
        assert sub.array.shape[0] == 5

    def test_quantized_barrier_coordinate_rounds_to_its_label(self) -> None:
        """A barrier coordinate is a discrete label, but quantized storage leaves
        it a fraction of a step off the integer (uint16 over 0-252 has a step of
        ~0.0038), so it must round rather than truncate."""
        vol = np.random.default_rng(2).random((4, 3, 3, 3)).astype(np.float32)
        for coord in (1.9996, 2.0004):
            sub = select_sub_volume(
                vol,
                ndim=4,
                barrier_dims=(3,),
                barrier_coords=(coord,),
                volume_axes=(1, 2, 3, 0),
            )
            np.testing.assert_array_equal(sub.array, vol[2])

    def test_out_of_range_barrier_coordinate_raises(self) -> None:
        vol = np.zeros((3, 4, 4, 4), np.float32)
        with pytest.raises(ValueError, match="disagree about this axis"):
            select_sub_volume(
                vol,
                ndim=4,
                barrier_dims=(3,),
                barrier_coords=(7,),
                volume_axes=(1, 2, 3, 0),
            )

    def test_volume_must_span_every_center_dim(self) -> None:
        """The barrier dims are sliced internally, so they must still be present
        in the target — a pre-reduced 3D array for 4D splats is a caller error,
        not something to guess at."""
        with pytest.raises(ValueError, match="span every center dim"):
            select_sub_volume(np.zeros((4, 4, 4), np.float32), ndim=4)

    def test_lazy_store_is_only_ever_sliced(self) -> None:
        """The reason a 431 GB timelapse is usable at all: the array-like is
        indexed, never coerced whole."""

        class Spy:
            def __init__(self, arr: np.ndarray) -> None:
                self._a = arr
                self.whole = 0

            @property
            def shape(self):  # noqa: ANN201 - mirrors the array protocol
                return self._a.shape

            def __getitem__(self, idx):  # noqa: ANN001, ANN204
                return self._a[idx]

            def __array__(self, *a, **k):  # noqa: ANN002, ANN003, ANN204
                self.whole += 1
                return self._a

        spy = Spy(np.random.default_rng(3).random((3, 4, 4, 4)).astype(np.float32))
        sub = select_sub_volume(
            spy,
            ndim=4,
            barrier_dims=(3,),
            barrier_coords=(1,),
            volume_axes=(1, 2, 3, 0),
        )
        assert sub.array.shape == (4, 4, 4)
        assert spy.whole == 0, "the whole store was materialised"


class TestProjectRestore:
    def test_projection_is_the_true_marginal(self) -> None:
        data = _splats(16, 4, barrier_value=3.0)
        proj = project_to_dims(data, (0, 1, 2))
        full = unpack_tril(np.asarray(data.cholesky_factors), 4)
        sigma = full @ full.transpose(0, 2, 1)
        got = unpack_tril(np.asarray(proj.cholesky_factors), 3)
        np.testing.assert_allclose(
            got @ got.transpose(0, 2, 1), sigma[:, :3, :3], rtol=1e-5, atol=1e-6
        )

    def test_restore_keeps_the_barrier_block_and_stays_positive_definite(self) -> None:
        """The lift recombines in FACTOR space precisely so this holds: with the
        free dims first, ``L = [[A, 0], [B, C]]``, so substituting the re-fit's
        ``A`` leaves a lower-triangular factor with a positive diagonal."""
        data = _splats(16, 4, barrier_value=2.0)
        proj = project_to_dims(data, (0, 1, 2))
        # Stand in for a re-fit that genuinely moved and rescaled the free block.
        moved = GSplatData(
            centers=(np.asarray(proj.centers) + 0.4).astype(np.float32),
            amplitudes=np.asarray(proj.amplitudes),
            cholesky_factors=(np.asarray(proj.cholesky_factors) * 1.25).astype(
                np.float32
            ),
        )
        back = restore_dims(moved, data, (0, 1, 2))

        np.testing.assert_array_equal(
            np.asarray(back.centers)[:, 3], np.asarray(data.centers)[:, 3]
        )
        before = unpack_tril(np.asarray(data.cholesky_factors), 4)
        after = unpack_tril(np.asarray(back.cholesky_factors), 4)
        sig_b = before @ before.transpose(0, 2, 1)
        sig_a = after @ after.transpose(0, 2, 1)
        np.testing.assert_allclose(sig_a[:, 3, 3], sig_b[:, 3, 3], rtol=1e-4, atol=1e-9)
        assert np.linalg.eigvalsh(sig_a).min() > 0.0

    def test_restore_preserves_colors_and_truncation_radius(self) -> None:
        """The re-fit is identity-preserving row-for-row, so per-splat payload the
        fit never sees has to survive the round trip."""
        rng = np.random.default_rng(4)
        base = _splats(12, 4, barrier_value=1.0)
        colors = rng.integers(0, 255, (12, 3)).astype(np.uint8)
        data = GSplatData(
            centers=base.centers,
            amplitudes=base.amplitudes,
            cholesky_factors=base.cholesky_factors,
            colors=colors,
            truncation_radius=2.75,
        )
        proj = project_to_dims(data, (0, 1, 2))
        assert proj.truncation_radius == 2.75
        back = restore_dims(proj, data, (0, 1, 2))
        np.testing.assert_array_equal(np.asarray(back.colors), colors)
        assert back.truncation_radius == 2.75

    def test_restore_refuses_a_row_misaligned_refit(self) -> None:
        data = _splats(12, 4, barrier_value=0.0)
        proj = project_to_dims(data, (0, 1, 2))
        short = GSplatData(
            centers=np.asarray(proj.centers)[:5],
            amplitudes=np.asarray(proj.amplitudes)[:5],
            cholesky_factors=np.asarray(proj.cholesky_factors)[:5],
        )
        with pytest.raises(ValueError, match="row-aligned"):
            restore_dims(short, data, (0, 1, 2))

    def test_all_dims_is_a_no_op_on_both_sides(self) -> None:
        data = _splats(10, 3)
        assert project_to_dims(data, (0, 1, 2)) is data
        assert restore_dims(data, data, (0, 1, 2)) is data


class TestStatsAggregation:
    @staticmethod
    def _one(group: dict, weight: int = 10) -> dict:
        sink: dict = {}
        merge_volume_refit_stats(sink, group, weight=weight)
        return finalize_volume_refit_stats(sink)

    def test_mse_stored_follows_the_verdict_not_the_minimum(self) -> None:
        """Regression. ``mse_stored`` once took ``min(seed, refit)``, which is only
        the same thing while MSE is the sole arbiter. A re-fit rejected for
        leaving its tile can hold the LOWER MSE and still not be what was stored,
        and the minimum then credits the level with an error it never achieved."""
        escaped = self._one(
            {
                "mse_seed": 1.0,
                "mse_refit": 0.1,  # looked better...
                "improved": False,
                "seed_won": True,  # ...but was rejected
                "tile_escape": True,
                "n_seed": 10,
                "n_refit": 10,
                "iters": 5,
                "wall_s": 0.1,
            }
        )
        assert escaped["mse_stored"] == pytest.approx(1.0), (
            "the seed was stored, so mse_stored must be the seed's error"
        )
        assert escaped["mse_stored"] <= escaped["mse_seed"] + 1e-12

    def test_mse_stored_takes_the_refit_when_the_refit_was_kept(self) -> None:
        won = self._one(
            {
                "mse_seed": 1.0,
                "mse_refit": 0.1,
                "improved": True,
                "seed_won": False,
                "tile_escape": False,
                "n_seed": 10,
                "n_refit": 10,
                "iters": 5,
                "wall_s": 0.1,
            }
        )
        assert won["mse_stored"] == pytest.approx(0.1)

    def test_means_are_weighted_by_piece_size(self) -> None:
        """Each piece's MSE is a mean over its own voxels, so pieces of different
        size must not get equal say."""
        sink: dict = {}
        merge_volume_refit_stats(
            sink,
            {"mse_seed": 1.0, "mse_refit": 0.5, "improved": True, "seed_won": False},
            weight=1,
        )
        merge_volume_refit_stats(
            sink,
            {"mse_seed": 3.0, "mse_refit": 2.0, "improved": False, "seed_won": True},
            weight=3,
        )
        out = finalize_volume_refit_stats(sink)
        assert out["mse_seed"] == pytest.approx((1.0 * 1 + 3.0 * 3) / 4)
        # refit kept for the first piece, seed for the second.
        assert out["mse_stored"] == pytest.approx((0.5 * 1 + 3.0 * 3) / 4)
        assert out["improved_frac"] == pytest.approx(0.5)
        assert out["n_pieces"] == 2

    def test_output_is_flat_and_json_safe(self) -> None:
        """The zarr writer and the provenance tests both require it."""
        import json

        out = self._one(
            {
                "mse_seed": 1.0,
                "mse_refit": 0.5,
                "improved": True,
                "seed_won": False,
                "n_seed": 4,
                "n_refit": 4,
                "iters": 2,
                "wall_s": 0.5,
            }
        )
        assert "_mse_w" not in out, "internal bookkeeping leaked into the artifact"
        assert all(not isinstance(v, (dict, list)) for v in out.values())
        json.dumps(out)

    def test_frame_mismatch_contributes_no_mse(self) -> None:
        """A skipped re-fit reports no MSEs at all, and must not be averaged in
        as a zero."""
        out = self._one(
            {
                "frame_mismatch": True,
                "improved": False,
                "seed_won": True,
                "n_seed": 3,
                "n_refit": 3,
                "iters": 0,
                "wall_s": 0.0,
            }
        )
        assert "mse_seed" not in out
        assert out["frame_mismatch"] is True
        assert out["frame_mismatch_frac"] == pytest.approx(1.0)


class TestArgumentValidation:
    """An axis map that is quietly wrong is worse than one that raises: it sends
    the re-fit at the wrong axis, every piece trips the frame guard, and the
    command reports success while refining nothing."""

    @pytest.mark.parametrize(
        "axes", [(1, 1, 2, 3), (0, 1, 2, 9), (0, 1, 2), (0, 1, 2, 3, 4)]
    )
    def test_volume_axes_must_be_a_permutation(self, axes) -> None:
        """Length alone is not enough — a duplicate or out-of-range entry
        otherwise surfaces as numpy's "repeated axis in transpose" or a bare
        IndexError, neither of which names the argument at fault."""
        with pytest.raises(ValueError, match="must be a permutation"):
            select_sub_volume(
                np.zeros((3, 4, 5, 6), np.float32),
                ndim=4,
                barrier_dims=(3,),
                barrier_coords=(0,),
                volume_axes=axes,
            )

    def test_box_arity_must_match_the_retained_dims(self) -> None:
        with pytest.raises(ValueError, match="one \\(low, high\\) per retained dim"):
            select_sub_volume(
                np.zeros((4, 4, 4), np.float32), ndim=3, box=[(0, 1), (0, 1)]
            )

    def test_orphan_volume_axes_is_rejected(self) -> None:
        """Mirrors the neighbouring ``volume``-without-``refine`` check and the
        CLI's orphan-selector rejection: an axis map with nothing to describe is
        a mistake, not a no-op."""
        from luxar.gsplats.lod.substitutive import make_substitutive_lod

        data = _splats(8, 3)
        with pytest.raises(ValueError, match="describes the layout of `volume`"):
            make_substitutive_lod(
                data,
                levels=1,
                compression_factor=2,
                volume_axes=(2, 1, 0),
                device="cpu",
            )
