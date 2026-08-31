"""Tests for INRIA PLY export (the inverse of classical import).

The primary fidelity gate is the closed loop: ``GSplatData → export →
read_inria_ply → classical_to_gsplat_data`` must reproduce centers and
covariances, and the reverse loop ``import → export (--opacity amplitude) →
import`` must be an identity in the source frame.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.interop._color import linear_to_srgb
from luxar.gsplats.interop.classical_splats import (
    classical_to_gsplat_data,
    import_gsplats,
    quat_to_rotmat,
    read_inria_ply,
)
from luxar.gsplats.interop.inria_export import (
    export_inria_ply,
    gsplat_data_to_inria_ply,
)
from luxar.gsplats.interop.tests._synthetic import make_ground_truth, write_inria_ply
from luxar.gsplats.utils.trils import pack_tril, unpack_tril


def _cov_relF_p95(ref: np.ndarray, got: np.ndarray, d: int = 3) -> float:
    Lr = unpack_tril(ref.astype(np.float64), d)
    Lg = unpack_tril(got.astype(np.float64), d)
    Sr = Lr @ Lr.transpose(0, 2, 1)
    Sg = Lg @ Lg.transpose(0, 2, 1)
    rel = np.linalg.norm(Sg - Sr, axis=(1, 2)) / (
        np.linalg.norm(Sr, axis=(1, 2)) + 1e-30
    )
    return float(np.percentile(rel, 95))


def _synthetic_gsplat_data(n: int = 24, seed: int = 5) -> GSplatData:
    """A native-style GSplatData (anisotropic rotated covariances, colors)."""
    gt = make_ground_truth(n=n, seed=seed)
    R = quat_to_rotmat(gt.quaternions)
    s2 = gt.scales.astype(np.float64) ** 2
    sigma = (R * s2[:, None, :]) @ R.transpose(0, 2, 1)
    return GSplatData(
        centers=gt.positions,
        amplitudes=gt.opacities,  # in (0, 1) so 'amplitude' policy is lossless
        cholesky_factors=pack_tril(np.linalg.cholesky(sigma)).astype(np.float32),
        colors=gt.colors,
    )


def _parse_ply(payload: bytes, tmp: Path) -> "object":
    path = tmp / "exported.ply"
    path.write_bytes(payload)
    return read_inria_ply(path)


class TestClosedLoop:
    def test_export_then_import_reproduces_geometry(self) -> None:
        data = _synthetic_gsplat_data()
        payload = gsplat_data_to_inria_ply(data, opacity_policy="amplitude")
        with tempfile.TemporaryDirectory() as tmp:
            cs = _parse_ply(payload, Path(tmp))
            back = classical_to_gsplat_data(cs, rotate_x180=False)

        assert back.n_splats == data.n_splats
        assert np.allclose(back.centers, data.centers, atol=1e-4)
        assert _cov_relF_p95(data.cholesky_factors, back.cholesky_factors) < 1e-4
        # Re-import lands the (0, 1) amplitudes in the color alpha channel
        # (amplitudes = 1): the round-trip invariant is EFFECTIVE mass A·a.
        from luxar.gsplats.utils.alpha import effective_amplitudes

        assert np.allclose(back.amplitudes, 1.0)
        assert np.allclose(effective_amplitudes(back), data.amplitudes, atol=1e-4)
        assert back.colors is not None
        assert np.allclose(back.colors[:, :3], data.colors, atol=1e-3)

    def test_import_export_import_identity(self) -> None:
        gt = make_ground_truth(n=16, seed=9)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "source.ply"
            write_inria_ply(src, gt)
            first = import_gsplats(src)  # default orientation applied
            # undo_orientation (default) must land back in the source frame.
            payload = gsplat_data_to_inria_ply(first, opacity_policy="amplitude")
            cs = _parse_ply(payload, Path(tmp))

        assert np.allclose(cs.positions, gt.positions, atol=1e-4)
        # eigh returns eigenvalues ascending: the exported (scale, quat) pair
        # is equivalent but not component-wise identical — compare sorted
        # scales and the reconstructed covariance instead.
        assert np.allclose(
            np.sort(cs.scales, axis=1), np.sort(gt.scales, axis=1), rtol=1e-3
        )
        back = classical_to_gsplat_data(cs, rotate_x180=False)
        R = quat_to_rotmat(gt.quaternions)
        s2 = gt.scales.astype(np.float64) ** 2
        expected = pack_tril(
            np.linalg.cholesky((R * s2[:, None, :]) @ R.transpose(0, 2, 1))
        )
        assert _cov_relF_p95(expected, back.cholesky_factors) < 1e-3
        assert np.allclose(cs.opacities, gt.opacities, atol=1e-4)
        assert np.allclose(cs.colors, gt.colors, atol=1e-3)

    def test_keep_orientation_stays_in_luxar_frame(self) -> None:
        gt = make_ground_truth(n=16, seed=9)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "source.ply"
            write_inria_ply(src, gt)
            first = import_gsplats(src)
            payload = gsplat_data_to_inria_ply(
                first, opacity_policy="amplitude", undo_orientation=False
            )
            cs = _parse_ply(payload, Path(tmp))
        assert np.allclose(cs.positions, first.centers, atol=1e-4)


class TestEigendecomposition:
    def test_isotropic_covariance(self) -> None:
        n = 8
        data = GSplatData(
            centers=np.zeros((n, 3), np.float32),
            amplitudes=np.full(n, 0.5, np.float32),
            cholesky_factors=np.tile(
                np.array([0.3, 0, 0.3, 0, 0, 0.3], np.float32), (n, 1)
            ),
        )
        payload = gsplat_data_to_inria_ply(data)
        with tempfile.TemporaryDirectory() as tmp:
            cs = _parse_ply(payload, Path(tmp))
        assert np.allclose(cs.scales, 0.3, rtol=1e-5)

    def test_near_singular_covariance_stays_finite(self) -> None:
        n = 4
        chol = np.tile(np.array([0.5, 0, 1e-12, 0, 0, 0.2], np.float32), (n, 1))
        data = GSplatData(
            centers=np.zeros((n, 3), np.float32),
            amplitudes=np.full(n, 0.5, np.float32),
            cholesky_factors=chol,
        )
        payload = gsplat_data_to_inria_ply(data)
        with tempfile.TemporaryDirectory() as tmp:
            cs = _parse_ply(payload, Path(tmp))
        assert np.isfinite(cs.scales).all()
        assert np.isfinite(cs.quaternions).all()

    def test_quaternions_are_proper_rotations(self) -> None:
        data = _synthetic_gsplat_data(n=50, seed=13)
        payload = gsplat_data_to_inria_ply(data)
        with tempfile.TemporaryDirectory() as tmp:
            cs = _parse_ply(payload, Path(tmp))
        R = quat_to_rotmat(cs.quaternions)
        assert np.allclose(np.linalg.det(R), 1.0, atol=1e-6)


class TestOpacityPolicies:
    def test_normalized_rescales_unbounded_amplitudes(self) -> None:
        data = _synthetic_gsplat_data()
        boosted = data.scale_intensity(1000.0)  # unbounded emission weights
        payload = gsplat_data_to_inria_ply(boosted, opacity_policy="normalized")
        with tempfile.TemporaryDirectory() as tmp:
            cs = _parse_ply(payload, Path(tmp))
        assert cs.opacities.max() <= 1.0
        assert cs.opacities.max() > 0.5  # top amplitudes near full opacity
        # Monotone: ordering of amplitudes preserved.
        order_in = np.argsort(boosted.amplitudes)
        order_out = np.argsort(cs.opacities)
        assert np.array_equal(order_in, order_out)

    def test_constant_policy(self) -> None:
        data = _synthetic_gsplat_data()
        payload = gsplat_data_to_inria_ply(
            data, opacity_policy="constant", constant_opacity=0.25
        )
        with tempfile.TemporaryDirectory() as tmp:
            cs = _parse_ply(payload, Path(tmp))
        assert np.allclose(cs.opacities, 0.25, atol=1e-5)

    def test_unknown_policy_raises(self) -> None:
        with pytest.raises(ValueError, match="opacity policy"):
            gsplat_data_to_inria_ply(
                _synthetic_gsplat_data(),
                opacity_policy="nope",  # type: ignore[arg-type]
            )


class TestColorSources:
    def test_white_fallback_without_colors(self) -> None:
        data = _synthetic_gsplat_data()
        no_colors = GSplatData(
            centers=data.centers,
            amplitudes=data.amplitudes,
            cholesky_factors=data.cholesky_factors,
        )
        payload = gsplat_data_to_inria_ply(no_colors)
        with tempfile.TemporaryDirectory() as tmp:
            cs = _parse_ply(payload, Path(tmp))
        assert np.allclose(cs.colors, 1.0, atol=1e-3)

    def test_colormap_bakes_amplitudes(self) -> None:
        data = _synthetic_gsplat_data()
        no_colors = GSplatData(
            centers=data.centers,
            amplitudes=data.amplitudes,
            cholesky_factors=data.cholesky_factors,
        )
        payload = gsplat_data_to_inria_ply(no_colors, colormap="viridis")
        with tempfile.TemporaryDirectory() as tmp:
            cs = _parse_ply(payload, Path(tmp))
        assert cs.colors.std() > 0.01  # actually varied, not white

    def test_uint8_rgba_colors_not_clipped_to_white(self) -> None:
        # Regression (#728): uint8 colors store values at full scale (0..255);
        # they must be normalized to [0, 1] before export, or the RGB channels
        # clip to white in linear_to_srgb and the alpha channel saturates to
        # fully-opaque in _opacity_logits. Here every splat is (200, 100, 50)
        # RGB with alpha 128 — the exported DC must reproduce
        # linear_to_srgb([200/255, 100/255, 50/255]) and opacity ~= 128/255.
        n = 12
        colors = np.tile(np.array([200, 100, 50, 128], dtype=np.uint8), (n, 1))
        data = GSplatData(
            centers=np.zeros((n, 3), np.float32),
            amplitudes=np.ones(n, np.float32),  # amplitude policy -> opacity = alpha
            cholesky_factors=np.tile(
                np.array([0.3, 0, 0.3, 0, 0, 0.3], np.float32), (n, 1)
            ),
            colors=colors,
        )
        payload = gsplat_data_to_inria_ply(data, opacity_policy="amplitude")
        with tempfile.TemporaryDirectory() as tmp:
            cs = _parse_ply(payload, Path(tmp))

        expected_rgb = linear_to_srgb(
            np.array([200.0 / 255.0, 100.0 / 255.0, 50.0 / 255.0])
        )
        # Tight tolerance: only float32 f_dc/logit quantization (~1e-7) should
        # separate us from the exact value, so a ~1/255 off-by-one normalizer
        # (e.g. dividing by 256) would still be caught.
        assert np.allclose(cs.colors, expected_rgb, atol=1e-4)
        # Not washed to white.
        assert cs.colors.max() < 0.999
        # Opacity reflects alpha 128/255 ~= 0.502, NOT a saturated 1.0.
        assert np.allclose(cs.opacities, 128.0 / 255.0, atol=1e-4)
        assert cs.opacities.max() < 0.6

    def test_colors_source_requires_colors(self) -> None:
        data = _synthetic_gsplat_data()
        no_colors = GSplatData(
            centers=data.centers,
            amplitudes=data.amplitudes,
            cholesky_factors=data.cholesky_factors,
        )
        with pytest.raises(ValueError, match="no colors"):
            gsplat_data_to_inria_ply(no_colors, color_source="colors")


class TestShDegree:
    @pytest.mark.parametrize("degree", [0, 1, 3])
    def test_f_rest_bands_written_as_zeros(self, degree: int) -> None:
        data = _synthetic_gsplat_data()
        payload = gsplat_data_to_inria_ply(data, sh_degree=degree)
        with tempfile.TemporaryDirectory() as tmp:
            cs = _parse_ply(payload, Path(tmp))
        assert cs.sh_degree == degree
        assert cs.n_splats == data.n_splats


class TestNdPolicy:
    def _4d_data(self) -> GSplatData:
        # Stack two timepoints; embed_dimension APPENDS the new (time) column,
        # matching `gsplat merge --as-dimension`.
        t0 = _synthetic_gsplat_data(n=24, seed=5).embed_dimension(values=0.0)
        t1 = _synthetic_gsplat_data(n=24, seed=6).embed_dimension(values=1.0)
        return GSplatData.concatenate([t0, t1])

    def test_4d_without_slice_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly 3D"):
            gsplat_data_to_inria_ply(self._4d_data())

    def test_4d_with_timepoint_selects_slice(self) -> None:
        data = self._4d_data()
        payload = gsplat_data_to_inria_ply(
            data, timepoint=0, opacity_policy="amplitude"
        )
        with tempfile.TemporaryDirectory() as tmp:
            cs = _parse_ply(payload, Path(tmp))
        assert cs.n_splats == 24

    def test_timepoint_and_slice_dim_conflict(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            gsplat_data_to_inria_ply(
                self._4d_data(), timepoint=0, slice_dim=3, slice_index=0
            )

    def test_2d_data_is_embedded(self) -> None:
        n = 8
        data = GSplatData(
            centers=np.random.default_rng(1).uniform(-1, 1, (n, 2)).astype(np.float32),
            amplitudes=np.full(n, 0.5, np.float32),
            cholesky_factors=np.tile(np.array([0.2, 0.0, 0.3], np.float32), (n, 1)),
        )
        payload = gsplat_data_to_inria_ply(data)
        with tempfile.TemporaryDirectory() as tmp:
            cs = _parse_ply(payload, Path(tmp))
        assert cs.n_splats == n
        assert np.allclose(cs.positions[:, 2], 0.0, atol=1e-6)


class TestFileLevel:
    def test_export_inria_ply_round_trip(self) -> None:
        data = _synthetic_gsplat_data()
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "data.gsplats.zarr"
            data.save(store, ordering="none")
            out = Path(tmp) / "exported.ply"
            n = export_inria_ply(store, out, opacity_policy="amplitude")
            assert n == data.n_splats
            cs = read_inria_ply(out)
        assert cs.n_splats == data.n_splats
        assert np.allclose(
            np.sort(cs.positions[:, 0]), np.sort(data.centers[:, 0]), atol=1e-2
        )

    def test_orientation_round_trips_through_disk(self) -> None:
        # Regression: the orientation matrix lives in stats["interop"], which
        # only survives GSplatData.load(include_stats=True) — export must load
        # with stats or undo_orientation silently no-ops.
        gt = make_ground_truth(n=16, seed=21)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "source.ply"
            write_inria_ply(src, gt)
            store = Path(tmp) / "imported.gsplats.zarr"
            import_gsplats(src).save(store, ordering="none")
            out = Path(tmp) / "back.ply"
            export_inria_ply(store, out, opacity_policy="amplitude")
            cs = read_inria_ply(out)
        assert np.allclose(cs.positions, gt.positions, atol=1e-2)

    def test_export_refuses_categorical_channel(self, tmp_path: Path) -> None:
        data = _synthetic_gsplat_data(n=4)
        labeled = GSplatData(
            centers=data.centers,
            amplitudes=data.amplitudes,
            cholesky_factors=data.cholesky_factors,
            colors=data.colors,
            label_ids=np.arange(4, dtype=np.uint8),
            label_vocabulary={i: str(i) for i in range(4)},
        )
        store = tmp_path / "labeled.gsplats.zarr"
        labeled.save(store, ordering="none")

        with pytest.raises(ValueError, match="cannot export.*label_ids"):
            export_inria_ply(store, tmp_path / "out.ply")

    def test_empty_export_raises(self) -> None:
        empty = GSplatData(
            centers=np.zeros((0, 3), np.float32),
            amplitudes=np.zeros(0, np.float32),
            cholesky_factors=np.zeros((0, 6), np.float32),
        )
        with pytest.raises(ValueError, match="empty"):
            gsplat_data_to_inria_ply(empty)
