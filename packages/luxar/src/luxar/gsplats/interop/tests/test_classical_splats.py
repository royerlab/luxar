"""Tests for classical (photogrammetric) Gaussian-splat import.

Fixture files are generated on the fly by the miniature per-dialect writers in
:mod:`._synthetic`, so every reader is tested for round-trip parity against a
shared ground truth without shipping binary fixtures. Fidelity is asserted at
two layers: reader-level parameter parity (tolerances mirror each format's
quantization), and covariance fidelity through the full
``read → convert → save → load`` pipeline (the ``_cov_relF_p95`` harness from
``io/tests/test_save_load.py``).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from luxar.gsplats.interop.classical_splats import (
    CLASSICAL_FORMATS,
    ClassicalSplats,
    classical_to_gsplat_data,
    detect_classical_format,
    import_gsplats,
    quat_to_rotmat,
    rotmat_to_quat,
)
from luxar.gsplats.interop.tests._synthetic import (
    SUFFIXES,
    WRITERS,
    GroundTruth,
    make_ground_truth,
)
from luxar.gsplats.utils.trils import unpack_tril

# Reader-level tolerances per dialect (dominated by each format's quantization).
# SOG: 16-bit log-domain means, 8-bit smallest-three quats, 256-entry codebooks
# for scales/DC (synthetic writer uses a linspace codebook, so error ≤ half a
# codebook step — the reader must tolerate that).
_POSITION_ATOL = {
    "inria": 1e-6, "splat": 1e-6, "spz": 5e-4, "supersplat": 5e-3, "sog": 5e-3
}
_SCALE_RTOL = {
    "inria": 1e-6, "splat": 1e-6, "spz": 0.04, "supersplat": 0.02, "sog": 0.02
}
# SPZ v2 stores (x, y, z) and recomputes w = sqrt(1 - |xyz|²): the 1/127.5
# xyz quantization error is amplified into w when w is small, so SPZ's real
# error profile is looser than the raw 8-bit step.
_QUAT_ATOL = {
    "inria": 1e-6, "splat": 0.01, "spz": 0.04, "supersplat": 0.002, "sog": 0.02
}
_OPACITY_ATOL = {
    "inria": 1e-6, "splat": 1 / 255, "spz": 1 / 255, "supersplat": 1 / 255,
    "sog": 1 / 255,
}
_COLOR_ATOL = {
    "inria": 1e-6, "splat": 1 / 255, "spz": 0.01, "supersplat": 0.01, "sog": 0.01
}


def _write_fixture(fmt: str, gt: GroundTruth, tmp: Path) -> Path:
    path = tmp / f"fixture_{fmt}{SUFFIXES[fmt]}"
    WRITERS[fmt](path, gt)
    return path


def _quat_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance between unit quaternions up to the q ≡ -q sign ambiguity."""
    return np.minimum(np.abs(a - b).max(axis=1), np.abs(a + b).max(axis=1))


@pytest.fixture(scope="module")
def ground_truth() -> GroundTruth:
    return make_ground_truth(n=16, seed=7)


class TestQuaternionHelpers:
    def test_round_trip(self) -> None:
        rng = np.random.default_rng(3)
        q = rng.normal(size=(200, 4))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        q[q[:, 0] < 0] *= -1
        back = rotmat_to_quat(quat_to_rotmat(q))
        assert np.allclose(back, q, atol=1e-6)

    def test_round_trip_near_180_degrees(self) -> None:
        # w ≈ 0 exercises the non-trace Shepperd branches.
        q = np.array([[1e-9, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        back = rotmat_to_quat(quat_to_rotmat(q))
        assert np.all(_quat_dist(back, q) < 1e-6)

    def test_unnormalized_and_zero_quaternions(self) -> None:
        R = quat_to_rotmat(np.array([[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]))
        assert np.allclose(R[0], np.eye(3))  # scale-invariant
        assert np.allclose(R[1], np.eye(3))  # zero-norm → identity

    def test_rotmat_is_orthonormal(self) -> None:
        rng = np.random.default_rng(4)
        q = rng.normal(size=(50, 4))
        R = quat_to_rotmat(q)
        eye = np.einsum("nij,nkj->nik", R, R)
        assert np.allclose(eye, np.eye(3)[None], atol=1e-12)
        assert np.allclose(np.linalg.det(R), 1.0, atol=1e-12)


class TestOpacitySanitization:
    """Non-finite opacity must never ride into the color alpha channel."""

    @staticmethod
    def _splats(opacities: np.ndarray) -> ClassicalSplats:
        n = opacities.shape[0]
        return ClassicalSplats(
            positions=np.zeros((n, 3), dtype=np.float32),
            scales=np.ones((n, 3), dtype=np.float32),
            quaternions=np.tile(
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n, 1)
            ),
            opacities=opacities.astype(np.float32),
            colors=np.full((n, 3), 0.5, dtype=np.float32),
        )

    def test_nonfinite_opacity_becomes_finite_alpha(self) -> None:
        # A corrupt source (e.g. a malformed INRIA float opacity → sigmoid(NaN))
        # yields NaN/±inf opacity; classical_to_gsplat_data must map it to a
        # finite alpha (NaN/+inf → opaque 1.0, −inf → 0.0) so it can't poison
        # effective_amplitudes ranking or the INRIA re-export logit.
        cs = self._splats(np.array([np.nan, np.inf, -np.inf, 0.4]))
        data = classical_to_gsplat_data(cs)
        assert data.colors is not None and data.colors.shape[1] == 4
        alpha = data.colors[:, 3]
        assert np.all(np.isfinite(alpha))
        np.testing.assert_allclose(alpha, [1.0, 1.0, 0.0, 0.4], atol=1e-6)
        # Downstream ranking stays finite (was NaN pre-fix).
        from luxar.gsplats.utils.alpha import effective_amplitudes

        assert np.all(np.isfinite(effective_amplitudes(data)))


class TestReaders:
    @pytest.mark.parametrize("fmt", CLASSICAL_FORMATS)
    def test_reader_parity(self, fmt: str, ground_truth: GroundTruth) -> None:
        from luxar.gsplats.interop.classical_splats import _READERS

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_fixture(fmt, ground_truth, Path(tmp))
            cs = _READERS[fmt](path)

        assert isinstance(cs, ClassicalSplats)
        assert cs.source_format == fmt
        assert cs.n_splats == ground_truth.positions.shape[0]
        assert np.allclose(
            cs.positions, ground_truth.positions, atol=_POSITION_ATOL[fmt]
        )
        assert np.allclose(
            cs.scales, ground_truth.scales, rtol=_SCALE_RTOL[fmt], atol=1e-6
        )
        assert np.all(
            _quat_dist(cs.quaternions.astype(np.float64), ground_truth.quaternions)
            < _QUAT_ATOL[fmt]
        )
        assert np.allclose(
            cs.opacities, ground_truth.opacities, atol=_OPACITY_ATOL[fmt]
        )
        assert np.allclose(cs.colors, ground_truth.colors, atol=_COLOR_ATOL[fmt])

    def test_inria_sh_degree_derived_from_header(
        self, ground_truth: GroundTruth
    ) -> None:
        from luxar.gsplats.interop.classical_splats import read_inria_ply
        from luxar.gsplats.interop.tests._synthetic import write_inria_ply

        with tempfile.TemporaryDirectory() as tmp:
            for degree in (0, 1, 3):
                path = Path(tmp) / f"deg{degree}.ply"
                write_inria_ply(path, ground_truth, sh_degree=degree)
                assert read_inria_ply(path).sh_degree == degree

    def test_supersplat_12_property_chunk(self, ground_truth: GroundTruth) -> None:
        # Older SuperSplat files omit the 6 per-chunk color bounds (12-prop
        # chunk); color is then a raw unorm with no lerp. The reader must
        # accept both layouts.
        from luxar.gsplats.interop.classical_splats import read_supersplat_ply
        from luxar.gsplats.interop.tests._synthetic import write_supersplat_ply

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chunk12.ply"
            write_supersplat_ply(path, ground_truth, color_bounds=False)
            cs = read_supersplat_ply(path)
        assert cs.n_splats == ground_truth.positions.shape[0]
        assert np.allclose(cs.colors, ground_truth.colors, atol=1 / 255)
        assert np.allclose(cs.positions, ground_truth.positions, atol=5e-3)

    def test_spz_declares_y_up_others_do_not(self, ground_truth: GroundTruth) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for fmt in CLASSICAL_FORMATS:
                path = _write_fixture(fmt, ground_truth, Path(tmp))
                from luxar.gsplats.interop.classical_splats import _READERS

                assert _READERS[fmt](path).y_up is (fmt == "spz")


class TestDetection:
    @pytest.mark.parametrize("fmt", CLASSICAL_FORMATS)
    def test_detects_all_dialects(self, fmt: str, ground_truth: GroundTruth) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_fixture(fmt, ground_truth, Path(tmp))
            assert detect_classical_format(path) == fmt

    def test_rejects_plain_ply(self) -> None:
        header = (
            "ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
            "property float x\nproperty float y\nproperty float z\nend_header\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plain.ply"
            path.write_bytes(header.encode() + np.zeros(3, "<f4").tobytes())
            with pytest.raises(ValueError, match="not a recognized"):
                detect_classical_format(path)

    def test_rejects_unknown_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.xyz"
            path.write_bytes(b"nope")
            with pytest.raises(ValueError, match="unrecognized extension"):
                detect_classical_format(path)

    def test_rejects_spz_v4_with_clear_error(self) -> None:
        from luxar.gsplats.interop.classical_splats import read_spz

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v4.spz"
            path.write_bytes((0x5053474E).to_bytes(4, "little") + b"\x00" * 28)
            with pytest.raises(ValueError, match="v4"):
                read_spz(path)


class TestConversion:
    def test_covariance_matches_ground_truth(self, ground_truth: GroundTruth) -> None:
        # No orientation: Σ must equal R diag(s²) Rᵀ from the ground truth.
        cs = ClassicalSplats(
            positions=ground_truth.positions,
            scales=ground_truth.scales,
            quaternions=ground_truth.quaternions,
            opacities=ground_truth.opacities,
            colors=ground_truth.colors,
            source_format="inria",
        )
        data = classical_to_gsplat_data(cs, rotate_x180=False)
        L = unpack_tril(data.cholesky_factors.astype(np.float64), 3)
        sigma = L @ L.transpose(0, 2, 1)
        R = quat_to_rotmat(ground_truth.quaternions)
        s2 = ground_truth.scales.astype(np.float64) ** 2
        expected = (R * s2[:, None, :]) @ R.transpose(0, 2, 1)
        assert np.allclose(sigma, expected, rtol=1e-5, atol=1e-9)
        assert np.allclose(data.centers, ground_truth.positions, atol=1e-6)
        # Learned opacity rides in the color ALPHA channel (per-splat opacity);
        # amplitudes are constant 1 so the alpha factor is not double-counted.
        assert np.allclose(data.amplitudes, 1.0)
        assert data.colors is not None
        assert data.colors.shape[1] == 4
        assert np.allclose(data.colors[:, 3], ground_truth.opacities, atol=1e-6)
        # GSplatData stores LINEAR color; the source DC is display-referred
        # (sRGB), so the conversion applies sRGB → linear. Alpha is coverage,
        # not light — it gets no sRGB transfer.
        from luxar.gsplats.interop._color import srgb_to_linear

        assert np.allclose(
            data.colors[:, :3], srgb_to_linear(ground_truth.colors), atol=1e-6
        )

    def test_default_reorientation_rotates_x180(
        self, ground_truth: GroundTruth
    ) -> None:
        cs = ClassicalSplats(
            positions=ground_truth.positions,
            scales=ground_truth.scales,
            quaternions=ground_truth.quaternions,
            opacities=ground_truth.opacities,
            colors=ground_truth.colors,
            source_format="inria",
        )
        plain = classical_to_gsplat_data(cs, rotate_x180=False)
        rotated = classical_to_gsplat_data(cs)  # None → dialect default (on)
        M = np.diag([1.0, -1.0, -1.0])
        assert np.allclose(rotated.centers, plain.centers @ M.T, atol=1e-6)
        Lp = unpack_tril(plain.cholesky_factors.astype(np.float64), 3)
        Lr = unpack_tril(rotated.cholesky_factors.astype(np.float64), 3)
        sig_p = Lp @ Lp.transpose(0, 2, 1)
        sig_r = Lr @ Lr.transpose(0, 2, 1)
        assert np.allclose(sig_r, M @ sig_p @ M.T, rtol=1e-5, atol=1e-9)

    def test_y_up_source_skips_default_reorientation(
        self, ground_truth: GroundTruth
    ) -> None:
        cs = ClassicalSplats(
            positions=ground_truth.positions,
            scales=ground_truth.scales,
            quaternions=ground_truth.quaternions,
            opacities=ground_truth.opacities,
            colors=ground_truth.colors,
            source_format="spz",
            y_up=True,
        )
        data = classical_to_gsplat_data(cs)
        assert np.allclose(data.centers, ground_truth.positions, atol=1e-6)

    def test_flip_mirrors_axis_and_stays_pd(self, ground_truth: GroundTruth) -> None:
        cs = ClassicalSplats(
            positions=ground_truth.positions,
            scales=ground_truth.scales,
            quaternions=ground_truth.quaternions,
            opacities=ground_truth.opacities,
            colors=ground_truth.colors,
            source_format="inria",
        )
        flipped = classical_to_gsplat_data(cs, rotate_x180=False, flip="x")
        expected = ground_truth.positions.copy()
        expected[:, 0] *= -1
        assert np.allclose(flipped.centers, expected, atol=1e-6)
        L = unpack_tril(flipped.cholesky_factors.astype(np.float64), 3)
        assert np.all(np.diagonal(L, axis1=1, axis2=2) > 0)  # PD survived the flip

    def test_flip_rejects_bad_axis(self, ground_truth: GroundTruth) -> None:
        cs = ClassicalSplats(
            positions=ground_truth.positions,
            scales=ground_truth.scales,
            quaternions=ground_truth.quaternions,
            opacities=ground_truth.opacities,
            colors=ground_truth.colors,
            source_format="inria",
        )
        with pytest.raises(ValueError, match="flip axes"):
            classical_to_gsplat_data(cs, flip="q")

    def test_degenerate_scales_survive_non_pd_guard(self) -> None:
        gt = make_ground_truth(n=8, seed=11)
        scales = gt.scales.copy()
        scales[0] = 0.0  # fully degenerate splat
        scales[1, 2] = 0.0  # one collapsed axis
        cs = ClassicalSplats(
            positions=gt.positions[:8],
            scales=scales,
            quaternions=gt.quaternions,
            opacities=gt.opacities,
            colors=gt.colors,
            source_format="splat",
        )
        data = classical_to_gsplat_data(cs, rotate_x180=False)
        L = unpack_tril(data.cholesky_factors.astype(np.float64), 3)
        assert np.isfinite(L).all()
        assert np.all(np.diagonal(L, axis1=1, axis2=2) > 0)

    def test_empty_input_raises(self) -> None:
        empty = ClassicalSplats(
            positions=np.zeros((0, 3), np.float32),
            scales=np.zeros((0, 3), np.float32),
            quaternions=np.zeros((0, 4), np.float32),
            opacities=np.zeros(0, np.float32),
            colors=np.zeros((0, 3), np.float32),
            source_format="splat",
        )
        with pytest.raises(ValueError, match="empty"):
            classical_to_gsplat_data(empty)


class TestEndToEndFidelity:
    """read → convert → save → load: covariance survives the full pipeline."""

    @staticmethod
    def _cov_relF_p95(ref: np.ndarray, got: np.ndarray, d: int) -> float:
        Lr = unpack_tril(ref.astype(np.float64), d)
        Lg = unpack_tril(got.astype(np.float64), d)
        Sr = Lr @ Lr.transpose(0, 2, 1)
        Sg = Lg @ Lg.transpose(0, 2, 1)
        rel = np.linalg.norm(Sg - Sr, axis=(1, 2)) / (
            np.linalg.norm(Sr, axis=(1, 2)) + 1e-30
        )
        return float(np.percentile(rel, 95))

    @pytest.mark.parametrize("fmt", CLASSICAL_FORMATS)
    def test_import_save_load_round_trip(
        self, fmt: str, ground_truth: GroundTruth
    ) -> None:
        from luxar.gsplats.gsplat_data import GSplatData

        with tempfile.TemporaryDirectory() as tmp:
            src = _write_fixture(fmt, ground_truth, Path(tmp))
            data = import_gsplats(src)
            out = Path(tmp) / "imported.gsplats.zarr"
            # ordering="none": the default hilbert ordering permutes splats on
            # disk, which is fine for rendering but breaks row-wise comparison.
            data.save(out, ordering="none")
            loaded = GSplatData.load(out)

        assert loaded.n_splats == ground_truth.positions.shape[0]
        assert np.allclose(loaded.centers, data.centers, atol=1e-3)
        # AUTO-encoding covariance certificate bound (test_save_load.py contract).
        assert (
            self._cov_relF_p95(data.cholesky_factors, loaded.cholesky_factors, 3)
            <= 0.05
        )
        assert loaded.colors is not None
        assert np.allclose(loaded.colors, data.colors, atol=2 / 255)
        assert np.allclose(loaded.amplitudes, data.amplitudes, rtol=0.02, atol=5e-3)


class TestSceneApi:
    def test_add_gsplats_from_file_accepts_classical_file(
        self, ground_truth: GroundTruth
    ) -> None:
        from luxar import LuxarZarrCompiler
        from luxar.core.dimension_inference import build_dimensions_from_data

        with tempfile.TemporaryDirectory() as tmp:
            src = _write_fixture("splat", ground_truth, Path(tmp))
            data = import_gsplats(src)
            scene_path = Path(tmp) / "scene.luxar.zarr"
            with LuxarZarrCompiler(scene_path) as compiler:
                scene = compiler.create_scene(
                    dimensions=build_dimensions_from_data(data.centers)
                )
                node = scene.add_gsplats_from_file(
                    "imported", src, blending_mode="normal"
                )
                assert node.blending_mode == "normal"
            assert scene_path.exists()


class TestSog:
    """SOG-specific paths beyond the shared parametrized reader parity."""

    def test_reads_from_meta_json_path(self, ground_truth: GroundTruth) -> None:
        from luxar.gsplats.interop.classical_splats import read_sog
        from luxar.gsplats.interop.tests._synthetic import write_sog

        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            write_sog(bundle, ground_truth)
            # Pointing at meta.json resolves the same bundle as the directory.
            cs_dir = read_sog(bundle)
            cs_meta = read_sog(bundle / "meta.json")
        assert np.allclose(cs_dir.positions, cs_meta.positions)
        assert detect_classical_format(bundle / "meta.json") == "sog"

    def test_reads_from_sog_zip(self, ground_truth: GroundTruth) -> None:
        import zipfile

        from luxar.gsplats.interop.classical_splats import read_sog
        from luxar.gsplats.interop.tests._synthetic import write_sog

        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            write_sog(bundle, ground_truth)
            zpath = Path(tmp) / "scene.sog"
            with zipfile.ZipFile(zpath, "w") as zf:
                for member in bundle.iterdir():
                    zf.write(member, member.name)
            assert detect_classical_format(zpath) == "sog"
            cs = read_sog(zpath)
        assert cs.n_splats == ground_truth.positions.shape[0]
        assert np.allclose(cs.positions, ground_truth.positions, atol=5e-3)

    def test_import_end_to_end_records_source_format(
        self, ground_truth: GroundTruth
    ) -> None:
        from luxar.gsplats.interop.tests._synthetic import write_sog

        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            write_sog(bundle, ground_truth)
            data = import_gsplats(bundle)
        assert data.n_splats == ground_truth.positions.shape[0]
        assert data.colors is not None
        assert data.stats["interop"]["source_format"] == "sog"

    def test_rejects_unsupported_version(self, ground_truth: GroundTruth) -> None:
        import json

        from luxar.gsplats.interop.classical_splats import read_sog
        from luxar.gsplats.interop.tests._synthetic import write_sog

        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            write_sog(bundle, ground_truth)
            meta = json.loads((bundle / "meta.json").read_text())
            meta["version"] = 1
            (bundle / "meta.json").write_text(json.dumps(meta))
            with pytest.raises(ValueError, match="SOG version"):
                read_sog(bundle)

    def test_shn_present_in_meta_is_dropped(self, ground_truth: GroundTruth) -> None:
        # shN is optional and Luxar bakes DC only; a meta with an shN block must
        # still decode (DC color), reporting the source SH degree from bands.
        import json

        from luxar.gsplats.interop.classical_splats import read_sog
        from luxar.gsplats.interop.tests._synthetic import write_sog

        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            write_sog(bundle, ground_truth)
            meta = json.loads((bundle / "meta.json").read_text())
            meta["shN"] = {"count": 1, "bands": 2, "codebook": [0.0] * 256,
                           "files": ["shN_centroids.webp", "shN_labels.webp"]}
            (bundle / "meta.json").write_text(json.dumps(meta))
            cs = read_sog(bundle)  # must NOT try to read the (absent) shN files
        assert cs.sh_degree == 2
        assert np.allclose(cs.colors, ground_truth.colors, atol=0.01)

    def test_rejects_wrong_channel_count(self, ground_truth: GroundTruth) -> None:
        # quats/sh0 must be RGBA; a malformed RGB image should fail with a clear
        # message, not a bare IndexError deep in the decode.
        from PIL import Image

        from luxar.gsplats.interop.classical_splats import read_sog
        from luxar.gsplats.interop.tests._synthetic import write_sog

        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            write_sog(bundle, ground_truth)
            n = ground_truth.positions.shape[0]
            # Re-save sh0 as RGB (drop the alpha/opacity channel).
            rgba = np.asarray(Image.open(bundle / "sh0.webp"))
            Image.fromarray(rgba[:, :, :3], mode="RGB").save(
                bundle / "sh0.webp", format="WEBP", lossless=True
            )
            with pytest.raises(ValueError, match="channel"):
                read_sog(bundle)
        assert n > 0  # sanity: fixture was non-empty

    def test_rejects_count_exceeding_pixels(self, ground_truth: GroundTruth) -> None:
        import json

        from luxar.gsplats.interop.classical_splats import read_sog
        from luxar.gsplats.interop.tests._synthetic import write_sog

        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            write_sog(bundle, ground_truth)
            meta = json.loads((bundle / "meta.json").read_text())
            meta["count"] = 10_000_000  # far more than the fixture's pixels
            (bundle / "meta.json").write_text(json.dumps(meta))
            with pytest.raises(ValueError, match="pixels"):
                read_sog(bundle)

    def test_golden_decode_independent_of_writer(self) -> None:
        # Hardcoded byte-level bundle → spec-computed expectations, decoded
        # WITHOUT write_sog, so a shared encoder/decoder spec-misread can't hide.
        import json

        from PIL import Image

        from luxar.gsplats.interop.classical_splats import SH_C0, read_sog

        with tempfile.TemporaryDirectory() as tmp:
            b = Path(tmp)
            # 3 Gaussians in a 1x3 image (row-major, count=3).
            def wr(name, rows, mode):
                arr = np.array(rows, np.uint8).reshape(1, len(rows), len(rows[0]))
                Image.fromarray(arr, mode=mode).save(b / name, format="WEBP", lossless=True)

            # means: q16 = 0, 65535, and an ASYMMETRIC low=1/high=0 (=1) on x so
            # a swapped hi/lo byte order (→256) is caught, not just symmetric ends.
            wr("means_l.webp", [[0, 0, 0], [255, 255, 255], [1, 0, 0]], "RGB")
            wr("means_u.webp", [[0, 0, 0], [255, 255, 255], [0, 0, 0]], "RGB")
            wr("scales.webp", [[0, 0, 0], [1, 1, 1], [0, 0, 0]], "RGB")
            # quats: splat0 alpha=253 → mode 1 (x largest); splat1 alpha=252 → w.
            wr("quats.webp",
               [[128, 128, 128, 253], [128, 128, 128, 252], [128, 128, 128, 252]], "RGBA")
            # sh0: rgb idx into DC codebook; alpha = opacity byte.
            wr("sh0.webp", [[0, 0, 0, 51], [1, 1, 1, 255], [0, 0, 0, 128]], "RGBA")
            meta = {
                "version": 2, "count": 3,
                "means": {"mins": [-2.0, -2.0, -2.0], "maxs": [2.0, 2.0, 2.0],
                          "files": ["means_l.webp", "means_u.webp"]},
                "scales": {"codebook": [np.log(0.5)] + [np.log(2.0)] + [0.0] * 254,
                           "files": ["scales.webp"]},
                "quats": {"files": ["quats.webp"]},
                "sh0": {"codebook": [1.0, -1.0] + [0.0] * 254, "files": ["sh0.webp"]},
            }
            (b / "meta.json").write_text(json.dumps(meta))
            cs = read_sog(b)

        # Means: q16=0 → log-min=-2 → sign*expm1(2)= -(e^2-1); q16=65535 → +(e^2-1).
        exp_lo = -(np.expm1(2.0))
        exp_hi = np.expm1(2.0)
        assert np.allclose(cs.positions[0], exp_lo, atol=1e-3)
        assert np.allclose(cs.positions[1], exp_hi, atol=1e-3)
        # Asymmetric splat2 x: q16=1 (NOT 256) → decodes just above the min; a
        # swapped hi/lo byte order would give q16=256 and a very different x.
        n2 = -2.0 + 4.0 * (1.0 / 65535.0)
        exp_x2 = np.sign(n2) * np.expm1(abs(n2))
        assert np.allclose(cs.positions[2, 0], exp_x2, atol=1e-4)
        # Scales: exp(codebook[0])=0.5, exp(codebook[1])=2.0.
        assert np.allclose(cs.scales[0], 0.5, rtol=1e-3)
        assert np.allclose(cs.scales[1], 2.0, rtol=1e-3)
        # Quat modes: splat0 largest = x (index 1); splat1 largest = w (index 0).
        assert int(np.argmax(np.abs(cs.quaternions[0]))) == 1
        assert int(np.argmax(np.abs(cs.quaternions[1]))) == 0
        # Opacity from alpha byte: 51/255, 255/255, 128/255.
        assert np.allclose(cs.opacities, [51 / 255, 1.0, 128 / 255], atol=1e-6)
        # Color: 0.5 + SH_C0*codebook[idx]; idx0→+1.0, idx1→-1.0.
        assert np.allclose(cs.colors[0], np.clip(0.5 + SH_C0 * 1.0, 0, 1), atol=1e-6)
        assert np.allclose(cs.colors[1], np.clip(0.5 + SH_C0 * -1.0, 0, 1), atol=1e-6)


class TestColorSpace:
    """The classical DC color is sRGB (display-referred); Luxar stores linear."""

    def test_srgb_linear_are_inverse(self) -> None:
        from luxar.gsplats.interop._color import linear_to_srgb, srgb_to_linear

        c = np.linspace(0.0, 1.0, 257).reshape(-1, 1).repeat(3, axis=1)
        assert np.allclose(linear_to_srgb(srgb_to_linear(c)), c, atol=1e-5)
        assert np.allclose(srgb_to_linear(linear_to_srgb(c)), c, atol=1e-5)

    def test_srgb_to_linear_darkens_midtones(self) -> None:
        # The whole point: a mid sRGB 0.5 becomes ~0.214 linear, so after the
        # viewer's output OETF it lands back near 0.5 instead of washing bright.
        from luxar.gsplats.interop._color import srgb_to_linear

        assert abs(float(srgb_to_linear(np.array([0.5]))[0]) - 0.214) < 0.005
        # Endpoints are fixed points.
        assert float(srgb_to_linear(np.array([0.0]))[0]) == 0.0
        assert abs(float(srgb_to_linear(np.array([1.0]))[0]) - 1.0) < 1e-6

    def test_import_stores_linear_color(self, ground_truth: GroundTruth) -> None:
        # Every dialect: GSplatData.colors == srgb_to_linear(reader sRGB color).
        from luxar.gsplats.interop._color import srgb_to_linear
        from luxar.gsplats.interop.classical_splats import _READERS

        for fmt in CLASSICAL_FORMATS:
            with tempfile.TemporaryDirectory() as tmp:
                path = _write_fixture(fmt, ground_truth, Path(tmp))
                cs = _READERS[fmt](path)  # reader → sRGB display color
                data = import_gsplats(path)  # → GSplatData (linear)
            assert np.allclose(
                data.colors[:, :3], srgb_to_linear(cs.colors), atol=1e-5
            ), f"{fmt}: GSplatData.colors must be linear(reader color)"
            # Alpha = the reader's opacity, untouched by the sRGB transfer.
            assert np.allclose(
                data.colors[:, 3], np.clip(cs.opacities, 0.0, 1.0), atol=1e-6
            ), f"{fmt}: color alpha must carry the reader opacity verbatim"

    def test_import_export_color_round_trips_in_srgb(
        self, ground_truth: GroundTruth
    ) -> None:
        # A source PLY's sRGB DC survives import(→linear)→export(→sRGB) exactly.
        from luxar.gsplats.interop.inria_export import gsplat_data_to_inria_ply
        from luxar.gsplats.interop.tests._synthetic import write_inria_ply

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.ply"
            write_inria_ply(src, ground_truth)
            data = import_gsplats(src, rotate_x180=False)
            payload = gsplat_data_to_inria_ply(data, opacity_policy="amplitude")
            out = Path(tmp) / "out.ply"
            out.write_bytes(payload)
            from luxar.gsplats.interop.classical_splats import read_inria_ply

            reround = read_inria_ply(out)  # sRGB DC again
        assert np.allclose(reround.colors, ground_truth.colors, atol=2e-3)
