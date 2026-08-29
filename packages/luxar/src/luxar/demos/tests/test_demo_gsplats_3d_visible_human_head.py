"""Smoke tests for demo_gsplats_3d_visible_human_head.

Covers the deterministic image helpers and the scene builder's authored
blending — no network, no PNG IO, no GPU fit. The demo is loaded by file path
(see test_demo_ppi_flow_field).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar.demos import is_lfs_pointer, voxel_sampled_payload_agreement
from luxar.demos._cinematic_camera import CINEMATIC_FOV_DEG
from luxar.gsplats.gsplat_data import GSplatData

pytest.importorskip("scipy")

_DEMO_PATH = (
    Path(__file__).resolve().parents[1] / "demo_gsplats_3d_visible_human_head.py"
)
_DATA_MANIFEST_PATH = _DEMO_PATH.parent / "data_manifest.json"
# The pair SHIPPED IN THIS REPO — the 20,572,128-byte fit and the sidecar #1911
# resampled against it (1,911,192 rows, matching that fit's splat count). These
# track the manifest's `sha256`, deliberately: the sibling deep check validates
# them against MATERIALIZED git-LFS bytes, so they can only ever describe the
# in-repo copy.
#
# The cc-by deposition holds a different, also self-consistent pair — the refit
# (29,378,476 B, 1,908,888 rows) with its own resampled sidecar — and the
# manifest now records that separately as `hosted_sha256`/`hosted_bytes`. Do not
# "fix" these constants to the hosted digests: the deep check cannot verify bytes
# that are not on disk, and the two pairs must never be mixed. Pinning one file
# from each pair is the #1670 mis-ordering all over again, so each pair moves
# together or not at all.
_SHIPPED_FIT_SHA256 = "c6ebbab8c2d1bdff0d5fd35f7032c6a375724b5fe5e6d33e8ca4e6af6ab139f6"
_SHIPPED_COLORS_SHA256 = (
    "63bce184e56d6d3b5f8f6817100c66310984c45da65fd94cdbc0a42ac05abb44"
)


def _load_demo_module():
    name = "_luxar_demo_vh_head_for_tests"
    spec = importlib.util.spec_from_file_location(name, _DEMO_PATH)
    if spec is None or spec.loader is None:
        pytest.skip(f"Could not locate demo at {_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_demo = _load_demo_module()
_LFS_DIR = _DEMO_PATH.parent / "data" / _demo.DEMO_NAME
_LFS_FIT = _LFS_DIR / _demo.FIT_FILE
_LFS_COLORS = _LFS_DIR / _demo.COLORS_FILE
create_luxar_scene = _demo.create_luxar_scene
luminance = _demo.luminance
tissue_mask = _demo.tissue_mask
mask_background = _demo.mask_background
crop_to_content = _demo.crop_to_content
sample_colors = _demo.sample_colors
_save_colors_u8 = _demo._save_colors_u8
_load_colors_f32 = _demo._load_colors_f32


class TestColorRoundtrip:
    def test_uint8_roundtrip_within_quantization(self, tmp_path) -> None:
        colors = np.array([[0.0, 0.5, 1.0], [0.25, 0.75, 0.1]], dtype=np.float32)
        p = tmp_path / "c.npz"
        _save_colors_u8(colors, p)
        assert p.stat().st_size > 0
        loaded = _load_colors_f32(p)
        assert loaded.dtype == np.float32
        assert loaded.min() >= 0.0 and loaded.max() <= 1.0
        np.testing.assert_allclose(loaded, colors, atol=1.0 / 255 + 1e-6)

    def test_loads_legacy_float_npz(self, tmp_path) -> None:
        p = tmp_path / "cf.npz"
        np.savez_compressed(p, colors=np.array([[0.2, 0.4, 0.6]], dtype=np.float32))
        loaded = _load_colors_f32(p)
        assert loaded.dtype == np.float32
        np.testing.assert_allclose(loaded, [[0.2, 0.4, 0.6]], atol=1e-6)


class TestLuminance:
    def test_white_and_black(self) -> None:
        rgb = np.array([[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]], dtype=np.float32)
        lum = luminance(rgb)
        assert lum.shape == (1, 2)
        np.testing.assert_allclose(lum[0, 0], 1.0, atol=1e-6)
        np.testing.assert_allclose(lum[0, 1], 0.0, atol=1e-6)

    def test_rec601_weights(self) -> None:
        assert abs(float(luminance(np.array([[0.0, 1.0, 0.0]]))[0]) - 0.587) < 1e-6


class TestTissueMask:
    def test_warm_kept_blue_and_dark_rejected(self) -> None:
        rgb = np.array(
            [
                [0.8, 0.6, 0.4],  # warm tissue → keep
                [0.1, 0.2, 0.9],  # blue gel → reject
                [0.02, 0.02, 0.02],  # near-black → reject
            ],
            dtype=np.float32,
        )
        m = tissue_mask(rgb)
        assert list(m) == [True, False, False]

    def test_mask_background_zeros_nontissue(self) -> None:
        rgb = np.array([[[0.8, 0.6, 0.4], [0.1, 0.2, 0.9]]], dtype=np.float32)
        out = mask_background(rgb)
        np.testing.assert_allclose(out[0, 0], [0.8, 0.6, 0.4], atol=1e-6)
        np.testing.assert_array_equal(out[0, 1], [0.0, 0.0, 0.0])


class TestCropToContent:
    def test_crops_to_bounding_box(self) -> None:
        vol = np.zeros((3, 8, 8, 3), dtype=np.float32)
        vol[1, 2:5, 3:6] = [0.5, 0.3, 0.2]  # a small warm block
        cropped, box = crop_to_content(vol, pad=0)
        z0, z1, y0, y1, x0, x1 = box
        assert (z0, z1) == (1, 2)
        assert (y0, y1) == (2, 5)
        assert (x0, x1) == (3, 6)
        assert cropped.shape == (1, 3, 3, 3)

    def test_empty_volume_returns_unchanged(self) -> None:
        vol = np.zeros((2, 4, 4, 3), dtype=np.float32)
        cropped, box = crop_to_content(vol)
        assert cropped.shape == vol.shape


def _tiny_gsplat_data(n: int = 8, seed: int = 0) -> GSplatData:
    """A handful of valid splats — no GPU fit, enough to build the scene."""
    rng = np.random.default_rng(seed)
    return GSplatData(
        centers=rng.uniform(-4.0, 4.0, (n, 3)).astype(np.float32),
        amplitudes=rng.uniform(0.2, 1.0, n).astype(np.float32),
        cholesky_factors=np.tile([1, 0, 1, 0, 0, 1], (n, 1)).astype(np.float32),
    )


class TestSceneBlending:
    def test_scene_bakes_volumetric_blending(self, tmp_path) -> None:
        # The demo exists to render photographic-color anatomy under
        # volumetric (emission-absorption) compositing; pin the authored mode
        # so a silent revert to additive glow is caught (the helper smoke
        # tests never build the scene). See the interop demos' blending test.
        fit = _tiny_gsplat_data()
        colors = (
            np.random.default_rng(1)
            .uniform(0.1, 0.9, (fit.n_splats, 3))
            .astype(np.float32)
        )
        out = create_luxar_scene(fit, colors, tmp_path / "vh.luxar.zarr")
        root = zarr.open_group(str(out), mode="r")
        node = root["visible_human_head"]
        assert dict(node.attrs).get("blending_mode") == "volumetric"
        camera = dict(dict(root.attrs)["viewer_config"])["camera"]
        assert camera["up"] == [-1.0, 0.0, 0.0]
        assert camera["position"][1] < 0.0
        centered = fit.center_at_centroid()
        half = np.maximum(
            np.abs(centered.centers.max(axis=0)),
            np.abs(centered.centers.min(axis=0)),
        )
        expected_distance = (
            max(half[0], half[2]) * 1.15 / np.tan(np.radians(CINEMATIC_FOV_DEG / 2.0))
        )
        assert camera["position"][1] == pytest.approx(-expected_distance)
        assert "fov" not in camera


class TestSampleColors:
    def test_nearest_sample_and_clamp(self) -> None:
        vol = np.zeros((2, 2, 2, 3), dtype=np.float32)
        vol[0, 0, 0] = [1.0, 0.0, 0.0]
        vol[1, 1, 1] = [0.0, 0.0, 1.0]
        centers = np.array(
            [[0.0, 0.0, 0.0], [1.4, 1.4, 1.4], [99.0, 99.0, 99.0]], dtype=np.float32
        )
        cols = sample_colors(vol, centers)
        assert cols.shape == (3, 3)
        np.testing.assert_allclose(cols[0], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(cols[1], [0.0, 0.0, 1.0])  # rounds to (1,1,1)
        np.testing.assert_allclose(cols[2], [0.0, 0.0, 1.0])  # clamped to (1,1,1)


def _scattered_gsplat_data(n: int, extent: float, seed: int = 0) -> GSplatData:
    """Splats scattered over a small voxel grid — no GPU fit needed."""
    rng = np.random.default_rng(seed)
    return GSplatData(
        centers=rng.uniform(0.0, extent, (n, 3)).astype(np.float32),
        amplitudes=rng.uniform(0.2, 1.0, n).astype(np.float32),
        cholesky_factors=np.tile([1, 0, 1, 0, 0, 1], (n, 1)).astype(np.float32),
    )


class TestColorSidecarOrdering:
    """The colors sidecar must be sampled in the SAVED store's splat order.

    ``GSplatData.save`` reorders splats spatially, so colors sampled at the
    in-memory fit's centers describe different splats than the ones ``load``
    hands back — every splat would render some other splat's color. See #1670.
    """

    @staticmethod
    def _through_sidecar(colors: np.ndarray, tmp_path, name: str) -> np.ndarray:
        """Colors as the uint8 sidecar round-trip returns them.

        Written and read through the demo's OWN ``_save_colors_u8`` /
        ``_load_colors_f32`` rather than a re-implementation of the quantization
        here — a local copy of the rounding rule would keep agreeing with a
        changed one only by luck.
        """
        p = tmp_path / name
        _save_colors_u8(colors, p)
        return _load_colors_f32(p)

    def test_colors_align_with_the_returned_fit_not_the_input(
        self, tmp_path, monkeypatch
    ) -> None:
        n = 6000
        rng = np.random.default_rng(7)
        rgb_vol = rng.uniform(0.0, 1.0, (16, 16, 16, 3)).astype(np.float32)
        fit = _scattered_gsplat_data(n, extent=15.49)
        monkeypatch.setattr(_demo, "LOCAL_FIT", tmp_path / _demo.FIT_FILE)
        monkeypatch.setattr(_demo, "LOCAL_COLORS", tmp_path / _demo.COLORS_FILE)

        _, colors = _demo.save_and_sample_colors(fit, rgb_vol)

        # Non-vacuity: the writer really did permute this input. Without it the
        # pre/post-save samplings would agree and prove nothing.
        moved = np.any(np.rint(stored.centers) != np.rint(fit.centers), axis=1)
        assert int(moved.sum()) > n // 2, "hilbert ordering did not permute the input"

        # An INDEPENDENT nearest-voxel sampling at the returned centers, taken
        # through the sidecar's own quantization. Comparing the returned `colors`
        # against `_load_colors_f32(LOCAL_COLORS)` instead would be vacuous:
        # `save_and_sample_colors` RETURNS exactly that read-back.
        expected = self._through_sidecar(
            sample_colors(rgb_vol, stored.centers), tmp_path, "expected.npz"
        )
        # Aligned with the fit that is RETURNED (== what the cache reloads)...
        np.testing.assert_array_equal(colors, expected)
        # ...and the sidecar ON DISK carries those same rows.
        np.testing.assert_array_equal(_load_colors_f32(_demo.LOCAL_COLORS), expected)
        # ...and NOT the pre-save sampling the old code persisted.
        assert not np.array_equal(
            expected,
            self._through_sidecar(
                sample_colors(rgb_vol, fit.centers), tmp_path, "presave.npz"
            ),
        )

    def test_guard_accepts_the_aligned_pair_and_rejects_a_permuted_one(
        self, tmp_path, monkeypatch
    ) -> None:
        n = 6000
        rng = np.random.default_rng(8)
        rgb_vol = rng.uniform(0.0, 1.0, (16, 16, 16, 3)).astype(np.float32)
        fit = _scattered_gsplat_data(n, extent=15.49, seed=1)
        monkeypatch.setattr(_demo, "LOCAL_FIT", tmp_path / _demo.FIT_FILE)
        monkeypatch.setattr(_demo, "LOCAL_COLORS", tmp_path / _demo.COLORS_FILE)

        stored, colors = _demo.save_and_sample_colors(fit, rgb_vol)
        assert _demo._colors_match_fit(stored, colors, "aligned")
        # A permuted sidecar (the #1670 bug) and a wrong-length one are refused.
        permuted = colors[rng.permutation(n)]
        assert not _demo._colors_match_fit(stored, permuted, "permuted")
        assert not _demo._colors_match_fit(stored, colors[:-1], "truncated")

    def test_a_pair_too_sparse_to_judge_is_accepted(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Unverifiable is NOT a failure — the guard must accept and move on.

        A fit whose splats never share a voxel gives the helper no evidence
        (``None``). Rejecting there would refit every sparse dataset forever, so
        this branch is load-bearing; it is also the one a stubbed test uses. The
        accept must be TRACED though: silently accepting is how a misordered
        sidecar on a sparse fit would render unnoticed.
        """
        # One splat per voxel on a coarse lattice → no same-voxel pair at all.
        grid = (
            np.stack(np.meshgrid(*[np.arange(6.0)] * 3, indexing="ij"), axis=-1)
            .reshape(-1, 3)
            .astype(np.float32)
        )
        fit = GSplatData(
            centers=grid,
            amplitudes=np.ones(len(grid), dtype=np.float32),
            cholesky_factors=np.tile([1, 0, 1, 0, 0, 1], (len(grid), 1)).astype(
                np.float32
            ),
        )
        rng = np.random.default_rng(9)
        colors = rng.uniform(0.0, 1.0, (len(grid), 3)).astype(np.float32)
        assert voxel_sampled_payload_agreement(fit.centers, colors) is None, (
            "fixture must be unverifiable for this branch to be exercised"
        )
        assert _demo._colors_match_fit(fit, colors, "unverifiable")
        assert "UNVERIFIED" in capsys.readouterr().out


def _same_voxel_pairs(centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(order, collides)`` — the guard's own pair structure, recomputed here.

    Mirrors ``voxel_sampled_payload_agreement``: sort by voxel key (every center
    column), then take adjacent equal-voxel positions. Needed so a test can break
    an EXACT number of pairs.
    """
    voxels = np.rint(centers).astype(np.int64)
    order = np.lexsort(voxels.T[::-1])
    v = voxels[order]
    return order, np.flatnonzero((v[1:] == v[:-1]).all(axis=1))


def _lone_pairs(collides: np.ndarray) -> np.ndarray:
    """Colliding positions whose voxel holds EXACTLY two splats.

    Breaking one of those breaks exactly one pair; in a 3-splat voxel the middle
    splat belongs to two pairs, so editing it would move the score by two.
    """
    isolated = ~np.isin(collides - 1, collides) & ~np.isin(collides + 1, collides)
    return collides[isolated]


class TestColorAgreementThreshold:
    """``MIN_COLOR_AGREEMENT`` must be the value the guard actually decides on.

    Without this, the constant survived being set to 0.5 with every test green.
    """

    @staticmethod
    def _saved_pair(tmp_path, monkeypatch, tag: str):
        rng = np.random.default_rng(41)
        rgb_vol = rng.uniform(0.0, 1.0, (16, 16, 16, 3)).astype(np.float32)
        fit = _scattered_gsplat_data(6000, extent=15.49, seed=3)
        monkeypatch.setattr(_demo, "LOCAL_FIT", tmp_path / (tag + _demo.FIT_FILE))
        monkeypatch.setattr(_demo, "LOCAL_COLORS", tmp_path / (tag + _demo.COLORS_FILE))
        return _demo.save_and_sample_colors(fit, rgb_vol)

    def _pair_with_broken(self, tmp_path, monkeypatch, n_broken: int):
        """A saved pair with EXACTLY ``n_broken`` same-voxel pairs disagreeing."""
        stored, colors = self._saved_pair(tmp_path, monkeypatch, "probe-")
        order, collides = _same_voxel_pairs(stored.centers)
        lone = _lone_pairs(collides)
        assert len(lone) >= n_broken, "not enough two-splat voxels to break"
        broken = colors.copy()
        for c in lone[:n_broken]:
            broken[order[c + 1]] = 1.0 - colors[order[c]]  # a different row
        return stored, broken

    def _tolerated_breaks(self, tmp_path, monkeypatch) -> int:
        """How many broken pairs the constant still tolerates, for this fixture.

        Counted on the SAVED fit (the writer permutes, so the pre-save centers are
        the wrong thing to count), hence the throwaway save.
        """
        stored, _ = self._saved_pair(tmp_path, monkeypatch, "count-")
        _, collides = _same_voxel_pairs(stored.centers)
        return int(np.floor((1.0 - _demo.MIN_COLOR_AGREEMENT) * len(collides)))

    def test_just_above_the_threshold_is_accepted(self, tmp_path, monkeypatch) -> None:
        n_broken = self._tolerated_breaks(tmp_path, monkeypatch)
        stored, broken = self._pair_with_broken(tmp_path, monkeypatch, n_broken)
        measured = voxel_sampled_payload_agreement(stored.centers, broken)
        assert measured is not None and measured >= _demo.MIN_COLOR_AGREEMENT
        assert _demo._colors_match_fit(stored, broken, "just above")

    def test_just_below_the_threshold_is_rejected(self, tmp_path, monkeypatch) -> None:
        n_broken = self._tolerated_breaks(tmp_path, monkeypatch) + 1
        stored, broken = self._pair_with_broken(tmp_path, monkeypatch, n_broken)
        measured = voxel_sampled_payload_agreement(stored.centers, broken)
        assert measured is not None and measured < _demo.MIN_COLOR_AGREEMENT
        assert not _demo._colors_match_fit(stored, broken, "just below")

    def test_a_near_miss_reordering_is_rejected(self, tmp_path, monkeypatch) -> None:
        """An ABSOLUTE pin, independent of what the constant currently says.

        A near-miss reordering — a genuinely misindexed sidecar that happens to
        keep most same-voxel pairs together — lands JUST under the gate. Measured
        on the CT demo's shipped pair (660,934 splats) by permuting its aligned
        sidecar the way each mistake would have written it: the writer's own
        morton order instead of hilbert scores 0.901, a roll-by-one 0.955, an
        adjacent-pair swap 0.962. So the pin is set at ~0.972, just ABOVE the
        worst of them: any constant at or below 0.972 accepts this pair and turns
        the test red, which is what stops the gate being lowered under a real
        near miss (0.90 — the level pinned before — was under three of them).
        """
        stored, _ = self._saved_pair(tmp_path, monkeypatch, "nm-")
        _, collides = _same_voxel_pairs(stored.centers)
        stored, broken = self._pair_with_broken(
            tmp_path, monkeypatch, int(np.floor(0.028 * len(collides)))
        )
        measured = voxel_sampled_payload_agreement(stored.centers, broken)
        # Strictly above 0.97, so `MIN_COLOR_AGREEMENT = 0.97` ACCEPTS this and
        # goes red rather than passing on a rounding coincidence.
        assert measured is not None and 0.970 < measured < 0.975
        assert not _demo._colors_match_fit(stored, broken, "near miss")


class TestRejectedPairFallsThroughToRefit:
    """A rejected (fit, colors) pair must trigger a REFIT, not render nonsense.

    The user-visible point of #1670. The shipped sidecar has since been
    regenerated against the shipped store, so this is no longer the state the
    demo is in — but detecting a mismatch is only half the contract:
    ``load_or_build`` has to actually fall through to download-and-refit, which
    is what keeps a future mis-ordered pair from rendering nonsense.
    """

    @staticmethod
    def _sentinel_setup(tmp_path, monkeypatch, *, permute: bool):
        n = 6000
        rng = np.random.default_rng(42)
        rgb_vol = rng.uniform(0.0, 1.0, (16, 16, 16, 3)).astype(np.float32)
        fit = _scattered_gsplat_data(n, extent=15.49, seed=5)
        monkeypatch.setattr(_demo, "LOCAL_FIT", tmp_path / _demo.FIT_FILE)
        monkeypatch.setattr(_demo, "LOCAL_COLORS", tmp_path / _demo.COLORS_FILE)
        _, colors = _demo.save_and_sample_colors(fit, rgb_vol)
        if permute:
            _save_colors_u8(colors[rng.permutation(n)], _demo.LOCAL_COLORS)

        monkeypatch.setattr(_demo, "RECOMPUTE", False)

        def _manifest_unavailable(*args, **kwargs):
            raise _demo.DatasetUnavailable("no cached, LFS, or hosted pair")

        monkeypatch.setattr(_demo, "ensure_dataset", _manifest_unavailable)
        monkeypatch.setattr(_demo, "warn_if_no_cuda_gpu", lambda: None)

        sentinel_fit = _scattered_gsplat_data(4, extent=1.0, seed=6)
        sentinel_colors = np.zeros((4, 3), dtype=np.float32)
        monkeypatch.setattr(_demo, "download_head_slices", lambda: tmp_path)
        monkeypatch.setattr(_demo, "assemble_volume", lambda *a, **k: (None, None))
        monkeypatch.setattr(
            _demo, "fit_head", lambda *a, **k: (sentinel_fit, sentinel_colors)
        )
        return colors, sentinel_fit, sentinel_colors

    def test_permuted_sidecar_falls_through_to_the_refit(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        _, sentinel_fit, sentinel_colors = self._sentinel_setup(
            tmp_path, monkeypatch, permute=True
        )
        got_fit, got_colors = _demo.load_or_build()
        assert got_fit is sentinel_fit, "a rejected pair was rendered anyway"
        assert got_colors is sentinel_colors
        assert "Using this machine's own earlier refit" not in capsys.readouterr().out

    def test_aligned_sidecar_is_used_instead_of_refitting(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        colors, sentinel_fit, _ = self._sentinel_setup(
            tmp_path, monkeypatch, permute=False
        )
        got_fit, got_colors = _demo.load_or_build()
        assert got_fit is not sentinel_fit, "an aligned pair triggered a refit"
        np.testing.assert_array_equal(got_colors, colors)
        assert "Using this machine's own earlier refit" in capsys.readouterr().out

    def test_a_corrupt_local_fit_self_heals_instead_of_raising(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """Rubble in the local-fit namespace must not brick every future launch.

        These bytes have no checksum, no remote and no second copy, so an
        unguarded ``GSplatData.load`` here raised ``BadZipFile`` out of
        ``load_or_build`` on EVERY launch, with a manual delete as the only
        recovery. The refit below already overwrites the file; it just has to be
        reached.
        """
        _, sentinel_fit, sentinel_colors = self._sentinel_setup(
            tmp_path, monkeypatch, permute=False
        )
        _demo.LOCAL_FIT.write_bytes(b"a Ctrl-C mid-save, not a zip")

        got_fit, got_colors = _demo.load_or_build()

        assert got_fit is sentinel_fit, "a corrupt local fit was not healed"
        assert got_colors is sentinel_colors
        out = capsys.readouterr().out
        assert "could not be loaded" in out and _demo.LOCAL_FIT.name in out, (
            "the failure must be reported loudly, with the path and the error"
        )


class TestManifestPairIsGuardedToo:
    """The manifest-resolved pair must still run the semantic alignment guard.

    Manifest digests prove that each file is expected, not that two newly pinned
    positional files belong to the same generation. The pair guard remains the
    last defence against rendering a fit with another generation's colors.
    """

    @staticmethod
    def _manifest_setup(tmp_path, monkeypatch, *, permute: bool):
        """Materialize a manifest-shaped pair under tmp_path and stub the refit."""
        n = 6000
        rng = np.random.default_rng(43)
        rgb_vol = rng.uniform(0.0, 1.0, (16, 16, 16, 3)).astype(np.float32)
        fit = _scattered_gsplat_data(n, extent=15.49, seed=7)

        manifest_dir = tmp_path / "manifest"
        monkeypatch.setattr(_demo, "LOCAL_FIT", manifest_dir / _demo.FIT_FILE)
        monkeypatch.setattr(_demo, "LOCAL_COLORS", manifest_dir / _demo.COLORS_FILE)
        stored, colors = _demo.save_and_sample_colors(fit, rgb_vol)
        if permute:
            _save_colors_u8(
                colors[rng.permutation(n)], manifest_dir / _demo.COLORS_FILE
            )
            colors = _load_colors_f32(manifest_dir / _demo.COLORS_FILE)

        monkeypatch.setattr(_demo, "LOCAL_FIT", tmp_path / "local" / _demo.FIT_FILE)
        monkeypatch.setattr(
            _demo, "LOCAL_COLORS", tmp_path / "local" / _demo.COLORS_FILE
        )
        monkeypatch.setattr(
            _demo,
            "ensure_dataset",
            lambda *a, **k: [
                manifest_dir / _demo.FIT_FILE,
                manifest_dir / _demo.COLORS_FILE,
            ],
        )

        monkeypatch.setattr(_demo, "RECOMPUTE", False)
        monkeypatch.setattr(_demo, "warn_if_no_cuda_gpu", lambda: None)
        sentinel_fit = _scattered_gsplat_data(4, extent=1.0, seed=8)
        sentinel_colors = np.zeros((4, 3), dtype=np.float32)
        monkeypatch.setattr(_demo, "download_head_slices", lambda: tmp_path)
        monkeypatch.setattr(_demo, "assemble_volume", lambda *a, **k: (None, None))
        monkeypatch.setattr(
            _demo, "fit_head", lambda *a, **k: (sentinel_fit, sentinel_colors)
        )
        return colors, sentinel_fit, sentinel_colors

    def test_a_permuted_manifest_sidecar_falls_through_to_the_refit(
        self, tmp_path, monkeypatch
    ) -> None:
        _, sentinel_fit, sentinel_colors = self._manifest_setup(
            tmp_path, monkeypatch, permute=True
        )
        got_fit, got_colors = _demo.load_or_build()
        assert got_fit is sentinel_fit, "a rejected manifest pair was rendered anyway"
        assert got_colors is sentinel_colors

    def test_an_aligned_manifest_sidecar_is_used_instead_of_refitting(
        self, tmp_path, monkeypatch
    ) -> None:
        colors, sentinel_fit, _ = self._manifest_setup(
            tmp_path, monkeypatch, permute=False
        )
        got_fit, got_colors = _demo.load_or_build()
        assert got_fit is not sentinel_fit, "an aligned manifest pair triggered a refit"
        np.testing.assert_array_equal(got_colors, colors)


class TestShippedPairIsAligned:
    """The SHIPPED (fit, colors) pair must pass the demo's own guard.

    Every other test here exercises the guard on synthetic pairs, which is why
    #1670 survived: the mechanism was covered and the artifact was not. The
    shipped sidecar had been sampled in the pre-save splat order, disagreed with
    the shipped store at 0.00097, and the demo silently fell through to a 1.1 GB
    download and refit on every run for anyone who pulled Git LFS.

    A sidecar carries no positions, so a mis-ordered one cannot be repaired in
    place — it has to be resampled from the volume. The manifest-pin test runs
    without materialized LFS assets in CI; bump its constants only after this
    deep pair check passes on a checkout where ``git lfs pull`` has run.

    Know the deep check's resolution before trusting it. The metric only sees
    splats that SHARE a voxel — 136,703 of 1,911,192 here, 7.15% — so it catches
    wholesale reordering and nothing finer. Measured against the real sidecar: a
    full permutation scores 0.00001, a roll by one 0.03546, a length change is
    caught outright, but swapping two arbitrary rows still passes. That is the
    right trade for the failure this guards (a sampling pass writing the whole
    array in the wrong order, i.e. #1670), and the wrong tool for per-splat
    corruption.

    It also cannot detect a resample taken in a DRIFTED coordinate frame — see
    the rejection branch in ``load_or_build``, and the three out-of-band frame
    checks recorded in the demo's module docstring.
    """

    def test_manifest_keeps_the_verified_pair_pinned(self) -> None:
        manifest = json.loads(_DATA_MANIFEST_PATH.read_text())
        files = {
            entry["name"]: entry["sha256"]
            for entry in manifest["datasets"]["gsplats_visible_human_head"]["files"]
        }

        expected = {
            _demo.FIT_FILE: _SHIPPED_FIT_SHA256,
            _demo.COLORS_FILE: _SHIPPED_COLORS_SHA256,
        }
        assert expected.items() <= files.items(), (
            "the shipped fit/colors pair changed; materialize Git LFS, rerun "
            "test_shipped_colors_belong_to_the_shipped_fit, then update both "
            "verified sha256 constants together"
        )

    @pytest.mark.slow
    def test_shipped_colors_belong_to_the_shipped_fit(self) -> None:
        if is_lfs_pointer(_LFS_FIT) or not _LFS_FIT.exists():
            pytest.skip(
                "Visible Human Git LFS assets are not materialized (run 'git lfs pull')"
            )
        if is_lfs_pointer(_LFS_COLORS) or not _LFS_COLORS.exists():
            pytest.skip(
                "Visible Human Git LFS colors sidecar is not materialized "
                "(run 'git lfs pull')"
            )

        fit = GSplatData.load(_LFS_FIT, include_stats=False)
        colors = _demo._load_colors_f32(_LFS_COLORS)

        assert len(colors) == len(fit.centers), (
            f"{len(colors):,} colors for {len(fit.centers):,} splats — the "
            "sidecar does not belong to this fit"
        )
        agreement = _demo.voxel_sampled_payload_agreement(fit.centers, colors)
        assert agreement is not None, "too few same-voxel splats to verify"
        assert agreement >= _demo.MIN_COLOR_AGREEMENT, (
            f"shipped pair agrees at {agreement:.5f} < "
            f"{_demo.MIN_COLOR_AGREEMENT} — the colors are not in the shipped "
            "store's splat order, so the demo will refit on every run (#1670). "
            "Resample the sidecar at the shipped store's centers."
        )
