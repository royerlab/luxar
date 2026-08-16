"""Smoke tests for demo_gsplats_3d_visible_human_head.

Covers the deterministic image helpers and the scene builder's authored
blending — no network, no PNG IO, no GPU fit. The demo is loaded by file path
(see test_demo_ppi_flow_field).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar.demos import voxel_sampled_payload_agreement
from luxar.gsplats.gsplat_data import GSplatData

pytest.importorskip("scipy")

_DEMO_PATH = (
    Path(__file__).resolve().parents[1] / "demo_gsplats_3d_visible_human_head.py"
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
        node = zarr.open_group(str(out), mode="r")["visible_human_head"]
        assert dict(node.attrs).get("blending_mode") == "volumetric"


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
        monkeypatch.setattr(_demo, "CACHE_FIT", tmp_path / _demo.FIT_FILE)
        monkeypatch.setattr(_demo, "CACHE_COLORS", tmp_path / _demo.COLORS_FILE)

        stored, colors = _demo.save_and_sample_colors(fit, rgb_vol)

        # Non-vacuity: the writer really did permute this input. Without it the
        # pre/post-save samplings would agree and prove nothing.
        moved = np.any(np.rint(stored.centers) != np.rint(fit.centers), axis=1)
        assert int(moved.sum()) > n // 2, "hilbert ordering did not permute the input"

        # An INDEPENDENT nearest-voxel sampling at the returned centers, taken
        # through the sidecar's own quantization. Comparing the returned `colors`
        # against `_load_colors_f32(CACHE_COLORS)` instead would be vacuous:
        # `save_and_sample_colors` RETURNS exactly that read-back.
        expected = self._through_sidecar(
            sample_colors(rgb_vol, stored.centers), tmp_path, "expected.npz"
        )
        # Aligned with the fit that is RETURNED (== what the cache reloads)...
        np.testing.assert_array_equal(colors, expected)
        # ...and the sidecar ON DISK carries those same rows.
        np.testing.assert_array_equal(_load_colors_f32(_demo.CACHE_COLORS), expected)
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
        monkeypatch.setattr(_demo, "CACHE_FIT", tmp_path / _demo.FIT_FILE)
        monkeypatch.setattr(_demo, "CACHE_COLORS", tmp_path / _demo.COLORS_FILE)

        stored, colors = _demo.save_and_sample_colors(fit, rgb_vol)
        assert _demo._colors_match_fit(stored, colors, "aligned")
        # A permuted sidecar (the #1670 bug) and a wrong-length one are refused.
        permuted = colors[rng.permutation(n)]
        assert not _demo._colors_match_fit(stored, permuted, "permuted")
        assert not _demo._colors_match_fit(stored, colors[:-1], "truncated")

    def test_a_pair_too_sparse_to_judge_is_accepted(self) -> None:
        """Unverifiable is NOT a failure — the guard must accept and move on.

        A fit whose splats never share a voxel gives the helper no evidence
        (``None``). Rejecting there would refit every sparse dataset forever, so
        this branch is load-bearing; it is also the one a stubbed test uses.
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
        monkeypatch.setattr(_demo, "CACHE_FIT", tmp_path / (tag + _demo.FIT_FILE))
        monkeypatch.setattr(_demo, "CACHE_COLORS", tmp_path / (tag + _demo.COLORS_FILE))
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

    The user-visible point of #1670 — and the state this demo is in today, since
    its shipped sidecar is misordered. Detecting the mismatch is only half of it:
    ``load_or_build`` has to actually fall through to download-and-refit.
    """

    @staticmethod
    def _sentinel_setup(tmp_path, monkeypatch, *, permute: bool):
        n = 6000
        rng = np.random.default_rng(42)
        rgb_vol = rng.uniform(0.0, 1.0, (16, 16, 16, 3)).astype(np.float32)
        fit = _scattered_gsplat_data(n, extent=15.49, seed=5)
        monkeypatch.setattr(_demo, "CACHE_FIT", tmp_path / _demo.FIT_FILE)
        monkeypatch.setattr(_demo, "CACHE_COLORS", tmp_path / _demo.COLORS_FILE)
        _, colors = _demo.save_and_sample_colors(fit, rgb_vol)
        if permute:
            _save_colors_u8(colors[rng.permutation(n)], _demo.CACHE_COLORS)

        monkeypatch.setattr(_demo, "RECOMPUTE", False)
        # The shipped LFS assets must not rescue (or mask) the outcome.
        monkeypatch.setattr(_demo, "LFS_FIT", tmp_path / "absent.gsplats.zarr.zip")
        monkeypatch.setattr(_demo, "LFS_COLORS", tmp_path / "absent.npz")
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
        self, tmp_path, monkeypatch
    ) -> None:
        _, sentinel_fit, sentinel_colors = self._sentinel_setup(
            tmp_path, monkeypatch, permute=True
        )
        got_fit, got_colors = _demo.load_or_build()
        assert got_fit is sentinel_fit, "a rejected pair was rendered anyway"
        assert got_colors is sentinel_colors

    def test_aligned_sidecar_is_used_instead_of_refitting(
        self, tmp_path, monkeypatch
    ) -> None:
        colors, sentinel_fit, _ = self._sentinel_setup(
            tmp_path, monkeypatch, permute=False
        )
        got_fit, got_colors = _demo.load_or_build()
        assert got_fit is not sentinel_fit, "an aligned pair triggered a refit"
        np.testing.assert_array_equal(got_colors, colors)


class TestShippedLfsPairIsGuardedToo:
    """The SHIPPED (Git LFS) branch of ``load_or_build`` must run the guard too.

    ``load_or_build`` has two accept doors — the processed cache and the packaged
    LFS assets — and each one calls ``_colors_match_fit`` separately. The cache
    door is covered above; without these two the LFS call could be replaced by
    ``if True:`` with the whole suite still green, because every other test points
    ``LFS_*`` at absent paths so that branch never runs. That is not academic
    here: today the shipped pair is the one that must be REFUSED (#1670).
    """

    @staticmethod
    def _lfs_setup(tmp_path, monkeypatch, *, permute: bool):
        """Materialize an LFS-shaped pair under tmp_path and stub the refit."""
        n = 6000
        rng = np.random.default_rng(43)
        rgb_vol = rng.uniform(0.0, 1.0, (16, 16, 16, 3)).astype(np.float32)
        fit = _scattered_gsplat_data(n, extent=15.49, seed=7)

        # Build the pair straight into the "shipped" location.
        lfs_dir = tmp_path / "lfs"
        lfs_dir.mkdir()
        monkeypatch.setattr(_demo, "CACHE_FIT", lfs_dir / _demo.FIT_FILE)
        monkeypatch.setattr(_demo, "CACHE_COLORS", lfs_dir / _demo.COLORS_FILE)
        _, colors = _demo.save_and_sample_colors(fit, rgb_vol)
        if permute:
            _save_colors_u8(colors[rng.permutation(n)], lfs_dir / _demo.COLORS_FILE)
            colors = _load_colors_f32(lfs_dir / _demo.COLORS_FILE)
        monkeypatch.setattr(_demo, "LFS_FIT", lfs_dir / _demo.FIT_FILE)
        monkeypatch.setattr(_demo, "LFS_COLORS", lfs_dir / _demo.COLORS_FILE)

        # …and empty the processed cache the LFS branch copies INTO, so the cache
        # door misses and the LFS door is the one under test. CACHE_DIR must move
        # too: the branch mkdirs it before `shutil.copy2`.
        cache_dir = tmp_path / "cache"
        monkeypatch.setattr(_demo, "CACHE_DIR", cache_dir)
        monkeypatch.setattr(_demo, "CACHE_FIT", cache_dir / _demo.FIT_FILE)
        monkeypatch.setattr(_demo, "CACHE_COLORS", cache_dir / _demo.COLORS_FILE)

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

    def test_a_permuted_shipped_sidecar_falls_through_to_the_refit(
        self, tmp_path, monkeypatch
    ) -> None:
        _, sentinel_fit, sentinel_colors = self._lfs_setup(
            tmp_path, monkeypatch, permute=True
        )
        got_fit, got_colors = _demo.load_or_build()
        assert got_fit is sentinel_fit, "a rejected SHIPPED pair was rendered anyway"
        assert got_colors is sentinel_colors

    def test_an_aligned_shipped_sidecar_is_used_instead_of_refitting(
        self, tmp_path, monkeypatch
    ) -> None:
        colors, sentinel_fit, _ = self._lfs_setup(tmp_path, monkeypatch, permute=False)
        got_fit, got_colors = _demo.load_or_build()
        assert got_fit is not sentinel_fit, "an aligned SHIPPED pair triggered a refit"
        np.testing.assert_array_equal(got_colors, colors)

    def test_the_shipped_pair_is_currently_rejected(self) -> None:
        """TRIPWIRE — the shipped artifact is misordered, and this pins that.

        ``vh_head_colors.npz`` was sampled in the pre-save splat order and does
        NOT correspond to the ``vh_head.gsplats.zarr.zip`` beside it (#1670), so
        the guard refuses the pair and the demo's DEFAULT path is a ~1.1 GB
        download plus a 4M-splat refit.

        WHEN THIS TEST GOES RED the artifact has been regenerated — that is good
        news, and it is the signal to REVERT every claim that documents today's
        slow path back to the fast-path wording:

          * ``DEMO_META["requirements"]["download_mb"]`` 1100 → 25
          * ``DEMO_META["requirements"]["compute"]`` "heavy" → "medium"
          * the "NO WORKING FAST PATH TODAY (#1670)" block in the demo's module
            docstring
          * the **Requires** paragraph in ``demos/README.md``
          * the ``gsplats_visible_human_head/`` row in ``demos/data/README.md``
          * the ``gsplats_3d_visible_human_head`` entry in
            ``scripts/gallery/generate_gallery_datasets.py``'s ``UNBUILDABLE_IDS``
            (and the ``unbuildable`` sentence in ``scripts/gallery/README.md``)

        Nothing else pins those, so without this tripwire they would quietly stay
        wrong forever. Swap this test for ``assert _colors_match_fit(...)`` then.
        """
        from luxar.demos import is_lfs_pointer

        for path in (_demo.LFS_FIT, _demo.LFS_COLORS):
            if not path.exists() or is_lfs_pointer(path):
                pytest.skip(f"{path.name} not materialized (run `git lfs pull`)")
        fit = GSplatData.load(_demo.LFS_FIT, include_stats=False)
        colors = _load_colors_f32(_demo.LFS_COLORS)
        assert not _demo._colors_match_fit(fit, colors, "shipped"), (
            "the shipped pair now PASSES the guard — see this test's docstring: "
            "revert download_mb, compute, the docstring, both READMEs and the "
            "gallery UNBUILDABLE_IDS entry to the fast-path wording"
        )
