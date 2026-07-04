"""Tests for :mod:`luxar.gsplats.lod.volume_refit` (the volume re-fit engine).

Engine-level coverage (the ``make_substitutive_lod(refine="volume")``
integration is tested in :mod:`test_substitutive`):

- Never-worse-than-seed: the returned splats always render at most the seed's
  MSE against the volume — including under an adversarial 1-iteration budget.
- On a fittable volume the warm-start re-fit genuinely improves (the benchmark
  behavior this feature productizes).
- Splat count and colors are preserved (identity-preserving fit: no cull, no
  dynamic ops), so the level slots back into the ladder unchanged in shape.
- Stats dict is flat, JSON-safe, and carries the documented keys.
"""

from __future__ import annotations

import numpy as np
import pytest

from luxar.gsplats.fit_gsplats import fit_gaussian_splats
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.lod.substitutive import make_substitutive_lod
from luxar.gsplats.lod.volume_refit import VolumeRefitConfig, volume_refine_splats

# ─────────────────────────────────────────────────────────────────────
# Fixtures — one tiny volume + fine fit + merge seed shared per module
# ─────────────────────────────────────────────────────────────────────


def _blob_volume(side: int = 20, n_blobs: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    grid = np.mgrid[0:side, 0:side, 0:side].astype(np.float32)
    vol = np.zeros((side,) * 3, dtype=np.float32)
    for _ in range(n_blobs):
        c = rng.uniform(0.2 * side, 0.8 * side, 3)
        sigma = rng.uniform(1.5, 2.5)
        r2 = sum((grid[d] - c[d]) ** 2 for d in range(3))
        vol += rng.uniform(0.4, 1.0) * np.exp(-r2 / (2 * sigma**2))
    return vol


def _mse(data: GSplatData, volume: np.ndarray) -> float:
    rendered = data.render_to_volume(shape=volume.shape, device="cpu")
    return float(np.mean((rendered.astype(np.float32) - volume) ** 2))


@pytest.fixture(scope="module")
def volume() -> np.ndarray:
    return _blob_volume()


@pytest.fixture(scope="module")
def merge_seed(volume: np.ndarray) -> GSplatData:
    """A coarse level's merge output: fine fit -> K=4 merge -> level 1."""
    fine = fit_gaussian_splats(
        volume, seeds=80, n_iters=200, device="cpu", verbose=False
    )
    ladder = make_substitutive_lod(fine, compression_factor=4, levels=1, device="cpu")
    return ladder.at_substitutive(1)


_FAST = VolumeRefitConfig(iters=60)


class TestNeverWorse:
    def test_result_mse_at_most_seed(self, merge_seed, volume) -> None:
        out, stats = volume_refine_splats(
            merge_seed, volume, config=_FAST, device="cpu"
        )
        assert _mse(out, volume) <= stats["mse_seed"] + 1e-12
        assert stats["mse_refit"] <= stats["mse_seed"] or stats["seed_won"]

    def test_adversarial_one_iter_still_never_worse(self, merge_seed, volume) -> None:
        out, stats = volume_refine_splats(
            merge_seed,
            volume,
            config=VolumeRefitConfig(iters=1),
            device="cpu",
        )
        assert _mse(out, volume) <= stats["mse_seed"] + 1e-12

    def test_seed_wins_on_mismatched_frame(self, merge_seed, volume) -> None:
        """A volume the seed does NOT describe (shifted frame): the fit either
        genuinely improves MSE or the untouched seed comes back — the returned
        splats are never a degradation."""
        wrong = np.roll(volume, shift=volume.shape[0] // 2, axis=0)
        out, stats = volume_refine_splats(merge_seed, wrong, config=_FAST, device="cpu")
        if stats["seed_won"]:
            np.testing.assert_array_equal(out.centers, merge_seed.centers)
        else:
            assert stats["mse_refit"] < stats["mse_seed"]


class TestQuality:
    def test_improves_on_fittable_volume(self, merge_seed, volume) -> None:
        out, stats = volume_refine_splats(
            merge_seed, volume, config=_FAST, device="cpu"
        )
        assert stats["improved"] is True
        assert stats["mse_refit"] < stats["mse_seed"]
        # And by a real margin — the benchmark regime, not a numerical wiggle.
        assert stats["mse_refit"] < 0.9 * stats["mse_seed"]


class TestIdentityPreservation:
    def test_count_preserved(self, merge_seed, volume) -> None:
        out, stats = volume_refine_splats(
            merge_seed, volume, config=_FAST, device="cpu"
        )
        assert out.n_splats == merge_seed.n_splats
        assert stats["n_refit"] == stats["n_seed"]

    def test_colors_carried_over(self, volume, merge_seed) -> None:
        colored = GSplatData(
            centers=merge_seed.centers,
            amplitudes=merge_seed.amplitudes,
            cholesky_factors=merge_seed.cholesky_factors,
            colors=np.full((merge_seed.n_splats, 3), 200, dtype=np.uint8),
        )
        out, _ = volume_refine_splats(colored, volume, config=_FAST, device="cpu")
        assert out.colors is not None
        np.testing.assert_array_equal(out.colors, colored.colors)


class TestStatsAndConfig:
    def test_stats_keys_json_safe(self, merge_seed, volume) -> None:
        import json

        _, stats = volume_refine_splats(merge_seed, volume, config=_FAST, device="cpu")
        for key in (
            "mse_seed",
            "mse_refit",
            "improved",
            "seed_won",
            "n_seed",
            "n_refit",
            "iters",
            "wall_s",
        ):
            assert key in stats, key
        json.dumps(stats)  # flat + JSON-safe (round-trips into pipeline/)

    def test_never_worse_false_returns_raw_refit(self, merge_seed, volume) -> None:
        out, stats = volume_refine_splats(
            merge_seed,
            volume,
            config=VolumeRefitConfig(iters=5, never_worse=False),
            device="cpu",
        )
        assert "mse_seed" not in stats  # guard renders skipped entirely
        assert out.n_splats == merge_seed.n_splats

    def test_config_defaults(self) -> None:
        cfg = VolumeRefitConfig()
        assert cfg.iters == 300
        assert cfg.never_worse is True
        assert cfg.conserve_mass is True


class TestSeedWinsBranch:
    """The never-worse guard's else-branch (return the seed when the re-fit
    renders WORSE) — the feature's headline safety promise. Forced with a
    monkeypatched fit that returns a deliberately worse candidate, so a
    mutation dropping the comparison (`result = refit`) is caught."""

    def test_seed_returned_when_refit_is_worse(
        self, merge_seed, volume, monkeypatch
    ) -> None:
        import luxar.gsplats.fit_gsplats as fg

        # A worse-but-NOT-relocated candidate: identical centers (so the
        # relocation frame check passes — it inspects center moments only)
        # with hugely inflated covariances -> a blurred render whose MSE is
        # worse than the seed's. Mass-pinning restores its DC, so the loss is
        # purely the blur; only the never-worse comparison can reject it.
        def _bad_fit(vol, seeds=None, **kwargs):
            return GSplatData(
                centers=np.asarray(seeds.centers).copy(),
                amplitudes=np.asarray(seeds.amplitudes).copy(),
                cholesky_factors=(np.asarray(seeds.cholesky_factors) * 8.0).astype(
                    np.float32
                ),
            )

        monkeypatch.setattr(fg, "fit_gaussian_splats", _bad_fit)
        out, stats = volume_refine_splats(
            merge_seed, volume, config=_FAST, device="cpu"
        )
        assert stats["seed_won"] is True
        assert stats["improved"] is False
        # The UNTOUCHED seed is returned, not the worse candidate — assert on
        # the field the bad fit corrupted (centers are identical by design, so
        # they cannot discriminate seed from refit here).
        np.testing.assert_array_equal(out.cholesky_factors, merge_seed.cholesky_factors)


class TestFrameGuard:
    def test_out_of_frame_seed_skips_refit(self, merge_seed, volume) -> None:
        """A seed whose centers sit far outside the volume's voxel range (a
        different coordinate frame) is returned untouched with a warning —
        never silently re-fit into the wrong place."""
        shifted = GSplatData(
            centers=(np.asarray(merge_seed.centers) + 10_000.0).astype(np.float32),
            amplitudes=merge_seed.amplitudes,
            cholesky_factors=merge_seed.cholesky_factors,
        )
        with pytest.warns(RuntimeWarning, match="coordinate frame"):
            out, stats = volume_refine_splats(
                shifted, volume, config=_FAST, device="cpu"
            )
        assert stats["frame_mismatch"] is True
        np.testing.assert_array_equal(out.centers, shifted.centers)

    def test_shrunk_frame_relocated_refit_rejected(
        self, merge_seed, volume, monkeypatch
    ) -> None:
        """A SHRUNK physical frame (e.g. voxel_size 0.25) sits INSIDE the bbox
        check; the fit then relocates the splats wholesale to the voxel-frame
        signal, and the relocated result would WIN the MSE guard while being
        misplaced relative to the ladder. The post-fit relocation check must
        reject it and keep the seed. (Deterministic: the fit is monkeypatched
        to return the relocated result the real fit converges to — the real
        relocation magnitude varies with iteration budget.)"""
        import luxar.gsplats.fit_gsplats as fg

        shrunk = GSplatData(
            centers=(np.asarray(merge_seed.centers) * 0.25).astype(np.float32),
            amplitudes=merge_seed.amplitudes,
            cholesky_factors=(np.asarray(merge_seed.cholesky_factors) * 0.25).astype(
                np.float32
            ),
        )

        def _relocating_fit(vol, seeds=None, **kwargs):
            # What the real fit converges to: splats dragged back to the
            # voxel-frame signal (4x the shrunk coordinates).
            return GSplatData(
                centers=(np.asarray(seeds.centers) * 4.0).astype(np.float32),
                amplitudes=np.asarray(seeds.amplitudes).copy(),
                cholesky_factors=(np.asarray(seeds.cholesky_factors) * 4.0).astype(
                    np.float32
                ),
            )

        monkeypatch.setattr(fg, "fit_gaussian_splats", _relocating_fit)
        with pytest.warns(RuntimeWarning, match="relocat"):
            out, stats = volume_refine_splats(
                shrunk, volume, config=VolumeRefitConfig(iters=10), device="cpu"
            )
        assert stats["frame_mismatch"] is True
        np.testing.assert_array_equal(out.centers, shrunk.centers)

    def test_relocated_displacement_unit(self) -> None:
        """The relocation detector: per-splat displacement (rows are 1:1 in
        the identity-preserving fit). Flags every frame-mismatch shape —
        shift, rescale, ROTATION, MIRROR (which aggregate CoM/RMS moments are
        provably blind to) — while passing legit warm-start drift and
        AMPLITUDE-ONLY changes (which amplitude-weighted moments falsely
        flagged). All deterministic."""
        from luxar.gsplats.lod.volume_refit import _relocated

        rng = np.random.default_rng(0)
        vol = np.zeros((64, 64, 64), np.float32)
        centers = rng.uniform(10, 50, (50, 3)).astype(np.float32)
        com = centers.mean(axis=0)

        def gs(c, amps=None):
            return GSplatData(
                centers=c.astype(np.float32),
                amplitudes=(amps if amps is not None else np.ones(len(c))).astype(
                    np.float32
                ),
                cholesky_factors=np.tile(
                    np.array([1, 0, 1, 0, 0, 1], np.float32), (len(c), 1)
                ),
            )

        seed = gs(centers)
        # Small drift (typical warm-start motion): NOT a relocation.
        assert not _relocated(seed, gs(centers + 0.5), vol)
        # Amplitude-only change (the fit zeroing a spurious splat, centers
        # untouched): NOT a relocation — the pre-fix amplitude-weighted RMS
        # falsely flagged exactly this.
        amps = np.ones(len(centers))
        amps[1:] = 1e-6
        assert not _relocated(seed, gs(centers, amps), vol)
        # Wholesale shift by more than the seed's own spread: relocation.
        assert _relocated(seed, gs(centers + 40.0), vol)
        # 4x spread rescale (frame scale change): relocation.
        assert _relocated(seed, gs((centers - com) * 4.0 + com), vol)
        # 90-degree rotation about the CoM (CoM and RMS invariant — the
        # moment-based detector was blind to this): relocation.
        rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], np.float32)
        assert _relocated(seed, gs((centers - com) @ rot.T + com), vol)
        # Point mirror about the CoM: relocation.
        assert _relocated(seed, gs(com - (centers - com)), vol)
        # A splat correcting within its OWN footprint: a TIGHT cluster (rms
        # ~2, so 0.5*rms and the 5%-diag floor are both below the motion) of
        # BIG sigma=8 splats moving 6 voxels — legit only thanks to the
        # median-splat-sigma slack term. Kills the drop-sigma-slack mutant.
        tight = rng.uniform(30, 34, (50, 3)).astype(np.float32)
        big = GSplatData(
            centers=tight,
            amplitudes=np.ones(len(tight), np.float32),
            cholesky_factors=np.tile(
                np.array([8, 0, 8, 0, 0, 8], np.float32), (len(tight), 1)
            ),
        )
        moved = GSplatData(
            centers=(tight + np.array([6.0, 0.0, 0.0], np.float32)).astype(np.float32),
            amplitudes=np.ones(len(tight), np.float32),
            cholesky_factors=big.cholesky_factors,
        )
        assert not _relocated(big, moved, vol)


class TestStatsCoherence:
    def test_seed_win_resets_mass_pin_stats(
        self, merge_seed, volume, monkeypatch
    ) -> None:
        """When the seed wins, the STORED artifact is the untouched seed —
        stats must not claim mass_pinned/mass_scale of the discarded candidate
        (they describe what was kept). Fails pre-fix (mass_pinned stayed True)."""
        import luxar.gsplats.fit_gsplats as fg

        def _bad_fit(vol, seeds=None, **kwargs):
            return GSplatData(
                centers=np.asarray(seeds.centers).copy(),
                amplitudes=np.asarray(seeds.amplitudes).copy(),
                cholesky_factors=(np.asarray(seeds.cholesky_factors) * 8.0).astype(
                    np.float32
                ),
            )

        monkeypatch.setattr(fg, "fit_gaussian_splats", _bad_fit)
        _, stats = volume_refine_splats(merge_seed, volume, config=_FAST, device="cpu")
        assert stats["seed_won"] is True
        assert stats["mass_pinned"] is False
        assert stats["mass_scale"] == 1.0


class TestMassPinning:
    def test_refit_dc_pinned_to_seed_when_conserve_mass(
        self, merge_seed, volume
    ) -> None:
        """conserve_mass rescales the re-fit's rendered DC to the seed's, so a
        level swap does not brighten (the pop the ladder-wide conserve_mass
        exists to prevent). Without pinning the re-fit tracks the volume's DC."""
        pinned, st_pin = volume_refine_splats(
            merge_seed,
            volume,
            config=VolumeRefitConfig(iters=60, conserve_mass=True),
            device="cpu",
        )
        raw, st_raw = volume_refine_splats(
            merge_seed,
            volume,
            config=VolumeRefitConfig(iters=60, conserve_mass=False),
            device="cpu",
        )

        def dc(d):
            return float(
                np.asarray(d.render_to_volume(shape=volume.shape, device="cpu")).sum()
            )

        dc_seed = dc(merge_seed)
        assert st_pin["mass_pinned"] is True
        # Pinned DC matches the seed to within a few percent (render truncation).
        assert abs(dc(pinned) - dc_seed) / dc_seed < 0.03
        # The raw re-fit tracks the volume's DC instead -> materially different.
        assert st_raw["mass_pinned"] is False
        assert abs(dc(raw) - dc_seed) / dc_seed > 0.05
