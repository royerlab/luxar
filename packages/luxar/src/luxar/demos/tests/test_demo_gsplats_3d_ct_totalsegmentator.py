"""Smoke tests for demo_gsplats_3d_ct_totalsegmentator.

Covers the deterministic array helpers and the scene builder's authored
blending — no network, no nibabel IO, no GPU fit. The demo is loaded by file
path (see test_demo_ppi_flow_field).
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
    Path(__file__).resolve().parents[1] / "demo_gsplats_3d_ct_totalsegmentator.py"
)


def _load_demo_module():
    name = "_luxar_demo_ct_totalseg_for_tests"
    spec = importlib.util.spec_from_file_location(name, _DEMO_PATH)
    if spec is None or spec.loader is None:
        pytest.skip(f"Could not locate demo at {_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_demo = _load_demo_module()
tissue_group = _demo.tissue_group
organ_palette = _demo.organ_palette
window_ct = _demo.window_ct
sample_labels = _demo.sample_labels
label_colors = _demo.label_colors
splat_layer_indices = _demo.splat_layer_indices
organ_label_text = _demo.organ_label_text
crop_to_content = _demo.crop_to_content
_save_labels_u8 = _demo._save_labels_u8
_load_labels = _demo._load_labels
CLASS_MAP = _demo.CLASS_MAP
SUPERGROUPS = _demo.SUPERGROUPS
create_luxar_scene = _demo.create_luxar_scene


def _tiny_gsplat_data(n: int, seed: int = 0) -> GSplatData:
    """A handful of valid splats — no GPU fit, enough to build the scene."""
    rng = np.random.default_rng(seed)
    return GSplatData(
        centers=rng.uniform(-4.0, 4.0, (n, 3)).astype(np.float32),
        amplitudes=rng.uniform(0.2, 1.0, n).astype(np.float32),
        cholesky_factors=np.tile([1, 0, 1, 0, 0, 1], (n, 1)).astype(np.float32),
    )


def _gsplat_layers(scene_path: Path) -> list:
    """Attr dicts of the scene's top-level gsplats layer nodes."""
    root = zarr.open_group(str(scene_path), mode="r")
    return [
        dict(group.attrs)
        for _name, group in root.groups()
        if dict(group.attrs).get("type") == "gsplats"
    ]


class TestSceneBlending:
    def test_scene_bakes_additive_blending_on_every_layer(self, tmp_path) -> None:
        # The per-tissue toggle layers all overlap the same body. The viewer
        # assigns one global order slot per node, which cannot interleave those
        # volumes; additive is order-independent. Pin it so a revert to the old
        # volumetric look is caught (the helper smoke tests never build the
        # scene). Seed one label per supergroup so the scene really carries
        # every layer — a layer that is never built could not be checked.
        all_ids = np.array(sorted(CLASS_MAP), dtype=np.int32)
        super_idx = splat_layer_indices(all_ids)
        labels = []
        for i, (layer_name, *_rest) in enumerate(SUPERGROUPS):
            members = all_ids[super_idx == i]
            assert members.size, f"no CLASS_MAP label lands in layer {layer_name!r}"
            labels.append(int(members[0]))
        fit = _tiny_gsplat_data(len(labels))
        out = create_luxar_scene(
            fit, np.array(labels, dtype=np.int32), tmp_path / "ct.luxar.zarr"
        )
        layers = _gsplat_layers(out)
        assert len(layers) == len(SUPERGROUPS)
        for attrs in layers:
            assert attrs.get("blending_mode") == "additive"


class TestTissueGroup:
    def test_known_structures(self) -> None:
        assert tissue_group("liver") == "abdominal_organ"
        assert tissue_group("aorta") == "vessel"
        assert tissue_group("iliac_vena_left") == "vessel"
        assert tissue_group("lung_upper_lobe_left") == "lung"
        assert tissue_group("vertebrae_L3") == "bone"
        assert tissue_group("rib_right_5") == "bone"
        assert tissue_group("heart") == "heart"
        assert tissue_group("gluteus_maximus_left") == "muscle"
        assert tissue_group("brain") == "brain"
        assert tissue_group("spinal_cord") == "spinal"
        assert tissue_group("colon") == "gi"
        assert tissue_group("kidney_left") == "urinary"

    def test_every_class_maps_to_a_known_group(self) -> None:
        groups = {tissue_group(n) for n in CLASS_MAP.values()}
        assert groups <= set(_demo.GROUP_COLORS.keys())


class TestOrganPalette:
    def test_shape_and_gamut(self) -> None:
        pal = organ_palette()
        assert pal.shape == (118, 3)
        assert pal.dtype == np.float32
        assert pal.min() >= 0.0 and pal.max() <= 1.0
        # background distinct from any organ
        assert not any(np.allclose(pal[0], pal[i]) for i in range(1, 118))

    def test_deterministic(self) -> None:
        np.testing.assert_array_equal(organ_palette(), organ_palette())


class TestWindowCt:
    def test_clip_and_normalize(self) -> None:
        ct = np.array(
            [-1000.0, _demo.HU_LO, 0.0, _demo.HU_HI, 3000.0], dtype=np.float32
        )
        out = window_ct(ct)
        assert out.min() == 0.0 and out.max() == 1.0
        assert out[0] == 0.0 and out[1] == 0.0  # below/at lo → 0
        assert out[3] == 1.0 and out[4] == 1.0  # at/above hi → 1
        assert 0.0 < out[2] < 1.0


class TestSampleLabelsAndColors:
    def test_nearest_label_and_clamp(self) -> None:
        vol = np.zeros((2, 2, 2), dtype=np.int32)
        vol[0, 0, 0] = 5  # liver
        vol[1, 1, 1] = 52  # aorta
        centers = np.array(
            [[0.0, 0.0, 0.0], [1.4, 1.4, 1.4], [99.0, 99.0, 99.0]], dtype=np.float32
        )
        ids = sample_labels(vol, centers)
        assert ids.tolist() == [5, 52, 52]  # 3rd clamps into (1,1,1)

    def test_label_colors_via_palette(self) -> None:
        pal = organ_palette()
        cols = label_colors(np.array([0, 5, 52], dtype=np.int32), pal)
        assert cols.shape == (3, 3) and cols.dtype == np.float32
        np.testing.assert_allclose(cols[1], pal[5])
        np.testing.assert_allclose(cols[2], pal[52])


class TestCropToContent:
    def test_bbox(self) -> None:
        m = np.zeros((10, 10, 10), dtype=bool)
        m[3:6, 4:5, 2:8] = True
        z0, z1, y0, y1, x0, x1 = crop_to_content(m, pad=0)
        assert (z0, z1, y0, y1, x0, x1) == (3, 6, 4, 5, 2, 8)

    def test_empty(self) -> None:
        m = np.zeros((4, 4, 4), dtype=bool)
        assert crop_to_content(m) == (0, 4, 0, 4, 0, 4)


class TestLayerSplit:
    def test_every_label_maps_to_a_layer(self) -> None:
        # every one of the 117 structures lands in exactly one supergroup
        idx = splat_layer_indices(np.arange(1, 118, dtype=np.int32))
        assert idx.min() >= 0 and idx.max() < len(SUPERGROUPS)
        # background label 0 → -1 (no layer)
        assert splat_layer_indices(np.array([0]))[0] == -1

    def test_known_label_layers(self) -> None:
        names = [g[0] for g in SUPERGROUPS]
        skeleton = names.index("Skeleton")
        organs = names.index("Organs")
        muscles = names.index("Muscles")
        # 5=liver→Organs, 27=vertebrae_L5→Skeleton, 80=gluteus→Muscles
        got = splat_layer_indices(np.array([5, 27, 80], dtype=np.int32))
        assert got.tolist() == [organs, skeleton, muscles]

    def test_muscle_layer_semi_transparent_and_boosted(self) -> None:
        muscle = next(g for g in SUPERGROUPS if g[0] == "Muscles")
        assert 0.0 < muscle[2] < 1.0  # semi-transparent so organs read through
        assert muscle[3] >= 2.0  # amplitude boost so low-HU muscle is visible

    def test_layer_tuple_shape(self) -> None:
        # (name, tissue groups, opacity, amplitude boost)
        for name, groups, opacity, boost in SUPERGROUPS:
            assert isinstance(name, str) and len(groups) >= 1
            assert 0.0 < opacity <= 1.0 and boost >= 1.0


class TestOrganLabelText:
    def test_pretty_names(self) -> None:
        assert organ_label_text(5) == "Liver"
        assert organ_label_text(2) == "Kidney right"
        assert organ_label_text(52) == "Aorta"
        assert organ_label_text(0) == ""


class TestLabelRoundtrip:
    def test_uint8_roundtrip(self, tmp_path) -> None:
        labels = np.array([0, 5, 52, 117, 80], dtype=np.int32)
        p = tmp_path / "labels.npz"
        _save_labels_u8(labels, p)
        loaded = _load_labels(p)
        assert loaded.dtype == np.int32
        np.testing.assert_array_equal(loaded, labels)


def _legacy_pair(label_vol: np.ndarray, fit: GSplatData):
    """Write the LEGACY shape: a fit with NO label channel plus a sidecar.

    This is what the in-repo Git-LFS payload is, and it is now the only shape on
    which the ordering guard can fire at all — the hosted archive carries its
    organ ids natively, so it has no second array to misorder. Tests of the guard
    therefore build this shape explicitly; ``save_and_sample_labels`` no longer
    produces it. Retires with the payloads (#2354).

    Written through the demo's own ``_save_atlas_fit`` / ``_save_labels_u8`` so
    the recipe and the quantization rule cannot drift from the real ones.
    """
    _demo._save_atlas_fit(fit)
    stored = GSplatData.load(_demo.LOCAL_FIT)
    assert _demo._native_labels(stored) is None, "the legacy shape must carry no ids"
    labels = _demo.sample_labels(label_vol, stored.centers)
    _save_labels_u8(labels, _demo.LOCAL_LABELS)
    return stored, _load_labels(_demo.LOCAL_LABELS)


def _scattered_gsplat_data(n: int, extent: float, seed: int = 0) -> GSplatData:
    """Splats scattered over a small voxel grid — no GPU fit needed."""
    rng = np.random.default_rng(seed)
    return GSplatData(
        centers=rng.uniform(0.0, extent, (n, 3)).astype(np.float32),
        amplitudes=rng.uniform(0.2, 1.0, n).astype(np.float32),
        cholesky_factors=np.tile([1, 0, 1, 0, 0, 1], (n, 1)).astype(np.float32),
        stats={"psnr_db": 42.5, "source_shape": [16, 16, 16]},
    )


class TestLabelSidecarOrdering:
    """The labels sidecar must be sampled in the SAVED store's splat order.

    ``GSplatData.save`` reorders splats spatially, so labels sampled at the
    in-memory fit's centers describe different splats than the ones ``load``
    hands back — every color, layer and tooltip would be wrong. See issue #1670.
    """

    def test_labels_align_with_the_returned_fit_not_the_input(
        self, tmp_path, monkeypatch
    ) -> None:
        n = 6000
        rng = np.random.default_rng(7)
        label_vol = rng.integers(0, 118, (16, 16, 16)).astype(np.int32)
        fit = _scattered_gsplat_data(n, extent=15.49)
        monkeypatch.setattr(_demo, "LOCAL_FIT", tmp_path / _demo.FIT_FILE)
        monkeypatch.setattr(_demo, "LOCAL_LABELS", tmp_path / _demo.LABELS_FILE)

        stored, labels = _legacy_pair(label_vol, fit)

        # Non-vacuity: the writer really did permute this input. Without it the
        # pre/post-save samplings would agree and prove nothing.
        moved = np.any(np.rint(stored.centers) != np.rint(fit.centers), axis=1)
        assert int(moved.sum()) > n // 2, "hilbert ordering did not permute the input"

        # Aligned with the fit that is RETURNED (== what the cache reloads)...
        np.testing.assert_array_equal(labels, sample_labels(label_vol, stored.centers))
        # ...and NOT with the pre-save sampling the old code persisted.
        assert not np.array_equal(labels, sample_labels(label_vol, fit.centers))
        # The sidecar on disk carries those same rows.
        np.testing.assert_array_equal(_load_labels(_demo.LOCAL_LABELS), labels)

    def test_guard_accepts_the_aligned_pair_and_rejects_a_permuted_one(
        self, tmp_path, monkeypatch
    ) -> None:
        n = 6000
        rng = np.random.default_rng(8)
        label_vol = rng.integers(0, 118, (16, 16, 16)).astype(np.int32)
        fit = _scattered_gsplat_data(n, extent=15.49, seed=1)
        monkeypatch.setattr(_demo, "LOCAL_FIT", tmp_path / _demo.FIT_FILE)
        monkeypatch.setattr(_demo, "LOCAL_LABELS", tmp_path / _demo.LABELS_FILE)

        stored, labels = _legacy_pair(label_vol, fit)
        assert _demo._labels_match_fit(stored, labels, "aligned")
        # A permuted sidecar (the #1670 bug) and a wrong-length one are refused.
        permuted = labels[rng.permutation(n)]
        assert not _demo._labels_match_fit(stored, permuted, "permuted")
        assert not _demo._labels_match_fit(stored, labels[:-1], "truncated")

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
        labels = np.random.default_rng(9).integers(0, 118, len(grid)).astype(np.int32)
        assert voxel_sampled_payload_agreement(fit.centers, labels) is None, (
            "fixture must be unverifiable for this branch to be exercised"
        )
        assert _demo._labels_match_fit(fit, labels, "unverifiable")
        assert "UNVERIFIED" in capsys.readouterr().out

    def test_shipped_pair_is_accepted(self) -> None:
        """The pair actually in Git LFS must pass the guard it is checked by.

        A guard nobody can satisfy is a guard that always refits. Only the ACCEPT
        verdict is asserted: the numbers themselves (agreement, splat count) are
        properties of the artifact and must be free to change when it is
        regenerated.
        """
        from luxar.demos import is_lfs_pointer

        for path in (_demo.LFS_FIT, _demo.LFS_LABELS):
            if not path.exists() or is_lfs_pointer(path):
                pytest.skip(f"{path.name} not materialized (run `git lfs pull`)")
        fit = GSplatData.load(_demo.LFS_FIT, include_stats=False)
        labels = _load_labels(_demo.LFS_LABELS)
        assert _demo._labels_match_fit(fit, labels, "shipped")


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


class TestLabelAgreementThreshold:
    """``MIN_LABEL_AGREEMENT`` must be the value the guard actually decides on.

    Without this, the constant survived being set to 0.5 with every test green.
    """

    @staticmethod
    def _pair_with_broken(
        tmp_path, monkeypatch, n_broken: int
    ) -> tuple[GSplatData, np.ndarray]:
        n = 6000
        rng = np.random.default_rng(31)
        label_vol = rng.integers(0, 118, (16, 16, 16)).astype(np.int32)
        fit = _scattered_gsplat_data(n, extent=15.49, seed=3)
        monkeypatch.setattr(_demo, "LOCAL_FIT", tmp_path / _demo.FIT_FILE)
        monkeypatch.setattr(_demo, "LOCAL_LABELS", tmp_path / _demo.LABELS_FILE)
        stored, labels = _legacy_pair(label_vol, fit)

        order, collides = _same_voxel_pairs(stored.centers)
        lone = _lone_pairs(collides)
        assert len(lone) >= n_broken, "not enough two-splat voxels to break"
        broken = labels.copy()
        for c in lone[:n_broken]:
            broken[order[c + 1]] = (int(labels[order[c]]) + 1) % 118
        return stored, broken

    @staticmethod
    def _tolerated_breaks(tmp_path, monkeypatch) -> int:
        """How many broken pairs the constant still tolerates, for this fixture.

        Derived from the SAVED fit's own pair count (the writer permutes, so the
        pre-save centers are the wrong thing to count), hence the throwaway save.
        """
        monkeypatch.setattr(_demo, "LOCAL_FIT", tmp_path / ("probe-" + _demo.FIT_FILE))
        monkeypatch.setattr(
            _demo, "LOCAL_LABELS", tmp_path / ("probe-" + _demo.LABELS_FILE)
        )
        stored, _ = _legacy_pair(
            np.zeros((16, 16, 16), dtype=np.int32),
            _scattered_gsplat_data(6000, extent=15.49, seed=3),
        )
        _, collides = _same_voxel_pairs(stored.centers)
        return int(np.floor((1.0 - _demo.MIN_LABEL_AGREEMENT) * len(collides)))

    def test_just_above_the_threshold_is_accepted(self, tmp_path, monkeypatch) -> None:
        n_broken = self._tolerated_breaks(tmp_path, monkeypatch)
        stored, broken = self._pair_with_broken(tmp_path, monkeypatch, n_broken)
        measured = voxel_sampled_payload_agreement(stored.centers, broken)
        assert measured is not None and measured >= _demo.MIN_LABEL_AGREEMENT
        assert _demo._labels_match_fit(stored, broken, "just above")

    def test_just_below_the_threshold_is_rejected(self, tmp_path, monkeypatch) -> None:
        n_broken = self._tolerated_breaks(tmp_path, monkeypatch) + 1
        stored, broken = self._pair_with_broken(tmp_path, monkeypatch, n_broken)
        measured = voxel_sampled_payload_agreement(stored.centers, broken)
        assert measured is not None and measured < _demo.MIN_LABEL_AGREEMENT
        assert not _demo._labels_match_fit(stored, broken, "just below")

    def test_a_near_miss_reordering_is_rejected(self, tmp_path, monkeypatch) -> None:
        """An ABSOLUTE pin, independent of what the constant currently says.

        A near-miss reordering — a genuinely misindexed sidecar that happens to
        keep most same-voxel pairs together — lands JUST under the gate. Measured
        on the shipped CT pair (660,934 splats) by permuting its aligned sidecar
        the way each mistake would have written it: the writer's own morton order
        instead of hilbert scores 0.901, a roll-by-one 0.955, an adjacent-pair
        swap 0.962. So the pin is set at ~0.972, just ABOVE the worst of them:
        any constant at or below 0.972 accepts this pair and turns the test red,
        which is what stops the gate being lowered under a real near miss (0.90
        — the level pinned before — was under three of them).
        """
        monkeypatch.setattr(_demo, "LOCAL_FIT", tmp_path / ("nm-" + _demo.FIT_FILE))
        monkeypatch.setattr(
            _demo, "LOCAL_LABELS", tmp_path / ("nm-" + _demo.LABELS_FILE)
        )
        stored, _ = _legacy_pair(
            np.zeros((16, 16, 16), dtype=np.int32),
            _scattered_gsplat_data(6000, extent=15.49, seed=3),
        )
        _, collides = _same_voxel_pairs(stored.centers)
        stored, broken = self._pair_with_broken(
            tmp_path, monkeypatch, int(np.floor(0.028 * len(collides)))
        )
        measured = voxel_sampled_payload_agreement(stored.centers, broken)
        # Strictly above 0.97, so `MIN_LABEL_AGREEMENT = 0.97` ACCEPTS this and
        # goes red rather than passing on a rounding coincidence.
        assert measured is not None and 0.970 < measured < 0.975
        assert not _demo._labels_match_fit(stored, broken, "near miss")


class TestRawCentersArePrecondition:
    """The guard MUST run on the centers straight off ``load``.

    ``voxel_sampled_payload_agreement`` reads the payload's nearest-voxel sampling
    invariant off the voxel each center rounds to, so ANY coordinate change
    invalidates it. ``create_luxar_scene`` re-centres the fit
    (``center_at_centroid``) before rendering — checking THAT fit would reject a
    perfectly aligned pair. Measured on the real shipped CT pair the recentred
    agreement is 0.9848, already under the 0.99 gate; on this synthetic pair it
    collapses much further.
    """

    def test_centring_the_fit_destroys_the_verdict(self) -> None:
        n = 6000
        rng = np.random.default_rng(32)
        label_vol = rng.integers(0, 118, (16, 16, 16)).astype(np.int32)
        fit = _scattered_gsplat_data(n, extent=15.49, seed=4)
        labels = sample_labels(label_vol, fit.centers)

        raw = voxel_sampled_payload_agreement(fit.centers, labels)
        assert raw == 1.0

        recentred = voxel_sampled_payload_agreement(
            fit.center_at_centroid().centers, labels
        )
        assert recentred is not None
        assert recentred < _demo.MIN_LABEL_AGREEMENT

        # ...so the guard must be handed the RAW fit. Checking a transformed one
        # would refuse a perfectly good pair.
        assert _demo._labels_match_fit(fit, labels, "raw")
        assert not _demo._labels_match_fit(
            fit.center_at_centroid(), labels, "recentred"
        )


class TestNativeLabelsWinOverTheSidecar:
    """A fit carrying its own organ ids must never consult a sidecar.

    This retires the #1670/#2334 failure mode instead of guarding against it. It
    matters more here than for vh_head: ct_atlas's misindexing is the DANGEROUS
    kind, because organs are large and contiguous, so a reordered attach still
    lands ~37% of splats in the right organ and renders plausible anatomy with the
    rest silently mislabelled. A native channel cannot be reordered relative to
    the splats it describes.
    """

    @staticmethod
    def _fixture(tmp_path, monkeypatch, tag: str):
        rng = np.random.default_rng(91)
        label_vol = rng.integers(0, 118, (16, 16, 16)).astype(np.int32)
        fit = _scattered_gsplat_data(4000, extent=15.49, seed=13)
        monkeypatch.setattr(_demo, "LOCAL_FIT", tmp_path / (tag + _demo.FIT_FILE))
        monkeypatch.setattr(_demo, "LOCAL_LABELS", tmp_path / (tag + _demo.LABELS_FILE))
        return label_vol, fit

    def test_a_permuted_sidecar_beside_a_native_fit_is_ignored(
        self, tmp_path, monkeypatch
    ) -> None:
        label_vol, fit = self._fixture(tmp_path, monkeypatch, "native-")
        stored, labels = _demo.save_and_sample_labels(fit, label_vol)
        rng = np.random.default_rng(5)
        _save_labels_u8(labels[rng.permutation(len(labels))], _demo.LOCAL_LABELS)

        got = _demo._labels_for(stored, _demo.LOCAL_FIT, _demo.LOCAL_LABELS)

        np.testing.assert_array_equal(got, labels)
        # Non-vacuity: the sidecar really did hold a different ordering.
        assert not np.array_equal(_load_labels(_demo.LOCAL_LABELS), labels)

    def test_the_ids_survive_the_save_exactly_not_merely_in_range(
        self, tmp_path, monkeypatch
    ) -> None:
        """A renumbering passes a range check, so check the classes themselves.

        #2334's hard requirement: the channel is CATEGORICAL, so an
        amplitude-style encoder that returns class 51 as 50 or 52 would attribute
        splats to the wrong anatomy with no error anywhere. Set equality alone is
        also not enough — a swap of two classes preserves the set — so the
        per-class POPULATIONS are compared too.
        """
        label_vol, fit = self._fixture(tmp_path, monkeypatch, "exact-")
        stored, labels = _demo.save_and_sample_labels(fit, label_vol)

        assert stored.label_ids is not None
        ids = np.asarray(stored.label_ids).astype(np.int32)
        expected = _demo.sample_labels(label_vol, stored.centers)
        np.testing.assert_array_equal(labels, expected)
        np.testing.assert_array_equal(ids, expected)
        assert set(np.unique(ids).tolist()) == set(np.unique(expected).tolist())
        for cls in np.unique(expected):
            assert int((ids == cls).sum()) == int((expected == cls).sum()), (
                f"class {int(cls)} changed population through the save"
            )
        cached = GSplatData.load(_demo.LOCAL_FIT, include_stats=True)
        assert cached.stats["psnr_db"] == 42.5
        assert cached.stats["source_shape"] == [16, 16, 16]
        # The vocabulary has to name every id present, background included.
        vocab = {int(k): str(v) for k, v in (stored.label_vocabulary or {}).items()}
        assert set(np.unique(ids).tolist()) <= set(vocab), "an id has no name"
        assert vocab.get(0) == "background"

    def test_a_legacy_fit_still_falls_back_to_its_sidecar(
        self, tmp_path, monkeypatch
    ) -> None:
        """The in-repo Git-LFS path, which must keep working until #2354."""
        label_vol, fit = self._fixture(tmp_path, monkeypatch, "legacy-")
        stored, labels = _legacy_pair(label_vol, fit)

        got = _demo._labels_for(stored, _demo.LOCAL_FIT, _demo.LOCAL_LABELS)

        np.testing.assert_array_equal(got, labels)

    def test_a_legacy_fit_with_a_misordered_sidecar_is_rejected(
        self, tmp_path, monkeypatch
    ) -> None:
        """The guard must still fire on the shape where it can."""
        label_vol, fit = self._fixture(tmp_path, monkeypatch, "broken-")
        stored, labels = _legacy_pair(label_vol, fit)
        rng = np.random.default_rng(6)
        _save_labels_u8(labels[rng.permutation(len(labels))], _demo.LOCAL_LABELS)

        assert _demo._labels_for(stored, _demo.LOCAL_FIT, _demo.LOCAL_LABELS) is None

    def test_a_legacy_fit_with_no_sidecar_yields_nothing(
        self, tmp_path, monkeypatch
    ) -> None:
        """Never render unlabelled anatomy — the callers must refit instead."""
        label_vol, fit = self._fixture(tmp_path, monkeypatch, "none-")
        stored, _ = _legacy_pair(label_vol, fit)

        assert _demo._labels_for(stored, _demo.LOCAL_FIT, None) is None

    def test_a_cache_that_drops_native_labels_is_removed(
        self, tmp_path, monkeypatch
    ) -> None:
        label_vol, fit = self._fixture(tmp_path, monkeypatch, "dropped-")
        monkeypatch.setattr(_demo, "_native_labels", lambda _fit: None)

        with pytest.raises(RuntimeError, match="cannot be cached without its labels"):
            _demo.save_and_sample_labels(fit, label_vol)

        assert not _demo.LOCAL_FIT.exists()


class TestRejectedPairFallsThroughToRefit:
    """A rejected (fit, labels) pair must trigger a REFIT, not render nonsense.

    The user-visible point of #1670: detecting the mismatch is only half of it —
    ``load_or_build`` has to actually fall through to download-and-refit.
    """

    @staticmethod
    def _sentinel_setup(tmp_path, monkeypatch, *, permute: bool):
        """Cache an aligned pair (optionally permuting the sidecar), stub the refit."""
        n = 6000
        rng = np.random.default_rng(33)
        label_vol = rng.integers(0, 118, (16, 16, 16)).astype(np.int32)
        fit = _scattered_gsplat_data(n, extent=15.49, seed=5)
        # The refit writes to the local-fit namespace; the FETCHED sidecar door
        # (CACHE_LABELS) is pointed at that same file, which is what
        # ensure_dataset would have put there.
        monkeypatch.setattr(_demo, "LOCAL_FIT", tmp_path / _demo.FIT_FILE)
        monkeypatch.setattr(_demo, "LOCAL_LABELS", tmp_path / _demo.LABELS_FILE)
        monkeypatch.setattr(_demo, "CACHE_LABELS", tmp_path / _demo.LABELS_FILE)
        stored, labels = _legacy_pair(label_vol, fit)
        if permute:
            _save_labels_u8(labels[rng.permutation(n)], _demo.LOCAL_LABELS)

        monkeypatch.setattr(_demo, "RECOMPUTE", False)
        # The shipped LFS assets must not rescue (or mask) the outcome.
        monkeypatch.setattr(_demo, "LFS_FIT", tmp_path / "absent.gsplats.zarr.zip")
        monkeypatch.setattr(_demo, "LFS_LABELS", tmp_path / "absent.npz")
        # The manifest fetch resolves to the cached fit, as ensure_dataset would.
        monkeypatch.setattr(
            _demo,
            "load_dataset_gsplats",
            lambda *a, **k: [GSplatData.load(_demo.LOCAL_FIT, include_stats=False)],
        )
        monkeypatch.setattr(_demo, "warn_if_no_cuda_gpu", lambda: None)

        sentinel_fit = _scattered_gsplat_data(4, extent=1.0, seed=6)
        sentinel_labels = np.zeros(4, dtype=np.int32)
        monkeypatch.setattr(
            _demo, "load_ct_and_labels", lambda: (None, None, None, None)
        )
        monkeypatch.setattr(
            _demo, "fit_atlas", lambda *a, **k: (sentinel_fit, sentinel_labels)
        )
        return labels, sentinel_fit, sentinel_labels

    def test_permuted_sidecar_falls_through_to_the_refit(
        self, tmp_path, monkeypatch
    ) -> None:
        _, sentinel_fit, sentinel_labels = self._sentinel_setup(
            tmp_path, monkeypatch, permute=True
        )
        got_fit, got_labels = _demo.load_or_build()
        assert got_fit is sentinel_fit, "a rejected pair was rendered anyway"
        assert got_labels is sentinel_labels

    def test_aligned_sidecar_is_used_instead_of_refitting(
        self, tmp_path, monkeypatch
    ) -> None:
        labels, sentinel_fit, _ = self._sentinel_setup(
            tmp_path, monkeypatch, permute=False
        )
        got_fit, got_labels = _demo.load_or_build()
        assert got_fit is not sentinel_fit, "an aligned pair triggered a refit"
        np.testing.assert_array_equal(got_labels, labels)


class TestShippedLfsPairIsGuardedToo:
    """The SHIPPED (Git LFS) branch of ``load_or_build`` must run the guard too.

    ``load_or_build`` has two accept doors — the manifest-fetched cache and the
    packaged LFS assets — and each one calls ``_labels_match_fit`` separately.
    The manifest-cache door is covered above; without these two the LFS call
    could be replaced by ``if True:`` with the whole suite still green, because
    every other test points ``LFS_*`` at absent paths so that branch never runs.
    """

    @staticmethod
    def _lfs_setup(tmp_path, monkeypatch, *, permute: bool):
        """Materialize an LFS-shaped pair under tmp_path and stub the refit."""
        n = 6000
        rng = np.random.default_rng(34)
        label_vol = rng.integers(0, 118, (16, 16, 16)).astype(np.int32)
        fit = _scattered_gsplat_data(n, extent=15.49, seed=7)

        # Build the pair straight into the "shipped" location.
        lfs_dir = tmp_path / "lfs"
        lfs_dir.mkdir()
        monkeypatch.setattr(_demo, "LOCAL_FIT", lfs_dir / _demo.FIT_FILE)
        monkeypatch.setattr(_demo, "LOCAL_LABELS", lfs_dir / _demo.LABELS_FILE)
        _, labels = _legacy_pair(label_vol, fit)
        if permute:
            _save_labels_u8(labels[rng.permutation(n)], lfs_dir / _demo.LABELS_FILE)
        monkeypatch.setattr(_demo, "LFS_FIT", lfs_dir / _demo.FIT_FILE)
        monkeypatch.setattr(_demo, "LFS_LABELS", lfs_dir / _demo.LABELS_FILE)

        # …and make the OTHER two doors miss, so the LFS one is under test: no
        # fetched dataset, no fetched sidecar, and no local refit either. The
        # last of those is only true because `load_or_build` reads the local pair
        # through these constants; while it re-derived the path from the cache
        # root instead, this redirect was inert and the door opened onto the
        # developer's real ~/.cache (#1618 review, A).
        monkeypatch.setattr(
            _demo, "CACHE_LABELS", tmp_path / "cache" / _demo.LABELS_FILE
        )
        monkeypatch.setattr(_demo, "LOCAL_FIT", tmp_path / "local" / _demo.FIT_FILE)
        monkeypatch.setattr(
            _demo, "LOCAL_LABELS", tmp_path / "local" / _demo.LABELS_FILE
        )
        monkeypatch.setattr(_demo, "load_dataset_gsplats", lambda *a, **k: None)

        monkeypatch.setattr(_demo, "RECOMPUTE", False)
        monkeypatch.setattr(_demo, "warn_if_no_cuda_gpu", lambda: None)
        sentinel_fit = _scattered_gsplat_data(4, extent=1.0, seed=8)
        sentinel_labels = np.zeros(4, dtype=np.int32)
        monkeypatch.setattr(
            _demo, "load_ct_and_labels", lambda: (None, None, None, None)
        )
        monkeypatch.setattr(
            _demo, "fit_atlas", lambda *a, **k: (sentinel_fit, sentinel_labels)
        )
        return labels, sentinel_fit, sentinel_labels

    def test_a_permuted_shipped_sidecar_falls_through_to_the_refit(
        self, tmp_path, monkeypatch
    ) -> None:
        _, sentinel_fit, sentinel_labels = self._lfs_setup(
            tmp_path, monkeypatch, permute=True
        )
        got_fit, got_labels = _demo.load_or_build()
        assert got_fit is sentinel_fit, "a rejected SHIPPED pair was rendered anyway"
        assert got_labels is sentinel_labels

    def test_an_aligned_shipped_sidecar_is_used_instead_of_refitting(
        self, tmp_path, monkeypatch
    ) -> None:
        labels, sentinel_fit, _ = self._lfs_setup(tmp_path, monkeypatch, permute=False)
        got_fit, got_labels = _demo.load_or_build()
        assert got_fit is not sentinel_fit, "an aligned SHIPPED pair triggered a refit"
        np.testing.assert_array_equal(got_labels, labels)


class TestTheLocalDoorOpensOnTheSecondLaunch:
    """The headline behaviour of #1618, for the one demo with a PAIR of artifacts.

    Every other test in this file drives ``load_or_build`` with a pre-planted
    pair; none of them lets the demo write one and then find it again. The CT
    demo is the interesting case because both halves — the fit and the labels
    sidecar — have to land in, and be read back out of, the local-fit namespace,
    and because its local door was the one that ignored its own ``LOCAL_FIT``
    constant and read the developer's real ``~/.cache`` instead.

    Stubbed: the manifest fetch (nothing hosted, nothing in-repo — the situation
    that sends the demo down this path), the 3.2 GB TotalSegmentator download,
    and the GPU fit. The save, the reload, the label sampling and the alignment
    guard all run for real.
    """

    @staticmethod
    def _setup(tmp_path, monkeypatch):
        from luxar.demos import DatasetUnavailable

        monkeypatch.setattr(_demo, "RECOMPUTE", False)
        # BOTH halves redirected, and the door must honour them: the refit writes
        # through these constants, so a read that re-derives its path from the
        # cache root would reach past the redirect (#1618 review, A).
        monkeypatch.setattr(_demo, "LOCAL_FIT", tmp_path / "local" / _demo.FIT_FILE)
        monkeypatch.setattr(
            _demo, "LOCAL_LABELS", tmp_path / "local" / _demo.LABELS_FILE
        )
        monkeypatch.setattr(
            _demo, "CACHE_LABELS", tmp_path / "cache" / _demo.LABELS_FILE
        )
        monkeypatch.setattr(_demo, "LFS_FIT", tmp_path / "absent.gsplats.zarr.zip")
        monkeypatch.setattr(_demo, "LFS_LABELS", tmp_path / "absent.npz")

        def _nothing_hosted(*args, **kwargs):
            raise DatasetUnavailable("no cached copy, no in-repo copy, no record")

        monkeypatch.setattr(_demo, "load_dataset_gsplats", _nothing_hosted)
        monkeypatch.setattr(_demo, "warn_if_no_cuda_gpu", lambda: None)
        monkeypatch.setattr(_demo, "detect_device", lambda: "cpu")

        rng = np.random.default_rng(41)
        label_vol = rng.integers(0, 118, (16, 16, 16)).astype(np.int32)
        downloads: list[int] = []
        fits: list[int] = []

        def _download():
            downloads.append(1)
            return np.zeros((16, 16, 16), dtype=np.float32), label_vol, None, None

        def _fit(*args, **kwargs):
            fits.append(1)
            return _scattered_gsplat_data(6000, extent=15.49, seed=42)

        monkeypatch.setattr(_demo, "load_ct_and_labels", _download)
        import luxar.gsplats as _gsplats

        monkeypatch.setattr(_gsplats, "fit_progressive_gaussian_splats", _fit)
        return downloads, fits

    def test_the_second_launch_reuses_the_first_launch_s_pair(
        self, tmp_path, monkeypatch
    ) -> None:
        downloads, fits = self._setup(tmp_path, monkeypatch)

        first_fit, first_labels = _demo.load_or_build()
        assert (downloads, fits) == ([1], [1]), "launch 1 should download and fit once"
        # Completeness is now a property of ONE file: the refit writes a fit that
        # carries its own organ ids and emits no sidecar, so "both halves landed"
        # is the wrong question — "does the fit describe itself" is the right one.
        assert _demo.LOCAL_FIT.exists(), "launch 1 left no fit behind"
        assert not _demo.LOCAL_LABELS.exists(), "the refit still wrote a sidecar"
        cached = GSplatData.load(_demo.LOCAL_FIT)
        assert _demo._native_labels(cached) is not None, (
            "launch 1 cached a fit with no organ ids, so launch 2 must refit"
        )

        second_fit, second_labels = _demo.load_or_build()
        assert fits == [1], "launch 2 refitted — the local door never opened"
        assert downloads == [1], "launch 2 re-downloaded the 3.2 GB subset"
        np.testing.assert_array_equal(second_fit.centers, first_fit.centers)
        np.testing.assert_array_equal(second_labels, first_labels)

    def test_a_legacy_fit_with_no_sidecar_sends_the_second_launch_back_to_the_refit(
        self, tmp_path, monkeypatch
    ) -> None:
        """Half a pair is still not a usable answer, where a pair is what exists.

        A fit that carries its own ids cannot be half of anything, so the case
        this covers is the LEGACY shape — the in-repo Git-LFS generation, which
        has no label channel — with its sidecar gone. That branch is live until
        those payloads retire (#2354), and it must refit rather than render.
        """
        _, fits = self._setup(tmp_path, monkeypatch)
        _demo.load_or_build()
        assert fits == [1]

        # Replace the cached self-describing fit with the legacy shape, then take
        # the sidecar away.
        _legacy_pair(
            np.zeros((16, 16, 16), dtype=np.int32),
            _scattered_gsplat_data(6000, extent=15.49, seed=42),
        )
        assert _demo._native_labels(GSplatData.load(_demo.LOCAL_FIT)) is None
        _demo.LOCAL_LABELS.unlink()

        _demo.load_or_build()
        assert fits == [1, 1], "a fit with neither native ids nor a sidecar was used"

    def test_recompute_bypasses_the_local_door(self, tmp_path, monkeypatch) -> None:
        """``--recompute`` must refit even with a perfectly good local pair present."""
        _, fits = self._setup(tmp_path, monkeypatch)
        _demo.load_or_build()
        assert _demo.LOCAL_FIT.exists()

        monkeypatch.setattr(_demo, "RECOMPUTE", True)
        _demo.load_or_build()
        assert fits == [1, 1], "--recompute reused the cached local pair"
