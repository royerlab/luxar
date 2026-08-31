"""Insertion-time amplitude normalisation.

The rule under test: a fitted archive stores amplitudes in raw source units
(detector counts), and the shader turns those directly into emitted radiance and
volumetric optical depth with no display-side compensation available — the
colormap window only picks a LUT index and is clamped to [0, 1]. So amplitudes
must be normalised as the node ENTERS a scene, not corrected at display time.

See :mod:`luxar.core.group.gsplats_pipeline.amplitude_norm` for the shader
citations behind that claim.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar._zarr_compat import read_node_attrs
from luxar.core.group.gsplats_pipeline.amplitude_norm import (
    AMPLITUDE_REFERENCE_PERCENTILE,
    NORMALIZATION_FACTOR_ATTR,
    normalize_gsplat_data,
    normalize_node_in_place,
    resolve_factor,
)
from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData, SubstitutiveLevel
from luxar.gsplats.tree import GSplatLeaf, GSplatLodGroup

DIMS = Dimensions([Dimension(n, display=True, range=(0.0, 120.0)) for n in "XYZ"])


def _data(n: int = 2000, peak: float = 800.0, seed: int = 0) -> GSplatData:
    """A splat set whose amplitudes span [0, peak] with a heavy tail."""
    rng = np.random.default_rng(seed)
    return GSplatData(
        centers=(rng.random((n, 3)).astype(np.float32) * 100.0),
        amplitudes=(rng.random(n).astype(np.float32) ** 2 * peak),
        cholesky_factors=np.tile(np.array([2, 0, 2, 0, 0, 2], np.float32), (n, 1)),
    )


def _ladder_data(n: int, peak: float, seed: int) -> GSplatData:
    data = _data(n=n, peak=peak, seed=seed)
    split = n // 2
    return GSplatData(
        additive_sublods=[
            AdditiveSubLOD(
                centers=data.centers[start:stop],
                amplitudes=data.amplitudes[start:stop],
                cholesky_factors=data.cholesky_factors[start:stop],
            )
            for start, stop in ((0, split), (split, n))
        ]
    )


def _build(tmp_path, data, name="g", **kwargs):
    out = tmp_path / f"{name}.luxar.zarr"
    with LuxarZarrCompiler(out) as compiler:
        scene = compiler.create_scene(dimensions=DIMS)
        scene.add_gsplats_from_data(name=name, result=data, **kwargs)
    return out


def _assert_scene_labels_follow_centers(scene_path, expected: GSplatData) -> None:
    from luxar.encoding.decoder import ArrayDecoder

    assert expected.label_ids is not None
    root = zarr.open_group(str(scene_path / "g"), mode="r")
    leaves = [root]
    if "label_ids" not in root:
        leaves = [root[name] for name in sorted(root.group_keys())]
    decoder = ArrayDecoder()
    centers = np.concatenate(
        [np.asarray(decoder.decode(leaf["centers"], root)) for leaf in leaves]
    )
    label_ids = np.concatenate(
        [np.asarray(decoder.decode(leaf["label_ids"], root)) for leaf in leaves]
    )
    assert {
        int(label_id): name
        for label_id, name in leaves[0].attrs["label_vocabulary"].items()
    } == expected.label_vocabulary
    for center, label_id in zip(centers, label_ids):
        distances = np.linalg.norm(expected.centers - center, axis=1)
        source_index = int(np.argmin(distances))
        assert distances[source_index] < 0.02
        assert int(label_id) == int(expected.label_ids[source_index])


def _window(scene_dir, name="g"):
    """The stored [lo, hi] amplitude window — hi is the p99.9 of what was written."""
    return read_node_attrs(scene_dir / name)["amplitude_data_range"]


def _factor(scene_dir, name="g"):
    return read_node_attrs(scene_dir / name).get(NORMALIZATION_FACTOR_ATTR)


# ---------------------------------------------------------------- the default


def test_raw_source_units_are_normalised_by_default(tmp_path):
    """Detector-count amplitudes land in [0, ~1] with no caller action."""
    data = _data(peak=800.0)
    assert np.percentile(data.amplitudes, AMPLITUDE_REFERENCE_PERCENTILE) > 100

    scene = _build(tmp_path, data)

    lo, hi = _window(scene)
    assert hi == pytest.approx(1.0, abs=0.02), (
        f"the robust reference should land at 1.0, got {hi}"
    )
    assert lo >= 0.0
    assert _factor(scene) == pytest.approx(
        1.0 / np.percentile(data.amplitudes, AMPLITUDE_REFERENCE_PERCENTILE), rel=1e-3
    )


def test_already_in_range_data_is_left_alone(tmp_path):
    """``auto`` must not double-brighten a fit that is already normalised.

    Many demos fit a volume already scaled to [0, 1] and then dim it by a
    hand-tuned constant. Rescaling their p99.9 up to 1.0 would blow every one of
    them out, so the default only acts when the reference EXCEEDS 1.0.
    """
    data = _data(peak=0.08)  # a dimmed, already-normalised fit
    scene = _build(tmp_path, data)

    _lo, hi = _window(scene)
    assert hi < 0.1, f"a dimmed fit must not be rescaled, window hi={hi}"
    assert _factor(scene) is None, "no factor should be stamped when nothing was scaled"


def test_opt_out_ships_raw_units(tmp_path):
    data = _data(peak=800.0)
    scene = _build(tmp_path, data, normalize_amplitudes=False)

    _lo, hi = _window(scene)
    assert hi > 100.0, f"opt-out must preserve raw units, got {hi}"
    assert _factor(scene) is None


def test_explicit_target(tmp_path):
    data = _data(peak=800.0)
    scene = _build(tmp_path, data, normalize_amplitudes=0.5)

    _lo, hi = _window(scene)
    assert hi == pytest.approx(0.5, abs=0.01)


def test_normalisation_is_idempotent(tmp_path):
    """Inserting an already-normalised result again must change nothing."""
    data = _data(peak=800.0)
    once = _build(tmp_path, data, name="a")
    hi_once = _window(once, "a")[1]

    ref = float(np.percentile(data.amplitudes, AMPLITUDE_REFERENCE_PERCENTILE))
    twice = _build(tmp_path, data.scale_intensity(1.0 / ref), name="b")

    assert _window(twice, "b")[1] == pytest.approx(hi_once, rel=1e-4)
    assert _factor(twice, "b") is None


@pytest.mark.parametrize("laddered", [False, True])
def test_file_ingestion_preserves_labels_when_normalising(tmp_path, laddered):
    data = _ladder_data(40, 800.0, 4) if laddered else _data(40, 800.0, 4)
    labels = (np.arange(data.n_splats, dtype=np.uint16) * 3) % 7
    labeled = data.with_label_ids(labels, {i: f"class-{i}" for i in range(7)})
    source = tmp_path / f"source-{laddered}.gsplats.zarr"
    labeled.save(source, ordering="none")

    out = tmp_path / f"scene-{laddered}.luxar.zarr"
    with LuxarZarrCompiler(out) as compiler:
        scene = compiler.create_scene(dimensions=DIMS)
        scene.add_gsplats_from_file("g", source)

    assert _factor(out) is not None
    _assert_scene_labels_follow_centers(out, labeled)


def test_substitutive_levels_share_one_factor(tmp_path):
    """Every LOD level scaled by the SAME number — the levels must not move
    relative to each other.

    Regression for a real bug in the first cut of this feature. The lod
    dispatch writes each substitutive level back through
    ``add_gsplats_from_data_impl``, so each level re-normalised ITSELF. A
    coarser level's amplitudes are larger (merged representatives carry
    combined mass), so its own p99.9 is higher and it got scaled DOWN relative
    to its siblings: measured factors 0.1266 / 0.2549 / 0.0594 across three
    levels of one node. The visible symptom is a brightness pop at every LOD
    switch — exactly what mass conservation exists to prevent.
    """
    from luxar.gsplats.gsplat_data import AdditiveSubLOD, SubstitutiveLevel

    rng = np.random.default_rng(7)

    def level(n, peak):
        return SubstitutiveLevel(
            additive_sublods=[
                AdditiveSubLOD(
                    centers=rng.random((n, 3)).astype(np.float32) * 100.0,
                    amplitudes=(rng.random(n).astype(np.float32) * peak),
                    cholesky_factors=np.tile(
                        np.array([2, 0, 2, 0, 0, 2], np.float32), (n, 1)
                    ),
                )
            ]
        )

    # finest first, each coarser level carrying larger merged amplitudes
    data = GSplatData(
        substitutive_levels=[level(4000, 10.0), level(1000, 40.0), level(250, 160.0)]
    )

    out = tmp_path / "lod.luxar.zarr"
    with LuxarZarrCompiler(out) as compiler:
        scene = compiler.create_scene(dimensions=DIMS)
        scene.add_gsplats_from_data(name="g", result=data)

    factors = []
    for i in range(3):
        child = out / "g" / f"child_{i}"
        factors.append(read_node_attrs(child).get(NORMALIZATION_FACTOR_ATTR))
    assert all(f is not None for f in factors), factors
    assert len(set(round(f, 12) for f in factors)) == 1, (
        f"LOD levels were normalised independently, so they no longer sit on one "
        f"exposure and the brightness pops at every switch: {factors}"
    )


def test_data_normalisation_updates_level_energy_stamps(tmp_path):
    source_energies = (4.0e6, 9.0e6)
    levels = []
    for index, (peak, energy) in enumerate(zip((800.0, 200.0), source_energies)):
        data = _ladder_data(n=1000, peak=peak, seed=index)
        levels.append(
            SubstitutiveLevel(
                additive_sublods=list(data.additive_sublods),
                stats={
                    "reference_energy": energy,
                    "quality": 0.75,
                    "refine_stats": {"mse_seed": 1.0},
                    "label": f"level-{index}",
                },
            )
        )

    scaled, factor = normalize_gsplat_data(GSplatData(substitutive_levels=levels), True)
    assert factor is not None
    for index, level in enumerate(scaled.substitutive_levels):
        assert "refine_stats" not in level.stats
        assert level.stats["label"] == f"level-{index}"

    out = _build(tmp_path, scaled, normalize_amplitudes=False)

    for index, source_energy in enumerate(reversed(source_energies)):
        attrs = read_node_attrs(out / "g" / f"child_{index}")
        stats = attrs["level_stats"]
        assert stats["reference_energy"] == pytest.approx(source_energy * factor**2)
        assert stats["quality"] == 0.75
        assert "refine_stats" not in stats


# ------------------------------------------------------- relative structure


def test_relative_brightness_survives(tmp_path):
    """A single factor rescales everything; it must not flatten the data."""
    data = _data(peak=800.0)
    before = np.asarray(data.amplitudes, dtype=np.float64)

    scene = _build(tmp_path, data)
    factor = _factor(scene)

    # The stored window's top is the reference; the RATIOS inside the data are
    # what a scale must preserve, and a pure multiply does.
    assert factor is not None
    scaled = before * factor
    assert np.corrcoef(before, scaled)[0, 1] == pytest.approx(1.0)


def test_a_hot_outlier_does_not_darken_the_scene(tmp_path):
    """The reference is robust, so one saturated splat cannot compress the rest.

    This is why the target is the 99.9th percentile and not ``max``: a single
    cosmic-ray splat 100x the real peak would otherwise push everything else
    down by 100x, and would move the factor on every refit.
    """
    data = _data(peak=800.0)
    amps = np.asarray(data.amplitudes).copy()
    amps[0] = 80_000.0  # one absurd outlier
    hot = GSplatData(
        centers=data.centers,
        amplitudes=amps,
        cholesky_factors=data.cholesky_factors,
    )

    clean_hi = _window(_build(tmp_path, data, name="clean"), "clean")[1]
    hot_hi = _window(_build(tmp_path, hot, name="hot"), "hot")[1]

    assert hot_hi == pytest.approx(clean_hi, rel=0.05), (
        "a single outlier moved the normalisation; the reference is not robust"
    )


# ----------------------------------------------------------- the graft path


def test_partition_gets_one_tree_wide_factor(tmp_path):
    """Every part scaled by the SAME number, derived from the pooled tree.

    A per-part factor would be actively wrong, not merely redundant: the tile
    holding the brightest region has a higher p99.9 than its neighbours, so
    per-part normalisation would scale adjacent tiles differently and show a
    visible exposure step at every seam.
    """
    from luxar.core.group.gsplats_pipeline.from_io import graft_gsplat_node

    rng = np.random.default_rng(1)

    def part(peak, offset):
        return GSplatData(
            centers=(rng.random((1500, 3)).astype(np.float32) * 30 + offset),
            amplitudes=(rng.random(1500).astype(np.float32) ** 2 * peak),
            cholesky_factors=np.tile(
                np.array([2, 0, 2, 0, 0, 2], np.float32), (1500, 1)
            ),
        )

    # Deliberately uneven: part 1 is 4x hotter than part 0.
    parts = [part(200.0, 0.0), part(800.0, 40.0), part(300.0, 80.0)]
    pooled_reference = np.percentile(
        np.concatenate([item.amplitudes for item in parts]),
        AMPLITUDE_REFERENCE_PERCENTILE,
    )
    node = GSplatData.partition_from_regions(parts)

    out = tmp_path / "p.luxar.zarr"
    with LuxarZarrCompiler(out) as compiler:
        scene = compiler.create_scene(dimensions=DIMS)
        graft_gsplat_node(scene, name="g", node=node)

    factors = [
        read_node_attrs(out / "g" / f"part_{i}").get(NORMALIZATION_FACTOR_ATTR)
        for i in range(3)
    ]
    assert all(f is not None for f in factors), f"no factor recorded: {factors}"
    assert len(set(round(f, 12) for f in factors)) == 1, (
        f"parts were scaled by DIFFERENT factors, which steps at every seam: {factors}"
    )
    assert factors[0] == pytest.approx(1.0 / pooled_reference, rel=1e-3)


def test_graft_normalisation_updates_leaf_energy_stamps(tmp_path):
    from luxar.core.group.gsplats_pipeline.from_io import graft_gsplat_node

    source_energies = (1.0e6, 4.0e5)
    parts = [
        _ladder_data(n=1000, peak=peak, seed=index)
        for index, peak in enumerate((200.0, 800.0))
    ]
    for index, (part, energy) in enumerate(zip(parts, source_energies)):
        part.tree.meta["stats"] = {
            "reference_energy": energy,
            "quality": 0.5,
            "refine_stats": {"mse_refit": 2.0},
            "label": f"part-{index}",
        }
    node = GSplatData.partition_from_regions(parts)

    out = tmp_path / "energy-stamps.luxar.zarr"
    with LuxarZarrCompiler(out) as compiler:
        scene = compiler.create_scene(dimensions=DIMS)
        graft_gsplat_node(scene, name="g", node=node)

    for index, source_energy in enumerate(source_energies):
        attrs = read_node_attrs(out / "g" / f"part_{index}")
        factor = attrs[NORMALIZATION_FACTOR_ATTR]
        stats = attrs["level_stats"]
        assert stats["reference_energy"] == pytest.approx(source_energy * factor**2)
        assert stats["quality"] == 0.5
        assert "refine_stats" not in stats
    for index, leaf in enumerate(node.children):
        assert "refine_stats" not in leaf.meta["stats"]
        assert leaf.meta["stats"]["label"] == f"part-{index}"


def test_graft_reference_uses_only_the_finest_lod_level():
    """Coarse merged representatives must not darken the default finest view."""

    def leaf(peak: float, seed: int) -> GSplatLeaf:
        return GSplatLeaf(
            additive_sublods=list(_data(peak=peak, seed=seed).additive_sublods)
        )

    coarse = leaf(160.0, 1)
    finest = leaf(10.0, 2)
    node = GSplatLodGroup(children=[coarse, finest])

    factor = normalize_node_in_place(node, True)

    assert factor is not None
    finest_reference = np.percentile(
        finest.additive_sublods[0].amplitudes, AMPLITUDE_REFERENCE_PERCENTILE
    )
    assert finest_reference == pytest.approx(1.0, abs=0.02)


def test_partition_opt_out(tmp_path):
    from luxar.core.group.gsplats_pipeline.from_io import graft_gsplat_node

    rng = np.random.default_rng(2)

    def part(offset):
        return GSplatData(
            centers=(rng.random((800, 3)).astype(np.float32) * 30 + offset),
            amplitudes=(rng.random(800).astype(np.float32) * 800.0),
            cholesky_factors=np.tile(
                np.array([2, 0, 2, 0, 0, 2], np.float32), (800, 1)
            ),
        )

    node = GSplatData.partition_from_regions([part(0.0), part(40.0)])
    out = tmp_path / "q.luxar.zarr"
    with LuxarZarrCompiler(out) as compiler:
        scene = compiler.create_scene(dimensions=DIMS)
        graft_gsplat_node(scene, name="g", node=node, normalize_amplitudes=False)

    assert read_node_attrs(out / "g" / "part_0").get(NORMALIZATION_FACTOR_ATTR) is None
    assert read_node_attrs(out / "g" / "part_0")["amplitude_data_range"][1] > 100.0


@pytest.mark.parametrize("kind", ["lod", "partition"])
def test_hand_built_structure_does_not_normalise_each_child(tmp_path, kind):
    out = tmp_path / f"manual-{kind}.luxar.zarr"
    with LuxarZarrCompiler(out) as compiler:
        scene = compiler.create_scene(dimensions=DIMS)
        if kind == "lod":
            wrapper = scene.add_lod_group("g", display_type="gsplats")
            child_attrs = [dict(coverage_fraction=0.0), dict(coverage_fraction=1.0)]
        else:
            wrapper = scene.add_partition_group(
                "g", display_type="gsplats", max_elements=2000
            )
            child_attrs = [{}, {}]

        wrapper.add_gsplats_from_data("child_0", _data(peak=200.0), **child_attrs[0])
        wrapper.add_gsplats_from_data("child_1", _data(peak=800.0), **child_attrs[1])

    for child_name in ("child_0", "child_1"):
        attrs = read_node_attrs(out / "g" / child_name)
        assert NORMALIZATION_FACTOR_ATTR not in attrs
        assert attrs["amplitude_data_range"][1] > 100.0


def test_hand_built_partition_volume_fits_keep_shared_raw_exposure(tmp_path):
    axis = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    volume = np.exp(-8.0 * (x * x + y * y + z * z)).astype(np.float32)

    out = tmp_path / "manual-partition-volumes.luxar.zarr"
    with LuxarZarrCompiler(out) as compiler:
        scene = compiler.create_scene(dimensions=DIMS)
        wrapper = scene.add_partition_group(
            "g", display_type="gsplats", max_elements=2000
        )
        for index, peak in enumerate((200.0, 800.0)):
            wrapper.add_gsplats_from_volume(
                f"child_{index}",
                volume * peak,
                seeds=30,
                n_iters=10,
                device="cpu",
                verbose=False,
            )
        scene.add_gsplats_from_volume(
            "raw_opt_out",
            volume * 800.0,
            seeds=30,
            n_iters=10,
            device="cpu",
            normalize_amplitudes=False,
            verbose=False,
        )

    for child_name in ("child_0", "child_1"):
        attrs = read_node_attrs(out / "g" / child_name)
        assert NORMALIZATION_FACTOR_ATTR not in attrs
        assert attrs["amplitude_data_range"][1] > 10.0

    raw_attrs = read_node_attrs(out / "raw_opt_out")
    assert NORMALIZATION_FACTOR_ATTR not in raw_attrs
    assert raw_attrs["amplitude_data_range"][1] > 10.0


def test_hand_built_structure_can_explicitly_normalise_a_child(tmp_path):
    out = tmp_path / "manual-explicit.luxar.zarr"
    with LuxarZarrCompiler(out) as compiler:
        scene = compiler.create_scene(dimensions=DIMS)
        wrapper = scene.add_partition_group(
            "g", display_type="gsplats", max_elements=2000
        )
        wrapper.add_gsplats_from_data(
            "child", _data(peak=800.0), normalize_amplitudes=True
        )

    attrs = read_node_attrs(out / "g" / "child")
    assert attrs[NORMALIZATION_FACTOR_ATTR] > 0.0
    assert attrs["amplitude_data_range"][1] == pytest.approx(1.0, abs=0.02)


# ----------------------------------------------------------- spec resolution


@pytest.mark.parametrize(
    "spec,reference,expected",
    [
        (True, 800.0, 1.0 / 800.0),
        ("auto", 800.0, 1.0 / 800.0),
        (True, 0.5, None),  # already in range
        (True, 1.0, None),  # exactly at the target
        (False, 800.0, None),
        (None, 800.0, None),
        (2.0, 800.0, 2.0 / 800.0),
        (True, 0.0, None),  # all-zero amplitudes
    ],
)
def test_resolve_factor(spec, reference, expected):
    got = resolve_factor(spec, reference)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected)


@pytest.mark.parametrize("bad", ["yes", -1.0, 0.0, float("inf")])
@pytest.mark.parametrize("reference", [800.0, 0.0])
def test_resolve_factor_rejects_nonsense(bad, reference):
    with pytest.raises(ValueError):
        resolve_factor(bad, reference)


def test_all_zero_amplitudes_are_not_touched(tmp_path):
    """Scaling a degenerate set is meaningless and would only void its stamps."""
    data = GSplatData(
        centers=np.zeros((50, 3), np.float32),
        amplitudes=np.zeros(50, np.float32),
        cholesky_factors=np.tile(np.array([2, 0, 2, 0, 0, 2], np.float32), (50, 1)),
    )
    scene = _build(tmp_path, data)
    assert _factor(scene) is None
