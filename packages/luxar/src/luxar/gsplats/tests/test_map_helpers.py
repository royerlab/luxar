"""Direct tests for the A5.0 rebuild helpers ``_map_additive`` /
``_map_substitutive``.

These are exercised indirectly by filter/cull/transform on multi-LOD data, but
tested here in isolation so a regression in the offset bookkeeping or the
structure/metadata carry-over is caught directly rather than as a confusing
downstream failure.
"""

from __future__ import annotations

import numpy as np

from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData, SubstitutiveLevel


def _make_additive(n_lods: int = 3, per_lod: int = 10) -> GSplatData:
    rng = np.random.default_rng(42)
    lods = [
        AdditiveSubLOD(
            centers=(rng.random((per_lod, 3)) * 100).astype(np.float32),
            amplitudes=rng.random(per_lod).astype(np.float32),
            cholesky_factors=np.tile(
                np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (per_lod, 1)
            ),
            stats={"pass_index": i},
        )
        for i in range(n_lods)
    ]
    return GSplatData.from_additive_sublods(lods)


def test_map_additive_advances_offset_and_preserves_structure() -> None:
    data = _make_additive(n_lods=3, per_lod=10)

    # fn doubles centers and stamps each splat's amplitude with its LOD's start
    # offset — so the finest amplitudes must be [0]*10 + [10]*10 + [20]*10 if the
    # offset advances correctly.
    def _fn(lod: AdditiveSubLOD, offset: int, n: int) -> AdditiveSubLOD:
        return AdditiveSubLOD(
            centers=lod.centers * 2.0,
            amplitudes=np.full(n, float(offset), dtype=np.float32),
            cholesky_factors=lod.cholesky_factors,
            colors=lod.colors,
            stats=dict(lod.stats),
            truncation_radius=lod.truncation_radius,
        )

    out = data._map_additive(_fn)

    assert out.n_additive_sublods == 3
    assert out.n_splats == data.n_splats
    assert np.allclose(out.centers, data.centers * 2.0)
    expected_amps = np.concatenate(
        [np.full(10, 0.0), np.full(10, 10.0), np.full(10, 20.0)]
    ).astype(np.float32)
    assert np.array_equal(out.amplitudes, expected_amps)
    # per-sub-LOD stats carried through
    assert [lod.stats["pass_index"] for lod in out.additive_sublods] == [0, 1, 2]


def _pyramid() -> GSplatData:
    def _sub(n: int, seed: int) -> AdditiveSubLOD:
        rng = np.random.default_rng(seed)
        chol = np.zeros((n, 6), dtype=np.float32)
        chol[:, [0, 2, 5]] = rng.uniform(0.5, 2.0, size=(n, 3))
        return AdditiveSubLOD(
            centers=rng.uniform(0, 100, size=(n, 3)).astype(np.float32),
            amplitudes=rng.uniform(0.1, 1.0, size=(n,)).astype(np.float32),
            cholesky_factors=chol,
        )

    return GSplatData.from_substitutive_levels(
        [
            SubstitutiveLevel(additive_sublods=[_sub(30, 0)], level_index=0),
            SubstitutiveLevel(
                additive_sublods=[_sub(12, 1)], compression_factor=4, level_index=1
            ),
            SubstitutiveLevel(
                additive_sublods=[_sub(4, 2)], compression_factor=16, level_index=2
            ),
        ]
    )


def test_map_substitutive_preserves_levels_and_metadata() -> None:
    pyr = _pyramid()
    shift = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    before = [lvl.additive_sublods[0].centers.copy() for lvl in pyr.substitutive_levels]

    out = pyr._map_substitutive(lambda lvl: lvl.translate(shift))

    assert out.n_substitutive == 3
    # per-level compression_factor metadata carried through
    assert [lvl.compression_factor for lvl in out.substitutive_levels] == [1, 4, 16]
    # every level's centers translated by the same shift
    for i, lvl in enumerate(out.substitutive_levels):
        assert np.allclose(lvl.additive_sublods[0].centers, before[i] + shift)
