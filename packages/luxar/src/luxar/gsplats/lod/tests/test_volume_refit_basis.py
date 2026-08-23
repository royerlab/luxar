"""The LOD volume re-fit must run on the ladder's basis, not invent one (#1177).

`volume_refine_splats` gets a background-relative seed and a RAW volume. Left
unreconciled the inner fit inherits `floor="auto"` and re-estimates a background
from that volume, so a refined level can land on a different basis from its
siblings — which `conserve_mass` DC pinning partly hides, and is why it went
unnoticed.
"""

from __future__ import annotations

import numpy as np
import pytest

from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.lod import volume_refit as vr
from luxar.gsplats.lod.recipes import RecipeParams, build_part_lod


def _seed(n: int = 6) -> GSplatData:
    rng = np.random.default_rng(0)
    return GSplatData(
        centers=rng.uniform(4.0, 12.0, (n, 3)).astype(np.float32),
        amplitudes=rng.uniform(0.2, 1.0, n).astype(np.float32),
        cholesky_factors=np.tile([1.5, 0, 1.5, 0, 0, 1.5], (n, 1)).astype(np.float32),
    )


def _seed_with_basis(n: int = 8, image_min: float = 500.0) -> GSplatData:
    seed = _seed(n)
    return GSplatData(
        centers=seed.centers,
        amplitudes=seed.amplitudes,
        cholesky_factors=seed.cholesky_factors,
        stats={"image_min": image_min},
    )


def _volume(pedestal: float = 500.0) -> np.ndarray:
    v = np.full((16, 16, 16), pedestal, dtype=np.float32)
    v[6:10, 6:10, 6:10] += 1500.0
    return v


def test_config_carries_the_basis_and_defaults_to_absent() -> None:
    """Threaded on the config because it already reaches all three layers
    (`make_substitutive_lod` -> `_volume_refine_level` -> `volume_refine_splats`),
    so no signature had to grow a parameter."""
    assert vr.VolumeRefitConfig().image_min is None
    assert vr.VolumeRefitConfig(image_min=675.0).image_min == pytest.approx(675.0)


def test_a_known_basis_shifts_the_target_and_disables_the_inner_floor(
    monkeypatch,
) -> None:
    """Both halves are needed. Subtracting the level makes the volume
    background-relative like the seed; `floor="none"` then stops `auto` from
    estimating a *second* background on top of that."""
    captured: dict = {}

    def fake_fit(volume, **kwargs):
        captured["volume"] = np.asarray(volume).copy()
        captured["floor"] = kwargs.get("floor")
        return _seed()

    monkeypatch.setattr("luxar.gsplats.fit_gsplats.fit_gaussian_splats", fake_fit)
    cfg = vr.VolumeRefitConfig(
        image_min=500.0, iters=1, never_worse=False, conserve_mass=False
    )
    vr.volume_refine_splats(_seed(), _volume(500.0), config=cfg, device="cpu")

    assert captured["floor"] == "none"
    # The 500-count pedestal is gone; the blob's 1500 counts survive intact.
    assert float(captured["volume"].min()) == pytest.approx(0.0)
    assert float(captured["volume"].max()) == pytest.approx(1500.0)


def test_an_unknown_basis_leaves_behaviour_unchanged(monkeypatch) -> None:
    """A store that records nothing must keep today's behaviour rather than have
    a level guessed for it — guessing would be worse than the status quo."""
    captured: dict = {}

    def fake_fit(volume, **kwargs):
        captured["volume"] = np.asarray(volume).copy()
        captured["floor"] = kwargs.get("floor")
        return _seed()

    monkeypatch.setattr("luxar.gsplats.fit_gsplats.fit_gaussian_splats", fake_fit)
    cfg = vr.VolumeRefitConfig(
        image_min=None, iters=1, never_worse=False, conserve_mass=False
    )
    raw = _volume(500.0)
    vr.volume_refine_splats(_seed(), raw, config=cfg, device="cpu")

    assert captured["floor"] == "auto"
    np.testing.assert_array_equal(captured["volume"], raw)


def test_per_part_levels_forward_the_owning_fits_basis(monkeypatch) -> None:
    """A bare part tree has no top-level stats, so its owner must carry them."""
    captured = []

    def fake_refine(seed, volume, *, config, device=None):
        captured.append(config.image_min)
        return seed, {}

    monkeypatch.setattr(vr, "volume_refine_splats", fake_refine)
    params = RecipeParams(
        compression_factor=2,
        levels=1,
        refine="volume",
        refine_iters=1,
        volume=_volume(500.0),
        image_min=500.0,
        additive_ladders=False,
        quality_stamps=False,
        device="cpu",
    )
    build_part_lod(
        _seed(8).tree,
        "levels",
        params,
        cell=[(0.0, 16.0), (0.0, 16.0), (0.0, 16.0)],
    )

    assert captured and set(captured) == {500.0}


def test_fit_time_partition_carries_region_basis_to_part_recipe(monkeypatch) -> None:
    import luxar.gsplats.lod.recipes as recipes

    captured = []

    def fake_build(part, recipe, params, *, cell=None):
        captured.append(params.image_min)
        return part

    monkeypatch.setattr(recipes, "build_part_lod", fake_build)
    node = GSplatData.partition_from_regions(
        [_seed_with_basis()], recipe="levels", recipe_params=RecipeParams()
    )

    assert node is not None
    assert captured == [500.0]


def test_adaptive_recipe_carries_input_basis_to_every_part(monkeypatch) -> None:
    import luxar.gsplats.lod.recipes as recipes

    captured = []

    def fake_part(part, params, *, coverage, cell=None):
        captured.append(params.image_min)
        return part

    monkeypatch.setattr(recipes, "_substitutive_for_part", fake_part)
    node = recipes.build_adaptive(
        _seed_with_basis(), RecipeParams(max_elements=4, quality_stamps=False)
    )

    assert node.children
    assert captured and set(captured) == {500.0}


def test_batch_merge_carries_part_basis_to_part_recipe(monkeypatch) -> None:
    import luxar.gsplats.lod.recipes as recipes
    from luxar.gsplats.batch.merge_orchestrator import _finalize_part_node

    captured = []

    def fake_build(part, recipe, params, *, cell=None):
        captured.append(params.image_min)
        return part

    monkeypatch.setattr(recipes, "build_part_lod", fake_build)
    node = _finalize_part_node(_seed_with_basis(), "levels", None, 1)

    assert node is not None
    assert captured == [500.0]


def test_the_never_worse_guard_compares_on_one_basis(monkeypatch) -> None:
    """The guard renders seed and candidate against `volume`; once that is
    shifted, both sides and the persisted `mse_seed`/`mse_refit` are on the fit's
    basis instead of carrying a pedestal offset."""
    monkeypatch.setattr(
        "luxar.gsplats.fit_gsplats.fit_gaussian_splats",
        lambda volume, **k: _seed(),
    )
    cfg = vr.VolumeRefitConfig(
        image_min=500.0, iters=1, never_worse=True, conserve_mass=False
    )
    _out, stats = vr.volume_refine_splats(
        _seed(), _volume(500.0), config=cfg, device="cpu"
    )
    # A pedestal-offset MSE on this volume would sit at/above 500**2; on the
    # shifted basis the floor of the error is the signal only.
    for key in ("mse_seed", "mse_refit"):
        if key in stats and stats[key] is not None:
            assert float(stats[key]) < 500.0**2
