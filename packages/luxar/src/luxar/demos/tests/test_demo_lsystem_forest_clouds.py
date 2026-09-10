"""The forest's seasonal cloud cover (2026-09-10 review: "a very very light
cloud cover, multiscale Perlin style, that varies with seasons")."""

from __future__ import annotations

import numpy as np
import pytest

from luxar.demos import demo_lsystem_forest as demo


@pytest.fixture(scope="module")
def terrain() -> demo.Terrain:
    return demo.build_terrain(np.random.default_rng(3))


@pytest.fixture(scope="module")
def clouds(terrain: demo.Terrain) -> dict:
    out = demo._build_cloud_cover(np.random.default_rng(7), terrain)
    assert out is not None
    return out


def test_cloud_field_is_a_normalised_multiscale_noise() -> None:
    fbm = demo.FBm(np.random.default_rng(1), octaves=4, base_cell=42.0, amplitude=1.0)
    x = np.linspace(-80, 80, 200)
    field = demo.cloud_field(fbm, x[:, None], x[None, :])
    assert field.min() >= 0.0 and field.max() <= 1.0
    # A real field, not a constant: it spans a good part of the unit range
    # and its mean sits near the 0.5 the normalisation is built around.
    assert field.max() - field.min() > 0.4
    assert abs(field.mean() - 0.5) < 0.12


def test_every_season_has_a_deck_under_the_cap(clouds: dict) -> None:
    seasons = clouds["positions"][:, 0].astype(int)
    counts = np.bincount(seasons, minlength=4)
    assert (counts > 0).all()
    assert (counts <= demo.CLOUD_MAX_POINTS).all()
    assert len(clouds["colors"]) == len(clouds["radii"]) == len(clouds["labels"])
    assert len(clouds["labels"]) == len(clouds["positions"])


def test_coverage_follows_the_seasonal_thresholds(clouds: dict) -> None:
    """Lower threshold = more sky covered: winter > autumn > spring > summer."""
    counts = np.bincount(clouds["positions"][:, 0].astype(int), minlength=4)
    order = np.argsort(demo.CLOUD_THRESHOLDS)  # most covered first
    covered = [counts[i] for i in order]
    # Monotone unless the cap flattens the top of the ranking.
    for heavier, lighter in zip(covered, covered[1:], strict=False):
        assert heavier >= lighter or heavier == demo.CLOUD_MAX_POINTS
    assert counts[demo.WINTER] > counts[demo.SUMMER]


def test_deck_floats_well_above_the_terrain(
    clouds: dict, terrain: demo.Terrain
) -> None:
    z = clouds["positions"][:, 3]
    floor = terrain.h_max + demo.CLOUD_DECK_HEIGHT
    # The deck undulates by a few metres (a second fBm sample) around its floor.
    assert z.min() >= floor - 6.0
    assert z.max() <= floor + demo.CLOUD_DECK_THICKNESS + 6.0
    assert z.min() > terrain.h_max + 15.0, "well above any canopy"


def test_deck_is_faint(clouds: dict) -> None:
    """The brief was "very very light": a low additive gain and soft, pale points."""
    assert demo.CLOUD_INTENSITY < 0.1
    assert clouds["colors"].min() > 0.3  # pale tints only, no dark storm cloud
    assert clouds["radii"].min() >= 2.5  # broad soft blobs, not sparkle
