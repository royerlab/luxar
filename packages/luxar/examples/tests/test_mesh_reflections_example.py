"""Guards for the reflections example — the Phase 4 acceptance scene.

Pinned: the scene authors ``viewer_config.environment.source = "scene"`` (the whole
point), both spheres are ``material="physical"`` with the knobs their spec names, the
swirl is the emissive light source (additive, bright, colours in linear RGB), and the
chrome sphere sits at the probe (the scene centre) so the reflection is exact where it
matters. The glass keeps ``refract_data`` OFF so the baked-vs-live comparison the
Phase 4 verification recorded stays meaningful.
"""

from __future__ import annotations

import numpy as np
import zarr

from luxar.core.viewer_config import ViewerConfig

from ._load import load_example

MOD = load_example("mesh_reflections_example")


def test_scene_authors_a_scene_derived_environment(tmp_path) -> None:
    store = tmp_path / "reflections.luxar.zarr"
    MOD.create_scene(store)
    root = zarr.open_group(str(store), mode="r")
    vc = ViewerConfig.from_dict(dict(root.attrs)["viewer_config"])
    assert vc.environment is not None
    assert vc.environment.source == "scene"
    assert vc.environment.probe == "auto"
    # No baked map ships with the example: it is the `luxar env bake` TARGET.
    assert "environment" not in root


def test_spheres_are_physical_and_the_swirl_is_the_light(tmp_path) -> None:
    store = tmp_path / "reflections.luxar.zarr"
    MOD.create_scene(store)
    root = zarr.open_group(str(store), mode="r")
    chrome = dict(root["chrome"].attrs)
    assert chrome["material"] == "physical"
    assert chrome["metalness"] == 1.0
    assert chrome["roughness"] == 0.05
    assert "transmission" not in chrome
    glass = dict(root["glass"].attrs)
    assert glass["material"] == "physical"
    assert glass["transmission"] == 1.0
    assert glass["ior"] == 1.5
    assert glass["thickness"] == MOD.RADIUS * 2.0
    assert "refract_data" not in glass
    swirl = dict(root["swirl"].attrs)
    assert swirl["blending_mode"] == "additive"
    assert swirl["n_points"] == MOD.SWIRL_POINT_COUNT
    assert swirl["intensity"] > 1.0
    assert "material" not in swirl


def test_chrome_sits_at_the_probe_and_the_glass_beside_it() -> None:
    positions, colors = MOD.swirl_points(MOD.SWIRL_POINT_COUNT, MOD.SWIRL_SEED)
    # The swirl is centred on the origin — where `probe: auto` (the bounds centre)
    # lands and where the chrome sphere is — within the jitter.
    centre = positions.mean(axis=0)
    assert np.all(np.abs(centre) < MOD.RADIUS * 0.6)
    assert float(np.linalg.norm(MOD.GLASS_OFFSET)) > MOD.RADIUS * 2.0
    assert colors.shape == (MOD.SWIRL_POINT_COUNT, 3)
    assert colors.min() >= 0.0 and colors.max() <= 1.0
    # Bright enough to light a metal: the ramp's ends are near full intensity.
    assert colors.max(axis=0).max() > 0.9
