"""Guards for the physical-materials example (Phases 1–2 acceptance scene).

What is pinned is the CONTRACT it demonstrates rather than the picture it makes:

* the house-shader reference carries no ``material`` attr at all — an absent
  attr, not ``"luxar"``, is what every pre-existing scene has;
* every other sphere is stamped ``material="physical"`` with exactly the knobs
  its spec names and nothing else (written only when set);
* the marker shell is translucent through PER-VERTEX alpha, so the physical
  material's opacity path is the same one the house shader uses;
* the emissive point clusters sit inside the shell and the clear glass, the layout
  the §3.4 default (points draw on top, unrefracted) is stated against.
"""

from __future__ import annotations

import numpy as np
import zarr

from ._load import load_example

MOD = load_example("mesh_physical_materials_example")

PHYSICAL_KEYS = (
    "roughness",
    "metalness",
    "clearcoat",
    "iridescence",
    "sheen",
    "transmission",
    "ior",
    "thickness",
    "attenuation_color",
    "attenuation_distance",
    "dispersion",
    "refract_data",
)


def test_scene_stamps_the_material_contract(tmp_path) -> None:
    store = tmp_path / "physical.luxar.zarr"
    MOD.create_scene(store)
    root = zarr.open_group(str(store), mode="r")

    for spec in MOD.SPHERES:
        attrs = dict(root[spec["name"]].attrs)
        material = spec.get("material")
        if material is None:
            assert "material" not in attrs, spec["name"]
            continue
        for key, value in material.items():
            assert attrs[key] == value, (spec["name"], key)
        # Nothing the spec did not name is stamped.
        for key in PHYSICAL_KEYS:
            assert (key in attrs) == (key in material), (spec["name"], key)


def test_glass_row_and_backdrop(tmp_path) -> None:
    """Phase 2: three glass spheres, each transmitting, in front of a house backdrop."""
    glass = [s for s in MOD.SPHERES if s.get("material", {}).get("transmission")]
    assert [s["name"] for s in glass] == [
        "clear_glass",
        "amber_glass",
        "dispersive_crystal",
    ]
    for spec in glass:
        assert spec["material"]["transmission"] == 1.0
        assert spec["material"]["thickness"] > 0
        # This example shows the Phase 2 default; the lens example shows the flag.
        assert "refract_data" not in spec["material"]
    amber = next(s for s in glass if s["name"] == "amber_glass")
    assert amber["material"]["attenuation_color"] == "#f6d148"
    crystal = next(s for s in glass if s["name"] == "dispersive_crystal")
    assert crystal["material"]["dispersion"] > 0

    store = tmp_path / "physical.luxar.zarr"
    MOD.create_scene(store)
    root = zarr.open_group(str(store), mode="r")
    backdrop = root["backdrop"]
    assert "material" not in backdrop.attrs
    assert backdrop.attrs["shading"] == "flat"
    assert backdrop.attrs["n_faces"] == MOD.BACKDROP_TILES_X * MOD.BACKDROP_TILES_Y * 2
    assert MOD.BACKDROP_DEPTH < -MOD.RADIUS
    glass_index = next(
        i for i, s in enumerate(MOD.SPHERES) if s["name"] == "clear_glass"
    )
    positions = MOD.shell_points(MOD.sphere_centre(glass_index), MOD.RADIUS)
    assert (
        float(np.linalg.norm(positions - MOD.sphere_centre(glass_index), axis=1).max())
        < MOD.RADIUS
    )
    assert root["glass_points"].attrs["n_points"] == len(positions)
    assert root["glass_points"].attrs["blending_mode"] == "additive"


def test_shell_is_translucent_through_vertex_alpha(tmp_path) -> None:
    store = tmp_path / "physical.luxar.zarr"
    MOD.create_scene(store)
    root = zarr.open_group(str(store), mode="r")
    shell = root["clearcoat_shell"]
    colors = np.asarray(shell["colors"][:])
    assert colors.shape[1] == 4
    assert np.all(colors[:, 3] < 1.0)
    assert shell.attrs["clearcoat"] == 1.0
    assert shell.attrs["has_colors"] is True


def test_points_sit_inside_the_shell(tmp_path) -> None:
    shell_index = next(
        i for i, s in enumerate(MOD.SPHERES) if s["name"] == "clearcoat_shell"
    )
    centre = MOD.sphere_centre(shell_index)
    positions = MOD.shell_points(centre, MOD.RADIUS)
    distances = np.linalg.norm(positions - centre[None, :], axis=1)
    assert float(distances.max()) < MOD.RADIUS
    store = tmp_path / "physical.luxar.zarr"
    MOD.create_scene(store)
    root = zarr.open_group(str(store), mode="r")
    assert root["shell_points"].attrs["blending_mode"] == "additive"
    assert root["shell_points"].attrs["n_points"] == len(positions)


def test_row_is_centred_and_evenly_spaced() -> None:
    centres = np.stack([MOD.sphere_centre(i) for i in range(len(MOD.SPHERES))])
    assert np.allclose(centres.mean(axis=0), 0.0, atol=1e-6)
    gaps = np.diff(centres[:, 0])
    assert np.allclose(gaps, gaps[0])
