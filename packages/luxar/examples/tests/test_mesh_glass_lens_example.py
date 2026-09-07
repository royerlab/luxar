"""Guards for the glass-lens example — the Phase 3 showcase (spec §3.4).

Pinned: the two refracting spheres author ``refract_data=True`` with a transmission
above zero, the control sphere does NOT, the lattice is emissive and faint enough not
to saturate, and the spheres sit IN FRONT of the lattice so the lens has data to bend.
"""

from __future__ import annotations

import numpy as np
import zarr

from ._load import load_example

MOD = load_example("mesh_glass_lens_example")


def test_refracting_spheres_author_the_flag_and_the_control_does_not(tmp_path) -> None:
    store = tmp_path / "lens.luxar.zarr"
    MOD.create_scene(store)
    root = zarr.open_group(str(store), mode="r")
    for name in ("lens", "amber_bubble"):
        attrs = dict(root[name].attrs)
        assert attrs["material"] == "physical"
        assert attrs["transmission"] == 1.0
        assert attrs["refract_data"] is True
    control = dict(root["glass_first"].attrs)
    assert control["transmission"] == 1.0
    assert "refract_data" not in control
    bubble = dict(root["amber_bubble"].attrs)
    assert bubble["attenuation_color"] == "#f6d148"
    assert bubble["attenuation_distance"] > 0


def test_lattice_is_faint_emissive_data_behind_the_spheres(tmp_path) -> None:
    store = tmp_path / "lens.luxar.zarr"
    MOD.create_scene(store)
    root = zarr.open_group(str(store), mode="r")
    lattice = dict(root["lattice"].attrs)
    assert lattice["blending_mode"] == "additive"
    assert lattice["intensity"] <= 1.0
    positions, colors = MOD.lattice_points()
    assert lattice["n_points"] == len(positions)
    assert colors.min() >= 0.0 and colors.max() <= 1.0
    # Every lattice point is behind every sphere along the view axis (camera at +z).
    assert float(positions[:, 2].max()) < MOD.SPHERE_Z - MOD.RADIUS


def test_row_is_centred_and_the_explainer_is_present(tmp_path) -> None:
    assert sum(MOD.SPHERE_X) == 0.0
    store = tmp_path / "lens.luxar.zarr"
    MOD.create_scene(store)
    root = zarr.open_group(str(store), mode="r")
    assert "overlays" in root
    assert len(list(root["overlays"].group_keys())) >= 1
    assert np.isclose(MOD.LATTICE_INTENSITY, 0.8)
