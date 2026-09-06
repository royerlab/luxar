"""Guards for the physical-materials mesh demo.

The demo is the Phase 1 acceptance scene for ``material="physical"``
(``docs/guides/specs/MESH_PHYSICAL_MATERIALS_SPEC.md``), so what is pinned here is
the CONTRACT it demonstrates rather than the picture it makes:

* the house-shader reference carries no ``material`` attr at all — an absent
  attr, not ``"luxar"``, is what every pre-existing scene has;
* every other sphere is stamped ``material="physical"`` with exactly the knobs
  its spec names and nothing else (written only when set);
* the marker shell is translucent through PER-VERTEX alpha, so the physical
  material's opacity path is the same one the house shader uses;
* the emissive point cluster sits inside that shell, which is the layout the
  §3.4 caveat (points draw on top, unrefracted) is stated against.
"""

from __future__ import annotations

import numpy as np
import zarr

from luxar.demos.demo_mesh_physical_materials import (
    RADIUS,
    SPHERES,
    create_scene,
    shell_points,
    sphere_centre,
)


def test_scene_stamps_the_material_contract(tmp_path) -> None:
    store = tmp_path / "physical.luxar.zarr"
    create_scene(store)
    root = zarr.open_group(str(store), mode="r")

    for spec in SPHERES:
        attrs = dict(root[spec["name"]].attrs)
        material = spec.get("material")
        if material is None:
            assert "material" not in attrs, spec["name"]
            continue
        for key, value in material.items():
            assert attrs[key] == value, (spec["name"], key)
        # Nothing the spec did not name is stamped.
        for key in ("roughness", "metalness", "clearcoat", "iridescence", "sheen"):
            assert (key in attrs) == (key in material), (spec["name"], key)


def test_shell_is_translucent_through_vertex_alpha(tmp_path) -> None:
    store = tmp_path / "physical.luxar.zarr"
    create_scene(store)
    root = zarr.open_group(str(store), mode="r")
    shell = root["clearcoat_shell"]
    colors = np.asarray(shell["colors"][:])
    assert colors.shape[1] == 4
    assert np.all(colors[:, 3] < 1.0)
    assert shell.attrs["clearcoat"] == 1.0
    assert shell.attrs["has_colors"] is True


def test_points_sit_inside_the_shell(tmp_path) -> None:
    shell_index = next(
        i for i, s in enumerate(SPHERES) if s["name"] == "clearcoat_shell"
    )
    centre = sphere_centre(shell_index)
    # The cluster is judged on the array the demo authors — the stored
    # `positions` are a quantized COORDINATE encoding, not raw floats.
    positions = shell_points(centre, RADIUS)
    distances = np.linalg.norm(positions - centre[None, :], axis=1)
    assert float(distances.max()) < RADIUS
    store = tmp_path / "physical.luxar.zarr"
    create_scene(store)
    root = zarr.open_group(str(store), mode="r")
    assert root["shell_points"].attrs["blending_mode"] == "additive"
    assert root["shell_points"].attrs["n_points"] == len(positions)


def test_row_is_centred_and_evenly_spaced() -> None:
    centres = np.stack([sphere_centre(i) for i in range(len(SPHERES))])
    assert np.allclose(centres.mean(axis=0), 0.0, atol=1e-6)
    gaps = np.diff(centres[:, 0])
    assert np.allclose(gaps, gaps[0])
