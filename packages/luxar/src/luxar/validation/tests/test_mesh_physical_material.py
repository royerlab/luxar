"""Authoring tests for ``material="physical"`` on a mesh (Phase 1).

The contract under test is ``docs/guides/specs/MESH_PHYSICAL_MATERIALS_SPEC.md``
§3.1: an opt-in family selector, a closed set of ``[0, 1]`` knobs written only
when set, and — the load-bearing half — refusal of every pairing that has no
valid meaning, because a knob that writes cleanly and does nothing reads in the
viewer exactly like a working setting.

Every refusal is parametrized as a ``(kwargs, error_pattern, test_id)`` triple in
the style of ``test_mesh_validation.py``; each was verified to fail before the
adder cross-check existed.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from luxar import Dimensions, LuxarZarrCompiler
from luxar.core.group.compositing import MESH_ONLY_APPEARANCE_ATTRS
from luxar.validation.types import (
    HOUSE_SHADER_ONLY_ATTRS,
    MESH_MATERIALS,
    PHYSICAL_MATERIAL_ATTRS,
    PHYSICAL_MATERIAL_FRACTION_ATTRS,
    validate_hex_color,
    validate_mesh_material,
)

# A welded tetrahedron: the smallest closed surface, 4 vertices / 4 faces.
_V = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)
_F = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.uint32)
_N = np.array(
    [[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
    dtype=np.float32,
)


# =============================================================================
# The vocabulary itself
# =============================================================================


def test_vocabulary_is_closed_and_registered_everywhere() -> None:
    """The knob set is one frozenset, and every key is refused on non-mesh nodes."""
    assert MESH_MATERIALS == ("luxar", "physical")
    assert set(PHYSICAL_MATERIAL_FRACTION_ATTRS) | {"sheen_color"} == set(
        PHYSICAL_MATERIAL_ATTRS
    )
    assert HOUSE_SHADER_ONLY_ATTRS == {
        "ambient",
        "shade_exponent",
        "specular",
        "shininess",
    }
    assert PHYSICAL_MATERIAL_ATTRS | {"material"} <= MESH_ONLY_APPEARANCE_ATTRS
    # Phase 2+ knobs are deliberately not here (spec §3.4).
    for later in (
        "transmission",
        "ior",
        "thickness",
        "attenuation_color",
        "dispersion",
    ):
        assert later not in MESH_ONLY_APPEARANCE_ATTRS


@pytest.mark.parametrize("value", ["luxar", "physical"])
def test_validate_mesh_material_accepts_the_two_families(value: str) -> None:
    assert validate_mesh_material(value) == value


@pytest.mark.parametrize("value", ["physcial", "Physical", "", None, 1])
def test_validate_mesh_material_refuses_anything_else(value) -> None:
    with pytest.raises(ValueError, match="Material must be one of luxar, physical"):
        validate_mesh_material(value)


@pytest.mark.parametrize("value", ["#ff4d6d", "#FFFFFF", "#000000", "#AbCdEf"])
def test_validate_hex_color_accepts_six_digit_forms(value: str) -> None:
    assert validate_hex_color(value, "Sheen color") == value


@pytest.mark.parametrize(
    "value", ["ff4d6d", "#f00", "#gg0000", "#ff4d6d00", (1, 0, 0), None]
)
def test_validate_hex_color_refuses_other_spellings(value) -> None:
    with pytest.raises(ValueError, match="'#rrggbb' hex colour string"):
        validate_hex_color(value, "Sheen color")


# =============================================================================
# Authoring round-trip
# =============================================================================


def test_physical_attrs_are_written_only_when_set(tmp_path) -> None:
    """The selector and its knobs reach zarr; absent knobs stay absent.

    "Written only when set" is load-bearing: the viewer takes three's own default
    for an absent knob, and a stamped default would freeze today's value into
    every store.
    """
    store = tmp_path / "physical.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        mesh = scene.add_mesh(
            "shell",
            _V,
            _F,
            normals=_N,
            normal_dims=[0, 1, 2],
            material="physical",
            roughness=0.4,
            metalness=1.0,
            clearcoat=1.0,
            clearcoat_roughness=0.1,
            sheen=0.5,
            sheen_color="#FFcc00",
            alpha_cutoff=0.3,
            shading="flat",
        )
        assert mesh.attrs["material"] == "physical"
        assert mesh.attrs["roughness"] == 0.4
        assert mesh.attrs["sheen_color"] == "#FFcc00"
        assert "iridescence" not in mesh.attrs
        # Explicit "luxar" is accepted and persisted as written …
        house = scene.add_mesh("house", _V, _F, material="luxar")
        assert house.attrs["material"] == "luxar"
        # … and the default carries no material attr at all.
        plain = scene.add_mesh("plain", _V, _F)
        assert "material" not in plain.attrs

    root = zarr.open_group(str(store), mode="r")
    attrs = dict(root["shell"].attrs)
    assert attrs["material"] == "physical"
    assert attrs["clearcoat"] == 1.0
    assert attrs["alpha_cutoff"] == 0.3
    assert attrs["shading"] == "flat"
    assert "iridescence" not in attrs
    assert "material" not in root["plain"].attrs


def test_physical_keeps_smooth_and_flat_shading(tmp_path) -> None:
    """Flat-versus-smooth normals is a property of any lit surface, so both stay."""
    store = tmp_path / "shading.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_mesh(
            "smooth",
            _V,
            _F,
            normals=_N,
            normal_dims=[0, 1, 2],
            material="physical",
            shading="smooth",
        )
        scene.add_mesh("flat", _V, _F, material="physical", shading="flat")
    # `shading` is a WRITER stamp, so it is read back from the store.
    root = zarr.open_group(str(store), mode="r")
    assert root["smooth"].attrs["shading"] == "smooth"
    assert root["flat"].attrs["shading"] == "flat"


def test_physical_rides_the_partition_route(tmp_path) -> None:
    """A partitioned physical mesh stamps the family on every generated part."""
    store = tmp_path / "parts.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_mesh(
            "parts",
            _V,
            _F,
            material="physical",
            metalness=0.8,
            partition={"max_elements": 2},
        )
    root = zarr.open_group(str(store), mode="r")
    parts = [k for k in root["parts"].keys() if k.startswith("part_")]
    assert len(parts) >= 2
    for part in parts:
        assert root["parts"][part].attrs["material"] == "physical"
        assert root["parts"][part].attrs["metalness"] == 0.8


def test_physical_knob_survives_post_hoc_write_through_on_a_mesh(tmp_path) -> None:
    """The write-through guard is a NON-mesh guard; a mesh leaf may be tweaked."""
    store = tmp_path / "tweak.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        mesh = scene.add_mesh("m", _V, _F, material="physical")
        mesh.attrs["roughness"] = 0.25
    root = zarr.open_group(str(store), mode="r")
    assert root["m"].attrs["roughness"] == 0.25


# =============================================================================
# Refusals
# =============================================================================


@pytest.mark.parametrize(
    "kwargs,error_pattern,test_id",
    [
        (
            dict(material="physcial"),
            "Material must be one of luxar, physical",
            "material typo is a typo, not a missing opt-in",
        ),
        (
            dict(roughness=0.2),
            "physically based material knobs.*material='physical'",
            "physical knob without the opt-in",
        ),
        (
            dict(material="luxar", sheen_color="#ff0000"),
            r"\['sheen_color'\].*material='physical'",
            "explicit luxar plus a physical knob",
        ),
        (
            dict(material="physical", roughness=1.5),
            "Roughness must be finite and between 0.0 and 1.0",
            "fraction out of range",
        ),
        (
            dict(material="physical", clearcoat_roughness=float("nan")),
            "Clearcoat roughness must be finite",
            "NaN fraction",
        ),
        (
            dict(material="physical", metalness="shiny"),
            "Metalness must be convertible to float",
            "non-numeric fraction",
        ),
        (
            dict(material="physical", sheen_color="ff0000"),
            "Sheen color must be a '#rrggbb' hex colour string",
            "sheen colour without the hash",
        ),
        (
            dict(material="physical", ambient=0.2),
            r"material='physical' and \['ambient'\].*house shader",
            "house-shader knob under physical",
        ),
        (
            dict(material="physical", specular=0.1, shininess=8.0),
            r"\['shininess', 'specular'\]",
            "several house-shader knobs are all named",
        ),
        (
            dict(material="physical", blending_mode="additive"),
            "material='physical' and blending_mode='additive'",
            "blending mode under physical",
        ),
        (
            dict(material="physical", shading="none"),
            "material='physical' and shading='none'",
            "unlit shading under physical",
        ),
        (
            dict(material="physical", shading="smoooth"),
            "shading must be one of",
            "a shading typo is still the writer's diagnostic",
        ),
    ],
)
def test_physical_refusals(tmp_path, kwargs, error_pattern, test_id) -> None:
    """Every pairing with no valid meaning is refused before anything is written."""
    store = tmp_path / "refuse.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises((ValueError, TypeError), match=error_pattern):
            scene.add_mesh("m", _V, _F, **kwargs)
        assert all(child.name != "m" for child in scene.children), test_id
    root = zarr.open_group(str(store), mode="r")
    assert "m" not in root, test_id


def test_physical_refuses_colormap_and_texture(tmp_path) -> None:
    """Phase 1 takes its base colour from vertex colours only."""
    with LuxarZarrCompiler(tmp_path / "sources.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises(ValueError, match="material='physical' and a colormap"):
            scene.add_mesh(
                "cm",
                _V,
                _F,
                scalars=np.arange(4, dtype=np.float32),
                colormap="viridis",
                material="physical",
            )
        uvs = np.zeros((4, 2), dtype=np.float32)
        texture = np.zeros((2, 2, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="material='physical' and a texture"):
            scene.add_mesh("tex", _V, _F, uvs=uvs, texture=texture, material="physical")


def test_refusals_are_not_wrapped_as_write_failures(tmp_path) -> None:
    """A pairing refusal is an argument error and must not read as a write failure."""
    with LuxarZarrCompiler(tmp_path / "wrap.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises(ValueError) as excinfo:
            scene.add_mesh("m", _V, _F, material="physical", ambient=0.1)
        assert "Cannot add mesh 'm'" in str(excinfo.value)
