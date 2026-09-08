"""Authoring tests for ``material="physical"`` on a mesh (Phases 1 and 2).

The contract under test is ``docs/guides/specs/MESH_PHYSICAL_MATERIALS_SPEC.md``
§3.1 and §3.4: an opt-in family selector, a closed set of knobs written only when
set (the ``[0, 1]`` surface knobs, then the glass family with its own ranges), and
— the load-bearing half — refusal of every pairing that has no valid meaning,
because a knob that writes cleanly and does nothing reads in the viewer exactly
like a working setting.

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
    IOR_MAX,
    IOR_MIN,
    MESH_MATERIALS,
    PHYSICAL_MATERIAL_ATTRS,
    PHYSICAL_MATERIAL_FRACTION_ATTRS,
    PHYSICAL_TRANSMISSION_ATTRS,
    PHYSICAL_TRANSMISSION_DEPENDENT_ATTRS,
    validate_hex_color,
    validate_ior,
    validate_mesh_material,
    validate_non_negative_finite,
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
    assert (
        set(PHYSICAL_MATERIAL_FRACTION_ATTRS)
        | {"sheen_color"}
        | PHYSICAL_TRANSMISSION_ATTRS
    ) == set(PHYSICAL_MATERIAL_ATTRS)
    assert "transmission" in PHYSICAL_MATERIAL_FRACTION_ATTRS
    assert PHYSICAL_TRANSMISSION_DEPENDENT_ATTRS < PHYSICAL_TRANSMISSION_ATTRS
    assert {"transmission", "ior"} == (
        PHYSICAL_TRANSMISSION_ATTRS - PHYSICAL_TRANSMISSION_DEPENDENT_ATTRS
    )
    assert HOUSE_SHADER_ONLY_ATTRS == {
        "ambient",
        "shade_exponent",
        "specular",
        "shininess",
    }
    assert PHYSICAL_MATERIAL_ATTRS | {"material"} <= MESH_ONLY_APPEARANCE_ATTRS
    # The Phase 2 glass family is in (spec §3.4), and Phase 3's one authored knob,
    # the `refract_data` flag, rides the same pairing rule.
    assert PHYSICAL_TRANSMISSION_ATTRS <= MESH_ONLY_APPEARANCE_ATTRS
    assert "refract_data" in PHYSICAL_TRANSMISSION_DEPENDENT_ATTRS


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


@pytest.mark.parametrize("value", [IOR_MIN, 1.5, 2.0, IOR_MAX, "1.33"])
def test_validate_ior_accepts_three_s_range(value) -> None:
    """Vacuum to diamond, inclusive; strings that parse are fine like every knob."""
    assert validate_ior(value) == float(value)


@pytest.mark.parametrize("value", [0.99, 2.34, 15, float("inf"), float("nan")])
def test_validate_ior_refuses_outside_the_range(value) -> None:
    with pytest.raises(ValueError, match=f"between {IOR_MIN} and {IOR_MAX}"):
        validate_ior(value)


def test_validate_non_negative_finite_admits_zero_but_not_below() -> None:
    """``thickness=0`` is a thin-walled bubble and legal; negative and NaN are not."""
    assert validate_non_negative_finite(0, "Thickness") == 0.0
    assert validate_non_negative_finite(0.4, "Thickness") == 0.4
    with pytest.raises(ValueError, match="at least 0"):
        validate_non_negative_finite(-0.1, "Thickness")
    with pytest.raises(ValueError, match="finite"):
        validate_non_negative_finite(float("nan"), "Thickness")
    with pytest.raises(TypeError, match="convertible to float"):
        validate_non_negative_finite("thick", "Thickness")


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


def test_glass_family_round_trips_and_is_written_only_when_set(tmp_path) -> None:
    """Phase 2: the six glass knobs reach zarr as authored; absent ones stay absent."""
    store = tmp_path / "glass.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        amber = scene.add_mesh(
            "amber",
            _V,
            _F,
            normals=_N,
            normal_dims=[0, 1, 2],
            material="physical",
            roughness=0.05,
            transmission=1.0,
            ior=1.5,
            thickness=0.4,
            attenuation_color="#f6d148",
            attenuation_distance=0.3,
            dispersion=0.5,
            refract_data=True,
        )
        assert amber.attrs["transmission"] == 1.0
        assert amber.attrs["attenuation_color"] == "#f6d148"
        assert amber.attrs["refract_data"] is True
        # `ior` alone is meaningful (it sets an opaque surface's reflectance) …
        metal = scene.add_mesh("metal", _V, _F, material="physical", ior=2.0)
        assert metal.attrs["ior"] == 2.0
        assert "transmission" not in metal.attrs
        # … and `thickness=0` is a legal thin wall once transmission is on.
        bubble = scene.add_mesh(
            "bubble", _V, _F, material="physical", transmission=0.9, thickness=0
        )
        assert bubble.attrs["thickness"] == 0.0

    root = zarr.open_group(str(store), mode="r")
    attrs = dict(root["amber"].attrs)
    for key in PHYSICAL_TRANSMISSION_ATTRS:
        assert key in attrs, key
    assert attrs["ior"] == 1.5
    assert attrs["dispersion"] == 0.5
    assert attrs["refract_data"] is True
    assert "thickness" not in root["metal"].attrs
    # Absent means false: a Phase 2 glass that never asked stays glass-first.
    assert "refract_data" not in root["bubble"].attrs


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
        (
            dict(material="physical", transmission=1.2),
            "Transmission must be finite and between 0.0 and 1.0",
            "transmission is a fraction",
        ),
        (
            dict(material="physical", ior=3.0),
            f"Ior must be finite and between {IOR_MIN} and {IOR_MAX}",
            "ior outside three's range",
        ),
        (
            dict(material="physical", transmission=1.0, attenuation_distance=0),
            "Attenuation distance must be finite and greater than 0",
            "attenuation distance divides, so zero is refused",
        ),
        (
            dict(material="physical", transmission=1.0, thickness=-1),
            "Thickness must be finite and at least 0",
            "negative thickness",
        ),
        (
            dict(material="physical", transmission=1.0, attenuation_color="amber"),
            "Attenuation color must be a '#rrggbb' hex colour string",
            "attenuation colour spelling",
        ),
        (
            dict(material="physical", thickness=0.4),
            r"\['thickness'\] but no transmission above zero",
            "glass knob without transmission",
        ),
        (
            dict(material="physical", transmission=0.0, dispersion=0.5),
            r"\['dispersion'\] but no transmission above zero",
            "glass knob with transmission explicitly zero",
        ),
        (
            dict(material="physical", refract_data=True),
            r"\['refract_data'\] but no transmission above zero",
            "refract_data without transmission",
        ),
        (
            dict(material="physical", transmission=1.0, refract_data=1),
            "Refract data must be a bool",
            "refract_data as a number",
        ),
        (
            dict(
                material="physical",
                attenuation_color="#f6d148",
                attenuation_distance=0.3,
                dispersion=0.2,
            ),
            r"\['attenuation_color', 'attenuation_distance', 'dispersion'\]",
            "every dead glass knob is named",
        ),
        (
            dict(transmission=1.0),
            r"\['transmission'\].*material='physical'",
            "transmission without the opt-in",
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
