"""Tests for nD transform validation and composition."""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from luxar import (
    Dimension,
    Dimensions,
    LuxarScene,
    LuxarZarrCompiler,
    apply_nd_transform_to_bounds,
    compose_nd_transforms,
    validate_nd_transform,
)


class TestValidateNdTransform:
    """Test validate_nd_transform function."""

    def test_valid_affine(self) -> None:
        """Test valid affine entry."""
        result = validate_nd_transform({"Time": {"scale": 0.001, "offset": 50.0}})
        assert result == {"Time": {"scale": 0.001, "offset": 50.0}}

    def test_valid_affine_scale_only(self) -> None:
        """Test affine with scale only (offset defaults to 0)."""
        result = validate_nd_transform({"Time": {"scale": 2.0}})
        assert result == {"Time": {"scale": 2.0}}

    def test_valid_affine_offset_only(self) -> None:
        """Test affine with offset only (scale defaults to 1)."""
        result = validate_nd_transform({"Time": {"offset": 100.0}})
        assert result == {"Time": {"offset": 100.0}}

    def test_valid_permutation(self) -> None:
        """Test valid permutation entry."""
        result = validate_nd_transform({"Channel": {"permutation": [2, 1, 0]}})
        assert result == {"Channel": {"permutation": [2, 1, 0]}}

    def test_valid_mixed_dims(self) -> None:
        """Test mixed affine and permutation on different dimensions."""
        result = validate_nd_transform(
            {
                "Time": {"scale": 0.001, "offset": 50.0},
                "Channel": {"permutation": [2, 1, 0]},
            }
        )
        assert "Time" in result
        assert "Channel" in result

    def test_empty_dict_is_identity(self) -> None:
        """Test empty dict is valid (identity transform)."""
        result = validate_nd_transform({})
        assert result == {}

    def test_reject_non_dict(self) -> None:
        """Test rejection of non-dict input."""
        with pytest.raises(TypeError, match="must be a dict"):
            validate_nd_transform("not a dict")

    def test_reject_non_string_key(self) -> None:
        """Test rejection of non-string keys."""
        with pytest.raises(ValueError, match="keys must be strings"):
            validate_nd_transform({42: {"scale": 1.0}})

    def test_reject_non_dict_entry(self) -> None:
        """Test rejection of non-dict entry values."""
        with pytest.raises(ValueError, match="must be a dict"):
            validate_nd_transform({"Time": [1, 2, 3]})

    def test_reject_mixed_affine_and_permutation(self) -> None:
        """Test rejection of both affine and permutation on same dim."""
        with pytest.raises(ValueError, match="cannot have both"):
            validate_nd_transform({"Time": {"scale": 2.0, "permutation": [0, 1]}})

    def test_reject_empty_entry(self) -> None:
        """Test rejection of entry with no recognized keys."""
        with pytest.raises(ValueError, match="must have"):
            validate_nd_transform({"Time": {}})

    def test_reject_invalid_scale_type(self) -> None:
        """Test rejection of non-numeric scale."""
        with pytest.raises(ValueError, match="must be a number"):
            validate_nd_transform({"Time": {"scale": "fast"}})

    def test_reject_unknown_affine_keys(self) -> None:
        """Test rejection of unknown keys in affine entry."""
        with pytest.raises(ValueError, match="unknown keys"):
            validate_nd_transform({"Time": {"scale": 1.0, "rotation": 45}})

    def test_warn_scale_zero(self) -> None:
        """Test warning for scale=0."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_nd_transform({"Time": {"scale": 0.0}})
            assert len(w) == 1
            assert "scale is 0" in str(w[0].message)

    def test_reject_invalid_permutation_type(self) -> None:
        """Test rejection of non-list permutation."""
        with pytest.raises(ValueError, match="must be a list"):
            validate_nd_transform({"Ch": {"permutation": "abc"}})

    def test_reject_invalid_permutation_values(self) -> None:
        """Test rejection of non-int permutation elements."""
        with pytest.raises(ValueError, match="must be int"):
            validate_nd_transform({"Ch": {"permutation": [0.5, 1.5]}})

    def test_reject_invalid_permutation_not_permutation(self) -> None:
        """Test rejection of non-permutation list."""
        with pytest.raises(ValueError, match="valid permutation"):
            validate_nd_transform({"Ch": {"permutation": [0, 0, 1]}})

    def test_negative_scale_is_valid(self) -> None:
        """Test negative scale (dimension flip) is valid."""
        result = validate_nd_transform({"Time": {"scale": -1.0, "offset": 100.0}})
        assert result["Time"]["scale"] == -1.0


class TestValidateWithDimensions:
    """Test validation with dimension context."""

    @pytest.fixture
    def dims_5d(self) -> Dimensions:
        return Dimensions(
            [
                Dimension("X", display=True),
                Dimension("Y", display=True),
                Dimension("Z", display=True),
                Dimension("Time", display=False, range=(0, 100)),
                Dimension(
                    "Channel", display=False, categories=["DAPI", "GFP", "mCherry"]
                ),
            ]
        )

    def test_valid_with_dimensions(self, dims_5d) -> None:
        result = validate_nd_transform(
            {"Time": {"offset": 50.0}, "Channel": {"permutation": [2, 1, 0]}},
            dimensions=dims_5d,
        )
        assert "Time" in result
        assert "Channel" in result

    def test_reject_displayed_dimension(self, dims_5d) -> None:
        with pytest.raises(ValueError, match="displayed dimension"):
            validate_nd_transform(
                {"X": {"scale": 2.0}},
                dimensions=dims_5d,
            )

    def test_reject_unknown_dimension(self, dims_5d) -> None:
        with pytest.raises(ValueError, match="not found"):
            validate_nd_transform(
                {"NonExistent": {"offset": 1.0}},
                dimensions=dims_5d,
            )

    def test_reject_affine_on_categorical(self, dims_5d) -> None:
        with pytest.raises(ValueError, match="requires 'permutation'"):
            validate_nd_transform(
                {"Channel": {"scale": 2.0}},
                dimensions=dims_5d,
            )

    def test_reject_permutation_on_continuous(self, dims_5d) -> None:
        with pytest.raises(ValueError, match="requires 'scale'/'offset'"):
            validate_nd_transform(
                {"Time": {"permutation": [0, 1]}},
                dimensions=dims_5d,
            )

    def test_reject_wrong_permutation_length(self, dims_5d) -> None:
        with pytest.raises(ValueError, match="length"):
            validate_nd_transform(
                {"Channel": {"permutation": [0, 1]}},  # 2 elements, need 3
                dimensions=dims_5d,
            )

    def test_reject_empty_categorical_dimension(self) -> None:
        dims = SimpleNamespace(
            dimensions=[
                SimpleNamespace(
                    name="Category",
                    display=False,
                    is_categorical=True,
                    discrete=False,
                    categories=[],
                )
            ]
        )
        with pytest.raises(ValueError, match="at least one category"):
            validate_nd_transform(
                {"Category": {"permutation": []}},
                dimensions=dims,
            )


class TestComposeNdTransforms:
    """Test compose_nd_transforms function."""

    def test_empty_composition(self) -> None:
        assert compose_nd_transforms() == {}

    def test_single_transform(self) -> None:
        t = {"Time": {"scale": 2.0, "offset": 10.0}}
        assert compose_nd_transforms(t) == t

    def test_affine_composition(self) -> None:
        """Test: parent(child(x)) = s_p * (s_c * x + o_c) + o_p."""
        parent = {"Time": {"scale": 2.0, "offset": 100.0}}
        child = {"Time": {"scale": 0.5, "offset": 10.0}}
        # composed: scale = 2 * 0.5 = 1.0, offset = 2 * 10 + 100 = 120
        result = compose_nd_transforms(parent, child)
        # scale=1.0 is identity, should be omitted
        assert result["Time"]["offset"] == 120.0

    def test_affine_composition_scale(self) -> None:
        """Test scale composition."""
        parent = {"Time": {"scale": 3.0}}
        child = {"Time": {"scale": 2.0}}
        result = compose_nd_transforms(parent, child)
        assert result["Time"]["scale"] == 6.0

    def test_permutation_composition(self) -> None:
        """Test: composed[i] = parent[child[i]]."""
        parent = {"Ch": {"permutation": [2, 0, 1]}}  # 0→2, 1→0, 2→1
        child = {"Ch": {"permutation": [1, 2, 0]}}  # 0→1, 1→2, 2→0
        # composed: 0→child→1→parent→0, 1→child→2→parent→1, 2→child→0→parent→2
        result = compose_nd_transforms(parent, child)
        assert result["Ch"]["permutation"] == [0, 1, 2]  # identity!

    def test_different_dims(self) -> None:
        """Test composition of transforms on different dimensions."""
        t1 = {"Time": {"offset": 10.0}}
        t2 = {"Channel": {"permutation": [1, 0]}}
        result = compose_nd_transforms(t1, t2)
        assert "Time" in result
        assert "Channel" in result

    def test_three_level(self) -> None:
        """Test 3-level hierarchy composition."""
        root = {"Time": {"offset": 100.0}}
        mid = {"Time": {"scale": 2.0}}
        leaf = {"Time": {"offset": 5.0}}
        # leaf: x → x + 5
        # mid: x → 2x
        # root: x → x + 100
        # composed: x → 2(x + 5) + 100 = 2x + 110
        result = compose_nd_transforms(root, mid, leaf)
        assert result["Time"]["scale"] == 2.0
        assert result["Time"]["offset"] == 110.0

    def test_identity_elision(self) -> None:
        """Test that identity transforms are elided from result."""
        t1 = {"Time": {"scale": 2.0}}
        t2 = {"Time": {"scale": 0.5}}
        # scale = 2 * 0.5 = 1.0, offset = 2*0 + 0 = 0 → identity
        result = compose_nd_transforms(t1, t2)
        assert "Time" not in result  # Identity elided

    def test_mixed_types_raises(self) -> None:
        """Test that composing different types on same dim raises error."""
        parent = {"Ch": {"scale": 2.0}}  # affine
        child = {"Ch": {"permutation": [1, 0]}}  # permutation
        with pytest.raises(ValueError, match="mixed types"):
            compose_nd_transforms(parent, child)


class TestApplyNdTransformToBounds:
    """Test apply_nd_transform_to_bounds function."""

    @pytest.fixture
    def dims_4d(self) -> Dimensions:
        return Dimensions(
            [
                Dimension("X", display=True),
                Dimension("Y", display=True),
                Dimension("Z", display=True),
                Dimension("Time", display=False, range=(0, 100)),
            ]
        )

    def test_affine_positive_scale(self, dims_4d) -> None:
        bounds = {"min": [0, 0, 0, 0], "max": [10, 10, 10, 50]}
        nd_t = {"Time": {"scale": 2.0, "offset": 10.0}}
        result = apply_nd_transform_to_bounds(bounds, nd_t, dims_4d)
        # Time: min = 2*0 + 10 = 10, max = 2*50 + 10 = 110
        assert result["min"][3] == 10.0
        assert result["max"][3] == 110.0
        # Spatial dims unchanged
        assert result["min"][:3] == [0, 0, 0]
        assert result["max"][:3] == [10, 10, 10]

    def test_affine_negative_scale(self, dims_4d) -> None:
        bounds = {"min": [0, 0, 0, 10], "max": [1, 1, 1, 50]}
        nd_t = {"Time": {"scale": -1.0, "offset": 100.0}}
        result = apply_nd_transform_to_bounds(bounds, nd_t, dims_4d)
        # Time: -1*10+100=90, -1*50+100=50 → min=50, max=90 (swapped)
        assert result["min"][3] == 50.0
        assert result["max"][3] == 90.0

    def test_no_transform_unchanged(self, dims_4d) -> None:
        bounds = {"min": [0, 0, 0, 0], "max": [10, 10, 10, 50]}
        result = apply_nd_transform_to_bounds(bounds, {}, dims_4d)
        assert result == bounds

    def test_mismatched_bounds_lengths_raise_value_error(self, dims_4d) -> None:
        # EN-2: silently skipping a mismatched bounds entry would yield mixed
        # transformed and untransformed bounds for the trailing dims, skewing
        # scene extents downstream. Fail fast so corrupt zarr metadata
        # surfaces clearly at the call site.
        bounds = {"min": [0, 0, 0, 0], "max": [10, 10, 10]}
        nd_t = {"Time": {"scale": 2.0, "offset": 10.0}}

        with pytest.raises(ValueError, match="Bounds length mismatch"):
            apply_nd_transform_to_bounds(bounds, nd_t, dims_4d)

    def test_bounds_shorter_than_dims_raise_value_error(self, dims_4d) -> None:
        # The companion case: bounds arrays are well-formed (equal lengths)
        # but cover fewer dims than the dimensions schema. Silently skipping
        # the missing dim would still produce mixed transformed/untransformed
        # bounds for any dim a downstream caller assumes is present.
        bounds = {"min": [0, 0, 0], "max": [10, 10, 10]}
        nd_t = {"Time": {"scale": 2.0, "offset": 10.0}}

        with pytest.raises(ValueError, match="out of range for bounds arrays"):
            apply_nd_transform_to_bounds(bounds, nd_t, dims_4d)


class TestNodeNdTransformIntegration:
    """Test nd_transform integration with Node/Scene/Compiler."""

    def test_nd_transform_property(self, tmp_path) -> None:
        """Test set/get/remove nd_transform on nodes."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            dims = Dimensions(
                [
                    Dimension("X", display=True),
                    Dimension("Y", display=True),
                    Dimension("Z", display=True),
                    Dimension("Time", display=False, range=(0, 100)),
                ]
            )
            scene = compiler.create_scene(dimensions=dims)
            group = scene.add_group("G")

            # No nd_transform initially
            assert group.nd_transform is None

            # Set it
            group.nd_transform = {"Time": {"offset": 50.0}}
            assert group.nd_transform == {"Time": {"offset": 50.0}}

            # Remove it
            group.nd_transform = None
            assert group.nd_transform is None

    def test_nd_transform_via_add_group(self, tmp_path) -> None:
        """Test nd_transform passed via add_group kwargs."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            dims = Dimensions(
                [
                    Dimension("X", display=True),
                    Dimension("Y", display=True),
                    Dimension("Z", display=True),
                    Dimension("Time", display=False, range=(0, 100)),
                ]
            )
            scene = compiler.create_scene(dimensions=dims)
            group = scene.add_group(
                "G",
                nd_transform={"Time": {"scale": 0.001, "offset": 50.0}},
            )
            assert group.nd_transform == {"Time": {"scale": 0.001, "offset": 50.0}}

    def test_nd_transform_persists_to_zarr(self, tmp_path) -> None:
        """Test nd_transform is persisted to zarr and survives reopen."""
        store_path = tmp_path / "test.luxar.zarr"
        nd_t = {"Time": {"scale": 0.001, "offset": 50.0}}

        with LuxarZarrCompiler(store_path) as compiler:
            dims = Dimensions(
                [
                    Dimension("X", display=True),
                    Dimension("Y", display=True),
                    Dimension("Z", display=True),
                    Dimension("Time", display=False, range=(0, 100)),
                ]
            )
            scene = compiler.create_scene(dimensions=dims)
            scene.add_group("G", nd_transform=nd_t)

        # Reopen and verify
        reader = LuxarScene.load(store_path)
        group_meta = reader.get_group("G")
        assert "nd_transform" in group_meta
        assert group_meta["nd_transform"] == nd_t

    def test_world_nd_transform(self, tmp_path) -> None:
        """Test hierarchical composition via world_nd_transform."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            dims = Dimensions(
                [
                    Dimension("X", display=True),
                    Dimension("Y", display=True),
                    Dimension("Z", display=True),
                    Dimension("Time", display=False, range=(0, 100)),
                ]
            )
            scene = compiler.create_scene(dimensions=dims)

            g1 = scene.add_group("L1", nd_transform={"Time": {"offset": 100.0}})
            g2 = g1.add_group("L2", nd_transform={"Time": {"scale": 2.0}})
            g3 = g2.add_group("L3", nd_transform={"Time": {"offset": 5.0}})

            # g3 world: root → L1(offset=100) → L2(scale=2) → L3(offset=5)
            # composed: x → 2(x+5) + 100 = 2x + 110
            world = g3.world_nd_transform
            assert world["Time"]["scale"] == 2.0
            assert world["Time"]["offset"] == 110.0

            # g1 has only its own
            assert g1.world_nd_transform == {"Time": {"offset": 100.0}}

            # Node with no nd_transform → empty dict
            g_plain = scene.add_group("Plain")
            assert g_plain.world_nd_transform == {}

    def test_nd_transform_on_points(self, tmp_path) -> None:
        """Test nd_transform stored on points node."""
        store_path = tmp_path / "test.luxar.zarr"
        with LuxarZarrCompiler(store_path) as compiler:
            dims = Dimensions(
                [
                    Dimension("X", display=True),
                    Dimension("Y", display=True),
                    Dimension("Z", display=True),
                    Dimension("Time", display=False, range=(0, 100)),
                ]
            )
            scene = compiler.create_scene(dimensions=dims)
            positions = np.random.rand(100, 4).astype(np.float32)
            positions[:, 3] *= 50  # Time in [0, 50]
            scene.add_points(
                "pts",
                positions,
                nd_transform={"Time": {"offset": 25.0}},
            )

        reader = LuxarScene.load(store_path)
        nodes = reader.nodes
        pts_node = next(n for n in nodes if n["name"] == "pts")
        assert "nd_transform" in pts_node
        assert pts_node["nd_transform"] == {"Time": {"offset": 25.0}}
