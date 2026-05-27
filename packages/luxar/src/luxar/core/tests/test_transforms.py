"""
Test transform utilities and functionality.
"""

import numpy as np
import pytest

from luxar import Dimensions, LuxarZarrCompiler
from luxar.transforms import (
    compose,
    from_list,
    identity,
    inverse,
    look_at,
    rotate,
    rotate_x,
    rotate_y,
    rotate_z,
    scale,
    to_list,
    translate,
)


class TestTransformUtilities:
    """Test transform utility functions."""

    def test_identity(self) -> None:
        """Test identity transform creation."""
        t = identity()
        assert t.shape == (4, 4)
        assert t.dtype == np.float32
        assert np.allclose(t, np.eye(4))

    def test_translate(self) -> None:
        """Test translation transform."""
        # Test individual axes
        t = translate(x=5)
        assert np.allclose(t[0, 3], 5)
        assert np.allclose(t[1, 3], 0)
        assert np.allclose(t[2, 3], 0)

        # Test all axes
        t = translate(1, 2, 3)
        assert np.allclose(t[:3, 3], [1, 2, 3])

        # Check rest of matrix is identity
        assert np.allclose(t[:3, :3], np.eye(3))
        assert t[3, 3] == 1

    def test_scale(self) -> None:
        """Test scaling transform."""
        # Non-uniform scale
        t = scale(2, 3, 4)
        assert np.allclose(np.diag(t), [2, 3, 4, 1])

        # Uniform scale
        t = scale(uniform=2.5)
        assert np.allclose(np.diag(t), [2.5, 2.5, 2.5, 1])

        # Check no other elements
        t[np.diag_indices(4)] = 0
        assert np.allclose(t, 0)

    # Audit G2 (python-core-validation, P5): degenerate-scale boundary tests.
    @pytest.mark.parametrize(
        "uniform,expected_diag,test_id",
        [
            # Zero scale: degenerate (non-invertible) but the function should
            # still return a well-defined diag matrix with 0s in xyz.
            (0.0, [0, 0, 0, 1], "uniform_zero"),
            # Negative scale: reflection through origin.
            (-1.0, [-1, -1, -1, 1], "uniform_neg_one"),
            # Very small positive: near-zero edge.
            (1e-10, [1e-10, 1e-10, 1e-10, 1], "uniform_tiny"),
            # Very large positive.
            (1e6, [1e6, 1e6, 1e6, 1], "uniform_huge"),
        ],
        ids=lambda x: x if isinstance(x, str) else None,
    )
    def test_scale_boundary_values(
        self, uniform: float, expected_diag, test_id: str
    ) -> None:
        """Boundary uniform scale values (0, ±1, tiny, huge) must produce
        a well-formed 4x4 diagonal matrix without raising.
        Inversion of a zero-scale matrix is undefined — that contract is
        covered by `test_inverse_non_invertible` below.
        """
        t = scale(uniform=uniform)
        assert t.shape == (4, 4)
        assert np.allclose(np.diag(t), expected_diag)
        # Off-diagonal must remain zero.
        t_no_diag = t.copy()
        t_no_diag[np.diag_indices(4)] = 0
        assert np.allclose(t_no_diag, 0)

    def test_rotate_zero_degrees_is_identity(self) -> None:
        """Audit G6: rotate(0°, any axis) must return the identity matrix.

        Mutants that drop the `theta == 0` short-circuit or compute
        `sin/cos` at 0 with float drift could surface as small but
        non-zero off-diagonal entries.
        """
        for axis in ("x", "y", "z"):
            t = rotate(0, axis)
            assert np.allclose(t, identity(), atol=1e-15), (
                f"rotate(0, {axis!r}) drifted from identity"
            )

        # And the arbitrary-axis variant — rotate(0, [1, 2, 3]) must also
        # be identity regardless of axis magnitude.
        t = rotate(0, np.array([1.0, 2.0, 3.0]))
        assert np.allclose(t, identity(), atol=1e-15)

    @pytest.mark.parametrize(
        "rotation_func,angle,input_axis,expected_output,test_id",
        [
            # X-axis rotations: Y -> Z, Z -> -Y
            (rotate_x, 90, [0, 1, 0, 0], [0, 0, 1], "x_90_y_to_z"),
            (rotate_x, 180, [0, 1, 0, 0], [0, -1, 0], "x_180_y_to_neg_y"),
            (rotate_x, 90, [0, 0, 1, 0], [0, -1, 0], "x_90_z_to_neg_y"),
            (rotate_x, 270, [0, 1, 0, 0], [0, 0, -1], "x_270_y_to_neg_z"),
            # Y-axis rotations: Z -> X, X -> -Z
            (rotate_y, 90, [0, 0, 1, 0], [1, 0, 0], "y_90_z_to_x"),
            (rotate_y, 180, [1, 0, 0, 0], [-1, 0, 0], "y_180_x_to_neg_x"),
            (rotate_y, 90, [1, 0, 0, 0], [0, 0, -1], "y_90_x_to_neg_z"),
            # Z-axis rotations: X -> Y, Y -> -X
            (rotate_z, 90, [1, 0, 0, 0], [0, 1, 0], "z_90_x_to_y"),
            (rotate_z, 180, [1, 0, 0, 0], [-1, 0, 0], "z_180_x_to_neg_x"),
            (rotate_z, 90, [0, 1, 0, 0], [-1, 0, 0], "z_90_y_to_neg_x"),
            (rotate_z, -90, [1, 0, 0, 0], [0, -1, 0], "z_neg90_x_to_neg_y"),
        ],
        ids=lambda x: x if isinstance(x, str) else None,
    )
    def test_axis_rotations(
        self, rotation_func, angle, input_axis, expected_output, test_id
    ) -> None:
        """Test axis rotations with various angles and inputs."""
        t = rotation_func(angle)
        rotated = t @ np.array(input_axis)
        assert np.allclose(rotated[:3], expected_output, atol=1e-6)

    def test_rotate_arbitrary_axis(self) -> None:
        """Test rotation around arbitrary axis."""
        # Rotate around diagonal axis
        axis = np.array([1, 1, 1])
        t = rotate(120, axis)

        # 120 degree rotation around (1,1,1) should cycle X->Y->Z->X
        x_axis = np.array([1, 0, 0, 0])
        rotated = t @ x_axis
        assert np.allclose(rotated[:3], [0, 1, 0], atol=1e-6)  # X -> Y

    # [Python-R1/transforms-CRIT] Zero-length axis previously produced
    # a NaN-filled rotation matrix that silently corrupted every scene
    # using the result. Pin the new ValueError, and the related case of
    # a tuple zero-vector input.
    def test_rotate_raises_on_zero_length_axis_array(self) -> None:
        with pytest.raises(ValueError, match="zero length"):
            rotate(90, np.array([0.0, 0.0, 0.0]))

    def test_rotate_raises_on_zero_length_axis_tuple(self) -> None:
        with pytest.raises(ValueError, match="zero length"):
            rotate(45, (0.0, 0.0, 0.0))

    @pytest.mark.parametrize(
        "axis,angle,expected_func,test_id",
        [
            ("x", 45, lambda: rotate_x(45), "axis_x_lower"),
            ("X", 45, lambda: rotate_x(45), "axis_x_upper"),
            ("y", 90, lambda: rotate_y(90), "axis_y_lower"),
            ("Y", 30, lambda: rotate_y(30), "axis_y_upper"),
            ("z", 180, lambda: rotate_z(180), "axis_z_lower"),
            ("Z", -45, lambda: rotate_z(-45), "axis_z_upper"),
        ],
        ids=lambda x: x if isinstance(x, str) else None,
    )
    def test_rotate_string_axis_valid(
        self, axis, angle, expected_func, test_id
    ) -> None:
        """Test rotation with valid string axis names."""
        assert np.allclose(rotate(angle, axis), expected_func())

    @pytest.mark.parametrize(
        "invalid_axis",
        ["w", "a", "xx", "xy", ""],
        ids=["w", "a", "xx", "xy", "empty"],
    )
    def test_rotate_string_axis_invalid(self, invalid_axis) -> None:
        """Test rotation with invalid string axis names."""
        with pytest.raises(ValueError):
            rotate(45, invalid_axis)

    def test_compose(self) -> None:
        """Test transform composition."""
        # Test that compose works correctly
        t1 = translate(5, 0, 0)
        t2 = rotate_z(90)

        # Compose multiple transforms
        combined = compose(t1, t2)

        # Test on a simple point
        point = np.array([1, 0, 0, 1])
        result = combined @ point

        # Just verify the compose function runs without error
        assert result.shape == (4,)

        # Empty compose
        assert np.allclose(compose(), identity())

        # Single transform
        assert np.allclose(compose(t1), t1)

    # [Python-R1/transforms-CRIT] Property tests for the three algebraic
    # contracts compose() MUST satisfy: identity, associativity, and the
    # documented application-order semantics. The existing example tests
    # cover specific fixed orderings; a mutation that subtly broke
    # associativity (e.g., inverted the `reversed(transforms)` walk) could
    # still pass them.
    @pytest.mark.parametrize("seed", [0, 1, 7, 42, 137])
    def test_compose_identity_law(self, seed: int) -> None:
        """compose(I, T) == compose(T, I) == T for every T."""
        rng = np.random.default_rng(seed)
        t = compose(
            translate(*rng.uniform(-5, 5, 3)),
            rotate_z(rng.uniform(-180, 180)),
            scale(*rng.uniform(0.5, 2.0, 3)),
        )
        assert np.allclose(compose(identity(), t), t, atol=1e-6)
        assert np.allclose(compose(t, identity()), t, atol=1e-6)

    @pytest.mark.parametrize("seed", [0, 1, 7, 42, 137])
    def test_compose_associativity(self, seed: int) -> None:
        """compose(compose(T1, T2), T3) == compose(T1, compose(T2, T3))."""
        rng = np.random.default_rng(seed)
        t1 = translate(*rng.uniform(-5, 5, 3))
        t2 = rotate_z(rng.uniform(-180, 180))
        t3 = scale(*rng.uniform(0.5, 2.0, 3))

        left = compose(compose(t1, t2), t3)
        right = compose(t1, compose(t2, t3))
        # float32 accumulator → looser tolerance than default 1e-8
        assert np.allclose(left, right, atol=1e-5)

    @pytest.mark.parametrize("seed", [0, 1, 7, 42, 137])
    def test_compose_application_order_property(self, seed: int) -> None:
        """compose(T1, T2) applied to a point == T2(T1(point))."""
        rng = np.random.default_rng(seed)
        t1 = translate(*rng.uniform(-5, 5, 3))
        t2 = scale(*rng.uniform(0.5, 2.0, 3))
        point = np.array([*rng.uniform(-1, 1, 3), 1.0])

        combined = compose(t1, t2) @ point
        sequential = t2 @ (t1 @ point)
        assert np.allclose(combined, sequential, atol=1e-5)

    def test_compose_application_order(self) -> None:
        """Test that compose(T1, T2, T3) applies T1 first, then T2, then T3.

        This is critical because matrix multiplication is right-to-left,
        so compose(T1, T2) must produce T2 @ T1 (not T1 @ T2).
        """
        # Test Case 1: Translate then Scale (non-commutative for non-origin points)
        t1 = translate(10, 0, 0)  # Move right 10 units
        t2 = scale(2, 2, 2)  # Scale by 2x

        point = np.array([0, 0, 0, 1])

        # Manual application in order
        p1 = t1 @ point  # After T1: (10, 0, 0)
        expected = t2 @ p1  # After T2: (20, 0, 0)

        # Using compose
        combined = compose(t1, t2)
        result = combined @ point

        assert np.allclose(result[:3], expected[:3]), (
            f"Expected {expected[:3]}, got {result[:3]}"
        )

        # Test Case 2: Rotate then Translate (order really matters!)
        t1 = rotate_z(90)  # Rotate 90° around Z
        t2 = translate(10, 0, 0)  # Move right 10 units

        point = np.array([1, 0, 0, 1])  # Point on X-axis

        # Manual: Rotate (1,0,0)→(0,1,0), then translate→(10,1,0)
        p1 = t1 @ point  # Rotated to Y-axis: (~0, 1, 0)
        expected = t2 @ p1  # Translated: (10, 1, 0)

        combined = compose(t1, t2)
        result = combined @ point

        assert np.allclose(result[:3], expected[:3], atol=1e-6), (
            f"Rotate then translate failed: expected {expected[:3]}, got {result[:3]}"
        )

        # Test Case 3: Opposite order gives different result
        # Translate then Rotate should give (0, 11, 0) not (10, 1, 0)
        t1 = translate(10, 0, 0)  # Move right first
        t2 = rotate_z(90)  # Then rotate

        point = np.array([1, 0, 0, 1])

        # Manual: Translate (1,0,0)→(11,0,0), then rotate→(0,11,0)
        p1 = t1 @ point
        expected = t2 @ p1

        combined = compose(t1, t2)
        result = combined @ point

        assert np.allclose(result[:3], expected[:3], atol=1e-6), (
            f"Translate then rotate failed: expected {expected[:3]}, got {result[:3]}"
        )

        # Verify they're different (order matters!)
        combined_reverse = compose(t2, t1)
        result_reverse = combined_reverse @ point
        assert not np.allclose(result[:3], result_reverse[:3], atol=0.1), (
            "Order should matter for non-commutative transforms!"
        )

        # Test Case 4: Three transforms
        t1 = translate(5, 0, 0)
        t2 = rotate_z(90)
        t3 = scale(2, 2, 2)

        point = np.array([1, 0, 0, 1])

        # Manual application: T1, then T2, then T3
        p1 = t1 @ point  # (6, 0, 0)
        p2 = t2 @ p1  # (0, 6, 0)
        expected = t3 @ p2  # (0, 12, 0)

        combined = compose(t1, t2, t3)
        result = combined @ point

        assert np.allclose(result[:3], expected[:3], atol=1e-6), (
            f"Three-transform composition failed: expected {expected[:3]}, got {result[:3]}"
        )

    @pytest.mark.parametrize(
        "transform_factory,test_id",
        [
            # Translation inverses
            (lambda: translate(5, 3, -2), "translate_xyz"),
            (lambda: translate(0, 0, 10), "translate_z_only"),
            (lambda: translate(-100, 50, 0), "translate_large"),
            # Rotation inverses
            (lambda: rotate_z(45), "rotate_z_45"),
            (lambda: rotate_x(90), "rotate_x_90"),
            (lambda: rotate_y(180), "rotate_y_180"),
            (lambda: rotate(30, "x"), "rotate_x_30"),
            # Scale inverses
            (lambda: scale(2, 4, 0.5), "scale_non_uniform"),
            (lambda: scale(uniform=3.0), "scale_uniform"),
            (lambda: scale(0.1, 0.1, 0.1), "scale_small"),
            # Combined transforms
            (lambda: compose(translate(5, 0, 0), rotate_z(45)), "translate_rotate"),
            (lambda: compose(rotate_x(30), scale(2, 2, 2)), "rotate_scale"),
        ],
        ids=lambda x: x if isinstance(x, str) else None,
    )
    def test_inverse_valid(self, transform_factory, test_id) -> None:
        """Test that inverse(T) @ T = identity for various transforms."""
        t = transform_factory()
        t_inv = inverse(t)
        assert np.allclose(compose(t, t_inv), identity(), atol=1e-6)

    def test_inverse_non_invertible(self) -> None:
        """Test that non-invertible matrices raise ValueError."""
        bad = np.zeros((4, 4), dtype=np.float32)
        with pytest.raises(ValueError):
            inverse(bad)

    def test_look_at(self) -> None:
        """Test look-at transform."""
        # Look at origin from position
        t = look_at((10, 5, 10), (0, 0, 0))

        # Position should be set
        assert np.allclose(t[:3, 3], [10, 5, 10])

        # Forward vector should point toward target
        forward = -t[2, :3]  # Negative Z is forward
        expected_forward = np.array([-10, -5, -10])
        expected_forward = expected_forward / np.linalg.norm(expected_forward)
        assert np.allclose(forward, expected_forward)

        # Custom up vector
        t = look_at((1, 0, 0), (0, 0, 0), up=(0, 0, 1))
        # Y axis should be close to Z (up)
        assert t[1, 2] > 0.9  # Y points mostly in Z direction

    # [Python-R1/transforms-CRIT] look_at degenerate inputs previously
    # produced NaN-filled matrices that propagated downstream (cameras
    # silently render nothing). Pin the new actionable errors.
    def test_look_at_raises_when_eye_equals_target(self) -> None:
        with pytest.raises(ValueError, match="coincident"):
            look_at((5, 5, 5), (5, 5, 5))

    def test_look_at_raises_when_up_parallel_to_forward(self) -> None:
        # forward direction is +Y, up is also +Y → cross product is zero
        with pytest.raises(ValueError, match="parallel"):
            look_at((0, 0, 0), (0, 1, 0), up=(0, 1, 0))

    def test_look_at_raises_when_up_antiparallel_to_forward(self) -> None:
        # forward is +Y, up is -Y → cross product is zero (still parallel)
        with pytest.raises(ValueError, match="parallel"):
            look_at((0, 0, 0), (0, 1, 0), up=(0, -1, 0))

    def test_to_from_list(self) -> None:
        """Test conversion to/from list."""
        # Create a transform
        t = compose(translate(1, 2, 3), rotate_z(45), scale(2, 2, 2))

        # Convert to list and back
        values = to_list(t)
        assert isinstance(values, list)
        assert len(values) == 16

        t_recovered = from_list(values)
        assert np.allclose(t, t_recovered)

        # Test validation
        with pytest.raises(ValueError):
            from_list([1, 2, 3])  # Wrong size

    # [Python-R1/transforms-CRIT] The to_list / from_list pair is the
    # THREE.js bridge — NumPy stores 4x4 matrices row-major while THREE.js
    # is column-major, so the translation vector lives at indices
    # [0][3], [1][3], [2][3] in NumPy but at FLAT indices [12], [13], [14]
    # in the THREE.js list (post-transpose). The round-trip identity test
    # above hides this column/row pivot: a regression that dropped the
    # transpose on BOTH sides would still pass it. Pin the indexing
    # contract explicitly with a known translation; this kills mutations
    # that touch only one side of the transpose.
    def test_to_list_places_translation_at_three_js_indices_12_13_14(self) -> None:
        t = translate(5, 7, 11)
        values = to_list(t)
        assert values[12] == 5.0
        assert values[13] == 7.0
        assert values[14] == 11.0
        # The NumPy-row-major translation slot [3][3] (== flat index 15)
        # is the bottom-right `1`; pin it so a mutation that ravelled
        # without the transpose (NumPy [0][3]=5 → flat [3]) would also
        # fail this assertion.
        assert values[15] == 1.0
        assert values[3] == 0.0  # NOT 5 — this is the dead-give-away of a missing transpose

    def test_from_list_reads_translation_from_three_js_indices_12_13_14(self) -> None:
        # Build a list with translation at THREE.js indices [12, 13, 14]
        # and verify from_list places them at NumPy row-major [0][3],
        # [1][3], [2][3].
        values = [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            5, 7, 11, 1,   # translation row in THREE.js column-major flat
        ]
        m = from_list(values)
        assert m[0, 3] == 5.0
        assert m[1, 3] == 7.0
        assert m[2, 3] == 11.0
        assert m[3, 3] == 1.0

    def test_aliases(self) -> None:
        """Test function aliases."""
        from luxar.transforms import rotation, scaling, translation

        assert np.allclose(translation(1, 2, 3), translate(1, 2, 3))
        assert np.allclose(scaling(2, 2, 2), scale(2, 2, 2))
        assert np.allclose(rotation(45, "z"), rotate(45, "z"))


class TestNodeTransformIntegration:
    """Test transform integration with Node class."""

    def test_node_transform_validation(self, tmp_path) -> None:
        """Test that transforms are validated when creating nodes."""
        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())

            # Valid transform - pass numpy matrix directly (row-major)
            t = translate(5, 0, 0)
            group = scene.add_group("ValidTransform", transform=t)
            assert "transform" in group.attrs

            # Invalid transform - wrong size
            with pytest.raises(ValueError):
                scene.add_group("BadSize", transform=[1, 2, 3])

            # Invalid transform - not a list/array
            with pytest.raises(ValueError):
                scene.add_group("BadType", transform="not a transform")

    def test_node_transform_property(self, tmp_path) -> None:
        """Test the transform property on nodes."""
        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("TestGroup")

            # No transform initially
            assert group.transform is None

            # Set transform using property
            t = compose(translate(1, 2, 3), rotate_x(45))
            group.transform = t

            # Get transform
            retrieved = group.transform
            assert retrieved is not None
            assert np.allclose(retrieved, t)

            # Set from numpy matrix
            group.transform = scale(2, 2, 2)
            assert np.allclose(group.transform, scale(2, 2, 2))

            # Remove transform
            group.transform = None
            assert group.transform is None
            assert "transform" not in group.attrs

    def test_nested_transforms(self, tmp_path) -> None:
        """Test nested transform hierarchy."""
        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())

            # Create hierarchy with transforms (pass numpy matrices)
            g1 = scene.add_group("Level1", transform=translate(5, 0, 0))
            g2 = g1.add_group("Level2", transform=rotate_z(45))
            g3 = g2.add_group("Level3", transform=scale(2, 2, 2))

            # Verify each has its own transform
            assert g1.transform is not None
            assert g2.transform is not None
            assert g3.transform is not None
            assert np.allclose(g1.transform, translate(5, 0, 0))
            assert np.allclose(g2.transform, rotate_z(45))
            assert np.allclose(g3.transform, scale(2, 2, 2))

    def test_transform_with_points(self, tmp_path) -> None:
        """Test transforms work with points."""
        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())

            # Create transformed group (pass numpy matrix directly)
            t = compose(translate(10, 0, 0), scale(uniform=0.5))
            group = scene.add_group("TransformedPoints", transform=t)

            # Add points
            positions = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
            colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
            scene.add_points("Points", positions, colors=colors, parent=group)

            # Verify structure
            assert group.transform is not None
            assert len(group.children) == 1
            assert group.children[0].name == "Points"

    def test_world_transform(self, tmp_path) -> None:
        """Test world_transform property composes parent chain correctly."""
        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())

            # Node with no transform → identity
            g_no_xform = scene.add_group("NoTransform")
            assert np.allclose(g_no_xform.world_transform, identity())

            # Single transform → equals local transform
            t1 = translate(5, 0, 0)
            g1 = scene.add_group("Level1", transform=t1)
            assert np.allclose(g1.world_transform, t1)

            # Chain: translate → rotate → scale
            t2 = rotate_z(45)
            t3 = scale(2, 2, 2)
            g2 = g1.add_group("Level2", transform=t2)
            g3 = g2.add_group("Level3", transform=t3)

            # world_transform = compose(root_first, ..., leaf_last)
            expected_g2 = compose(t1, t2)
            expected_g3 = compose(t1, t2, t3)
            assert np.allclose(g2.world_transform, expected_g2, atol=1e-6)
            assert np.allclose(g3.world_transform, expected_g3, atol=1e-6)

            # Mixed: parent has no transform, child does
            g_plain = scene.add_group("Plain")
            g_child = g_plain.add_group("Child", transform=t1)
            assert np.allclose(g_child.world_transform, t1)

            # [Python-R4/transforms-MAJOR] Mixed hierarchy with a SKIPPED
            # level: grandparent has transform, parent has NONE, child
            # has transform. world_transform must compose grandparent +
            # child, treating the no-transform parent as identity. The
            # original test only covered "parent has none, child has X"
            # (depth 2) — the depth-3 skip is a structurally different
            # path through the recursive walk.
            g_gp = scene.add_group("Grandparent3", transform=t1)
            g_intermediate = g_gp.add_group("IntermediateNoXform")
            g_leaf = g_intermediate.add_group("Leaf", transform=t3)
            # compose(t1, identity, t3) == compose(t1, t3)
            assert np.allclose(g_leaf.world_transform, compose(t1, t3), atol=1e-6)
            # Intermediate's own world_transform should equal t1 (its
            # parent's) — the skipped level inherits without modification.
            assert np.allclose(g_intermediate.world_transform, t1, atol=1e-6)

    def test_world_transform_matches_manual_compose(self, tmp_path) -> None:
        """Test world_transform gives same result as manual point transformation."""
        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())

            t1 = translate(10, 0, 0)
            t2 = rotate_z(90)
            g1 = scene.add_group("Parent", transform=t1)
            g2 = g1.add_group("Child", transform=t2)

            # Apply world_transform to a point
            point = np.array([1, 0, 0, 1])
            world = g2.world_transform
            result = world @ point

            # Manual: translate first, then rotate
            p1 = t1 @ point  # (11, 0, 0, 1)
            expected = t2 @ p1  # (0, 11, 0, 1)
            assert np.allclose(result[:3], expected[:3], atol=1e-5)

    def test_transform_removal_persists_to_zarr(self, tmp_path) -> None:
        """Test that setting transform=None actually removes it from zarr store."""
        store_path = tmp_path / "test.zarr"
        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("TestGroup", transform=translate(1, 2, 3))

            # Verify transform is in cache
            assert "transform" in group.attrs

            # Remove it
            group.transform = None
            assert group.transform is None
            assert "transform" not in group.attrs

        # Reopen zarr store and verify attribute is gone from disk
        import zarr

        reopened = zarr.open_group(store_path, mode="r")
        assert "transform" not in reopened["TestGroup"].attrs

    def test_prepare_transform_roundtrip_from_list(self) -> None:
        """Test that row-major list input round-trips correctly."""
        from luxar.core.transforms import (
            prepare_transform_for_zarr,
            read_transform_from_zarr,
        )

        # Start with a numpy matrix
        original = compose(translate(1, 2, 3), rotate_z(45))

        # Convert to zarr format and back
        zarr_list = prepare_transform_for_zarr(original)
        recovered = read_transform_from_zarr(zarr_list)

        assert np.allclose(original, recovered, atol=1e-6)

        # Also test with a row-major flattened list
        flat_list = original.ravel().tolist()
        zarr_list2 = prepare_transform_for_zarr(flat_list)
        recovered2 = read_transform_from_zarr(zarr_list2)

        assert np.allclose(original, recovered2, atol=1e-6)
