"""Tests for scene-level dimensions."""

import numpy as np
import pytest

from luxar.dimensions import Dimension, Dimensions


class TestDimension:
    """Test the Dimension class."""

    def test_dimension_creation(self) -> None:
        """Test creating dimensions with various parameters."""
        # Basic dimension
        dim = Dimension("x", unit="um")
        assert dim.name == "x"
        assert dim.unit == "um"
        assert dim.display is True  # Default
        assert dim.scale == 1.0

        # Full specification
        dim = Dimension(
            "time",
            unit="s",
            range=(0, 10),
            step=0.5,
            display=False,
            discrete=True,
            cyclic=False,
            scale=2.0,
            description="Time dimension",
        )
        assert dim.name == "time"
        assert dim.range == (0, 10)
        assert dim.step == 0.5
        assert dim.display is False
        assert dim.discrete is True
        assert dim.description == "Time dimension"

    @pytest.mark.parametrize(
        "kwargs,error_pattern,test_id",
        [
            # Range validation
            ({"range": [0, 1, 2]}, "Range must be a tuple", "range_not_tuple"),
            ({"range": (10, 5)}, "min must be less than max", "range_min_gt_max"),
            ({"range": (5, 5)}, "min must be less than max", "range_equal"),
            # Step validation
            ({"step": -1}, "Step size must be positive", "step_negative"),
            ({"step": 0}, "Step size must be positive", "step_zero"),
            # Scale validation
            ({"scale": 0}, "Scale must be positive", "scale_zero"),
            ({"scale": -1}, "Scale must be positive", "scale_negative"),
        ],
        ids=lambda x: x if isinstance(x, str) else None,
    )
    def test_dimension_validation(self, kwargs, error_pattern, test_id) -> None:
        """Test dimension parameter validation with various invalid inputs."""
        with pytest.raises(ValueError, match=error_pattern):
            Dimension("x", **kwargs)

    @pytest.mark.parametrize(
        "kwargs,expected_step,test_id",
        [
            # Explicit step takes precedence
            ({"step": 0.25}, 0.25, "explicit_step"),
            ({"step": 5.0}, 5.0, "explicit_step_large"),
            # Discrete dimensions default to 1.0
            ({"discrete": True}, 1.0, "discrete_default"),
            ({"discrete": True, "step": 2.0}, 2.0, "discrete_explicit_step"),
            # Auto-calculated from range (1% of range)
            ({"range": (0, 100)}, 1.0, "range_100"),
            ({"range": (0, 1000)}, 10.0, "range_1000"),
            ({"range": (-50, 50)}, 1.0, "range_symmetric"),
            # Default fallback when no step, discrete, or range
            ({}, 0.1, "default_fallback"),
        ],
        ids=lambda x: x if isinstance(x, str) else None,
    )
    def test_get_step(self, kwargs, expected_step, test_id) -> None:
        """Test automatic step calculation with various configurations."""
        dim = Dimension("x", **kwargs)
        assert dim.get_step() == expected_step

    def test_dimension_serialization(self) -> None:
        """Test to_dict and from_dict."""
        dim = Dimension(
            "time", unit="ms", range=(0, 1000), step=10, display=False, discrete=True
        )

        # Serialize
        data = dim.to_dict()
        assert data["name"] == "time"
        assert data["unit"] == "ms"
        assert data["range"] == [0, 1000]
        assert data["step"] == 10
        assert data["display"] is False

        # Deserialize
        dim2 = Dimension.from_dict(data)
        assert dim2.name == dim.name
        assert dim2.unit == dim.unit
        assert dim2.range == dim.range
        assert dim2.step == dim.step
        assert dim2.display == dim.display

    # [Python-R1/dims-G4] Full-fidelity Dimension to_dict / from_dict
    # round-trip. The existing test above only verifies a subset of
    # fields; a mutation in from_dict that dropped `discrete`, `cyclic`,
    # `scale`, `spatial`, or `description` would silently lose metadata
    # for every scene that touched the affected dimension.
    def test_dimension_full_field_roundtrip(self) -> None:
        original = Dimension(
            name="theta",
            unit="rad",
            range=(0.0, 6.283185307),
            step=0.01,
            display=True,
            discrete=False,
            cyclic=True,
            scale=2.5,
            spatial=True,
            description="azimuthal angle around z-axis",
        )

        recovered = Dimension.from_dict(original.to_dict())
        # Pin every field individually so a single dropped key fails
        # with an obvious message rather than a vague dict-mismatch.
        assert recovered.name == original.name
        assert recovered.unit == original.unit
        assert recovered.range == original.range
        assert recovered.step == original.step
        assert recovered.display == original.display
        assert recovered.discrete == original.discrete
        assert recovered.cyclic == original.cyclic
        assert recovered.scale == original.scale
        assert recovered.spatial == original.spatial
        assert recovered.description == original.description

        # And the full dict round-trip identity (catches added-fields
        # drift in either to_dict OR from_dict).
        assert original.to_dict() == recovered.to_dict()


class TestDimensions:
    """Test the Dimensions container class."""

    def test_dimensions_creation(self) -> None:
        """Test creating dimension collections."""
        dims = Dimensions([Dimension("x"), Dimension("y"), Dimension("z")])
        assert dims.ndim == 3
        assert dims.names == ["x", "y", "z"]
        assert dims.displayed == [0, 1, 2]
        assert dims.non_displayed == []

    @pytest.mark.parametrize(
        "dims_factory,error_pattern,test_id",
        [
            # Duplicate names
            (
                lambda: [Dimension("x"), Dimension("x")],
                "must be unique",
                "duplicate_names",
            ),
            (
                lambda: [Dimension("a"), Dimension("b"), Dimension("a")],
                "must be unique",
                "duplicate_names_three",
            ),
            # Too many displayed (max 3)
            (
                lambda: [
                    Dimension("x", display=True),
                    Dimension("y", display=True),
                    Dimension("z", display=True),
                    Dimension("t", display=True),
                ],
                "Maximum 3 dimensions",
                "four_displayed",
            ),
            (
                lambda: [
                    Dimension("a", display=True),
                    Dimension("b", display=True),
                    Dimension("c", display=True),
                    Dimension("d", display=True),
                    Dimension("e", display=True),
                ],
                "Maximum 3 dimensions",
                "five_displayed",
            ),
            # None displayed
            (
                lambda: [Dimension("x", display=False), Dimension("y", display=False)],
                "At least one dimension",
                "none_displayed",
            ),
            (
                lambda: [Dimension("a", display=False)],
                "At least one dimension",
                "single_not_displayed",
            ),
        ],
        ids=lambda x: x if isinstance(x, str) else None,
    )
    def test_dimensions_validation(self, dims_factory, error_pattern, test_id) -> None:
        """Test dimensions validation rules with various invalid configurations."""
        with pytest.raises(ValueError, match=error_pattern):
            Dimensions(dims_factory())

    def test_dimension_access(self) -> None:
        """Test accessing dimensions."""
        dims = Dimensions(
            [
                Dimension("time", unit="s", display=False),
                Dimension("x", unit="px"),
                Dimension("y", unit="px"),
            ]
        )

        # Get by name
        time_dim = dims.get_dimension("time")
        assert time_dim is not None
        assert time_dim.unit == "s"

        # Get index
        assert dims.get_index("time") == 0
        assert dims.get_index("y") == 2

        # Non-existent
        assert dims.get_dimension("z") is None
        with pytest.raises(ValueError, match="not found"):
            dims.get_index("z")

    def test_position_validation_valid(self) -> None:
        """Test that valid positions pass validation."""
        dims = Dimensions(
            [Dimension("x", range=(-10, 10)), Dimension("y", range=(-5, 5))]
        )
        # Valid positions - should not raise
        positions = np.array([[0, 0], [5, 2], [-5, -2]])
        dims.validate_positions(positions)

    @pytest.mark.parametrize(
        "positions_factory,error_pattern,test_id",
        [
            # Wrong shape (1D instead of 2D)
            (lambda: np.array([1, 2, 3]), "must be a 2D array", "shape_1d"),
            (lambda: np.array(5), "must be a 2D array", "shape_scalar"),
            # Wrong number of dimensions (3 instead of 2)
            (lambda: np.array([[1, 2, 3]]), "has 3 dimensions", "ndim_mismatch_3"),
            (lambda: np.array([[1]]), "has 1 dimensions", "ndim_mismatch_1"),
            # Out of range values
            (lambda: np.array([[15, 0]]), "outside range", "out_of_range_x_high"),
            (lambda: np.array([[-15, 0]]), "outside range", "out_of_range_x_low"),
            (lambda: np.array([[0, 10]]), "outside range", "out_of_range_y_high"),
            (lambda: np.array([[0, -10]]), "outside range", "out_of_range_y_low"),
        ],
        ids=lambda x: x if isinstance(x, str) else None,
    )
    def test_position_validation_invalid(
        self, positions_factory, error_pattern, test_id
    ) -> None:
        """Test position validation with various invalid inputs."""
        dims = Dimensions(
            [Dimension("x", range=(-10, 10)), Dimension("y", range=(-5, 5))]
        )
        with pytest.raises(ValueError, match=error_pattern):
            dims.validate_positions(positions_factory())

    def test_dimensions_serialization(self) -> None:
        """Test serialization of dimension collections."""
        dims = Dimensions(
            [
                Dimension("t", unit="s", display=False),
                Dimension("x", unit="m"),
                Dimension("y", unit="m"),
            ]
        )

        # Serialize
        data = dims.to_dict()
        assert len(data["dimensions"]) == 3
        assert data["dimensions"][0]["name"] == "t"

        # Deserialize
        dims2 = Dimensions.from_dict(data)
        assert dims2.ndim == 3
        assert dims2.names == ["t", "x", "y"]
        assert dims2.displayed == [1, 2]

    def test_convenience_constructors(self) -> None:
        """Test convenience constructor methods."""
        # 2D
        dims = Dimensions.default_2d()
        assert dims.ndim == 2
        assert dims.names == ["x", "y"]
        assert len(dims.displayed) == 2

        # 3D
        dims = Dimensions.default_3d()
        assert dims.ndim == 3
        assert dims.names == ["x", "y", "z"]
        assert len(dims.displayed) == 3

        # Time series
        dims = Dimensions.default_timeseries(n_timepoints=100)
        assert dims.ndim == 4
        assert dims.names == ["t", "x", "y", "z"]
        assert dims.displayed == [1, 2, 3]  # t is hidden
        assert dims.dimensions[0].discrete is True

        # Multichannel
        dims = Dimensions.default_multichannel(n_channels=5)
        assert dims.ndim == 4
        assert dims.names == ["c", "x", "y", "z"]
        assert dims.dimensions[0].range == (0, 4)

    def test_from_positions(self) -> None:
        """Test inferring dimensions from positions."""
        # 3D positions
        positions = np.random.randn(100, 3)
        dims = Dimensions.from_positions(positions)
        assert dims.ndim == 3
        assert dims.names == ["x", "y", "z"]
        assert all(d.range is None for d in dims.dimensions)  # No ranges set

        # 5D with custom names
        positions = np.random.randn(50, 5)
        dims = Dimensions.from_positions(positions, names=["t", "x", "y", "z", "c"])
        assert dims.names == ["t", "x", "y", "z", "c"]
        assert dims.displayed == [0, 1, 2]  # First 3 dimensions are displayed

        # Invalid inputs
        with pytest.raises(ValueError, match="must be 2D array"):
            Dimensions.from_positions(np.array([1, 2, 3]))

        with pytest.raises(ValueError, match="Got 2 names for 3 dimensions"):
            Dimensions.from_positions(np.random.randn(10, 3), names=["x", "y"])


class TestCategoricalDimensions:
    """Test categorical dimension support."""

    def test_categorical_dimension_creation(self) -> None:
        """Test creating a categorical dimension with categories."""
        dim = Dimension(
            "channel",
            categories=["DAPI", "GFP", "mCherry"],
            display=False,
        )

        # Auto-behaviors
        assert dim.is_categorical is True
        assert dim.discrete is True  # Auto-set
        assert dim.range == (0, 2)  # Auto-set from len(categories)
        assert dim.step == 1.0  # Auto-set for categories
        assert dim.categories == ["DAPI", "GFP", "mCherry"]

    def test_categorical_dimension_serialization(self) -> None:
        """Test serialization/deserialization of categorical dimension."""
        dim = Dimension(
            "channel",
            categories=["DAPI", "GFP", "mCherry"],
            display=False,
            cyclic=True,
        )

        # Serialize
        data = dim.to_dict()
        assert data["categories"] == ["DAPI", "GFP", "mCherry"]
        assert data["discrete"] is True
        assert data["cyclic"] is True
        assert data["range"] == [0, 2]

        # Deserialize
        loaded = Dimension.from_dict(data)
        assert loaded.categories == ["DAPI", "GFP", "mCherry"]
        assert loaded.is_categorical is True
        assert loaded.discrete is True
        assert loaded.cyclic is True
        assert loaded.range == (0, 2)

    def test_categorical_with_explicit_range(self) -> None:
        """Test that explicit range overrides auto-calculation."""
        dim = Dimension(
            "channel",
            categories=["A", "B", "C"],
            range=(0, 5),  # Explicit range
            display=False,
        )
        # Explicit range should be respected
        assert dim.range == (0, 5)

    def test_non_categorical_dimension(self) -> None:
        """Test that non-categorical dimensions don't have categories."""
        dim = Dimension("x", unit="um", display=True)
        assert dim.is_categorical is False
        assert dim.categories is None

    @pytest.mark.parametrize("display", [True, False])
    def test_single_category_zero_width_range(self, display: bool) -> None:
        """Regression (#755): a single category must not raise.

        The auto-range for one category collapses to (0, 0); categorical
        dimensions are allowed a zero-width range (MIN_CATEGORIES == 1).
        """
        dim = Dimension("channel", categories=["DAPI"], display=display)
        assert dim.range == (0, 0)
        assert dim.discrete is True
        assert dim.is_categorical is True

    def test_two_category_range_unchanged(self) -> None:
        """Guard against regression: two categories still yield range (0, 1)."""
        dim = Dimension("channel", categories=["DAPI", "GFP"], display=False)
        assert dim.range == (0, 1)

    def test_inverted_categorical_range_rejected(self) -> None:
        """An explicitly inverted categorical range must still raise."""
        with pytest.raises(ValueError, match="min must not be greater than max"):
            Dimension("x", categories=["a", "b"], range=(2, 0), display=False)

    def test_degenerate_continuous_range_rejected(self) -> None:
        """A zero-width continuous range must still raise (no categories)."""
        with pytest.raises(ValueError, match="min must be less than max"):
            Dimension("x", range=(5.0, 5.0))

    @pytest.mark.parametrize(
        "categories,error_pattern,test_id",
        [
            # Empty categories list
            ([], "must have at least 1 element", "empty_list"),
            # Duplicate categories
            (["A", "B", "A"], "duplicate category name", "duplicate_ABA"),
            (["X", "X"], "duplicate category name", "duplicate_XX"),
            (["foo", "bar", "baz", "foo"], "duplicate category name", "duplicate_last"),
            # Empty string category
            (["A", "", "C"], "is empty string", "empty_middle"),
            (["", "B", "C"], "is empty string", "empty_first"),
            (["A", "B", ""], "is empty string", "empty_last"),
        ],
        ids=lambda x: x if isinstance(x, str) else None,
    )
    def test_invalid_categories_rejected(
        self, categories, error_pattern, test_id
    ) -> None:
        """Test that various invalid category configurations are rejected."""
        with pytest.raises(ValueError, match=error_pattern):
            Dimension("channel", categories=categories, display=False)

    def test_categorical_in_dimensions_collection(self) -> None:
        """Test categorical dimension in a Dimensions collection."""
        dims = Dimensions(
            [
                Dimension(
                    "channel",
                    categories=["DAPI", "GFP"],
                    display=False,
                ),
                Dimension("x", unit="um", display=True),
                Dimension("y", unit="um", display=True),
                Dimension("z", unit="um", display=True),
            ]
        )

        assert dims.ndim == 4
        assert dims.dimensions[0].is_categorical is True
        assert dims.dimensions[0].categories == ["DAPI", "GFP"]

        # Serialize and reload
        data = dims.to_dict()
        loaded = Dimensions.from_dict(data)
        assert loaded.dimensions[0].is_categorical is True
        assert loaded.dimensions[0].categories == ["DAPI", "GFP"]
