"""Test physical units support in Luxar through Dimensions system."""

import pytest

from luxar import Dimension, Dimensions, LuxarZarrCompiler


class TestPhysicalUnits:
    """Test that all supported physical units work correctly through Dimensions."""

    @pytest.mark.parametrize(
        "unit",
        [
            "nm",  # nanometer
            "um",  # micrometer
            "mm",  # millimeter
            "cm",  # centimeter
            "m",  # meter (short form)
            "metre",  # meter (British spelling)
            "meter",  # meter (American spelling)
            "km",  # kilometer
            "inch",  # inch
            "foot",  # foot
            "px",  # pixel
            "au",  # arbitrary units
        ],
    )
    def test_unit_acceptance(self, unit, tmp_path) -> None:
        """Test that each physical unit is accepted in Dimensions."""
        scene_path = tmp_path / f"test_{unit}.luxar.zarr"

        # Create dimensions with this unit
        dims = Dimensions(
            [
                Dimension("x", unit=unit, display=True),
                Dimension("y", unit=unit, display=True),
                Dimension("z", unit=unit, display=True),
            ]
        )

        # Should not raise any errors
        with LuxarZarrCompiler(scene_path) as compiler:
            compiler.create_scene(dimensions=dims)

        # Check the dimensions were stored correctly
        import zarr

        store = zarr.open_group(scene_path, mode="r")
        assert "scene_dimensions" in store.attrs
        dims_data = store.attrs["scene_dimensions"]
        assert dims_data["dimensions"][0]["unit"] == unit

    def test_invalid_unit_rejection(self) -> None:
        """Test that invalid units are rejected in Dimensions."""
        # Invalid unit should be caught during Dimension creation
        # PhysicalUnit.validate() is called by dimension validation
        from luxar.typing_utils.enums import PhysicalUnit

        with pytest.raises(ValueError, match="Invalid"):
            PhysicalUnit.validate("invalid_unit")

    def test_units_in_enum_match_types(self) -> None:
        """Every ``PhysicalUnit`` member validates and round-trips unchanged.

        Derived from the enum, which is the vocabulary. This used to iterate
        ``typing_utils.config.SUPPORTED_UNITS`` — a hand-copy of the same list
        in a module whose other ~25 names nothing read.
        """
        from luxar.typing_utils.enums import PhysicalUnit
        from luxar.validation.types import validate_physical_unit

        members = list(PhysicalUnit)
        assert members, "PhysicalUnit is empty — the loop would be vacuous"
        for member in members:
            # Each supported unit validates (no raise) AND is returned unchanged
            # — not merely "does not raise".
            assert validate_physical_unit(member.value) == member.value

    def test_dimensions_with_mixed_units(self, tmp_path) -> None:
        """Test that different dimensions can have different units."""
        scene_path = tmp_path / "test_mixed.luxar.zarr"

        dims = Dimensions(
            [
                Dimension("x", unit="um", display=True),  # micrometers
                Dimension("y", unit="um", display=True),  # micrometers
                Dimension("z", unit="nm", display=True),  # nanometers (different!)
            ]
        )

        with LuxarZarrCompiler(scene_path) as compiler:
            compiler.create_scene(dimensions=dims)

        # Verify dimensions stored correctly
        import zarr

        store = zarr.open_group(scene_path, mode="r")
        dims_data = store.attrs["scene_dimensions"]["dimensions"]
        assert dims_data[0]["unit"] == "um"
        assert dims_data[1]["unit"] == "um"
        assert dims_data[2]["unit"] == "nm"
