"""Test physical units support in Luxar."""

import pytest

from luxar import Scene


class TestPhysicalUnits:
    """Test that all supported physical units work correctly."""

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
    def test_unit_acceptance(self, unit, tmp_path):
        """Test that each physical unit is accepted by Scene."""
        scene_path = tmp_path / f"test_{unit}.zarr"
        scene = Scene(scene_path, units=unit)
        assert scene.attrs["units"] == unit

    def test_invalid_unit_rejection(self, tmp_path):
        """Test that invalid units are rejected."""
        scene_path = tmp_path / "test_invalid.zarr"
        with pytest.raises(ValueError, match="Invalid unit"):
            Scene(scene_path, units="invalid_unit")

    def test_units_in_config_match_types(self):
        """Test that SUPPORTED_UNITS in config matches validation in types."""
        from luxar.config import SUPPORTED_UNITS
        from luxar.types import validate_physical_unit

        # All units in config should be valid
        for unit in SUPPORTED_UNITS:
            assert validate_physical_unit(unit) == unit
