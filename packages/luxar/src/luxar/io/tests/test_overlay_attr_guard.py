"""Regression tests for the internal overlay-attribute namespace."""

import tempfile
from pathlib import Path

import pytest
import zarr

from luxar import Dimensions, LuxarZarrCompiler


def test_bare_overlays_group_does_not_bypass_unknown_attr_guard() -> None:
    """Only real ``overlays/<name>`` entries use the internal schema."""
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = Path(tmpdir) / "test.luxar.zarr"
        with LuxarZarrCompiler(zarr_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="Did you mean 'blending_mode'"):
                scene.add_group("overlays", blending="max")

        root = zarr.open_group(str(zarr_path), mode="r")
        assert "overlays" not in root
