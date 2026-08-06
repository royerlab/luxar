"""Guard tests to ensure package descriptions stay accurate and inclusive of all geometry types."""

import luxar


class TestPackageDescription:
    """Ensure the luxar package docstring reflects all supported geometry types."""

    def test_docstring_does_not_say_point_cloud_viewer(self) -> None:
        """The package is NOT a 'point cloud viewer' — it supports multiple geometry types."""
        docstring = (luxar.__doc__ or "").lower()
        assert "point cloud viewer" not in docstring
        assert "point cloud visualization" not in docstring
        assert "points visualization" not in docstring

    def test_docstring_mentions_multiple_geometry_types(self) -> None:
        """The package docstring should mention points, lines, Gaussian splats, and meshes."""
        docstring = (luxar.__doc__ or "").lower()
        assert "points" in docstring or "point" in docstring
        assert "lines" in docstring or "line" in docstring
        assert "gaussian splat" in docstring or "splat" in docstring
        assert "mesh" in docstring

    def test_all_geometry_classes_exported(self) -> None:
        """All four geometry types must be importable from the top-level package."""
        assert hasattr(luxar, "Points")
        assert hasattr(luxar, "Lines")
        assert hasattr(luxar, "GSplats")
        assert hasattr(luxar, "Mesh")
