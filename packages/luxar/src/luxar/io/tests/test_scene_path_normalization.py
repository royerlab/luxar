"""The scene compiler enforces the canonical ``.luxar.zarr`` extension.

Full Luxar scenes are written with the self-identifying ``.luxar.zarr``
extension. ``LuxarZarrCompiler`` normalizes its output path so a bare name or a
plain ``.zarr`` still produces a canonically-named scene, and exposes the final
path via ``store_path``.
"""

import numpy as np

from luxar import Dimensions, LuxarScene, LuxarZarrCompiler


def _write_minimal_scene(compiler) -> None:
    compiler.create_scene(dimensions=Dimensions.default_3d())
    compiler.write_points("pts", np.array([[1.0, 2.0, 3.0]], dtype=np.float32))


class TestSceneExtensionNormalization:
    def test_bare_name_gets_luxar_zarr(self, tmp_path):
        c = LuxarZarrCompiler(tmp_path / "scene")
        assert c.store_path.endswith("scene.luxar.zarr")

    def test_plain_zarr_is_corrected(self, tmp_path):
        # The "correct if needed" case: plain .zarr -> .luxar.zarr.
        c = LuxarZarrCompiler(tmp_path / "scene.zarr")
        assert c.store_path.endswith("scene.luxar.zarr")
        assert not c.store_path.endswith("scene.zarr.luxar.zarr")

    def test_canonical_is_unchanged(self, tmp_path):
        c = LuxarZarrCompiler(tmp_path / "scene.luxar.zarr")
        assert c.store_path.endswith("scene.luxar.zarr")

    def test_store_written_at_canonical_path(self, tmp_path):
        # Round-trip via the normalized store_path; the on-disk store lives at
        # the canonical location, NOT at the plain-.zarr path the caller passed.
        with LuxarZarrCompiler(tmp_path / "scene.zarr") as compiler:
            _write_minimal_scene(compiler)
            written = compiler.store_path

        assert written.endswith("scene.luxar.zarr")
        assert (tmp_path / "scene.luxar.zarr").exists()
        assert not (tmp_path / "scene.zarr").exists()

        scene = LuxarScene.load(written)
        assert scene.get_points("pts")["positions"].shape == (1, 3)
