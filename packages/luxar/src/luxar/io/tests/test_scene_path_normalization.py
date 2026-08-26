"""The scene compiler enforces the canonical ``.luxar.zarr`` extension.

Full Luxar scenes are written with the self-identifying ``.luxar.zarr``
extension. ``LuxarZarrCompiler`` normalizes its output path so a bare name or a
plain ``.zarr`` still produces a canonically-named scene, and exposes the final
path via ``store_path``.
"""

import zipfile
from pathlib import Path

import numpy as np
import pytest

from luxar import Dimensions, LuxarScene, LuxarZarrCompiler
from luxar.io import optimise as optimise_mod


def _write_minimal_scene(compiler) -> None:
    compiler.create_scene(dimensions=Dimensions.default_3d())
    compiler.write_points("pts", np.array([[1.0, 2.0, 3.0]], dtype=np.float32))


class TestSceneExtensionNormalization:
    def test_archive_initialization_log_uses_public_path(self, tmp_path, capsys):
        requested = tmp_path / "scene.luxar.zarr.zip"

        LuxarZarrCompiler(requested)
        output = capsys.readouterr().out

        assert f"Zarr compiler initialized at {requested}" in output
        assert ".compile-" not in output

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

    def test_zarr_zip_is_packaged_flat_and_round_trips(self, tmp_path):
        requested = tmp_path / "scene.luxar.zarr.zip"

        with LuxarZarrCompiler(requested) as compiler:
            _write_minimal_scene(compiler)
            assert Path(compiler.store_path).is_dir()

        assert compiler.store_path == str(requested)
        assert requested.is_file()
        assert not (tmp_path / "scene.luxar.zarr.zip.luxar.zarr").exists()
        with zipfile.ZipFile(requested) as archive:
            names = archive.namelist()
            assert names == sorted(names)
            assert len(names) == len(set(names))
            assert "zarr.json" in names
            assert all(
                info.compress_type == zipfile.ZIP_STORED for info in archive.infolist()
            )

        scene = LuxarScene.load(requested)
        np.testing.assert_array_equal(
            scene.get_points("pts")["positions"],
            np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        )

    def test_plain_zarr_zip_gets_canonical_inner_suffix(self, tmp_path):
        requested = tmp_path / "scene.zarr.zip"

        with LuxarZarrCompiler(requested) as compiler:
            _write_minimal_scene(compiler)

        assert compiler.store_path == str(tmp_path / "scene.luxar.zarr.zip")
        assert (tmp_path / "scene.luxar.zarr.zip").is_file()
        assert not requested.exists()

    def test_bare_zip_gets_canonical_inner_suffix(self, tmp_path):
        requested = tmp_path / "scene.zip"

        with LuxarZarrCompiler(requested) as compiler:
            _write_minimal_scene(compiler)

        assert compiler.store_path == str(tmp_path / "scene.luxar.zarr.zip")
        assert (tmp_path / "scene.luxar.zarr.zip").is_file()
        assert not requested.exists()

    def test_zip_without_archive_name_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="must include a name before .zip"):
            LuxarZarrCompiler(tmp_path / ".zip")

    def test_archive_destination_symlink_is_refused_before_authoring(self, tmp_path):
        target = tmp_path / "target.zip"
        target.write_bytes(b"previous archive")
        requested = tmp_path / "scene.luxar.zarr.zip"
        requested.symlink_to(target)

        with pytest.raises(ValueError, match="refuses to write.*symlink"):
            LuxarZarrCompiler(requested)

        assert requested.is_symlink()
        assert target.read_bytes() == b"previous archive"
        assert sorted(path.name for path in tmp_path.iterdir()) == [
            requested.name,
            target.name,
        ]

    def test_scene_to_zarr_can_explicitly_finalize_requested_archive(self, tmp_path):
        requested = tmp_path / "scene.luxar.zarr.zip"

        with LuxarZarrCompiler(requested) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_points("pts", np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
            scene.to_zarr(requested)

        assert requested.is_file()
        assert compiler.store_path == str(requested)
        assert LuxarScene.load(requested).get_points("pts")["positions"].shape == (
            1,
            3,
        )

    def test_scene_to_zarr_replaces_existing_requested_archive(self, tmp_path):
        requested = tmp_path / "scene.luxar.zarr.zip"
        with LuxarZarrCompiler(requested) as compiler:
            _write_minimal_scene(compiler)

        with LuxarZarrCompiler(requested) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_points(
                "replacement",
                np.array([[4.0, 5.0, 6.0]], dtype=np.float32),
            )
            scene.to_zarr(requested)

        with zipfile.ZipFile(requested) as archive:
            assert not any(name.startswith("pts/") for name in archive.namelist())
            assert any(name.startswith("replacement/") for name in archive.namelist())

    def test_scene_to_zarr_is_idempotent_for_finalized_archive(self, tmp_path):
        requested = tmp_path / "scene.luxar.zarr.zip"

        with LuxarZarrCompiler(requested) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_points("pts", np.zeros((1, 3), dtype=np.float32))
            scene.to_zarr(requested)

        scene.to_zarr(requested)

        assert requested.is_file()
        assert compiler.store_path == str(requested)

    def test_scene_to_zarr_rejects_other_destination_before_archive_publish(
        self, tmp_path
    ):
        requested = tmp_path / "scene.luxar.zarr.zip"
        copy = tmp_path / "copy.luxar.zarr"

        with LuxarZarrCompiler(requested) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_points("pts", np.zeros((1, 3), dtype=np.float32))
            with pytest.raises(ValueError, match="only supports its final destination"):
                scene.to_zarr(copy)
            assert not requested.exists()
            assert not copy.exists()

        assert requested.is_file()
        assert not copy.exists()

    def test_scene_to_zarr_uses_writer_path_normalization_consistently(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        destination = tmp_path / "copy.luxar.zarr"

        with LuxarZarrCompiler("~/scene.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_points("pts", np.zeros((1, 3), dtype=np.float32))
            scene.to_zarr(destination)

        assert destination.is_dir()
        assert LuxarScene.load(destination).get_points("pts")["positions"].shape == (
            1,
            3,
        )

    def test_scene_to_zarr_archive_log_uses_public_path(self, tmp_path, capsys):
        requested = tmp_path / "scene.luxar.zarr.zip"

        with LuxarZarrCompiler(requested) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_points("pts", np.zeros((1, 3), dtype=np.float32))
            capsys.readouterr()
            scene.to_zarr(requested)
            output = capsys.readouterr().out

        assert f"Finalized scene at {requested}" in output
        assert "Exporting scene from" not in output
        assert ".compile-" not in output

    def test_archive_packaging_reports_progress(self, tmp_path, capsys):
        requested = tmp_path / "scene.luxar.zarr.zip"

        with LuxarZarrCompiler(requested) as compiler:
            _write_minimal_scene(compiler)
            capsys.readouterr()

        output = capsys.readouterr().out
        assert f"Packaging scene archive at {requested}" in output

    def test_existing_zarr_zip_is_replaced_without_stale_members(self, tmp_path):
        requested = tmp_path / "scene.luxar.zarr.zip"
        with LuxarZarrCompiler(requested) as compiler:
            _write_minimal_scene(compiler)

        with LuxarZarrCompiler(requested) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_points(
                "replacement",
                np.array([[4.0, 5.0, 6.0]], dtype=np.float32),
            )

        scene = LuxarScene.load(requested)
        with zipfile.ZipFile(requested) as archive:
            assert not any(name.startswith("pts/") for name in archive.namelist())
            assert any(name.startswith("replacement/") for name in archive.namelist())
        assert scene.get_points("replacement")["positions"].shape == (1, 3)

    def test_failed_packaging_preserves_previous_archive_and_cleans_staging(
        self, tmp_path, monkeypatch
    ):
        requested = tmp_path / "scene.luxar.zarr.zip"
        requested.write_bytes(b"previous archive")

        def fail_packaging(staging, artifact):
            artifact.write_bytes(b"partial archive")
            raise OSError("disk full")

        monkeypatch.setattr(optimise_mod, "_package", fail_packaging)
        with pytest.raises(ValueError, match="disk full"):
            with LuxarZarrCompiler(requested) as compiler:
                _write_minimal_scene(compiler)

        assert requested.read_bytes() == b"previous archive"
        assert sorted(path.name for path in tmp_path.iterdir()) == [requested.name]

    def test_swallowed_packaging_failure_cannot_publish_empty_archive(
        self, tmp_path, monkeypatch
    ):
        requested = tmp_path / "scene.luxar.zarr.zip"
        with LuxarZarrCompiler(requested) as compiler:
            _write_minimal_scene(compiler)
        previous_archive = requested.read_bytes()

        attempts = 0

        def fail_once(staging, artifact):
            nonlocal attempts
            attempts += 1
            raise OSError("transient packaging failure")

        monkeypatch.setattr(optimise_mod, "_package", fail_once)
        with LuxarZarrCompiler(requested) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_points(
                "replacement",
                np.array([[4.0, 5.0, 6.0]], dtype=np.float32),
            )
            with pytest.raises(ValueError, match="transient packaging failure"):
                compiler.finalize()
            with pytest.raises(ValueError, match="previous finalization failure"):
                compiler.finalize()

        assert attempts == 1
        assert requested.read_bytes() == previous_archive
        assert sorted(path.name for path in tmp_path.iterdir()) == [requested.name]
        scene = LuxarScene.load(requested)
        assert scene.get_points("pts")["positions"].shape == (1, 3)
        with pytest.raises(KeyError):
            scene.get_points("replacement")

    def test_body_failure_does_not_publish_or_leave_staging(self, tmp_path, capsys):
        requested = tmp_path / "scene.luxar.zarr.zip"

        with pytest.raises(RuntimeError, match="authoring failed"):
            with LuxarZarrCompiler(requested) as compiler:
                _write_minimal_scene(compiler)
                raise RuntimeError("authoring failed")

        output = capsys.readouterr().out
        assert not requested.exists()
        assert list(tmp_path.iterdir()) == []
        assert f"discarding incomplete archive staging for {requested}" in output
        assert "leaving the store unfinalized" not in output
