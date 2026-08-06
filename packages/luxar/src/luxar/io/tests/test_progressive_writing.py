"""Test progressive writing architecture with LuxarZarrCompiler."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar.core.dimensions import Dimensions
from luxar.encoding import ArrayDecoder
from luxar.io.compiler import LuxarZarrCompiler
from luxar.io.reader import LuxarScene


class TestProgressiveWriting:
    """Test the new progressive writing architecture."""

    def test_compiler_context_manager(self) -> None:
        """Test LuxarZarrCompiler as context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.luxar.zarr"

            with LuxarZarrCompiler(output_path) as compiler:
                # Compiler should be initialized
                assert compiler.store_path == str(output_path)
                assert compiler.store is not None

                # Create scene
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                assert scene is not None
                assert scene._writer == compiler

            # After context exit, should be finalized
            assert compiler._is_finalized

            # Zarr store should exist with consolidated metadata
            assert output_path.exists()
            store = zarr.open_group(output_path, mode="r")
            assert store.attrs["type"] == "scene"
            assert ".zmetadata" in store.store

    def test_progressive_points_writing(self) -> None:
        """Test that points are written immediately without keeping in memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.luxar.zarr"

            # Create large array (but not too large for CI)
            n_points = 10000
            positions = np.random.randn(n_points, 3).astype(np.float32)
            colors = np.random.rand(n_points, 3).astype(np.float32)

            with LuxarZarrCompiler(output_path, enable_spatial_index=False) as compiler:
                compiler.create_scene(dimensions=Dimensions.default_3d())

                # Write points - should go directly to disk
                metadata = compiler.write_points(
                    "test_points", positions, colors=colors
                )

                # Check metadata
                assert metadata["n_points"] == n_points
                assert metadata["ndim"] == 3
                assert metadata["has_colors"] is True

                # Verify data is in Zarr store
                store = zarr.open_group(output_path, mode="r")
                assert "test_points" in store
                assert "test_points/positions" in store
                assert "test_points/colors" in store

                # Verify data matches. Positions are uint16 per-axis fixed-point
                # under the default AUTO mode, so decode before comparing.
                decoded_positions = ArrayDecoder().decode(
                    store["test_points/positions"], store
                )
                np.testing.assert_allclose(
                    decoded_positions,
                    positions,
                    atol=float(np.ptp(positions, axis=0).max()) / 65535 * 2,
                )

    def test_scene_with_dimensions(self) -> None:
        """Test scene creation with dimensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.luxar.zarr"

            from luxar import Dimension

            dims = Dimensions(
                [
                    Dimension("x", unit="um", display=True),
                    Dimension("y", unit="um", display=True),
                    Dimension("z", unit="um", display=True),
                    Dimension("time", unit="s", display=False),
                ]
            )

            with LuxarZarrCompiler(output_path) as compiler:
                scene = compiler.create_scene(dimensions=dims)

                assert scene._dimensions == dims

                # Dimensions should be stored in Zarr
                store = zarr.open_group(output_path, mode="r")
                assert "scene_dimensions" in store.attrs
                dims_dict = store.attrs["scene_dimensions"]
                assert len(dims_dict["dimensions"]) == 4

    def test_hierarchical_structure(self) -> None:
        """Test creating hierarchical structure with groups."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.luxar.zarr"

            with LuxarZarrCompiler(output_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())

                # Add a group
                group1 = scene.add_group("group1", opacity=0.8)
                assert group1.name == "group1"
                assert group1._writer == compiler

                # Add nested group
                group2 = group1.add_group("group2")
                assert group2.path == "group1/group2"

                # Verify structure in Zarr
                store = zarr.open_group(output_path, mode="r")
                assert "group1" in store
                assert "group1/group2" in store

    def test_no_memory_accumulation(self) -> None:
        """Test that data is not kept in memory after writing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.luxar.zarr"

            with LuxarZarrCompiler(output_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())

                # Create data
                positions = np.random.randn(1000, 3).astype(np.float32)

                # Write points
                metadata = compiler.write_points("points", positions)

                # Metadata should not contain actual data
                assert "positions" not in metadata
                assert metadata["n_points"] == 1000

                # Scene should not have the data
                assert not hasattr(scene, "_positions")

    def test_compiler_without_context_manager(self) -> None:
        """Test using compiler without context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.luxar.zarr"

            compiler = LuxarZarrCompiler(output_path)
            compiler.create_scene(dimensions=Dimensions.default_3d())

            # Add some data
            positions = np.random.randn(100, 3).astype(np.float32)
            compiler.write_points("points", positions)

            # Manual finalization
            compiler.finalize()
            assert compiler._is_finalized

            # Verify store exists
            assert output_path.exists()

    def test_resizable_dataset_creation(self) -> None:
        """Test creating resizable datasets for streaming."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.luxar.zarr"

            with LuxarZarrCompiler(output_path) as compiler:
                # Create resizable dataset
                dataset = compiler.create_resizable_dataset(
                    "streaming_points/positions",
                    dtype=np.float32,
                    shape=(0, 3),
                    maxshape=(None, 3),
                )

                assert dataset is not None
                assert dataset.shape == (0, 3)

                # Append data
                batch1 = np.random.randn(100, 3).astype(np.float32)
                dataset.resize((100, 3))
                dataset[:100] = batch1

                # Append more data
                batch2 = np.random.randn(50, 3).astype(np.float32)
                dataset.resize((150, 3))
                dataset[100:150] = batch2

                # Verify
                assert dataset.shape == (150, 3)
                np.testing.assert_array_almost_equal(dataset[:100], batch1)
                np.testing.assert_array_almost_equal(dataset[100:150], batch2)


class TestExitOnException:
    """__exit__ must not finalize a partial store when an exception propagates."""

    def test_owned_tempdir_cleaned_after_finalize_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed finalize removes the compiler-owned temporary scene."""
        compiler = LuxarZarrCompiler(store_path=None, enable_spatial_index=False)
        assert compiler._tmpdir is not None
        temp_path = Path(compiler._tmpdir.name)
        assert temp_path.exists()

        def fail_content_hashes(store: zarr.Group) -> None:
            raise RuntimeError("forced finalize failure")

        with pytest.raises(ValueError, match="Could not finalize") as exc_info:
            with compiler:
                compiler.create_scene(dimensions=Dimensions.default_3d())
                monkeypatch.setattr(
                    compiler, "_compute_content_hashes", fail_content_hashes
                )

        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert str(exc_info.value.__cause__) == "forced finalize failure"
        assert not temp_path.exists()

    def test_owned_tempdir_cleaned_after_success(self) -> None:
        """A successful context exit still removes its temporary scene."""
        compiler = LuxarZarrCompiler(store_path=None, enable_spatial_index=False)
        assert compiler._tmpdir is not None
        temp_path = Path(compiler._tmpdir.name)
        assert temp_path.exists()

        with compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())

        assert compiler._is_finalized is True
        assert not temp_path.exists()

    def test_exception_leaves_store_incomplete(self) -> None:
        """A raise inside the with block: no finalize, incomplete marker set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.luxar.zarr"

            with pytest.raises(RuntimeError, match="boom"):
                with LuxarZarrCompiler(
                    output_path, enable_spatial_index=False
                ) as compiler:
                    compiler.create_scene(dimensions=Dimensions.default_3d())
                    positions = np.random.randn(100, 3).astype(np.float32)
                    compiler.write_points("pts", positions)
                    raise RuntimeError("boom")

            # Store was NOT finalized.
            assert compiler._is_finalized is False

            # Root carries the incomplete marker.
            store = zarr.open_group(output_path, mode="r")
            assert store.attrs.get("incomplete") is True

            # The reader refuses to load the incomplete store.
            with pytest.raises(ValueError, match="incomplete"):
                LuxarScene.load(output_path)

    def test_finalize_failure_marks_incomplete(self, monkeypatch) -> None:
        """A finalize() that fails midway (even on a clean exit) must mark the
        store incomplete so LuxarScene.load rejects the half-finalized store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.luxar.zarr"

            with pytest.raises(ValueError, match="Could not finalize"):
                with LuxarZarrCompiler(
                    output_path, enable_spatial_index=False
                ) as compiler:
                    compiler.create_scene(dimensions=Dimensions.default_3d())
                    positions = np.random.randn(100, 3).astype(np.float32)
                    compiler.write_points("pts", positions)
                    # Break a finalize sub-step so finalize() fails on the
                    # clean exit (Scene.to_zarr()-style direct finalize path).
                    monkeypatch.setattr(
                        compiler,
                        "_compute_content_hashes",
                        lambda store: (_ for _ in ()).throw(RuntimeError("disk full")),
                    )

            # finalize() failed → the store is not finalized.
            assert compiler._is_finalized is False

            # The half-finalized store carries the incomplete marker.
            store = zarr.open_group(output_path, mode="r")
            assert store.attrs.get("incomplete") is True

            # The reader refuses to load the half-finalized store.
            with pytest.raises(ValueError, match="incomplete"):
                LuxarScene.load(output_path)

    def test_finalize_pre_phase_failure_marks_incomplete(self, monkeypatch) -> None:
        """A DIRECT finalize() (no with block) that fails BEFORE the finalize
        phases — hover injection raising — must still mark the store incomplete
        (the marker-clear + hover injection now run inside the boundary)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.luxar.zarr"

            compiler = LuxarZarrCompiler(output_path, enable_spatial_index=False)
            compiler.create_scene(dimensions=Dimensions.default_3d())
            positions = np.random.randn(100, 3).astype(np.float32)
            compiler.write_points("pts", positions)

            # Break hover injection, which runs before the finalize phases.
            monkeypatch.setattr(
                compiler._scene,
                "_auto_inject_hover_overlay",
                lambda: (_ for _ in ()).throw(RuntimeError("hover boom")),
            )

            with pytest.raises(ValueError, match="Could not finalize"):
                compiler.finalize()

            # finalize() failed → the store is not finalized.
            assert compiler._is_finalized is False

            # The partial store carries the incomplete marker.
            store = zarr.open_group(output_path, mode="r")
            assert store.attrs.get("incomplete") is True

            # The reader refuses to load the partial store.
            with pytest.raises(ValueError, match="incomplete"):
                LuxarScene.load(output_path)

    def test_finalize_keyboardinterrupt_marks_incomplete(self, monkeypatch) -> None:
        """A DIRECT finalize() interrupted by KeyboardInterrupt (a
        BaseException, not Exception) mid-phase must mark the store incomplete
        and re-raise the ORIGINAL exception, unwrapped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.luxar.zarr"

            compiler = LuxarZarrCompiler(output_path, enable_spatial_index=False)
            compiler.create_scene(dimensions=Dimensions.default_3d())
            positions = np.random.randn(100, 3).astype(np.float32)
            compiler.write_points("pts", positions)

            # Interrupt a finalize sub-step with a BaseException.
            monkeypatch.setattr(
                compiler,
                "_compute_content_hashes",
                lambda store: (_ for _ in ()).throw(KeyboardInterrupt()),
            )

            # Re-raised unchanged — NOT wrapped in ValueError.
            with pytest.raises(KeyboardInterrupt):
                compiler.finalize()

            # finalize() failed → the store is not finalized.
            assert compiler._is_finalized is False

            # The half-finalized store carries the incomplete marker.
            store = zarr.open_group(output_path, mode="r")
            assert store.attrs.get("incomplete") is True

            # The reader refuses to load the half-finalized store.
            with pytest.raises(ValueError, match="incomplete"):
                LuxarScene.load(output_path)

    def test_success_print_failure_does_not_mark_incomplete(self, monkeypatch) -> None:
        """A failure of the purely-informational success print (e.g.
        BrokenPipeError on a closed stdout) must not fail finalize() nor mark
        the already-complete store incomplete."""
        import luxar.io.compiler as compiler_mod

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.luxar.zarr"

            compiler = LuxarZarrCompiler(output_path, enable_spatial_index=False)
            compiler.create_scene(dimensions=Dimensions.default_3d())
            positions = np.random.randn(100, 3).astype(np.float32)
            compiler.write_points("pts", positions)

            def broken_pipe_on_success(msg, *args, **kwargs):
                if str(msg).startswith("✅ Zarr store finalized"):
                    raise BrokenPipeError("stdout closed")

            monkeypatch.setattr(compiler_mod, "aprint", broken_pipe_on_success)

            # Does not raise: the store is complete, only the print failed.
            compiler.finalize()
            assert compiler._is_finalized is True

            store = zarr.open_group(output_path, mode="r")
            assert not store.attrs.get("incomplete")

            scene = LuxarScene.load(output_path)
            assert scene is not None

    def test_failed_stale_marker_clear_fails_finalize(self, monkeypatch) -> None:
        """If a stale incomplete marker cannot be deleted, finalize() must FAIL
        (not silently complete with the marker left behind, which would be a
        valid store that LuxarScene.load permanently rejects). The failure
        keeps _is_finalized False, so a retry can still succeed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.luxar.zarr"

            compiler = LuxarZarrCompiler(output_path, enable_spatial_index=False)
            compiler.create_scene(dimensions=Dimensions.default_3d())
            positions = np.random.randn(100, 3).astype(np.float32)
            compiler.write_points("pts", positions)

            # Stale marker from a prior aborted attempt.
            compiler.store.attrs["incomplete"] = True

            def failing_delitem(self, key):
                raise RuntimeError("attrs delete failed")

            monkeypatch.setattr(
                type(compiler.store.attrs), "__delitem__", failing_delitem
            )

            with pytest.raises(ValueError, match="Could not finalize"):
                compiler.finalize()

            # Finalize failed → not finalized, marker still present, rejected.
            assert compiler._is_finalized is False
            store = zarr.open_group(output_path, mode="r")
            assert store.attrs.get("incomplete") is True
            with pytest.raises(ValueError, match="incomplete"):
                LuxarScene.load(output_path)

            # Once the transient failure clears, a retry recovers fully.
            monkeypatch.undo()
            compiler.finalize()
            assert compiler._is_finalized is True
            store = zarr.open_group(output_path, mode="r")
            assert not store.attrs.get("incomplete")
            scene = LuxarScene.load(output_path)
            assert scene is not None

    def test_finalize_clears_stale_incomplete_marker(self) -> None:
        """finalize() on a store bearing a stale incomplete marker (from a
        prior aborted attempt) clears it and yields a loadable store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.luxar.zarr"

            compiler = LuxarZarrCompiler(output_path, enable_spatial_index=False)
            compiler.create_scene(dimensions=Dimensions.default_3d())
            positions = np.random.randn(100, 3).astype(np.float32)
            compiler.write_points("pts", positions)

            # Simulate a prior aborted attempt having stamped the marker.
            compiler.store.attrs["incomplete"] = True
            assert compiler._is_finalized is False

            # A real finalize supersedes the stale marker.
            compiler.finalize()
            assert compiler._is_finalized is True

            store = zarr.open_group(output_path, mode="r")
            assert not store.attrs.get("incomplete")

            # The recovered store loads without error.
            scene = LuxarScene.load(output_path)
            assert scene is not None

    def test_clean_exit_finalizes_and_loads(self) -> None:
        """A normal with block finalizes, has no incomplete marker, loads."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.luxar.zarr"

            with LuxarZarrCompiler(output_path, enable_spatial_index=False) as compiler:
                compiler.create_scene(dimensions=Dimensions.default_3d())
                positions = np.random.randn(100, 3).astype(np.float32)
                compiler.write_points("pts", positions)

            assert compiler._is_finalized is True

            store = zarr.open_group(output_path, mode="r")
            assert not store.attrs.get("incomplete")

            # Loads without error.
            scene = LuxarScene.load(output_path)
            assert scene is not None

    def test_exception_after_finalize_still_loads(self) -> None:
        """An unrelated raise AFTER an explicit finalize must not mark the
        already-complete store incomplete."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.luxar.zarr"

            with pytest.raises(RuntimeError, match="boom"):
                with LuxarZarrCompiler(
                    output_path, enable_spatial_index=False
                ) as compiler:
                    compiler.create_scene(dimensions=Dimensions.default_3d())
                    positions = np.random.randn(100, 3).astype(np.float32)
                    compiler.write_points("pts", positions)
                    # Finalize explicitly inside the block (the Scene.to_zarr
                    # pattern), then raise something unrelated.
                    compiler.finalize()
                    raise RuntimeError("boom")

            # The store was already finalized; the exception must not undo it.
            assert compiler._is_finalized is True

            store = zarr.open_group(output_path, mode="r")
            assert not store.attrs.get("incomplete")

            # The complete store still loads.
            scene = LuxarScene.load(output_path)
            assert scene is not None

    def test_keyboardinterrupt_marks_incomplete(self) -> None:
        """A KeyboardInterrupt (BaseException) inside the block leaves the
        store unfinalized and marked incomplete."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.luxar.zarr"

            with pytest.raises(KeyboardInterrupt):
                with LuxarZarrCompiler(
                    output_path, enable_spatial_index=False
                ) as compiler:
                    compiler.create_scene(dimensions=Dimensions.default_3d())
                    positions = np.random.randn(100, 3).astype(np.float32)
                    compiler.write_points("pts", positions)
                    raise KeyboardInterrupt

            # Store was NOT finalized.
            assert compiler._is_finalized is False

            # Root carries the incomplete marker.
            store = zarr.open_group(output_path, mode="r")
            assert store.attrs.get("incomplete") is True

            # The reader refuses to load the incomplete store.
            with pytest.raises(ValueError, match="incomplete"):
                LuxarScene.load(output_path)
