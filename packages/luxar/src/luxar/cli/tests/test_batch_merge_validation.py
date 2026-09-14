"""CLI + plan-time validation tests for `batch-fit merge` recipe knobs.

Guards the friction fixes: recipe-specific merge knobs must be *rejected* (not
silently ignored) when given without a matching recipe, `--no-recipe` must be
able to override a manifest's `merge_recipe`, and `--flat` must be mutually
exclusive with `--recipe` at the CLI boundary (not only deep in the library).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import typer
import zarr
from typer.testing import CliRunner

from luxar.cli.gsplat_commands import app_gsplat
from luxar.cli.tests._testing import normalized_cli_output

runner = CliRunner()


def _write_manifest(
    output_dir: Path,
    *,
    merge_recipe: str | None = None,
    tile_names: list[str] | None = None,
) -> None:
    """Write a minimal BatchManifest (enough for `batch-fit merge` to load)."""
    from luxar.gsplats.batch.manifest import BatchJob, BatchManifest, save_manifest

    jobs = [
        BatchJob(
            task_id=i,
            timepoint=0,
            channel=0,
            tile_index=i,
            output_filename=name,
            estimated_wall_seconds=1.0,
        )
        for i, name in enumerate(tile_names or [])
    ]
    manifest = BatchManifest(
        input_path="/data/test.zarr",
        output_dir=str(output_dir),
        n_timepoints=1,
        n_channels=1,
        spatial_shape=(64, 64, 64),
        tile_size=64,
        n_tiles=max(1, len(jobs)),
        total_tasks=max(1, len(jobs)),
        merge_recipe=merge_recipe,
        jobs=jobs,
    )
    save_manifest(manifest, output_dir)


# ── plan-time: resolve_merge_recipe_args (the `submit`/`run` path) ──────────


class TestResolveMergeRecipeArgs:
    def test_orphaned_knob_without_recipe_raises(self) -> None:
        """A merge knob without --merge-recipe is a silent no-op pre-fix; it must
        now raise rather than return `{}`."""
        from luxar.cli.gsplat_ops.batch.planning import (
            MergeConfig,
            resolve_merge_recipe_args,
        )

        with pytest.raises(typer.BadParameter):
            resolve_merge_recipe_args(MergeConfig(recipe=None, n_lods=6))

    def test_no_knobs_no_recipe_returns_empty(self) -> None:
        from luxar.cli.gsplat_ops.batch.planning import (
            MergeConfig,
            resolve_merge_recipe_args,
        )

        assert resolve_merge_recipe_args(MergeConfig()) == {}

    def test_valid_recipe_with_knobs_resolves(self) -> None:
        from luxar.cli.gsplat_ops.batch.planning import (
            MergeConfig,
            resolve_merge_recipe_args,
        )

        args = resolve_merge_recipe_args(MergeConfig(recipe="stream", n_lods=6))
        assert args == {"n-lods": "6"}

    def test_cross_recipe_knob_still_rejected(self) -> None:
        from luxar.cli.gsplat_ops.batch.planning import (
            MergeConfig,
            resolve_merge_recipe_args,
        )

        with pytest.raises(typer.BadParameter):
            resolve_merge_recipe_args(
                MergeConfig(recipe="stream", compression_factor=4)
            )


class TestMergePipelineProvenance:
    """The merge's pipeline/ provenance must record the RECIPE separately from
    the reduction MECHANISM (lod_kind). A recipe name must never land in
    lod_kind — before the rename they coincided (additive/substitutive), which
    masked the conflation."""

    def test_recipe_and_lod_kind_are_distinct(self) -> None:
        from luxar.gsplats.batch.merge_orchestrator import _recipe_pipeline_info

        s = _recipe_pipeline_info("stream", None)
        assert s["recipe"] == "stream" and s["lod_kind"] == "additive"
        lv = _recipe_pipeline_info("levels", None)
        assert lv["recipe"] == "levels" and lv["lod_kind"] == "substitutive"
        # lod_kind is a mechanism word the viewer/migrator understand, NEVER a
        # recipe name.
        assert lv["lod_kind"] != "levels" and s["lod_kind"] != "stream"

    def test_refine_iters_none_sentinel_resolves(self) -> None:
        """RecipeParams.refine_iters=None (the engine-default sentinel) must
        record the value that actually runs — not crash on int(None), and not
        record None for an active refine. Fails pre-fix with TypeError."""
        from luxar.gsplats.batch.merge_orchestrator import _recipe_pipeline_info
        from luxar.gsplats.lod.recipes import RecipeParams

        assert (
            _recipe_pipeline_info("levels", RecipeParams(refine="l2"))["refine_iters"]
            == 120
        )
        assert (
            _recipe_pipeline_info("levels", RecipeParams(refine="l2", refine_iters=7))[
                "refine_iters"
            ]
            == 7
        )
        assert _recipe_pipeline_info("levels", RecipeParams())["refine_iters"] is None

    def test_merge_validates_refine_volume_front_door(self, tmp_path) -> None:
        """A merge-time volume re-fit re-opens the source and crops it per tile, so
        anything making the source unusable must be caught BEFORE the streaming
        merge writes a single part — the deep per-part guard would fire mid-stream
        and leave a half-written store.

        This replaces a test asserting the combination was refused outright. The
        refusal is now specific: it names what is missing instead of declaring the
        merge volume-free.
        """
        import numpy as np
        import pytest

        from luxar.gsplats.batch.manifest import BatchManifest
        from luxar.gsplats.batch.merge_orchestrator import merge_batch_results
        from luxar.gsplats.lod.recipes import RecipeParams

        out_dir = tmp_path / "batch"
        (out_dir / "tiles").mkdir(parents=True)
        params = RecipeParams(refine="volume", volume=np.zeros((4, 4, 4), np.float32))

        def _merge(manifest):
            return merge_batch_results(
                manifest,
                out_dir,
                verbose=False,
                recipe="levels",
                recipe_params=params,
            )

        # No source recorded at all.
        with pytest.raises(ValueError, match="records no input path"):
            _merge(BatchManifest(n_timepoints=1, n_channels=1, n_tiles=1))

        # A source that has since moved or been deleted.
        with pytest.raises(ValueError, match="no longer exists"):
            _merge(
                BatchManifest(
                    n_timepoints=1,
                    n_channels=1,
                    n_tiles=1,
                    input_path=str(tmp_path / "gone.zarr"),
                )
            )

        # Present, but planned without --axes: the stacked axis cannot be mapped,
        # and guessing it wrong sends every re-fit at the wrong axis.
        source = tmp_path / "src.npy"
        np.save(source, np.zeros((4, 4, 4), np.float32))
        with pytest.raises(ValueError, match="axis labels"):
            _merge(
                BatchManifest(
                    n_timepoints=1, n_channels=1, n_tiles=1, input_path=str(source)
                )
            )

        # An unrecognised axis label, named by the front door rather than
        # surfacing from the loader further down.
        with pytest.raises(ValueError, match="not recognised"):
            _merge(
                BatchManifest(
                    n_timepoints=1,
                    n_channels=1,
                    n_tiles=1,
                    input_path=str(source),
                    axes="q,z,y",
                )
            )

        # SEVERAL channels selected: a merged part carries ONE stacked axis (the
        # timepoints), and there is no single channel index to pin either.
        with pytest.raises(ValueError, match="several channels were selected"):
            _merge(
                BatchManifest(
                    n_timepoints=2,
                    n_channels=3,
                    n_tiles=1,
                    input_path=str(source),
                    axes="t,c,z,y,x",
                )
            )

        # Mis-FRAMED: the run's --config carried a real-space voxel_size, so the
        # planner recorded a grid_scale and every part's crop — taken in VOXELS
        # from the tile grid — would be a factor off (#1587). `gsplat fit`
        # already refuses --refine volume there; the batch front door must too.
        with pytest.raises(ValueError, match="frame scaled by"):
            _merge(
                BatchManifest(
                    n_timepoints=1,
                    n_channels=1,
                    n_tiles=1,
                    input_path=str(source),
                    axes="z,y,x",
                    grid_scale=[4.0, 1.0, 1.0],
                )
            )
        # Non-vacuity: the same manifest with the frames in agreement passes the
        # front door (an all-ones factor is no scale, and so is an absent one —
        # which is what every manifest written before #1587 says).
        from luxar.gsplats.batch.merge_orchestrator import _validate_merge_volume_refit

        for agreeing in ([1.0, 1.0, 1.0], None):
            _validate_merge_volume_refit(
                BatchManifest(
                    n_timepoints=1,
                    n_channels=1,
                    n_tiles=1,
                    input_path=str(source),
                    axes="z,y,x",
                    grid_scale=agreeing,
                )
            )

        # Every rejection happened before anything was written.
        assert not (out_dir / "merged").exists()

    def test_merge_refit_source_pins_the_axes_a_part_does_not_carry(
        self, tmp_path
    ) -> None:
        """The re-opened source is the FULL array — a merged part is not.

        The fit tasks sliced one channel (and, with a single timepoint, one
        timepoint) away, so those axes survive on the source but have no center
        column on the part. They must be PINNED to the index the fit used;
        left in, the axis map is the wrong length and every re-fit is aimed at
        the wrong axis. Pinning stays lazy — pre-slicing a zarr array
        materialises it, which is what opening it lazily exists to avoid.
        """
        import numpy as np
        import zarr

        from luxar.gsplats.batch.manifest import BatchManifest
        from luxar.gsplats.batch.merge_orchestrator import _merge_refit_volume

        # The canonical OME-Zarr shape: (t, c, z, y, x).
        arr = np.arange(3 * 2 * 4 * 5 * 6, dtype=np.uint16).reshape(3, 2, 4, 5, 6)
        source = tmp_path / "tczyx.zarr"
        z = zarr.open(str(source), mode="w", shape=arr.shape, dtype=arr.dtype)
        z[:] = arr

        # Three timepoints stacked, channel 1 selected: the time axis is WALKED
        # (one slice per barrier group), the channel axis is pinned.
        volume, axes = _merge_refit_volume(
            BatchManifest(
                input_path=str(source),
                axes="t,c,z,y,x",
                n_timepoints=3,
                n_channels=1,
                channel_indices=[1],
            )
        )
        assert not isinstance(volume, np.ndarray)  # still lazy
        assert tuple(volume.shape) == (3, 4, 5, 6)
        # Center dims are spatial-first with the stacked axis LAST.
        assert axes == (1, 2, 3, 0)
        np.testing.assert_array_equal(np.asarray(volume[2]), arr[2, 1])

        # One timepoint selected: the merge stacks nothing, so the TIME axis is
        # pinned too and the part is purely spatial.
        volume, axes = _merge_refit_volume(
            BatchManifest(
                input_path=str(source),
                axes="t,c,z,y,x",
                n_timepoints=1,
                n_channels=1,
                timepoint_indices=[2],
                channel_indices=[0],
            )
        )
        assert tuple(volume.shape) == (4, 5, 6)
        assert axes == (0, 1, 2)
        np.testing.assert_array_equal(np.asarray(volume[:]), arr[2, 0])

    def test_merge_refit_rejects_axes_that_do_not_describe_the_array(
        self, tmp_path
    ) -> None:
        """Label count vs the resolved array — a mismatch would mis-pin silently."""
        import numpy as np
        import zarr

        from luxar.gsplats.batch.manifest import BatchManifest
        from luxar.gsplats.batch.merge_orchestrator import _merge_refit_volume

        source = tmp_path / "zyx.zarr"
        z = zarr.open(str(source), mode="w", shape=(4, 4, 4), dtype="u2")
        z[:] = np.zeros((4, 4, 4), np.uint16)
        with pytest.raises(ValueError, match="labels but .* resolved to a 3D array"):
            _merge_refit_volume(
                BatchManifest(
                    input_path=str(source),
                    axes="t,z,y,x",
                    n_timepoints=2,
                    n_channels=1,
                )
            )

    def test_legacy_manifest_recipe_translates_in_provenance(self) -> None:
        from luxar.gsplats.batch.merge_orchestrator import _recipe_pipeline_info

        # A pre-rename manifest value flows in; provenance records the CURRENT
        # recipe name + the mechanism.
        info = _recipe_pipeline_info("substitutive", None)
        assert info["recipe"] == "levels" and info["lod_kind"] == "substitutive"
        info = _recipe_pipeline_info("additive", None)
        assert info["recipe"] == "stream" and info["lod_kind"] == "additive"

    def test_no_recipe_writes_no_provenance(self) -> None:
        from luxar.gsplats.batch.merge_orchestrator import _recipe_pipeline_info

        assert _recipe_pipeline_info(None, None) is None


class TestMergeStreamingKnobs:
    """--merge-target-ms trio: plan-time resolution to a stored stream:<c>
    breakpoints string (manifest schema unchanged) + the exclusion rules."""

    def test_target_ms_resolves_to_stored_stream_string(self) -> None:
        from luxar.cli.gsplat_ops.batch.planning import (
            MergeConfig,
            resolve_merge_recipe_args,
        )

        args = resolve_merge_recipe_args(
            MergeConfig(recipe="stream", target_ms=200.0), merged_ndim=4
        )
        bp = args["breakpoints"]
        assert bp.startswith("stream:")
        # 200 ms @ default 25 Mbps @ analytic 30 B (4D: centers u16,
        # amplitude u16, cholesky u8-certified) → 625000/30 = 20833.
        assert bp == "stream:20833"

    def test_target_ms_scales_for_slices_and_partition_parts(self) -> None:
        from luxar.cli.gsplat_ops.batch.planning import (
            MergeConfig,
            resolve_merge_recipe_args,
        )

        args = resolve_merge_recipe_args(
            MergeConfig(recipe="stream", target_ms=200.0),
            merged_ndim=4,
            slice_count=6,
            part_count=3,
        )

        assert args["breakpoints"] == "stream:41666"

    def test_store_survey_counts_hidden_coordinates_and_partition_parts(
        self, tmp_path: Path
    ) -> None:
        from luxar.cli.gsplat_ops.recipe_shared import survey_gsplat_streaming_layout

        store_path = tmp_path / "layout.gsplats.zarr"
        root = zarr.open_group(store_path, mode="w")
        root.attrs["kind"] = "partition"
        for part_index in range(2):
            leaf = root.create_group(f"part_{part_index}")
            leaf.attrs.update(
                {"type": "gsplats", "n_splats": 3, "n_additive_sublods": 1}
            )
            leaf.attrs["slice_dims"] = [3]
            leaf.create_array(
                "centers",
                data=np.asarray(
                    [
                        [part_index, 0, 0, 0],
                        [part_index, 1, 0, 1],
                        [part_index, 2, 0, 2],
                    ],
                    dtype=np.uint16,
                ),
            )

        assert survey_gsplat_streaming_layout(store_path) == (3, 2)

    def test_store_survey_ignores_spatial_integer_barriers(
        self, tmp_path: Path
    ) -> None:
        from luxar.cli.gsplat_ops.recipe_shared import survey_gsplat_streaming_layout

        store_path = tmp_path / "spatial-layout.gsplats.zarr"
        leaf = zarr.open_group(store_path, mode="w")
        leaf.attrs.update(
            {
                "type": "gsplats",
                "n_splats": 4,
                "n_additive_sublods": 1,
                "slice_dims": [0, 1, 2],
            }
        )
        leaf.create_array(
            "centers",
            data=np.asarray(
                [[0, 0, 0], [1, 2, 3], [2, 4, 6], [3, 6, 9]], dtype=np.uint16
            ),
        )

        assert survey_gsplat_streaming_layout(store_path) == (1, 1)

    def test_store_survey_ignores_partial_spatial_integer_barriers(
        self, tmp_path: Path
    ) -> None:
        from luxar.cli.gsplat_ops.recipe_shared import survey_gsplat_streaming_layout

        store_path = tmp_path / "partial-spatial-layout.gsplats.zarr"
        leaf = zarr.open_group(store_path, mode="w")
        leaf.attrs.update(
            {
                "type": "gsplats",
                "n_splats": 4,
                "n_additive_sublods": 1,
                "slice_dims": [0, 1],
            }
        )
        leaf.create_array(
            "centers",
            data=np.asarray(
                [[0, 0, 0.1], [0, 1, 0.2], [1, 0, 0.3], [1, 1, 0.4]],
                dtype=np.float32,
            ),
        )

        assert survey_gsplat_streaming_layout(store_path) == (1, 1)

    def test_store_survey_counts_2d_stacked_barrier_axis(self, tmp_path: Path) -> None:
        from luxar.cli.gsplat_ops.recipe_shared import survey_gsplat_streaming_layout

        store_path = tmp_path / "stacked-2d-layout.gsplats.zarr"
        leaf = zarr.open_group(store_path, mode="w")
        leaf.attrs.update(
            {
                "type": "gsplats",
                "n_splats": 4,
                "n_additive_sublods": 1,
                "slice_dims": [2],
            }
        )
        leaf.create_array(
            "centers",
            data=np.asarray(
                [[0, 0, 0], [1, 2, 0], [2, 4, 1], [3, 6, 1]], dtype=np.uint16
            ),
        )

        assert survey_gsplat_streaming_layout(store_path) == (2, 1)

    def test_store_survey_counts_only_stacked_barrier_axes(
        self, tmp_path: Path
    ) -> None:
        from luxar.cli.gsplat_ops.recipe_shared import survey_gsplat_streaming_layout

        store_path = tmp_path / "stacked-layout.gsplats.zarr"
        leaf = zarr.open_group(store_path, mode="w")
        leaf.attrs.update(
            {
                "type": "gsplats",
                "n_splats": 4,
                "n_additive_sublods": 1,
                "slice_dims": [0, 1, 2, 3],
            }
        )
        leaf.create_array(
            "centers",
            data=np.asarray(
                [[0, 0, 0, 0], [1, 2, 3, 0], [2, 4, 6, 1], [3, 6, 9, 1]],
                dtype=np.uint16,
            ),
        )

        assert survey_gsplat_streaming_layout(store_path) == (2, 1)

    def test_store_survey_excludes_spatial_axis_from_partial_barrier(
        self, tmp_path: Path
    ) -> None:
        from luxar.cli.gsplat_ops.recipe_shared import survey_gsplat_streaming_layout

        store_path = tmp_path / "partial-stacked-layout.gsplats.zarr"
        leaf = zarr.open_group(store_path, mode="w")
        leaf.attrs.update(
            {
                "type": "gsplats",
                "n_splats": 4,
                "n_additive_sublods": 1,
                "slice_dims": [2, 3],
            }
        )
        leaf.create_array(
            "centers",
            data=np.asarray(
                [
                    [0.1, 0.2, 0, 0],
                    [0.3, 0.4, 1, 0],
                    [0.5, 0.6, 0, 1],
                    [0.7, 0.8, 1, 1],
                ],
                dtype=np.float32,
            ),
        )

        assert survey_gsplat_streaming_layout(store_path) == (2, 1)

    def test_store_survey_decodes_each_rungs_coordinate_grid(
        self, tmp_path: Path
    ) -> None:
        from luxar.cli.gsplat_ops.recipe_shared import survey_gsplat_streaming_layout

        store_path = tmp_path / "encoded-layout.gsplats.zarr"
        leaf = zarr.open_group(store_path, mode="w")
        leaf.attrs.update({"type": "gsplats", "n_splats": 4, "n_additive_sublods": 2})
        for index, (low, rows) in enumerate(((10.0, [0, 1]), (12.0, [0, 1]))):
            rung = leaf.create_group(f"additive_{index}")
            rung.attrs.update({"type": "gsplats", "n_splats": 2, "slice_dims": [3]})
            centers = rung.create_array(
                "centers",
                data=np.column_stack(
                    [
                        np.zeros((2, 3), dtype=np.uint16),
                        np.asarray(rows, dtype=np.uint16),
                    ]
                ),
            )
            centers.attrs["encoding"] = {
                "name": "linear_perchannel_u16",
                "col_lo": [0.0, 0.0, 0.0, low],
                "col_hi": [0.0, 0.0, 0.0, low + 1.0],
                "bits": 16,
                "original_dtype": "float32",
            }

        assert survey_gsplat_streaming_layout(store_path) == (4, 1)

    def test_store_survey_decodes_lut_coordinates(self, tmp_path: Path) -> None:
        from luxar.cli.gsplat_ops.recipe_shared import survey_gsplat_streaming_layout

        store_path = tmp_path / "lut-layout.gsplats.zarr"
        leaf = zarr.open_group(store_path, mode="w")
        leaf.attrs.update(
            {
                "type": "gsplats",
                "n_splats": 4,
                "n_additive_sublods": 1,
                "slice_dims": [3],
            }
        )
        centers = leaf.create_array(
            "centers",
            data=np.asarray(
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]],
                dtype=np.uint8,
            ),
        )
        centers.attrs["encoding"] = {
            "name": "lut_uint8",
            "lut": [0.0, 4.0],
            "original_dtype": "float32",
        }

        assert survey_gsplat_streaming_layout(store_path) == (2, 1)

    def test_store_survey_reads_zipped_store(self, tmp_path: Path) -> None:
        import shutil

        from luxar.cli.gsplat_ops.recipe_shared import survey_gsplat_streaming_layout

        store_path = tmp_path / "zipped-layout.gsplats.zarr"
        leaf = zarr.open_group(store_path, mode="w")
        leaf.attrs.update(
            {
                "type": "gsplats",
                "n_splats": 3,
                "n_additive_sublods": 1,
                "slice_dims": [3],
            }
        )
        leaf.create_array(
            "centers",
            data=np.asarray(
                [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]], dtype=np.float32
            ),
        )
        archive = Path(shutil.make_archive(str(store_path), "zip", root_dir=store_path))

        assert survey_gsplat_streaming_layout(archive) == (3, 1)

    def test_stored_stream_string_reparses_at_merge_time(self) -> None:
        """The manifest round-trip: the stored string re-parses via
        parse_lod_breakpoints into the same deferred spec."""
        from luxar.cli.gsplat_ops.recipe_shared import parse_lod_breakpoints

        assert parse_lod_breakpoints("stream:20833") == "stream:20833"

    def test_breakpoint_error_lists_equi_energy(self) -> None:
        from luxar.cli.gsplat_ops.recipe_shared import parse_lod_breakpoints

        with pytest.raises(typer.BadParameter, match="equi-energy:<n>"):
            parse_lod_breakpoints("equienergy:4")

    def test_target_ms_sized_with_true_merged_ndim(self) -> None:
        """merged_ndim=3 (single timepoint, 3 spatial) must size against the
        3D analytic figure (21 B), not the hardcoded 4D default (30 B)."""
        from luxar.cli.gsplat_ops.batch.planning import (
            MergeConfig,
            resolve_merge_recipe_args,
        )

        args = resolve_merge_recipe_args(
            MergeConfig(recipe="stream", target_ms=200.0), merged_ndim=3
        )
        # 200 ms @ 25 Mbps @ analytic 21 B (3D) → 625000/21 = 29762.
        assert args["breakpoints"] == "stream:29762"

    def test_target_ms_accounts_for_channel_colors(self) -> None:
        """A color-carrying multi-channel merge adds ~4 B/splat to the analytic
        estimate, shrinking the first chunk accordingly."""
        from luxar.cli.gsplat_ops.batch.planning import (
            MergeConfig,
            resolve_merge_recipe_args,
        )

        args = resolve_merge_recipe_args(
            MergeConfig(recipe="stream", target_ms=200.0),
            merged_ndim=4,
            merged_has_colors=True,
        )
        # 30 + 4 = 34 B/splat → 625000/34 = 18382 (< the colorless 20833).
        assert args["breakpoints"] == "stream:18382"

    def test_target_ms_and_breakpoints_mutually_exclusive(self) -> None:
        from luxar.cli.gsplat_ops.batch.planning import (
            MergeConfig,
            resolve_merge_recipe_args,
        )

        with pytest.raises(typer.BadParameter) as exc:
            resolve_merge_recipe_args(
                MergeConfig(recipe="stream", target_ms=200.0, breakpoints="equal-count")
            )
        # The plan-time surface must name the --merge-* spellings.
        assert "--merge-target-ms" in str(exc.value)
        assert "--merge-breakpoints" in str(exc.value)

    def test_bandwidth_knob_requires_target_ms(self) -> None:
        from luxar.cli.gsplat_ops.batch.planning import (
            MergeConfig,
            resolve_merge_recipe_args,
        )

        with pytest.raises(typer.BadParameter) as exc:
            resolve_merge_recipe_args(MergeConfig(recipe="stream", bandwidth_mbps=50.0))
        assert "--merge-bandwidth-mbps" in str(exc.value)
        assert "--merge-target-ms" in str(exc.value)

    def test_target_ms_orphaned_without_recipe(self) -> None:
        from luxar.cli.gsplat_ops.batch.planning import (
            MergeConfig,
            resolve_merge_recipe_args,
        )

        with pytest.raises(typer.BadParameter):
            resolve_merge_recipe_args(MergeConfig(target_ms=200.0))

    def test_target_ms_cross_recipe_rejected(self) -> None:
        from luxar.cli.gsplat_ops.batch.planning import (
            MergeConfig,
            resolve_merge_recipe_args,
        )

        with pytest.raises(typer.BadParameter):
            resolve_merge_recipe_args(MergeConfig(recipe="levels", target_ms=200.0))

    def test_merge_cli_target_ms_without_recipe_rejected(self, tmp_path: Path) -> None:
        """The `batch-fit merge` CLI rejects --target-ms without a recipe."""
        _write_manifest(tmp_path, merge_recipe=None)
        res = runner.invoke(
            app_gsplat,
            ["batch-fit", "merge", str(tmp_path), "--target-ms", "200"],
        )
        assert res.exit_code != 0
        assert "recipe-specific but no recipe is in effect" in normalized_cli_output(
            res
        )


# ── runtime: `batch-fit merge` CLI validation ──────────────────────────────


class TestBatchMergeCliValidation:
    def test_knob_without_recipe_rejected(self, tmp_path: Path) -> None:
        """--n-lods without a recipe (and no manifest recipe) → BadParameter.

        Pre-fix this was silently ignored and the merge proceeded; post-fix the
        usage error fires before any merge work and steers to --recipe."""
        _write_manifest(tmp_path, merge_recipe=None)
        res = runner.invoke(
            app_gsplat, ["batch-fit", "merge", str(tmp_path), "--n-lods", "6"]
        )
        assert res.exit_code != 0
        io = normalized_cli_output(res)
        assert "recipe-specific but no recipe is in effect" in io
        # No --no-recipe/--flat here, so the hint should steer to --recipe.
        assert "Pass --recipe" in io

    def test_refine_knobs_are_recipe_gated_like_every_other(
        self, tmp_path: Path
    ) -> None:
        """--refine/--refine-iters must join the knob-relevance gate.

        `stream` has no coarse levels to refine and a recipe-less merge writes bare
        leaves, so in both cases the knob does nothing. Plan time already rejects
        `--merge-refine` there; left out of the merge CLI's gate, the same request
        was accepted and then silently dropped — and for `volume` that is an
        expensive re-fit the user believes ran.
        """
        _write_manifest(tmp_path, merge_recipe=None)
        res = runner.invoke(
            app_gsplat, ["batch-fit", "merge", str(tmp_path), "--refine", "volume"]
        )
        assert res.exit_code != 0
        assert "recipe-specific but no recipe is in effect" in normalized_cli_output(
            res
        )

        res = runner.invoke(
            app_gsplat,
            [
                "batch-fit",
                "merge",
                str(tmp_path),
                "--recipe",
                "stream",
                "--refine-iters",
                "5",
            ],
        )
        assert res.exit_code != 0
        assert "not used by --recipe stream" in normalized_cli_output(res)

    def test_unknown_recipe_name_reported_even_with_a_knob(
        self, tmp_path: Path
    ) -> None:
        """A bad recipe NAME must be reported as such — even when a valid knob is
        also passed. Pre-fix, the unvalidated eff_recipe reached the knob-relevance
        helper, whose empty allowed-set misreported the VALID knob as 'not used by
        --recipe addative', hiding the real error (the typo'd recipe name). Mirrors
        `gsplat lod`'s RECIPE_NAMES guard."""
        _write_manifest(tmp_path, merge_recipe=None)
        res = runner.invoke(
            app_gsplat,
            [
                "batch-fit",
                "merge",
                str(tmp_path),
                "--recipe",
                "addative",
                "--n-lods",
                "6",
            ],
        )
        assert res.exit_code != 0
        io = normalized_cli_output(res)
        assert "unknown per-part recipe" in io
        assert "addative" in io
        # Must NOT misreport the valid knob as the problem.
        assert "not used by" not in io

    def test_unknown_recipe_name_without_knob_also_clean(self, tmp_path: Path) -> None:
        """Same clean 'unknown per-part recipe' error with no knob present — so the
        diagnostic no longer depends on whether an unrelated knob was passed."""
        _write_manifest(tmp_path, merge_recipe=None)
        res = runner.invoke(
            app_gsplat, ["batch-fit", "merge", str(tmp_path), "--recipe", "addative"]
        )
        assert res.exit_code != 0
        assert "unknown per-part recipe" in normalized_cli_output(res)

    def test_flat_and_recipe_mutually_exclusive(self, tmp_path: Path) -> None:
        """--flat with --recipe is a clean CLI usage error (the message is the
        CLI-specific one, proving it fires before merge_batch_results)."""
        _write_manifest(tmp_path, merge_recipe=None)
        res = runner.invoke(
            app_gsplat,
            ["batch-fit", "merge", str(tmp_path), "--flat", "--recipe", "stream"],
        )
        assert res.exit_code != 0
        assert (
            "concatenates all tiles into a single bare leaf"
            in normalized_cli_output(res)
        )

    def test_recipe_and_no_recipe_mutually_exclusive(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, merge_recipe=None)
        res = runner.invoke(
            app_gsplat,
            [
                "batch-fit",
                "merge",
                str(tmp_path),
                "--recipe",
                "stream",
                "--no-recipe",
            ],
        )
        assert res.exit_code != 0
        assert "mutually exclusive" in normalized_cli_output(res)

    def test_no_recipe_overrides_manifest_recipe(self, tmp_path: Path) -> None:
        """With a manifest merge_recipe set, --no-recipe forces a recipe-less
        merge: a knob that WOULD be valid for the manifest recipe is now rejected
        as orphaned (proving eff_recipe became None)."""
        _write_manifest(tmp_path, merge_recipe="additive")
        res = runner.invoke(
            app_gsplat,
            ["batch-fit", "merge", str(tmp_path), "--no-recipe", "--n-lods", "6"],
        )
        assert res.exit_code != 0
        # eff_recipe is None despite the manifest → the knob is orphaned.
        assert "recipe-specific but no recipe is in effect" in normalized_cli_output(
            res
        )

    def test_no_recipe_hint_does_not_contradict_the_flag(self, tmp_path: Path) -> None:
        """The orphaned-knob hint must NOT tell a user who typed --no-recipe to
        'Pass --recipe' (a direct contradiction). It must reference --no-recipe
        and the 'drop these knobs' resolution instead."""
        _write_manifest(tmp_path, merge_recipe=None)
        res = runner.invoke(
            app_gsplat,
            ["batch-fit", "merge", str(tmp_path), "--no-recipe", "--n-lods", "6"],
        )
        assert res.exit_code != 0
        io = normalized_cli_output(res)
        assert "--no-recipe forces a recipe-less" in io
        assert "drop these" in io
        # The bare contradictory steer ("Pass --recipe ...") must be absent.
        assert "Pass --recipe" not in io

    def test_flat_with_knob_hint_references_flat(self, tmp_path: Path) -> None:
        """Same contradiction guard for --flat (even more explicitly no-LOD)."""
        _write_manifest(tmp_path, merge_recipe=None)
        res = runner.invoke(
            app_gsplat, ["batch-fit", "merge", str(tmp_path), "--flat", "--n-lods", "6"]
        )
        assert res.exit_code != 0
        io = normalized_cli_output(res)
        assert "--flat forces a recipe-less" in io
        assert "Pass --recipe" not in io


# ── shared streaming-knob validation helper (all five CLI surfaces) ─────────


class TestValidateStreamingKnobsHelper:
    """The single shared validator behind `gsplat lod`, `gsplat additive`,
    `fit --recipe`, `batch-fit submit/run` (--merge- prefix) and
    `batch-fit merge` — one source of truth for the exclusion messages."""

    def test_default_prefix_messages(self) -> None:
        from luxar.cli.gsplat_ops.recipe_shared import validate_streaming_knobs

        with pytest.raises(typer.BadParameter) as exc:
            validate_streaming_knobs(200.0, None, None, "equal-count")
        assert "--target-ms and --breakpoints are mutually exclusive" in str(exc.value)
        with pytest.raises(typer.BadParameter) as exc:
            validate_streaming_knobs(None, 50.0, None, None)
        assert "--bandwidth-mbps/--bytes-per-splat only apply with" in str(exc.value)

    def test_merge_prefix_renames_options(self) -> None:
        from luxar.cli.gsplat_ops.recipe_shared import validate_streaming_knobs

        with pytest.raises(typer.BadParameter) as exc:
            validate_streaming_knobs(
                200.0, None, None, "equal-count", prefix="--merge-"
            )
        assert "--merge-target-ms and --merge-breakpoints" in str(exc.value)
        with pytest.raises(typer.BadParameter) as exc:
            validate_streaming_knobs(None, None, 45.0, None, prefix="--merge-")
        assert "--merge-bandwidth-mbps/--merge-bytes-per-splat" in str(exc.value)

    def test_valid_combinations_pass(self) -> None:
        from luxar.cli.gsplat_ops.recipe_shared import validate_streaming_knobs

        validate_streaming_knobs(None, None, None, None)
        validate_streaming_knobs(None, None, None, "equal-count")
        validate_streaming_knobs(200.0, 50.0, 45.0, None)


# ── plan-time sizing matches merge-time (batch-fit submit/run vs merge) ─────


class TestPlanTimeStreamingSizing:
    """`plan_batch` must resolve --merge-target-ms with the SAME merged-ndim /
    colors inputs `batch-fit merge` uses — pre-fix it always assumed the 4D
    analytic 45 B/splat, so a single-timepoint (3D) plan and its merge sized
    DIFFERENT ladders for the same data."""

    @staticmethod
    def _make_zarr(path: Path, shape: tuple[int, ...]) -> None:
        import numpy as np
        import zarr

        z = zarr.open_array(
            str(path),
            mode="w",
            shape=shape,
            chunks=(1,) * (len(shape) - 3) + shape[-3:],
            dtype="f4",
        )
        z[:] = np.zeros(shape, np.float32)

    def _plan(self, tmp_path: Path, shape: tuple[int, ...], axes: list[str], merge):
        from luxar.cli.gsplat_ops.batch.planning import (
            ContentKnobs,
            DenoiseConfig,
            FitConfig,
            plan_batch,
        )

        src = tmp_path / "vol.zarr"
        self._make_zarr(src, shape)
        return plan_batch(
            input_path=src,
            output_dir=tmp_path / "out",
            tiling="uniform",
            tile_size=64,  # > 8 → single tile, no GPU profile needed
            tile_overlap=0,
            axes_list=axes,
            array_key=None,
            timepoints_slice=None,
            channels_slice=None,
            fit=FitConfig(),
            denoise=DenoiseConfig(),
            content=ContentKnobs(),
            merge=merge,
        )

    def test_multi_timepoint_plans_4d_ladder(self, tmp_path: Path) -> None:
        from luxar.cli.gsplat_ops.batch.planning import MergeConfig

        plan = self._plan(
            tmp_path,
            (2, 8, 8, 8),
            ["t", "z", "y", "x"],
            MergeConfig(recipe="stream", target_ms=200.0),
        )
        # The whole-node 20,833-splat budget is doubled for two hidden slices;
        # one partition part receives the resulting 41,666-splat first rung.
        assert plan.manifest.merge_recipe_args["breakpoints"] == "stream:41666"

    def test_invalid_merge_knobs_fail_before_plan_output(self, tmp_path: Path) -> None:
        from luxar.cli.gsplat_ops.batch.planning import MergeConfig

        with pytest.raises(typer.BadParameter, match="require a --merge-recipe"):
            self._plan(
                tmp_path,
                (2, 8, 8, 8),
                ["t", "z", "y", "x"],
                MergeConfig(recipe=None, n_lods=6),
            )
        assert not (tmp_path / "out").exists()

    def test_single_timepoint_plans_3d_ladder(self, tmp_path: Path) -> None:
        from luxar.cli.gsplat_ops.batch.planning import MergeConfig

        plan = self._plan(
            tmp_path,
            (1, 8, 8, 8),
            ["t", "z", "y", "x"],
            MergeConfig(recipe="stream", target_ms=200.0),
        )
        # No stacked axis → merged ndim 3 → 21 B → 29762 (pre-fix: 20833).
        assert plan.manifest.merge_recipe_args["breakpoints"] == "stream:29762"

    def test_multichannel_color_merge_plans_color_bytes(self, tmp_path: Path) -> None:
        from luxar.cli.gsplat_ops.batch.planning import MergeConfig

        plan = self._plan(
            tmp_path,
            (2, 2, 8, 8, 8),
            ["t", "c", "z", "y", "x"],
            MergeConfig(
                recipe="stream",
                target_ms=200.0,
                channel_colors="#ff0080,#00ff00",
            ),
        )
        # Colors will be written per-splat → 34 B → 18,382 for the whole node,
        # then doubled for the two hidden time slices.
        assert plan.manifest.merge_recipe_args["breakpoints"] == "stream:36764"


# ── merge-time --target-ms: measured from completed tiles, analytic fallback ─


class TestMergeTargetMsBytesSource:
    def test_measure_tiles_bytes_per_splat(self, tmp_path: Path) -> None:
        """Bytes summed over completed tile stores / stored splat counts read
        from every centers/.zarray — missing tiles are skipped."""
        import json

        from luxar.cli.gsplat_ops.batch.measurement import (
            measure_tiles_bytes_per_splat as _measure_tiles_bytes_per_splat,
        )

        tiles = tmp_path / "tiles"
        tile = tiles / "t00_c00_tile000.gsplats.zarr"
        (tile / "centers").mkdir(parents=True)
        zarray = json.dumps({"shape": [100, 3]})
        (tile / "centers" / ".zarray").write_text(zarray)
        (tile / "centers" / "0.0").write_bytes(b"\0" * 4000)
        bps, n = _measure_tiles_bytes_per_splat(
            tiles, ["t00_c00_tile000.gsplats.zarr", "missing.gsplats.zarr"]
        )
        assert n == 1
        assert bps == pytest.approx((4000 + len(zarray)) / 100)

    def test_measure_tiles_none_when_nothing_on_disk(self, tmp_path: Path) -> None:
        from luxar.cli.gsplat_ops.batch.measurement import (
            measure_tiles_bytes_per_splat as _measure_tiles_bytes_per_splat,
        )

        bps, n = _measure_tiles_bytes_per_splat(tmp_path / "tiles", ["a.gsplats.zarr"])
        assert bps is None and n == 0

    def test_merge_cli_measures_from_completed_tiles(self, tmp_path: Path) -> None:
        """With a completed tile on disk, `batch-fit merge --target-ms` sizes
        against the MEASURED bytes/splat and says so (the merge itself then
        fails on the stub tile, but the sizing log has already fired)."""
        import json

        name = "t00_c00_tile000.gsplats.zarr"
        _write_manifest(tmp_path, merge_recipe="additive", tile_names=[name])
        tile = tmp_path / "tiles" / name
        (tile / "centers").mkdir(parents=True)
        (tile / "centers" / ".zarray").write_text(json.dumps({"shape": [1000, 3]}))
        (tile / "centers" / "0.0").write_bytes(b"\0" * 45_000)
        res = runner.invoke(
            app_gsplat,
            ["batch-fit", "merge", str(tmp_path), "--target-ms", "200"],
        )
        io = normalized_cli_output(res)
        assert "measured from 1 completed tile store(s)" in io
        assert "stream:" in io

    def test_merge_cli_falls_back_to_analytic(self, tmp_path: Path) -> None:
        """No completed tiles → the analytic estimate for the merged parts,
        with the merged ndim derived from the manifest (3 spatial, T=1 → 3D)."""
        _write_manifest(tmp_path, merge_recipe="additive")
        res = runner.invoke(
            app_gsplat,
            ["batch-fit", "merge", str(tmp_path), "--target-ms", "200"],
        )
        io = normalized_cli_output(res)
        assert "analytic estimate" in io
        # 3D analytic 21 B → 29762 (the same figure plan-time now derives).
        assert "stream:29762" in io
