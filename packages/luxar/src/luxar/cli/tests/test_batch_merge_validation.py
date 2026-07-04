"""CLI + plan-time validation tests for `batch-fit merge` recipe knobs.

Guards the friction fixes: recipe-specific merge knobs must be *rejected* (not
silently ignored) when given without a matching recipe, `--no-recipe` must be
able to override a manifest's `merge_recipe`, and `--flat` must be mutually
exclusive with `--recipe` at the CLI boundary (not only deep in the library).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from luxar.cli.gsplat_commands import app_gsplat

runner = CliRunner()

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# Rich renders typer.BadParameter inside a box and HARD-WRAPS the message across
# lines with border glyphs — so a multi-word phrase is split by "│\n│". Strip the
# box-drawing glyphs and collapse all whitespace so substring matches survive
# wrapping (Rich wraps at spaces, so collapsed text reconstructs the prose).


def _io(result) -> str:
    out = result.stdout or ""
    try:
        err = result.stderr or ""
    except (ValueError, AttributeError):
        err = ""
    text = _ANSI.sub("", out + err)
    text = re.sub(r"[─│╭╮╰╯┄┆]", " ", text)
    return re.sub(r"\s+", " ", text)


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
        from luxar.cli.gsplat_ops.batch_planning import (
            MergeConfig,
            resolve_merge_recipe_args,
        )

        with pytest.raises(typer.BadParameter):
            resolve_merge_recipe_args(MergeConfig(recipe=None, n_lods=6))

    def test_no_knobs_no_recipe_returns_empty(self) -> None:
        from luxar.cli.gsplat_ops.batch_planning import (
            MergeConfig,
            resolve_merge_recipe_args,
        )

        assert resolve_merge_recipe_args(MergeConfig()) == {}

    def test_valid_recipe_with_knobs_resolves(self) -> None:
        from luxar.cli.gsplat_ops.batch_planning import (
            MergeConfig,
            resolve_merge_recipe_args,
        )

        args = resolve_merge_recipe_args(MergeConfig(recipe="stream", n_lods=6))
        assert args == {"n-lods": "6"}

    def test_cross_recipe_knob_still_rejected(self) -> None:
        from luxar.cli.gsplat_ops.batch_planning import (
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

    def test_merge_rejects_refine_volume_front_door(self, tmp_path) -> None:
        """refine='volume' must be rejected BEFORE the streaming merge writes
        anything (the deep per-part rejection would fire mid-stream, leaving a
        half-written store). Fails pre-fix (error surfaced only mid-merge)."""
        import numpy as np
        import pytest

        from luxar.gsplats.batch.manifest import BatchManifest
        from luxar.gsplats.batch.merge_orchestrator import merge_batch_results
        from luxar.gsplats.lod.recipes import RecipeParams

        manifest = BatchManifest(n_timepoints=1, n_channels=1, n_tiles=1)
        out_dir = tmp_path / "batch"
        (out_dir / "tiles").mkdir(parents=True)
        with pytest.raises(ValueError, match="volume-free by design"):
            merge_batch_results(
                manifest,
                out_dir,
                verbose=False,
                recipe="levels",
                recipe_params=RecipeParams(
                    refine="volume", volume=np.zeros((4, 4, 4), np.float32)
                ),
            )
        # Nothing was written before the rejection.
        assert not (out_dir / "merged").exists()

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
        from luxar.cli.gsplat_ops.batch_planning import (
            MergeConfig,
            resolve_merge_recipe_args,
        )

        args = resolve_merge_recipe_args(
            MergeConfig(recipe="stream", target_ms=200.0), merged_ndim=4
        )
        bp = args["breakpoints"]
        assert bp.startswith("stream:")
        # 200 ms @ default 25 Mbps @ analytic 45 B (4D) → 13889.
        assert bp == "stream:13889"

    def test_stored_stream_string_reparses_at_merge_time(self) -> None:
        """The manifest round-trip: the stored string re-parses via
        _parse_lod_breakpoints into the same deferred spec."""
        from luxar.cli.lod import _parse_lod_breakpoints

        assert _parse_lod_breakpoints("stream:13889") == "stream:13889"

    def test_target_ms_sized_with_true_merged_ndim(self) -> None:
        """merged_ndim=3 (single timepoint, 3 spatial) must size against the
        3D analytic figure (30 B), not the hardcoded 4D default (45 B)."""
        from luxar.cli.gsplat_ops.batch_planning import (
            MergeConfig,
            resolve_merge_recipe_args,
        )

        args = resolve_merge_recipe_args(
            MergeConfig(recipe="stream", target_ms=200.0), merged_ndim=3
        )
        # 200 ms @ 25 Mbps @ analytic 30 B (3D) → 625000/30 = 20833.
        assert args["breakpoints"] == "stream:20833"

    def test_target_ms_accounts_for_channel_colors(self) -> None:
        """A color-carrying multi-channel merge adds ~4 B/splat to the analytic
        estimate, shrinking the first chunk accordingly."""
        from luxar.cli.gsplat_ops.batch_planning import (
            MergeConfig,
            resolve_merge_recipe_args,
        )

        args = resolve_merge_recipe_args(
            MergeConfig(recipe="stream", target_ms=200.0),
            merged_ndim=4,
            merged_has_colors=True,
        )
        # 45 + 4 = 49 B/splat → 625000/49 = 12755 (< the colorless 13889).
        assert args["breakpoints"] == "stream:12755"

    def test_target_ms_and_breakpoints_mutually_exclusive(self) -> None:
        from luxar.cli.gsplat_ops.batch_planning import (
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
        from luxar.cli.gsplat_ops.batch_planning import (
            MergeConfig,
            resolve_merge_recipe_args,
        )

        with pytest.raises(typer.BadParameter) as exc:
            resolve_merge_recipe_args(MergeConfig(recipe="stream", bandwidth_mbps=50.0))
        assert "--merge-bandwidth-mbps" in str(exc.value)
        assert "--merge-target-ms" in str(exc.value)

    def test_target_ms_orphaned_without_recipe(self) -> None:
        from luxar.cli.gsplat_ops.batch_planning import (
            MergeConfig,
            resolve_merge_recipe_args,
        )

        with pytest.raises(typer.BadParameter):
            resolve_merge_recipe_args(MergeConfig(target_ms=200.0))

    def test_target_ms_cross_recipe_rejected(self) -> None:
        from luxar.cli.gsplat_ops.batch_planning import (
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
        assert "recipe-specific but no recipe is in effect" in _io(res)


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
        io = _io(res)
        assert "recipe-specific but no recipe is in effect" in io
        # No --no-recipe/--flat here, so the hint should steer to --recipe.
        assert "Pass --recipe" in io

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
        io = _io(res)
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
        assert "unknown per-part recipe" in _io(res)

    def test_flat_and_recipe_mutually_exclusive(self, tmp_path: Path) -> None:
        """--flat with --recipe is a clean CLI usage error (the message is the
        CLI-specific one, proving it fires before merge_batch_results)."""
        _write_manifest(tmp_path, merge_recipe=None)
        res = runner.invoke(
            app_gsplat,
            ["batch-fit", "merge", str(tmp_path), "--flat", "--recipe", "stream"],
        )
        assert res.exit_code != 0
        assert "concatenates all tiles into a single bare leaf" in _io(res)

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
        assert "mutually exclusive" in _io(res)

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
        assert "recipe-specific but no recipe is in effect" in _io(res)

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
        io = _io(res)
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
        io = _io(res)
        assert "--flat forces a recipe-less" in io
        assert "Pass --recipe" not in io


# ── shared streaming-knob validation helper (all five CLI surfaces) ─────────


class TestValidateStreamingKnobsHelper:
    """The single shared validator behind `gsplat lod`, `gsplat additive`,
    `fit --recipe`, `batch-fit submit/run` (--merge- prefix) and
    `batch-fit merge` — one source of truth for the exclusion messages."""

    def test_default_prefix_messages(self) -> None:
        from luxar.cli.lod import validate_streaming_knobs

        with pytest.raises(typer.BadParameter) as exc:
            validate_streaming_knobs(200.0, None, None, "equal-count")
        assert "--target-ms and --breakpoints are mutually exclusive" in str(exc.value)
        with pytest.raises(typer.BadParameter) as exc:
            validate_streaming_knobs(None, 50.0, None, None)
        assert "--bandwidth-mbps/--bytes-per-splat only apply with" in str(exc.value)

    def test_merge_prefix_renames_options(self) -> None:
        from luxar.cli.lod import validate_streaming_knobs

        with pytest.raises(typer.BadParameter) as exc:
            validate_streaming_knobs(
                200.0, None, None, "equal-count", prefix="--merge-"
            )
        assert "--merge-target-ms and --merge-breakpoints" in str(exc.value)
        with pytest.raises(typer.BadParameter) as exc:
            validate_streaming_knobs(None, None, 45.0, None, prefix="--merge-")
        assert "--merge-bandwidth-mbps/--merge-bytes-per-splat" in str(exc.value)

    def test_valid_combinations_pass(self) -> None:
        from luxar.cli.lod import validate_streaming_knobs

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
        from luxar.cli.gsplat_ops.batch_planning import (
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
        from luxar.cli.gsplat_ops.batch_planning import MergeConfig

        plan = self._plan(
            tmp_path,
            (2, 8, 8, 8),
            ["t", "z", "y", "x"],
            MergeConfig(recipe="stream", target_ms=200.0),
        )
        # merged ndim = 3 spatial + stacked-timepoint axis = 4 → 45 B → 13889:
        # exactly what `batch-fit merge --target-ms 200` derives from the
        # manifest (len(spatial_shape) + (n_timepoints > 1)).
        assert plan.manifest.merge_recipe_args["breakpoints"] == "stream:13889"

    def test_single_timepoint_plans_3d_ladder(self, tmp_path: Path) -> None:
        from luxar.cli.gsplat_ops.batch_planning import MergeConfig

        plan = self._plan(
            tmp_path,
            (1, 8, 8, 8),
            ["t", "z", "y", "x"],
            MergeConfig(recipe="stream", target_ms=200.0),
        )
        # No stacked axis → merged ndim 3 → 30 B → 20833 (pre-fix: 13889).
        assert plan.manifest.merge_recipe_args["breakpoints"] == "stream:20833"

    def test_multichannel_color_merge_plans_color_bytes(self, tmp_path: Path) -> None:
        from luxar.cli.gsplat_ops.batch_planning import MergeConfig

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
        # Colors will be written per-splat → 49 B → 12755 (pre-fix: 13889).
        assert plan.manifest.merge_recipe_args["breakpoints"] == "stream:12755"


# ── merge-time --target-ms: measured from completed tiles, analytic fallback ─


class TestMergeTargetMsBytesSource:
    def test_measure_tiles_bytes_per_splat(self, tmp_path: Path) -> None:
        """Bytes summed over completed tile stores / stored splat counts read
        from every centers/.zarray — missing tiles are skipped."""
        import json

        from luxar.cli.gsplat_ops.batch import _measure_tiles_bytes_per_splat

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
        from luxar.cli.gsplat_ops.batch import _measure_tiles_bytes_per_splat

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
        io = _io(res)
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
        io = _io(res)
        assert "analytic estimate" in io
        # 3D analytic 30 B → 20833 (the same figure plan-time now derives).
        assert "stream:20833" in io
