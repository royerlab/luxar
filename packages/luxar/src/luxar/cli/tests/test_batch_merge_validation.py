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


def _write_manifest(output_dir: Path, *, merge_recipe: str | None = None) -> None:
    """Write a minimal BatchManifest (enough for `batch-fit merge` to load)."""
    from luxar.gsplats.batch.manifest import BatchManifest, save_manifest

    manifest = BatchManifest(
        input_path="/data/test.zarr",
        output_dir=str(output_dir),
        n_timepoints=1,
        n_channels=1,
        spatial_shape=(64, 64, 64),
        tile_size=64,
        n_tiles=1,
        total_tasks=1,
        merge_recipe=merge_recipe,
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

        args = resolve_merge_recipe_args(MergeConfig(recipe="additive", n_lods=6))
        assert args == {"n-lods": "6"}

    def test_cross_recipe_knob_still_rejected(self) -> None:
        from luxar.cli.gsplat_ops.batch_planning import (
            MergeConfig,
            resolve_merge_recipe_args,
        )

        with pytest.raises(typer.BadParameter):
            resolve_merge_recipe_args(
                MergeConfig(recipe="additive", compression_factor=4)
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
        io = _io(res)
        assert "recipe-specific but no recipe is in effect" in io
        # No --no-recipe/--flat here, so the hint should steer to --recipe.
        assert "Pass --recipe" in io

    def test_unknown_recipe_name_reported_even_with_a_knob(self, tmp_path: Path) -> None:
        """A bad recipe NAME must be reported as such — even when a valid knob is
        also passed. Pre-fix, the unvalidated eff_recipe reached the knob-relevance
        helper, whose empty allowed-set misreported the VALID knob as 'not used by
        --recipe addative', hiding the real error (the typo'd recipe name). Mirrors
        `gsplat lod`'s RECIPE_NAMES guard."""
        _write_manifest(tmp_path, merge_recipe=None)
        res = runner.invoke(
            app_gsplat,
            ["batch-fit", "merge", str(tmp_path), "--recipe", "addative", "--n-lods", "6"],
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
            ["batch-fit", "merge", str(tmp_path), "--flat", "--recipe", "additive"],
        )
        assert res.exit_code != 0
        assert "concatenates all tiles into a single bare leaf" in _io(res)

    def test_recipe_and_no_recipe_mutually_exclusive(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, merge_recipe=None)
        res = runner.invoke(
            app_gsplat,
            ["batch-fit", "merge", str(tmp_path), "--recipe", "additive", "--no-recipe"],
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
