"""``luxar restamp-lod`` at the CLI boundary.

Lives here rather than beside :mod:`luxar.io.lod_restamp`'s own tests because
the layering contract forbids a domain package importing ``luxar.cli``. What is
checked here is only what the command adds over
:func:`~luxar.io.lod_restamp.restamp_lod_store`: the exit code (so it can gate a
pipeline), the audit trail it prints, and the argument plumbing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
from typer.testing import CliRunner

from luxar._zarr_compat import close, consolidate, create_root_group
from luxar.cli import app
from luxar.core.dimensions import Dimensions
from luxar.io.compiler import LuxarZarrCompiler
from luxar.typing_utils.constants import DERIVED_LOD_SELECTOR, LEGACY_LOD_SELECTOR

runner = CliRunner()

LEGACY_LADDER = [0.0, 4.0]


def _legacy_scene(tmp_path: Path, name: str = "scene.luxar.zarr") -> Path:
    """A compiled scene with one legacy whole-object ladder and one tiled pair."""
    store = tmp_path / name
    rng = np.random.default_rng(0)
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        tiled = scene.add_partition_group(
            "tiled", display_type="points", max_elements=100_000
        )
        for target, node in ((scene, "pts"), (tiled, "part_0"), (tiled, "part_1")):
            target.add_points(
                node,
                rng.normal(0, 20, (400, 3)).astype(np.float32),
                radii=0.5,
                substitutive_lod=dict(
                    levels=1, device="cpu", seed=0, coverage_fractions=LEGACY_LADDER
                ),
            )
    return store


def _node_attrs(store: Path) -> Dict[str, Dict[str, Any]]:
    from luxar._zarr_compat import read_consolidated_attrs

    return dict(read_consolidated_attrs(store))


def test_a_clean_run_exits_zero_and_prints_the_audit_trail(tmp_path: Path) -> None:
    """The old→new ladder is the whole point of the output: this command can be
    overriding an author's deliberate list, so the rewrite has to be legible."""
    store = _legacy_scene(tmp_path)

    result = runner.invoke(app, ["restamp-lod", str(store)])

    assert result.exit_code == 0, result.output
    assert "[0, 4] → [0, 0.5]" in result.output
    assert "[0, 4] → [0, 1]" in result.output, "the tile anchor must be visible too"
    # Anchor pinned TO ITS GROUP on one line: asserting the two labels appear
    # somewhere in the output passes just as happily when they are swapped, and
    # a swapped anchor is exactly the mistake worth catching.
    assert "pts: anchor whole-object 0.5" in result.output
    assert "tiled/part_0: anchor fills-screen (tile) 1" in result.output
    assert "tiled/part_1: anchor fills-screen (tile) 1" in result.output
    assert _node_attrs(store)["pts"]["selector"] == DERIVED_LOD_SELECTOR


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    store = _legacy_scene(tmp_path)
    before = _node_attrs(store)

    result = runner.invoke(app, ["restamp-lod", str(store), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Nothing was written." in result.output
    assert _node_attrs(store) == before


def test_group_is_repeatable(tmp_path: Path) -> None:
    store = _legacy_scene(tmp_path)

    result = runner.invoke(
        app,
        ["restamp-lod", str(store), "--group", "pts", "--group", "tiled/part_1"],
    )

    assert result.exit_code == 0, result.output
    attrs = _node_attrs(store)
    assert attrs["pts"]["selector"] == DERIVED_LOD_SELECTOR
    assert attrs["tiled/part_1"]["selector"] == DERIVED_LOD_SELECTOR
    assert attrs["tiled/part_0"]["selector"] == LEGACY_LOD_SELECTOR


def test_anchor_rederives_an_already_current_ladder(tmp_path: Path) -> None:
    store = _legacy_scene(tmp_path)
    assert runner.invoke(app, ["restamp-lod", str(store)]).exit_code == 0

    result = runner.invoke(
        app,
        ["restamp-lod", str(store), "--anchor", "0.25", "--group", "pts"],
    )

    assert result.exit_code == 0, result.output
    assert "[0, 0.5] → [0, 0.25]" in result.output
    assert "pts: anchor whole-object 0.25" in result.output
    assert _node_attrs(store)["pts/child_1"]["coverage_fraction"] == 0.25
    assert _node_attrs(store)["tiled/part_0/child_1"]["coverage_fraction"] == 1.0


def test_invalid_anchor_exits_one_without_writing(tmp_path: Path) -> None:
    store = _legacy_scene(tmp_path)
    before = _node_attrs(store)

    result = runner.invoke(app, ["restamp-lod", str(store), "--anchor", "nan"])

    assert result.exit_code == 1, result.output
    assert "--anchor must be finite and in" in result.output
    assert _node_attrs(store) == before


def test_an_unmatched_group_exits_one(tmp_path: Path) -> None:
    store = _legacy_scene(tmp_path)
    before = _node_attrs(store)

    result = runner.invoke(app, ["restamp-lod", str(store), "--group", "no/such"])

    assert result.exit_code == 1
    assert "not a kind=lod group" in result.output
    assert _node_attrs(store) == before


def test_a_missing_path_exits_one(tmp_path: Path) -> None:
    result = runner.invoke(app, ["restamp-lod", str(tmp_path / "gone.luxar.zarr")])

    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_a_non_luxar_store_exits_one(tmp_path: Path) -> None:
    import zarr

    store = tmp_path / "foreign.zarr"
    root = create_root_group(zarr.storage.LocalStore(str(store)))
    root.attrs["not"] = "ours"
    consolidate(root)
    close(root)

    result = runner.invoke(app, ["restamp-lod", str(store)])

    assert result.exit_code == 1
    assert "does not look like a Luxar" in result.output


def test_an_unsupported_selector_exits_one_but_still_writes_the_rest(
    tmp_path: Path,
) -> None:
    """A mixed store is repaired as far as it can be, and the exit code says so.

    Silence would be the wrong answer in both directions: refusing outright
    would strand the convertible ladders, and exiting 0 would let a pipeline
    believe the whole store had been migrated.
    """
    import zarr

    store = tmp_path / "mixed.luxar.zarr"
    root = create_root_group(zarr.storage.LocalStore(str(store)))
    root.attrs["type"] = "scene"
    for name, selector in (("ok", LEGACY_LOD_SELECTOR), ("stale", "pixel_size")):
        lod = root.create_group(name)
        lod.attrs.update(
            {
                "type": "group",
                "kind": "lod",
                "display_type": "points",
                "selector": selector,
            }
        )
        for index, (fraction, count) in enumerate(((0.0, 100), (4.0, 400))):
            child = lod.create_group(f"child_{index}")
            child.attrs.update(
                {
                    "type": "points",
                    "child_index": index,
                    "coverage_fraction": fraction,
                    "n_points": count,
                }
            )
    consolidate(root)
    close(root)

    refused = runner.invoke(app, ["restamp-lod", str(store), "--group", "stale"])

    assert refused.exit_code == 1, refused.output
    assert "Nothing was restamped — see below." in refused.output
    assert "every LOD ladder is already current" not in refused.output

    result = runner.invoke(app, ["restamp-lod", str(store)])

    assert result.exit_code == 1, result.output
    assert "migrate-format" in result.output
    attrs = _node_attrs(store)
    assert attrs["ok"]["selector"] == DERIVED_LOD_SELECTOR
    assert attrs["stale"]["selector"] == "pixel_size"
    assert attrs["stale/child_1"]["coverage_fraction"] == 4.0


def test_a_re_verification_residual_exits_one(tmp_path: Path, monkeypatch: Any) -> None:
    """The end of the verify → report → exit-code chain, on a REAL failure.

    The store is left with a stale consolidated index (``consolidate`` neutered
    for this run), which is the silent failure the read-back exists to catch:
    the per-node documents are right and the only document the viewer fetches is
    not. A pipeline gates on this exit code, so it is the link that has to hold.
    """
    from luxar.io import lod_restamp

    store = _legacy_scene(tmp_path)
    monkeypatch.setattr(lod_restamp, "consolidate", lambda group: None)

    result = runner.invoke(app, ["restamp-lod", str(store)])

    assert result.exit_code == 1, result.output
    assert "verify:" in result.output
    assert "consolidated index" in result.output


def test_a_store_with_no_digest_to_restamp_exits_one(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The ladders land, and the run still fails — because no client will see them.

    A ``kind=partition`` root carries neither a scene ``type`` nor a
    ``.gsplats.zarr`` ``content_hash``, so there is no digest to move; at zarr
    format 2, the legacy corpus's format, the viewer's ``zattrs-hash`` fallback
    digests the root ``.zattrs``, which a child's ladder edit does not touch
    either. A pipeline gating on this exit code has to learn that the store needs
    republishing under a new URL prefix, not that everything is fine.
    """
    import zarr

    from luxar import _zarr_compat

    monkeypatch.setattr(_zarr_compat, "ZARR_FORMAT", 2)
    store = tmp_path / "parts.luxar.zarr"
    root = create_root_group(zarr.storage.LocalStore(str(store)))
    root.attrs.update({"kind": "partition", "display_type": "points"})
    for part in (0, 1):
        lod = root.create_group(f"part_{part}")
        lod.attrs.update(
            {
                "type": "group",
                "kind": "lod",
                "display_type": "points",
                "selector": LEGACY_LOD_SELECTOR,
            }
        )
        for index, (fraction, count) in enumerate(((0.0, 100), (4.0, 400))):
            child = lod.create_group(f"child_{index}")
            child.attrs.update(
                {
                    "type": "points",
                    "child_index": index,
                    "coverage_fraction": fraction,
                    "n_points": count,
                }
            )
    consolidate(root)
    close(root)

    dry_result = runner.invoke(app, ["restamp-lod", str(store), "--dry-run"])

    assert dry_result.exit_code == 1, dry_result.output
    assert "no digest to restamp" in dry_result.output
    assert "Nothing was written" in dry_result.output
    assert _node_attrs(store)["part_0"]["selector"] == LEGACY_LOD_SELECTOR

    result = runner.invoke(app, ["restamp-lod", str(store)])

    assert result.exit_code == 1, result.output
    assert "no digest to restamp" in result.output
    assert "at zarr format 2" in result.output
    assert "new URL prefix" in result.output
    attrs = _node_attrs(store)
    assert attrs["part_0"]["selector"] == DERIVED_LOD_SELECTOR, (
        "the rewrite itself must still have landed"
    )


def test_a_second_run_exits_zero_and_reports_a_no_op(tmp_path: Path) -> None:
    store = _legacy_scene(tmp_path)
    assert runner.invoke(app, ["restamp-lod", str(store)]).exit_code == 0

    result = runner.invoke(app, ["restamp-lod", str(store)])

    assert result.exit_code == 0, result.output
    assert "Nothing to restamp" in result.output
