"""The CMU-1 channel order is load-bearing, so the guard that checks it is gated.

``create_luxar_scene`` assigns the red/green/blue colormaps by POSITION in the
resolved path list, so a channel arriving out of order paints hematoxylin red —
a wrong picture that renders perfectly happily. ``resolve_data`` refuses that
case, but its only live trigger is a real run against the hosted 114 MB
artifacts, and demos are omitted from coverage: the refusal had no test at all.

The refusal cases cover both shapes the guard exists for, not just the tidy one:

* a **permutation** of the three channel paths — the mis-colouring the guard is
  named for, though the generator's ``sorted(...)`` glob means no committed
  manifest can actually produce it;
* an **extra sidecar** entry, which is the REACHABLE trigger: two sibling
  datasets already ship exactly such a file next to their fit
  (``gsplats_ct_totalsegmentator`` → ``ct_atlas_labels.npz``,
  ``gsplats_visible_human_head`` → ``vh_head_colors.npz``). A sidecar that sorts
  *before* ``cmu1_ch0`` is the dangerous one — it shifts every channel one
  position along the positional colormap list — so it is exercised separately
  from one appended at the end.

The last test is the valuable one. It lifts the check off the runtime path
entirely by asserting the packaged ``data_manifest.json`` still lists exactly the
three channel files the demo expects, in order — so a manifest edit that would
make the demo die (or, worse, mis-colour) is caught in CI instead of on someone's
machine after a download.
"""

from __future__ import annotations

import importlib
import json
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from luxar.demos import registry
from luxar.gsplats.gsplat_data import GSplatData

_MANIFEST = registry._DEMOS_DIR / "data_manifest.json"

demo = importlib.import_module("luxar.demos.demo_gsplats_2d_cmu1_pathology")

# A sidecar named like this sorts BEFORE ``cmu1_ch0.gsplats.zarr.zip`` under the
# generator's ``sorted(...)`` glob, so it lands at index 0 and shifts every
# channel one position along ``CHANNEL_COLORMAPS``.
_SIDECAR_FIRST = Path("/nowhere/cmu1_atlas_labels.npz")
_SIDECAR_LAST = Path("/nowhere/cmu1_thumbnail.npz")


def _channel_paths() -> list[Path]:
    """The names ``ensure_dataset`` is contracted to hand back, in order."""
    return [
        Path(f"/nowhere/cmu1_ch{i}.gsplats.zarr.zip") for i in range(demo.N_CHANNELS)
    ]


@pytest.mark.parametrize(
    "corrupt",
    [
        # Rotate rather than index: a hardcoded paths[2] would raise IndexError
        # instead of failing readably if N_CHANNELS ever dropped to 2.
        pytest.param(lambda paths: paths[-1:] + paths[:-1], id="rotated_channels"),
        pytest.param(lambda paths: [*paths, _SIDECAR_LAST], id="sidecar_appended"),
        pytest.param(lambda paths: [_SIDECAR_FIRST, *paths], id="sidecar_sorts_first"),
    ],
)
def test_resolve_data_refuses_anything_but_the_exact_channel_list(
    monkeypatch: pytest.MonkeyPatch,
    corrupt: Callable[[list[Path]], list[Path]],
) -> None:
    bad = corrupt(_channel_paths())
    monkeypatch.setattr(demo, "ensure_dataset", lambda _name: bad)

    with pytest.raises(RuntimeError, match="does not match the expected"):
        demo.resolve_data()


def test_resolve_data_accepts_the_manifest_order_unchanged(monkeypatch) -> None:
    """Positive control: without it, the refusal above could pass for any reason."""
    paths = _channel_paths()
    monkeypatch.setattr(demo, "ensure_dataset", lambda _name: paths)

    assert demo.resolve_data() == paths


def test_the_packaged_manifest_lists_the_channels_the_demo_expects() -> None:
    datasets = json.loads(_MANIFEST.read_text())["datasets"]
    files = [f["name"] for f in datasets[demo.DATASET]["files"]]

    assert files == [f"cmu1_ch{i}.gsplats.zarr.zip" for i in range(demo.N_CHANNELS)], (
        f"{demo.DATASET} no longer lists exactly the three channel fits in "
        f"channel order, so the demo's resolve_data() would refuse it: {files}"
    )


def test_local_tiled_refit_preserves_overlap_contributions(
    tmp_path, monkeypatch
) -> None:
    """A post-fit amplitude cull removes the weak halves of both Hann windows."""
    monkeypatch.setattr(demo, "TILE_SIZE", 32)
    monkeypatch.setattr(demo, "OVERLAP", 8)
    monkeypatch.setattr(demo, "SEEDS_PER_TILE", 128)
    monkeypatch.setattr(demo, "N_ITERS", 20)
    monkeypatch.setattr(demo, "DEVICE", "cpu")

    y, x = np.mgrid[:24, :48]
    image = np.exp(-((y - 12) ** 2 / 50 + (x - 24) ** 2 / 450)).astype(np.float32)
    result = demo.fit_channel_tiled(image, "red", tmp_path / "red.gsplats.zarr.zip")

    assert result.n_splats == 256  # Both tiles keep their full seed budget.
    assert result.stats.get("culled") is not True
    reconstruction = result.render_to_volume(shape=image.shape, device="cpu")
    seam_mse = np.mean((reconstruction[:, 24:32] - image[:, 24:32]) ** 2)
    assert seam_mse < 0.016  # The culled fit measures about 0.019 here.


def test_local_fit_paths_rejects_a_truncated_channel(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(demo, "local_fit_path", lambda _dataset, name: tmp_path / name)
    for name in demo.GSPLATS_FILES:
        with zipfile.ZipFile(tmp_path / name, "w"):
            pass
    (tmp_path / demo.GSPLATS_FILES[1]).write_bytes(b"a Ctrl-C mid-save, not a zip")

    assert demo.local_fit_paths() is None


def test_local_fit_paths_accepts_a_complete_zip_set(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(demo, "local_fit_path", lambda _dataset, name: tmp_path / name)
    expected = [tmp_path / name for name in demo.GSPLATS_FILES]
    for path in expected:
        with zipfile.ZipFile(path, "w"):
            pass

    assert demo.local_fit_paths() == expected


# ---------------------------------------------------------------------------
# First paint: the grafted parts are re-laddered for a small first rung
# ---------------------------------------------------------------------------


def _flat_2d(n: int, seed: int) -> GSplatData:
    rng = np.random.default_rng(seed)
    chol = np.zeros((n, 3), dtype=np.float32)
    chol[:, [0, 2]] = 1.0  # packed lower-triangular diagonal for ndim=2
    return GSplatData(
        centers=(rng.random((n, 2)) * 1000).astype(np.float32),
        amplitudes=rng.random(n).astype(np.float32),
        cholesky_factors=chol,
    )


def _record_shaped_archive(path: Path, seed: int) -> Path:
    """A kind=partition store whose parts carry EQUAL-COUNT 4-rung ladders.

    The shape of the published cmu1 archives (four top-level parts, each with
    ``lod_breakpoints_kind == "equal-count"`` and rung 0 = a quarter of the part).
    """
    from luxar.gsplats.io.save_gsplats import write_gsplats_tree
    from luxar.gsplats.lod import RecipeParams, build_recipe

    node = build_recipe(
        _flat_2d(480, seed), "tiles", RecipeParams(max_elements=130, n_lods=4)
    )
    write_gsplats_tree(path, node)
    return path


def test_part_ladder_spec_fits_the_smallest_part(tmp_path, monkeypatch) -> None:
    archive = _record_shaped_archive(tmp_path / "ch.gsplats.zarr", seed=1)
    monkeypatch.setattr(demo, "FIRST_RUNG_SPLATS", 20)

    spec = demo.part_ladder_spec(archive)

    from luxar._zarr_compat import open_group
    from luxar.gsplats.io.tree_summary import read_gsplat_tree_summary

    smallest = min(read_gsplat_tree_summary(open_group(str(archive))).leaf_counts)
    assert spec["recompute"] is True
    assert spec["method"] == "self_energy"
    cuts = spec["breakpoints"]
    assert cuts[0] == 20, "first rung is the first-paint budget"
    assert cuts == sorted(set(cuts)), "strictly increasing cumulative cuts"
    assert cuts[-1] == smallest, "sized to the smallest part so every part accepts it"


def test_scene_reladders_every_grafted_part(tmp_path, monkeypatch) -> None:
    # The whole point: a record-shaped archive (equal-count rungs, rung 0 = 25%)
    # reaches the scene with a small first rung on EVERY part.
    import zarr

    monkeypatch.setattr(demo, "FIRST_RUNG_SPLATS", 20)
    paths = [
        _record_shaped_archive(tmp_path / f"cmu1_ch{i}.gsplats.zarr", seed=i)
        for i in range(demo.N_CHANNELS)
    ]
    out = demo.create_luxar_scene(paths, tmp_path / "cmu1.luxar.zarr")

    root = zarr.open_group(str(out), mode="r")
    parts_seen = 0
    for colormap in ("red", "green", "blue"):
        layer = root[f"gsplats_{colormap}"]
        assert layer.attrs["kind"] == "partition"
        for name, part in layer.groups():
            if not name.startswith("part_"):
                continue
            parts_seen += 1
            n_sub = int(part.attrs["n_additive_sublods"])
            n_total = int(part.attrs["n_splats"])
            assert n_sub >= 3, f"{colormap}/{name}: expected a real ladder, got {n_sub}"
            rung0 = part["additive_0"]
            assert int(rung0.attrs["n_splats"]) == 20, (
                f"{colormap}/{name}: rung 0 is {rung0.attrs['n_splats']}, "
                "not the first-paint budget"
            )
            stats = rung0.attrs["lod_stats"]
            assert stats["lod_breakpoints_kind"] != "equal-count", (
                "the archive's equal-count ladder survived the graft"
            )
            assert stats["lod_method"] == "self_energy"
            rungs = [int(part[f"additive_{i}"].attrs["n_splats"]) for i in range(n_sub)]
            assert sum(rungs) == n_total
    assert parts_seen >= 3 * 2, "expected several parts per channel"
