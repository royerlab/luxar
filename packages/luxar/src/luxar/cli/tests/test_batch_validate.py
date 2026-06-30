"""Regression tests for `gsplat batch validate` tile integrity (Blocker 1).

``_validate_tile`` is a data-loss-prevention gate: ``batch validate --fix``
deletes any tile it classifies as corrupt. After the v3.0 cutover this must:

* classify a well-formed v3.0 tile (leaf / additive ladder / kind=lod /
  kind=partition / nested) as ``"ok"``;
* classify a legacy (v2.0) tile as ``unsupported_format_version`` — RECOVERABLE,
  which the ``--fix`` loop must bucket as *unmigrated* and NEVER delete;
* still catch genuinely corrupt tiles (missing arrays / .zmetadata / .zattrs).

The ``--fix`` loop keys the non-deletable bucket on the
``"unsupported_format_version"`` prefix, so that exact contract is asserted here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import zarr

from luxar.cli.gsplat_commands import _validate_tile
from luxar.gsplats.gsplat_data import AdditiveSubLOD
from luxar.gsplats.io.save_gsplats import save_gsplats, write_gsplats_tree
from luxar.gsplats.tree import GSplatLeaf, GSplatLodGroup, GSplatPartition


def _splats(n: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, [0, 2, 5]] = rng.uniform(0.5, 2.0, size=(n, 3))
    return {
        "centers": (rng.uniform(0, 50, size=(n, 3))).astype(np.float32),
        "amplitudes": rng.uniform(0.1, 1.0, size=(n,)).astype(np.float32),
        "cholesky_factors": chol,
    }


def _leaf(n: int, seed: int = 0) -> GSplatLeaf:
    s = _splats(n, seed)
    return GSplatLeaf(additive_sublods=[AdditiveSubLOD(**s)])


def _make_v2_0_tile(path: Path, n: int = 5) -> None:
    """A legacy v2.0 substitutive_0/additive_0 tile (must be 'unmigrated')."""
    store = zarr.DirectoryStore(str(path))
    root = zarr.group(store=store, overwrite=True)
    root.attrs.update(
        {
            "format_version": "2.0",
            "format_type": "gsplats_zarr",
            "n_substitutive": 1,
            "default_substitutive": 0,
        }
    )
    splats = root.create_group("splats")
    sub = splats.create_group("substitutive_0")
    add = sub.create_group("additive_0")
    s = _splats(n)
    add.create_dataset("centers", data=s["centers"])
    add.create_dataset("amplitudes", data=s["amplitudes"])
    add.create_dataset("cholesky_factors", data=s["cholesky_factors"])
    zarr.consolidate_metadata(store)


# ── v3.0 tiles → "ok" ─────────────────────────────────────────────────────


def test_v3_leaf_tile_ok(tmp_path):
    p = tmp_path / "leaf.gsplats.zarr"
    save_gsplats(path=p, **_splats(40), ordering="none")
    assert _validate_tile(p) == "ok"


def test_v3_additive_ladder_tile_ok(tmp_path):
    p = tmp_path / "ladder.gsplats.zarr"
    leaf = GSplatLeaf(
        additive_sublods=[
            AdditiveSubLOD(**_splats(30, 0)),
            AdditiveSubLOD(**_splats(10, 1)),
        ]
    )
    write_gsplats_tree(p, leaf, ordering="none")
    assert _validate_tile(p) == "ok"


def test_v3_lod_partition_nested_tiles_ok(tmp_path):
    lod = tmp_path / "lod.gsplats.zarr"
    write_gsplats_tree(
        lod, GSplatLodGroup(children=[_leaf(50, 0), _leaf(8, 1)]), ordering="none"
    )
    assert _validate_tile(lod) == "ok"

    part = tmp_path / "part.gsplats.zarr"
    write_gsplats_tree(
        part, GSplatPartition(children=[_leaf(20, 0), _leaf(20, 1)]), ordering="none"
    )
    assert _validate_tile(part) == "ok"

    nested = tmp_path / "nested.gsplats.zarr"
    write_gsplats_tree(
        nested,
        GSplatPartition(
            children=[
                GSplatLodGroup(children=[_leaf(40, 0), _leaf(6, 1)]),
                _leaf(15, 2),
            ]
        ),
        ordering="none",
    )
    assert _validate_tile(nested) == "ok"


# ── legacy v2.0 tile → unmigrated (recoverable, NEVER deleted) ────────────


def test_v2_tile_classified_unmigrated_not_corrupt(tmp_path):
    p = tmp_path / "legacy.gsplats.zarr"
    _make_v2_0_tile(p, n=5)
    reason = _validate_tile(p)
    # The --fix loop buckets this as non-deletable iff it starts with this prefix.
    assert reason.startswith("unsupported_format_version"), reason


def test_unmigrated_bucket_prefix_contract(tmp_path):
    """Guard the exact prefix the batch-validate --fix loop keys on so a future
    edit can't silently move a recoverable tile into the deletable bucket."""
    import inspect

    from luxar.cli import gsplat_commands

    src = inspect.getsource(gsplat_commands.batch_validate_cmd)
    assert 'startswith("unsupported_format_version")' in src


# ── genuinely corrupt tiles → caught ──────────────────────────────────────


def test_missing_zmetadata_is_corrupt(tmp_path):
    p = tmp_path / "t.gsplats.zarr"
    save_gsplats(path=p, **_splats(10), ordering="none")
    (p / ".zmetadata").unlink()
    assert _validate_tile(p) == "no_zmetadata (save incomplete)"


def test_missing_centers_is_corrupt(tmp_path):
    import shutil

    p = tmp_path / "t.gsplats.zarr"
    save_gsplats(path=p, **_splats(10), ordering="none")
    shutil.rmtree(p / "centers")
    assert "missing_centers" in _validate_tile(p)


def test_child_missing_zattrs_is_corrupt(tmp_path):
    p = tmp_path / "lod.gsplats.zarr"
    write_gsplats_tree(
        p, GSplatLodGroup(children=[_leaf(50, 0), _leaf(8, 1)]), ordering="none"
    )
    (p / "child_0" / ".zattrs").unlink()
    assert _validate_tile(p).startswith("no_zattrs")


def test_bad_format_type_is_corrupt(tmp_path):
    p = tmp_path / "t.gsplats.zarr"
    p.mkdir()
    (p / ".zmetadata").write_text("{}")
    (p / ".zattrs").write_text(json.dumps({"format_type": "not_gsplats"}))
    assert _validate_tile(p).startswith("bad_format_type")


def test_missing_offdiag_for_dgt1_is_corrupt(tmp_path):
    """v3.1: a d>1 leaf with the diagonal but no off-diagonal array is a
    partial/corrupt write — the validator must flag it (recoverable via re-fit)
    rather than passing it as 'ok' (which would let --fix delete real data only
    after a confusing later decode failure)."""
    import shutil

    p = tmp_path / "t.gsplats.zarr"
    save_gsplats(path=p, **_splats(20), ordering="none")  # 3D → offdiag present
    assert (p / "cholesky_factors_offdiag").is_dir()  # precondition
    shutil.rmtree(p / "cholesky_factors_offdiag")
    assert "missing_cholesky_factors_offdiag" in _validate_tile(p)


def test_1d_tile_no_offdiag_is_ok(tmp_path):
    """A 1D leaf legitimately has no off-diagonal array (k - d == 0); the
    d>1 offdiag-required check must NOT false-flag it as corrupt."""
    rng = np.random.default_rng(0)
    p = tmp_path / "t1d.gsplats.zarr"
    save_gsplats(
        path=p,
        centers=rng.uniform(0, 50, size=(20, 1)).astype(np.float32),
        amplitudes=rng.uniform(0.1, 1.0, size=(20,)).astype(np.float32),
        cholesky_factors=rng.uniform(0.5, 2.0, size=(20, 1)).astype(np.float32),
        ordering="none",
    )
    assert not (p / "cholesky_factors_offdiag").exists()  # precondition: 1D
    assert _validate_tile(p) == "ok"
