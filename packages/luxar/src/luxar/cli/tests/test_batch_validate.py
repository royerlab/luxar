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
import os
from pathlib import Path

import numpy as np
import zarr

from luxar.cli.gsplat_ops.batch.validation import validate_tile as _validate_tile
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

    # Inspect the implementation that actually runs the --fix loop; the name
    # reachable via gsplat_commands is `batch/commands.py`'s thin Typer
    # registration surface, which only delegates here.
    from luxar.cli.gsplat_ops.batch.status_validate_cancel import (
        run_batch_validate_cmd,
    )

    src = inspect.getsource(run_batch_validate_cmd)
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


def test_validate_fix_reclaims_staging_leftovers_keeps_empty_marker(
    tmp_path, monkeypatch
):
    """`--fix` must reclaim per-attempt staging leftovers, not the old `.tmp`.

    Staging dirs are now `{tile}.tmp.<token>` (local host+pid) /
    `{tile}.tmp.<jobid>.<taskid>.<restart>` (Slurm), plus a possible
    `{tile}.tmp.<token>.empty` marker file and a possible
    `{tile}.tmp.<token>.old` set-aside prior tile (a --no-resume refit that died
    mid-promotion). The `{tile}.tmp*` glob must match all of them while NEVER
    touching the legitimate `{tile}.empty` marker, and the liveness probe must
    receive the bare token (`.empty`/`.old` stripped).
    """
    import luxar.cli.gsplat_ops.batch.status_validate_cancel as bsvc
    from luxar.cli.gsplat_ops.batch.status_validate_cancel import (
        run_batch_validate_cmd,
    )
    from luxar.gsplats.batch.manifest import BatchJob, BatchManifest, save_manifest

    # All leftovers belong to attempts that are verifiably gone; record the
    # tokens the probe receives so the suffix stripping is asserted too.
    seen_tokens: list[str] = []

    def _gone(token: str) -> bool:
        seen_tokens.append(token)
        return False

    monkeypatch.setattr(bsvc, "_staging_attempt_live", _gone)

    out_dir = tmp_path / "batch"
    tiles = out_dir / "tiles"
    tiles.mkdir(parents=True)
    tile = "t00_c00_tile000.gsplats.zarr"

    manifest = BatchManifest(
        input_path="/data/x.zarr",
        output_dir=str(out_dir),
        jobs=[
            BatchJob(
                task_id=0,
                timepoint=0,
                channel=0,
                tile_index=0,
                output_filename=tile,
                estimated_wall_seconds=1.0,
            )
        ],
    )
    save_manifest(manifest, out_dir)

    # Legitimate empty marker (task fit 0 splats) — MUST survive `--fix`.
    (tiles / f"{tile}.empty").touch()
    # Stale local staging dir + a Slurm-shaped stale `.empty` marker file + a
    # set-aside prior tile from a --no-resume refit that died mid-promotion.
    staging = tiles / f"{tile}.tmp.host-1234"
    staging.mkdir()
    (staging / "data").write_text("partial")
    stale_marker = tiles / f"{tile}.tmp.42.0.1.empty"
    stale_marker.touch()
    stale_aside = tiles / f"{tile}.tmp.host-1234.old"
    stale_aside.mkdir()
    (stale_aside / "data").write_text("prior")

    run_batch_validate_cmd(output_dir=out_dir, fix=True)

    assert not staging.exists()  # staging dir reclaimed
    assert not stale_marker.exists()  # stale staging marker reclaimed
    assert not stale_aside.exists()  # set-aside prior tile reclaimed
    assert (tiles / f"{tile}.empty").exists()  # legitimate marker untouched
    # The probe received bare tokens: `.empty` and `.old` suffixes stripped.
    assert set(seen_tokens) == {"host-1234", "42.0.1"}


def test_validate_report_counts_stray_staging_empty_markers(
    tmp_path, capsys, monkeypatch
):
    """Report-only `validate` (no --fix) must COUNT stray staging leftovers.

    A stray `{tile}.tmp.<token>.empty` marker and a stale staging dir both count
    toward STALE_TMP even without --fix, so the report is honest; neither is
    deleted in report mode.
    """
    import luxar.cli.gsplat_ops.batch.status_validate_cancel as bsvc
    from luxar.cli.gsplat_ops.batch.status_validate_cancel import (
        run_batch_validate_cmd,
    )
    from luxar.gsplats.batch.manifest import BatchJob, BatchManifest, save_manifest

    monkeypatch.setattr(bsvc, "_staging_attempt_live", lambda token: False)

    out_dir = tmp_path / "batch"
    tiles = out_dir / "tiles"
    tiles.mkdir(parents=True)
    tile = "t00_c00_tile000.gsplats.zarr"

    manifest = BatchManifest(
        input_path="/data/x.zarr",
        output_dir=str(out_dir),
        jobs=[
            BatchJob(
                task_id=0,
                timepoint=0,
                channel=0,
                tile_index=0,
                output_filename=tile,
                estimated_wall_seconds=1.0,
            )
        ],
    )
    save_manifest(manifest, out_dir)

    (tiles / f"{tile}.empty").touch()  # legitimate empty marker
    staging = tiles / f"{tile}.tmp.host-1234"
    staging.mkdir()
    stray_marker = tiles / f"{tile}.tmp.42.0.1.empty"
    stray_marker.touch()

    run_batch_validate_cmd(output_dir=out_dir, fix=False)

    out = capsys.readouterr().out
    # 1 staging dir + 1 stray .empty marker = 2 counted.
    assert "STALE_TMP:  2" in out
    # Report mode deletes nothing.
    assert staging.exists()
    assert stray_marker.exists()
    assert (tiles / f"{tile}.empty").exists()


def test_validate_fix_keeps_staging_of_running_attempt(tmp_path, capsys, monkeypatch):
    """`--fix` must NEVER delete staging owned by a still-running attempt.

    Running fits write into exactly these paths; a partial delete under a live
    zarr writer can end with a corrupt store being promoted by that attempt's
    atomic claim. Live (or unverifiable) staging is counted as ACTIVE and kept.
    """
    import luxar.cli.gsplat_ops.batch.status_validate_cancel as bsvc
    from luxar.cli.gsplat_ops.batch.status_validate_cancel import (
        run_batch_validate_cmd,
    )
    from luxar.gsplats.batch.manifest import BatchJob, BatchManifest, save_manifest

    monkeypatch.setattr(bsvc, "_staging_attempt_live", lambda token: True)

    out_dir = tmp_path / "batch"
    tiles = out_dir / "tiles"
    tiles.mkdir(parents=True)
    tile = "t00_c00_tile000.gsplats.zarr"

    manifest = BatchManifest(
        input_path="/data/x.zarr",
        output_dir=str(out_dir),
        jobs=[
            BatchJob(
                task_id=0,
                timepoint=0,
                channel=0,
                tile_index=0,
                output_filename=tile,
                estimated_wall_seconds=1.0,
            )
        ],
    )
    save_manifest(manifest, out_dir)

    staging = tiles / f"{tile}.tmp.host-1234"
    staging.mkdir()
    (staging / "data").write_text("in-progress")

    run_batch_validate_cmd(output_dir=out_dir, fix=True)

    out = capsys.readouterr().out
    assert staging.exists()  # the live attempt's store is untouched
    assert (staging / "data").read_text() == "in-progress"
    assert "STALE_TMP:  0" in out
    assert "ACTIVE_TMP: 1" in out
    assert "Kept: " in out


def test_staging_attempt_live_token_parsing(monkeypatch):
    """Token → liveness routing: local host+pid, Slurm jobid, legacy, unknown."""
    import socket as socket_mod

    import luxar.cli.gsplat_ops.batch.status_validate_cancel as bsvc

    host = socket_mod.gethostname()

    # Our own (running) pid on this host is live.
    assert bsvc._staging_attempt_live(f"{host}-{os.getpid()}") is True

    # A dead pid on this host is reclaimable.
    def _dead(pid, sig):  # type: ignore[no-untyped-def]
        raise ProcessLookupError()

    monkeypatch.setattr(bsvc.os, "kill", _dead)
    assert bsvc._staging_attempt_live(f"{host}-4242") is False
    monkeypatch.undo()

    # Another machine's pid can't be probed — keep, to be safe.
    assert bsvc._staging_attempt_live("some-other-host-4242") is None

    # Legacy shared `.tmp` (empty token) has no owner — always reclaimable.
    assert bsvc._staging_attempt_live("") is False


def test_staging_attempt_live_slurm_token(monkeypatch):
    """Slurm tokens are probed via squeue; no squeue means 'cannot verify'."""
    import luxar.cli.gsplat_ops.batch.status_validate_cancel as bsvc

    class _Res:
        def __init__(self, rc, out, err=""):
            self.returncode = rc
            self.stdout = out
            self.stderr = err

    calls: list[list[str]] = []

    def _fake_run(argv, **kw):  # type: ignore[no-untyped-def]
        calls.append(argv)
        return _Res(0, "RUNNING\n")

    monkeypatch.setattr(bsvc.subprocess, "run", _fake_run)
    assert bsvc._staging_attempt_live("42.7.1") is True
    assert calls and calls[0][:3] == ["squeue", "-h", "-j"] and calls[0][3] == "42"

    # Job left the queue (squeue rejects the unknown id) → reclaimable.
    monkeypatch.setattr(
        bsvc.subprocess,
        "run",
        lambda *a, **kw: _Res(1, "", "slurm_load_jobs error: Invalid job id specified"),
    )
    assert bsvc._staging_attempt_live("42.7.1") is False

    # squeue failed operationally (controller down, auth error) — that is NOT
    # proof the job is gone; deleting live staging on it could corrupt a store
    # a running attempt is about to promote. Keep, to be safe.
    monkeypatch.setattr(
        bsvc.subprocess,
        "run",
        lambda *a, **kw: _Res(
            1, "", "slurm_load_jobs error: Unable to contact slurm controller"
        ),
    )
    assert bsvc._staging_attempt_live("42.7.1") is None

    # No squeue on this machine → cannot verify → keep.
    def _no_squeue(*a, **kw):  # type: ignore[no-untyped-def]
        raise FileNotFoundError("squeue")

    monkeypatch.setattr(bsvc.subprocess, "run", _no_squeue)
    assert bsvc._staging_attempt_live("42.7.1") is None


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
