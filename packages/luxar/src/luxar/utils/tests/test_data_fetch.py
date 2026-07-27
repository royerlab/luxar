"""Tests for the manifest-driven demo-dataset fetch helper (R17).

Covers the manifest as the single source of truth (schema, disk consistency,
staleness guard) and the ``ensure_dataset`` resolution logic (cache → in-repo LFS
→ Zenodo) with no network — the Zenodo leg stays dormant until URLs are set.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from luxar.utils import data_fetch
from luxar.utils.data_fetch import (
    DatasetNotFound,
    LocalComputeDataset,
    ensure_dataset,
    load_dataset_gsplats,
    load_manifest,
)
from luxar.utils.download import QUARANTINE_SUFFIX, find_quarantined_files

REPO_ROOT = Path(__file__).resolve().parents[6]
DATA_DIR = Path(data_fetch._DEMOS_DATA_DIR)
VALID_BUCKETS = {"zenodo", "local-compute", "regenerate"}


def _all_files(dataset: dict) -> list[dict]:
    """Every file entry of a dataset — flat ``files`` or across ``variants``."""
    if "variants" in dataset:
        return [f for v in dataset["variants"].values() for f in v.get("files", [])]
    return dataset.get("files", [])


# --------------------------------------------------------------------------- #
# Manifest as single source of truth
# --------------------------------------------------------------------------- #
def test_manifest_loads_and_has_expected_shape():
    m = load_manifest()
    assert m["schema_version"] == 1
    assert m["records"] and m["datasets"]
    for key in ("cc-by", "cc-by-sa", "h2afva"):
        assert key in m["records"], f"missing record group {key}"


def test_every_dataset_has_valid_bucket_license_and_files():
    m = load_manifest()
    for name, d in m["datasets"].items():
        assert d["bucket"] in VALID_BUCKETS, f"{name}: bad bucket {d['bucket']!r}"
        assert d.get("license"), f"{name}: missing license"
        # A dataset carries either a flat file list or a variants map.
        if "variants" in d:
            assert isinstance(d["variants"], dict) and d["variants"], (
                f"{name}: bad variants"
            )
            defaults = [v for v in d["variants"].values() if v.get("default")]
            assert len(defaults) == 1, f"{name}: needs exactly one default variant"
            for vn, v in d["variants"].items():
                assert isinstance(v.get("files"), list), (
                    f"{name}/{vn}: files must be a list"
                )
        else:
            assert isinstance(d.get("files"), list), f"{name}: files must be a list"


def test_zenodo_datasets_reference_an_existing_record():
    m = load_manifest()
    for name, d in m["datasets"].items():
        if d["bucket"] == "zenodo":
            assert d["record"] in m["records"], (
                f"{name}: unknown record {d['record']!r}"
            )


def test_present_zenodo_files_have_checksums():
    """Every listed file (i.e. already in-repo) carries a sha256 + byte size.

    Pending-upload datasets (computed elsewhere, not yet committed) legitimately
    have an empty file list and are skipped.
    """
    m = load_manifest()
    for name, d in m["datasets"].items():
        if d["bucket"] != "zenodo":
            continue
        for f in _all_files(d):
            assert f.get("sha256"), f"{name}/{f['name']}: missing sha256"
            assert f.get("bytes"), f"{name}/{f['name']}: missing byte size"


def test_pending_upload_datasets_are_declared_and_empty():
    m = load_manifest()
    # neuromast + h2afva are computed on obsidian; not yet in the repo.
    for name in ("gsplats_4d_neuromast_2ch", "h2afva"):
        assert name in m["datasets"], f"{name} should be listed as a pending upload"
        assert _all_files(m["datasets"][name]) == [], f"{name} should have no files yet"


def test_h2afva_has_light_default_and_full_variant():
    """The 16 GB timelapse ships as an opt-in; the demo default is the light cut."""
    variants = load_manifest()["datasets"]["h2afva"]["variants"]
    assert set(variants) == {"51tp", "253tp"}
    assert variants["51tp"]["default"] is True
    assert variants["253tp"]["default"] is False
    assert variants["253tp"]["approx_bytes"] > variants["51tp"]["approx_bytes"]


def test_gaia_is_local_compute_not_hosted():
    """Gaia is CC BY-NC → must NOT be a hosted (zenodo) dataset."""
    d = load_manifest()["datasets"]["milky_way_gaia_3m"]
    assert d["bucket"] == "local-compute"
    assert d.get("redistribute") is False


def test_manifest_matches_files_on_disk():
    """Every in-repo dataset file appears in the manifest with a matching name."""
    m = load_manifest()
    listed = {f["name"] for d in m["datasets"].values() for f in _all_files(d)}
    for sub in DATA_DIR.iterdir():
        if sub.is_dir() and sub.name != "tests":
            for f in sub.glob("*"):
                if f.is_file() and not f.name.startswith("."):
                    assert f.name in listed, (
                        f"{sub.name}/{f.name} missing from manifest"
                    )


def test_generator_check_reports_manifest_current():
    """The committed manifest matches what the generator would produce."""
    res = subprocess.run(
        [sys.executable, "scripts/gen_data_manifest.py", "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"manifest stale:\n{res.stdout}\n{res.stderr}"


# --------------------------------------------------------------------------- #
# ensure_dataset resolution (no network)
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


@pytest.fixture()
def fake_repo(tmp_path, monkeypatch):
    """A tiny manifest + in-repo LFS dir + cache root, wired via monkeypatch."""
    lfs_root = tmp_path / "demos_data"
    (lfs_root / "gsplats_toy").mkdir(parents=True)
    payload = lfs_root / "gsplats_toy" / "toy_ch0.gsplats.zarr.zip"
    payload.write_bytes(b"toy-splat-bytes")

    # A variant dataset: light (default) + full, one file each in its own subdir.
    (lfs_root / "toy_ts" / "light").mkdir(parents=True)
    (lfs_root / "toy_ts" / "full").mkdir(parents=True)
    light = lfs_root / "toy_ts" / "light" / "ts.gsplats.zarr.zip"
    light.write_bytes(b"light-bytes")
    full = lfs_root / "toy_ts" / "full" / "ts.gsplats.zarr.zip"
    full.write_bytes(b"full-timelapse-bytes")

    manifest = {
        "schema_version": 1,
        "cache_root": "~/.cache/luxar",
        "records": {
            "cc-by": {"license": "cc-by-4.0", "zenodo_record": None, "base_url": None}
        },
        "datasets": {
            "gsplats_toy": {
                "bucket": "zenodo",
                "record": "cc-by",
                "license": "cc0-1.0",
                "files": [
                    {
                        "name": "toy_ch0.gsplats.zarr.zip",
                        "sha256": _sha256(payload),
                        "bytes": payload.stat().st_size,
                    }
                ],
            },
            "toy_ts": {
                "bucket": "zenodo",
                "record": "cc-by",
                "license": "cc-by-4.0",
                "variants": {
                    "light": {
                        "default": True,
                        "files": [
                            {
                                "name": "ts.gsplats.zarr.zip",
                                "sha256": _sha256(light),
                                "bytes": light.stat().st_size,
                            }
                        ],
                    },
                    "full": {
                        "default": False,
                        "files": [
                            {
                                "name": "ts.gsplats.zarr.zip",
                                "sha256": _sha256(full),
                                "bytes": full.stat().st_size,
                            }
                        ],
                    },
                },
            },
            "toy_gaia": {
                "bucket": "local-compute",
                "redistribute": False,
                "license": "cc-by-nc-3.0-igo",
                "reason": "NC",
                "strategy": "query archive",
                "files": [],
            },
            "toy_dipc": {
                "bucket": "regenerate",
                "license": "cc-by-4.0",
                "strategy": "cpu build",
                "files": [],
            },
        },
    }
    monkeypatch.setattr(data_fetch, "_DEMOS_DATA_DIR", lfs_root)
    return manifest, tmp_path / "cache"


def test_variant_default_is_light(fake_repo):
    manifest, cache = fake_repo
    paths = ensure_dataset("toy_ts", manifest=manifest, cache_root=cache, verbose=False)
    assert paths[0].read_bytes() == b"light-bytes"
    # Variant files cache in their own subdir.
    assert paths[0].parent == cache / "toy_ts" / "light"


def test_variant_explicit_full(fake_repo):
    manifest, cache = fake_repo
    paths = ensure_dataset(
        "toy_ts", variant="full", manifest=manifest, cache_root=cache, verbose=False
    )
    assert paths[0].read_bytes() == b"full-timelapse-bytes"
    assert paths[0].parent == cache / "toy_ts" / "full"


def test_unknown_variant_raises(fake_repo):
    manifest, cache = fake_repo
    with pytest.raises(ValueError, match="Unknown variant"):
        ensure_dataset(
            "toy_ts", variant="nope", manifest=manifest, cache_root=cache, verbose=False
        )


def test_variant_on_nonvariant_dataset_raises(fake_repo):
    manifest, cache = fake_repo
    with pytest.raises(ValueError, match="no variants"):
        ensure_dataset(
            "gsplats_toy",
            variant="light",
            manifest=manifest,
            cache_root=cache,
            verbose=False,
        )


def test_ensure_dataset_copies_from_inrepo_lfs(fake_repo):
    manifest, cache = fake_repo
    paths = ensure_dataset(
        "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
    )
    assert len(paths) == 1
    assert paths[0].exists() and paths[0].read_bytes() == b"toy-splat-bytes"
    assert paths[0].parent == cache / "gsplats_toy"


def test_ensure_dataset_cache_hit_is_reused(fake_repo, monkeypatch):
    manifest, cache = fake_repo
    ensure_dataset("gsplats_toy", manifest=manifest, cache_root=cache, verbose=False)
    # Now break the in-repo source: a cache hit must not need it.
    monkeypatch.setattr(data_fetch, "_DEMOS_DATA_DIR", cache / "does-not-exist")
    paths = ensure_dataset(
        "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
    )
    assert paths[0].read_bytes() == b"toy-splat-bytes"


def test_local_compute_and_regenerate_raise(fake_repo):
    manifest, cache = fake_repo
    for name in ("toy_gaia", "toy_dipc"):
        with pytest.raises(LocalComputeDataset):
            ensure_dataset(name, manifest=manifest, cache_root=cache, verbose=False)


def test_recompute_forces_local_path_even_for_zenodo(fake_repo):
    manifest, cache = fake_repo
    with pytest.raises(LocalComputeDataset):
        ensure_dataset(
            "gsplats_toy",
            manifest=manifest,
            cache_root=cache,
            recompute=True,
            verbose=False,
        )


def test_unknown_dataset_raises(fake_repo):
    manifest, cache = fake_repo
    with pytest.raises(DatasetNotFound):
        ensure_dataset("nope", manifest=manifest, cache_root=cache, verbose=False)


def test_missing_and_unhosted_raises_clear_error(fake_repo):
    """No cache, no in-repo file, no Zenodo URL → actionable FileNotFoundError."""
    manifest, cache = fake_repo
    manifest["datasets"]["gsplats_toy"]["files"][0]["name"] = "absent.zip"
    with pytest.raises(FileNotFoundError, match="Zenodo record URL is not set"):
        ensure_dataset(
            "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
        )


# --------------------------------------------------------------------------- #
# Cache integrity: the checksum is the authority at every step
# --------------------------------------------------------------------------- #
def _corrupt_in_place_preserving_stat(path: Path) -> None:
    """Overwrite *path* with same-length garbage and restore its (atime, mtime).

    This is exactly the failure a (size, mtime) staleness test cannot see, and
    the shape of the bug this module was fixed for.
    """
    st = path.stat()
    path.write_bytes(b"X" * st.st_size)
    os.utime(path, (st.st_atime, st.st_mtime))
    assert path.stat().st_size == st.st_size


def test_same_size_same_mtime_corruption_is_quarantined_and_repaired(fake_repo):
    """REGRESSION: a checksum-failing cache entry must never be handed back.

    Before the fix, step 1 logged the sha256 mismatch and fell through to step 2,
    where ``_cache_is_stale`` compared only (size, mtime) — identical for an
    in-place corruption — so nothing was re-copied and the corrupt path was
    returned to the caller.
    """
    manifest, cache = fake_repo
    (good,) = ensure_dataset(
        "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
    )
    _corrupt_in_place_preserving_stat(good)

    (repaired,) = ensure_dataset(
        "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
    )

    assert repaired == good
    assert repaired.read_bytes() == b"toy-splat-bytes"
    quarantined = find_quarantined_files(good)
    assert quarantined == [good.with_name(good.name + QUARANTINE_SUFFIX)]
    assert quarantined[0].read_bytes() == b"X" * len(b"toy-splat-bytes")


def test_corrupt_cache_without_a_source_raises_and_still_quarantines(
    fake_repo, monkeypatch
):
    """Repair-or-raise: when nothing can fix it, the bad bytes still move aside."""
    manifest, cache = fake_repo
    (good,) = ensure_dataset(
        "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
    )
    _corrupt_in_place_preserving_stat(good)
    monkeypatch.setattr(data_fetch, "_DEMOS_DATA_DIR", cache / "does-not-exist")

    with pytest.raises(FileNotFoundError):
        ensure_dataset(
            "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
        )

    assert not good.exists(), "corrupt bytes left under the canonical name"
    assert find_quarantined_files(good)


def test_inrepo_source_failing_its_own_checksum_is_never_used(fake_repo):
    """A bad packaged copy is reported, never loaded, and never renamed.

    demos/data is git-tracked, so a .corrupt file there would dirty the working
    tree and break test_manifest_matches_files_on_disk. With no Zenodo URL yet
    there is no good copy, so the only correct outcome is a clear error.
    """
    manifest, cache = fake_repo
    payload = data_fetch._DEMOS_DATA_DIR / "gsplats_toy" / "toy_ch0.gsplats.zarr.zip"
    payload.write_bytes(b"toy-splat-BYTES")  # same length, different content

    with pytest.raises(FileNotFoundError, match="fails its manifest sha256"):
        ensure_dataset(
            "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
        )

    assert payload.read_bytes() == b"toy-splat-BYTES", "in-repo source was modified"
    assert find_quarantined_files(payload) == []
    assert not (cache / "gsplats_toy" / "toy_ch0.gsplats.zarr.zip").exists()


def test_zenodo_leg_is_never_handed_a_preexisting_file(fake_repo, monkeypatch):
    """robust_download RESUMES onto whatever sits at the destination (#731).

    So step 3's contract is that `dest` must not exist when it is called.
    """
    manifest, cache = fake_repo
    manifest["records"]["cc-by"]["base_url"] = "https://example.invalid/files"
    (good,) = ensure_dataset(
        "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
    )
    _corrupt_in_place_preserving_stat(good)
    monkeypatch.setattr(data_fetch, "_DEMOS_DATA_DIR", cache / "does-not-exist")

    def _fake_download(url, output_path, expected_sha256=None, **kw):
        assert not Path(output_path).exists(), (
            "download_with_checksum was handed an existing file — "
            "robust_download would append to (resume onto) its bytes"
        )
        Path(output_path).write_bytes(b"toy-splat-bytes")
        return Path(output_path)

    monkeypatch.setattr("luxar.utils.download.download_with_checksum", _fake_download)

    (path,) = ensure_dataset(
        "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
    )
    assert path.read_bytes() == b"toy-splat-bytes"


def test_lfs_pointer_in_cache_is_quarantined_and_replaced(fake_repo):
    manifest, cache = fake_repo
    dest = cache / "gsplats_toy" / "toy_ch0.gsplats.zarr.zip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(
        b"version https://git-lfs.github.com/spec/v1\noid sha256:"
        + b"0" * 64
        + b"\nsize 15\n"
    )

    (path,) = ensure_dataset(
        "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
    )

    assert path.read_bytes() == b"toy-splat-bytes"
    assert find_quarantined_files(dest)


def test_entry_without_a_checksum_is_reused_unverified(fake_repo):
    """sha256=None means 'unverifiable', not 'verified'.

    verify_file_checksum returns True when handed no expected hash, so this path
    is gated explicitly rather than left to that vacuous behaviour.
    """
    manifest, cache = fake_repo
    manifest["datasets"]["gsplats_toy"]["files"][0]["sha256"] = None

    (path,) = ensure_dataset(
        "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
    )
    path.write_bytes(b"anything-at-all")
    (again,) = ensure_dataset(
        "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
    )

    assert again.read_bytes() == b"anything-at-all"
    assert find_quarantined_files(again) == []


def test_cache_copy_leaves_no_temporary_files(fake_repo):
    """The LFS->cache copy is atomic; its temp sibling must not survive."""
    manifest, cache = fake_repo
    ensure_dataset("gsplats_toy", manifest=manifest, cache_root=cache, verbose=False)

    leftovers = [
        p.name for p in (cache / "gsplats_toy").iterdir() if p.name.startswith(".tmp_")
    ]
    assert leftovers == []


# --------------------------------------------------------------------------- #
# load_dataset_gsplats (the migration wrapper)
# --------------------------------------------------------------------------- #
@pytest.fixture()
def fake_gsplats_repo(tmp_path, monkeypatch):
    """Like ``fake_repo``, but the payload is a real (tiny) gsplats archive."""
    import numpy as np

    from luxar.gsplats.gsplat_data import GSplatData

    lfs_root = tmp_path / "demos_data"
    (lfs_root / "gsplats_toy").mkdir(parents=True)
    payload = lfs_root / "gsplats_toy" / "toy_ch0.gsplats.zarr.zip"
    n = 8
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, [0, 2, 5]] = 1.0
    GSplatData(
        centers=np.random.rand(n, 3).astype(np.float32),
        amplitudes=np.ones(n, dtype=np.float32),
        cholesky_factors=chol,
    ).save(payload, ordering="none", compress="zip")

    # A non-gsplat sidecar, exactly as gsplats_ct_totalsegmentator really ships.
    sidecar = lfs_root / "gsplats_toy" / "toy_labels.npz"
    np.savez(sidecar, labels=np.zeros(n, dtype=np.uint8))

    manifest = {
        "schema_version": 1,
        "records": {"cc-by": {"zenodo_record": None, "base_url": None}},
        "datasets": {
            "gsplats_toy": {
                "bucket": "zenodo",
                "record": "cc-by",
                "license": "cc0-1.0",
                "files": [
                    {"name": p.name, "sha256": _sha256(p), "bytes": p.stat().st_size}
                    for p in (payload, sidecar)
                ],
            },
            "toy_local": {
                "bucket": "local-compute",
                "license": "cc-by-nc-3.0-igo",
                "reason": "not redistributable",
                "strategy": "query the archive and fit",
                "files": [],
            },
        },
    }
    monkeypatch.setattr(data_fetch, "_DEMOS_DATA_DIR", lfs_root)
    return manifest, tmp_path / "cache"


def test_wrapper_loads_only_the_gsplat_files(fake_gsplats_repo):
    manifest, cache = fake_gsplats_repo

    out = load_dataset_gsplats(
        "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
    )

    assert out is not None and len(out) == 1, "the .npz sidecar must be skipped"
    assert len(out[0].amplitudes) == 8


def test_wrapper_recompute_returns_none(fake_gsplats_repo):
    manifest, cache = fake_gsplats_repo
    assert (
        load_dataset_gsplats(
            "gsplats_toy",
            recompute=True,
            manifest=manifest,
            cache_root=cache,
            verbose=False,
        )
        is None
    )


def test_wrapper_maps_local_compute_to_the_none_sentinel(fake_gsplats_repo):
    """The sentinel reconciliation: ensure_dataset RAISES, the wrapper returns None."""
    manifest, cache = fake_gsplats_repo
    with pytest.raises(LocalComputeDataset):
        ensure_dataset("toy_local", manifest=manifest, cache_root=cache, verbose=False)

    assert (
        load_dataset_gsplats(
            "toy_local", manifest=manifest, cache_root=cache, verbose=False
        )
        is None
    )


def test_wrapper_rejects_a_file_that_is_not_a_manifest_entry(fake_gsplats_repo):
    manifest, cache = fake_gsplats_repo
    with pytest.raises(FileNotFoundError, match="runtime-computed file list"):
        load_dataset_gsplats(
            "gsplats_toy",
            ["zebrafish_frame0007.gsplats.zarr.zip"],
            manifest=manifest,
            cache_root=cache,
            verbose=False,
        )


def test_wrapper_propagates_unknown_dataset(fake_gsplats_repo):
    manifest, cache = fake_gsplats_repo
    with pytest.raises(DatasetNotFound):
        load_dataset_gsplats("nope", manifest=manifest, cache_root=cache, verbose=False)


def test_wrapper_repairs_a_corrupt_cache_before_loading(fake_gsplats_repo):
    """End to end: the wrapper inherits ensure_dataset's checksum authority."""
    manifest, cache = fake_gsplats_repo
    first = load_dataset_gsplats(
        "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
    )
    cached = cache / "gsplats_toy" / "toy_ch0.gsplats.zarr.zip"
    _corrupt_in_place_preserving_stat(cached)

    second = load_dataset_gsplats(
        "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
    )

    assert first is not None and second is not None
    assert len(second[0].amplitudes) == len(first[0].amplitudes)
    assert find_quarantined_files(cached)
