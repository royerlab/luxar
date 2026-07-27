"""Tests for the manifest-driven demo-dataset fetch helper (R17).

Covers the manifest as the single source of truth (schema, disk consistency,
staleness guard) and the ``ensure_dataset`` resolution logic (cache → in-repo LFS
→ Zenodo) with no network — the Zenodo leg stays dormant until URLs are set.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from luxar.utils import data_fetch
from luxar.utils.data_fetch import (
    DatasetNotFound,
    LocalComputeDataset,
    ensure_dataset,
    load_manifest,
)

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
            assert (
                isinstance(d["variants"], dict) and d["variants"]
            ), f"{name}: bad variants"
            defaults = [v for v in d["variants"].values() if v.get("default")]
            assert len(defaults) == 1, f"{name}: needs exactly one default variant"
            for vn, v in d["variants"].items():
                assert isinstance(
                    v.get("files"), list
                ), f"{name}/{vn}: files must be a list"
        else:
            assert isinstance(d.get("files"), list), f"{name}: files must be a list"


def test_zenodo_datasets_reference_an_existing_record():
    m = load_manifest()
    for name, d in m["datasets"].items():
        if d["bucket"] == "zenodo":
            assert (
                d["record"] in m["records"]
            ), f"{name}: unknown record {d['record']!r}"


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
                    assert (
                        f.name in listed
                    ), f"{sub.name}/{f.name} missing from manifest"


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
