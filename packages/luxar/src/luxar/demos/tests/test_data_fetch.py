"""Tests for the manifest-driven demo-dataset fetch helper (R17).

Covers the manifest as the single source of truth (schema, disk consistency,
staleness guard) and the ``ensure_dataset`` resolution logic (cache → in-repo LFS
→ Zenodo) with no network — the Zenodo leg stays dormant until URLs are set.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import runpy
import shutil
import sys
from pathlib import Path

import pytest

from luxar.demos._support.datasets import data_fetch
from luxar.demos._support.datasets.data_fetch import (
    LOCAL_FIT_DIRNAME,
    MANIFEST_PATH,
    DatasetNotFound,
    DatasetUnavailable,
    LocalComputeDataset,
    clear_manifest_cache,
    dataset_spec,
    declared_file_names,
    ensure_dataset,
    load_dataset_gsplats,
    load_local_fit_gsplats,
    load_local_fit_gsplats_at,
    load_manifest,
    local_fit_path,
)
from luxar.demos._support.downloads.download import (
    QUARANTINE_SUFFIX,
    find_quarantined_files,
)

REPO_ROOT = Path(__file__).resolve().parents[6]
DATA_DIR = Path(data_fetch._DEMOS_DATA_DIR)
VALID_BUCKETS = {"zenodo", "local-compute", "regenerate"}
GEN_SCRIPT = REPO_ROOT / "scripts" / "gen_data_manifest.py"
_NO_SCRIPT = "generator script not present (packaged install without repo scripts/)"

# This contract deliberately tracks the pinned digests and their history depths.
# Update it as part of a reviewed re-pin.
#
# The digests below are unchanged by the Git-LFS teardown (#2354), but the field
# holding them moved. With no in-repo payload left, the record digest is
# `sha256` and `hosted_sha256` is gone. What the test guards is unchanged:
# both members of a positional pair must move together, at equal history depth.
_EXPECTED_POSITIONAL_CONTRACTS = {
    # Re-pinned when the hosted fit gained an exact NATIVE categorical label
    # channel (#2387), retiring the sidecar's ordering hazard for the hosted copy
    # (#2334). The pair moved atomically: the sidecar was re-exported FROM the new
    # archive's own ids, so it is aligned by construction rather than by a
    # measurement that happened to pass. Both pins became the single record
    # contract when the in-repo Git-LFS payloads retired (#2354).
    (
        "gsplats_ct_totalsegmentator",
        "ct_atlas.gsplats.zarr.zip",
    ): ("60610346018089761f96067dc19d12cb152be6077f3afa23a3f35f3bb92fc648", 1),
    (
        "gsplats_ct_totalsegmentator",
        "ct_atlas_labels.npz",
    ): ("53731d93bfbb3746c64e21abc3a63d4151e24a42097a855d084650ea4e263b19", 1),
    # Re-pinned when the hosted fit gained NATIVE per-splat colors, retiring the
    # sidecar's ordering hazard for the hosted copy (#2334). The pair moved
    # atomically, as the generator and this test both require: the sidecar was
    # re-exported FROM the new archive's own colors, so it is aligned by
    # construction rather than by a measurement that happened to pass. Both
    # pins became the single record contract when the Git-LFS payloads retired.
    (
        "gsplats_visible_human_head",
        "vh_head.gsplats.zarr.zip",
    ): ("547fbbf3d3a6ffabbac38e751a6dccec587d44a9a2f83423a1d0ba3b66410e21", 1),
    (
        "gsplats_visible_human_head",
        "vh_head_colors.npz",
    ): ("f24d7fc074f8709df15201e414189a03ed30a5e58499cbd50d500530fa8880e1", 1),
}


def _load_generator():
    """Import scripts/gen_data_manifest.py as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("gen_data_manifest", GEN_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
    for key in ("cc-by", "cc-by-sa", "h2afva", "droso-timelapse"):
        assert key in m["records"], f"missing record group {key}"


def test_positional_sidecar_pairs_move_atomically():
    """Each positional group advances together and matches its record contract."""
    expected_pairs = {
        ("gsplats_ct_totalsegmentator", "ct_atlas"),
        ("gsplats_visible_human_head", "vh_head"),
    }
    manifest = load_manifest()
    groups: dict[tuple[str, str], list[dict]] = {}
    for dataset_name, dataset in manifest["datasets"].items():
        for entry in _all_files(dataset):
            pair = entry.get("positional_pair")
            if pair:
                groups.setdefault((dataset_name, pair), []).append(entry)

    assert set(groups) == expected_pairs
    pair_members = {
        (dataset_name, entry["name"])
        for (dataset_name, _pair), entries in groups.items()
        for entry in entries
    }
    assert pair_members == set(_EXPECTED_POSITIONAL_CONTRACTS)
    for (dataset_name, pair), entries in groups.items():
        assert len(entries) >= 2, f"{dataset_name}/{pair} has no positional partner"
        history_lengths = {
            len(entry.get("superseded_sha256") or ()) for entry in entries
        }
        assert len(history_lengths) == 1, (
            f"{dataset_name}/{pair} members did not move atomically"
        )
        for entry in entries:
            history = entry.get("superseded_sha256") or ()
            expected_contract = _EXPECTED_POSITIONAL_CONTRACTS[
                (dataset_name, entry["name"])
            ]
            assert (entry.get("sha256"), len(history)) == expected_contract, (
                f"{dataset_name}/{entry['name']} pin or history depth changed; "
                "update the committed expectation only after reviewing the re-pin"
            )
            if history:
                assert entry.get("sha256") != history[-1], (
                    f"{dataset_name}/{entry['name']} still names its "
                    "superseded pin as sha256"
                )


def test_load_manifest_hands_out_an_independent_copy():
    """A caller must not be able to poison the cached parse.

    The parse is memoised process-wide, so returning it directly meant one
    caller mutating the manifest — or a nested ``dataset_spec`` out of it —
    changed what every later reader saw.
    """
    first = load_manifest()
    first["schema_version"] = 999
    first["datasets"]["not_a_real_dataset"] = {"bucket": "zenodo"}
    dataset_spec("milky_way_gaia_3m", first)["bucket"] = "zenodo"

    second = load_manifest()
    assert second is not first
    assert second["schema_version"] == 1
    assert "not_a_real_dataset" not in second["datasets"]
    assert second["datasets"]["milky_way_gaia_3m"]["bucket"] == "local-compute"


def test_manifest_parse_is_cached_and_can_be_invalidated(tmp_path):
    """The cache still exists (and is still droppable) after the copy-on-read split.

    Moving the ``lru_cache`` onto a private helper must not take the ability to
    invalidate it with it: a caller that rewrites a manifest on disk needs the
    next read to see the new bytes.
    """
    custom = tmp_path / "data_manifest.json"
    base = {"schema_version": 1, "records": {}, "datasets": {}}
    custom.write_text(json.dumps(base))
    assert load_manifest(str(custom))["datasets"] == {}

    custom.write_text(json.dumps({**base, "datasets": {"toy": {"bucket": "zenodo"}}}))
    assert load_manifest(str(custom))["datasets"] == {}, "parse is no longer cached"

    clear_manifest_cache()
    try:
        assert "toy" in load_manifest(str(custom))["datasets"]
    finally:
        # maxsize=1: don't leave the tmp manifest occupying the slot.
        clear_manifest_cache()


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
            sha = f.get("sha256")
            assert sha, f"{name}/{f['name']}: missing sha256"
            # Shape-check it too: a pin transcribed by hand (truncated, or an md5
            # pasted where the sha256 belongs) is unverifiable data that would
            # only surface as a checksum failure on someone else's download.
            assert len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), (
                f"{name}/{f['name']}: {sha!r} is not a lowercase hex sha256"
            )
            assert f.get("bytes"), f"{name}/{f['name']}: missing byte size"
            # `hosted_sha256` is OPTIONAL — present only where the hosted
            # artifact differs from the in-repo copy — but when present it is a
            # download contract and gets the same shape check. A malformed one
            # would reject every download of a file that is actually fine.
            hosted = f.get("hosted_sha256")
            if hosted is not None:
                assert len(hosted) == 64 and all(
                    c in "0123456789abcdef" for c in hosted
                ), f"{name}/{f['name']}: {hosted!r} is not a lowercase hex sha256"
            # `hosted_bytes` travels with it so the digest and size preserve a
            # complete description of the hosted artifact as a pair.
            hosted_bytes = f.get("hosted_bytes")
            if hosted_bytes is not None:
                assert isinstance(hosted_bytes, int) and hosted_bytes > 0, (
                    f"{name}/{f['name']}: hosted_bytes {hosted_bytes!r} is not a "
                    "positive int"
                )
                assert hosted is not None, (
                    f"{name}/{f['name']}: hosted_bytes without hosted_sha256"
                )


def _file_lists(dataset: dict) -> list[tuple[str, list[dict]]]:
    """``(label, files)`` per variant, or one entry for a flat dataset."""
    if "variants" in dataset:
        return [(vn, v.get("files", [])) for vn, v in dataset["variants"].items()]
    return [("files", dataset.get("files", []))]


def test_pending_upload_flag_matches_the_file_lists():
    """The flag and the file lists must agree, in BOTH directions.

    A zenodo dataset with an empty file list cannot be fetched at all, so leaving
    one unflagged hides a broken demo. A dataset whose files are all pinned no
    longer needs the flag, and a flag left behind is how four datasets came to
    claim they were still awaiting upload after their files were live in the
    records: nothing checked the two against each other.
    """
    m = load_manifest()
    for name, d in m["datasets"].items():
        if d["bucket"] != "zenodo":
            continue
        empty = [label for label, files in _file_lists(d) if not files]
        if d.get("pending_upload"):
            assert empty, (
                f"{name}: flagged pending_upload, but every file list is "
                f"populated — the flag is stale and should be removed"
            )
        else:
            assert not empty, (
                f"{name}: not flagged pending_upload, but {empty} file list(s) "
                f"are empty, so the demo cannot fetch it"
            )


def test_h2afva_has_light_default_and_full_variant():
    """The 5.9 GB timelapse ships as an opt-in; the demo default is the light cut."""
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


# Local scratch the generator deliberately omits (mirrors gen_data_manifest's
# _SCRATCH_SUFFIXES) — kept inline so this dependency-free guard still runs on a
# packaged install with no scripts/.
_SCRATCH_SUFFIXES = (".corrupt", ".part", ".part.validator", ".tmp")


def test_manifest_matches_files_on_disk():
    """Every in-repo dataset file appears in the manifest with a matching name."""
    if not DATA_DIR.is_dir():  # installed wheel, or post-R17 checkout
        pytest.skip("demos/data/ not available (data no longer in the repo)")
    m = load_manifest()
    listed = {f["name"] for d in m["datasets"].values() for f in _all_files(d)}
    for sub in DATA_DIR.iterdir():
        if sub.is_dir() and sub.name != "tests":
            for f in sub.glob("*"):
                if (
                    f.is_file()
                    and not f.name.startswith(".")
                    and not f.name.endswith(_SCRATCH_SUFFIXES)
                ):
                    assert f.name in listed, (
                        f"{sub.name}/{f.name} missing from manifest"
                    )


@pytest.mark.skipif(not GEN_SCRIPT.exists(), reason=_NO_SCRIPT)
def test_committed_manifest_matches_generator():
    """The same drift gate CI runs: the committed manifest must be current."""
    argv = sys.argv[:]
    sys.argv = [str(GEN_SCRIPT), "--check"]
    try:
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(str(GEN_SCRIPT), run_name="__main__")
    finally:
        sys.argv = argv
    assert exc.value.code == 0, (
        "demo-data manifest is stale — run `make gen-data-manifest` and commit"
    )


@pytest.mark.skipif(not GEN_SCRIPT.exists(), reason=_NO_SCRIPT)
def test_drift_gate_detects_a_stale_manifest(monkeypatch):
    """The gate must actually gate: a mutated dataset table makes --check fail."""
    mod = _load_generator()
    monkeypatch.setitem(
        mod.DATASETS,
        "totally_not_a_dataset",
        dict(bucket="regenerate", license="cc0-1.0", dir=""),
    )
    monkeypatch.setattr(sys, "argv", [str(GEN_SCRIPT), "--check"])
    assert mod.main() == 1, "drift gate should return non-zero on a stale manifest"


@pytest.mark.skipif(not GEN_SCRIPT.exists(), reason=_NO_SCRIPT)
def test_regeneration_preserves_entries_when_the_data_is_gone(tmp_path, monkeypatch):
    """R17 step 4 safety: `git rm`-ing demos/data must not wipe the checksums."""
    mod = _load_generator()
    monkeypatch.setattr(mod, "DATA_DIR", tmp_path / "absent")
    committed = json.loads(mod.MANIFEST.read_text())

    preserved = mod.build(committed, prune=False)

    assert preserved["datasets"] == committed["datasets"], (
        "a checkout that cannot see the data must not rewrite the manifest"
    )
    # --prune is the explicit opt-in that DOES empty them.
    assert mod.build(committed, prune=True)["datasets"]["gsplats_kidney"]["files"] == []


@pytest.mark.skipif(not GEN_SCRIPT.exists(), reason=_NO_SCRIPT)
def test_generator_skips_local_scratch_files(tmp_path):
    """Quarantined/partial/temp scratch must never be admitted into the manifest.

    ``ensure_dataset`` could never resolve a ``.corrupt``/``.part``/``.tmp`` copy,
    so listing one would ship a permanently-unfetchable entry.
    """
    mod = _load_generator()
    assert mod._SCRATCH_SUFFIXES == _SCRATCH_SUFFIXES, (
        "the inline copy above has drifted from the generator's list, so the "
        "disk-consistency guard and the generator no longer agree"
    )
    (tmp_path / "dataset.npz").write_bytes(b"data")
    for scratch in (
        "dataset.npz.corrupt",
        "dataset.npz.part",
        "dataset.npz.part.validator",
        "notes.tmp",
    ):
        (tmp_path / scratch).write_bytes(b"scratch")

    kept = sorted(p.name for p in tmp_path.iterdir() if mod._keep(p))
    assert kept == ["dataset.npz"]


@pytest.mark.skipif(not GEN_SCRIPT.exists(), reason=_NO_SCRIPT)
def test_generator_rejects_a_corrupt_lfs_pointer(tmp_path):
    """A malformed LFS pointer must fail loudly, naming the file.

    A non-numeric ``size`` used to abort with a bare ``ValueError``; a pointer
    missing ``oid``/``size`` used to fall through to hashing the ~130-byte stub,
    silently shipping a plausible-but-bogus checksum. An oid that is not a
    lowercase hex sha256 (or a negative size) is just as unusable: the downstream
    checksum compare is case-sensitive against ``hexdigest()``, so such an entry
    could never verify.
    """
    mod = _load_generator()
    header = "version https://git-lfs.github.com/spec/v1\n"

    good = tmp_path / "good.bin"
    good.write_text(f"{header}oid sha256:{'a' * 64}\nsize 123\n")
    assert mod._pointer_checksum(good) == {"sha256": "a" * 64, "bytes": 123}

    for bad_body in (
        f"oid sha256:{'a' * 64}\nsize notanumber\n",  # non-numeric size
        "size 123\n",  # no oid line
        f"oid sha256:{'a' * 64}\n",  # no size line
        "oid sha256:not-a-digest\nsize 123\n",  # oid is not hex
        f"oid sha256:{'a' * 63}\nsize 123\n",  # oid too short
        f"oid sha256:{'A' * 64}\nsize 123\n",  # oid not lowercase
        f"oid sha256:{'a' * 64}\nsize -1\n",  # negative size
    ):
        corrupt = tmp_path / "corrupt.bin"
        corrupt.write_text(header + bad_body)
        with pytest.raises(ValueError, match="corrupt git-LFS pointer"):
            mod._pointer_checksum(corrupt)


# --------------------------------------------------------------------------- #
# The manifest must remain shippable
# --------------------------------------------------------------------------- #
def test_manifest_lives_outside_the_lfs_data_tree():
    """Dependency-free companion guard — runs in any env, tooling or not.

    demos/data/ is ~450 MB of git-LFS payload excluded from BOTH the wheel and
    the sdist. The manifest must never drift back inside it.
    """
    assert data_fetch._DEMOS_DATA_DIR not in MANIFEST_PATH.parents, (
        f"{MANIFEST_PATH} is inside the excluded demos/data/ tree — "
        "load_manifest() would raise FileNotFoundError for every pip user"
    )


def test_manifest_is_shippable_in_the_wheel_and_sdist():
    """The manifest must be inside the packaged tree and outside every exclude.

    Re-runs hatchling's own matcher over the committed build config, so a future
    glob cannot silently un-ship the manifest again. Hatchling compiles
    ``exclude`` with ``pathspec.GitIgnoreSpec.from_lines`` over the whole pattern
    list (hatchling/builders/config.py), so the spec is built the same way here —
    per-pattern matching would misjudge gitignore negation precedence.
    """
    import tomllib  # stdlib (the project floor is 3.12)

    # pathspec rides in via mypy, i.e. the `dev` feature CI's test env uses.
    pathspec = pytest.importorskip("pathspec")

    pyproject = REPO_ROOT / "pyproject.toml"
    if not pyproject.is_file():  # installed wheel — no source tree to check
        pytest.skip("pyproject.toml not available (installed package)")
    try:
        rel = MANIFEST_PATH.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        pytest.skip("manifest resolves outside the source tree (installed package)")

    targets = tomllib.loads(pyproject.read_text(encoding="utf-8"))["tool"]["hatch"][
        "build"
    ]["targets"]
    for target in ("wheel", "sdist"):
        roots = targets[target]["packages"]
        assert any(rel.startswith(f"{root.rstrip('/')}/") for root in roots), (
            f"{rel} is outside the {target} target's packages={roots}"
        )
        patterns = targets[target].get("exclude", [])
        if pathspec.GitIgnoreSpec.from_lines(patterns).match_file(rel):
            culprits = [
                p
                for p in patterns
                if pathspec.GitIgnoreSpec.from_lines([p]).match_file(rel)
            ]
            raise AssertionError(
                f"{rel} is excluded from the {target} by {culprits}. It is a "
                "packaged resource that load_manifest() opens at runtime — move "
                "the manifest or narrow the glob."
            )


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


def test_file_selection_uses_the_resolved_variant(fake_repo):
    manifest, cache = fake_repo

    paths = ensure_dataset(
        "toy_ts",
        variant="full",
        file_names={"ts.gsplats.zarr.zip"},
        manifest=manifest,
        cache_root=cache,
        verbose=False,
    )

    assert [path.read_bytes() for path in paths] == [b"full-timelapse-bytes"]
    assert declared_file_names("toy_ts", variant="full", manifest=manifest) == {
        "ts.gsplats.zarr.zip"
    }


def test_unknown_selected_file_names_the_resolved_variant(fake_repo):
    manifest, cache = fake_repo

    with pytest.raises(ValueError, match="dataset 'toy_ts' variant 'light'.*nope.zip"):
        ensure_dataset(
            "toy_ts",
            file_names={"nope.zip"},
            manifest=manifest,
            cache_root=cache,
            verbose=False,
        )


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
    from luxar.demos._support.runtime.provenance import (
        INPUT_DIGESTS_ATTR,
        _clear_input_digests,
        stamp_input_digests,
    )

    manifest, cache = fake_repo
    _clear_input_digests()
    paths = ensure_dataset(
        "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
    )
    assert len(paths) == 1
    assert paths[0].exists() and paths[0].read_bytes() == b"toy-splat-bytes"
    assert paths[0].parent == cache / "gsplats_toy"
    assert paths.input_digests == {paths[0].name: _sha256(paths[0])}
    scene = type("Scene", (), {"attrs": {}})()
    stamp_input_digests(scene)
    assert scene.attrs[INPUT_DIGESTS_ATTR] == paths.input_digests
    _clear_input_digests()


def test_ensure_dataset_resolves_only_requested_files(fake_repo):
    manifest, cache = fake_repo
    dataset = manifest["datasets"]["gsplats_toy"]
    lfs_dir = data_fetch._DEMOS_DATA_DIR / "gsplats_toy"
    sidecar = lfs_dir / "toy_tracks.npz"
    sidecar.write_bytes(b"toy-track-bytes")
    dataset["files"].append(
        {
            "name": sidecar.name,
            "sha256": _sha256(sidecar),
            "bytes": sidecar.stat().st_size,
        }
    )

    paths = ensure_dataset(
        "gsplats_toy",
        file_names={sidecar.name},
        manifest=manifest,
        cache_root=cache,
        verbose=False,
    )

    assert [path.name for path in paths] == [sidecar.name]
    assert paths[0].read_bytes() == b"toy-track-bytes"
    assert not (cache / "gsplats_toy" / "toy_ch0.gsplats.zarr.zip").exists()


def test_ensure_dataset_rejects_an_unknown_requested_file(fake_repo):
    manifest, cache = fake_repo

    with pytest.raises(ValueError, match="not declared.*missing.npz"):
        ensure_dataset(
            "gsplats_toy",
            file_names={"missing.npz"},
            manifest=manifest,
            cache_root=cache,
            verbose=False,
        )

    assert not (cache / "gsplats_toy").exists()


def test_ensure_dataset_accepts_an_empty_file_selection(fake_repo):
    manifest, cache = fake_repo

    paths = ensure_dataset(
        "gsplats_toy",
        file_names=set(),
        manifest=manifest,
        cache_root=cache,
        verbose=False,
    )

    assert paths == []
    assert not (cache / "gsplats_toy").exists()


def test_ensure_dataset_rejects_part_of_a_positional_pair(fake_repo):
    manifest, cache = fake_repo
    entries = manifest["datasets"]["gsplats_toy"]["files"]
    lfs_dir = data_fetch._DEMOS_DATA_DIR / "gsplats_toy"
    sidecar = lfs_dir / "toy_tracks.npz"
    sidecar.write_bytes(b"toy-track-bytes")
    entries[0]["positional_pair"] = "toy"
    entries.append(
        {
            "name": sidecar.name,
            "sha256": _sha256(sidecar),
            "bytes": sidecar.stat().st_size,
            "positional_pair": "toy",
        }
    )

    with pytest.raises(ValueError, match="positional pair.*toy"):
        ensure_dataset(
            "gsplats_toy",
            file_names={entries[0]["name"]},
            manifest=manifest,
            cache_root=cache,
            verbose=False,
        )

    assert not (cache / "gsplats_toy").exists()


def test_ensure_dataset_resolves_a_complete_positional_pair_only(fake_repo):
    manifest, cache = fake_repo
    entries = manifest["datasets"]["gsplats_toy"]["files"]
    lfs_dir = data_fetch._DEMOS_DATA_DIR / "gsplats_toy"
    sidecar = lfs_dir / "toy_tracks.npz"
    sidecar.write_bytes(b"toy-track-bytes")
    entries[0]["positional_pair"] = "toy"
    entries.extend(
        [
            {
                "name": sidecar.name,
                "sha256": _sha256(sidecar),
                "bytes": sidecar.stat().st_size,
                "positional_pair": "toy",
            },
            {"name": "unselected.bin", "sha256": "f" * 64},
        ]
    )
    pair_names = {entries[0]["name"], sidecar.name}

    paths = ensure_dataset(
        "gsplats_toy",
        file_names=pair_names,
        manifest=manifest,
        cache_root=cache,
        verbose=False,
    )

    assert [path.name for path in paths] == [entries[0]["name"], sidecar.name]
    assert not (cache / "gsplats_toy" / "unselected.bin").exists()


def test_ensure_dataset_resolves_toplevel_dir(tmp_path, monkeypatch):
    """Regression for #821: a ``dir: ""`` dataset lives at the top level of
    demos/data (no ``<name>/`` subdir) and must still resolve from in-repo LFS.

    Pre-fix, ensure_dataset reconstructed the in-repo path as
    ``_DEMOS_DATA_DIR / name``, so it looked in a non-existent ``<name>/``
    subdir, missed the top-level file, found no Zenodo URL, and raised
    FileNotFoundError for a file sitting right there with a correct checksum.
    """
    lfs_root = tmp_path / "demos_data"
    lfs_root.mkdir(parents=True)
    # File written DIRECTLY into the lfs root — no <name>/ subdir.
    payload = lfs_root / "census_umap_1m.npz"
    payload.write_bytes(b"toplevel-census-bytes")

    manifest = {
        "schema_version": 1,
        "records": {
            "cc-by": {"license": "cc-by-4.0", "zenodo_record": None, "base_url": None}
        },
        "datasets": {
            "census_umap_1m": {
                "bucket": "zenodo",
                "record": "cc-by",
                "license": "cc-by-4.0",
                "dir": "",  # top level of demos/data
                "files": [
                    {
                        "name": "census_umap_1m.npz",
                        "sha256": _sha256(payload),
                        "bytes": payload.stat().st_size,
                    }
                ],
            },
        },
    }
    monkeypatch.setattr(data_fetch, "_DEMOS_DATA_DIR", lfs_root)
    cache = tmp_path / "cache"

    paths = ensure_dataset(
        "census_umap_1m", manifest=manifest, cache_root=cache, verbose=False
    )

    assert len(paths) == 1
    assert paths[0].exists()
    assert paths[0].read_bytes() == b"toplevel-census-bytes"
    # Cache still namespaces by dataset name (top-level layout is in-repo only).
    assert paths[0].parent == cache / "census_umap_1m"


def test_ensure_dataset_cache_hit_is_reused(fake_repo, monkeypatch):
    manifest, cache = fake_repo
    ensure_dataset("gsplats_toy", manifest=manifest, cache_root=cache, verbose=False)
    # Now break the in-repo source: a cache hit must not need it.
    monkeypatch.setattr(data_fetch, "_DEMOS_DATA_DIR", cache / "does-not-exist")
    paths = ensure_dataset(
        "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
    )
    assert paths[0].read_bytes() == b"toy-splat-bytes"


def test_ensure_dataset_warm_cache_hit_is_silent_when_quiet(fake_repo, capsys):
    """A warm cache hit under ``verbose=False`` must print nothing at all.

    The manifest sha256 is re-verified on every hit, so a ``verify_file_checksum``
    that ignores ``verbose`` puts a "Verifying …/Computing SHA256…/✓ verified"
    block on screen per file for a caller that explicitly asked for silence.
    """
    manifest, cache = fake_repo
    ensure_dataset("gsplats_toy", manifest=manifest, cache_root=cache, verbose=False)
    capsys.readouterr()  # discard the cold-resolution output

    ensure_dataset("gsplats_toy", manifest=manifest, cache_root=cache, verbose=False)

    assert capsys.readouterr().out == ""


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


# --------------------------------------------------------------------------- #
# DatasetUnavailable: the ONE FileNotFoundError a demo may route around
# --------------------------------------------------------------------------- #
def test_nothing_obtainable_anywhere_is_a_routable_absence(fake_repo):
    """No cache, no in-repo copy, no record → the demo may build its own."""
    manifest, cache = fake_repo
    manifest["datasets"]["gsplats_toy"]["files"][0]["name"] = "absent.zip"
    with pytest.raises(DatasetUnavailable):
        ensure_dataset(
            "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
        )


def test_a_pending_upload_entry_is_a_routable_absence(fake_repo):
    """A dataset the manifest pins no files for yet is likewise just "not there"."""
    manifest, cache = fake_repo
    manifest["datasets"]["gsplats_toy"]["files"] = []
    with pytest.raises(DatasetUnavailable, match="pending upload"):
        ensure_dataset(
            "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
        )


def test_a_bad_in_repo_copy_is_a_fault_not_an_absence(fake_repo):
    """The bytes are RIGHT THERE and wrong — a broken checkout or stale manifest.

    Every demo that falls back to a multi-minute refit when its manifest fetch
    comes up empty catches this narrowly — eleven of them ``except
    DatasetUnavailable``, and ``nexrad_supercell`` that plus
    :class:`~luxar.demos._support.datasets.bundles.BundleMemberNotFound`, the bundle-side routable
    absence (its per-frame member names carry ``--dbz-floor`` and friends, so a
    non-default run legitimately asks the shipped bundle for frames it cannot
    hold). This must not be one of the things any of them swallow: it would
    present a repo fault as a routine rebuild, forever.
    """
    manifest, cache = fake_repo
    payload = data_fetch._DEMOS_DATA_DIR / "gsplats_toy" / "toy_ch0.gsplats.zarr.zip"
    payload.write_bytes(b"toy-splat-BYTES")  # same length, different content

    with pytest.raises(FileNotFoundError) as excinfo:
        ensure_dataset(
            "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
        )
    assert not isinstance(excinfo.value, DatasetUnavailable)


def test_an_unknown_requested_file_is_a_fault_not_an_absence(fake_gsplats_repo):
    """A file list that is not the manifest's is a bug in the demo, not missing data."""
    manifest, cache = fake_gsplats_repo
    with pytest.raises(FileNotFoundError) as excinfo:
        load_dataset_gsplats(
            "gsplats_toy",
            ["invented_at_runtime.gsplats.zarr.zip"],
            manifest=manifest,
            cache_root=cache,
            verbose=False,
        )
    assert not isinstance(excinfo.value, DatasetUnavailable)


def test_a_missing_packaged_manifest_is_a_fault_not_an_absence(tmp_path):
    """A broken install must not look like a demo whose data is not up yet."""
    clear_manifest_cache()
    try:
        with pytest.raises(FileNotFoundError) as excinfo:
            load_manifest(str(tmp_path / "no_such_manifest.json"))
        assert not isinstance(excinfo.value, DatasetUnavailable)
    finally:
        clear_manifest_cache()


def test_missing_and_unhosted_raises_clear_error(fake_repo):
    """No cache, no in-repo file, no Zenodo URL → actionable FileNotFoundError."""
    manifest, cache = fake_repo
    manifest["datasets"]["gsplats_toy"]["files"][0]["name"] = "absent.zip"
    with pytest.raises(FileNotFoundError, match="builds no Zenodo URL for it yet"):
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


def test_unpulled_lfs_pointer_names_the_available_remedies(fake_repo):
    """An unhydrated source checkout must name Git LFS and fallback remedies."""
    manifest, cache = fake_repo
    payload = data_fetch._DEMOS_DATA_DIR / "gsplats_toy" / "toy_ch0.gsplats.zarr.zip"
    payload.write_text(
        f"version https://git-lfs.github.com/spec/v1\noid sha256:{'0' * 64}\nsize 15\n"
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        ensure_dataset(
            "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
        )

    message = str(exc_info.value)
    assert "unpublished draft" in message
    assert "publish the record" in message.lower()
    assert "--recompute" in message
    assert "git lfs pull" in message


def test_hosted_only_archive_does_not_recommend_git_lfs(fake_repo):
    """An archive absent from the repository cannot be hydrated with Git LFS."""
    manifest, cache = fake_repo
    payload = data_fetch._DEMOS_DATA_DIR / "gsplats_toy" / "toy_ch0.gsplats.zarr.zip"
    payload.unlink()

    with pytest.raises(DatasetUnavailable) as exc_info:
        ensure_dataset(
            "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
        )

    message = str(exc_info.value)
    assert "hosted-only" in message
    assert "unpublished draft" in message
    assert "git lfs pull" not in message


def test_installed_package_recommends_source_checkout_git_lfs(fake_repo):
    """A wheel has no data tree, even when the source checkout has the archive."""
    manifest, cache = fake_repo
    shutil.rmtree(data_fetch._DEMOS_DATA_DIR)

    with pytest.raises(DatasetUnavailable) as exc_info:
        ensure_dataset(
            "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
        )

    message = str(exc_info.value)
    assert "installed package ships no demo payloads" in message
    assert "source checkout" in message
    assert "git lfs pull" in message
    assert "hosted-only" not in message


def test_inrepo_source_failing_its_own_checksum_is_never_used(fake_repo):
    """A bad packaged copy is reported, never loaded, and never renamed.

    demos/data is git-tracked, so a .corrupt file there would dirty the working
    tree and break test_manifest_matches_files_on_disk. With no Zenodo URL yet
    there is no good copy, so the only correct outcome is a clear error.
    """
    manifest, cache = fake_repo
    payload = data_fetch._DEMOS_DATA_DIR / "gsplats_toy" / "toy_ch0.gsplats.zarr.zip"
    payload.write_bytes(b"toy-splat-BYTES")  # same length, different content

    # Names BOTH contracts: the point is that neither the in-repo digest nor a
    # hosted one accepted these bytes, so there is nothing to fall back to.
    with pytest.raises(
        FileNotFoundError, match="matches neither its in-repo nor its hosted sha256"
    ):
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

    monkeypatch.setattr(
        "luxar.demos._support.downloads.download.download_with_checksum", _fake_download
    )

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
    assert out.input_digests == {
        "toy_ch0.gsplats.zarr.zip": _sha256(
            cache / "gsplats_toy" / "toy_ch0.gsplats.zarr.zip"
        )
    }


def test_wrapper_points_a_multi_part_store_at_the_graft_entry(
    fake_gsplats_repo, tmp_path, monkeypatch
):
    """A ``kind=partition`` payload has no flat form, so say what does open it.

    Pins the sentinel as well as the wording: the branch keys on the shape
    error's own "matrix-shaped" text, so if that message is ever reworded the
    wrapper silently goes back to surfacing a bare internal error — with a
    remedy the caller cannot guess.
    """
    import numpy as np

    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.io.save_gsplats import write_gsplats_tree

    manifest, cache = fake_gsplats_repo
    n = 64
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, [0, 2, 5]] = 1.0
    data = GSplatData(
        centers=(np.random.rand(n, 3) * 100).astype(np.float32),
        amplitudes=np.ones(n, dtype=np.float32),
        cholesky_factors=chol,
    )
    partition = data.to_spatial_partition(max_elements=16)
    assert len(partition.children) > 1, "the fixture must really be multi-part"
    payload = tmp_path / "demos_data" / "gsplats_toy" / "toy_ch0.gsplats.zarr.zip"
    payload.unlink()
    write_gsplats_tree(payload, partition, compress="zip")
    entry = manifest["datasets"]["gsplats_toy"]["files"][0]
    entry["sha256"] = _sha256(payload)
    entry["bytes"] = payload.stat().st_size

    with pytest.raises(ValueError) as excinfo:
        load_dataset_gsplats(
            "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
        )

    message = str(excinfo.value)
    assert "multi-part" in message
    assert "ensure_dataset('gsplats_toy')" in message
    assert "add_gsplats_from_file" in message
    # The original shape error is kept, not swallowed.
    assert "matrix-shaped" in str(excinfo.value.__cause__)


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


# --------------------------------------------------------------------------- #
# local_fit_path / load_local_fit_gsplats: the demo's OWN artifacts (#1618)
# --------------------------------------------------------------------------- #
def _tiny_gsplats(n: int = 8):
    import numpy as np

    from luxar.gsplats.gsplat_data import GSplatData

    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, [0, 2, 5]] = 1.0
    return GSplatData(
        centers=np.random.rand(n, 3).astype(np.float32),
        amplitudes=np.ones(n, dtype=np.float32),
        cholesky_factors=chol,
    )


def test_local_fit_path_is_namespaced_under_the_dataset_cache_dir(tmp_path):
    path = local_fit_path(
        "gsplats_toy", "toy_ch0.gsplats.zarr.zip", cache_root=tmp_path
    )
    assert path == tmp_path / "gsplats_toy" / "local" / "toy_ch0.gsplats.zarr.zip"
    assert path.parent.name == LOCAL_FIT_DIRNAME
    # The point of the namespace: it is NOT the manifest's destination.
    assert path != tmp_path / "gsplats_toy" / "toy_ch0.gsplats.zarr.zip"


def test_local_fit_path_mirrors_the_variant_layout(tmp_path):
    """A variant's manifest dest is ``<name>/<variant>/<file>``; stay under it."""
    path = local_fit_path("toy_ts", "ts.zip", variant="light", cache_root=tmp_path)
    assert path == tmp_path / "toy_ts" / "light" / "local" / "ts.zip"


def test_local_fit_path_refuses_the_one_colliding_variant_name(tmp_path):
    with pytest.raises(ValueError, match="collide"):
        local_fit_path("toy_ts", "ts.zip", variant="local", cache_root=tmp_path)


def test_a_local_fit_survives_a_later_ensure_dataset(fake_repo):
    """THE regression (#1618): the checksum gate must not see the local fit.

    Written to the manifest's own ``<name>/<file>`` — what every migrated demo
    used to do — the next fetch hashes it, fails, and renames it ``.corrupt``,
    so the demo recomputes on every launch. Asserted both ways here: the local
    copy is untouched and unquarantined, while the same bytes at the colliding
    path ARE quarantined by the same call.
    """
    manifest, cache = fake_repo
    payload = b"an eleven-minute GPU fit that matches no manifest hash"

    local = local_fit_path("gsplats_toy", "toy_ch0.gsplats.zarr.zip", cache_root=cache)
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(payload)

    colliding = cache / "gsplats_toy" / "toy_ch0.gsplats.zarr.zip"
    colliding.parent.mkdir(parents=True, exist_ok=True)
    colliding.write_bytes(payload)

    ensure_dataset("gsplats_toy", manifest=manifest, cache_root=cache, verbose=False)

    assert local.is_file(), "the local fit was deleted by the fetch"
    assert local.read_bytes() == payload, "the local fit was overwritten"
    assert not find_quarantined_files(local), "the local fit was quarantined"
    # The control: at the manifest's own path those same bytes are destroyed.
    assert find_quarantined_files(colliding)
    assert colliding.read_bytes() != payload


def test_load_local_fit_returns_the_data_when_every_file_is_present(tmp_path):
    names = ["a.gsplats.zarr.zip", "b.gsplats.zarr.zip"]
    for i, name in enumerate(names):
        path = local_fit_path("toy", name, cache_root=tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _tiny_gsplats(4 + i).save(path, ordering="none", compress="zip")

    out = load_local_fit_gsplats("toy", names, cache_root=tmp_path, verbose=False)

    assert out is not None
    assert [len(g.amplitudes) for g in out] == [4, 5], "order must follow file_names"


def test_load_local_fit_returns_none_when_any_file_is_missing(tmp_path):
    """A partial set is not an answer: the refit rewrites all of them anyway."""
    names = ["a.gsplats.zarr.zip", "b.gsplats.zarr.zip"]
    present = local_fit_path("toy", names[0], cache_root=tmp_path)
    present.parent.mkdir(parents=True, exist_ok=True)
    _tiny_gsplats().save(present, ordering="none", compress="zip")

    assert (
        load_local_fit_gsplats("toy", names, cache_root=tmp_path, verbose=False) is None
    )
    assert (
        load_local_fit_gsplats("toy", [names[1]], cache_root=tmp_path, verbose=False)
        is None
    )


def test_load_local_fit_reports_a_corrupt_file_and_returns_none(tmp_path, capsys):
    """Unreadable local bytes must not crash the demo — but must not be silent.

    There is no checksum, no remote and no second copy for these files, so the
    only recovery is the refit the caller can already do. Raising would strand
    the demo on rubble it can heal; staying quiet would hide a machine that has
    started refitting on every launch.
    """
    path = local_fit_path("toy", "a.gsplats.zarr.zip", cache_root=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a zip at all")

    out = load_local_fit_gsplats(
        "toy", ["a.gsplats.zarr.zip"], cache_root=tmp_path, verbose=False
    )

    assert out is None
    assert "could not be loaded" in capsys.readouterr().out
    assert path.is_file(), "the bad file is left in place for inspection"


def test_load_local_fit_raises_on_a_multi_part_store(tmp_path):
    """The one failure a rebuild cannot fix, so the one it must not hide.

    A ``kind=partition`` / non-leaf lod tree has no flat ``GSplatData`` form at
    all. Swallowed into ``None``, the caller refits — and the refit writes the
    same unloadable shape, so the demo refits on EVERY launch: exactly the #1618
    symptom this namespace exists to end, with a printed "delete the file if the
    rebuild keeps happening" that cannot help. ``load_dataset_gsplats`` already
    translates this error; so does its local sibling.
    """
    from luxar.demos._lod_policy import save_with_lod

    path = local_fit_path("toy", "a.gsplats.zarr.zip", cache_root=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_with_lod(
        _tiny_gsplats(64),
        path,
        recipe="adaptive",
        device="cpu",
        quiet=True,
        compress="zip",
    )

    with pytest.raises(ValueError, match="multi-part gsplats store"):
        load_local_fit_gsplats(
            "toy", ["a.gsplats.zarr.zip"], cache_root=tmp_path, verbose=False
        )


def test_load_local_fit_at_reads_the_paths_it_is_given(tmp_path):
    """The path-based door does NOT re-derive anything from the cache root.

    A demo that publishes a ``LOCAL_FIT`` constant writes its refit through it;
    if the read door recomputed the path instead, the two halves could point at
    different files and a redirected constant would be silently ignored — which
    is what ct_totalsegmentator did (#1618 review, A).
    """
    elsewhere = tmp_path / "not" / "the" / "cache" / "fit.gsplats.zarr.zip"
    elsewhere.parent.mkdir(parents=True)
    _tiny_gsplats(7).save(elsewhere, ordering="none", compress="zip")

    out = load_local_fit_gsplats_at([elsewhere], verbose=False)

    assert out is not None and len(out[0].amplitudes) == 7
    assert load_local_fit_gsplats_at([tmp_path / "absent.zip"], verbose=False) is None


def test_asking_for_no_files_at_all_is_a_caller_bug(tmp_path):
    """``[]`` is neither a loaded set nor "rebuild it", so it must not be returned.

    Every caller writes ``if fits is not None: fits[0]``; an empty list passes
    that test and then raises ``IndexError`` somewhere else entirely.
    """
    with pytest.raises(ValueError, match="no paths"):
        load_local_fit_gsplats("toy", [], cache_root=tmp_path, verbose=False)
    with pytest.raises(ValueError, match="no paths"):
        load_local_fit_gsplats_at([], verbose=False)


# --------------------------------------------------------------------------- #
# load_dataset_bundle: the manifest-driven bundle path
# --------------------------------------------------------------------------- #
def _write_bundle(path: Path, members: dict[str, bytes]) -> None:
    import zipfile

    with zipfile.ZipFile(path, "w") as zf:
        for name, blob in members.items():
            zf.writestr(name, blob)


@pytest.mark.parametrize("hosted_mode", ["agree", "diverge", "hosted_only"])
def test_load_dataset_bundle_verifies_the_outer_zip_then_extracts(
    tmp_path, monkeypatch, hosted_mode
):
    """The bundle is checksum-verified, then its members are extracted and loaded.

    The point of routing bundles through the manifest is that the OUTER zip -- the
    unit that is actually downloaded -- gets verified. Members are covered by
    verifying the container, so they are not pinned individually.
    """
    from luxar.demos._support.datasets import bundles

    inner = {
        "frame0.gsplats.zarr.zip": b"PK-not-really",
        "frame1.gsplats.zarr.zip": b"x",
    }
    lfs_dir = tmp_path / "repo" / "bundle_ds"
    lfs_dir.mkdir(parents=True)
    bundle = lfs_dir / "b.gsplats.zarr.zip"
    _write_bundle(bundle, inner)
    sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
    local_sha = None if hosted_mode == "hosted_only" else sha
    hosted_sha = sha if hosted_mode in {"agree", "hosted_only"} else "f" * 64

    manifest = {
        "schema_version": 1,
        "records": {"r": {"published": False}},
        "datasets": {
            "bundle_ds": {
                "bucket": "zenodo",
                "record": "r",
                "license": "cc0-1.0",
                "dir": "bundle_ds",
                "files": [
                    {
                        "name": "b.gsplats.zarr.zip",
                        "hosted_sha256": hosted_sha,
                        "superseded_sha256": ["a" * 64],
                        "bytes": bundle.stat().st_size,
                        **({"sha256": local_sha} if local_sha is not None else {}),
                    }
                ],
            }
        },
    }
    monkeypatch.setattr(data_fetch, "_DEMOS_DATA_DIR", tmp_path / "repo")
    loaded: list[tuple[Path, dict]] = []

    def _spy(bp, bn, cd, fn, **kwargs):
        loaded.append((bp, kwargs))
        return list(fn)

    monkeypatch.setattr(bundles, "_extract_bundle_and_load", _spy)

    out = bundles.load_dataset_bundle(
        "bundle_ds",
        "b.gsplats.zarr.zip",
        list(inner),
        cache_root=tmp_path / "cache",
        manifest=manifest,
        verbose=False,
    )
    assert out == list(inner)
    # Resolved through ensure_dataset, so it is the verified CACHE copy that gets
    # extracted, not the working-tree file.
    assert loaded and loaded[0][0].parent == tmp_path / "cache" / "bundle_ds"
    assert loaded[0][0].read_bytes() == bundle.read_bytes()
    # The extracted frames are keyed on the digest that was just verified, not on
    # a (size, mtime) guess a same-size re-upload could reproduce.
    assert loaded[0][1]["stamp"] == f"sha256:{sha}"


def test_load_dataset_bundle_refreshes_frames_after_superseded_bundle_is_replaced(
    tmp_path, monkeypatch
):
    """A superseded extraction must not inherit the current bundle's stamp."""
    from luxar.demos._support.datasets import bundles
    from luxar.gsplats import gsplat_data

    monkeypatch.setattr(
        gsplat_data.GSplatData,
        "load",
        classmethod(lambda cls, path, **kwargs: path.read_bytes()),
    )

    cache_root = tmp_path / "cache"
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    bundle_name = "b.gsplats.zarr.zip"
    frame_name = "frame0.gsplats.zarr.zip"
    cached_bundle = cache_root / "bundle_ds" / bundle_name
    cached_bundle.parent.mkdir(parents=True)
    _write_bundle(cached_bundle, {frame_name: b"old-frame"})
    old_digest = hashlib.sha256(cached_bundle.read_bytes()).hexdigest()

    current_bundle = tmp_path / "current.zip"
    _write_bundle(current_bundle, {frame_name: b"current-frame-is-longer"})
    current_digest = hashlib.sha256(current_bundle.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "records": {"r": {"published": False}},
        "datasets": {
            "bundle_ds": {
                "bucket": "zenodo",
                "record": "r",
                "license": "cc0-1.0",
                "dir": "bundle_ds",
                "files": [
                    {
                        "name": bundle_name,
                        "sha256": current_digest,
                        "bytes": current_bundle.stat().st_size,
                        "superseded_sha256": [old_digest],
                    }
                ],
            }
        },
    }
    monkeypatch.setattr(data_fetch, "_DEMOS_DATA_DIR", repo_root)

    assert bundles.load_dataset_bundle(
        "bundle_ds",
        bundle_name,
        [frame_name],
        cache_root=cache_root,
        manifest=manifest,
        verbose=False,
    ) == [b"old-frame"]

    repo_bundle = repo_root / "bundle_ds" / bundle_name
    repo_bundle.parent.mkdir()
    repo_bundle.write_bytes(current_bundle.read_bytes())

    assert bundles.load_dataset_bundle(
        "bundle_ds",
        bundle_name,
        [frame_name],
        cache_root=cache_root,
        manifest=manifest,
        verbose=False,
    ) == [b"current-frame-is-longer"]


def test_load_dataset_bundle_rejects_a_bundle_that_is_not_a_manifest_file(
    tmp_path, monkeypatch
):
    """Naming a bundle the manifest does not list must raise, not fetch something else."""
    from luxar.demos._support.datasets import bundles

    lfs_dir = tmp_path / "repo" / "bundle_ds"
    lfs_dir.mkdir(parents=True)
    bundle = lfs_dir / "b.gsplats.zarr.zip"
    _write_bundle(bundle, {"f.gsplats.zarr.zip": b"x"})
    manifest = {
        "schema_version": 1,
        "records": {"r": {"published": False}},
        "datasets": {
            "bundle_ds": {
                "bucket": "zenodo",
                "record": "r",
                "license": "cc0-1.0",
                "dir": "bundle_ds",
                "files": [
                    {
                        "name": "b.gsplats.zarr.zip",
                        "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                        "bytes": bundle.stat().st_size,
                    }
                ],
            }
        },
    }
    monkeypatch.setattr(data_fetch, "_DEMOS_DATA_DIR", tmp_path / "repo")
    with pytest.raises(FileNotFoundError, match="not a manifest file"):
        bundles.load_dataset_bundle(
            "bundle_ds",
            "wrong.zip",
            ["f.gsplats.zarr.zip"],
            cache_root=tmp_path / "cache",
            manifest=manifest,
            verbose=False,
        )


def test_load_dataset_bundle_returns_none_for_a_local_compute_dataset():
    """A non-hosted dataset hands control back so the demo builds it itself."""
    from luxar.demos._support.datasets import bundles

    manifest = {
        "schema_version": 1,
        "records": {},
        "datasets": {
            "lc": {
                "bucket": "local-compute",
                "license": "none",
                "reason": "not redistributable",
                "files": [],
            }
        },
    }
    assert (
        bundles.load_dataset_bundle(
            "lc", "b.zip", ["f.zip"], manifest=manifest, verbose=False
        )
        is None
    )


def test_load_dataset_bundle_honours_recompute():
    from luxar.demos._support.datasets import bundles

    assert (
        bundles.load_dataset_bundle(
            "anything", "b.zip", ["f.zip"], recompute=True, verbose=False
        )
        is None
    )


# --------------------------------------------------------------------------- #
# The Zenodo leg: ids are recorded long before the record is public
# --------------------------------------------------------------------------- #
def test_unpublished_record_builds_no_url_even_with_ids():
    """`published: false` keeps the leg dormant, ids notwithstanding.

    Zenodo reserves a DOI at deposition time and the deposition id becomes the
    record id on publication, so both are recorded as soon as the draft exists --
    but a file URL into an unpublished draft 404s for everyone. Returning None
    keeps the caller on the in-repo copy and its clean "not hosted yet" message
    instead of turning that into an HTTP error.
    """
    rec = {
        "zenodo_record": "21912280",
        "zenodo_doi": "10.5281/zenodo.21912280",
        "published": False,
    }
    assert data_fetch.zenodo_file_url(rec, "kidney_ch0.gsplats.zarr.zip") is None


def test_published_record_derives_the_standard_file_url():
    rec = {"zenodo_record": "21912280", "published": True}
    assert data_fetch.zenodo_file_url(rec, "a.zip") == (
        "https://zenodo.org/records/21912280/files/a.zip?download=1"
    )


def test_explicit_base_url_wins_over_the_derived_form():
    rec = {
        "zenodo_record": "21912280",
        "base_url": "https://example.org/files/",
        "published": True,
    }
    assert data_fetch.zenodo_file_url(rec, "a.zip") == (
        "https://example.org/files/a.zip?download=1"
    )


def test_base_url_is_not_gated_by_published():
    """The Sandbox rehearsal: a mirror URL on a record that is still a draft.

    The migration runbook rehearses against sandbox.zenodo.org by pointing a
    record's `base_url` there while keeping the production record an unpublished
    (still deletable) draft. `published` describes THAT record, so it must not
    silence a `base_url` aimed somewhere else — otherwise the rehearsal needs
    `published: true` on a draft, which the audit script would report as LIVE.
    """
    rec = {
        "zenodo_record": "21912280",
        "base_url": "https://sandbox.zenodo.org/records/1234/files",
        "published": False,
    }
    assert data_fetch.zenodo_file_url(rec, "a.zip") == (
        "https://sandbox.zenodo.org/records/1234/files/a.zip?download=1"
    )


def test_dataset_base_url_overrides_only_its_record_download(tmp_path, monkeypatch):
    payload = b"mirrored dataset bytes"
    digest = hashlib.sha256(payload).hexdigest()
    manifest = {
        "schema_version": 1,
        "records": {
            "cc-by": {
                "zenodo_record": "21912280",
                "base_url": None,
                "published": True,
            }
        },
        "datasets": {
            "mirrored": {
                "bucket": "zenodo",
                "record": "cc-by",
                "base_url": "https://example.org/mirrored/",
                "license": "cc-by-4.0",
                "files": [
                    {"name": "data.zip", "sha256": digest, "bytes": len(payload)}
                ],
            }
        },
    }
    requested: list[tuple[str, str]] = []

    def _fake_download(url, dest, *, expected_sha256):
        requested.append((url, expected_sha256))
        dest.write_bytes(payload)
        return dest

    monkeypatch.setattr(data_fetch, "_DEMOS_DATA_DIR", tmp_path / "repo")
    monkeypatch.setattr(
        "luxar.demos._support.downloads.download.download_with_checksum",
        _fake_download,
    )

    (path,) = ensure_dataset(
        "mirrored", manifest=manifest, cache_root=tmp_path / "cache", verbose=False
    )

    assert path.read_bytes() == payload
    assert requested == [("https://example.org/mirrored/data.zip?download=1", digest)]
    assert manifest["records"]["cc-by"]["base_url"] is None


def test_record_without_ids_builds_no_url():
    assert data_fetch.zenodo_file_url({}, "a.zip") is None


def test_a_record_that_omits_published_is_treated_as_reachable():
    """Only an explicit `published: false` gates the URL.

    The generator always emits the flag, so every SHIPPED record carries one --
    but a hand-rolled record (a fixture here, or a `base_url` aimed at a one-off
    mirror during the Sandbox rehearsal) has no reason to, and must not be
    silenced by its absence. Pinned because the audit script reads the same
    field and the two have to agree on what a missing one means.
    """
    assert data_fetch.zenodo_file_url({"zenodo_record": "21912280"}, "a.zip") == (
        "https://zenodo.org/records/21912280/files/a.zip?download=1"
    )
    assert data_fetch.zenodo_file_url(
        {"base_url": "https://example.org/files"}, "a.zip"
    ) == ("https://example.org/files/a.zip?download=1")


def test_every_shipped_record_agrees_with_its_published_flag():
    """Each record's id is recorded, and `published` decides whether a URL exists.

    Guards against wiring a live URL by accident: while a record is an unpublished
    draft it must build no URL, so no demo can start 404ing against something that
    is not public yet. The other direction matters just as much — a record flipped
    to published has to actually resolve to a URL, or the flip is silently inert.
    Written both ways so publication needs no edit here beyond the flag itself.
    """
    m = load_manifest()
    for name, rec in m["records"].items():
        assert rec.get("zenodo_record"), f"{name}: record id should be recorded"
        url = data_fetch.zenodo_file_url(rec, "x.zip")
        # `base_url` is an explicit "the files are HERE" override that outranks the
        # flag (the Sandbox rehearsal), so it belongs on the reachable side here —
        # this has to mirror `zenodo_file_url`, not restate a subset of it.
        if rec.get("published") or rec.get("base_url"):
            assert url, f"{name}: reachable per its flags but builds no URL"
        else:
            assert url is None, f"{name}: unpublished draft, but the leg is live"


def test_neuromast_resolves_to_the_r2_mirror_not_zenodo():
    """The Z-corrected neuromast pair (#2713) must fetch from the R2 mirror.

    Goes through the PRODUCTION resolution path — ``resolved_record`` overlays
    the dataset-level ``base_url`` onto the record (its "the dataset mirror
    deliberately outranks the provenance record URL" contract), which a
    raw-record ``zenodo_file_url`` probe skips. A wiring that resolved to the
    un-updated Zenodo record would serve the OLD Z-stretched bytes (now in
    ``superseded_sha256``) and pass every other test in this file — a
    digest-only re-pin does not touch the resolved host.
    """
    m = load_manifest()
    spec = m["datasets"]["gsplats_4d_neuromast_2ch"]
    assert spec.get("files"), "neuromast dataset lost its file pins"
    record = data_fetch.resolved_record(m, spec)
    for f in spec["files"]:
        url = data_fetch.zenodo_file_url(record, f["name"])
        assert url and "data.luxarviewer.dev/inputs/neuromast-z-2713" in url, (
            f"{f['name']} resolves to {url!r}, not the R2 mirror"
        )


# ---------------------------------------------------------------------------
# Two checksum contracts (`sha256` = the repo's copy, `hosted_sha256` = the
# record's). These are the cases that exist only once the two can disagree — the
# state a refit creates, and the one that used to force Zenodo publication onto
# the critical path of every demo-data PR.
# ---------------------------------------------------------------------------

_HOSTED_ONLY = "f" * 64  # a plausible digest that describes no local bytes


def _diverge(manifest: dict, dataset: str = "gsplats_toy") -> dict:
    """Pin a hosted digest that disagrees with the in-repo one. Returns the entry."""
    entry = manifest["datasets"][dataset]["files"][0]
    entry["hosted_sha256"] = _HOSTED_ONLY
    return entry


def test_a_divergent_hosted_pin_still_resolves_from_the_repo(fake_repo):
    """The whole point: truthful hosted pins must not break a working checkout.

    Before the split there was one field, so pinning the refit's digest made the
    in-repo fallback fail its own checksum — which is why the payloads had to be
    deleted in the same PR, which is why the record had to be published first.
    """
    manifest, cache = fake_repo
    _diverge(manifest)

    (path,) = ensure_dataset(
        "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
    )
    assert path.read_bytes() == b"toy-splat-bytes"
    assert find_quarantined_files(path) == []


def test_a_divergent_pin_does_not_churn_the_cache_on_the_second_call(fake_repo):
    """The second call is the real test — the cache slot has no provenance.

    ``dest`` is keyed on (dataset, variant, basename) and nothing else, with no
    record of which source filled it. A cache leg stricter than the leg that
    wrote it would quarantine its own copy and re-copy it every single run.
    """
    manifest, cache = fake_repo
    _diverge(manifest)

    (first,) = ensure_dataset(
        "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
    )
    stat_before = first.stat()
    (second,) = ensure_dataset(
        "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
    )

    assert second == first
    assert second.read_bytes() == b"toy-splat-bytes"
    assert find_quarantined_files(second) == [], "cache churned on a divergent pin"
    assert second.stat().st_mtime == stat_before.st_mtime, "file was rewritten"


def test_a_divergent_pin_is_reported_not_silently_accepted(
    fake_repo, capsys, monkeypatch
):
    """Serving the older generation is allowed, but it has to be said out loud."""
    manifest, cache = fake_repo
    _diverge(manifest)
    real_sha256 = hashlib.sha256
    hash_passes = 0

    def _counting_sha256(*args, **kwargs):
        nonlocal hash_passes
        hash_passes += 1
        return real_sha256(*args, **kwargs)

    monkeypatch.setattr(hashlib, "sha256", _counting_sha256)

    ensure_dataset("gsplats_toy", manifest=manifest, cache_root=cache, verbose=True)

    out = capsys.readouterr().out
    assert hash_passes == 1, out
    assert "SHA256 mismatch" not in out, out
    assert "hosted_sha256 differs" in out, out


def test_a_current_hosted_positional_pair_is_not_reverted(fake_repo, capsys):
    """The current hosted pair survives with an older in-repo copy available.

    This is the #2454 defect 2 shape missing from the existing hosted-cache
    coverage: both files belong to one positional pair, carry superseded history,
    and one still has an in-repo generation the resolver could copy over it. The
    assertion is intentionally scoped to the current record generation; a cached
    superseded generation is still refreshed when a current source exists, as
    ``test_a_superseded_cache_is_replaced_when_a_route_exists`` requires.
    """
    manifest, cache = fake_repo
    entries = manifest["datasets"]["gsplats_toy"]["files"]
    entries[:] = [
        {
            "name": "fit.gsplats.zarr.zip",
            "sha256": hashlib.sha256(b"local-fit").hexdigest(),
            "hosted_sha256": hashlib.sha256(b"hosted-fit").hexdigest(),
            "superseded_sha256": [hashlib.sha256(b"old-hosted-fit").hexdigest()],
            "positional_pair": "toy",
        },
        {
            "name": "colors.npz",
            "sha256": hashlib.sha256(b"local-colors").hexdigest(),
            "hosted_sha256": hashlib.sha256(b"hosted-colors").hexdigest(),
            "superseded_sha256": [hashlib.sha256(b"old-hosted-colors").hexdigest()],
            "positional_pair": "toy",
        },
    ]
    lfs_dir = data_fetch._DEMOS_DATA_DIR / "gsplats_toy"
    (lfs_dir / "fit.gsplats.zarr.zip").write_bytes(b"local-fit")

    cache_dir = cache / "gsplats_toy"
    cache_dir.mkdir(parents=True)
    (cache_dir / "fit.gsplats.zarr.zip").write_bytes(b"hosted-fit")
    (cache_dir / "colors.npz").write_bytes(b"hosted-colors")

    paths = ensure_dataset(
        "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
    )

    assert [path.read_bytes() for path in paths] == [
        b"hosted-fit",
        b"hosted-colors",
    ], "cache was reverted to the in-repo copy"
    assert [find_quarantined_files(path) for path in paths] == [
        [],
        [],
    ], "the correctly-seeded pair was churned"
    assert "Using SUPERSEDED positional pair" not in capsys.readouterr().out


def test_agreeing_pins_report_nothing_unusual(fake_repo, capsys):
    """A hosted pin EQUAL to the local one is the 16-dataset majority case.

    It must be indistinguishable from having no hosted pin at all, or the notice
    becomes noise that trains people to ignore it.
    """
    manifest, cache = fake_repo
    entry = manifest["datasets"]["gsplats_toy"]["files"][0]
    entry["hosted_sha256"] = entry["sha256"]

    ensure_dataset("gsplats_toy", manifest=manifest, cache_root=cache, verbose=True)

    out = capsys.readouterr().out
    assert "hosted_sha256 differs" not in out, out


def test_the_download_leg_is_strict_on_the_hosted_digest(fake_repo, monkeypatch):
    """Bytes from the record must be the RECORD's bytes.

    Accepting either contract is for bytes already in hand. A download verified
    against the in-repo digest would happily store the wrong artifact.
    """
    manifest, cache = fake_repo
    entry = _diverge(manifest)
    manifest["records"]["cc-by"]["base_url"] = "https://example.invalid/files"
    # No in-repo copy, so step 3 is the only leg left.
    monkeypatch.setattr(data_fetch, "_DEMOS_DATA_DIR", cache / "does-not-exist")

    seen: dict[str, object] = {}

    def _fake_download(url, output_path, expected_sha256=None, **kw):
        seen["expected"] = expected_sha256
        Path(output_path).write_bytes(b"hosted-bytes")
        return Path(output_path)

    monkeypatch.setattr(
        "luxar.demos._support.downloads.download.download_with_checksum", _fake_download
    )
    ensure_dataset("gsplats_toy", manifest=manifest, cache_root=cache, verbose=False)

    assert seen["expected"] == entry["hosted_sha256"], (
        "the download leg must verify against the hosted digest, not the in-repo one"
    )


def test_bytes_matching_neither_contract_are_still_quarantined(fake_repo, capsys):
    """Widening acceptance to two digests must not widen it to anything.

    The corruption case is what the quarantine exists for, and a two-contract
    check that accepted a third thing would be worse than the one-contract check
    it replaced.
    """
    manifest, cache = fake_repo
    _diverge(manifest)
    (good,) = ensure_dataset(
        "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
    )
    _corrupt_in_place_preserving_stat(good)

    (repaired,) = ensure_dataset(
        "gsplats_toy", manifest=manifest, cache_root=cache, verbose=True
    )

    out = capsys.readouterr().out
    assert "SHA256 mismatch" in out
    assert "Expected hosted:" in out
    assert "Expected in-repo:" in out
    assert repaired.read_bytes() == b"toy-splat-bytes"
    assert [p.name for p in find_quarantined_files(good)] == [good.name + ".corrupt"]


def test_collapsed_record_pin_is_labelled_as_record_on_mismatch(fake_repo, capsys):
    manifest, cache = fake_repo
    entry = manifest["datasets"]["gsplats_toy"]["files"][0]
    entry["superseded_sha256"] = ["0" * 64]
    (good,) = ensure_dataset(
        "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
    )
    _corrupt_in_place_preserving_stat(good)

    ensure_dataset("gsplats_toy", manifest=manifest, cache_root=cache, verbose=True)

    out = capsys.readouterr().out
    assert "Expected record:" in out
    assert "Expected in-repo:" not in out
    assert "matches neither the record nor a superseded sha256" in out


def test_a_divergent_pin_on_a_variant_needs_no_extra_plumbing(fake_repo):
    """Variants resolve through the same single loop, so they inherit this.

    ``resolve_variant`` returns the same file-entry shape either way — worth
    pinning, because h2afva (the dataset the refit actually diverged) declares
    its files only under ``variants``.
    """
    manifest, cache = fake_repo
    entry = manifest["datasets"]["toy_ts"]["variants"]["full"]["files"][0]
    entry["hosted_sha256"] = _HOSTED_ONLY

    (path,) = ensure_dataset(
        "toy_ts", variant="full", manifest=manifest, cache_root=cache, verbose=False
    )
    assert path.read_bytes() == b"full-timelapse-bytes"
    assert find_quarantined_files(path) == []


@pytest.mark.skipif(not GEN_SCRIPT.exists(), reason=_NO_SCRIPT)
def test_a_hosted_pin_survives_regeneration_when_the_data_is_visible(
    tmp_path, monkeypatch
):
    """The generator cannot DERIVE a hosted digest, so it must carry it forward.

    This is the silent-loss case: a dataset whose data dir is visible has its
    entries rebuilt from disk, so a hand-added ``hosted_sha256`` would evaporate
    on the next ``make gen-data-manifest`` — while a dataset with no dir on disk
    kept its own, because those keep the committed list whole. A field that
    survives in some rows and vanishes in others is worse than one that never
    worked at all.
    """
    mod = _load_generator()
    data = tmp_path / "data" / "gsplats_kidney"
    data.mkdir(parents=True)
    (data / "kidney_ch0.gsplats.zarr.zip").write_bytes(b"kidney-bytes")
    monkeypatch.setattr(mod, "DATA_DIR", tmp_path / "data")

    hosted = "b" * 64
    committed = {
        "datasets": {
            "gsplats_kidney": {
                "files": [
                    {
                        "name": "kidney_ch0.gsplats.zarr.zip",
                        "sha256": "a" * 64,  # deliberately NOT the on-disk digest
                        "bytes": 1,
                        "hosted_sha256": hosted,
                        "hosted_bytes": 4242,
                    }
                ]
            }
        }
    }
    entry = mod.build(committed, prune=False)["datasets"]["gsplats_kidney"]["files"][0]

    assert entry["hosted_sha256"] == hosted, "hosted pin lost on regeneration"
    # The SIZE has to survive too so the hosted artifact remains fully
    # described by a paired digest and size.
    assert entry["hosted_bytes"] == 4242, "hosted size lost on regeneration"
    # The local digest is still re-derived from disk — that is the whole point of
    # the split, and it must not be frozen along with the hosted one.
    assert entry["sha256"] == hashlib.sha256(b"kidney-bytes").hexdigest()


@pytest.mark.skipif(not GEN_SCRIPT.exists(), reason=_NO_SCRIPT)
def test_regeneration_adds_no_hosted_key_where_there_was_none(tmp_path, monkeypatch):
    """Absent must stay absent, or the drift gate reddens for every dataset.

    ``--check`` is a whole-file byte comparison, and
    ``test_regeneration_preserves_entries_when_the_data_is_gone`` asserts an
    exact dict equality — an unconditional ``hosted_sha256: None`` would fail
    both for all 30 datasets at once.
    """
    mod = _load_generator()
    data = tmp_path / "data" / "gsplats_kidney"
    data.mkdir(parents=True)
    (data / "kidney_ch0.gsplats.zarr.zip").write_bytes(b"kidney-bytes")
    monkeypatch.setattr(mod, "DATA_DIR", tmp_path / "data")

    # The committed pin must MATCH the bytes on disk here. A mismatch is a
    # legitimate pin change, which now records the outgoing digest as
    # `superseded_sha256` — correct behaviour, but a different subject than the
    # one this test is about, and conflating them would make the assertion below
    # untestable.
    committed = {
        "datasets": {
            "gsplats_kidney": {
                "files": [
                    {
                        "name": "kidney_ch0.gsplats.zarr.zip",
                        "sha256": hashlib.sha256(b"kidney-bytes").hexdigest(),
                        "bytes": len(b"kidney-bytes"),
                    }
                ]
            }
        }
    }
    entry = mod.build(committed, prune=False)["datasets"]["gsplats_kidney"]["files"][0]
    assert "hosted_sha256" not in entry
    assert "superseded_sha256" not in entry, "an unchanged pin recorded history"
    assert list(entry) == ["name", "sha256", "bytes"], "entry key order changed"


@pytest.mark.skipif(not GEN_SCRIPT.exists(), reason=_NO_SCRIPT)
def test_a_hosted_pin_survives_regeneration_on_a_variant(tmp_path, monkeypatch):
    """Variants take a separate code path, and h2afva is the diverged dataset.

    Its files are declared only under ``variants``, so a carry-forward that
    covered only the flat list would miss the one dataset this exists for.
    """
    mod = _load_generator()
    base = tmp_path / "data" / "h2afva" / "253tp"
    base.mkdir(parents=True)
    (base / "big.gsplats.zarr.zip").write_bytes(b"big-bytes")
    monkeypatch.setattr(mod, "DATA_DIR", tmp_path / "data")
    monkeypatch.setitem(
        mod.DATASETS,
        "h2afva",
        {"bucket": "zenodo", "record": "h2afva", "variants": {"253tp": {}}},
    )

    hosted = "c" * 64
    committed = {
        "datasets": {
            "h2afva": {
                "variants": {
                    "253tp": {
                        "files": [
                            {
                                "name": "big.gsplats.zarr.zip",
                                "sha256": "a" * 64,
                                "bytes": 1,
                                "hosted_sha256": hosted,
                            }
                        ]
                    }
                }
            }
        }
    }
    built = mod.build(committed, prune=False)["datasets"]["h2afva"]
    assert built["variants"]["253tp"]["files"][0]["hosted_sha256"] == hosted


@pytest.mark.skipif(not GEN_SCRIPT.exists(), reason=_NO_SCRIPT)
def test_a_mixed_dataset_gets_hosted_pins_only_where_it_had_them(tmp_path, monkeypatch):
    """One file diverged, its sibling did not — the real Visible Human shape.

    Guards the seam the two single-shape tests both miss: with SOME hosted pin
    present the carry-forward runs, so a lookup that cannot distinguish "no pin
    for this file" from "no pins at all" stamps ``hosted_sha256: None`` onto the
    sibling. That reddens the byte-exact drift gate, and a null download contract
    would reject a file that is perfectly fine.
    """
    mod = _load_generator()
    data = tmp_path / "data" / "gsplats_visible_human_head"
    data.mkdir(parents=True)
    (data / "vh_head.gsplats.zarr.zip").write_bytes(b"fit-bytes")
    (data / "vh_head_colors.npz").write_bytes(b"colour-bytes")
    monkeypatch.setattr(mod, "DATA_DIR", tmp_path / "data")

    hosted = "d" * 64
    committed = {
        "datasets": {
            "gsplats_visible_human_head": {
                "files": [
                    {
                        "name": "vh_head.gsplats.zarr.zip",
                        "sha256": "a" * 64,
                        "bytes": 1,
                        "hosted_sha256": hosted,
                    },
                    {"name": "vh_head_colors.npz", "sha256": "b" * 64, "bytes": 1},
                ]
            }
        }
    }
    built = mod.build(committed, prune=False)["datasets"]
    entries = {e["name"]: e for e in built["gsplats_visible_human_head"]["files"]}

    assert entries["vh_head.gsplats.zarr.zip"]["hosted_sha256"] == hosted
    assert "hosted_sha256" not in entries["vh_head_colors.npz"], (
        "a file with no hosted pin was stamped with one"
    )


def test_a_hosted_only_pin_is_still_enforced(fake_repo, monkeypatch):
    """``sha256`` absent but ``hosted_sha256`` present must NOT read as unverifiable.

    A reachable state: a ``pending_upload`` row's pin has always described the
    hosted artifact (there is nothing on disk to hash), so moving it to
    ``hosted_sha256`` leaves no local digest. If "unverifiable" were keyed on
    ``sha256`` alone, that row would accept any bytes at all — turning a row we
    know the digest of into the one row we check least.
    """
    manifest, cache = fake_repo
    entry = manifest["datasets"]["gsplats_toy"]["files"][0]
    entry["hosted_sha256"] = entry.pop("sha256")
    monkeypatch.setattr(data_fetch, "_DEMOS_DATA_DIR", cache / "does-not-exist")

    dest = cache / "gsplats_toy" / "toy_ch0.gsplats.zarr.zip"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"wrong-bytes-XXX")  # same length as the real payload

    with pytest.raises(DatasetUnavailable):
        ensure_dataset(
            "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
        )
    assert [p.name for p in find_quarantined_files(dest)] == [dest.name + ".corrupt"]


def test_a_hosted_only_pin_accepts_the_bytes_it_describes(fake_repo, monkeypatch):
    """The other half: correct bytes under a hosted-only pin are reused, not rejected."""
    manifest, cache = fake_repo
    entry = manifest["datasets"]["gsplats_toy"]["files"][0]
    entry["hosted_sha256"] = entry.pop("sha256")
    monkeypatch.setattr(data_fetch, "_DEMOS_DATA_DIR", cache / "does-not-exist")

    dest = cache / "gsplats_toy" / "toy_ch0.gsplats.zarr.zip"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"toy-splat-bytes")

    (path,) = ensure_dataset(
        "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
    )
    assert path == dest
    assert find_quarantined_files(dest) == []


# ---------------------------------------------------------------------------
# `superseded_sha256`: a re-pin must never brick a dataset
#
# #1734 re-pinned `gsplats_3d_drosophila_gastrulation` to the bytes its Zenodo
# record holds. Every existing cache held the previous generation, there is no
# in-repo copy, and the record is an unpublished draft — so the cache was
# quarantined as a mismatch and the demo became unbuildable by anyone. Ten of
# thirty datasets are hosted-only, so this was a class, not an incident.
#
# Same shape as #854, where `cached_download` quarantined a COMPLETE file
# against a stale `expected_size` and re-downloaded it forever. There the fix
# was to stop quarantining. Here a digest mismatch is ambiguous — corrupt and
# superseded look identical — so the previous digests are recorded to tell them
# apart, and only then is the quarantine relaxed.
# ---------------------------------------------------------------------------


def _pin_superseded(manifest, dataset="gsplats_toy", digests=()):
    entry = manifest["datasets"][dataset]["files"][0]
    entry["superseded_sha256"] = list(digests)
    return entry


def test_a_superseded_cache_is_kept_when_nothing_can_replace_it(fake_repo, monkeypatch):
    """The droso case exactly: re-pinned, no in-repo copy, record unpublished.

    Quarantining here destroys the last copy in existence and leaves the dataset
    unobtainable. Keeping it is the only outcome that is not strictly worse.
    """
    manifest, cache = fake_repo
    entry = manifest["datasets"]["gsplats_toy"]["files"][0]
    old_digest = entry["sha256"]
    # Seed the cache with the OLD generation, then re-pin to a new one.
    dest = cache / "gsplats_toy" / "toy_ch0.gsplats.zarr.zip"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"toy-splat-bytes")
    entry["sha256"] = "9" * 64
    entry["superseded_sha256"] = [old_digest]
    # No in-repo copy, and no Zenodo URL: irreplaceable.
    monkeypatch.setattr(data_fetch, "_DEMOS_DATA_DIR", cache / "does-not-exist")

    (path,) = ensure_dataset(
        "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
    )

    assert path == dest
    assert path.read_bytes() == b"toy-splat-bytes"
    assert find_quarantined_files(dest) == [], "the last copy was destroyed"


def test_a_positional_pair_rejects_mixed_current_and_superseded_caches(
    fake_repo, monkeypatch
):
    """An irreplaceable fallback must not pair stale payload with a current fit."""
    manifest, cache = fake_repo
    entries = manifest["datasets"]["gsplats_toy"]["files"]
    entries[:] = [
        {
            "name": "fit.gsplats.zarr.zip",
            "sha256": hashlib.sha256(b"current-fit").hexdigest(),
            "superseded_sha256": [hashlib.sha256(b"old-fit").hexdigest()],
            "positional_pair": "toy",
        },
        {
            "name": "colors.npz",
            "sha256": hashlib.sha256(b"current-colors").hexdigest(),
            "superseded_sha256": [hashlib.sha256(b"old-colors").hexdigest()],
            "positional_pair": "toy",
        },
    ]
    pair_cache = cache / "gsplats_toy"
    pair_cache.mkdir(parents=True)
    (pair_cache / "fit.gsplats.zarr.zip").write_bytes(b"current-fit")
    (pair_cache / "colors.npz").write_bytes(b"old-colors")
    repo_root = cache / "repo"
    repo_pair = repo_root / "gsplats_toy"
    repo_pair.mkdir(parents=True)
    pointer = (
        f"version https://git-lfs.github.com/spec/v1\noid sha256:{'0' * 64}\nsize 15\n"
    )
    (repo_pair / "fit.gsplats.zarr.zip").write_text(pointer)
    monkeypatch.setattr(data_fetch, "_DEMOS_DATA_DIR", repo_root)

    with pytest.raises(DatasetUnavailable, match="positional pair.*toy") as excinfo:
        ensure_dataset(
            "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
        )

    message = str(excinfo.value)
    assert "gsplats_toy" in message
    assert "fit.gsplats.zarr.zip: In a source checkout, run `git lfs pull`." in message
    assert (
        "colors.npz: This archive is hosted-only and has no in-repo Git LFS copy."
        in message
    )


def test_a_complete_superseded_positional_pair_remains_usable(fake_repo, monkeypatch):
    """One unavailable member keeps every partner on the complete prior generation."""
    manifest, cache = fake_repo
    entries = manifest["datasets"]["gsplats_toy"]["files"]
    entries[:] = [
        {
            "name": "fit.gsplats.zarr.zip",
            "sha256": hashlib.sha256(b"current-fit").hexdigest(),
            "superseded_sha256": [hashlib.sha256(b"old-fit").hexdigest()],
            "positional_pair": "toy",
        },
        {
            "name": "colors.npz",
            "sha256": hashlib.sha256(b"current-colors").hexdigest(),
            "superseded_sha256": [hashlib.sha256(b"old-colors").hexdigest()],
            "positional_pair": "toy",
        },
    ]
    pair_cache = cache / "gsplats_toy"
    pair_cache.mkdir(parents=True)
    (pair_cache / "fit.gsplats.zarr.zip").write_bytes(b"old-fit")
    (pair_cache / "colors.npz").write_bytes(b"old-colors")
    partial_repo = cache / "partial-repo"
    repo_pair = partial_repo / "gsplats_toy"
    repo_pair.mkdir(parents=True)
    (repo_pair / "fit.gsplats.zarr.zip").write_bytes(b"current-fit")
    monkeypatch.setattr(data_fetch, "_DEMOS_DATA_DIR", partial_repo)

    paths = ensure_dataset(
        "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
    )

    assert [path.read_bytes() for path in paths] == [b"old-fit", b"old-colors"]
    assert list(paths.input_digests) == ["colors.npz", "fit.gsplats.zarr.zip"]
    assert paths.input_digests == {
        "colors.npz": hashlib.sha256(b"old-colors").hexdigest(),
        "fit.gsplats.zarr.zip": hashlib.sha256(b"old-fit").hexdigest(),
    }


def test_a_complete_positional_pair_refreshes_when_current_sources_exist(
    fake_repo, monkeypatch, capsys
):
    """Available current members must not be pinned to a complete stale pair."""
    manifest, cache = fake_repo
    entries = manifest["datasets"]["gsplats_toy"]["files"]
    entries[:] = [
        {
            "name": "fit.gsplats.zarr.zip",
            "sha256": hashlib.sha256(b"current-fit").hexdigest(),
            "superseded_sha256": [hashlib.sha256(b"old-fit").hexdigest()],
            "positional_pair": "toy",
        },
        {
            "name": "colors.npz",
            "sha256": hashlib.sha256(b"current-colors").hexdigest(),
            "superseded_sha256": [hashlib.sha256(b"old-colors").hexdigest()],
            "positional_pair": "toy",
        },
    ]
    pair_cache = cache / "gsplats_toy"
    pair_cache.mkdir(parents=True)
    (pair_cache / "fit.gsplats.zarr.zip").write_bytes(b"old-fit")
    (pair_cache / "colors.npz").write_bytes(b"old-colors")
    repo_root = cache / "repo"
    repo_pair = repo_root / "gsplats_toy"
    repo_pair.mkdir(parents=True)
    (repo_pair / "fit.gsplats.zarr.zip").write_bytes(b"current-fit")
    (repo_pair / "colors.npz").write_bytes(b"current-colors")
    monkeypatch.setattr(data_fetch, "_DEMOS_DATA_DIR", repo_root)

    paths = ensure_dataset(
        "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
    )

    assert [path.read_bytes() for path in paths] == [b"current-fit", b"current-colors"]
    captured = capsys.readouterr()
    assert "Using SUPERSEDED positional pair" not in captured.out + captured.err


def test_a_current_positional_pair_reports_each_verification_once(
    fake_repo, monkeypatch, capsys
):
    """Pair preflight must not narrate the resolver's checksum pass twice."""
    manifest, cache = fake_repo
    entries = manifest["datasets"]["gsplats_toy"]["files"]
    entries[:] = [
        {
            "name": "fit.gsplats.zarr.zip",
            "sha256": hashlib.sha256(b"current-fit").hexdigest(),
            "superseded_sha256": [hashlib.sha256(b"old-fit").hexdigest()],
            "positional_pair": "toy",
        },
        {
            "name": "colors.npz",
            "sha256": hashlib.sha256(b"current-colors").hexdigest(),
            "superseded_sha256": [hashlib.sha256(b"old-colors").hexdigest()],
            "positional_pair": "toy",
        },
    ]
    pair_cache = cache / "gsplats_toy"
    pair_cache.mkdir(parents=True)
    (pair_cache / "fit.gsplats.zarr.zip").write_bytes(b"current-fit")
    (pair_cache / "colors.npz").write_bytes(b"current-colors")
    monkeypatch.setattr(data_fetch, "_DEMOS_DATA_DIR", cache / "does-not-exist")

    ensure_dataset("gsplats_toy", manifest=manifest, cache_root=cache)

    out = capsys.readouterr().out
    assert out.count("Verifying fit.gsplats.zarr.zip") == 1
    assert out.count("Verifying colors.npz") == 1
    assert out.index("Ensuring dataset (gsplats_toy)") < out.index(
        "Verifying fit.gsplats.zarr.zip"
    )


def test_a_positional_pair_rejects_different_superseded_depths(fake_repo, monkeypatch):
    """Two last-history matches are not one generation when their depths differ."""
    manifest, cache = fake_repo
    entries = manifest["datasets"]["gsplats_toy"]["files"]
    entries[:] = [
        {
            "name": "fit.gsplats.zarr.zip",
            "sha256": hashlib.sha256(b"current-fit").hexdigest(),
            "superseded_sha256": [hashlib.sha256(b"old-fit").hexdigest()],
            "positional_pair": "toy",
        },
        {
            "name": "colors.npz",
            "sha256": hashlib.sha256(b"current-colors").hexdigest(),
            "superseded_sha256": [
                hashlib.sha256(b"ancient-colors").hexdigest(),
                hashlib.sha256(b"old-colors").hexdigest(),
            ],
            "positional_pair": "toy",
        },
    ]
    pair_cache = cache / "gsplats_toy"
    pair_cache.mkdir(parents=True)
    (pair_cache / "fit.gsplats.zarr.zip").write_bytes(b"old-fit")
    (pair_cache / "colors.npz").write_bytes(b"old-colors")
    monkeypatch.setattr(data_fetch, "_DEMOS_DATA_DIR", cache / "does-not-exist")

    with pytest.raises(DatasetUnavailable, match="not the same generation"):
        ensure_dataset(
            "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
        )


def test_a_superseded_cache_with_an_lfs_pointer_names_git_lfs(fake_repo, capsys):
    """An unpulled current copy is obtainable by hydrating its LFS pointer."""
    manifest, cache = fake_repo
    entry = manifest["datasets"]["gsplats_toy"]["files"][0]
    old_digest = entry["sha256"]
    dest = cache / "gsplats_toy" / "toy_ch0.gsplats.zarr.zip"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"toy-splat-bytes")
    entry["sha256"] = "9" * 64
    entry["superseded_sha256"] = [old_digest]
    payload = data_fetch._DEMOS_DATA_DIR / "gsplats_toy" / dest.name
    payload.write_text(
        f"version https://git-lfs.github.com/spec/v1\noid sha256:{'0' * 64}\nsize 15\n"
    )

    (path,) = ensure_dataset(
        "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
    )

    assert path == dest
    notice = capsys.readouterr().out
    assert "git lfs pull" in notice
    assert "hosted-only" not in notice


def test_a_superseded_positional_pair_names_each_source_remedy(
    fake_repo, monkeypatch, capsys
):
    """Every bypassed pair member keeps its own actionable source remedy."""
    manifest, cache = fake_repo
    entries = manifest["datasets"]["gsplats_toy"]["files"]
    entries[:] = [
        {
            "name": "fit.gsplats.zarr.zip",
            "sha256": hashlib.sha256(b"current-fit").hexdigest(),
            "superseded_sha256": [hashlib.sha256(b"old-fit").hexdigest()],
            "positional_pair": "toy",
        },
        {
            "name": "colors.npz",
            "sha256": hashlib.sha256(b"current-colors").hexdigest(),
            "superseded_sha256": [hashlib.sha256(b"old-colors").hexdigest()],
            "positional_pair": "toy",
        },
    ]
    pair_cache = cache / "gsplats_toy"
    pair_cache.mkdir(parents=True)
    (pair_cache / "fit.gsplats.zarr.zip").write_bytes(b"old-fit")
    (pair_cache / "colors.npz").write_bytes(b"old-colors")
    repo_root = cache / "repo"
    repo_pair = repo_root / "gsplats_toy"
    repo_pair.mkdir(parents=True)
    pointer = (
        f"version https://git-lfs.github.com/spec/v1\noid sha256:{'0' * 64}\nsize 15\n"
    )
    (repo_pair / "fit.gsplats.zarr.zip").write_text(pointer)
    monkeypatch.setattr(data_fetch, "_DEMOS_DATA_DIR", repo_root)

    paths = ensure_dataset(
        "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
    )

    assert [path.read_bytes() for path in paths] == [b"old-fit", b"old-colors"]
    notice = capsys.readouterr().out
    assert "gsplats_toy" in notice
    assert "fit.gsplats.zarr.zip: In a source checkout, run `git lfs pull`." in notice
    assert (
        "colors.npz: This archive is hosted-only and has no in-repo Git LFS copy."
        in notice
    )


def test_a_superseded_cache_is_replaced_when_a_route_exists(fake_repo, monkeypatch):
    """Keeping it is a LAST resort, not a preference.

    If the current bytes are obtainable, the superseded copy must lose — otherwise
    a stale cache would pin a machine to an old generation forever.
    """
    manifest, cache = fake_repo
    entry = manifest["datasets"]["gsplats_toy"]["files"][0]
    old_digest = entry["sha256"]
    dest = cache / "gsplats_toy" / "toy_ch0.gsplats.zarr.zip"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"toy-splat-bytes")
    entry["superseded_sha256"] = [old_digest]
    # Re-pin to whatever the in-repo copy hashes to, so step 2 is a live route.
    payload = data_fetch._DEMOS_DATA_DIR / "gsplats_toy" / "toy_ch0.gsplats.zarr.zip"
    payload.write_bytes(b"NEWER-generation")
    entry["sha256"] = _sha256(payload)

    (path,) = ensure_dataset(
        "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
    )

    assert path.read_bytes() == b"NEWER-generation", "kept a superseded copy anyway"
    assert [p.name for p in find_quarantined_files(dest)] == [dest.name + ".corrupt"]


def test_a_superseded_cache_is_replaced_when_zenodo_is_reachable(
    fake_repo, monkeypatch
):
    """A live download route also makes the superseded cache replaceable."""
    manifest, cache = fake_repo
    entry = manifest["datasets"]["gsplats_toy"]["files"][0]
    old_digest = entry["sha256"]
    dest = cache / "gsplats_toy" / "toy_ch0.gsplats.zarr.zip"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"toy-splat-bytes")
    entry["superseded_sha256"] = [old_digest]
    entry["sha256"] = hashlib.sha256(b"NEWER-generation").hexdigest()
    manifest["records"]["cc-by"]["base_url"] = "https://example.invalid/files"
    monkeypatch.setattr(data_fetch, "_DEMOS_DATA_DIR", cache / "does-not-exist")

    def _fake_download(url, output_path, expected_sha256=None, **kw):
        Path(output_path).write_bytes(b"NEWER-generation")
        return Path(output_path)

    monkeypatch.setattr(
        "luxar.demos._support.downloads.download.download_with_checksum", _fake_download
    )

    (path,) = ensure_dataset(
        "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
    )

    assert path.read_bytes() == b"NEWER-generation", "kept a superseded copy anyway"
    assert [p.name for p in find_quarantined_files(dest)] == [dest.name + ".corrupt"]


def test_corrupt_bytes_are_still_quarantined_even_when_irreplaceable(
    fake_repo, monkeypatch
):
    """The relaxation must not become "accept anything when stuck".

    Bytes matching NO recorded digest are corruption, and serving them would be
    worse than failing. This is why the previous digests had to be recorded
    rather than the checksum simply loosened.
    """
    manifest, cache = fake_repo
    entry = manifest["datasets"]["gsplats_toy"]["files"][0]
    entry["superseded_sha256"] = ["a" * 64]  # names something else entirely
    dest = cache / "gsplats_toy" / "toy_ch0.gsplats.zarr.zip"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"corrupted-bytes")
    monkeypatch.setattr(data_fetch, "_DEMOS_DATA_DIR", cache / "does-not-exist")

    with pytest.raises(DatasetUnavailable):
        ensure_dataset(
            "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
        )
    assert [p.name for p in find_quarantined_files(dest)] == [dest.name + ".corrupt"]


def test_using_a_superseded_copy_is_announced(fake_repo, monkeypatch, capsys):
    """Serving out-of-date data silently is how a stale tile gets published."""
    manifest, cache = fake_repo
    entry = manifest["datasets"]["gsplats_toy"]["files"][0]
    old_digest = entry["sha256"]
    dest = cache / "gsplats_toy" / "toy_ch0.gsplats.zarr.zip"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"toy-splat-bytes")
    entry["sha256"] = "9" * 64
    entry["superseded_sha256"] = [old_digest]
    monkeypatch.setattr(data_fetch, "_DEMOS_DATA_DIR", cache / "does-not-exist")

    ensure_dataset("gsplats_toy", manifest=manifest, cache_root=cache, verbose=False)

    out = capsys.readouterr().out
    assert "SUPERSEDED" in out, out
    assert "not corrupt" in out, out


def test_the_superseded_verdict_ranks_below_both_contracts(fake_repo):
    """A digest that is current must never be reported as superseded."""
    payload = data_fetch._DEMOS_DATA_DIR / "gsplats_toy" / "toy_ch0.gsplats.zarr.zip"
    current = _sha256(payload)
    # The same digest named as BOTH current and superseded: current must win.
    assert data_fetch._accepted_contract(payload, current, None, False, [current]) == (
        "local",
        current,
    )
    assert data_fetch._accepted_contract(payload, None, current, False, [current]) == (
        "hosted",
        current,
    )


def test_only_the_most_recent_generation_is_accepted(fake_repo, monkeypatch):
    """The manifest keeps the whole history; acceptance reaches back exactly one.

    The list's effect IS how far back "acceptable" reaches, and the oldest entry
    is the likeliest to be genuinely wrong. Anything older than one generation
    degrades to a build failure — loud and recoverable — rather than to a
    silently stale artifact.
    """
    manifest, cache = fake_repo
    entry = manifest["datasets"]["gsplats_toy"]["files"][0]
    ancient = entry["sha256"]  # what the cache actually holds
    dest = cache / "gsplats_toy" / "toy_ch0.gsplats.zarr.zip"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"toy-splat-bytes")
    entry["sha256"] = "9" * 64
    # Two generations of history: the cached bytes are the OLDER one.
    entry["superseded_sha256"] = [ancient, "8" * 64]
    monkeypatch.setattr(data_fetch, "_DEMOS_DATA_DIR", cache / "does-not-exist")

    with pytest.raises(DatasetUnavailable):
        ensure_dataset(
            "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
        )

    # Reachable again the moment that digest becomes the most recent entry.
    entry["superseded_sha256"] = ["8" * 64, ancient]
    dest.write_bytes(b"toy-splat-bytes")
    (path,) = ensure_dataset(
        "gsplats_toy", manifest=manifest, cache_root=cache, verbose=False
    )
    assert path.read_bytes() == b"toy-splat-bytes"


@pytest.mark.skipif(not GEN_SCRIPT.exists(), reason=_NO_SCRIPT)
def test_a_changed_pin_records_the_outgoing_digest(tmp_path, monkeypatch):
    """The generator must capture history at the moment it is still knowable.

    When a visible in-repo pin changes, the outgoing digest is what every
    existing cache holds. If regeneration does not record it while those bytes
    are available, the information is gone and a re-pin can strand every cache.
    """
    mod = _load_generator()
    data = tmp_path / "data" / "gsplats_kidney"
    data.mkdir(parents=True)
    (data / "kidney_ch0.gsplats.zarr.zip").write_bytes(b"NEW-generation")
    monkeypatch.setattr(mod, "DATA_DIR", tmp_path / "data")

    outgoing = hashlib.sha256(b"OLD-generation").hexdigest()
    committed = {
        "datasets": {
            "gsplats_kidney": {
                "files": [
                    {
                        "name": "kidney_ch0.gsplats.zarr.zip",
                        "sha256": outgoing,
                        "bytes": len(b"OLD-generation"),
                    }
                ]
            }
        }
    }
    entry = mod.build(committed, prune=False)["datasets"]["gsplats_kidney"]["files"][0]

    assert entry["sha256"] == hashlib.sha256(b"NEW-generation").hexdigest()
    assert entry["superseded_sha256"] == [outgoing]

    # Idempotent: regenerating again must not duplicate the entry.
    again = mod.build(
        {"datasets": {"gsplats_kidney": {"files": [entry]}}}, prune=False
    )["datasets"]["gsplats_kidney"]["files"][0]
    assert again["superseded_sha256"] == [outgoing], "history duplicated on re-run"


@pytest.mark.skipif(not GEN_SCRIPT.exists(), reason=_NO_SCRIPT)
def test_a_reverted_pin_moves_the_outgoing_digest_to_the_end(tmp_path, monkeypatch):
    """The newest superseded generation must remain the final history entry."""
    mod = _load_generator()
    data = tmp_path / "data" / "gsplats_kidney"
    data.mkdir(parents=True)
    payload = data / "kidney_ch0.gsplats.zarr.zip"
    monkeypatch.setattr(mod, "DATA_DIR", tmp_path / "data")

    def _regenerate(previous, generation):
        payload.write_bytes(generation)
        return mod.build(previous, prune=False)["datasets"]["gsplats_kidney"]["files"][
            0
        ]

    digest_a = hashlib.sha256(b"A").hexdigest()
    digest_b = hashlib.sha256(b"B").hexdigest()
    initial = {
        "datasets": {
            "gsplats_kidney": {
                "files": [
                    {
                        "name": payload.name,
                        "sha256": digest_a,
                        "bytes": 1,
                    }
                ]
            }
        }
    }

    pin_b = _regenerate(initial, b"B")
    pin_a = _regenerate({"datasets": {"gsplats_kidney": {"files": [pin_b]}}}, b"A")
    pin_c = _regenerate({"datasets": {"gsplats_kidney": {"files": [pin_a]}}}, b"C")

    assert pin_b["superseded_sha256"] == [digest_a]
    assert pin_a["superseded_sha256"] == [digest_a, digest_b]
    assert pin_c["superseded_sha256"] == [digest_b, digest_a]


@pytest.mark.skipif(not GEN_SCRIPT.exists(), reason=_NO_SCRIPT)
def test_a_hosted_only_repin_history_must_be_authored_before_regeneration(
    tmp_path, monkeypatch
):
    """Without bytes on disk, regeneration cannot discover the outgoing pin."""
    mod = _load_generator()
    monkeypatch.setattr(mod, "DATA_DIR", tmp_path / "data")
    repinned = {
        "datasets": {
            "gsplats_3d_drosophila_gastrulation": {
                "files": [
                    {
                        "name": "droso_gastrulation.gsplats.zarr.zip",
                        "sha256": "b" * 64,
                        "bytes": 2,
                    }
                ]
            }
        }
    }

    entry = mod.build(repinned, prune=False)["datasets"][
        "gsplats_3d_drosophila_gastrulation"
    ]["files"][0]

    assert (
        entry == repinned["datasets"]["gsplats_3d_drosophila_gastrulation"]["files"][0]
    )
    assert "superseded_sha256" not in entry
