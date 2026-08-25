"""The generated record text must not claim more than the archives support.

A Zenodo record is the one place where a wrong number is *published* rather than
merely wrong, so the generator derives every figure from the manifest and the
archives' own stamps. These tests cover the ways that can go wrong quietly: an
unreadable or non-splat file being described as though it were a fit, a
publication flag leaking into the text, and a dataset silently missing from its
own record.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest

#: .../packages/luxar/src/luxar/demos/tests/this_file.py -> repo root is 6 up.
_SCRIPT = Path(__file__).resolve().parents[6] / "scripts" / "gen_zenodo_records.py"


def _load_module() -> Any:
    """Import the script by path; `scripts/` is not an importable package."""
    # Deliberately an error, not a skip: a wrong path here silently turned every
    # test in this file into a pass once already.
    assert _SCRIPT.exists(), (
        f"generator not found at {_SCRIPT} — fix the path rather than skipping, "
        "or these tests all pass while checking nothing"
    )
    spec = importlib.util.spec_from_file_location("gen_zenodo_records", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["gen_zenodo_records"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen() -> Any:
    return _load_module()


@pytest.fixture(scope="module")
def manifest(gen: Any) -> dict[str, Any]:
    return json.loads(gen.MANIFEST.read_text())


def _write_frame(
    path: Path,
    *,
    n_splats: int = 100,
    psnr: float = 40.0,
    source_bytes: int = 1_000_000,
    gsplats: bool = True,
) -> None:
    """One single-store `.gsplats.zarr.zip`, as a bundle's frames are."""
    root = f"{path.name.split('.')[0]}.gsplats.zarr"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{root}/.zgroup", json.dumps({"zarr_format": 2}))
        zf.writestr(
            f"{root}/.zattrs",
            json.dumps(
                {
                    "format_type": "gsplats_zarr" if gsplats else "luxar_zarr",
                    "n_splats": n_splats,
                }
            ),
        )
        zf.writestr(f"{root}/fitting/.zgroup", json.dumps({"zarr_format": 2}))
        zf.writestr(
            f"{root}/fitting/.zattrs",
            json.dumps({"psnr_db": psnr, "source_bytes": source_bytes}),
        )


def _write_bundle(path: Path, frames: list[dict[str, Any]], tmp_path: Path) -> None:
    """An outer zip of per-frame `.gsplats.zarr.zip` members, as the timelapses are."""
    members = []
    for index, kwargs in enumerate(frames):
        member = tmp_path / f"frame{index:04d}.gsplats.zarr.zip"
        _write_frame(member, **kwargs)
        members.append(member)
    with zipfile.ZipFile(path, "w") as zf:
        for member in members:
            zf.write(member, member.name)


def test_every_record_renders(gen: Any, manifest: dict[str, Any]) -> None:
    for key in manifest["records"]:
        text = gen.render_record(key, manifest)
        assert manifest["records"][key]["title"] in text
        assert manifest["records"][key]["zenodo_doi"] in text


def test_every_hosted_dataset_appears_in_its_record(
    gen: Any, manifest: dict[str, Any]
) -> None:
    """A dataset missing from its record would simply never be described."""
    rendered = {key: gen.render_record(key, manifest) for key in manifest["records"]}
    for name, entry in manifest["datasets"].items():
        if entry.get("bucket") != "zenodo":
            continue
        record = entry.get("record")
        assert record in rendered, f"{name} names record {record!r}"
        assert f"`{name}`" in rendered[record], (
            f"{name} is hosted under {record!r} but does not appear in its text"
        )


def test_a_draft_record_says_so(gen: Any, manifest: dict[str, Any]) -> None:
    """While `published` is false the text must carry the draft marker."""
    for key, record in manifest["records"].items():
        text = gen.render_record(key, manifest)
        if record.get("published"):
            assert "DRAFT" not in text
        else:
            assert "DRAFT" in text, f"{key} is unpublished but the text does not say so"


def test_the_text_never_contains_a_zenodo_file_url(
    gen: Any, manifest: dict[str, Any]
) -> None:
    """A file URL into an unpublished record 404s for every reader."""
    for key, record in manifest["records"].items():
        if record.get("published"):
            continue
        text = gen.render_record(key, manifest)
        assert f"zenodo.org/records/{record['zenodo_record']}/files" not in text


class TestNonSplatFilesAreNotDescribedAsFits:
    """The bug this class exists for: an `.npz` is also a zip.

    Opened blindly it parses, every field comes back defaulted, and the record
    ends up asserting "single level" and an absent PSNR for a file that is not a
    splat fit at all.
    """

    def test_an_npz_is_not_read_as_a_splat_store(
        self, gen: Any, tmp_path: Path
    ) -> None:
        import numpy as np

        path = tmp_path / "labels.npz"
        np.savez(path, labels=np.zeros((4, 4), np.uint8))
        assert gen._read_archive(path) is None

    def test_a_zarr_store_that_is_not_gsplats_is_rejected(
        self, gen: Any, tmp_path: Path
    ) -> None:
        path = tmp_path / "points.luxar.zarr.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("points.luxar.zarr/.zgroup", json.dumps({"zarr_format": 2}))
            zf.writestr(
                "points.luxar.zarr/.zattrs",
                json.dumps({"format_type": "luxar_zarr", "type": "points"}),
            )
        assert gen._read_archive(path) is None

    def test_a_gsplats_store_is_accepted(self, gen: Any, tmp_path: Path) -> None:
        """The negative tests above would pass even if nothing were readable."""
        path = tmp_path / "x.gsplats.zarr.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("x.gsplats.zarr/.zgroup", json.dumps({"zarr_format": 2}))
            zf.writestr(
                "x.gsplats.zarr/.zattrs",
                json.dumps(
                    {
                        "format_type": "gsplats_zarr",
                        "n_splats": 1234,
                        "n_additive_sublods": 4,
                    }
                ),
            )
            zf.writestr(
                "x.gsplats.zarr/fitting/.zgroup", json.dumps({"zarr_format": 2})
            )
            zf.writestr(
                "x.gsplats.zarr/fitting/.zattrs",
                json.dumps({"psnr_db": 41.3, "foreground_psnr_db": 28.2}),
            )
        info = gen._read_archive(path)
        assert info is not None
        assert info["n_splats"] == 1234
        assert info["psnr_db"] == pytest.approx(41.3)
        assert info["foreground_psnr_db"] == pytest.approx(28.2)
        assert "4 steps" in info["topology"]

    def test_a_corrupt_file_is_absent_not_fatal(self, gen: Any, tmp_path: Path) -> None:
        path = tmp_path / "truncated.gsplats.zarr.zip"
        path.write_bytes(b"not a zip at all")
        assert gen._read_archive(path) is None

    def test_unparseable_attrs_are_absent_not_fatal(
        self, gen: Any, tmp_path: Path
    ) -> None:
        """A readable zip whose metadata is not JSON must not raise mid-record."""
        path = tmp_path / "garbled.gsplats.zarr.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("garbled.gsplats.zarr/.zgroup", "{")
            zf.writestr("garbled.gsplats.zarr/.zattrs", "not json at all")
        assert gen._read_archive(path) is None


class TestABundleIsDescribedWhole:
    """A timelapse bundle is a zip of per-frame zips, and the record describes it.

    Every column is a statement about the ARCHIVE, so reading only the first
    frame published a 400-frame bundle's splat count 349-fold too low and quoted
    its best frame's PSNR as the bundle's.
    """

    def test_the_splat_count_covers_every_frame(self, gen: Any, tmp_path: Path) -> None:
        path = tmp_path / "movie.gsplats.zarr.zip"
        _write_bundle(
            path,
            [{"n_splats": 10}, {"n_splats": 20}, {"n_splats": 30}],
            tmp_path,
        )
        info = gen._read_archive(path)
        assert info is not None
        assert info["frames"] == 3
        assert info["n_splats"] == 60, "the first frame's count is not the archive's"

    def test_the_source_bytes_cover_every_frame(self, gen: Any, tmp_path: Path) -> None:
        """`vs raw voxels` divides these by the WHOLE bundle's stored size."""
        path = tmp_path / "movie.gsplats.zarr.zip"
        _write_bundle(path, [{"source_bytes": 1000}] * 4, tmp_path)
        assert gen._read_archive(path)["source_bytes"] == 4000

    def test_quality_is_reported_as_the_range_across_frames(
        self, gen: Any, tmp_path: Path
    ) -> None:
        path = tmp_path / "movie.gsplats.zarr.zip"
        _write_bundle(path, [{"psnr": 31.2}, {"psnr": 54.9}, {"psnr": 40.0}], tmp_path)
        assert gen._db(gen._read_archive(path)["psnr_db"]) == "31.2–54.9"

    def test_quality_range_survives_the_committed_sidecar(
        self, gen: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        archive = tmp_path / "ds" / "movie.gsplats.zarr.zip"
        archive.parent.mkdir()
        _write_bundle(
            archive,
            [{"psnr": 31.2}, {"psnr": 54.9}, {"psnr": 40.0}],
            tmp_path,
        )
        monkeypatch.setattr(gen, "CHARACTERISTICS", tmp_path / "chars.json")
        manifest = _fake_manifest(
            [_entry("movie.gsplats.zarr.zip", gen._sha256_of(archive))]
        )

        gen.refresh_characteristics(manifest, tmp_path)
        chars = gen.load_characteristics()
        (row,) = gen._dataset_rows("ds", manifest["datasets"]["ds"], chars)

        assert row["file"] == "movie.gsplats.zarr.zip (3 frames)"
        assert row["psnr"] == "31.2–54.9"

    def test_one_value_when_every_frame_agrees(self, gen: Any, tmp_path: Path) -> None:
        path = tmp_path / "movie.gsplats.zarr.zip"
        _write_bundle(path, [{"psnr": 40.0}] * 3, tmp_path)
        assert gen._db(gen._read_archive(path)["psnr_db"]) == "40.0"

    def test_an_unreadable_frame_makes_the_totals_absent(
        self, gen: Any, tmp_path: Path
    ) -> None:
        """A partial sum published as a total is the falsehood, not a lesser one."""
        path = tmp_path / "movie.gsplats.zarr.zip"
        _write_bundle(path, [{"n_splats": 10}, {"n_splats": 20}], tmp_path)
        broken = tmp_path / "frame0002.gsplats.zarr.zip"
        broken.write_bytes(b"truncated")
        with zipfile.ZipFile(path, "a") as zf:
            zf.write(broken, broken.name)
        info = gen._read_archive(path)
        assert info is not None, "two readable frames still make this a splat bundle"
        assert info["frames"] == 3
        assert info["n_splats"] is None
        assert info["psnr_db"] is None


class TestFiguresAreAbsentRatherThanInvented:
    def test_a_missing_number_renders_as_absent(self, gen: Any) -> None:
        assert gen._db(None) == gen._ABSENT
        assert gen._db("41.3") == gen._ABSENT, "a string must not pass as a measurement"
        assert gen._db(float("nan")) == gen._ABSENT
        assert gen._db(float("inf")) == gen._ABSENT, (
            "an infinite PSNR means an exact fit on a degenerate case, not a score"
        )
        assert gen._db(41.34) == "41.3"

    def test_a_ratio_needs_both_sides(self, gen: Any) -> None:
        assert gen._ratio(None, 100) == gen._ABSENT
        assert gen._ratio(100, None) == gen._ABSENT
        assert gen._ratio(100, 0) == gen._ABSENT, "no dividing by an empty archive"
        assert gen._ratio("100", 10) == gen._ABSENT
        assert gen._ratio(100, "10") == gen._ABSENT
        assert gen._ratio(35_000_000, 100_000) == "350:1"

    def test_an_incomparable_acquisition_states_the_reason(self, gen: Any) -> None:
        entry = {
            "acquisition": {
                "description": "377 colour photographs",
                "comparable": False,
                "reason": "the fit converts colour to greyscale",
            }
        }
        line = gen._acquisition_line(entry, 1_000_000)
        assert "colour to greyscale" in line
        assert ":1" not in line, "an incomparable source must not be given a ratio"

    def test_a_comparable_acquisition_quotes_the_download_ratio(self, gen: Any) -> None:
        entry = {
            "acquisition": {
                "description": "the kidney sample",
                "comparable": True,
                "stored_bytes": 21_000_000,
            }
        }
        line = gen._acquisition_line(entry, 2_100_000)
        assert "10:1" in line
        assert "as downloaded" in line

    def test_a_comparable_acquisition_awaiting_measurement_quotes_nothing(
        self, gen: Any
    ) -> None:
        entry = {
            "acquisition": {"description": "the kidney sample", "comparable": True}
        }
        line = gen._acquisition_line(entry, 2_100_000)
        assert ":1" not in line
        assert "kidney sample" in line

    def test_variants_get_no_summed_ratio(self, gen: Any) -> None:
        """Size variants are alternative downloads; summing them double-counts."""
        entry = {
            "acquisition": {
                "description": "the timelapse",
                "comparable": True,
                "stored_bytes": 21_000_000,
            }
        }
        line = gen._acquisition_line(entry, None)
        assert ":1" not in line
        assert "total" not in line, "there is no single total to divide by"
        assert "20.0 MiB stored" in line


class TestSizeVariantsAreDescribed:
    """h2afva lists its archives under `variants`, not a flat `files`.

    Reading only `files` left the record whose title is the timelapse saying "No
    files uploaded yet", with its whole table and its work-list rows missing.
    """

    ENTRY = {
        "bucket": "zenodo",
        "record": "r",
        "dir": "movie",
        "license": "cc-by-4.0",
        "variants": {
            "light": {"default": True, "note": "the lighter fit", "files": []},
            "full": {
                "note": "every timepoint",
                "files": [{"name": "movie_full.gsplats.zarr.zip", "bytes": 4_000_000}],
            },
        },
    }

    def test_a_variant_file_becomes_a_row(self, gen: Any) -> None:
        rows = gen._dataset_rows("movie", self.ENTRY)
        assert [row["file"] for row in rows] == ["full/movie_full.gsplats.zarr.zip"]

    def test_the_record_lists_the_variant_and_its_note(self, gen: Any) -> None:
        manifest = {
            "records": {"r": {"title": "T", "license": "cc-by-4.0", "zenodo_doi": "d"}},
            "datasets": {"movie": self.ENTRY},
        }
        text = gen.render_record("r", manifest)
        assert "No files uploaded yet" not in text
        assert "movie_full.gsplats.zarr.zip" in text
        assert "the lighter fit" in text and "every timepoint" in text

    def test_a_declared_variant_is_not_reported_as_no_files(self, gen: Any) -> None:
        problems, _ = gen._gaps({"datasets": {"movie": self.ENTRY}})
        assert "movie: no files uploaded" not in problems

    def test_the_cache_namespaces_by_dataset_name_and_variant(
        self, gen: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`ensure_dataset` caches under <name>/<variant>/, not the manifest dir."""
        monkeypatch.setattr(gen, "DATA_DIR", tmp_path / "repo")
        monkeypatch.setattr(gen, "CACHE_DIR", tmp_path / "cache")
        cached = (
            tmp_path / "cache" / "movie_ds" / "full" / "movie_full.gsplats.zarr.zip"
        )
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"")
        found = gen._locate(
            "movie_ds", self.ENTRY, "full", "movie_full.gsplats.zarr.zip"
        )
        assert found == cached


class TestAnUnreadableFitIsNotCalledANonFit:
    """A `.gsplats.zarr.zip` that is not on this machine is unknown, not "not a fit".

    Without the distinction a fresh clone (no `git lfs pull`, no demo cache)
    renders every splat dataset as the plain file list reserved for tabular data,
    and `--check` reports a clean bill of health having read nothing.
    """

    ENTRY = {
        "bucket": "zenodo",
        "record": "r",
        "dir": "d",
        "license": "cc-by-4.0",
        "files": [{"name": "absent.gsplats.zarr.zip", "bytes": 4_000_000}],
    }

    @pytest.fixture(autouse=True)
    def _empty_roots(self, gen: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(gen, "DATA_DIR", tmp_path / "repo")
        monkeypatch.setattr(gen, "CACHE_DIR", tmp_path / "cache")

    def test_it_still_gets_the_splat_table(self, gen: Any) -> None:
        manifest = {
            "records": {"r": {"title": "T", "license": "cc-by-4.0", "zenodo_doi": "d"}},
            "datasets": {"nope": self.ENTRY},
        }
        text = gen.render_record("r", manifest)
        assert "| Splats |" in text, "a fit with unknown figures is still a fit"

    def test_check_says_it_read_nothing(self, gen: Any) -> None:
        problems, unread = gen._gaps({"datasets": {"nope": self.ENTRY}})
        assert unread == ["nope/absent.gsplats.zarr.zip"]
        assert not [p for p in problems if "absent" in p], (
            "an archive that was never read owes no stamps yet"
        )

    def test_a_real_sidecar_is_still_not_a_fit(self, gen: Any) -> None:
        entry = dict(self.ENTRY, files=[{"name": "labels.npz", "bytes": 12}])
        problems, unread = gen._gaps({"datasets": {"side": entry}})
        assert unread == []
        assert problems == []


# ---------------------------------------------------------------------------
# The committed measurements (`scripts/demo_archive_characteristics.json`)
#
# Reading the archives at render time only worked while the archives were in the
# repo. They are moving to Zenodo, so measurement had to separate from rendering
# — and the migration had already made the old design report absent figures for
# data that has them.
# ---------------------------------------------------------------------------


def _entry(name: str, sha: str, **extra: Any) -> dict[str, Any]:
    return {"name": name, "sha256": sha, "bytes": 1024, **extra}


def _fake_manifest(files: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": {"cc-by": {"license": "cc-by-4.0", "zenodo_record": 1}},
        "datasets": {
            "ds": {"bucket": "zenodo", "record": "cc-by", "dir": "ds", "files": files}
        },
    }


def test_a_committed_measurement_beats_a_local_archive(
    gen: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The record describes the artifact it SERVES, not whatever is on this disk.

    The local copy is routinely a pre-refit generation (or a scratch fit), so
    preferring it is how the descriptions went stale in the first place.
    """
    monkeypatch.setattr(
        gen,
        "load_characteristics",
        lambda: {
            "ds/a.gsplats.zarr.zip": {
                "n_splats": 12345,
                "topology": "single level",
                "psnr_db": 41.5,
                "foreground_psnr_db": 30.25,
                "source_bytes": None,
                "frames": None,
            }
        },
    )
    # Any local read would have to go through _locate; make it impossible.
    monkeypatch.setattr(gen, "_locate", lambda *a, **k: None)

    (row,) = gen._dataset_rows(
        "ds", _fake_manifest([_entry("a.gsplats.zarr.zip", "a" * 64)])["datasets"]["ds"]
    )
    assert row["splats"] == "12,345"
    assert "41.5" in row["psnr"]
    assert "30.2" in row["fg_psnr"] or "30.3" in row["fg_psnr"]


def test_a_staged_measurement_is_not_clobbered_by_a_local_refresh(
    gen: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Equal pinned bytes retain staged provenance across local refreshes."""
    key = "ds/a.gsplats.zarr.zip"
    pinned_sha = "p" * 64
    staged = {
        key: {
            "n_splats": 999,
            "psnr_db": 43.0,
            "foreground_psnr_db": 30.6,
            "topology": "single level",
            "measured_from": "staged",
            "measured_sha256": pinned_sha,
        }
    }
    monkeypatch.setattr(gen, "CHARACTERISTICS", tmp_path / "chars.json")
    monkeypatch.setattr(gen, "load_characteristics", lambda: staged)
    monkeypatch.setattr(gen, "_locate", lambda *a, **k: tmp_path / "local.zip")
    monkeypatch.setattr(gen, "_sha256_of", lambda p: pinned_sha)
    monkeypatch.setattr(
        gen,
        "_read_archive",
        lambda p: {
            "n_splats": 111,
            "psnr_db": 30.0,
            "foreground_psnr_db": None,
            "topology": "single level",
        },
    )

    counts = gen.refresh_characteristics(
        _fake_manifest([_entry("a.gsplats.zarr.zip", pinned_sha)])
    )

    assert counts == (1, 1, 0, 0)
    written = json.loads((tmp_path / "chars.json").read_text())["archives"][key]
    assert written["measured_from"] == "staged", "a local read outranked the staged one"
    assert written["foreground_psnr_db"] == 30.6
    assert written["n_splats"] == 999


def test_an_unpinned_local_measurement_cannot_replace_an_absent_figure(
    gen: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deliberate absence must survive refreshes from superseded bytes."""
    key = "ds/a.gsplats.zarr.zip"
    absent = {
        key: {
            "n_splats": None,
            "psnr_db": None,
            "foreground_psnr_db": None,
            "topology": None,
            "measured_from": None,
            "measured_sha256": None,
        }
    }
    monkeypatch.setattr(gen, "CHARACTERISTICS", tmp_path / "chars.json")
    monkeypatch.setattr(gen, "load_characteristics", lambda: absent)
    monkeypatch.setattr(gen, "_locate", lambda *a, **k: tmp_path / "local.zip")
    monkeypatch.setattr(gen, "_sha256_of", lambda p: "l" * 64)
    monkeypatch.setattr(
        gen,
        "_read_archive",
        lambda p: {"n_splats": 111, "psnr_db": 30.0, "topology": "single level"},
    )
    manifest = _fake_manifest(
        [
            _entry(
                "a.gsplats.zarr.zip",
                "l" * 64,
                hosted_sha256="h" * 64,
            )
        ]
    )

    counts = gen.refresh_characteristics(manifest)

    assert counts == (1, 0, 1, 0)
    written = json.loads((tmp_path / "chars.json").read_text())["archives"][key]
    assert written == absent[key]


def test_a_current_local_measurement_replaces_a_stale_staged_one(
    gen: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Digest validity wins before staged/local provenance precedence."""
    key = "ds/a.gsplats.zarr.zip"
    stale = {
        key: {
            "n_splats": 999,
            "measured_from": "staged",
            "measured_sha256": "s" * 64,
        }
    }
    monkeypatch.setattr(gen, "CHARACTERISTICS", tmp_path / "chars.json")
    monkeypatch.setattr(gen, "load_characteristics", lambda: stale)
    monkeypatch.setattr(gen, "_locate", lambda *a, **k: tmp_path / "local.zip")
    monkeypatch.setattr(gen, "_sha256_of", lambda p: "p" * 64)
    monkeypatch.setattr(gen, "_read_archive", lambda p: {"n_splats": 111})

    counts = gen.refresh_characteristics(
        _fake_manifest([_entry("a.gsplats.zarr.zip", "p" * 64)])
    )

    assert counts == (1, 0, 0, 0)
    written = json.loads((tmp_path / "chars.json").read_text())["archives"][key]
    assert written["measured_sha256"] == "p" * 64
    assert written["n_splats"] == 111


def test_refresh_preserves_entries_whose_archive_is_absent(
    gen: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial checkout is the normal case now, so it must not delete figures.

    Same rule ``gen_data_manifest`` follows for file lists: what you cannot see,
    you do not get to erase.
    """
    key = "ds/a.gsplats.zarr.zip"
    monkeypatch.setattr(gen, "CHARACTERISTICS", tmp_path / "chars.json")
    monkeypatch.setattr(
        gen, "load_characteristics", lambda: {key: {"n_splats": 7, "psnr_db": 1.0}}
    )
    monkeypatch.setattr(gen, "_locate", lambda *a, **k: None)

    read, retained, rejected, preserved = gen.refresh_characteristics(
        _fake_manifest([_entry("a.gsplats.zarr.zip", "a" * 64)])
    )

    assert (read, retained, rejected, preserved) == (0, 0, 0, 1)
    assert (
        json.loads((tmp_path / "chars.json").read_text())["archives"][key]["n_splats"]
        == 7
    )


def test_a_prefix_matching_cache_path_is_not_labelled_staged(
    gen: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged_root = tmp_path / "lux"
    cache_archive = tmp_path / "luxar" / "ds" / "fit.gsplats.zarr.zip"
    cache_archive.parent.mkdir(parents=True)
    cache_archive.write_bytes(b"archive")
    monkeypatch.setattr(gen, "CHARACTERISTICS", tmp_path / "chars.json")
    monkeypatch.setattr(gen, "_locate", lambda *args: cache_archive)
    monkeypatch.setattr(gen, "_read_archive", lambda path: {"n_splats": 7})
    monkeypatch.setattr(gen, "_sha256_of", lambda path: "a" * 64)

    gen.refresh_characteristics(
        _fake_manifest([_entry(cache_archive.name, "a" * 64)]), staged_root
    )

    entry = gen.load_characteristics()[f"ds/{cache_archive.name}"]
    assert entry["measured_from"] == "cache"


def test_refresh_reports_and_discards_an_unpinned_read_without_a_fallback(
    gen: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = "ds/a.gsplats.zarr.zip"
    monkeypatch.setattr(gen, "CHARACTERISTICS", tmp_path / "chars.json")
    monkeypatch.setattr(gen, "load_characteristics", lambda: {})
    monkeypatch.setattr(gen, "_locate", lambda *a, **k: tmp_path / "local.zip")
    monkeypatch.setattr(gen, "_sha256_of", lambda p: "l" * 64)
    monkeypatch.setattr(gen, "_read_archive", lambda p: {"n_splats": 111})

    counts = gen.refresh_characteristics(
        _fake_manifest([_entry("a.gsplats.zarr.zip", "p" * 64)])
    )

    assert counts == (1, 0, 1, 0)
    assert key not in json.loads((tmp_path / "chars.json").read_text())["archives"]


def test_a_partial_sidecar_entry_renders_absent_fields(gen: Any) -> None:
    manifest = _fake_manifest([_entry("a.gsplats.zarr.zip", "a" * 64)])

    (row,) = gen._dataset_rows(
        "ds",
        manifest["datasets"]["ds"],
        {"ds/a.gsplats.zarr.zip": {"n_splats": 7, "source_bytes": "2048"}},
    )

    assert row["splats"] == "7"
    assert row["topology"] == gen._ABSENT
    assert row["psnr"] == gen._ABSENT
    assert row["vs_raw"] == gen._ABSENT


def test_load_characteristics_skips_non_object_entries(
    gen: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "chars.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "archives": {"good": {"n_splats": 7}, "bad": "not an object"},
            }
        )
    )
    monkeypatch.setattr(gen, "CHARACTERISTICS", path)

    assert gen.load_characteristics() == {"good": {"n_splats": 7}}


def test_load_characteristics_rejects_an_unknown_schema(
    gen: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "chars.json"
    path.write_text(json.dumps({"schema_version": 2, "archives": {}}))
    monkeypatch.setattr(gen, "CHARACTERISTICS", path)

    with pytest.raises(ValueError, match="unsupported characteristics schema"):
        gen.load_characteristics()


def test_a_measurement_from_superseded_bytes_is_reported(
    gen: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale figure is worse than an absent one: absent prints as a dash.

    This is the guard the hand-written descriptions never had — a refit changes
    the artifact without touching the measurements taken from the old one.
    """
    monkeypatch.setattr(
        gen,
        "load_characteristics",
        lambda: {"ds/a.gsplats.zarr.zip": {"measured_sha256": "b" * 64}},
    )
    stale = gen._stale_characteristics(
        _fake_manifest([_entry("a.gsplats.zarr.zip", "a" * 64)])
    )
    assert len(stale) == 1
    assert "b" * 12 in stale[0] and "a" * 12 in stale[0]


def test_check_fails_when_a_measurement_is_stale(
    gen: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = {
        "n_splats": 7,
        "topology": "single level",
        "psnr_db": 40.0,
        "foreground_psnr_db": 30.0,
        "source_bytes": 2048,
        "measured_sha256": "b" * 64,
    }
    path = tmp_path / "chars.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "archives": {"ds/a.gsplats.zarr.zip": archive},
            }
        )
    )
    monkeypatch.setattr(gen, "CHARACTERISTICS", path)
    manifest = _fake_manifest([_entry("a.gsplats.zarr.zip", "a" * 64)])

    assert gen._run_check(manifest) == 1


def test_the_hosted_digest_is_what_a_measurement_is_judged_against(gen: Any) -> None:
    """Once the two contracts diverge, the record's copy is the relevant one.

    `sha256` describes what the repo ships; `hosted_sha256` what the record
    serves. A figure printed in a record must be judged against the latter.
    """
    assert gen._pinned_digest({"sha256": "a" * 64}) == "a" * 64
    assert (
        gen._pinned_digest({"sha256": "a" * 64, "hosted_sha256": "h" * 64}) == "h" * 64
    )


def test_no_measurement_is_stale_against_its_own_digest(
    gen: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The quiet case must stay quiet, or the report is noise."""
    monkeypatch.setattr(
        gen,
        "load_characteristics",
        lambda: {"ds/a.gsplats.zarr.zip": {"measured_sha256": "a" * 64}},
    )
    assert (
        gen._stale_characteristics(
            _fake_manifest([_entry("a.gsplats.zarr.zip", "a" * 64)])
        )
        == []
    )


def test_committed_measurements_match_the_hosted_manifest_pins(gen: Any) -> None:
    manifest = json.loads(gen.MANIFEST.read_text())
    assert gen._stale_characteristics(manifest) == []


def test_every_committed_measurement_names_a_manifest_archive(gen: Any) -> None:
    manifest = json.loads(gen.MANIFEST.read_text())
    manifest_keys = {
        gen._char_key(dataset, variant, spec["name"])
        for dataset, entry in manifest["datasets"].items()
        if entry.get("bucket") == "zenodo"
        for variant, spec in gen._files_of(entry)
    }
    assert set(gen.load_characteristics()) <= manifest_keys


def test_a_record_quotes_the_size_a_reader_will_download(gen: Any) -> None:
    """`bytes` is the repo's copy; `hosted_bytes` is the record's.

    They diverge for every refitted dataset, so a record's own table quoting the
    in-repo size tells a reader the wrong download size — 36.4 MiB against an
    actual 80.6 MiB for cmu1_ch0.
    """
    assert gen.hosted_size({"bytes": 38201205}) == 38201205
    assert gen.hosted_size({"bytes": 38201205, "hosted_bytes": 84492218}) == 84492218


def test_the_finest_count_is_not_a_naive_sum(
    gen: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tree's groups combine three different ways; summing them is far off.

    `part_N` are disjoint tiles (SUM), `child_N` are substitutive levels that
    REPLACE each other (MAX), and `additive_N` sum to their own parent. Summing
    everything gives 2,574,354 for ct_atlas against a true 647,083.
    """
    import zipfile

    # A partition of two parts, each a 2-level lod. The first finest level has no
    # own stamp, so it must sum its additive chunks; the second has an own stamp,
    # which must win over its deliberately different additive sum. Correct:
    # (20 + 30) + 500 = 550, not 50 + 501.
    groups = {
        "part_0",
        "part_0/child_0",
        "part_0/child_1",
        "part_0/child_1/additive_0",
        "part_0/child_1/additive_1",
        "part_1",
        "part_1/child_0",
        "part_1/child_1",
        "part_1/child_1/additive_0",
        "part_1/child_1/additive_1",
    }
    counts = {
        "": {"kind": "partition"},
        "part_0": {"kind": "lod"},
        "part_0/child_0": {"n_splats": 5},
        "part_0/child_1": {},
        "part_0/child_1/additive_0": {"n_splats": 20},
        "part_0/child_1/additive_1": {"n_splats": 30},
        "part_1": {"kind": "lod"},
        "part_1/child_0": {"n_splats": 7},
        "part_1/child_1": {"n_splats": 500},
        "part_1/child_1/additive_0": {"n_splats": 200},
        "part_1/child_1/additive_1": {"n_splats": 301},
    }
    monkey = lambda zf, root, node="": counts.get(node.rstrip("/"), {})  # noqa: E731
    monkeypatch.setattr(gen, "_attrs", monkey)
    total = gen._finest_elements(zipfile.ZipFile.__new__(zipfile.ZipFile), "r/", groups)
    assert total == 550, "expected sum-over-parts of max-over-levels"


def test_a_partition_count_is_absent_if_any_part_is_unreadable(
    gen: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    counts = {"": {"kind": "partition"}, "part_0": {"n_splats": 10}, "part_1": {}}
    monkeypatch.setattr(
        gen, "_attrs", lambda zf, root, node="": counts.get(node.rstrip("/"), {})
    )

    total = gen._finest_elements(
        zipfile.ZipFile.__new__(zipfile.ZipFile), "r/", {"part_0", "part_1"}
    )

    assert total is None, "a partial sum must not be published as a total"


def test_an_additive_count_is_absent_if_any_chunk_is_unreadable(
    gen: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    counts = {"": {}, "additive_0": {"n_splats": 10}, "additive_1": {}}
    monkeypatch.setattr(
        gen, "_attrs", lambda zf, root, node="": counts.get(node.rstrip("/"), {})
    )

    total = gen._finest_elements(
        zipfile.ZipFile.__new__(zipfile.ZipFile),
        "r/",
        {"additive_0", "additive_1"},
    )

    assert total is None, "a partial sum must not be published as a total"
