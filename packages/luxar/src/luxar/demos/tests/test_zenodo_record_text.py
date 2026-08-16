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
