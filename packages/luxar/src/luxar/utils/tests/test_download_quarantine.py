"""Tests for the shared quarantined-cache (``.corrupt``) detection helpers.

The demo cache helpers quarantine a truncated/unreadable artifact by renaming it
to ``<name>.corrupt``. A quarantined file is never reused, so the next run
silently restarts a potentially multi-gigabyte download. These helpers make the
download chokepoint say so, with the path and size, before spending the
bandwidth.
"""

from __future__ import annotations

from pathlib import Path

from luxar.utils.download import (
    QUARANTINE_SUFFIX,
    _format_bytes,
    find_quarantined_files,
    format_quarantine_notice,
    robust_download,
    warn_if_quarantined,
)


class TestFormatBytes:
    def test_scales_units(self) -> None:
        assert _format_bytes(512) == "512 B"
        assert _format_bytes(2048) == "2.00 KB"
        assert _format_bytes(3 * 1024**2) == "3.00 MB"
        assert _format_bytes(1644036096) == "1.53 GB"


class TestFindQuarantinedFiles:
    def test_clean_cache_reports_nothing(self, tmp_path: Path) -> None:
        target = tmp_path / "embeddings.npy"
        target.write_bytes(b"ok")
        assert find_quarantined_files(target) == []
        assert find_quarantined_files(tmp_path) == []

    def test_appended_suffix_convention(self, tmp_path: Path) -> None:
        """``foo.npy`` → ``foo.npy.corrupt`` (the download/np.load convention)."""
        corrupt = tmp_path / f"embeddings.npy{QUARANTINE_SUFFIX}"
        corrupt.write_bytes(b"x" * 10)
        assert find_quarantined_files(tmp_path / "embeddings.npy") == [corrupt]

    def test_with_suffix_convention(self, tmp_path: Path) -> None:
        """``foo.pkl`` → ``foo.corrupt`` (the ``cache_computed`` convention)."""
        corrupt = tmp_path / f"cache_v1{QUARANTINE_SUFFIX}"
        corrupt.write_bytes(b"x" * 10)
        assert find_quarantined_files(tmp_path / "cache_v1.pkl") == [corrupt]

    def test_directory_target_lists_all_and_dedupes(self, tmp_path: Path) -> None:
        first = tmp_path / f"a.npy{QUARANTINE_SUFFIX}"
        second = tmp_path / f"b.zip{QUARANTINE_SUFFIX}"
        first.write_bytes(b"a")
        second.write_bytes(b"b")
        (tmp_path / "fine.npy").write_bytes(b"ok")
        assert find_quarantined_files(tmp_path) == [first, second]

    def test_directories_named_corrupt_are_ignored(self, tmp_path: Path) -> None:
        (tmp_path / f"a.zarr{QUARANTINE_SUFFIX}").mkdir()
        assert find_quarantined_files(tmp_path) == []


class TestFormatQuarantineNotice:
    def test_empty_input_yields_empty_string(self) -> None:
        assert format_quarantine_notice([]) == ""

    def test_notice_names_path_size_and_action(self, tmp_path: Path) -> None:
        corrupt = tmp_path / f"embeddings.npy{QUARANTINE_SUFFIX}"
        corrupt.write_bytes(b"x" * 4096)

        notice = format_quarantine_notice([corrupt], action="do the thing")

        assert str(corrupt) in notice
        assert "4.00 KB" in notice
        assert "QUARANTINED" in notice
        assert "do the thing" in notice


class TestWarnIfQuarantined:
    def test_returns_and_prints_for_quarantined_file(self, tmp_path, capsys) -> None:
        corrupt = tmp_path / f"embeddings.npy{QUARANTINE_SUFFIX}"
        corrupt.write_bytes(b"x" * 1024)

        found = warn_if_quarantined(tmp_path / "embeddings.npy")

        assert found == [corrupt]
        out = capsys.readouterr().out
        assert corrupt.name in out
        assert "1.00 KB" in out

    def test_silent_when_clean(self, tmp_path, capsys) -> None:
        assert warn_if_quarantined(tmp_path / "embeddings.npy") == []
        assert capsys.readouterr().out == ""

    def test_verbose_false_suppresses_output(self, tmp_path, capsys) -> None:
        corrupt = tmp_path / f"embeddings.npy{QUARANTINE_SUFFIX}"
        corrupt.write_bytes(b"x")

        assert warn_if_quarantined(tmp_path / "embeddings.npy", verbose=False) == [
            corrupt
        ]
        assert capsys.readouterr().out == ""


class TestRobustDownloadChokepoint:
    """``robust_download`` must announce a quarantined sibling before fetching."""

    def test_warns_before_starting_the_download(self, tmp_path, capsys) -> None:
        dest = tmp_path / "huge.npy"
        corrupt = tmp_path / f"huge.npy{QUARANTINE_SUFFIX}"
        corrupt.write_bytes(b"x" * 2048)

        # No network: an unroutable scheme makes the very first request fail, so
        # the test only observes the pre-download warning.
        try:
            robust_download(
                "http://127.0.0.1:9/never-served",
                dest,
                max_retries=0,
                timeout=1,
            )
        except Exception:  # noqa: BLE001 - the failure itself is not under test
            pass

        out = capsys.readouterr().out
        assert str(corrupt) in out
        assert "2.00 KB" in out
        assert "QUARANTINED" in out
