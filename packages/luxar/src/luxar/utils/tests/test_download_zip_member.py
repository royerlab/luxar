"""Tests for HTTP-Range remote-zip single-member extraction.

A stdlib ``http.server`` subclass that honors ``Range`` requests serves real
zip files built with ``zipfile`` (classic, STORED, and forced-Zip64 layouts),
so the extractor is verified byte-for-byte against ``zipfile``'s own
extraction — no network access.
"""

from __future__ import annotations

import http.server
import threading
import zipfile
from pathlib import Path

import numpy as np
import pytest

from luxar.utils.download import download_zip_member


class _RangeHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler + RFC 7233 single-range GET support."""

    def log_message(self, *args: object) -> None:  # keep test output quiet
        pass

    def send_head(self):  # noqa: ANN201 - stdlib override
        path = Path(self.translate_path(self.path))
        range_header = self.headers.get("Range")
        if range_header is None or not path.is_file():
            return super().send_head()

        size = path.stat().st_size
        spec = range_header.replace("bytes=", "").strip()
        start_s, _, end_s = spec.partition("-")
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else size - 1
        end = min(end, size - 1)
        if start > end or start >= size:
            self.send_error(416, "Requested Range Not Satisfiable")
            return None

        f = open(path, "rb")
        f.seek(start)
        self._range_remaining = end - start + 1
        self.send_response(206)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        return _LimitedFile(f, end - start + 1)


class _LimitedFile:
    """File-like that stops after ``limit`` bytes (for copyfile)."""

    def __init__(self, f, limit: int) -> None:  # noqa: ANN001
        self._f = f
        self._remaining = limit

    def read(self, n: int = -1) -> bytes:
        if self._remaining <= 0:
            return b""
        n = self._remaining if n < 0 else min(n, self._remaining)
        data = self._f.read(n)
        self._remaining -= len(data)
        return data

    def close(self) -> None:
        self._f.close()


class _RedirectRangeHTTPHandler(_RangeHTTPHandler):
    """Range handler that 302-redirects ``/dl/<x>`` → ``/<x>`` with
    ``Content-Length: 0`` on the redirect — mimicking GitHub/HF release-asset
    hosts that bounce the canonical URL to a signed CDN. Reproduces the bug
    where an unfollowed HEAD reads the 302's zero length. Both GET and HEAD
    route through ``send_head``, so this covers both verbs.
    """

    def send_head(self):  # noqa: ANN201 - stdlib override
        if self.path.startswith("/dl/"):
            self.send_response(302)
            self.send_header("Location", self.path[3:])  # strip "/dl"
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        return super().send_head()


def _serve(directory: Path, handler_cls):  # noqa: ANN001, ANN202
    handler = lambda *a, **kw: handler_cls(*a, directory=str(directory), **kw)  # noqa: E731
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


@pytest.fixture()
def range_server(tmp_path: Path):
    """Serve tmp_path over HTTP with Range support; yields the base URL."""
    server, thread = _serve(tmp_path, _RangeHTTPHandler)
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


@pytest.fixture()
def redirect_range_server(tmp_path: Path):
    """Serve tmp_path with Range support + a 302-redirecting ``/dl/`` prefix."""
    server, thread = _serve(tmp_path, _RedirectRangeHTTPHandler)
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _make_payloads(seed: int = 3) -> dict[str, bytes]:
    rng = np.random.default_rng(seed)
    return {
        # Compressible (deflate does something) and incompressible payloads.
        "models/garden/point_cloud.ply": b"ply\n" + b"splat " * 50_000,
        "models/train/point_cloud.ply": rng.integers(0, 256, 300_000)
        .astype(np.uint8)
        .tobytes(),
        "readme.txt": b"hello",
    }


def _forge_central_uncompressed_size(
    archive: Path, member: str, forged_size: int
) -> None:
    """Overwrite the *central-directory* uncompressed-size field (offset 24,
    little-endian uint32) of ``member`` with ``forged_size`` — simulating a
    corrupt/malicious archive that under-declares a member that inflates large.
    """
    data = bytearray(archive.read_bytes())
    name = member.encode()
    sig = b"\x50\x4b\x01\x02"  # central-directory file header
    idx = 0
    while True:
        idx = data.find(sig, idx)
        if idx < 0:
            raise AssertionError(f"central-dir record for {member!r} not found")
        name_len = int.from_bytes(data[idx + 28 : idx + 30], "little")
        rec_name = bytes(data[idx + 46 : idx + 46 + name_len])
        if rec_name == name:
            data[idx + 24 : idx + 28] = int(forged_size).to_bytes(4, "little")
            break
        idx += 4
    archive.write_bytes(bytes(data))


def _build_zip(
    path: Path,
    payloads: dict[str, bytes],
    *,
    stored: bool = False,
    force_zip64: bool = False,
) -> None:
    compression = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(path, "w", compression=compression) as zf:
        for name, data in payloads.items():
            with zf.open(name, "w", force_zip64=force_zip64) as f:
                f.write(data)


class TestDownloadZipMember:
    @pytest.mark.parametrize("stored", [False, True], ids=["deflate", "stored"])
    @pytest.mark.parametrize("zip64", [False, True], ids=["classic", "zip64"])
    def test_extracts_byte_exact(
        self, range_server: str, tmp_path: Path, stored: bool, zip64: bool
    ) -> None:
        payloads = _make_payloads()
        archive = tmp_path / "archive.zip"
        _build_zip(archive, payloads, stored=stored, force_zip64=zip64)

        member = "models/garden/point_cloud.ply"
        out = tmp_path / "extracted.ply"
        result = download_zip_member(f"{range_server}/archive.zip", member, out)
        assert result == out
        assert out.read_bytes() == payloads[member]

    def test_second_member_and_expected_size(
        self, range_server: str, tmp_path: Path
    ) -> None:
        payloads = _make_payloads()
        archive = tmp_path / "archive.zip"
        _build_zip(archive, payloads)
        member = "models/train/point_cloud.ply"
        out = tmp_path / "train.ply"
        download_zip_member(
            f"{range_server}/archive.zip",
            member,
            out,
            expected_size=len(payloads[member]),
        )
        assert out.read_bytes() == payloads[member]
        # Second call skips (already complete) without error.
        download_zip_member(
            f"{range_server}/archive.zip",
            member,
            out,
            expected_size=len(payloads[member]),
        )

    def test_missing_member_lists_contents(
        self, range_server: str, tmp_path: Path
    ) -> None:
        _build_zip(tmp_path / "archive.zip", _make_payloads())
        with pytest.raises(KeyError, match="not found"):
            download_zip_member(
                f"{range_server}/archive.zip",
                "models/nope.ply",
                tmp_path / "o.ply",
            )

    def test_rejects_traversal_member_path(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unsafe zip path"):
            download_zip_member(
                "http://127.0.0.1:1/never-contacted.zip",
                "../evil.ply",
                tmp_path / "o.ply",
            )

    def test_expected_size_mismatch_raises(
        self, range_server: str, tmp_path: Path
    ) -> None:
        payloads = _make_payloads()
        _build_zip(tmp_path / "archive.zip", payloads)
        with pytest.raises(ValueError, match="size mismatch"):
            download_zip_member(
                f"{range_server}/archive.zip",
                "readme.txt",
                tmp_path / "o.txt",
                expected_size=999,
            )

    def test_extracts_through_a_redirecting_host(
        self, redirect_range_server: str, tmp_path: Path
    ) -> None:
        """Regression: a HEAD that 302-redirects with Content-Length: 0 must
        be followed (allow_redirects=True), or archive_size reads as 0 and the
        tail range becomes `bytes=0--1` — which strict CDNs reject with 501.
        """
        payloads = _make_payloads()
        archive = tmp_path / "archive.zip"
        _build_zip(archive, payloads)
        member = "models/train/point_cloud.ply"
        out = tmp_path / "via_redirect.ply"
        # /dl/archive.zip 302-redirects to /archive.zip on every hop.
        download_zip_member(f"{redirect_range_server}/dl/archive.zip", member, out)
        assert out.read_bytes() == payloads[member]

    def test_rejects_decompression_bomb(
        self, range_server: str, tmp_path: Path
    ) -> None:
        """A member whose forged central-dir uncompressed size (1024) is far
        smaller than what its DEFLATE stream actually inflates to must be
        rejected mid-stream as a decompression bomb, with no ``.part`` staging
        file (or output) left behind.
        """
        member = "bomb.bin"
        # Highly compressible: tiny DEFLATE stream, ~5 MB inflated.
        payload = b"A" * 5_000_000
        archive = tmp_path / "bomb.zip"
        _build_zip(archive, {member: payload})
        _forge_central_uncompressed_size(archive, member, 1024)

        out = tmp_path / "bomb.out"
        with pytest.raises(ValueError, match="decompression bomb|exceeds"):
            download_zip_member(f"{range_server}/bomb.zip", member, out)
        assert not out.exists()
        assert list(tmp_path.glob("*.part")) == []

    def test_max_uncompressed_size_ceiling(
        self, range_server: str, tmp_path: Path
    ) -> None:
        """``max_uncompressed_size`` rejects a member declaring more than the
        ceiling before any streaming; a generous ceiling still extracts fine.
        """
        payloads = _make_payloads()
        _build_zip(tmp_path / "archive.zip", payloads)
        member = "models/train/point_cloud.ply"  # 300_000 bytes
        out = tmp_path / "train.ply"

        with pytest.raises(ValueError, match="ceiling|exceeding"):
            download_zip_member(
                f"{range_server}/archive.zip",
                member,
                out,
                max_uncompressed_size=1024,
            )
        assert not out.exists()
        assert list(tmp_path.glob("*.part")) == []

        # A generous ceiling leaves normal extraction unaffected.
        download_zip_member(
            f"{range_server}/archive.zip",
            member,
            out,
            max_uncompressed_size=10_000_000,
        )
        assert out.read_bytes() == payloads[member]

    def test_max_uncompressed_size_none_disables_ceiling(
        self, range_server: str, tmp_path: Path
    ) -> None:
        """Passing ``max_uncompressed_size=None`` disables the absolute ceiling:
        a normal member still extracts byte-exact (proving ``None`` is a valid
        disable value that does not trip the ``min``/early-rejection logic). The
        legitimate stream also exercises the compressed-bytes counter without a
        false trigger (cumulative read == comp_size).
        """
        payloads = _make_payloads()
        _build_zip(tmp_path / "archive.zip", payloads)
        member = "models/train/point_cloud.ply"
        out = tmp_path / "train.ply"
        download_zip_member(
            f"{range_server}/archive.zip",
            member,
            out,
            max_uncompressed_size=None,
        )
        assert out.read_bytes() == payloads[member]

    def test_matches_zipfile_extraction(
        self, range_server: str, tmp_path: Path
    ) -> None:
        payloads = _make_payloads(seed=11)
        archive = tmp_path / "archive.zip"
        _build_zip(archive, payloads, force_zip64=True)
        member = "models/train/point_cloud.ply"
        out = tmp_path / "ranged.bin"
        download_zip_member(f"{range_server}/archive.zip", member, out)
        with zipfile.ZipFile(archive) as zf:
            reference = zf.read(member)
        assert out.read_bytes() == reference
