"""Extract individual members from remote ZIP archives via HTTP ranges."""

from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import Any, BinaryIO, Optional

from arbol import aprint, asection

from .download import _make_host_scoped_session

_EOCD_SIGNATURE = b"PK\x05\x06"
_EOCD64_LOCATOR_SIGNATURE = b"PK\x06\x07"
_EOCD64_SIGNATURE = b"PK\x06\x06"
_CENTRAL_DIR_SIGNATURE = b"PK\x01\x02"
_LOCAL_HEADER_SIGNATURE = b"PK\x03\x04"
#: EOCD is 22 bytes + up to 64 KiB of trailing comment.
_EOCD_TAIL_BYTES = 22 + 65536
#: Absolute ceiling on a single extracted member's uncompressed size (256 GiB),
#: independent of the archive's own (attacker-controlled) metadata. Mirrors
#: gsplats/io/_archive.py.
_MAX_MEMBER_UNCOMPRESSED_BYTES = 256 * 1024**3
#: Ceiling on the central directory we buffer via ``response.content`` (64 MiB).
#: A real central directory is small — one ~46-byte record plus the member
#: name/extra/comment (≈76 bytes for a typical name) per member — so the archives
#: we fetch run from tens of KB to at most a few MB. 64 MiB is a deliberately
#: generous ceiling that still blocks the whole-archive forgery: a forged EOCD
#: (e.g. cd_offset=0, cd_size=archive_size) would otherwise make us buffer the
#: entire archive into memory — a DoS on the metadata path, distinct from the
#: DEFLATE decompression-bomb guard on the member stream.
_MAX_CENTRAL_DIR_BYTES = 64 * 1024**2


def _ranged_get(
    session: Any,
    url: str,
    start: int,
    end: int,
    *,
    timeout: int,
    extra_headers: Optional[dict] = None,
    stream: bool = False,
) -> Any:
    """Closed-interval Range GET (inclusive byte range); asserts HTTP 206."""
    headers = dict(extra_headers or {})
    headers["Range"] = f"bytes={start}-{end}"
    response = session.get(url, headers=headers, timeout=timeout, stream=stream)
    # On any error path close the (possibly ``stream=True``) response
    # deterministically rather than leaking the connection to the GC — the
    # success path hands ownership to the caller, which closes it in turn.
    try:
        response.raise_for_status()
        if response.status_code != 206:
            raise ValueError(
                f"Server ignored the Range request (HTTP {response.status_code}) — "
                "remote-zip extraction needs Accept-Ranges: bytes"
            )
    except BaseException:
        response.close()
        raise
    return response


def _ranged_get_bytes(
    session: Any,
    url: str,
    start: int,
    end: int,
    *,
    timeout: int,
    extra_headers: Optional[dict] = None,
) -> bytes:
    """Read exactly one closed byte range without trusting the response size.

    ``requests`` buffers an entire response before exposing ``.content``. A
    broken or malicious HTTP 206 server can therefore ignore the requested end
    and stream to EOF, defeating any bound placed on the requested range. Read
    incrementally instead and abort after the first byte beyond the expected
    range, while also rejecting a truncated response.
    """
    expected = end - start + 1
    if expected <= 0:
        raise ValueError(f"Invalid byte range {start}-{end}")

    data = bytearray()
    chunk_size = min(64 * 1024, expected + 1)
    with contextlib.closing(
        _ranged_get(
            session,
            url,
            start,
            end,
            timeout=timeout,
            extra_headers=extra_headers,
            stream=True,
        )
    ) as response:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            received = len(data) + len(chunk)
            if received > expected:
                raise ValueError(
                    f"Server returned more bytes ({received:,}) than requested "
                    f"({expected:,}) for Range {start}-{end}"
                )
            data.extend(chunk)

    if len(data) != expected:
        raise ValueError(
            f"Server returned {len(data):,} bytes for Range {start}-{end}; "
            f"expected {expected:,}"
        )
    return bytes(data)


def _parse_remote_zip_directory(
    session: Any, url: str, *, timeout: int, extra_headers: Optional[dict]
) -> tuple[bytes, int]:
    """Fetch and locate the central directory of a remote zip via Range requests.

    Returns ``(central_directory_bytes, archive_size)``. Handles both classic
    and Zip64 archives (mandatory for >4 GiB files like INRIA's models.zip).
    """
    import struct

    # allow_redirects=True is REQUIRED: requests' HEAD does not follow
    # redirects by default, so on a redirecting host (GitHub release assets,
    # Hugging Face /resolve/ → signed CDN) an unfollowed HEAD returns the 3xx
    # with Content-Length: 0, making archive_size 0 and the tail range
    # `bytes=0--1` — which such CDNs reject with 501. The ranged GETs below
    # follow redirects on their own; HEAD must match.
    head = session.head(
        url, timeout=timeout, headers=dict(extra_headers or {}), allow_redirects=True
    )
    head.raise_for_status()
    if "content-length" not in head.headers:
        raise ValueError("Remote server did not report Content-Length")
    archive_size = int(head.headers["content-length"])

    tail_start = max(0, archive_size - _EOCD_TAIL_BYTES)
    tail = _ranged_get_bytes(
        session,
        url,
        tail_start,
        archive_size - 1,
        timeout=timeout,
        extra_headers=extra_headers,
    )

    eocd_local = tail.rfind(_EOCD_SIGNATURE)
    if eocd_local < 0:
        raise ValueError("No end-of-central-directory record found — not a zip?")
    eocd = tail[eocd_local:]
    cd_size = struct.unpack_from("<I", eocd, 12)[0]
    cd_offset = struct.unpack_from("<I", eocd, 16)[0]

    # Zip64: the classic fields saturate at 0xFFFFFFFF and the real values
    # live in the Zip64 EOCD record, found via its locator (20 bytes,
    # immediately before the EOCD).
    needs_zip64 = cd_offset == 0xFFFFFFFF or cd_size == 0xFFFFFFFF
    locator_local = eocd_local - 20
    has_locator = (
        locator_local >= 0
        and tail[locator_local : locator_local + 4] == _EOCD64_LOCATOR_SIGNATURE
    )
    if needs_zip64 or has_locator:
        if not has_locator:
            raise ValueError("Zip64 archive without a Zip64 EOCD locator")
        eocd64_offset = struct.unpack_from("<Q", tail, locator_local + 8)[0]
        # `<Q` is unsigned, so eocd64_offset >= 0 always — only the upper bound
        # can be violated.
        if eocd64_offset + 56 > archive_size:
            raise ValueError(
                f"Zip64 EOCD offset {eocd64_offset} lies outside the archive "
                f"(size {archive_size}) — refusing to fetch"
            )
        eocd64 = _ranged_get_bytes(
            session,
            url,
            eocd64_offset,
            eocd64_offset + 55,
            timeout=timeout,
            extra_headers=extra_headers,
        )
        if eocd64[:4] != _EOCD64_SIGNATURE:
            raise ValueError("Bad Zip64 end-of-central-directory signature")
        cd_size = struct.unpack_from("<Q", eocd64, 40)[0]
        cd_offset = struct.unpack_from("<Q", eocd64, 48)[0]

    # Validate the EOCD-derived (attacker-controlled) central-directory bounds
    # before the ranged GET buffers the response into memory. Covers both the
    # classic and Zip64 paths, since both resolve into cd_size/cd_offset here.
    # A forged EOCD (e.g. cd_offset=0, cd_size=archive_size) would otherwise
    # make us buffer the whole archive — a memory-exhaustion DoS.
    # cd_size/cd_offset come from unsigned struct fields (<I/<Q), so neither can
    # be negative — only the empty-zip and upper-bound cases are reachable.
    if cd_size == 0:
        raise ValueError("Remote zip has an empty central directory (no members)")
    if cd_size > _MAX_CENTRAL_DIR_BYTES:
        raise ValueError(
            f"Central-directory size {cd_size} is out of range "
            f"(1..{_MAX_CENTRAL_DIR_BYTES} bytes) — refusing to fetch"
        )
    if cd_offset >= archive_size:
        raise ValueError(
            f"Central-directory offset {cd_offset} lies outside the archive "
            f"(size {archive_size}) — refusing to fetch"
        )
    if cd_offset + cd_size > archive_size:
        raise ValueError(
            f"Central directory (offset {cd_offset}, size {cd_size}) overruns "
            f"the archive (size {archive_size}) — refusing to fetch"
        )

    central_dir = _ranged_get_bytes(
        session,
        url,
        cd_offset,
        cd_offset + cd_size - 1,
        timeout=timeout,
        extra_headers=extra_headers,
    )
    return central_dir, archive_size


def _find_member_in_central_dir(
    central_dir: bytes, member: str
) -> tuple[int, int, int, int, int]:
    """Locate ``member`` in central-directory bytes.

    Returns ``(local_header_offset, compressed_size, uncompressed_size, crc32,
    method)`` (``method`` is 0=STORED or 8=DEFLATE), resolving Zip64 extra
    fields where the classic 32-bit fields saturate.
    """
    import struct

    offset = 0
    names = []
    while offset + 46 <= len(central_dir):
        if central_dir[offset : offset + 4] != _CENTRAL_DIR_SIGNATURE:
            break
        method = struct.unpack_from("<H", central_dir, offset + 10)[0]
        crc32 = struct.unpack_from("<I", central_dir, offset + 16)[0]
        comp_size = struct.unpack_from("<I", central_dir, offset + 20)[0]
        uncomp_size = struct.unpack_from("<I", central_dir, offset + 24)[0]
        name_len = struct.unpack_from("<H", central_dir, offset + 28)[0]
        extra_len = struct.unpack_from("<H", central_dir, offset + 30)[0]
        comment_len = struct.unpack_from("<H", central_dir, offset + 32)[0]
        local_offset = struct.unpack_from("<I", central_dir, offset + 42)[0]
        name = central_dir[offset + 46 : offset + 46 + name_len].decode(
            "utf-8", errors="replace"
        )

        if name == member:
            # Zip64 extra field (id 0x0001): 8-byte values appear in a fixed
            # order, but ONLY for the classic fields that saturated.
            extra = central_dir[
                offset + 46 + name_len : offset + 46 + name_len + extra_len
            ]
            e = 0
            while e + 4 <= len(extra):
                field_id, field_len = struct.unpack_from("<HH", extra, e)
                if field_id == 0x0001:
                    v = e + 4
                    if uncomp_size == 0xFFFFFFFF:
                        uncomp_size = struct.unpack_from("<Q", extra, v)[0]
                        v += 8
                    if comp_size == 0xFFFFFFFF:
                        comp_size = struct.unpack_from("<Q", extra, v)[0]
                        v += 8
                    if local_offset == 0xFFFFFFFF:
                        local_offset = struct.unpack_from("<Q", extra, v)[0]
                        v += 8
                    break
                e += 4 + field_len
            if method not in (0, 8):
                raise ValueError(
                    f"Member {member!r} uses unsupported compression method "
                    f"{method} (only STORED and DEFLATE are supported)"
                )
            return local_offset, comp_size, uncomp_size, crc32, method

        names.append(name)
        offset += 46 + name_len + extra_len + comment_len

    preview = ", ".join(names[:8]) + ("…" if len(names) > 8 else "")
    raise KeyError(f"Member {member!r} not found in remote zip (has: {preview})")


def download_zip_member(
    url: str,
    member: str,
    output_path: Path,
    *,
    expected_size: Optional[int] = None,
    max_retries: int = 3,
    timeout: int = 300,
    chunk_size: int = 1024 * 1024,
    extra_headers: Optional[dict] = None,
    max_uncompressed_size: Optional[int] = _MAX_MEMBER_UNCOMPRESSED_BYTES,
) -> Path:
    """Extract ONE member from a remote zip via HTTP Range requests.

    Downloads only the bytes of the requested member (plus a few KB of zip
    bookkeeping) instead of the whole archive — e.g. a 1.5 GB scene out of
    INRIA's 14.7 GB ``models.zip``. Requires the server to honor
    ``Range`` requests (``Accept-Ranges: bytes``); redirects (e.g. Hugging
    Face ``/resolve/`` → signed CDN) are followed.

    Flow: ranged GET of the archive tail → EOCD (+ mandatory Zip64 records
    for >4 GiB archives) → ranged GET of the central directory → locate the
    member → ranged streaming GET of its compressed bytes → inflate
    (``zlib`` raw DEFLATE) or raw copy (STORED) to disk → CRC32 + size
    verification against the central directory.

    Args:
        url: Archive URL.
        member: Exact member path inside the zip (POSIX separators).
        output_path: Where to write the extracted (decompressed) member.
        expected_size: Optional expected uncompressed size — used both to
            skip an already-complete download and to sanity-check the zip.
        max_retries: Retry attempts for the member body download.
        timeout: Per-request timeout (seconds).
        chunk_size: Streaming chunk size (bytes).
        extra_headers: Extra HTTP headers scoped to the original URL.
        max_uncompressed_size: Absolute ceiling (bytes) on the member's
            uncompressed size, independent of the archive's own
            (attacker-controlled) metadata. Defaults to 256 GiB
            (``_MAX_MEMBER_UNCOMPRESSED_BYTES``). A member whose
            central-directory uncompressed size exceeds it is rejected before
            any streaming; the inflate itself is then bounded by that
            (already-within-ceiling) declared size — so a self-consistent
            decompression bomb (one whose declared size, actual inflated size,
            and CRC all agree) still cannot write unbounded to disk. Pass a
            smaller int to tighten it, or ``None`` to disable the ceiling
            entirely (the output is then bounded only by the declared
            member size). The INRIA / cluster-fly demo callers pass a tight
            ``expected_size`` and are unaffected by this default.

    Returns:
        Path to the extracted member.
    """
    import zlib

    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    from .zip_safety import _validate_zip_member_path

    _validate_zip_member_path(member)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and expected_size:
        if output_path.stat().st_size == expected_size:
            aprint(f"✓ Member already extracted: {output_path}")
            return output_path

    session = _make_host_scoped_session(url, (extra_headers or {}).keys())
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
    )
    session.mount("http://", HTTPAdapter(max_retries=retry_strategy))
    session.mount("https://", HTTPAdapter(max_retries=retry_strategy))

    with asection(f"Extracting {member} from remote zip"):
        central_dir, archive_size = _parse_remote_zip_directory(
            session, url, timeout=timeout, extra_headers=extra_headers
        )
        local_offset, comp_size, uncomp_size, crc_expected, cd_method = (
            _find_member_in_central_dir(central_dir, member)
        )
        aprint(
            f"Member: {comp_size / 1e6:.1f} MB compressed → "
            f"{uncomp_size / 1e6:.1f} MB (archive: {archive_size / 1e9:.2f} GB)"
        )
        if expected_size is not None and uncomp_size != expected_size:
            raise ValueError(
                f"Member size mismatch: zip declares {uncomp_size:,} bytes, "
                f"expected {expected_size:,}"
            )
        if max_uncompressed_size is not None and uncomp_size > max_uncompressed_size:
            raise ValueError(
                f"Member declares {uncomp_size:,} uncompressed bytes, exceeding "
                f"the {max_uncompressed_size:,}-byte ceiling "
                f"(max_uncompressed_size); refusing to extract"
            )
        # Never inflate beyond the declared member size — guards against a
        # decompression bomb whose small DEFLATE stream expands without bound.
        # A member declaring more than the absolute ceiling was already rejected
        # above (before any streaming), so uncomp_size is here always the
        # tighter of the two bounds.
        size_limit = uncomp_size
        if output_path.exists() and output_path.stat().st_size == uncomp_size:
            aprint(f"✓ Member already extracted: {output_path}")
            return output_path

        if comp_size == 0:
            # Empty STORED member: the body range GET below would be an inverted
            # ``bytes=data_start-(data_start-1)`` that servers answer 416/200, so
            # never fetch a body. A legitimate empty member is STORED (method 0)
            # with zero uncompressed bytes and CRC 0 — the shortest empty DEFLATE
            # stream is two bytes, so a DEFLATE member (or any nonzero size/CRC)
            # with zero compressed bytes is a malformed header. Reject it —
            # deliberately stricter than stdlib ``zipfile``, which accepts such a
            # forged empty DEFLATE entry (a zero-byte raw stream flushes to empty
            # and CRC 0 matches) — instead of silently extracting empty.
            # A valid empty member writes through the same ``.part``-then-
            # ``replace`` promotion the streaming path uses.
            if cd_method != 0 or uncomp_size != 0 or crc_expected != 0:
                raise ValueError(
                    f"Member {member!r} has zero compressed bytes but is not a "
                    f"valid empty STORED entry (method {cd_method}, "
                    f"{uncomp_size:,} uncompressed bytes, CRC {crc_expected:#010x})"
                )
            tmp_path = output_path.with_suffix(output_path.suffix + ".part")
            tmp_path.write_bytes(b"")
            tmp_path.replace(output_path)
            aprint(f"✓ Extracted empty member: {output_path}")
            return output_path

        # The local header repeats name/extra with potentially DIFFERENT
        # lengths than the central directory — read it to find the data start.
        import struct

        local_header = _ranged_get_bytes(
            session,
            url,
            local_offset,
            local_offset + 29,
            timeout=timeout,
            extra_headers=extra_headers,
        )
        if local_header[:4] != _LOCAL_HEADER_SIGNATURE:
            raise ValueError("Bad local file header signature in remote zip")
        method = struct.unpack_from("<H", local_header, 8)[0]
        name_len = struct.unpack_from("<H", local_header, 26)[0]
        extra_len = struct.unpack_from("<H", local_header, 28)[0]
        data_start = local_offset + 30 + name_len + extra_len

        crc = 0
        written = 0
        last_report = 0

        def _emit(f: BinaryIO, data: bytes) -> None:
            """Write one decompressed slice, updating CRC/counters, and abort
            if the running output exceeds the bound (decompression-bomb guard).
            """
            nonlocal crc, written, last_report
            if not data:
                return
            if written + len(data) > size_limit:
                # Check BEFORE writing so over-limit bytes never touch disk.
                raise ValueError(
                    f"Decompressed output ({written + len(data):,} bytes) "
                    f"exceeds the declared member size ({size_limit:,} bytes) "
                    f"for {member!r} — possible decompression bomb; aborting "
                    f"extraction"
                )
            f.write(data)
            crc = zlib.crc32(data, crc)
            written += len(data)
            if written - last_report >= 100 * 1024 * 1024:
                aprint(f"  {written / 1e6:.0f} / {uncomp_size / 1e6:.0f} MB")
                last_report = written

        attempt = 0
        tmp_path = output_path.with_suffix(output_path.suffix + ".part")
        while True:
            try:
                decompressor = zlib.decompressobj(-15) if method == 8 else None
                crc = 0
                written = 0
                last_report = 0
                compressed_read = 0
                with (
                    contextlib.closing(
                        _ranged_get(
                            session,
                            url,
                            data_start,
                            data_start + comp_size - 1,
                            timeout=timeout,
                            extra_headers=extra_headers,
                            stream=True,
                        )
                    ) as response,
                    open(tmp_path, "wb") as f,
                ):
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        # Secondary defense: the ranged GET requested exactly
                        # comp_size bytes, so a well-formed member reads exactly
                        # that. Abort if a misbehaving/malicious server streams
                        # more (which would otherwise let iter_content run
                        # unbounded), before touching the decompressor.
                        compressed_read += len(chunk)
                        if compressed_read > comp_size:
                            raise ValueError(
                                f"Server returned more compressed bytes "
                                f"({compressed_read:,}) than member {member!r} "
                                f"declares ({comp_size:,}); aborting extraction"
                            )
                        if decompressor is None:
                            # STORED: bound the raw copy by the same check.
                            _emit(f, chunk)
                            continue
                        # DEFLATE: drain in <=chunk_size slices, feeding the
                        # unconsumed tail back, so a single compressed chunk
                        # cannot inflate unbounded in memory.
                        buf = chunk
                        while buf:
                            data = decompressor.decompress(buf, chunk_size)
                            _emit(f, data)
                            buf = decompressor.unconsumed_tail
                    if decompressor is not None:
                        _emit(f, decompressor.flush())
                if written != uncomp_size:
                    raise ValueError(
                        f"Extracted {written:,} bytes; zip declares {uncomp_size:,}"
                    )
                if crc != crc_expected:
                    raise ValueError(
                        f"CRC32 mismatch: got {crc:#010x}, "
                        f"zip declares {crc_expected:#010x}"
                    )
                break
            except (
                requests.ConnectionError,
                requests.Timeout,
                requests.exceptions.ChunkedEncodingError,
            ) as exc:
                attempt += 1
                tmp_path.unlink(missing_ok=True)
                if attempt > max_retries:
                    raise
                wait = 2**attempt
                aprint(f"⚠️  Network error ({exc}); retrying in {wait}s…")
                time.sleep(wait)
            except BaseException:
                # Size/CRC/decompression failure (or anything non-retryable):
                # never leave the oversized/partial staging file behind.
                tmp_path.unlink(missing_ok=True)
                raise

        # Promote OUTSIDE the retry/cleanup guard above: the member is now fully
        # written and CRC-verified, so a failure to rename it into place (e.g.
        # the destination already exists as a directory, or a read-only parent)
        # must NOT trip the `except BaseException` cleanup and delete the
        # verified bytes.
        tmp_path.replace(output_path)
        aprint(f"✓ Extracted + CRC-verified: {output_path}")
        return output_path
