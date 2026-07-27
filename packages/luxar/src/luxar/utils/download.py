"""Robust download utilities with retry logic, resume capability, and progress tracking.

This module provides production-grade download functionality for large datasets,
including automatic retry on failure, partial download resume, and integrity verification.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Optional, Union

from arbol import aprint, asection

# ─────────────────────────────────────────────────────────────────────────────
# Quarantined-cache detection
# ─────────────────────────────────────────────────────────────────────────────

#: Suffix appended by the demo/cache helpers when a cached artifact is found to
#: be truncated, unreadable or the wrong shape. A quarantined file is NEVER
#: reused — it is kept only so the user can inspect or salvage it.
QUARANTINE_SUFFIX = ".corrupt"


def _format_bytes(n_bytes: int) -> str:
    """Format a byte count as a compact human-readable size."""
    size = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"  # pragma: no cover - loop always returns


def find_quarantined_files(target: Union[str, Path]) -> list[Path]:
    """Return the quarantined ``.corrupt`` files associated with *target*.

    Args:
        target: A cache *file* (both the appended form ``foo.npy.corrupt`` that
            :func:`quarantine_file` writes and the bare ``Path.with_suffix``
            form ``foo.corrupt`` are checked, so older hand-rolled quarantines
            stay discoverable) or a cache *directory* (every ``*.corrupt``
            inside it is reported).

    Returns:
        Existing quarantined paths, sorted and de-duplicated (empty when clean).
    """
    target = Path(target)
    if target.is_dir():
        return sorted(p for p in target.glob(f"*{QUARANTINE_SUFFIX}") if p.is_file())

    candidates = [target.with_name(target.name + QUARANTINE_SUFFIX)]
    if target.suffix:
        candidates.append(target.with_suffix(QUARANTINE_SUFFIX))
    seen: dict[Path, None] = {}
    for path in candidates:
        if path.is_file():
            seen[path] = None
    return sorted(seen)


def format_quarantine_notice(
    paths: list[Path],
    *,
    indent: str = "  ",
    action: str = "re-download the full file (or delete the quarantined copy)",
) -> str:
    """Build an actionable multi-line notice for quarantined cache files.

    Returns an empty string when *paths* is empty, so callers can splice the
    result straight into a larger message.
    """
    if not paths:
        return ""
    lines = [
        f"{indent}⚠ A previously cached copy was QUARANTINED as corrupt and will "
        f"not be reused:"
    ]
    for path in paths:
        try:
            size = _format_bytes(path.stat().st_size)
        except OSError:  # pragma: no cover - raced deletion
            size = "unknown size"
        lines.append(f"{indent}    {path}  ({size})")
    lines.append(f"{indent}  To proceed you must {action}.")
    return "\n".join(lines)


def warn_if_quarantined(
    target: Union[str, Path],
    *,
    verbose: bool = True,
    action: str = "re-download the full file (or delete the quarantined copy)",
) -> list[Path]:
    """Print an actionable warning if *target* has quarantined ``.corrupt`` files.

    Called at the download chokepoint so a user who is about to re-fetch a
    multi-gigabyte artifact is told *why* — a truncated earlier copy is sitting
    next to it — instead of silently watching a huge download start over.

    Returns:
        The quarantined paths found (empty when clean).
    """
    quarantined = find_quarantined_files(target)
    if quarantined and verbose:
        aprint(format_quarantine_notice(quarantined, indent="", action=action))
    return quarantined


def quarantine_file(
    path: Union[str, Path],
    *,
    reason: str = "",
    verbose: bool = True,
) -> Path:
    """Rename a rejected cache artifact out of the way; return its new path.

    The producer side of the ``.corrupt`` convention that
    :func:`find_quarantined_files` and :func:`warn_if_quarantined` already read.

    Quarantining rather than deleting keeps the bytes for inspection or salvage
    and, critically, gets them out from under the canonical name so that
    :func:`robust_download` cannot RESUME onto them — it opens the destination in
    append mode whenever a file is already there, so a complete-but-wrong file
    left in place would be appended to rather than replaced.

    The suffix is APPENDED (``foo.zip`` -> ``foo.zip.corrupt``) so the original
    name and extension survive intact; that is also the form ``luxar demo clear``
    classifies correctly. A pre-existing quarantine for the same file is
    REPLACED, not stacked: one slot per file, so a repeated corrupt-fetch loop
    cannot fill a disk with copies of a multi-gigabyte artifact, and the finders
    (which match the exact ``.corrupt`` name) keep working.

    Args:
        path: The rejected artifact. Must be an existing regular file.
        reason: Short cause ("sha256 mismatch", "truncated"), shown in the notice.
        verbose: Print the notice. The rename happens either way.

    Returns:
        Path of the quarantined file.

    Raises:
        FileNotFoundError: *path* is missing or is not a regular file.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Cannot quarantine (not a regular file): {path}")

    target = path.with_name(path.name + QUARANTINE_SUFFIX)
    path.replace(target)  # atomic; silently clobbers an older quarantine
    if verbose:
        detail = f" ({reason})" if reason else ""
        aprint(
            f"⚠️  Quarantined cache file{detail}: {path.name} → {target.name}. "
            "It will never be reused; delete it (or run 'luxar demo clear') to "
            f"reclaim {_format_bytes(target.stat().st_size)}."
        )
    return target


def robust_download(
    url: str,
    output_path: Path,
    max_retries: int = 3,
    timeout: int = 300,
    chunk_size: int = 1024 * 1024,  # 1MB chunks
    verify_size: bool = True,
    expected_size: Optional[int] = None,
    extra_headers: Optional[dict] = None,
) -> Path:
    """Download a file with automatic retry, resume capability, and progress tracking.

    Features:
    - Automatic retry on network errors (exponential backoff)
    - Resume partial downloads (HTTP Range requests)
    - Progress tracking with ETA
    - File size verification
    - Cleanup of corrupted partial downloads

    Args:
        url: URL to download from
        output_path: Where to save the downloaded file
        max_retries: Maximum number of retry attempts (default: 3)
        timeout: Timeout in seconds for initial connection (default: 300)
        chunk_size: Size of download chunks in bytes (default: 1MB)
        verify_size: Whether to verify final file size matches Content-Length
        expected_size: Expected file size in bytes (optional, for validation)
        extra_headers: Extra HTTP headers to send on every request (e.g. an
            API key: ``{"api-key": "..."}``). Merged with the Range header.

    Returns:
        Path to downloaded file

    Raises:
        requests.HTTPError: If HTTP error occurs after all retries
        requests.ConnectionError: If connection fails after all retries
        ValueError: If downloaded file size doesn't match expected size

    Example:
        >>> from luxar.utils.download import robust_download
        >>> path = robust_download(
        ...     "https://example.com/large_dataset.zip",
        ...     Path("data/dataset.zip"),
        ...     max_retries=5
        ... )
        >>> aprint(f"Downloaded to {path}")
    """
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if file already exists and is complete
    if output_path.exists() and expected_size:
        current_size = output_path.stat().st_size
        if current_size == expected_size:
            aprint(f"✓ File already downloaded: {output_path}")
            aprint(f"  Size: {current_size / (1024**3):.2f} GB")
            return output_path

    # A quarantined `.corrupt` sibling means an earlier copy was rejected as
    # truncated/unreadable. Say so BEFORE re-fetching, so a multi-GB download
    # never starts unexplained. The action differs from the default here: this
    # call site IS the re-download, so telling the user to re-download would
    # name the thing already happening — the only thing left for them to do is
    # reclaim the space the rejected copy is holding.
    warn_if_quarantined(
        output_path,
        action=(
            "delete the quarantined copy to reclaim its disk space — this run "
            "is already re-fetching the file from scratch"
        ),
    )

    # Set up session with retry logic
    session = requests.Session()
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=2,  # Exponential backoff: 2s, 4s, 8s...
        status_forcelist=[429, 500, 502, 503, 504],  # Retry on these HTTP codes
        allowed_methods=["HEAD", "GET", "OPTIONS"],  # Methods to retry
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Determine if we can resume
    resume_byte_pos = 0
    if output_path.exists():
        resume_byte_pos = output_path.stat().st_size
        aprint(f"📂 Partial download found: {resume_byte_pos / (1024**2):.1f} MB")
        aprint("   Attempting to resume...")

    attempt = 0
    while attempt <= max_retries:
        try:
            # Set up headers for resume
            headers = dict(extra_headers or {})
            if resume_byte_pos > 0:
                headers["Range"] = f"bytes={resume_byte_pos}-"

            with asection(f"Download Attempt {attempt + 1}/{max_retries + 1}"):
                # Make request
                response = session.get(
                    url, headers=headers, stream=True, timeout=timeout
                )
                response.raise_for_status()

                # Check if resume was accepted
                if resume_byte_pos > 0 and response.status_code == 206:
                    aprint(f"✓ Resuming from {resume_byte_pos / (1024**2):.1f} MB")
                    mode = "ab"  # Append mode
                elif resume_byte_pos > 0:
                    aprint("⚠️  Server doesn't support resume, restarting download")
                    resume_byte_pos = 0
                    mode = "wb"
                else:
                    mode = "wb"

                # Get total size
                if "content-length" in response.headers:
                    content_length = int(response.headers["content-length"])
                    total_size = content_length + resume_byte_pos
                elif "content-range" in response.headers:
                    # For resumed downloads: "bytes start-end/total"
                    content_range = response.headers["content-range"]
                    total_size = int(content_range.split("/")[-1])
                else:
                    total_size = 0

                if total_size > 0:
                    aprint(f"📦 Total size: {total_size / (1024**3):.2f} GB")
                else:
                    aprint("📦 Size: Unknown (no Content-Length header)")

                # Download with progress
                downloaded = resume_byte_pos
                last_progress_mb = downloaded / (1024 * 1024)
                start_time = time.time()

                with open(output_path, mode) as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                            # Progress update every 100MB
                            progress_mb = downloaded / (1024 * 1024)
                            if progress_mb - last_progress_mb >= 100:
                                if total_size > 0:
                                    percent = downloaded / total_size * 100
                                    elapsed = time.time() - start_time
                                    rate_mbps = (
                                        (downloaded - resume_byte_pos)
                                        / (1024 * 1024)
                                        / elapsed
                                        if elapsed > 0
                                        else 0
                                    )
                                    remaining_bytes = total_size - downloaded
                                    eta_seconds = (
                                        remaining_bytes / (rate_mbps * 1024 * 1024)
                                        if rate_mbps > 0
                                        else 0
                                    )

                                    aprint(
                                        f"  {downloaded / (1024**3):.2f} GB / {total_size / (1024**3):.2f} GB "
                                        f"({percent:.1f}%) - {rate_mbps:.1f} MB/s - ETA: {eta_seconds / 60:.0f}min"
                                    )
                                else:
                                    aprint(
                                        f"  Downloaded: {downloaded / (1024**3):.2f} GB"
                                    )

                                last_progress_mb = progress_mb

                # Verify download completed
                final_size = output_path.stat().st_size
                aprint("✓ Download complete!")
                aprint(f"  Final size: {final_size / (1024**3):.2f} GB")

                # Verify size if expected
                if verify_size and total_size > 0:
                    if final_size != total_size:
                        raise ValueError(
                            f"Downloaded file size mismatch: expected {total_size} bytes, "
                            f"got {final_size} bytes"
                        )
                    aprint(f"✓ Size verified: {final_size} bytes")

                if expected_size and final_size != expected_size:
                    aprint(
                        f"⚠️  Warning: File size ({final_size}) doesn't match expected ({expected_size})"
                    )

                return output_path

        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ) as e:
            attempt += 1
            if attempt <= max_retries:
                wait_time = 2**attempt  # Exponential backoff
                aprint(f"❌ Download error: {type(e).__name__}: {e}")
                aprint(
                    f"   Retrying in {wait_time} seconds... (attempt {attempt}/{max_retries})"
                )
                time.sleep(wait_time)
                # Keep partial file for resume attempt
            else:
                aprint(f"❌ Download failed after {max_retries + 1} attempts")
                if output_path.exists():
                    aprint(f"   Partial download saved at: {output_path}")
                    aprint(
                        f"   You can retry to resume from {output_path.stat().st_size / (1024**2):.1f} MB"
                    )
                raise

        except requests.exceptions.HTTPError as e:
            aprint(f"❌ HTTP error: {e}")
            if output_path.exists():
                output_path.unlink()  # Clean up on HTTP errors (bad URL, etc.)
            raise

        except Exception as e:
            aprint(f"❌ Unexpected error: {type(e).__name__}: {e}")
            # Keep partial file - might be resumable
            raise

    # Should never reach here
    raise RuntimeError("Download failed after all retry attempts")


def verify_file_checksum(
    file_path: Path,
    expected_md5: Optional[str] = None,
    expected_sha256: Optional[str] = None,
) -> bool:
    """Verify file integrity using checksums.

    Args:
        file_path: Path to file to verify
        expected_md5: Expected MD5 hash (optional)
        expected_sha256: Expected SHA256 hash (optional)

    Returns:
        True if file matches expected checksum(s), False otherwise

    Example:
        >>> verify_file_checksum(
        ...     Path("dataset.zip"),
        ...     expected_sha256="abc123..."
        ... )
        True
    """
    if not file_path.exists():
        return False

    with asection(f"Verifying {file_path.name}"):
        if expected_md5:
            aprint("Computing MD5...")
            # Integrity check of a downloaded artifact, not a security control:
            # usedforsecurity=False documents intent and clears bandit B324.
            md5_hash = hashlib.md5(usedforsecurity=False)
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192 * 128), b""):
                    md5_hash.update(chunk)
            actual_md5 = md5_hash.hexdigest()

            if actual_md5 == expected_md5:
                aprint(f"✓ MD5 verified: {actual_md5}")
            else:
                aprint("❌ MD5 mismatch!")
                aprint(f"   Expected: {expected_md5}")
                aprint(f"   Actual:   {actual_md5}")
                return False

        if expected_sha256:
            aprint("Computing SHA256...")
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192 * 128), b""):
                    sha256_hash.update(chunk)
            actual_sha256 = sha256_hash.hexdigest()

            if actual_sha256 == expected_sha256:
                aprint(f"✓ SHA256 verified: {actual_sha256}")
            else:
                aprint("❌ SHA256 mismatch!")
                aprint(f"   Expected: {expected_sha256}")
                aprint(f"   Actual:   {actual_sha256}")
                return False

    return True


def download_with_checksum(
    url: str,
    output_path: Path,
    expected_md5: Optional[str] = None,
    expected_sha256: Optional[str] = None,
    **kwargs: Any,
) -> Path:
    """Download file and verify checksum.

    Combines robust_download with checksum verification for data integrity.

    Args:
        url: URL to download from
        output_path: Where to save the file
        expected_md5: Expected MD5 hash (optional)
        expected_sha256: Expected SHA256 hash (optional)
        **kwargs: Additional arguments passed to robust_download

    Returns:
        Path to downloaded and verified file

    Raises:
        ValueError: If checksum verification fails

    Example:
        >>> download_with_checksum(
        ...     "https://example.com/data.zip",
        ...     Path("data.zip"),
        ...     expected_sha256="abc123...",
        ...     max_retries=5
        ... )
    """
    # Download file
    output_path = robust_download(url, output_path, **kwargs)

    # Verify checksum if provided
    if expected_md5 or expected_sha256:
        if not verify_file_checksum(output_path, expected_md5, expected_sha256):
            # Checksum failed - delete corrupted file
            aprint("❌ Checksum verification failed - file may be corrupted")
            output_path.unlink()
            raise ValueError(
                "Downloaded file failed checksum verification. "
                "File has been deleted. Please try downloading again."
            )

    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Remote-zip single-member extraction (HTTP Range)
# ─────────────────────────────────────────────────────────────────────────────

_EOCD_SIGNATURE = b"PK\x05\x06"
_EOCD64_LOCATOR_SIGNATURE = b"PK\x06\x07"
_EOCD64_SIGNATURE = b"PK\x06\x06"
_CENTRAL_DIR_SIGNATURE = b"PK\x01\x02"
_LOCAL_HEADER_SIGNATURE = b"PK\x03\x04"
#: EOCD is 22 bytes + up to 64 KiB of trailing comment.
_EOCD_TAIL_BYTES = 22 + 65536


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
    response.raise_for_status()
    if response.status_code != 206:
        raise ValueError(
            f"Server ignored the Range request (HTTP {response.status_code}) — "
            "remote-zip extraction needs Accept-Ranges: bytes"
        )
    return response


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
    tail = _ranged_get(
        session,
        url,
        tail_start,
        archive_size - 1,
        timeout=timeout,
        extra_headers=extra_headers,
    ).content

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
        eocd64 = _ranged_get(
            session,
            url,
            eocd64_offset,
            eocd64_offset + 55,
            timeout=timeout,
            extra_headers=extra_headers,
        ).content
        if eocd64[:4] != _EOCD64_SIGNATURE:
            raise ValueError("Bad Zip64 end-of-central-directory signature")
        cd_size = struct.unpack_from("<Q", eocd64, 40)[0]
        cd_offset = struct.unpack_from("<Q", eocd64, 48)[0]

    central_dir = _ranged_get(
        session,
        url,
        cd_offset,
        cd_offset + cd_size - 1,
        timeout=timeout,
        extra_headers=extra_headers,
    ).content
    return central_dir, archive_size


def _find_member_in_central_dir(
    central_dir: bytes, member: str
) -> tuple[int, int, int, int]:
    """Locate ``member`` in central-directory bytes.

    Returns ``(local_header_offset, compressed_size, uncompressed_size, crc32)``,
    resolving Zip64 extra fields where the classic 32-bit fields saturate.
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
            return local_offset, comp_size, uncomp_size, crc32

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
        extra_headers: Extra HTTP headers for every request.

    Returns:
        Path to the extracted member.
    """
    import zlib

    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    from luxar.utils.demos import _validate_zip_member_path

    _validate_zip_member_path(member)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and expected_size:
        if output_path.stat().st_size == expected_size:
            aprint(f"✓ Member already extracted: {output_path}")
            return output_path

    session = requests.Session()
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
        local_offset, comp_size, uncomp_size, crc_expected = (
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
        if output_path.exists() and output_path.stat().st_size == uncomp_size:
            aprint(f"✓ Member already extracted: {output_path}")
            return output_path

        # The local header repeats name/extra with potentially DIFFERENT
        # lengths than the central directory — read it to find the data start.
        import struct

        local_header = _ranged_get(
            session,
            url,
            local_offset,
            local_offset + 29,
            timeout=timeout,
            extra_headers=extra_headers,
        ).content
        if local_header[:4] != _LOCAL_HEADER_SIGNATURE:
            raise ValueError("Bad local file header signature in remote zip")
        method = struct.unpack_from("<H", local_header, 8)[0]
        name_len = struct.unpack_from("<H", local_header, 26)[0]
        extra_len = struct.unpack_from("<H", local_header, 28)[0]
        data_start = local_offset + 30 + name_len + extra_len

        attempt = 0
        while True:
            try:
                tmp_path = output_path.with_suffix(output_path.suffix + ".part")
                decompressor = zlib.decompressobj(-15) if method == 8 else None
                crc = 0
                written = 0
                response = _ranged_get(
                    session,
                    url,
                    data_start,
                    data_start + comp_size - 1,
                    timeout=timeout,
                    extra_headers=extra_headers,
                    stream=True,
                )
                last_report = 0
                with open(tmp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        data = decompressor.decompress(chunk) if decompressor else chunk
                        f.write(data)
                        crc = zlib.crc32(data, crc)
                        written += len(data)
                        if written - last_report >= 100 * 1024 * 1024:
                            aprint(
                                f"  {written / 1e6:.0f} / {uncomp_size / 1e6:.0f} MB"
                            )
                            last_report = written
                    if decompressor:
                        data = decompressor.flush()
                        f.write(data)
                        crc = zlib.crc32(data, crc)
                        written += len(data)
                if written != uncomp_size:
                    raise ValueError(
                        f"Extracted {written:,} bytes; zip declares {uncomp_size:,}"
                    )
                if crc != crc_expected:
                    raise ValueError(
                        f"CRC32 mismatch: got {crc:#010x}, "
                        f"zip declares {crc_expected:#010x}"
                    )
                tmp_path.replace(output_path)
                aprint(f"✓ Extracted + CRC-verified: {output_path}")
                return output_path
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
