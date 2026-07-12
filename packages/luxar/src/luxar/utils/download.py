"""Robust download utilities with retry logic, resume capability, and progress tracking.

This module provides production-grade download functionality for large datasets,
including automatic retry on failure, partial download resume, and integrity verification.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Optional

from arbol import aprint, asection


def robust_download(
    url: str,
    output_path: Path,
    max_retries: int = 3,
    timeout: int = 300,
    chunk_size: int = 1024 * 1024,  # 1MB chunks
    verify_size: bool = True,
    expected_size: Optional[int] = None,
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
            headers = {}
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
