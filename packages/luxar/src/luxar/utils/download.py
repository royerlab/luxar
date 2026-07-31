"""Robust download utilities with retry logic, resume capability, and progress tracking.

This module provides production-grade download functionality for large datasets,
including automatic retry on failure, partial download resume, and integrity verification.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import time
from pathlib import Path
from typing import Any, BinaryIO, Optional, Union

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


def _force_identity_encoding(headers: dict) -> None:
    """Default Accept-Encoding to identity unless the caller already set it
    (case-insensitively). An identity byte stream is required: ``total_size``
    comes from Content-Length but the body is decoded, and ``.part`` byte-range
    resume assumes identity."""
    if not any(k.lower() == "accept-encoding" for k in headers):
        headers["Accept-Encoding"] = "identity"


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
    :func:`robust_download` cannot mistake them for a valid cache — a file at the
    canonical ``output_path`` is treated as COMPLETE (in-progress bytes stage in a
    sibling ``.part`` file), so a complete-but-wrong file left in place would be
    trusted and returned rather than re-fetched.

    The suffix is APPENDED (``foo.zip`` -> ``foo.zip.corrupt``) so the original
    name and extension survive intact; that is also the form
    ``luxar demo cache clear`` classifies correctly. A pre-existing quarantine
    for the same file is REPLACED, not stacked: one slot per file, so a
    repeated corrupt-fetch loop cannot fill a disk with copies of a
    multi-gigabyte artifact, and the finders (which match the exact
    ``.corrupt`` name) keep working.

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
            "It will never be reused; delete it (or run 'luxar demo cache clear') to "
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
    - Resume partial downloads (HTTP Range requests), validated with
      ``If-Range`` against the recorded ETag/Last-Modified of the staged bytes
      so a remote that changed is re-fetched clean, never spliced
    - Progress tracking with ETA
    - File size verification
    - Atomic staging: bytes land in a sibling ``<dest>.part`` and are renamed
      onto the destination only once complete and size-verified, so an
      interrupted download leaves a resumable ``.part`` rather than a truncated
      file under the canonical name (and a pre-existing cache is never deleted
      on error)
    - A 416 (Range Not Satisfiable) while resuming at/after EOF is non-fatal:
      the cache is proven at-least-complete, and is returned untouched unless
      the 416's authoritative total contradicts its size (one clean restart)

    When the destination already exists — and a matching ``expected_size``
    hasn't already short-circuited the call — one or more size-probe requests
    (a HEAD, and possibly an unranged GET) are issued up front to decide
    whether to resume, restart, or return the cache as-is.

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
            Unless it already contains an ``Accept-Encoding`` (any casing),
            requests default to ``Accept-Encoding: identity`` so size
            verification and ``.part`` resume see the raw byte stream.

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
    # In-progress bytes are staged in a sibling `.part` file; a file existing at
    # `output_path` therefore means it is COMPLETE. Only a fully-downloaded,
    # size-verified `.part` is atomically renamed (os.replace) onto `output_path`.
    # This is the invariant that lets a size-less cache check trust output_path:
    # an interrupted download (Ctrl-C, OOM, exhausted retries) leaves its
    # truncated bytes in the `.part` file, never at the canonical name (#732).
    # `.with_name(...+".part")` APPENDS the suffix (foo.zip → foo.zip.part), so
    # the original name/extension survive; `.with_suffix` would replace them.
    part_path = output_path.with_name(output_path.name + ".part")
    # The staging file's provenance: the strong validator (ETag/Last-Modified)
    # of the remote representation its bytes came from. Sent as `If-Range` on
    # resume, so a remote that changed since the partial was written answers
    # with a 200 full body (clean restart) instead of a 206 tail that would
    # splice `old_prefix + new_tail` into a corrupt file passing size checks.
    # (Re)written whenever the staging file is started from scratch; removed
    # whenever the staging file is promoted or discarded. Absent (e.g. the
    # server offers no validator), resume stays best-effort as before.
    validator_path = output_path.with_name(output_path.name + ".part.validator")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if file already exists and is complete
    if output_path.exists() and expected_size:
        current_size = output_path.stat().st_size
        if current_size == expected_size:
            aprint(f"✓ File already downloaded: {output_path}")
            aprint(f"  Size: {current_size / (1024**3):.2f} GB")
            # Drop any stale staging file orphaned by an aborted prior run.
            part_path.unlink(missing_ok=True)
            validator_path.unlink(missing_ok=True)
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

    def _parse_len(headers: Any) -> Optional[int]:
        """Parse a *positive* Content-Length, else ``None`` (unknown).

        A non-positive value (``Content-Length: 0`` from a chunked/dynamic
        host) or a malformed/duplicated header (``"100, 100"``) must be treated
        as *unknown* — never as a real size — so it can't spuriously trigger the
        ``local > remote`` truncate-and-restart branch.
        """
        raw = headers.get("content-length")
        if raw is None:
            return None
        try:
            n = int(raw)
        except (TypeError, ValueError):
            return None
        return n if n > 0 else None

    def _parse_content_range_total(headers: Any) -> Optional[int]:
        """Parse the authoritative total from a ``Content-Range`` header.

        RFC 9110 §14.4: a 416 response carries ``Content-Range: bytes */<total>``
        (and a 206 carries ``bytes <start>-<end>/<total>``). Returns the trailing
        ``/<total>`` integer, or ``None`` for a missing header, a ``*`` total, or
        any malformed value — e.g. ``"bytes */12345" -> 12345``,
        ``"bytes 0-99/12345" -> 12345``, ``"bytes */*" -> None``,
        ``"bytes */0" -> 0``.
        """
        raw = headers.get("content-range")
        if raw is None:
            return None
        total = raw.rsplit("/", 1)[-1].strip()
        try:
            n = int(total)
        except (TypeError, ValueError):
            return None
        # Unlike _parse_len (Content-Length:0 → unknown), a Content-Range total
        # of 0 is a REAL size (the remote was replaced by an empty file); only
        # the "*" marker means unknown, and that already failed int() above.
        return n if n >= 0 else None

    def _parse_content_range_start(headers: Any) -> Optional[int]:
        """Parse ``<start>`` from a 206 ``Content-Range: bytes <start>-<end>/<total>``.

        Returns ``None`` for a missing/malformed header or the unsatisfied-range
        form ``bytes */<total>`` (whose range part is ``*``, not an integer).
        """
        raw = headers.get("content-range")
        if raw is None:
            return None
        value = raw.strip()
        if not value.lower().startswith("bytes"):
            return None
        span = value[5:].strip().split("/", 1)[0]
        try:
            return int(span.partition("-")[0].strip())
        except (TypeError, ValueError):
            return None

    def _strong_validator(headers: Any) -> Optional[str]:
        """Pick a validator usable in ``If-Range`` (RFC 9110 §13.1.5).

        Prefer a strong ETag (a weak ``W/`` tag must never be sent in
        If-Range); fall back to ``Last-Modified`` (an HTTP-date is the other
        allowed form). ``None`` when the response offers neither — resume then
        stays best-effort, exactly as before.
        """
        etag = str(headers.get("etag") or "").strip()
        if etag and not etag.startswith("W/"):
            return etag
        last_modified = str(headers.get("last-modified") or "").strip()
        return last_modified or None

    def _record_part_validator(headers: Any) -> None:
        """Persist the response's validator alongside a from-scratch staging file.

        A later run resumes the staged bytes with ``If-Range: <validator>``, so
        a remote that changed in between answers 200 (full body → clean
        restart) instead of a 206 tail that would splice old and new bytes.
        Without a usable validator the sidecar is removed so a stale one can
        never vouch for bytes it doesn't describe.
        """
        validator = _strong_validator(headers)
        if validator is None:
            validator_path.unlink(missing_ok=True)
        else:
            validator_path.write_text(validator, encoding="utf-8")

    def _read_part_validator() -> Optional[str]:
        """Read the staged file's recorded validator (``None`` if absent/empty)."""
        try:
            validator = validator_path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return validator or None

    def _resolve_remote_size() -> Optional[int]:
        """Best-effort remote Content-Length (``None`` if unknowable).

        Tries a lightweight HEAD first, then falls back to an unranged
        streaming GET whose headers we read and then close the body — some
        servers don't support HEAD or omit Content-Length on it. Uses the
        same session (retry strategy) and ``extra_headers`` as the download.
        """
        probe_headers = dict(extra_headers or {})
        # Force identity so the server reports the true (decoded) resource size
        # that matches the on-disk file — a `Content-Encoding: gzip` response
        # would otherwise report the COMPRESSED length and make a complete
        # cache look "larger than remote", truncating it.
        _force_identity_encoding(probe_headers)
        try:
            head = session.head(
                url, timeout=timeout, headers=probe_headers, allow_redirects=True
            )
            head.raise_for_status()
            size = _parse_len(head.headers)
            if size is not None:
                return size
        except requests.exceptions.RequestException:
            pass
        try:
            probe = session.get(
                url, timeout=timeout, headers=probe_headers, stream=True
            )
            try:
                probe.raise_for_status()
                size = _parse_len(probe.headers)
                if size is not None:
                    return size
            finally:
                probe.close()
        except requests.exceptions.RequestException:
            pass
        return None

    # Fast cache-hit + resume determination. A COMPLETE file at output_path is
    # returned untouched (never re-downloaded); any in-progress bytes to resume
    # live in part_path, NOT output_path. Resolve the remote size ONCE, BEFORE
    # issuing any Range request: a Range starting at (or past) EOF is
    # unsatisfiable and the server answers HTTP 416, so we must not blindly
    # resume from the staged file's size. The cached value is reused by the 416
    # handler below.
    resume_byte_pos = 0
    remote_size: Optional[int] = None
    # Whether we have already spent a remote-size probe this call (so the
    # `.part` block below doesn't probe a second time on a size-less host).
    remote_size_resolved = False
    # Track a pre-existing output_path we MIGRATE into the staging path (below)
    # so a transient failure can restore it untouched (offline-safe).
    migrated_from_output = False
    migrated_size = 0
    if output_path.exists():
        local_size = output_path.stat().st_size
        remote_size = _resolve_remote_size()
        remote_size_resolved = True
        if remote_size is not None and local_size == remote_size:
            # Already complete — return it untouched (do NOT re-download).
            aprint(f"✓ File already downloaded: {output_path}")
            aprint(f"  Size: {local_size / (1024**3):.2f} GB")
            if expected_size and local_size != expected_size:
                aprint(
                    f"⚠️  Warning: File size ({local_size}) doesn't match "
                    f"expected ({expected_size})"
                )
            # Drop any stale staging file orphaned by an aborted prior run.
            part_path.unlink(missing_ok=True)
            validator_path.unlink(missing_ok=True)
            return output_path

    # A pre-existing output_path whose completeness CANNOT be confirmed by size —
    # i.e. ONLY when the remote size is UNKNOWN (a size-less/chunked host, so
    # `_resolve_remote_size()` returned None) — is MIGRATED into the staging path,
    # so the resume/416 probe below can still CONFIRM its completeness via a Range
    # at EOF (a 416 → the 416-complete branch promotes it straight back onto
    # output_path, ZERO re-download) instead of blindly re-fetching the whole
    # file. We do NOT migrate on a KNOWN size mismatch (remote_size is not None
    # and local_size != remote_size): migrating there would feed a stale, wrong
    # file to the resume path and splice `stale[:local] + remote[local:]` into a
    # corrupt file that passes size verification. Instead we leave output_path in
    # place, fetch fresh into `.part`, and promote on success (a terminal failure
    # preserves the stale file — never destroyed on a transient error). If a
    # `.part` already exists we prefer those staged bytes and let a stale
    # non-complete output_path be overwritten by the eventual promotion.
    if output_path.exists() and remote_size is None and not part_path.exists():
        migrated_size = output_path.stat().st_size
        os.replace(output_path, part_path)
        # A validator sidecar left by an older aborted run described a staging
        # file that no longer exists — it must not vouch for the migrated bytes.
        validator_path.unlink(missing_ok=True)
        migrated_from_output = True

    def _restore_migrated_cache() -> None:
        """Restore a migrated pre-existing cache on a terminal failure.

        If the staging file was NEVER grown (offline, or an error before any
        body byte was written) it is byte-identical to the pre-existing cache,
        so move it back onto output_path — a transient failure must never
        destroy a usable cache, and it stays usable offline. A staging file that
        WAS grown is a genuine in-progress partial and is left in place
        (resumable next run).
        """
        if (
            migrated_from_output
            and part_path.exists()
            and not output_path.exists()
            and part_path.stat().st_size == migrated_size
        ):
            os.replace(part_path, output_path)

    if part_path.exists():
        part_size = part_path.stat().st_size
        if remote_size is None and not remote_size_resolved:
            remote_size = _resolve_remote_size()
            remote_size_resolved = True
        if remote_size is not None and part_size > remote_size:
            # Staged partial is LARGER than the remote asset (e.g. a re-uploaded,
            # smaller file): a Range at EOF would 416 — restart from scratch.
            aprint(
                "⚠️  Staged partial is larger than the remote source "
                f"({part_size} > {remote_size} bytes); restarting from scratch"
            )
            resume_byte_pos = 0
            # Discard the proven-stale staged bytes NOW rather than via the
            # loop's "wb" open: if the fetch fails before that open, the
            # condemned bytes would survive in `.part` and a LATER run could
            # resume onto them (e.g. after the remote grows past their length)
            # and splice a corrupt file that passes size verification.
            part_path.unlink(missing_ok=True)
            validator_path.unlink(missing_ok=True)
        else:
            # Genuinely partial (0 < part < remote) or unknown remote size:
            # best-effort resume. A resulting 416 is handled gracefully below.
            resume_byte_pos = part_size
            aprint(f"📂 Partial download found: {resume_byte_pos / (1024**2):.1f} MB")
            aprint("   Attempting to resume...")

    # A 416 while resuming triggers at most one clean full restart (no Range);
    # a second 416 with no Range is a genuine error and is re-raised.
    restarted_after_416 = False

    promoted = False  # True once the staging file is promoted onto output_path
    attempt = 0
    try:
        while attempt <= max_retries:
            try:
                # Set up headers for resume
                headers = dict(extra_headers or {})
                # Force an identity byte stream (see _force_identity_encoding):
                # size verification + `.part` byte-range resume require it.
                _force_identity_encoding(headers)
                sent_if_range = False
                if resume_byte_pos > 0:
                    headers["Range"] = f"bytes={resume_byte_pos}-"
                    # Prove the staged bytes still belong to the current remote
                    # representation: with `If-Range`, a server whose content
                    # changed answers 200 (full body → the clean-restart branch
                    # below) instead of a 206 tail that would splice
                    # `old_prefix + new_tail`. A migrated pre-existing cache
                    # never has a recorded validator (the sidecar is removed at
                    # migration), so its 416/206 confirm probe is unaffected.
                    resume_validator = _read_part_validator()
                    if resume_validator is not None:
                        headers["If-Range"] = resume_validator
                        sent_if_range = True

                # Bind the streamed GET in contextlib.closing so its socket is
                # released deterministically on EVERY exit — success (return),
                # 416 restart (`continue`), or re-raise — instead of leaking to
                # the GC. raise_for_status still leaves `e.response` usable
                # afterward: the headers are already buffered, only the socket is
                # released (the 416 handler reads headers, never the body).
                with (
                    asection(f"Download Attempt {attempt + 1}/{max_retries + 1}"),
                    contextlib.closing(
                        session.get(url, headers=headers, stream=True, timeout=timeout)
                    ) as response,
                ):
                    response.raise_for_status()

                    # Check if resume was accepted
                    if resume_byte_pos > 0 and response.status_code == 206:
                        if migrated_from_output:
                            # A MIGRATED file is, by our invariant, COMPLETE — a
                            # 206 to a Range at its EOF PROVES the remote has
                            # more/different bytes, so the migrated cache is STALE
                            # (not a genuine partial). Appending would splice
                            # `stale[:N] + remote[N:]`; instead discard the stale
                            # bytes and re-fetch from scratch. Reset resume_byte_pos
                            # and `continue` to re-issue the request WITHOUT a Range
                            # header (this 206 response only carries the tail
                            # `remote[N:]`; writing it as-is would leave that tail
                            # AS the whole file). The unranged retry gets a 200 full
                            # body and, since resume_byte_pos is now 0, cannot
                            # re-enter this branch — so it restarts exactly once.
                            # Clearing the migrated flag stops _restore_migrated_cache()
                            # resurrecting the (proven-stale) copy on a later failure:
                            # re-fetching a PROVEN-stale cache may legitimately
                            # discard it; it self-heals next run.
                            aprint(
                                "⚠️  Cached copy is stale (remote has changed); "
                                "re-downloading from scratch"
                            )
                            resume_byte_pos = 0
                            migrated_from_output = False
                            response.close()
                            # Discard the proven-stale staged bytes NOW, not via
                            # the next iteration's "wb" open: if the unranged
                            # re-fetch fails before that open, a full stale copy
                            # would survive in `.part` and a LATER run would
                            # resume it (206) and splice `stale[:N] + remote[N:]`
                            # into a corrupt file that passes size verification.
                            part_path.unlink(missing_ok=True)
                            validator_path.unlink(missing_ok=True)
                            continue
                        range_start = _parse_content_range_start(response.headers)
                        if range_start != resume_byte_pos:
                            # A single-part 206 MUST carry `Content-Range:
                            # bytes <start>-<end>/<total>` with <start> equal to
                            # the requested offset (RFC 9110). A missing header
                            # or a different start means appending would land
                            # the bytes at the wrong offset — corrupting the
                            # file while still passing size verification.
                            # Restart clean instead of appending blind.
                            aprint(
                                "⚠️  Resume response is misaligned (Content-Range "
                                f"start {range_start!r}, requested "
                                f"{resume_byte_pos}); restarting from scratch"
                            )
                            resume_byte_pos = 0
                            response.close()
                            part_path.unlink(missing_ok=True)
                            validator_path.unlink(missing_ok=True)
                            continue
                        aprint(f"✓ Resuming from {resume_byte_pos / (1024**2):.1f} MB")
                        mode = "ab"  # Append mode
                    elif resume_byte_pos > 0:
                        if sent_if_range:
                            # The If-Range validator did not match: the remote
                            # representation changed since the partial was
                            # written, and the server correctly sent the full
                            # body instead of a tail to splice.
                            aprint(
                                "⚠️  Remote content changed since the partial was "
                                "written; restarting download"
                            )
                        else:
                            aprint(
                                "⚠️  Server doesn't support resume, restarting download"
                            )
                        resume_byte_pos = 0
                        mode = "wb"
                        # Overwriting the migrated bytes → no longer a trustworthy
                        # pre-existing cache to auto-restore.
                        migrated_from_output = False
                    else:
                        mode = "wb"

                    if mode == "wb":
                        # (Re)starting the staging file from scratch: record THIS
                        # response's validator so a later resume of these bytes
                        # can be checked against the remote representation.
                        # Discard the condemned bytes FIRST: the `"wb"` open below
                        # is what truncates them, so if anything between here and
                        # there fails (Ctrl-C, or the open itself hitting ENOSPC /
                        # EMFILE) the old bytes would survive in `.part` paired
                        # with a validator that vouches for the NEW
                        # representation — and the next run would resume them,
                        # get a 206, and splice `old_prefix + new_tail` into a
                        # corrupt file that passes size verification.
                        part_path.unlink(missing_ok=True)
                        _record_part_validator(response.headers)

                    # Get total size. Route both headers through the same hardened
                    # parsers the resume probe uses, so a duplicated
                    # ``Content-Length: "100, 100"`` or a ``Content-Range: .../*``
                    # degrades to "unknown size" instead of raising ValueError into
                    # the generic handler and aborting the download non-retryably.
                    content_length = _parse_len(response.headers)
                    content_range_total = _parse_content_range_total(response.headers)
                    if content_length is not None:
                        total_size = content_length + resume_byte_pos
                    elif content_range_total is not None:
                        # For resumed downloads: "bytes start-end/total"
                        total_size = content_range_total
                    else:
                        total_size = 0

                    if total_size > 0:
                        aprint(f"📦 Total size: {total_size / (1024**3):.2f} GB")
                    else:
                        aprint(
                            "📦 Size: Unknown (no usable Content-Length or "
                            "Content-Range header)"
                        )

                    # Download with progress
                    downloaded = resume_byte_pos
                    last_progress_mb = downloaded / (1024 * 1024)
                    start_time = time.time()

                    with open(part_path, mode) as f:
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

                    # Verify download completed (measured on the STAGING file, which
                    # holds the bytes just written — output_path is untouched until
                    # the atomic promotion below).
                    final_size = part_path.stat().st_size
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

                    # Atomically promote the fully-downloaded, size-verified staging
                    # file onto the canonical path (same directory → atomic rename,
                    # overwriting any stale file already there). Only now does
                    # output_path exist, and it is COMPLETE by construction.
                    promoted = True
                    os.replace(part_path, output_path)
                    validator_path.unlink(missing_ok=True)
                    return output_path

            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError,
            ) as e:
                attempt += 1
                if attempt <= max_retries:
                    # #721: this attempt may have written more bytes to the staging
                    # file before dropping. Refresh the resume offset from its ACTUAL
                    # size so the retry's `Range: bytes=<offset>-` matches what is on
                    # disk — otherwise the append-mode retry re-requests (and appends)
                    # a byte range already written, silently duplicating it. Only when
                    # resuming (resume_byte_pos > 0); a fresh "wb" download that drops
                    # keeps resume_byte_pos == 0 and correctly restarts from scratch.
                    if resume_byte_pos > 0 and part_path.exists():
                        resume_byte_pos = part_path.stat().st_size
                    wait_time = 2**attempt  # Exponential backoff
                    aprint(f"❌ Download error: {type(e).__name__}: {e}")
                    aprint(
                        f"   Retrying in {wait_time} seconds... (attempt {attempt}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    # Keep the staging (.part) file for the resume attempt.
                else:
                    aprint(f"❌ Download failed after {max_retries + 1} attempts")
                    # Restore a migrated pre-existing cache that was never grown
                    # (offline / dropped before any body byte) BEFORE reporting a
                    # resumable partial — a genuine, grown partial stays in `.part`.
                    _restore_migrated_cache()
                    if part_path.exists():
                        aprint(f"   Partial download saved at: {part_path}")
                        aprint(
                            f"   You can retry to resume from {part_path.stat().st_size / (1024**2):.1f} MB"
                        )
                    raise

            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                if status == 416:
                    # Range Not Satisfiable: a 416 to `Range: bytes=<resume>-`
                    # proves `part >= total` (RFC 9110), so the STAGED file is AT
                    # LEAST complete. NEVER fatal, and NEVER delete the staged bytes.
                    if resume_byte_pos > 0 and part_path.exists():
                        local_size = part_path.stat().st_size
                        # The 416 response itself carries the authoritative total in
                        # its `Content-Range: bytes */<total>` header (RFC 9110), which
                        # disambiguates a genuinely-complete staged file from a stale,
                        # oversized one — even on a chunked/dynamic host that omits
                        # Content-Length (so the earlier size probe returned None).
                        # Fall back to the earlier-resolved remote_size when absent.
                        content_range_total = (
                            _parse_content_range_total(e.response.headers)
                            if e.response is not None
                            else None
                        )
                        effective_total = (
                            content_range_total
                            if content_range_total is not None
                            else remote_size
                        )
                        if effective_total is not None:
                            if local_size == effective_total:
                                # Exact match: the staged file is complete — promote
                                # it WITHOUT truncating or re-downloading.
                                if expected_size and local_size != expected_size:
                                    aprint(
                                        f"⚠️  Warning: File size ({local_size}) doesn't "
                                        f"match expected ({expected_size})"
                                    )
                                aprint(f"✓ Download complete: {output_path}")
                                promoted = True
                                os.replace(part_path, output_path)
                                validator_path.unlink(missing_ok=True)
                                return output_path
                            if not restarted_after_416:
                                # Any size mismatch against the authoritative total
                                # (a stale/oversized staged file, or a contradictory
                                # smaller-total 416 from a misbehaving server /
                                # concurrently-truncated file): restart cleanly.
                                aprint(
                                    "⚠️  Range not satisfiable (416); staged partial "
                                    f"size ({local_size}) doesn't match the remote "
                                    f"({effective_total}) — restarting from scratch"
                                )
                                resume_byte_pos = 0
                                restarted_after_416 = True
                                # Discarding the migrated bytes → no longer a
                                # trustworthy pre-existing cache to auto-restore.
                                migrated_from_output = False
                                # Discard the proven-stale staged bytes NOW rather
                                # than via the next iteration's "wb" open: if the
                                # re-fetch fails before that open, the condemned
                                # bytes would survive in `.part` and a LATER run
                                # could resume onto them (e.g. after the remote
                                # grows past their length) and splice a corrupt
                                # file that passes size verification.
                                part_path.unlink(missing_ok=True)
                                validator_path.unlink(missing_ok=True)
                                continue
                        else:
                            # No total anywhere (a truly header-less 416, no
                            # Content-Range and no resolved remote size): a 416 still
                            # proves `part >= total`, so the staged file is AT LEAST
                            # complete — promote it, mirroring the pre-loop complete
                            # path's expected_size mismatch warning.
                            if expected_size and local_size != expected_size:
                                aprint(
                                    f"⚠️  Warning: File size ({local_size}) doesn't "
                                    f"match expected ({expected_size})"
                                )
                            aprint(f"✓ Download complete: {output_path}")
                            promoted = True
                            os.replace(part_path, output_path)
                            validator_path.unlink(missing_ok=True)
                            return output_path
                    # 416 with no Range, no staged file, or after we already restarted
                    # once is a genuine error — surface it without touching the file
                    # (but restore a migrated, never-grown pre-existing cache first).
                    aprint(f"❌ HTTP error: {e}")
                    _restore_migrated_cache()
                    raise
                aprint(f"❌ HTTP error: {e}")
                # Restore a migrated pre-existing cache that was never grown (an HTTP
                # error is raised at raise_for_status, before any body byte is
                # written, so a migrated staging file is still byte-identical). Under
                # `.part` staging we NEVER create output_path on an error path, so
                # there is no THIS-call file to clean up here.
                _restore_migrated_cache()
                raise

            except Exception as e:
                aprint(f"❌ Unexpected error: {type(e).__name__}: {e}")
                # Keep the staging (.part) file - a grown partial might be resumable.
                # Restore a migrated pre-existing cache that was never grown (the
                # helper no-ops when the staging file grew, e.g. a size-mismatch
                # ValueError or a mid-write OSError, leaving the partial in .part).
                _restore_migrated_cache()
                raise

    finally:
        # BaseException (Ctrl-C during the retry backoff sleep) backstop: no
        # `except` clause catches a KeyboardInterrupt, so restore the untouched
        # migrated cache here too (no-op once promoted / already restored).
        if not promoted:
            _restore_migrated_cache()

    # Should never reach here
    raise RuntimeError("Download failed after all retry attempts")


def verify_file_checksum(
    file_path: Path,
    expected_md5: Optional[str] = None,
    expected_sha256: Optional[str] = None,
    verbose: bool = True,
) -> bool:
    """Verify file integrity using checksums.

    Args:
        file_path: Path to file to verify
        expected_md5: Expected MD5 hash (optional)
        expected_sha256: Expected SHA256 hash (optional)
        verbose: Emit the "Verifying…/Computing…/verified" progress lines. Set
            False for a quiet check — ``ensure_dataset(verbose=False)`` re-hashes
            every warm cache hit, and the header/line-per-file noise is pure spam
            for a caller that asked for silence.

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

    with (
        asection(f"Verifying {file_path.name}") if verbose else contextlib.nullcontext()
    ):
        if expected_md5:
            if verbose:
                aprint("Computing MD5...")
            # Integrity check of a downloaded artifact, not a security control:
            # usedforsecurity=False documents intent and clears bandit B324.
            md5_hash = hashlib.md5(usedforsecurity=False)
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192 * 128), b""):
                    md5_hash.update(chunk)
            actual_md5 = md5_hash.hexdigest()

            if actual_md5 == expected_md5:
                if verbose:
                    aprint(f"✓ MD5 verified: {actual_md5}")
            else:
                if verbose:
                    aprint("❌ MD5 mismatch!")
                    aprint(f"   Expected: {expected_md5}")
                    aprint(f"   Actual:   {actual_md5}")
                return False

        if expected_sha256:
            if verbose:
                aprint("Computing SHA256...")
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192 * 128), b""):
                    sha256_hash.update(chunk)
            actual_sha256 = sha256_hash.hexdigest()

            if actual_sha256 == expected_sha256:
                if verbose:
                    aprint(f"✓ SHA256 verified: {actual_sha256}")
            else:
                if verbose:
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
        extra_headers: Extra HTTP headers for every request.
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
