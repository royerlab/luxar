"""Robust download utilities with retry logic, resume capability, and progress tracking.

This module provides production-grade download functionality for large datasets,
including automatic retry on failure, partial download resume, and integrity verification.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from arbol import aprint, asection

if TYPE_CHECKING:
    import requests

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


def _make_host_scoped_session(
    original_url: str, sensitive_headers: Iterable[str]
) -> requests.Session:
    """Build a session that scopes caller credentials to the original URL.

    ``requests`` strips only the ``Authorization`` header when a redirect crosses
    to a different host; it leaves arbitrary custom headers (an ``api-key``, an
    ``x-api-key``, a Zenodo/S3 bearer token, …) in place, so a credential-bearing
    ``extra_headers`` entry would leak to whatever host a redirect points at. The
    returned session drops every *sensitive_headers* entry whenever requests
    would strip ``Authorization`` from a redirect (host, port, or unsafe scheme
    change), closing that leak while preserving requests' upgrade carve-out.

    Args:
        original_url: Original request URL used as the redirect-policy anchor.
        sensitive_headers: Header names to scope to the original URL (typically
            the caller's ``extra_headers`` keys); compared case-insensitively.
    """
    import requests

    sensitive = {str(name).lower() for name in sensitive_headers}

    class _HostScopedSession(requests.Session):
        def rebuild_auth(self, prepared_request: Any, response: Any) -> None:  # noqa: ANN401 - matches requests' signature
            # Preserve requests' own cross-host Authorization stripping first.
            super().rebuild_auth(prepared_request, response)
            if not sensitive:
                return
            if self.should_strip_auth(original_url, prepared_request.url):
                headers = prepared_request.headers
                for name in [k for k in headers if k.lower() in sensitive]:
                    del headers[name]

    return _HostScopedSession()


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
        extra_headers: Extra HTTP headers scoped to the original URL (e.g. an
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
        >>> from luxar.demos import robust_download
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
    # call site IS the re-fetch, so telling the user to re-download would name
    # the thing already happening — the only thing left for them to do is
    # reclaim the space the rejected copy is holding. Keep the wording neutral
    # about HOW the replacement arrives: whether this run downloads fresh or
    # resumes a `.part` staging file is only determined later (the `.part` probe
    # below), so asserting "from scratch" here could contradict the resume
    # notice ("Attempting to resume...") emitted downstream (#716).
    warn_if_quarantined(
        output_path,
        action=(
            "delete the quarantined copy to reclaim its disk space — this run "
            "is already fetching a replacement"
        ),
    )

    # Set up session with retry logic. Scope caller-supplied credential headers
    # (extra_headers) with requests' own Authorization redirect policy so a
    # host, port, or unsafe scheme change cannot leak arbitrary custom headers
    # (an api-key, an x-api-key, a Zenodo/S3 token, …).
    session = _make_host_scoped_session(url, (extra_headers or {}).keys())
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
            for a caller that asked for silence. False silences the MISMATCH
            report too, not just the progress lines, so a caller that asks for
            silence owns surfacing the False return. Both in-tree callers
            quarantine the rejected file, which leaves it discoverable as a
            ``.corrupt`` sibling (``warn_if_quarantined`` / ``luxar demo cache
            list``) even when nothing was printed here.

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
