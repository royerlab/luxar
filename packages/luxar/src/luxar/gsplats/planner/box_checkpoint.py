"""Per-box checkpoints for content-tiled fits: finished boxes survive a kill.

A content fit (``gsplat fit --tiling content``) fits its plan's boxes one by one
(or ``-j N`` at a time). Without checkpoints a finished box lived only in memory
(sequential) or in a staging directory the next run wiped (parallel), so killing
a multi-hour fit at box 90 of 100 cost all 90. Each finished box is now written
to ``<output>.boxes/box_{i}.gsplats.zarr`` (or a ``.empty`` marker for a 0-splat
box) followed by a ``box_{i}.done.json`` marker; a rerun skips every box whose
marker matches the current run's fingerprint, and the directory is removed only
after the final output is saved.

The marker is written AFTER the box store, atomically (temp file + rename), so a
kill mid-save leaves a box without a marker -- refitted, never half-read; a box
whose marker is present but whose store cannot be read is discarded and refitted
too (:meth:`BoxCheckpoint.load_or_discard`). The fingerprint
(:func:`plan_fingerprint`) covers the luxar version, the plan, the per-box fit
configuration, the input file's identity (resolved path, size, mtime) and a
strided digest of the volume, so a changed plan, changed fit knobs, an upgraded
fitter or a different input never reuse stale boxes.

Two runs targeting the same output must not share boxes mid-run (#1040), so
:meth:`BoxCheckpoint.lock` takes an exclusive ``flock`` on ``.lock`` (released
by the kernel when the holder dies, so there is no stale-lock guessing); a run
that finds it held falls back to a private staging directory (no resume) rather
than touching the other run's boxes. ``LUXAR_CHECKPOINT_FORCE_UNLOCK=1`` breaks
a lock that is known to be dead (e.g. left on a network filesystem by a host
that went away).
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import os
import shutil
import socket
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Optional

import numpy as np
from arbol import aprint

if TYPE_CHECKING:
    from luxar.gsplats.gsplat_data import GSplatData

    from .spec import FitPlan

#: Marker schema version; bump to invalidate every existing checkpoint.
_CHECKPOINT_VERSION = 1
#: Volume elements hashed into the fingerprint (strided sample).
_VOLUME_DIGEST_SAMPLES = 1 << 20
_LOCK_NAME = ".lock"
#: What a torn / truncated / foreign box store raises on load.
#: numcodecs raises RuntimeError on a corrupted compressed chunk ("error during
#: blosc decompression"); MemoryError is not a RuntimeError subclass.
_UNREADABLE_STORE_ERRORS = (
    OSError,
    ValueError,
    KeyError,
    TypeError,
    IndexError,
    RuntimeError,
)
#: Set to 1 to break a checkpoint lock known to be dead.
FORCE_UNLOCK_ENV = "LUXAR_CHECKPOINT_FORCE_UNLOCK"


def _force_unlock_requested() -> bool:
    """Whether ``LUXAR_CHECKPOINT_FORCE_UNLOCK`` asks to break a held lock."""
    return os.environ.get(FORCE_UNLOCK_ENV, "").strip().lower() in ("1", "true", "yes")


class CheckpointBusy(RuntimeError):
    """Another live process holds the checkpoint directory's lock."""


def checkpoint_dir_for(output: Path) -> Path:
    """The stable per-output box checkpoint directory, ``<output>.boxes``."""
    output = Path(output)
    return output.parent / f"{output.name}.boxes"


def _volume_digest(volume: Any) -> str:
    """SHA-256 of the shape, dtype and a <= 1 M-element strided sample of ``volume``."""
    v = np.asarray(volume)
    flat = v.reshape(-1)
    step = max(1, -(-flat.size // _VOLUME_DIGEST_SAMPLES))
    sample = np.ascontiguousarray(flat[::step])
    h = hashlib.sha256()
    h.update(repr((v.shape, str(v.dtype))).encode())
    h.update(sample.tobytes())
    return h.hexdigest()


def _source_identity(source: Path) -> str:
    """Resolved path + size + mtime (per top-level entry for a directory store)."""
    path = Path(source).resolve()
    try:
        st = path.stat()
    except OSError:
        return str(path)
    parts = [str(path), str(st.st_size), str(st.st_mtime_ns)]
    if path.is_dir():
        with contextlib.suppress(OSError):
            for entry in sorted(os.scandir(path), key=lambda e: e.name):
                est = entry.stat()
                parts.append(f"{entry.name}:{est.st_size}:{est.st_mtime_ns}")
    return "|".join(parts)


def plan_fingerprint(
    plan: "FitPlan",
    *,
    fit_kwargs: Optional[Mapping[str, Any]] = None,
    volume: Any = None,
    source: "Optional[Path]" = None,
) -> str:
    """A digest identifying "these boxes, fitted this way, from this volume".

    ``fit_kwargs`` is folded in via ``repr`` of its sorted items, so any knob
    that changes a box's fit (iterations, loss, the resolved floor, the
    normalization range, ...) changes the fingerprint. ``device`` and
    ``verbose`` are excluded: they change how a box is computed, not what it
    represents. ``source`` (the input file) adds its resolved path, size and
    mtime; ``volume`` adds a strided content digest; the luxar version is
    always included.
    """
    from dataclasses import asdict

    from luxar import __version__

    h = hashlib.sha256()
    h.update(f"v{_CHECKPOINT_VERSION} luxar {__version__}".encode())
    h.update(json.dumps(asdict(plan), sort_keys=True, default=str).encode())
    if fit_kwargs is not None:
        items = sorted(
            (k, repr(v))
            for k, v in fit_kwargs.items()
            if k not in ("device", "verbose")
        )
        h.update(repr(items).encode())
    if source is not None:
        h.update(_source_identity(source).encode())
    if volume is not None:
        h.update(_volume_digest(volume).encode())
    return h.hexdigest()


class BoxCheckpoint:
    """Finished-box store for one content fit (see the module docstring)."""

    def __init__(self, directory: Path, fingerprint: str) -> None:
        """Checkpoint in ``directory`` for boxes fitted under ``fingerprint``."""
        self.directory = Path(directory)
        self.fingerprint = fingerprint
        self._lock_fd: Optional[int] = None

    # -- paths ---------------------------------------------------------------
    def box_path(self, i: int) -> Path:
        """Store path of box ``i`` (``box_{i}.gsplats.zarr``)."""
        return self.directory / f"box_{i}.gsplats.zarr"

    def empty_marker(self, i: int) -> Path:
        """Marker a 0-splat box writes instead of a store (the writer rejects empty ones)."""
        return Path(str(self.box_path(i)) + ".empty")

    def done_marker(self, i: int) -> Path:
        """JSON marker recording that box ``i`` finished, and under which fingerprint."""
        return self.directory / f"box_{i}.done.json"

    # -- state ---------------------------------------------------------------
    def is_done(self, i: int) -> bool:
        """True when box ``i`` finished under THIS fingerprint and is on disk."""
        try:
            meta = json.loads(self.done_marker(i).read_text())
        except (OSError, ValueError):
            return False
        if meta.get("fingerprint") != self.fingerprint or meta.get("box") != i:
            return False
        return self.box_path(i).exists() or self.empty_marker(i).exists()

    def discard(self, i: int) -> None:
        """Remove every trace of box ``i`` (marker first, then the data)."""
        self.done_marker(i).unlink(missing_ok=True)
        self.empty_marker(i).unlink(missing_ok=True)
        shutil.rmtree(self.box_path(i), ignore_errors=True)

    def mark_done(self, i: int, n_splats: int) -> None:
        """Atomically record box ``i`` as finished (call AFTER its data is saved)."""
        payload = {
            "version": _CHECKPOINT_VERSION,
            "fingerprint": self.fingerprint,
            "box": i,
            "n_splats": int(n_splats),
        }
        tmp = self.done_marker(i).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, self.done_marker(i))

    def save_region(self, i: int, region: "GSplatData") -> None:
        """Persist a freshly fitted box, then mark it done."""
        self.directory.mkdir(parents=True, exist_ok=True)
        self.discard(i)
        if region.n_splats == 0:
            self.empty_marker(i).write_text("")  # writer rejects empty stores
        else:
            region.save(self.box_path(i), amplitude_bits="auto")
        self.mark_done(i, int(region.n_splats))

    def load_region(self, i: int) -> "Optional[GSplatData]":
        """Reload a finished box (``None`` for a 0-splat box)."""
        from luxar.gsplats.gsplat_data import GSplatData

        if not self.box_path(i).exists():
            return None
        return GSplatData.load(self.box_path(i), include_stats=True)

    def load_or_discard(self, i: int) -> "tuple[bool, Optional[GSplatData]]":
        """``(True, region)`` for a readable finished box, else discard it.

        A box marked done whose store no longer reads (truncated by a full
        disk, edited by hand, ...) is discarded and reported ``(False, None)``
        so the caller refits it instead of failing on every rerun.
        """
        try:
            return True, self.load_region(i)
        except _UNREADABLE_STORE_ERRORS:
            # Read/format failures only (zarr's errors subclass ValueError);
            # MemoryError / KeyboardInterrupt still propagate.
            self.discard(i)
            return False, None

    def remove(self) -> None:
        """Delete the whole checkpoint directory (call after the final output is saved)."""
        shutil.rmtree(self.directory, ignore_errors=True)

    # -- locking ---------------------------------------------------------------
    @property
    def lock_path(self) -> Path:
        """The ``.lock`` file :meth:`lock` takes an ``flock`` on."""
        return self.directory / _LOCK_NAME

    def lock(self) -> None:
        """Take the directory's exclusive lock (release with :meth:`unlock`).

        An ``flock`` on an open descriptor: the kernel releases it when the
        holder exits or dies, so a crashed fit never leaves a stale lock and
        there is no owner-file parsing to race on. Raises
        :class:`CheckpointBusy` (naming the holder, the lock path and how to
        clear it) when another process holds it. With
        ``LUXAR_CHECKPOINT_FORCE_UNLOCK=1`` a HELD lock is broken -- loudly --
        by replacing the lock file. A filesystem without ``flock`` support
        (ENOLCK / ENOSYS / EOPNOTSUPP on some Lustre / NFS mounts), or a
        platform without ``fcntl`` (Windows), proceeds WITHOUT a lock, with a
        warning that concurrent fits to the same output are then unprotected.
        """
        try:
            import fcntl
        except ImportError:  # pragma: no cover - Windows
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            holder = os.pread(fd, 256, 0).decode(errors="replace").strip()
            os.close(fd)
            if exc.errno not in (errno.EWOULDBLOCK, errno.EAGAIN):
                aprint(
                    f"⚠ Cannot lock {self.lock_path} ({exc.strerror or exc}): "
                    "this filesystem does not support flock. Continuing WITHOUT "
                    "a checkpoint lock -- do not run two fits to the same "
                    "output at once."
                )
                return
            if not _force_unlock_requested():
                raise CheckpointBusy(
                    f"{self.directory} is in use by another fit "
                    f"(holder: {holder or 'unknown'}). If no such fit is running, "
                    f"remove {self.lock_path} or rerun with {FORCE_UNLOCK_ENV}=1"
                ) from None
            aprint(
                f"⚠⚠ {FORCE_UNLOCK_ENV} set: BREAKING the lock on "
                f"{self.directory} held by {holder or 'an unknown process'}. If "
                "that fit is still running, both will write the same boxes."
            )
            self.lock_path.unlink(missing_ok=True)
            fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                # A third process locked the fresh file first.
                os.close(fd)
                raise CheckpointBusy(
                    f"{self.directory} was re-locked by another fit while "
                    "breaking its lock"
                ) from None
        os.ftruncate(fd, 0)
        os.pwrite(fd, f"{socket.gethostname()} {os.getpid()}".encode(), 0)
        self._lock_fd = fd

    def unlock(self) -> None:
        """Release the lock if held (the lock file itself is left in place).

        Never unlinked here: a process already blocked on this inode would
        otherwise lock a deleted file while a third locks a new one.
        """
        if self._lock_fd is None:
            return
        fd, self._lock_fd = self._lock_fd, None
        with contextlib.suppress(OSError):
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            os.close(fd)

    @contextlib.contextmanager
    def acquire(self) -> Iterator["BoxCheckpoint"]:
        """:meth:`lock` for the duration of a ``with`` block."""
        self.lock()
        try:
            yield self
        finally:
            self.unlock()


__all__ = [
    "BoxCheckpoint",
    "CheckpointBusy",
    "FORCE_UNLOCK_ENV",
    "checkpoint_dir_for",
    "plan_fingerprint",
]
