"""Selection, threading and warm-up of the native (numba) CPU splatting kernels.

``LUXAR_CPU_BACKEND`` picks how a CPU fit renders:

* ``auto`` (default / unset) -- the numba kernels when numba imports, else the
  PyTorch renderer;
* ``native`` -- the numba kernels, raising if numba is missing (forcing exists
  for testing and benchmarking, where a quiet fallback would report the wrong
  backend's numbers);
* ``torch`` -- always the PyTorch renderer.

``LUXAR_CPU_THREADS`` sets the kernels' thread count (default: the physical
cores available to this process). numba keeps its own thread pool, so this
neither reads nor changes torch's intra-op setting.

Nothing compiles at import: the kernels compile on first use and numba's
``cache=True`` stores them -- beside the module in a writable checkout, in the
user cache directory (``~/.cache/numba`` or the platform equivalent) when the
package lives in a read-only site-packages. ``NUMBA_CACHE_DIR`` overrides both.
"""

from __future__ import annotations

import getpass
import importlib
import os
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from types import ModuleType
from typing import Any

from arbol import aprint

from luxar.gsplats.utils.device import CPU_THREADS_ENV, physical_core_count

BACKEND_ENV = "LUXAR_CPU_BACKEND"
THREADS_ENV = CPU_THREADS_ENV
VALID_CHOICES = ("auto", "native", "torch")
_KERNELS_MODULE = __name__.rsplit(".", 1)[0] + ".kernels"

# One kernel launch at a time: numba's ``workqueue`` threading layer (the only
# one available without TBB/OpenMP, e.g. on macOS) aborts the PROCESS on a
# concurrent launch from two Python threads.
LAUNCH_LOCK = threading.Lock()

_STATE_LOCK = threading.Lock()
_KERNELS: ModuleType | None = None
_UNAVAILABLE_REASON: str | None = None
_NUMBA_MISSING = False
_WARMED = False
#: Read by ``kernels.py`` at import: False once caching proved impossible.
_CACHE_ENABLED = True


@dataclass(frozen=True)
class CpuNativeStatus:
    """What a successful selection resolved to (for the model-class log line)."""

    threads: int
    threading_layer: str
    warmup_seconds: float
    from_cache: bool


def reset_for_tests() -> None:
    """Forget the cached import result, warm-up and thread count (tests only)."""
    global _KERNELS, _UNAVAILABLE_REASON, _NUMBA_MISSING, _WARMED, _CACHE_ENABLED  # noqa: PLW0603
    with _STATE_LOCK:
        _KERNELS = None
        _UNAVAILABLE_REASON = None
        _NUMBA_MISSING = False
        _WARMED = False
        _CACHE_ENABLED = True
    _thread_count_for.cache_clear()


def kernel_cache_enabled() -> bool:
    """Whether ``kernels.py`` should decorate with ``cache=True``."""
    return _CACHE_ENABLED


def requested_backend() -> str:
    """The normalized ``LUXAR_CPU_BACKEND`` choice (``auto`` when unset)."""
    raw = os.environ.get(BACKEND_ENV, "").strip().lower() or "auto"
    if raw not in VALID_CHOICES:
        raise ValueError(
            f"{BACKEND_ENV}={raw!r} is not one of {', '.join(VALID_CHOICES)}"
        )
    return raw


@lru_cache(maxsize=8)
def _thread_count_for(raw: str) -> int:
    """Thread count for one raw ``LUXAR_CPU_THREADS`` value (warns once per value)."""
    if raw and raw != "0":
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value >= 1:
            return value
        aprint(
            f"WARNING: ignoring {THREADS_ENV}={raw!r} (expected a positive "
            "integer); using the physical core count."
        )
    return physical_core_count()


def resolve_thread_count() -> int:
    """``LUXAR_CPU_THREADS`` if set (a positive int), else the physical cores."""
    return _thread_count_for(os.environ.get(THREADS_ENV, "").strip())


def configure_threads(numba_module: Any) -> int:
    """Apply the resolved thread count to numba's pool (clamped to its size)."""
    wanted = resolve_thread_count()
    count = max(1, min(wanted, int(numba_module.config.NUMBA_NUM_THREADS)))
    numba_module.set_num_threads(count)
    return count


def _temp_cache_dir() -> str:
    """A per-user numba cache directory under the system temp dir."""
    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001 # no passwd entry / no USER: any failure
        user = str(os.getuid()) if hasattr(os, "getuid") else "user"
    path = os.path.join(tempfile.gettempdir(), f"luxar-numba-cache-{user}")
    os.makedirs(path, mode=0o700, exist_ok=True)
    return path


def _use_temp_cache_dir() -> str:
    """Point numba's cache at :func:`_temp_cache_dir` (this process only)."""
    import numba

    path = _temp_cache_dir()
    config: Any = numba.config  # attributes are set from the environment
    config.CACHE_DIR = path
    return path


def _import_kernel_module() -> ModuleType:
    """Import ``kernels.py``, surviving a cache location numba cannot write.

    ``cache=True`` raises ``RuntimeError`` at decoration time when no cache
    locator works (read-only site-packages AND an unwritable or missing HOME).
    Retry with a per-user cache under the temp dir; if that fails too, compile
    without a cache rather than give up the kernels.
    """
    global _CACHE_ENABLED  # noqa: PLW0603
    try:
        return importlib.import_module(_KERNELS_MODULE)
    except RuntimeError as first:
        sys.modules.pop(_KERNELS_MODULE, None)
        try:
            path = _use_temp_cache_dir()
            module = importlib.import_module(_KERNELS_MODULE)
            aprint(
                f"Native CPU kernels: cannot cache next to the package or in the "
                f"user cache ({first}); caching in {path} instead."
            )
            return module
        except (RuntimeError, OSError) as second:
            sys.modules.pop(_KERNELS_MODULE, None)
            _CACHE_ENABLED = False
            module = importlib.import_module(_KERNELS_MODULE)
            aprint(
                "Native CPU kernels: no writable numba cache "
                f"({second}); compiling without one (every process pays the JIT)."
            )
            return module


def _import_kernels() -> ModuleType | None:
    """Import the kernel module, recording why when it cannot be used."""
    global _KERNELS, _UNAVAILABLE_REASON, _NUMBA_MISSING  # noqa: PLW0603
    with _STATE_LOCK:
        if _KERNELS is not None or _UNAVAILABLE_REASON is not None:
            return _KERNELS
        try:
            kernels = _import_kernel_module()
        except ImportError as exc:
            _NUMBA_MISSING = True
            _UNAVAILABLE_REASON = f"numba is not installed ({exc})"
            return None
        # A broken numba/llvmlite install raises more than ImportError (e.g.
        # an LLVM symbol error); any of them must fall back, not abort the fit.
        except Exception as exc:  # noqa: BLE001
            _UNAVAILABLE_REASON = f"numba failed to load the kernels ({exc!r})"
            return None
        _KERNELS = kernels
        return kernels


def numba_is_missing() -> bool:
    """True when the kernels are unavailable because numba does not import."""
    return _NUMBA_MISSING


def get_cpu_kernels() -> ModuleType | None:
    """The numba kernel module per ``LUXAR_CPU_BACKEND``, or None for torch.

    ``native`` with numba unusable raises ``RuntimeError``; ``torch`` returns
    None without importing numba.
    """
    choice = requested_backend()
    if choice == "torch":
        return None
    kernels = _import_kernels()
    if kernels is None and choice == "native":
        advice = (
            "Install numba (it ships with `pip install luxar[gsplats]`)"
            if _NUMBA_MISSING
            else "Fix the numba installation"
        )
        raise RuntimeError(
            f"{BACKEND_ENV}=native but the CPU kernels cannot run: "
            f"{_UNAVAILABLE_REASON}. {advice} or unset the variable."
        )
    return kernels


def cpu_native_unavailable_reason() -> str | None:
    """Why the native kernels are not in use (None when they are or untried)."""
    try:
        if requested_backend() == "torch":
            return f"{BACKEND_ENV}=torch"
    except ValueError as exc:
        return str(exc)
    return _UNAVAILABLE_REASON


def _warm_up(kernels: ModuleType) -> tuple[float, bool]:
    """Compile (or load from cache) the float32 kernels on a tiny problem.

    Returns ``(seconds, from_cache)``. Run once per process so the fit's
    first iteration is not the one that pays the JIT.
    """
    import numpy as np

    from .gsplat_model_cpu import cpu_render_backward_numpy, cpu_render_numpy

    centers = np.array([[2.0, 2.0, 2.0]], np.float32)
    Ls = np.eye(3, dtype=np.float32)[None] * np.float32(1.0)
    amps = np.ones(1, np.float32)
    start = time.perf_counter()
    out = cpu_render_numpy((5, 5, 5), centers, Ls, amps, 3.0, 1e-5)
    cpu_render_backward_numpy(np.ones_like(out), centers, Ls, amps, 3.0, 1e-5)
    seconds = time.perf_counter() - start
    hits = sum(
        sum(fn.stats.cache_hits.values())
        for fn in (kernels.forward_3d, kernels.backward_3d)
    )
    return seconds, hits > 0


def prepare_cpu_native() -> CpuNativeStatus | None:
    """Select, thread and warm up the native kernels; None means use torch."""
    global _WARMED  # noqa: PLW0603
    kernels = get_cpu_kernels()
    if kernels is None:
        return None
    import numba

    threads = configure_threads(numba)
    seconds, from_cache = (0.0, True)
    if not _WARMED:
        seconds, from_cache = _warm_up(kernels)
        _WARMED = True
    try:
        # getattr: numba does not re-export it for mypy (attr-defined).
        layer = str(getattr(numba, "threading_layer")())  # noqa: B009
    except ValueError:  # no parallel region has run yet
        layer = "unknown"
    return CpuNativeStatus(threads, layer, seconds, from_cache)
