# mypy: warn-unreachable=False
# Reason: availability probes early-return on `sys.platform != "darwin"`; mypy
# narrows sys.platform per host. Same file-scope rationale as metal/__init__.py.

"""Selection of the Metal splatting kernel backend.

Three ways to run the 3D splatting forward/backward on an Apple-silicon GPU,
tried in this order:

1. ``aot``     -- the ahead-of-time torch extension (``metal_splatting_backend``)
   built explicitly with ``python setup.py build_ext --inplace --force`` in this
   directory (needs Xcode). Kept first so dev checkouts that built it behave
   as before. A build older than its sources is skipped, never used stale.
2. ``runtime`` -- the same ``kernels.metal`` compiled on first use by
   ``torch.mps.compile_shader`` (:mod:`.runtime_backend`), i.e. by the Metal
   compiler that ships with macOS. Needs torch >= 2.7 (2.6 via its private
   API): no Xcode, no compiled extension, no torch-ABI coupling.
3. ``torch``   -- no custom kernels; the pure-PyTorch renderer on MPS.

``LUXAR_METAL_BACKEND=aot|runtime|torch`` forces one (``auto``/unset = the
order above). A forced kernel backend that cannot run raises instead of
silently degrading -- forcing exists for testing and benchmarking, where a
quiet fallback would report the wrong backend's numbers.

Nothing here builds anything: importing Luxar never compiles into the package
directory (which, for a pip install, is site-packages).
"""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from arbol import aprint

BACKEND_ENV = "LUXAR_METAL_BACKEND"
VALID_CHOICES = ("auto", "aot", "runtime", "torch")

_METAL_DIR = Path(__file__).parent
_SRC_DIR = _METAL_DIR / "src"
# Filter compiled extensions to the running Python's ABI tag so a stale build
# from a different interpreter (e.g. cpython-311 alongside cpython-312) is not
# picked up; loading it would fail at import time with a confusing error.
_PYTHON_ABI_TAG = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
_EXTENSION_PATTERN = f"metal_splatting_backend.{_PYTHON_ABI_TAG}-*.so"

#: How to build the optional AOT extension (a developer action, never automatic).
AOT_BUILD_HINT = (
    "build it with `python setup.py build_ext --inplace --force` in "
    "packages/luxar/src/luxar/gsplats/models/gsplats/metal (needs full Xcode)"
)


@dataclass(frozen=True)
class MetalKernelBackend:
    """A usable kernel backend; both expose the AOT extension's three entry points."""

    name: str  # "aot" or "runtime"
    forward_splat_3d: Callable[..., Any]
    backward_splat_3d: Callable[..., Any]
    compute_conic_metal: Callable[..., Any]


# =============================================================================
# AOT extension discovery (no building)
# =============================================================================


def find_extension() -> Path | None:
    """The newest compiled extension matching the active Python ABI, if any."""
    extensions = list(_METAL_DIR.glob(_EXTENSION_PATTERN))
    if extensions:
        return max(extensions, key=lambda path: path.stat().st_mtime)
    return None


def _extension_sources() -> list[Path]:
    """Source files whose change makes the compiled extension stale."""
    return [_METAL_DIR / "setup.py", _SRC_DIR / "bindings.mm"]


def extension_is_stale(extension: Path) -> bool:
    """True when compiled Metal artifacts are older than their sources."""
    extension_mtime = extension.stat().st_mtime
    if any(
        src.exists() and src.stat().st_mtime > extension_mtime
        for src in _extension_sources()
    ):
        return True

    metallib = _SRC_DIR / "default.metallib"
    kernels = _SRC_DIR / "kernels.metal"
    return not metallib.exists() or (
        kernels.exists() and kernels.stat().st_mtime > metallib.stat().st_mtime
    )


def _validate_mps_interop() -> str | None:
    """Check the torch internals bindings.mm relies on; None when they hold.

    The AOT extension reinterprets ``untyped_storage().data_ptr()`` as an
    ``MTLBuffer`` and applies ``storage_offset``. Only the AOT path needs this.
    """
    try:
        import torch

        t = torch.randn(10, device="mps")
        if not t.untyped_storage().data_ptr():
            return "untyped_storage().data_ptr() returned null"
        t_slice = torch.randn(100, device="mps")[25:75]
        if t_slice.storage_offset() != 25:
            return "storage_offset not working correctly"
    except Exception as exc:  # noqa: BLE001 -- any failure disables the AOT path
        return f"MPS-Metal interop validation failed: {exc}"
    return None


def _import_aot() -> tuple[ModuleType | None, str]:
    """The compiled extension, or (None, why it cannot be used)."""
    extension = find_extension()
    if extension is None:
        return None, f"compiled extension not built ({AOT_BUILD_HINT})"
    if extension_is_stale(extension):
        return None, f"compiled extension {extension.name} is stale ({AOT_BUILD_HINT})"
    interop_problem = _validate_mps_interop()
    if interop_problem is not None:
        return None, interop_problem
    if str(_METAL_DIR) not in sys.path:
        sys.path.insert(0, str(_METAL_DIR))
    try:
        import metal_splatting_backend
    except (ImportError, OSError) as exc:
        return None, f"compiled extension {extension.name} failed to import: {exc}"
    module: ModuleType = metal_splatting_backend
    metallib = _SRC_DIR / "default.metallib"
    if metallib.exists() and hasattr(module, "set_library_path"):
        try:
            module.set_library_path(str(metallib))
        except RuntimeError:
            pass  # already set: a context exists, its library path is final
    return module, ""


_aot_cache: list[tuple[ModuleType | None, str]] = []


def _aot_backend() -> tuple[MetalKernelBackend | None, str]:
    """The compiled extension wrapped as a backend, or (None, why not)."""
    if not _aot_cache:
        _aot_cache.append(_import_aot())
    module, why = _aot_cache[0]
    if module is None:
        return None, why
    return (
        MetalKernelBackend(
            "aot",
            module.forward_splat_3d,
            module.backward_splat_3d,
            module.compute_conic_metal,
        ),
        "",
    )


def _runtime_backend() -> tuple[MetalKernelBackend | None, str]:
    """The runtime-compiled backend (compiled here, once), or (None, why not)."""
    from luxar.gsplats.models.gsplats.metal import runtime_backend

    reason = runtime_backend.runtime_unavailable_reason()
    if reason is not None:
        return None, reason
    try:
        runtime_backend.prepare()
    except RuntimeError as exc:  # includes MetalCompileError
        return None, str(exc)
    return (
        MetalKernelBackend(
            "runtime",
            runtime_backend.forward_splat_3d,
            runtime_backend.backward_splat_3d,
            runtime_backend.compute_conic_metal,
        ),
        "",
    )


def mps_unavailable_reason() -> str | None:
    """Why there is no MPS device to run Metal kernels on (None when there is)."""
    if sys.platform != "darwin":
        return "Metal is only available on macOS"
    try:
        import torch
    except ImportError:
        return "PyTorch is not installed"
    if not torch.backends.mps.is_available():
        return "PyTorch cannot access an MPS device"
    return None


# =============================================================================
# Selection
# =============================================================================


def requested_backend(environ: dict[str, str] | None = None) -> str:
    """The ``LUXAR_METAL_BACKEND`` choice, validated (``auto`` when unset)."""
    env = os.environ if environ is None else environ
    choice = env.get(BACKEND_ENV, "").strip().lower() or "auto"
    if choice not in VALID_CHOICES:
        raise ValueError(
            f"{BACKEND_ENV}={env.get(BACKEND_ENV)!r} is not one of "
            f"{', '.join(VALID_CHOICES)}"
        )
    return choice


_resolved: dict[str, MetalKernelBackend | None] = {}
_reasons: dict[str, str] = {}
_lock = threading.Lock()


def _resolve(choice: str) -> MetalKernelBackend | None:
    """Resolve one ``LUXAR_METAL_BACKEND`` choice (uncached; see the module docstring)."""
    if choice == "torch":
        _reasons[choice] = f"{BACKEND_ENV}=torch"
        return None
    no_mps = mps_unavailable_reason()
    if no_mps is not None:
        if choice != "auto":
            raise RuntimeError(f"{BACKEND_ENV}={choice} but {no_mps}")
        _reasons[choice] = no_mps
        return None
    aot_why = ""
    if choice in ("auto", "aot"):
        aot, aot_why = _aot_backend()
        if aot is not None:
            return aot
        if choice == "aot":
            raise RuntimeError(f"{BACKEND_ENV}=aot but the {aot_why}")
    runtime, runtime_why = _runtime_backend()
    if runtime is not None:
        return runtime
    if choice == "runtime":
        raise RuntimeError(
            f"{BACKEND_ENV}=runtime but runtime Metal compilation is unavailable: "
            f"{runtime_why}"
        )
    _reasons[choice] = (
        f"runtime Metal compilation unavailable: {runtime_why}; and the {aot_why}"
    )
    return None


_DESCRIPTIONS = {
    "aot": "compiled extension (setup.py build_ext)",
    "runtime": "compiled at runtime by macOS's Metal compiler via torch.mps",
}


def get_metal_backend() -> MetalKernelBackend | None:
    """The Metal kernel backend to use, or None for the pure-PyTorch path.

    Resolved once per ``LUXAR_METAL_BACKEND`` value and logged once; cheap
    after the first call. Resolving the runtime backend compiles
    ``kernels.metal`` (milliseconds once macOS has cached the shader), so a
    returned backend is known to work.

    Raises
    ------
    RuntimeError
        A forced backend (``aot``/``runtime``) cannot run.
    ValueError
        ``LUXAR_METAL_BACKEND`` holds an unknown value.
    """
    choice = requested_backend()
    with _lock:
        if choice in _resolved:
            return _resolved[choice]
        backend = _resolve(choice)
        _resolved[choice] = backend
    if backend is not None:
        aprint(
            f"Metal splatting kernels: {backend.name} ({_DESCRIPTIONS[backend.name]})"
        )
    elif mps_unavailable_reason() is None:
        aprint(
            f"Metal splatting kernels: none, using PyTorch ({_reasons.get(choice, '')})"
        )
    return backend


def metal_backend_unavailable_reason() -> str | None:
    """Why :func:`get_metal_backend` returned None (None when it did not)."""
    if get_metal_backend() is not None:
        return None
    return _reasons.get(requested_backend(), "unknown")


def reset_for_tests() -> None:
    """Forget resolved backends (tests only)."""
    with _lock:
        _resolved.clear()
        _reasons.clear()
        _aot_cache.clear()
