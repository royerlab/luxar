# mypy: warn-unreachable=False
# Reason: the availability probe early-returns on `sys.platform != "darwin"`;
# mypy narrows sys.platform per host, so the macOS branch is "unreachable" on
# Linux and reachable on macOS. Same file-scope rationale as metal/__init__.py.

"""Runtime-compiled Metal splatting kernels (no Xcode, no compiled extension).

The SAME ``src/kernels.metal`` the ahead-of-time extension ships is compiled
on first use by ``torch.mps.compile_shader``, which hands the source to the
Metal compiler built into macOS itself. Nothing here shells out to ``xcrun``,
so a ``pip install "luxar[gsplats]"`` on an Apple-silicon Mac gets the fast
kernels without Xcode and without a binary coupled to one torch ABI.

The functions mirror the AOT extension's contract exactly
(``forward_splat_3d`` / ``backward_splat_3d`` / ``compute_conic_metal``: same
arguments, validation, return shapes, output zeroing), so this module is a
drop-in for ``metal_splatting_backend``.

Two binding details matter and are easy to get wrong:

* a Python int or list binds as **int64** by default. The kernels read
  ``constant uint3& shape_dhw`` and ``constant uint& n_splats``, so an int64
  ``[D, H, W]`` is read as ``(D, 0, H)`` -- a silent wrong render. Every
  integer is therefore bound as 32-bit via ``arg_casts`` (supported by both
  the public torch >= 2.7 API and the private torch 2.6 ``_compile_shader``).
* the forward/backward kernels are splat-centric: ONE threadgroup of 64
  threads per splat, i.e. ``dispatchThreadgroups(N, 64)``. That is
  ``threads=N * 64, group_size=64`` here. The kernels index the splat by
  ``threadgroup_position_in_grid`` with no offset argument, so one dispatch
  must cover all splats and ``N * 64`` must fit a 32-bit thread count: the
  runtime path caps N at :data:`MAX_SPLATS` (~67.1M). The AOT extension's
  ``dispatchThreadgroups`` counts threadgroups, so it accepts up to 2**32 - 1.

Kernels are enqueued on torch's own MPS stream, so unlike the AOT bindings
there is no ``torch.mps.synchronize()`` + ``waitUntilCompleted`` per call.
macOS caches compiled shader binaries itself; the compiled library is
memoized per process.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any, Callable, Sequence

#: The kernel source -- shared verbatim with the AOT extension's ``setup.py``.
KERNELS_SOURCE = Path(__file__).parent / "src" / "kernels.metal"

#: Must equal ``THREADGROUP_SIZE`` in kernels.metal (the AOT ``kThreadgroupSize``).
THREADGROUP_SIZE = 64

#: Same hard cap as the AOT ``validate_shape_3d`` (MET-3: per-splat AABB
#: voxel counters are 32-bit inside the kernels).
MAX_VOXELS = 2_000_000_000

#: Integers are bound as int32 (see the module docstring), so cap at int32.
MAX_INT32 = 2**31 - 1

#: Splat-centric dispatch launches ``N * THREADGROUP_SIZE`` threads in one grid
#: whose size must fit 32 bits (see the module docstring).
MAX_SPLATS = (2**32 - 1) // THREADGROUP_SIZE


class MetalCompileError(RuntimeError):
    """``kernels.metal`` failed to compile with the runtime Metal compiler."""


def _shader_compiler() -> Callable[[str], Any] | None:
    """The torch shader compiler, or None when this torch has none.

    ``torch.mps.compile_shader`` (public) from torch 2.7; the private
    ``torch.mps._compile_shader`` in 2.6 (same kernel call API, ``arg_casts``
    included); nothing earlier.
    """
    import torch

    public = getattr(torch.mps, "compile_shader", None)
    if public is not None:
        return public  # type: ignore[no-any-return]
    private: Callable[[str], Any] | None = getattr(torch.mps, "_compile_shader", None)
    return private


def runtime_unavailable_reason() -> str | None:
    """Why the runtime backend cannot run here, or None when it can.

    Cheap: does not compile anything (see :func:`prepare` for that).
    """
    if sys.platform != "darwin":
        return "Metal is only available on macOS"
    try:
        import torch
    except ImportError:
        return "PyTorch is not installed"
    if not torch.backends.mps.is_available():
        return "PyTorch cannot access an MPS device"
    compiler = _shader_compiler()
    if compiler is None:
        return (
            f"torch {torch.__version__} has no torch.mps.compile_shader "
            "(runtime Metal compilation needs torch >= 2.7, or 2.6 via its "
            "private API)"
        )
    if not KERNELS_SOURCE.is_file():
        return f"Metal kernel source missing: {KERNELS_SOURCE}"
    return None


_library: Any = None
_lock = threading.Lock()


def prepare() -> None:
    """Compile ``kernels.metal`` once per process (thread-safe; idempotent).

    Raises
    ------
    RuntimeError
        The runtime backend cannot run here (see :func:`runtime_unavailable_reason`).
    MetalCompileError
        The Metal compiler rejected the source.
    """
    _get_library()


def _get_library() -> Any:
    """The compiled shader library, compiling it on first call (double-checked lock)."""
    global _library
    if _library is not None:
        return _library
    with _lock:
        if _library is not None:
            return _library
        reason = runtime_unavailable_reason()
        if reason is not None:
            raise RuntimeError(f"Runtime Metal backend unavailable: {reason}")
        compiler = _shader_compiler()
        assert compiler is not None  # guaranteed by runtime_unavailable_reason()
        source = KERNELS_SOURCE.read_text(encoding="utf-8")
        try:
            library = compiler(source)
        except Exception as exc:  # noqa: BLE001 -- torch raises a bare RuntimeError
            raise MetalCompileError(
                f"Failed to compile {KERNELS_SOURCE.name} with the runtime Metal "
                f"compiler: {exc}"
            ) from exc
        _library = library
        return _library


def reset_for_tests() -> None:
    """Forget the compiled library (tests only)."""
    global _library
    with _lock:
        _library = None


def _launch(
    kernel_name: str,
    args: list[Any],
    uint_indices: Sequence[int],
    *,
    threads: int,
    group_size: int,
) -> None:
    """Dispatch one kernel, binding ``args[i]`` for i in uint_indices as 32-bit."""
    library = _get_library()
    kernel = getattr(library, kernel_name)
    kernel(
        *args,
        threads=threads,
        group_size=group_size,
        arg_casts={i: "int32" for i in uint_indices},
    )


# =============================================================================
# Validation (ported from bindings.mm)
# =============================================================================


def check_splat_count(n: int) -> None:
    """Reject splat counts the single splat-centric dispatch cannot cover."""
    if n > MAX_SPLATS:
        raise ValueError(
            f"Runtime Metal kernels support at most {MAX_SPLATS} splats per call "
            f"(got {n}): the splat-centric dispatch launches N * {THREADGROUP_SIZE} "
            "threads in one 32-bit grid. Split the splats into batches, or build "
            "the AOT extension (LUXAR_METAL_BACKEND=aot), which counts threadgroups."
        )


def _validate_shape_3d(shape: Sequence[int]) -> list[int]:
    """Validate a ``(D, H, W)`` shape like ``validate_shape_3d`` in bindings.mm."""
    dims = [int(s) for s in shape]
    if len(dims) != 3:
        raise ValueError("3D Metal kernels require shape (D, H, W)")
    if any(s <= 0 for s in dims):
        raise ValueError("shape dimensions must be positive")
    total = dims[0] * dims[1] * dims[2]
    if total > MAX_VOXELS:
        raise ValueError(
            f"Volume D*H*W={total} exceeds Metal backend's per-splat AABB capacity "
            f"({MAX_VOXELS} voxels). This is a hard cap to prevent silent splat "
            "drops; reduce volume size or tile the workload."
        )
    return dims


def _check_mps_float32_contiguous(names: Sequence[str], tensors: Sequence[Any]) -> None:
    """Require MPS, float32 and contiguous, naming the offending tensor."""
    import torch

    for name, t in zip(names, tensors, strict=True):
        if t.device.type != "mps":
            raise ValueError(f"{name} must be on MPS device, got {t.device}")
        if t.dtype != torch.float32:
            raise TypeError(f"{name} must be float32, got {t.dtype}")
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")


def _validate_splat_tensors_3d(
    centers: Any, conic: Any, amps: Any, shape: Sequence[int]
) -> list[int]:
    """Validate splat tensors + shape like ``validate_splat_tensors_3d``."""
    dims = _validate_shape_3d(shape)
    _check_mps_float32_contiguous(("centers", "conic", "amps"), (centers, conic, amps))
    if centers.dim() != 2 or centers.shape[1] != 3:
        raise ValueError("centers must have shape (N, 3)")
    if conic.dim() != 2 or conic.shape[1] != 6:
        raise ValueError("conic must have shape (N, 6)")
    if amps.dim() != 1:
        raise ValueError("amps must have shape (N,)")
    n = centers.shape[0]
    if conic.shape[0] != n or amps.shape[0] != n:
        raise ValueError("centers, conic, and amps batch dimensions must match")
    check_splat_count(n)
    return dims


# =============================================================================
# Kernel entry points (same contract as the AOT metal_splatting_backend)
# =============================================================================


def compute_conic_metal(Ls: Any) -> Any:
    """Packed 3D conic (Σ⁻¹) from ``(N, 3, 3)`` Cholesky factors, ``[Z, Y, X]`` order."""
    import torch

    _check_mps_float32_contiguous(("Ls",), (Ls,))
    if Ls.dim() != 3 or Ls.shape[1] != 3 or Ls.shape[2] != 3:
        raise ValueError("Ls must have shape (N, 3, 3)")
    n = int(Ls.shape[0])
    if n > MAX_INT32:
        raise ValueError(f"compute_conic_metal supports at most {MAX_INT32} splats")
    conic = torch.empty((n, 6), device=Ls.device, dtype=torch.float32)
    if n == 0:
        return conic
    _launch(
        "compute_conic_from_L_3d",
        [Ls, conic, n],
        (2,),
        threads=n,
        group_size=min(THREADGROUP_SIZE, n),
    )
    return conic


def forward_splat_3d(
    centers: Any,
    conic: Any,
    amps: Any,
    shape: Sequence[int],
    truncate: float,
    intensity_floor: float,
) -> Any:
    """Splat-centric 3D forward pass: the ``(D, H, W)`` rendered volume."""
    import torch

    dims = _validate_splat_tensors_3d(centers, conic, amps, shape)
    # The kernel accumulates with atomics, so the output starts at zero.
    output = torch.zeros(dims, device=centers.device, dtype=torch.float32)
    n = int(centers.shape[0])
    if n == 0:
        return output
    _launch(
        "rasterize_forward_splat_centric_3d",
        [
            centers,
            conic,
            amps,
            output,
            dims,
            n,
            float(truncate),
            float(intensity_floor),
        ],
        (4, 5),
        threads=n * THREADGROUP_SIZE,
        group_size=THREADGROUP_SIZE,
    )
    return output


def backward_splat_3d(
    grad_output: Any,
    centers: Any,
    conic: Any,
    amps: Any,
    shape: Sequence[int],
    truncate: float,
    intensity_floor: float,
) -> tuple[Any, Any, Any]:
    """Splat-centric 3D backward pass: ``(d_centers, d_conic, d_amps)``."""
    import torch

    dims = _validate_splat_tensors_3d(centers, conic, amps, shape)
    _check_mps_float32_contiguous(("grad_output",), (grad_output,))
    if tuple(grad_output.shape) != tuple(dims):
        raise ValueError("grad_output shape must match the provided 3D shape")
    n = int(centers.shape[0])
    opts: dict[str, Any] = {"device": centers.device, "dtype": torch.float32}
    if n == 0:
        return (
            torch.zeros((0, 3), **opts),
            torch.zeros((0, 6), **opts),
            torch.zeros((0,), **opts),
        )
    # Every element is written by its splat's threadgroup (tid 0 after the
    # reduction), so no zeroing is needed -- same as the AOT bindings.
    d_centers = torch.empty((n, 3), **opts)
    d_conic = torch.empty((n, 6), **opts)
    d_amps = torch.empty((n,), **opts)
    _launch(
        "rasterize_backward_splat_centric_3d",
        [
            grad_output,
            centers,
            conic,
            amps,
            d_centers,
            d_conic,
            d_amps,
            dims,
            n,
            float(truncate),
            float(intensity_floor),
        ],
        (7, 8),
        threads=n * THREADGROUP_SIZE,
        group_size=THREADGROUP_SIZE,
    )
    return d_centers, d_conic, d_amps
