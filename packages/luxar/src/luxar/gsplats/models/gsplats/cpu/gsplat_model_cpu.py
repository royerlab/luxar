"""Gaussian splat model whose 3D CPU render runs on fused numba kernels.

:class:`GaussianSplatModelCPUNative` is :class:`GaussianSplatModel` with its
``forward`` routed through :class:`CpuSplatFunction`, a
``torch.autograd.Function`` whose forward and analytic backward are the numba
kernels in :mod:`.kernels`. Parameters, constraints (``_build_L``, sigmoid
centers, softplus amplitudes) and the optimizer are unchanged -- autograd
carries the gradient from ``(centers, L, amps)`` back to the raw parameters.

The kernels cover the production 3D case; any other dimensionality uses the
PyTorch renderer, as the Metal model does.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, cast

import numpy as np
import torch
from arbol import aprint

from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel
from luxar.gsplats.models.gsplats.rendering_core import render_gaussians
from luxar.typing_utils.constants import DEFAULT_TRUNCATION_RADIUS

#: Whether the one non-finite-splat notice has been printed in this process.
_WARNED_NONFINITE = False

#: Target number of forward work items (z-plane x row band) per thread.
_ITEMS_PER_THREAD = 8


def _kernels() -> Any:
    """The kernel module, or RuntimeError naming why it cannot be used."""
    from .backend import get_cpu_kernels

    kernels = get_cpu_kernels()
    if kernels is None:
        from .backend import cpu_native_unavailable_reason

        raise RuntimeError(
            "Native CPU splatting kernels are not available "
            f"({cpu_native_unavailable_reason()})."
        )
    return kernels


def _row_band(depth: int, height: int) -> int:
    """Rows per forward work item, so there are enough items to balance."""
    import numba

    items_wanted = _ITEMS_PER_THREAD * int(numba.get_num_threads())  # type: ignore[no-untyped-call]
    bands_wanted = max(1, -(-items_wanted // max(1, depth)))
    return max(1, -(-height // bands_wanted))


def _floor_args(intensity_floor: Optional[float]) -> Tuple[bool, float]:
    """``(enabled, value)`` for the kernels' amplitude-aware box shrink."""
    on = intensity_floor is not None and intensity_floor > 0
    return on, float(intensity_floor) if on and intensity_floor else 0.0


def _boxes(
    kernels: Any,
    shape: Sequence[int],
    centers: np.ndarray,
    Ls: np.ndarray,
    amps: np.ndarray,
    truncate: float,
    intensity_floor: Optional[float],
) -> np.ndarray:
    """Per-splat AABBs ``(N, 6)`` exactly as the PyTorch renderer bounds them.

    Non-finite splats get an empty box; the first time any appear, one line
    says how many (an optimizer that produced NaNs is worth knowing about, but
    not once per render).
    """
    global _WARNED_NONFINITE  # noqa: PLW0603
    floor_on, floor_value = _floor_args(intensity_floor)
    boxes, nonfinite = kernels.compute_boxes(
        centers,
        Ls,
        amps,
        int(shape[0]),
        int(shape[1]),
        int(shape[2]),
        float(truncate),
        floor_on,
        floor_value,
    )
    if not _WARNED_NONFINITE:
        count = int(nonfinite.sum())
        if count:
            _WARNED_NONFINITE = True
            aprint(
                f"WARNING: native CPU kernels skipped {count} splat(s) with "
                "non-finite parameters (they render nothing); further "
                "occurrences are not reported."
            )
    return cast(np.ndarray, boxes)


def cpu_render_numpy(
    shape: Sequence[int],
    centers: np.ndarray,
    Ls: np.ndarray,
    amps: np.ndarray,
    truncate: float = DEFAULT_TRUNCATION_RADIUS,
    intensity_floor: Optional[float] = 1e-5,
) -> np.ndarray:
    """Render ``(N, 3)`` splats into a ``shape`` volume with the numba kernels.

    Same contract and numbers as :func:`render_gaussians` for 3D; the output
    dtype follows ``centers`` (float32 in fitting, float64 for gradchecks).
    """
    from .backend import LAUNCH_LOCK, configure_threads

    if len(shape) != 3:
        raise ValueError(f"native CPU kernels render 3D volumes, got {len(shape)}D")
    kernels = _kernels()
    out = np.zeros(tuple(int(s) for s in shape), dtype=centers.dtype)
    if centers.shape[0] == 0:
        return out
    import numba

    with LAUNCH_LOCK:
        configure_threads(numba)
        boxes = _boxes(kernels, shape, centers, Ls, amps, truncate, intensity_floor)
        start, splats = kernels.plane_lists(boxes, int(shape[0]))
        band = _row_band(int(shape[0]), int(shape[1]))
        with numba.parallel_chunksize(1):
            kernels.forward_3d(
                centers, Ls, amps, boxes, start, splats, out, float(truncate), band
            )
    return out


def cpu_render_backward_numpy(
    grad_output: np.ndarray,
    centers: np.ndarray,
    Ls: np.ndarray,
    amps: np.ndarray,
    truncate: float = DEFAULT_TRUNCATION_RADIUS,
    intensity_floor: Optional[float] = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytic ``(d_centers, d_Ls, d_amps)`` of ``sum(grad_output * render)``."""
    from .backend import LAUNCH_LOCK, configure_threads

    kernels = _kernels()
    d_centers = np.zeros_like(centers)
    d_Ls = np.zeros_like(Ls)
    d_amps = np.zeros_like(amps)
    if centers.shape[0] == 0:
        return d_centers, d_Ls, d_amps
    import numba

    with LAUNCH_LOCK:
        configure_threads(numba)
        boxes = _boxes(
            kernels, grad_output.shape, centers, Ls, amps, truncate, intensity_floor
        )
        # Splats differ widely in cost; small chunks let idle threads steal.
        with numba.parallel_chunksize(16):
            kernels.backward_3d(
                grad_output,
                centers,
                Ls,
                amps,
                boxes,
                d_centers,
                d_Ls,
                d_amps,
                float(truncate),
            )
    return d_centers, d_Ls, d_amps


def _as_numpy(tensor: torch.Tensor, dtype: torch.dtype) -> np.ndarray:
    """Zero-copy contiguous numpy view of a detached CPU tensor."""
    return tensor.detach().to(dtype).contiguous().numpy()


class CpuSplatFunction(torch.autograd.Function):
    """Autograd bridge: numba forward, numba analytic backward (3D, CPU)."""

    @staticmethod
    def forward(
        ctx: Any,
        centers: torch.Tensor,
        Ls: torch.Tensor,
        amps: torch.Tensor,
        shape: Tuple[int, ...],
        truncate: float,
        intensity_floor: Optional[float] = 1e-5,
    ) -> torch.Tensor:
        """Render through the numba forward; saves the inputs for backward."""
        if len(shape) != 3:
            raise ValueError(f"CpuSplatFunction requires 3D volumes, got {len(shape)}D")
        for name, tensor in (("centers", centers), ("Ls", Ls), ("amps", amps)):
            if tensor.device.type != "cpu":
                raise ValueError(
                    f"CpuSplatFunction requires CPU tensors ({name} is on "
                    f"{tensor.device})"
                )
        dtype = torch.float64 if centers.dtype == torch.float64 else torch.float32
        ctx.shape = tuple(int(s) for s in shape)
        ctx.truncate = float(truncate)
        ctx.intensity_floor = intensity_floor
        ctx.dtype = dtype
        out = cpu_render_numpy(
            ctx.shape,
            _as_numpy(centers, dtype),
            _as_numpy(Ls, dtype),
            _as_numpy(amps, dtype),
            ctx.truncate,
            intensity_floor,
        )
        ctx.save_for_backward(centers, Ls, amps)
        return torch.from_numpy(out)

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """Analytic gradients for ``(centers, Ls, amps)``; None for the rest."""
        centers, Ls, amps = ctx.saved_tensors
        d_centers, d_Ls, d_amps = cpu_render_backward_numpy(
            _as_numpy(grad_output, ctx.dtype),
            _as_numpy(centers, ctx.dtype),
            _as_numpy(Ls, ctx.dtype),
            _as_numpy(amps, ctx.dtype),
            ctx.truncate,
            ctx.intensity_floor,
        )
        return (
            torch.from_numpy(d_centers).to(centers.dtype),
            torch.from_numpy(d_Ls).to(Ls.dtype),
            torch.from_numpy(d_amps).to(amps.dtype),
            None,
            None,
            None,
        )


def render_gaussians_cpu_native(
    shape: Sequence[int],
    centers: torch.Tensor,
    Ls: torch.Tensor,
    amps: torch.Tensor,
    truncate: float = DEFAULT_TRUNCATION_RADIUS,
    intensity_floor: Optional[float] = 1e-5,
) -> torch.Tensor:
    """Differentiable 3D CPU render through the numba kernels."""
    return cast(
        torch.Tensor,
        CpuSplatFunction.apply(  # type: ignore[no-untyped-call, unused-ignore]
            centers, Ls, amps, tuple(shape), truncate, intensity_floor
        ),
    )


class GaussianSplatModelCPUNative(GaussianSplatModel):
    """Gaussian splat model rendered by fused numba kernels on the CPU.

    Same constructor and parameterization as :class:`GaussianSplatModel`
    (always on the CPU). 3D renders go through :class:`CpuSplatFunction`;
    other dimensionalities use the PyTorch renderer.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        centers0: np.ndarray,
        L0: np.ndarray,
        amps0: np.ndarray,
        sigma_min_diag: Sequence[float],
        sigma_max_diag: Optional[Sequence[float]] = None,
        amp_max: Optional[float] = None,
        max_eccentricity: Optional[float] = None,
        truncate: float = DEFAULT_TRUNCATION_RADIUS,
        intensity_floor: float = 1e-5,
        voxel_size: Optional[np.ndarray] = None,
        device: Optional[torch.device | str] = None,
    ) -> None:
        """Build the model on the CPU; raises if the kernels are unavailable."""
        resolved = torch.device(device) if device is not None else torch.device("cpu")
        if resolved.type != "cpu":
            raise ValueError(
                f"GaussianSplatModelCPUNative requires the CPU (got {resolved})."
            )
        _kernels()  # fail at construction, not at the first forward
        super().__init__(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=sigma_min_diag,
            sigma_max_diag=sigma_max_diag,
            amp_max=amp_max,
            max_eccentricity=max_eccentricity,
            truncate=truncate,
            voxel_size=voxel_size,
            device=resolved,
        )
        self._intensity_floor = float(intensity_floor)

    @property
    def kernel_backend(self) -> str:
        """Kernel backend name, as reported in the fit's backend summary."""
        return "numba"

    @property
    def intensity_floor(self) -> float:
        """Amplitude-aware truncation floor used by every render."""
        return self._intensity_floor

    @property
    def _uses_native(self) -> bool:
        """True when renders go through the numba kernels (3D on the CPU)."""
        return self.dim == 3 and self.raw_mu.device.type == "cpu"

    def forward(self) -> torch.Tensor:
        """Render all splats (numba kernels in 3D, PyTorch renderer otherwise)."""
        centers, Ls, amps = self.current_params()
        if self._uses_native:
            return render_gaussians_cpu_native(
                self.shape, centers, Ls, amps, self.truncate, self._intensity_floor
            )
        return render_gaussians(
            self.shape,
            centers,
            Ls,
            amps,
            truncate=self.truncate,
            intensity_floor=self._intensity_floor,
        )

    def __repr__(self) -> str:
        """Splat count, shape and which renderer is active."""
        backend = "numba" if self._uses_native else "pytorch"
        return (
            "GaussianSplatModelCPUNative("
            f"n_splats={self.n_splats()}, shape={self.shape}, backend={backend})"
        )
