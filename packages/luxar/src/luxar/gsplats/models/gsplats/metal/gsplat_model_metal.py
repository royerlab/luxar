"""
Metal/MPS Gaussian splatting model for Apple Silicon.

The optimized custom Metal path currently targets the production-critical 3D
case. The public model mirrors :class:`GaussianSplatModelCUDA` and uses
Luxar's PyTorch renderer for other supported MPS dimensions, so callers can use
the same model-management interface on MPS and CUDA systems.
"""

from __future__ import annotations

import builtins
from typing import Any, Optional, Sequence, Tuple, cast

import numpy as np
import torch

from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel
from luxar.gsplats.models.gsplats.rendering_core import render_gaussians
from luxar.typing_utils.constants import DEFAULT_TRUNCATION_RADIUS

try:
    import metal_splatting_backend

    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False


def cholesky_to_conic(L: torch.Tensor) -> torch.Tensor:
    """Convert Cholesky factors to packed inverse-covariance form.

    Parameters
    ----------
    L:
        ``(N, d, d)`` lower-triangular Cholesky factors where
        ``Σ = L @ L.T``.

    Returns
    -------
    torch.Tensor
        ``(N, d * (d + 1) // 2)`` packed upper-triangular elements of
        ``Σ⁻¹`` in row-major order. For 3D this is
        ``[c00, c01, c02, c11, c12, c22]`` in the input coordinate order
        (Luxar volumes use ``[Z, Y, X]``).
    """
    if L.ndim != 3 or L.shape[1] != L.shape[2]:
        raise ValueError(f"Expected L with shape (N, d, d), got {tuple(L.shape)}")

    n_splats, dim, _ = L.shape
    device = L.device

    if dim == 2:
        l00 = L[:, 0, 0]
        l10 = L[:, 1, 0]
        l11 = L[:, 1, 1]

        k00 = 1.0 / (l00 + 1e-9)
        k11 = 1.0 / (l11 + 1e-9)
        k10 = -l10 * k00 * k11

        c00 = k00 * k00 + k10 * k10
        c01 = k10 * k11
        c11 = k11 * k11
        return torch.stack([c00, c01, c11], dim=1)

    if dim == 3:
        l00 = L[:, 0, 0]
        l10 = L[:, 1, 0]
        l11 = L[:, 1, 1]
        l20 = L[:, 2, 0]
        l21 = L[:, 2, 1]
        l22 = L[:, 2, 2]

        k00 = 1.0 / (l00 + 1e-9)
        k11 = 1.0 / (l11 + 1e-9)
        k22 = 1.0 / (l22 + 1e-9)
        k10 = -l10 * k00 * k11
        k21 = -l21 * k11 * k22
        k20 = -(l20 * k00 + l21 * k10) * k22

        c00 = k00 * k00 + k10 * k10 + k20 * k20
        c01 = k10 * k11 + k20 * k21
        c02 = k20 * k22
        c11 = k11 * k11 + k21 * k21
        c12 = k21 * k22
        c22 = k22 * k22
        return torch.stack([c00, c01, c02, c11, c12, c22], dim=1)

    eye = torch.eye(dim, device=device, dtype=L.dtype).expand(n_splats, -1, -1)
    l_inv = torch.linalg.solve_triangular(L, eye, upper=False)
    sigma_inv = l_inv.transpose(-2, -1) @ l_inv
    rows, cols = torch.triu_indices(dim, dim, device=device)
    result: torch.Tensor = sigma_inv[:, rows, cols]
    return result


class MetalSplatFunction(torch.autograd.Function):
    """Custom autograd function for the splat-centric 3D Metal renderer."""

    @staticmethod
    def forward(
        ctx: Any,
        centers: torch.Tensor,
        Ls: torch.Tensor,
        amps: torch.Tensor,
        shape: Tuple[int, ...],
        truncate: float,
        intensity_floor: float = 1e-5,
        use_metal_conic: bool = False,
    ) -> torch.Tensor:
        dim = len(shape)
        if not METAL_AVAILABLE:
            raise RuntimeError("Metal splatting backend is not available")
        if dim != 3:
            raise ValueError(f"MetalSplatFunction requires 3D volumes, got {dim}D")
        if (
            centers.device.type != "mps"
            or Ls.device.type != "mps"
            or amps.device.type != "mps"
        ):
            raise ValueError(
                "MetalSplatFunction requires MPS tensors, got "
                f"centers={centers.device}, Ls={Ls.device}, amps={amps.device}"
            )

        ctx.shape = tuple(int(s) for s in shape)
        ctx.truncate = float(truncate)
        ctx.intensity_floor = float(intensity_floor)

        if (
            centers.dtype != torch.float32
            or Ls.dtype != torch.float32
            or amps.dtype != torch.float32
        ):
            raise TypeError(
                "Metal custom kernels require float32 centers, Ls, and amps"
            )

        # L -> conic. The splat-centric kernels use Luxar/PyTorch's native
        # [Z, Y, X] coordinate order and row-major packed upper triangle. We
        # compute the conic ONCE per forward (via the Metal kernel when
        # ``use_metal_conic`` is set, otherwise via the PyTorch helper) so the
        # rasterization kernel does not redo the Cholesky → conic math
        # per-splat-per-thread.
        Ls_for_conic = Ls.detach().clone().requires_grad_(True)
        if use_metal_conic:
            conic_zyx = metal_splatting_backend.compute_conic_metal(
                Ls.contiguous()
            ).detach()
        else:
            conic_zyx = cholesky_to_conic(Ls_for_conic).detach().contiguous()

        output = cast(
            torch.Tensor,
            metal_splatting_backend.forward_splat_3d(
                centers.contiguous(),
                conic_zyx.contiguous(),
                amps.contiguous(),
                list(shape),
                float(truncate),
                float(intensity_floor),
            ),
        )
        if output.shape != torch.Size(shape):
            output = output.view(shape)

        ctx.save_for_backward(centers, Ls_for_conic, conic_zyx, amps)
        return output

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], ...]:
        centers, Ls_for_conic, conic_zyx, amps = ctx.saved_tensors

        d_centers, d_conic_zyx, d_amps = metal_splatting_backend.backward_splat_3d(
            grad_output.contiguous(),
            centers.contiguous(),
            conic_zyx.contiguous(),
            amps.contiguous(),
            list(ctx.shape),
            ctx.truncate,
            ctx.intensity_floor,
        )

        # Chain rule: Metal returns d(conic) in the same [Z, Y, X] packed
        # order as cholesky_to_conic(), so no coordinate reorder is needed.
        with torch.enable_grad():
            conic_recomputed_zyx = cholesky_to_conic(Ls_for_conic)
        (d_Ls,) = torch.autograd.grad(
            outputs=conic_recomputed_zyx,
            inputs=Ls_for_conic,
            grad_outputs=d_conic_zyx,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )

        return d_centers, d_Ls, d_amps, None, None, None, None


class GaussianSplatModelMetal(GaussianSplatModel):
    """Gaussian splat model with an optimized Metal/MPS rendering backend.

    The constructor mirrors :class:`GaussianSplatModelCUDA` where practical. The
    custom Metal kernel is used for 3D MPS tensors. For 2D and 4D-8D MPS
    shapes, the model uses Luxar's PyTorch renderer while preserving the same
    parameter-management interface.
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
        use_fp16: bool = False,
        use_metal_conic: bool = False,
        voxel_size: Optional[np.ndarray] = None,
        device: Optional[torch.device | str] = None,
    ) -> None:
        dim = len(shape)
        if dim < 2 or dim > 8:
            raise ValueError(
                f"Metal backend supports 2D-8D volumes (got {dim}D). "
                "For higher dimensions, use GaussianSplatModel."
            )
        if not METAL_AVAILABLE:
            raise RuntimeError(
                "Metal extension is not loaded. Check is_metal_available() before "
                "constructing GaussianSplatModelMetal."
            )
        if use_fp16:
            raise ValueError(
                "Metal backend does not support FP16 kernels yet. "
                "Use use_fp16=False with float32 parameters."
            )
        if device is None:
            if not torch.backends.mps.is_available():
                raise ValueError(
                    "Metal backend requires an available MPS device. "
                    "For CPU, use GaussianSplatModel."
                )
            resolved_device = torch.device("mps")
        elif isinstance(device, str):
            resolved_device = torch.device(device)
        else:
            resolved_device = device

        if resolved_device.type != "mps":
            raise ValueError(
                f"Metal backend requires an MPS device (got {resolved_device}). "
                "For CPU, use GaussianSplatModel. For CUDA, use GaussianSplatModelCUDA."
            )

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
            device=resolved_device,
        )

        self._intensity_floor = float(intensity_floor)
        self._use_metal_conic = bool(use_metal_conic)
        self._use_fp16 = False

    @property
    def _uses_custom_metal(self) -> bool:
        return METAL_AVAILABLE and self.dim == 3 and self.raw_mu.device.type == "mps"

    def forward(self) -> torch.Tensor:
        centers, Ls, amps = self.current_params()
        if self._uses_custom_metal:
            # `Function.apply` is an untyped classmethod up to torch 2.13 (so
            # `no-untyped-call` fires) and an `Any` alias from 2.14 (so the
            # ignore goes unused). Both torches are in range, hence both codes.
            return cast(
                torch.Tensor,
                MetalSplatFunction.apply(  # type: ignore[no-untyped-call, unused-ignore]
                    centers,
                    Ls,
                    amps,
                    self.shape,
                    self.truncate,
                    self._intensity_floor,
                    self._use_metal_conic,
                ),
            )

        return render_gaussians(
            self.shape,
            centers,
            Ls,
            amps,
            truncate=self.truncate,
            intensity_floor=self._intensity_floor,
        )

    @staticmethod
    def _validate_mps_tensor(
        name: str, tensor: torch.Tensor, *, require_float32: bool = True
    ) -> None:
        if tensor.device.type != "mps":
            raise ValueError(
                f"Metal backend requires {name} to be on MPS (got {tensor.device})."
            )
        if require_float32 and tensor.dtype != torch.float32:
            raise ValueError(
                f"Metal backend requires {name} to be float32 (got {tensor.dtype})."
            )

    def replace_with(
        self, centers: torch.Tensor, Ls: torch.Tensor, amps: torch.Tensor
    ) -> None:
        """Replace all splats, requiring MPS float32 tensors."""
        self._validate_mps_tensor("centers", centers)
        self._validate_mps_tensor("Ls", Ls)
        self._validate_mps_tensor("amps", amps)
        super().replace_with(centers, Ls, amps)

    def append_(
        self, centers: torch.Tensor, Ls: torch.Tensor, amps: torch.Tensor
    ) -> None:
        """Append splats, requiring MPS float32 tensors."""
        self._validate_mps_tensor("centers", centers)
        self._validate_mps_tensor("Ls", Ls)
        self._validate_mps_tensor("amps", amps)
        super().append_(centers, Ls, amps)

    def prune_(self, mask: torch.Tensor) -> None:
        """Prune splats, requiring an MPS boolean keep mask."""
        self._validate_mps_tensor("mask", mask, require_float32=False)
        if mask.dtype != torch.bool:
            raise ValueError(
                f"Metal backend requires mask to be bool (got {mask.dtype})."
            )
        super().prune_(mask)

    @staticmethod
    def _validate_requested_device(device: torch.device | str | int | None) -> None:
        if device is None:
            return
        if isinstance(device, int):
            raise ValueError(
                f"Metal backend requires an MPS device (got device index {device}). "
                "For CPU, use GaussianSplatModel. For CUDA, use GaussianSplatModelCUDA."
            )
        resolved = torch.device(device)
        if resolved.type != "mps":
            raise ValueError(
                f"Metal backend requires an MPS device (got {resolved}). "
                "For CPU, use GaussianSplatModel. For CUDA, use GaussianSplatModelCUDA."
            )

    @staticmethod
    def _validate_requested_dtype(dtype: torch.dtype | None) -> None:
        if dtype is not None and dtype != torch.float32:
            raise ValueError(
                f"Metal backend requires float32 parameters (got {dtype}). "
                "FP16/BF16/FP64 Metal kernels are not implemented."
            )

    def _validate_to_args(self, *args: Any, **kwargs: Any) -> None:
        for arg in args:
            if isinstance(arg, torch.Tensor):
                self._validate_requested_device(arg.device)
                self._validate_requested_dtype(arg.dtype)
            elif isinstance(arg, torch.device):
                self._validate_requested_device(arg)
            elif isinstance(arg, str):
                self._validate_requested_device(arg)
            elif isinstance(arg, torch.dtype):
                self._validate_requested_dtype(arg)

        device_kw = kwargs.get("device")
        if device_kw is not None:
            self._validate_requested_device(device_kw)
        dtype_kw = kwargs.get("dtype")
        if dtype_kw is not None:
            self._validate_requested_dtype(dtype_kw)

    def to(self, *args: Any, **kwargs: Any) -> "GaussianSplatModelMetal":
        """Move the model while preserving Metal's MPS/float32 invariants."""
        self._validate_to_args(*args, **kwargs)
        super().to(*args, **kwargs)
        return self

    def to_empty(
        self, *, device: torch.device | str | int | None, recurse: bool = True
    ) -> "GaussianSplatModelMetal":
        """Move storage while preserving Metal's MPS-device invariant."""
        self._validate_requested_device(device)
        super().to_empty(device=device, recurse=recurse)
        return self

    def type(self, dst_type: Any) -> "GaussianSplatModelMetal":
        """Tensor-type migration is disabled to preserve MPS/float32 invariants."""
        _ = dst_type
        raise ValueError(
            "Metal backend requires MPS float32 tensors; use .to('mps') or .float()."
        )

    def cpu(self) -> "GaussianSplatModelMetal":
        """CPU tensors are not valid for the Metal backend."""
        raise ValueError(
            "Metal backend requires an MPS device; use GaussianSplatModel for CPU."
        )

    def cuda(self, device: Any = None) -> "GaussianSplatModelMetal":
        """CUDA tensors are not valid for the Metal backend."""
        _ = device
        raise ValueError(
            "Metal backend requires an MPS device; use GaussianSplatModelCUDA for CUDA."
        )

    def half(self) -> "GaussianSplatModelMetal":
        """FP16 tensors are not valid for the current Metal kernels."""
        raise ValueError(
            "Metal backend requires float32 parameters; FP16 is unsupported."
        )

    def bfloat16(self) -> "GaussianSplatModelMetal":
        """BF16 tensors are not valid for the current Metal kernels."""
        raise ValueError(
            "Metal backend requires float32 parameters; BF16 is unsupported."
        )

    def double(self) -> "GaussianSplatModelMetal":
        """FP64 tensors are not valid for the current Metal kernels."""
        raise ValueError(
            "Metal backend requires float32 parameters; FP64 is unsupported."
        )

    def float(self) -> "GaussianSplatModelMetal":
        """Keep the model in the supported float32 dtype."""
        super().float()
        return self

    @property
    def use_fp16(self) -> bool:
        return self._use_fp16

    @property
    def intensity_floor(self) -> builtins.float:
        return self._intensity_floor

    @property
    def use_metal_conic(self) -> bool:
        return self._use_metal_conic

    def __repr__(self) -> str:
        backend = "metal" if self._uses_custom_metal else "pytorch"
        return (
            "GaussianSplatModelMetal("
            f"n_splats={self.n_splats()}, "
            f"shape={self.shape}, "
            f"device={next(self.parameters()).device}, "
            f"backend={backend})"
        )
