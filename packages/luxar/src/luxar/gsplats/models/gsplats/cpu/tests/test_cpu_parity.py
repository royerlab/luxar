"""Parity of the numba CPU kernels with the PyTorch reference renderer.

The kernels are meant to be the reference's numbers, fused: the same AABB, the
same clamped forward substitution, the same shifted Gaussian. So the forward is
compared voxel for voxel and the analytic backward against autograd through
:func:`render_gaussians`, plus a float64 ``gradcheck`` of the backward against
finite differences of the kernel's own forward.
"""

from __future__ import annotations

from typing import Tuple

import pytest
import torch

pytest.importorskip("numba")

from luxar.gsplats.models.gsplats.cpu import (  # noqa: E402
    GaussianSplatModelCPUNative,
    render_gaussians_cpu_native,
)
from luxar.gsplats.models.gsplats.rendering_core import (  # noqa: E402
    render_gaussians,
)

Splats = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _random_splats(
    n: int,
    shape: Tuple[int, int, int],
    sigma: Tuple[float, float],
    seed: int,
    dtype: torch.dtype = torch.float32,
    off_diag: float = 0.3,
) -> Splats:
    gen = torch.Generator().manual_seed(seed)
    centers = torch.rand(n, 3, generator=gen, dtype=dtype) * torch.tensor(
        shape, dtype=dtype
    )
    lo, hi = sigma
    diag = lo + (hi - lo) * torch.rand(n, 3, generator=gen, dtype=dtype)
    Ls = torch.diag_embed(diag)
    for i, j in ((1, 0), (2, 0), (2, 1)):
        Ls[:, i, j] = off_diag * torch.randn(n, generator=gen, dtype=dtype)
    amps = 0.1 + torch.rand(n, generator=gen, dtype=dtype)
    return centers, Ls, amps


def _assert_forward_matches(
    shape: Tuple[int, int, int], splats: Splats, **kwargs: object
) -> None:
    centers, Ls, amps = splats
    ref = render_gaussians(shape, centers, Ls, amps, **kwargs)  # type: ignore[arg-type]
    out = render_gaussians_cpu_native(shape, centers, Ls, amps, **kwargs)  # type: ignore[arg-type]
    assert out.shape == ref.shape and out.dtype == ref.dtype
    scale = max(float(ref.abs().max()), 1e-12)
    assert float((out - ref).abs().max()) <= 1e-5 * scale


def _gradients(fn: object, shape: Tuple[int, int, int], splats: Splats, weight):  # type: ignore[no-untyped-def]
    leaves = [t.detach().clone().requires_grad_(True) for t in splats]
    out = fn(shape, *leaves)  # type: ignore[operator]
    (out * weight).sum().backward()
    return [leaf.grad for leaf in leaves]


def _assert_gradients_match(shape: Tuple[int, int, int], splats: Splats) -> None:
    weight = torch.randn(shape, generator=torch.Generator().manual_seed(7))
    ref = _gradients(render_gaussians, shape, splats, weight)
    got = _gradients(render_gaussians_cpu_native, shape, splats, weight)
    for name, r, g in zip(("centers", "Ls", "amps"), ref, got, strict=True):
        scale = max(float(r.abs().max()), 1e-12)
        err = float((g - r).abs().max())
        assert err <= 1e-4 * scale, f"d{name}: max err {err:.3g} vs scale {scale:.3g}"


SHAPE = (14, 17, 19)


@pytest.mark.parametrize(
    ("seed", "n", "sigma", "off_diag"),
    [
        pytest.param(1, 40, (0.7, 2.5), 0.3, id="typical"),
        pytest.param(2, 30, (0.05, 0.3), 0.02, id="tiny_sub_voxel"),
        pytest.param(3, 3, (6.0, 12.0), 1.0, id="huge"),
        pytest.param(4, 25, (0.4, 4.0), 1.2, id="anisotropic"),
    ],
)
def test_forward_and_gradients_match_reference(
    seed: int, n: int, sigma: Tuple[float, float], off_diag: float
) -> None:
    splats = _random_splats(n, SHAPE, sigma, seed=seed, off_diag=off_diag)
    _assert_forward_matches(SHAPE, splats)
    _assert_gradients_match(SHAPE, splats)


def test_splats_on_and_beyond_the_border_match_reference() -> None:
    centers, Ls, amps = _random_splats(12, SHAPE, (1.0, 2.0), seed=3)
    corners = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [13.0, 16.0, 18.0],
            [0.0, 16.0, 9.5],
            [6.5, 0.0, 18.0],
            [-0.4, 8.0, 19.3],
            [13.9, -0.2, 4.0],
        ]
    )
    centers[: len(corners)] = corners
    _assert_forward_matches(SHAPE, (centers, Ls, amps))
    _assert_gradients_match(SHAPE, (centers, Ls, amps))


def test_intensity_floor_disabled_matches_reference() -> None:
    splats = _random_splats(20, SHAPE, (0.8, 2.0), seed=5)
    _assert_forward_matches(SHAPE, splats, intensity_floor=None)
    _assert_forward_matches(SHAPE, splats, truncate=2.0, intensity_floor=1e-3)


def test_empty_splat_set_renders_zeros_and_has_empty_gradients() -> None:
    centers = torch.zeros((0, 3), requires_grad=True)
    Ls = torch.zeros((0, 3, 3), requires_grad=True)
    amps = torch.zeros((0,), requires_grad=True)
    out = render_gaussians_cpu_native(SHAPE, centers, Ls, amps)
    assert out.shape == SHAPE and float(out.detach().abs().sum()) == 0.0
    out.sum().backward()
    assert centers.grad is not None and centers.grad.shape == (0, 3)


def test_forward_is_deterministic() -> None:
    splats = _random_splats(60, SHAPE, (0.7, 3.0), seed=11)
    first = render_gaussians_cpu_native(SHAPE, *splats)
    second = render_gaussians_cpu_native(SHAPE, *splats)
    assert torch.equal(first, second)


def test_float64_backward_passes_gradcheck() -> None:
    shape = (9, 10, 11)
    splats = _random_splats(6, shape, (0.8, 1.8), seed=13, dtype=torch.float64)
    leaves = tuple(t.detach().clone().requires_grad_(True) for t in splats)

    def render(c: torch.Tensor, L: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return render_gaussians_cpu_native(shape, c, L, a)

    assert torch.autograd.gradcheck(render, leaves, eps=1e-6, atol=1e-6, rtol=1e-4)


def test_model_forward_and_backward_match_the_pytorch_model() -> None:
    from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

    centers, Ls, amps = _random_splats(30, SHAPE, (0.8, 2.0), seed=17)
    kwargs = dict(
        shape=SHAPE,
        centers0=centers.numpy(),
        L0=Ls.numpy(),
        amps0=amps.numpy(),
        sigma_min_diag=[0.3, 0.3, 0.3],
        amp_max=2.0,
        max_eccentricity=8.0,
        device="cpu",
    )
    native = GaussianSplatModelCPUNative(**kwargs)  # type: ignore[arg-type]
    reference = GaussianSplatModel(**kwargs)  # type: ignore[arg-type]
    target = torch.rand(SHAPE, generator=torch.Generator().manual_seed(1))
    losses = []
    for model in (native, reference):
        loss = ((model() - target) ** 2).mean()
        loss.backward()
        losses.append(loss)
    assert float(losses[0]) == pytest.approx(float(losses[1]), rel=1e-5)
    for (name, p_native), p_ref in zip(
        native.named_parameters(), reference.parameters(), strict=True
    ):
        assert p_native.grad is not None and p_ref.grad is not None
        scale = max(float(p_ref.grad.abs().max()), 1e-12)
        assert float((p_native.grad - p_ref.grad).abs().max()) <= 1e-4 * scale, name
    assert native.kernel_backend == "numba"
    assert "backend=numba" in repr(native)


def test_model_renders_2d_with_the_pytorch_renderer() -> None:
    model = GaussianSplatModelCPUNative(
        shape=(12, 12),
        centers0=torch.tensor([[5.0, 6.0]]).numpy(),
        L0=torch.eye(2)[None].numpy() * 1.5,
        amps0=torch.tensor([0.8]).numpy(),
        sigma_min_diag=[0.3, 0.3],
    )
    out = model()
    assert out.shape == (12, 12) and float(out.max()) > 0.5
    assert "backend=pytorch" in repr(model)


def test_model_refuses_a_non_cpu_device() -> None:
    with pytest.raises(ValueError, match="requires the CPU"):
        GaussianSplatModelCPUNative(
            shape=(4, 4, 4),
            centers0=torch.zeros(1, 3).numpy(),
            L0=torch.eye(3)[None].numpy(),
            amps0=torch.ones(1).numpy(),
            sigma_min_diag=[0.3, 0.3, 0.3],
            device="meta",
        )
