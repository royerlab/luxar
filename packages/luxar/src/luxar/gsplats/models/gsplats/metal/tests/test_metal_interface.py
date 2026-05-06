"""Interface-parity tests for the Metal Gaussian splat model."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from luxar.gsplats.models.gsplats.metal import is_metal_available

pytestmark = pytest.mark.skipif(
    not is_metal_available(), reason="Metal backend not available"
)

if is_metal_available():
    from luxar.gsplats.models.gsplats.metal import GaussianSplatModelMetal
else:
    GaussianSplatModelMetal = None  # type: ignore[misc, assignment]


class TestMetalCudaInterfaceParity:
    """Verify the Metal model exposes the CUDA/base-model management API."""

    def test_two_dimensional_model_uses_pytorch_renderer(self) -> None:
        model = GaussianSplatModelMetal(
            shape=(8, 8),
            centers0=np.array([[4.0, 4.0]], dtype=np.float32),
            L0=np.array([np.eye(2)], dtype=np.float32),
            amps0=np.array([1.0], dtype=np.float32),
            sigma_min_diag=[0.2, 0.2],
            device="mps",
        )

        output = model()
        assert output.shape == (8, 8)
        assert output.device.type == "mps"
        assert output.max() > 0

        output.sum().backward()
        assert all(param.grad is not None for param in model.parameters())

    def test_four_dimensional_model_uses_pytorch_renderer(self) -> None:
        model = GaussianSplatModelMetal(
            shape=(4, 4, 4, 4),
            centers0=np.array([[2.0, 2.0, 2.0, 2.0]], dtype=np.float32),
            L0=np.array([np.eye(4)], dtype=np.float32),
            amps0=np.array([1.0], dtype=np.float32),
            sigma_min_diag=[0.2, 0.2, 0.2, 0.2],
            device="mps",
        )

        output = model()
        assert output.shape == (4, 4, 4, 4)
        assert output.device.type == "mps"
        assert output.max() > 0

        output.sum().backward()
        assert all(param.grad is not None for param in model.parameters())

    def test_rejects_cpu_device(self) -> None:
        with pytest.raises(ValueError, match="requires an MPS device"):
            GaussianSplatModelMetal(
                shape=(8, 8, 8),
                centers0=np.array([[4.0, 4.0, 4.0]], dtype=np.float32),
                L0=np.array([np.eye(3)], dtype=np.float32),
                amps0=np.array([1.0], dtype=np.float32),
                sigma_min_diag=[0.2, 0.2, 0.2],
                device="cpu",
            )

    def test_to_cpu_and_non_float32_dtype_are_rejected(self) -> None:
        model = GaussianSplatModelMetal(
            shape=(8, 8, 8),
            centers0=np.array([[4.0, 4.0, 4.0]], dtype=np.float32),
            L0=np.array([np.eye(3)], dtype=np.float32),
            amps0=np.array([1.0], dtype=np.float32),
            sigma_min_diag=[0.2, 0.2, 0.2],
            device="mps",
        )

        assert model.to("mps") is model
        assert model.float() is model
        with pytest.raises(ValueError, match="requires an MPS device"):
            model.to("cpu")
        with pytest.raises(ValueError, match="requires an MPS device"):
            model.cpu()
        with pytest.raises(ValueError, match="float32"):
            model.to(torch.float64)
        with pytest.raises(ValueError, match="FP16"):
            model.half()
        with pytest.raises(ValueError, match="requires an MPS device"):
            model.to_empty(device="cpu")
        with pytest.raises(ValueError, match="MPS float32"):
            model.type(torch.FloatTensor)
        with pytest.raises(ValueError, match="centers to be on MPS"):
            model.append_(torch.zeros((1, 3)), torch.eye(3).unsqueeze(0), torch.ones(1))
        with pytest.raises(ValueError, match="mask to be on MPS"):
            model.prune_(torch.tensor([True]))

    def test_rejects_dimensions_outside_cuda_range(self) -> None:
        dim = 9
        with pytest.raises(ValueError, match="2D-8D"):
            GaussianSplatModelMetal(
                shape=(2,) * dim,
                centers0=np.zeros((1, dim), dtype=np.float32),
                L0=np.array([np.eye(dim)], dtype=np.float32),
                amps0=np.array([1.0], dtype=np.float32),
                sigma_min_diag=[0.2] * dim,
                device="mps",
            )

    def test_rejects_fp16_until_kernels_support_it(self) -> None:
        with pytest.raises(ValueError, match="does not support FP16"):
            GaussianSplatModelMetal(
                shape=(8, 8, 8),
                centers0=np.array([[4.0, 4.0, 4.0]], dtype=np.float32),
                L0=np.array([np.eye(3)], dtype=np.float32),
                amps0=np.array([1.0], dtype=np.float32),
                sigma_min_diag=[0.2, 0.2, 0.2],
                use_fp16=True,
                device="mps",
            )

    def test_replace_append_prune_and_state_dict_match_base_interface(self) -> None:
        model = GaussianSplatModelMetal(
            shape=(8, 8, 8),
            centers0=np.array([[4.0, 4.0, 4.0]], dtype=np.float32),
            L0=np.array([np.eye(3)], dtype=np.float32),
            amps0=np.array([1.0], dtype=np.float32),
            sigma_min_diag=[0.2, 0.2, 0.2],
            amp_max=1.0,
            max_eccentricity=4.0,
            device="mps",
        )

        assert model.n_splats() == 1
        assert model.amp_max == 1.0
        assert model.max_eccentricity == 4.0
        assert model.use_fp16 is False

        centers, Ls, amps = model.current_params()
        model.append_(centers, Ls, amps)
        assert model.n_splats() == 2

        keep_mask = torch.tensor([True, False], device="mps")
        model.prune_(keep_mask)
        assert model.n_splats() == 1

        new_centers = torch.tensor([[3.0, 3.0, 3.0]], device="mps")
        new_Ls = torch.eye(3, device="mps").unsqueeze(0)
        new_amps = torch.tensor([0.5], device="mps")
        model.replace_with(new_centers, new_Ls, new_amps)
        assert model.n_splats() == 1

        state = model.state_dict()
        assert "raw_mu" in state
        assert all(not key.startswith("_base.") for key in state)

        output = model()
        assert output.shape == (8, 8, 8)

    def test_prune_all_splats_renders_zero_volume(self) -> None:
        model = GaussianSplatModelMetal(
            shape=(8, 8, 8),
            centers0=np.array([[4.0, 4.0, 4.0]], dtype=np.float32),
            L0=np.array([np.eye(3)], dtype=np.float32),
            amps0=np.array([1.0], dtype=np.float32),
            sigma_min_diag=[0.2, 0.2, 0.2],
            device="mps",
        )

        model.prune_(torch.tensor([False], device="mps"))
        output = model()
        output.sum().backward()

        assert model.n_splats() == 0
        assert output.shape == (8, 8, 8)
        assert output.sum().item() == 0.0

    def test_nested_module_state_dict_uses_clean_parameter_names(self) -> None:
        class Wrapper(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = GaussianSplatModelMetal(
                    shape=(8, 8, 8),
                    centers0=np.array([[4.0, 4.0, 4.0]], dtype=np.float32),
                    L0=np.array([np.eye(3)], dtype=np.float32),
                    amps0=np.array([1.0], dtype=np.float32),
                    sigma_min_diag=[0.2, 0.2, 0.2],
                    device="mps",
                )

        wrapper = Wrapper()
        state = wrapper.state_dict()
        assert "model.raw_mu" in state
        assert all("._base." not in key for key in state)
        assert all("._base." not in name for name, _ in wrapper.named_parameters())

        clone = Wrapper()
        clone.load_state_dict(state)
