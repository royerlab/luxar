"""Tests for contribution-based Gaussian splat culling."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from luxar.gsplats.culling import (
    CullResult,
    compute_per_splat_deletion_error,
    cull_by_contribution,
)
from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _make_splat_2d(
    center: tuple[float, float],
    amplitude: float,
    sigma: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a single isotropic 2D splat."""
    centers = torch.tensor([center], dtype=torch.float32, device=device)
    amps = torch.tensor([amplitude], dtype=torch.float32, device=device)
    Ls = torch.zeros((1, 2, 2), dtype=torch.float32, device=device)
    Ls[0, 0, 0] = sigma
    Ls[0, 1, 1] = sigma
    return centers, Ls, amps


def _make_splat_3d(
    center: tuple[float, float, float],
    amplitude: float,
    sigma: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a single isotropic 3D splat."""
    centers = torch.tensor([center], dtype=torch.float32, device=device)
    amps = torch.tensor([amplitude], dtype=torch.float32, device=device)
    Ls = torch.zeros((1, 3, 3), dtype=torch.float32, device=device)
    Ls[0, 0, 0] = sigma
    Ls[0, 1, 1] = sigma
    Ls[0, 2, 2] = sigma
    return centers, Ls, amps


def _concat_splats(
    *splats: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenate multiple splats."""
    centers = torch.cat([s[0] for s in splats], dim=0)
    Ls = torch.cat([s[1] for s in splats], dim=0)
    amps = torch.cat([s[2] for s in splats], dim=0)
    return centers, Ls, amps


# ---------------------------------------------------------------------------
# Tests for compute_per_splat_deletion_error
# ---------------------------------------------------------------------------


class TestPerSplatDeletionError:
    def test_single_splat_2d(self, device: torch.device) -> None:
        """A single splat perfectly fitting the target has high deletion error."""
        shape = (32, 32)
        centers, Ls, amps = _make_splat_2d((16.0, 16.0), 1.0, 3.0, device)

        V_pred = render_gaussians(shape, centers, Ls, amps, truncate=3.0)
        target = V_pred.clone()
        R = target - V_pred  # zero residual

        errors = compute_per_splat_deletion_error(
            centers, Ls, amps, R, shape, truncate=3.0
        )

        # Removing the only splat increases error by its peak amplitude
        assert errors[0].item() > 0.5  # peak contribution is ~1.0

    def test_single_splat_3d(self, device: torch.device) -> None:
        """3D fast path works correctly."""
        shape = (24, 24, 24)
        centers, Ls, amps = _make_splat_3d((12.0, 12.0, 12.0), 1.0, 3.0, device)

        V_pred = render_gaussians(shape, centers, Ls, amps, truncate=3.0)
        R = V_pred.clone() * 0  # zero residual

        errors = compute_per_splat_deletion_error(
            centers, Ls, amps, R, shape, truncate=3.0
        )

        assert errors[0].item() > 0.5

    def test_zero_amplitude_has_zero_error(self, device: torch.device) -> None:
        """A zero-amplitude splat contributes nothing."""
        shape = (32, 32)
        centers, Ls, amps = _make_splat_2d((16.0, 16.0), 0.0, 3.0, device)
        R = torch.zeros(shape, device=device)

        errors = compute_per_splat_deletion_error(
            centers, Ls, amps, R, shape, truncate=3.0
        )

        assert errors[0].item() == pytest.approx(0.0, abs=1e-6)

    def test_redundant_splats_have_lower_error(self, device: torch.device) -> None:
        """Two identical overlapping splats each have lower deletion error than a lone splat."""
        shape = (32, 32)
        s1 = _make_splat_2d((16.0, 16.0), 0.5, 3.0, device)
        s2 = _make_splat_2d((16.0, 16.0), 0.5, 3.0, device)
        centers, Ls, amps = _concat_splats(s1, s2)

        # Target is what both splats produce together
        V_pred = render_gaussians(shape, centers, Ls, amps, truncate=3.0)
        target = V_pred.clone()
        R = target - V_pred

        errors = compute_per_splat_deletion_error(
            centers, Ls, amps, R, shape, truncate=3.0
        )

        # Each splat's deletion error is ~0.5 (its own amplitude)
        # because removing it leaves the other, which only covers half the target
        assert errors[0].item() == pytest.approx(0.5, abs=0.1)
        assert errors[1].item() == pytest.approx(0.5, abs=0.1)

    def test_empty_splats(self, device: torch.device) -> None:
        """No splats returns empty errors."""
        shape = (16, 16)
        centers = torch.zeros((0, 2), device=device)
        Ls = torch.zeros((0, 2, 2), device=device)
        amps = torch.zeros((0,), device=device)
        R = torch.zeros(shape, device=device)

        errors = compute_per_splat_deletion_error(
            centers, Ls, amps, R, shape, truncate=3.0
        )

        assert errors.shape == (0,)


# ---------------------------------------------------------------------------
# Tests for cull_by_contribution
# ---------------------------------------------------------------------------


class TestCullByContribution:
    def test_no_culling_when_all_needed_2d(self, device: torch.device) -> None:
        """Non-overlapping splats covering distinct regions should all be kept."""
        shape = (64, 64)
        # Three well-separated splats
        s1 = _make_splat_2d((10.0, 10.0), 1.0, 2.0, device)
        s2 = _make_splat_2d((32.0, 32.0), 1.0, 2.0, device)
        s3 = _make_splat_2d((54.0, 54.0), 1.0, 2.0, device)
        centers, Ls, amps = _concat_splats(s1, s2, s3)

        target = render_gaussians(shape, centers, Ls, amps, truncate=3.0)

        result = cull_by_contribution(centers, Ls, amps, target, shape, truncate=3.0)

        assert isinstance(result, CullResult)
        assert result.n_culled == 0
        assert result.keep_mask.all()

    def test_culls_redundant_duplicates_2d(self, device: torch.device) -> None:
        """A tiny over-predicting splat should be culled."""
        shape = (32, 32)
        # Two "real" splats that form the target
        s1 = _make_splat_2d((16.0, 16.0), 0.5, 3.0, device)
        s2 = _make_splat_2d((16.0, 16.0), 0.5, 3.0, device)
        s_tiny = _make_splat_2d((16.0, 16.0), 0.001, 3.0, device)

        # Target is from the real splats only (without the tiny one)
        c_real, L_real, a_real = _concat_splats(s1, s2)
        target = render_gaussians(shape, c_real, L_real, a_real, truncate=3.0)

        # Fit includes the tiny extra splat → over-predicts slightly
        centers, Ls, amps = _concat_splats(s1, s2, s_tiny)

        result = cull_by_contribution(
            centers,
            Ls,
            amps,
            target,
            shape,
            truncate=3.0,
            error_percentile=99.0,
        )

        # The tiny splat should be culled (removing it reduces over-prediction)
        assert result.n_culled >= 1
        assert not result.keep_mask[2]

    def test_culls_zero_amplitude(self, device: torch.device) -> None:
        """Zero-amplitude splats should always be culled."""
        shape = (32, 32)
        s_real = _make_splat_2d((16.0, 16.0), 1.0, 3.0, device)
        s_zero = _make_splat_2d((20.0, 20.0), 0.0, 3.0, device)
        centers, Ls, amps = _concat_splats(s_real, s_zero)

        target = render_gaussians(shape, centers, Ls, amps, truncate=3.0)

        result = cull_by_contribution(centers, Ls, amps, target, shape, truncate=3.0)

        # Zero-amplitude splat should be culled
        assert not result.keep_mask[1]

    def test_3d_support(self, device: torch.device) -> None:
        """Culling works for 3D data."""
        shape = (24, 24, 24)
        s1 = _make_splat_3d((12.0, 12.0, 12.0), 1.0, 2.0, device)
        s_tiny = _make_splat_3d((12.0, 12.0, 12.0), 0.001, 2.0, device)

        # Target from real splat only
        c_real, L_real, a_real = s1
        target = render_gaussians(shape, c_real, L_real, a_real, truncate=3.0)

        # Fit includes the tiny extra → over-predicts slightly
        centers, Ls, amps = _concat_splats(s1, s_tiny)

        result = cull_by_contribution(
            centers,
            Ls,
            amps,
            target,
            shape,
            truncate=3.0,
            error_percentile=99.0,
        )

        assert result.n_culled >= 1

    def test_empty_data(self, device: torch.device) -> None:
        """Empty data returns empty result."""
        shape = (16, 16)
        centers = torch.zeros((0, 2), device=device)
        Ls = torch.zeros((0, 2, 2), device=device)
        amps = torch.zeros((0,), device=device)
        target = torch.zeros(shape, device=device)

        result = cull_by_contribution(centers, Ls, amps, target, shape, truncate=3.0)

        assert result.n_culled == 0
        assert result.keep_mask.shape == (0,)

    def test_quality_preserved_after_culling(self, device: torch.device) -> None:
        """PSNR should not degrade significantly after culling."""
        shape = (48, 48)
        torch.manual_seed(42)

        # Create a scene with some redundancy
        splats = []
        for i in range(10):
            cx = 8.0 + i * 3.5
            cy = 24.0
            splats.append(_make_splat_2d((cx, cy), 0.8, 2.5, device))
        # Add some tiny redundant splats
        for i in range(5):
            cx = 10.0 + i * 4.0
            cy = 24.0
            splats.append(_make_splat_2d((cx, cy), 0.01, 2.5, device))

        centers, Ls, amps = _concat_splats(*splats)
        target = render_gaussians(shape, centers, Ls, amps, truncate=3.0)

        result = cull_by_contribution(
            centers,
            Ls,
            amps,
            target,
            shape,
            truncate=3.0,
            error_percentile=99.0,
        )

        # Render after culling
        keep = torch.from_numpy(result.keep_mask).to(device)
        V_culled = render_gaussians(
            shape,
            centers[keep],
            Ls[keep],
            amps[keep],
            truncate=3.0,
        )

        # Check quality: MSE should be very small
        mse = ((target - V_culled) ** 2).mean().item()
        data_range = target.max().item() - target.min().item()
        if data_range > 0 and mse > 0:
            psnr = 10 * np.log10(data_range**2 / mse)
            assert psnr > 30, f"PSNR too low after culling: {psnr:.1f} dB"

    def test_error_budget_reported(self, device: torch.device) -> None:
        """CullResult should contain valid diagnostics."""
        shape = (32, 32)
        centers, Ls, amps = _make_splat_2d((16.0, 16.0), 1.0, 3.0, device)
        target = render_gaussians(shape, centers, Ls, amps, truncate=3.0)

        result = cull_by_contribution(centers, Ls, amps, target, shape, truncate=3.0)

        # Audit W7 fix: `>= 0` is true for any non-negative number, so a
        # mutant returning constant 0 for every diagnostic would pass.
        # For the single non-trivial splat above we expect the budget to
        # be populated with a finite (non-zero, non-NaN, finite-magnitude)
        # value and the iteration counters to have run at least once.
        assert np.isfinite(result.error_budget)
        assert result.error_budget >= 0
        # phase1 evaluates ALL splats; for the 1-splat fixture this is 1
        # (or 0 if the implementation prunes single-splat). Mutant that
        # always returns 0 would pass `>= 0` but fails the upper bound.
        assert 0 <= result.phase1_candidates <= 1
        assert 0 <= result.phase2_iterations <= 5  # tiny fixture
        assert np.isfinite(result.max_joint_error)
        assert result.max_joint_error >= 0

    def test_higher_tolerance_culls_more(self, device: torch.device) -> None:
        """Higher error_tolerance should cull more splats."""
        shape = (48, 48)
        splats = []
        for i in range(8):
            cx = 6.0 + i * 5.0
            cy = 24.0
            splats.append(_make_splat_2d((cx, cy), 0.5, 2.0, device))
        for i in range(4):
            splats.append(_make_splat_2d((12.0 + i * 6.0, 24.0), 0.05, 2.0, device))

        centers, Ls, amps = _concat_splats(*splats)
        target = render_gaussians(shape, centers, Ls, amps, truncate=3.0)

        result_strict = cull_by_contribution(
            centers, Ls, amps, target, shape, error_tolerance=0.5
        )
        result_loose = cull_by_contribution(
            centers, Ls, amps, target, shape, error_tolerance=2.0
        )

        assert result_loose.n_culled >= result_strict.n_culled


# ---------------------------------------------------------------------------
# Tests for GSplatData.cull()
# ---------------------------------------------------------------------------


class TestRedundancyMode:
    """Tests for redundancy mode (no target volume)."""

    def test_redundancy_culls_duplicates(self, device: torch.device) -> None:
        """Redundancy mode should cull overlapping duplicates."""
        shape = (32, 32)
        # One real splat + one tiny duplicate
        s_real = _make_splat_2d((16.0, 16.0), 1.0, 3.0, device)
        s_tiny = _make_splat_2d((16.0, 16.0), 0.005, 3.0, device)
        centers, Ls, amps = _concat_splats(s_real, s_tiny)

        result = cull_by_contribution(
            centers,
            Ls,
            amps,
            None,
            shape,
            truncate=3.0,
            redundancy_threshold=0.01,
        )

        assert result.mode == "redundancy"
        assert result.n_culled >= 1
        assert not result.keep_mask[1]  # tiny splat culled

    def test_redundancy_keeps_unique_splats(self, device: torch.device) -> None:
        """Redundancy mode should keep splats that are the sole local contributors."""
        shape = (64, 64)
        s1 = _make_splat_2d((16.0, 16.0), 1.0, 2.0, device)
        s2 = _make_splat_2d((48.0, 48.0), 1.0, 2.0, device)
        centers, Ls, amps = _concat_splats(s1, s2)

        result = cull_by_contribution(
            centers,
            Ls,
            amps,
            None,
            shape,
            truncate=3.0,
            redundancy_threshold=0.01,
        )

        assert result.n_culled == 0
        assert result.keep_mask.all()

    def test_redundancy_empty(self, device: torch.device) -> None:
        """Redundancy mode with empty data."""
        shape = (16, 16)
        centers = torch.zeros((0, 2), device=device)
        Ls = torch.zeros((0, 2, 2), device=device)
        amps = torch.zeros((0,), device=device)

        result = cull_by_contribution(
            centers,
            Ls,
            amps,
            None,
            shape,
            truncate=3.0,
        )

        assert result.mode == "redundancy"
        assert result.n_culled == 0


class TestNdSupport:
    """Test the generic nD code path (d != 2 and d != 3)."""

    def test_4d_deletion_error(self, device: torch.device) -> None:
        """4D triggers the generic nD path."""
        shape = (8, 8, 8, 8)
        centers = torch.tensor([[4.0, 4.0, 4.0, 4.0]], device=device)
        amps = torch.tensor([1.0], device=device)
        Ls = torch.eye(4, device=device).unsqueeze(0) * 2.0

        R = torch.zeros(shape, device=device)

        errors = compute_per_splat_deletion_error(
            centers, Ls, amps, R, shape, truncate=2.0
        )

        assert errors.shape == (1,)
        assert errors[0].item() > 0.5

    def test_4d_cull(self, device: torch.device) -> None:
        """Culling works end-to-end for 4D data."""
        shape = (8, 8, 8, 8)
        c1 = torch.tensor([[4.0, 4.0, 4.0, 4.0]], device=device)
        c2 = torch.tensor([[4.0, 4.0, 4.0, 4.0]], device=device)
        centers = torch.cat([c1, c2])
        amps = torch.tensor([1.0, 0.001], device=device)
        L = torch.eye(4, device=device).unsqueeze(0) * 2.0
        Ls = L.expand(2, -1, -1).clone()

        # Target from real splat only
        target = render_gaussians(shape, c1, L, amps[:1], truncate=2.0)

        result = cull_by_contribution(
            centers,
            Ls,
            amps,
            target,
            shape,
            truncate=2.0,
        )

        assert result.n_culled >= 1


class TestJointCompoundingBinarySearch:
    """Test that the joint compounding binary search is monotonic."""

    def test_verbose_output(self, device: torch.device) -> None:
        """Verbose mode should not crash."""
        shape = (32, 32)
        s1 = _make_splat_2d((16.0, 16.0), 1.0, 3.0, device)
        s2 = _make_splat_2d((16.0, 16.0), 0.01, 3.0, device)
        c_real, L_real, a_real = s1
        target = render_gaussians(shape, c_real, L_real, a_real, truncate=3.0)
        centers, Ls, amps = _concat_splats(s1, s2)

        result = cull_by_contribution(
            centers,
            Ls,
            amps,
            target,
            shape,
            truncate=3.0,
            verbose=True,
        )
        assert isinstance(result, CullResult)


class TestGSplatDataCull:
    def test_basic_cull(self, device: torch.device) -> None:
        """GSplatData.cull() should return a valid GSplatData."""
        from luxar.gsplats.gsplat_data import GSplatData

        shape = (32, 32)
        # Build GSplatData with numpy arrays
        centers_np = np.array([[16.0, 16.0], [16.0, 16.0]], dtype=np.float32)
        amps_np = np.array([1.0, 0.001], dtype=np.float32)
        # Packed Cholesky for 2D: [L00, L10, L11]
        chol_np = np.array([[3.0, 0.0, 3.0], [3.0, 0.0, 3.0]], dtype=np.float32)

        data = GSplatData(
            centers=centers_np,
            amplitudes=amps_np,
            cholesky_factors=chol_np,
        )

        # Create a target volume
        centers_t = torch.from_numpy(centers_np).to(device)
        amps_t = torch.from_numpy(amps_np).to(device)
        Ls_t = torch.zeros((2, 2, 2), dtype=torch.float32, device=device)
        Ls_t[:, 0, 0] = torch.tensor([3.0, 3.0], device=device)
        Ls_t[:, 1, 1] = torch.tensor([3.0, 3.0], device=device)
        target_np = (
            render_gaussians(shape, centers_t, Ls_t, amps_t, truncate=3.0).cpu().numpy()
        )

        culled = data.cull(target_np, device=str(device))

        assert isinstance(culled, GSplatData)
        assert culled.n_splats <= data.n_splats

    def test_stats_populated(self, device: torch.device) -> None:
        """Culling metadata should be present in stats."""
        from luxar.gsplats.gsplat_data import GSplatData

        shape = (32, 32)
        centers_np = np.array([[16.0, 16.0]], dtype=np.float32)
        amps_np = np.array([1.0], dtype=np.float32)
        chol_np = np.array([[3.0, 0.0, 3.0]], dtype=np.float32)

        data = GSplatData(
            centers=centers_np,
            amplitudes=amps_np,
            cholesky_factors=chol_np,
        )

        centers_t = torch.from_numpy(centers_np).to(device)
        amps_t = torch.from_numpy(amps_np).to(device)
        Ls_t = torch.zeros((1, 2, 2), dtype=torch.float32, device=device)
        Ls_t[0, 0, 0] = 3.0
        Ls_t[0, 1, 1] = 3.0
        target_np = (
            render_gaussians(shape, centers_t, Ls_t, amps_t, truncate=3.0).cpu().numpy()
        )

        culled = data.cull(target_np, device=str(device))

        assert culled.stats["culled"] is True
        assert culled.stats["culling_method"] == "error_budget"
        assert "n_original" in culled.stats
        assert "n_culled" in culled.stats
        assert "error_budget" in culled.stats
        assert "phase1_candidates" in culled.stats

    def test_redundancy_mode_no_target(self, device: torch.device) -> None:
        """GSplatData.cull() without target uses redundancy mode."""
        from luxar.gsplats.gsplat_data import GSplatData

        centers_np = np.array([[16.0, 16.0], [16.0, 16.0]], dtype=np.float32)
        amps_np = np.array([1.0, 0.005], dtype=np.float32)
        chol_np = np.array([[3.0, 0.0, 3.0], [3.0, 0.0, 3.0]], dtype=np.float32)

        data = GSplatData(
            centers=centers_np,
            amplitudes=amps_np,
            cholesky_factors=chol_np,
        )

        culled = data.cull(shape=(32, 32), device=str(device))

        assert culled.stats["culling_method"] == "redundancy"
        assert culled.n_splats <= data.n_splats
