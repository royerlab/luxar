"""Z-order (Morton code) sorting of Gaussian splats for memory locality.

Periodically reordering splats by their spatial Morton code improves GPU cache
coherence during rendering and gradient computation, following the approach
described in Faster-GS (Hahlbohm et al., 2026).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from luxar.io.ordering import morton_encode_nd, normalize_coords_to_grid

if TYPE_CHECKING:
    from luxar.gsplats.fitting.dynamic_ops import RecentlyRelocatedTracker
    from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel


def sort_splats_by_morton_order(
    model: "GaussianSplatModel",
    optimizer: torch.optim.Optimizer,
    relocation_tracker: Optional["RecentlyRelocatedTracker"] = None,
) -> None:
    """Reorder all splat parameters and optimizer state by Morton code.

    This sorts splats so that spatially nearby Gaussians are also contiguous
    in memory, improving cache locality for tiled rendering kernels.

    Parameters
    ----------
    model : GaussianSplatModel
        The model whose parameters will be permuted in-place.
    optimizer : torch.optim.Optimizer
        The optimizer whose per-parameter state (exp_avg, exp_avg_sq) will
        be permuted to match.
    relocation_tracker : RecentlyRelocatedTracker, optional
        If provided, its per-splat cooldown state is also permuted.
    """
    with torch.no_grad():
        centers = model.current_params()[0]  # (N, d) activated voxel coords
        centers_np = centers.detach().cpu().numpy()

    n_splats, ndim = centers_np.shape
    if n_splats <= 1:
        return

    # Compute Morton codes using existing infrastructure
    bits_per_dim = min(21, 64 // ndim)
    min_coords = centers_np.min(axis=0)
    max_coords = centers_np.max(axis=0)
    grid_coords = normalize_coords_to_grid(
        centers_np, min_coords, max_coords, 2**bits_per_dim
    )
    morton_codes = morton_encode_nd(grid_coords, bits_per_dim)
    sort_indices = np.argsort(morton_codes)

    # Early exit if already sorted
    if np.all(sort_indices == np.arange(n_splats)):
        return

    perm = torch.tensor(sort_indices, dtype=torch.long, device=centers.device)

    # Permute all model parameters in-place (preserves optimizer state keys)
    params = [
        model.raw_mu,
        model.raw_L_diag,
        model.L_off,
        model.raw_a,
    ]
    for param in params:
        param.data[:] = param.data[perm]

    # Permute optimizer state to match
    for param in params:
        if param in optimizer.state:
            state = optimizer.state[param]
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if key in state and isinstance(state[key], torch.Tensor):
                    state[key].data[:] = state[key].data[perm]

    # Permute relocation tracker state
    if relocation_tracker is not None and hasattr(
        relocation_tracker, "last_relocation_step"
    ):
        tracker_tensor = relocation_tracker.last_relocation_step
        if isinstance(tracker_tensor, torch.Tensor):
            tracker_tensor.data[:] = tracker_tensor.data[perm]
