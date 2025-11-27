#!/usr/bin/env python3
"""
Quick test of the per-splat Adam optimizer.
"""

import numpy as np
import torch

from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel
from luxar.gsplats.optim import (
    PerSplatAdam,
    PerSplatReduceLROnPlateau,
    create_per_splat_optimizer_setup,
)


def test_per_splat_adam() -> None:
    """Test basic per-splat Adam functionality."""
    print("🧪 Testing Per-Splat Adam Optimizer")

    # Create a simple model
    shape = (32, 32)
    n_splats = 5
    centers0 = np.random.uniform(5, 25, (n_splats, 2)).astype(np.float32)
    L0 = np.stack([np.eye(2) * 1.5] * n_splats).astype(np.float32)
    amps0 = np.random.uniform(0.1, 1.0, n_splats).astype(np.float32)

    model = GaussianSplatModel(
        shape=shape,
        centers0=centers0,
        L0=L0,
        amps0=amps0,
        sigma_min_diag=[0.5, 0.5],
        device=torch.device("cpu"),
    )

    # Create per-splat optimizer
    optimizer = PerSplatAdam(model, lr=0.01)
    scheduler = PerSplatReduceLROnPlateau(optimizer)

    print(f"✓ Created optimizer for {model.n_splats()} splats")
    print(f"✓ Initial learning rates: {optimizer.get_effective_learning_rates()}")

    # Test optimization step
    target = torch.randn_like(torch.zeros(shape))

    for i in range(3):
        optimizer.zero_grad()
        pred = model()
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())

        print(f"Step {i + 1}: loss={loss.item():.6f}")

    print(f"✓ Final learning rates: {optimizer.get_effective_learning_rates()}")

    # Test adding splats
    print("\n🔧 Testing splat addition...")
    centers_new = torch.tensor([[10.0, 10.0], [20.0, 20.0]], dtype=torch.float32)
    Ls_new = torch.stack([torch.eye(2) * 1.0, torch.eye(2) * 1.2], dim=0)
    amps_new = torch.tensor([0.5, 0.7], dtype=torch.float32)
    sharpness_new = torch.tensor([2.0, 2.0], dtype=torch.float32)

    old_n = model.n_splats()
    model.append_(centers_new, Ls_new, amps_new, sharpness_new)
    optimizer.add_splats(2, lr_new=0.1)  # Higher LR for new splats
    scheduler.add_splats(2)

    print(f"✓ Added 2 splats: {old_n} → {model.n_splats()}")
    print(f"✓ New learning rates: {optimizer.get_effective_learning_rates()}")

    # Test removing splats
    print("\n🔧 Testing splat removal...")
    keep_mask = torch.tensor(
        [True, False, True, False, True, True, True]
    )  # Remove 2nd and 4th
    model.prune_(keep_mask)
    optimizer.remove_splats(keep_mask)
    scheduler.remove_splats(keep_mask)

    print(f"✓ Pruned splats: 7 → {model.n_splats()}")
    print(f"✓ Remaining learning rates: {optimizer.get_effective_learning_rates()}")

    # Test optimization continues to work
    print("\n🔧 Testing post-topology optimization...")
    for i in range(2):
        optimizer.zero_grad()
        pred = model()
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())

        print(f"Post-topology step {i + 1}: loss={loss.item():.6f}")

    print("✅ Per-splat Adam test passed!")


def test_factory_function() -> None:
    """Test the factory function."""
    print("\n🏭 Testing factory function...")

    shape = (16, 16)
    centers0 = np.random.uniform(2, 14, (3, 2)).astype(np.float32)
    L0 = np.stack([np.eye(2)] * 3).astype(np.float32)
    amps0 = np.ones(3, dtype=np.float32)

    model = GaussianSplatModel(
        shape=shape,
        centers0=centers0,
        L0=L0,
        amps0=amps0,
        sigma_min_diag=[0.5, 0.5],
    )

    optimizer, scheduler, coordinator = create_per_splat_optimizer_setup(
        model, lr=0.02, scheduler_type="plateau"
    )

    print(
        f"✓ Factory created optimizer with {len(optimizer.splat_states)} splat states"
    )
    print(f"✓ Coordinator status: {coordinator.get_status()}")

    # Test coordinated operations
    centers_new = torch.tensor([[8.0, 8.0]], dtype=torch.float32)
    Ls_new = torch.stack([torch.eye(2)], dim=0)
    amps_new = torch.tensor([0.8], dtype=torch.float32)
    sharpness_new = torch.tensor([2.0], dtype=torch.float32)

    n_added = coordinator.add_splats(
        centers_new, Ls_new, amps_new, sharpness_new, lr_new=0.05
    )
    print(f"✓ Coordinator added {n_added} splats")
    print(f"✓ New status: {coordinator.get_status()}")

    print("✅ Factory function test passed!")


if __name__ == "__main__":
    test_per_splat_adam()
    test_factory_function()
    print("\n🎉 All tests passed! Per-splat optimizer is ready.")
