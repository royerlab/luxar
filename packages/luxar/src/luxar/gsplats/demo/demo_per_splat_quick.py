#!/usr/bin/env python3
"""
Quick demo showcasing per-splat Adam optimizer benefits.

Shows the key difference: standard optimizer loses ALL momentum during
dynamic operations, while per-splat optimizer preserves momentum.
"""

import numpy as np
import torch
from arbol import aprint, asection

from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel
from luxar.gsplats.optim import PerSplatAdam


def demo_momentum_preservation():
    """Demonstrate momentum preservation with per-splat optimizer."""

    aprint("🧪 Per-Splat Optimizer Momentum Preservation Demo")

    # Create a simple 2D model
    shape = (32, 32)
    n_splats = 5
    centers0 = np.random.uniform(8, 24, (n_splats, 2)).astype(np.float32)
    L0 = np.stack([np.eye(2) * 1.5] * n_splats).astype(np.float32)
    amps0 = np.random.uniform(0.3, 0.8, n_splats).astype(np.float32)

    model = GaussianSplatModel(
        shape=shape,
        centers0=centers0,
        L0=L0,
        amps0=amps0,
        sigma_min_diag=[0.5, 0.5],
        device=torch.device("cpu"),
    )

    # Create target for optimization
    target = torch.randn(shape) * 0.5 + 0.5

    with asection("Phase 1: Build momentum (10 steps)"):
        optimizer = PerSplatAdam(model, lr=0.05)

        # Optimize to build momentum
        losses = []
        for step in range(10):
            optimizer.zero_grad()
            pred = model()
            loss = torch.nn.functional.mse_loss(pred, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if step % 3 == 0:
                aprint(f"  Step {step + 1}: loss={loss.item():.6f}")

        aprint(f"✓ Built momentum over {len(losses)} steps")
        aprint("✓ All splats have accumulated momentum in optimizer")

        # Show that splats have momentum
        n_with_momentum = sum(
            1
            for state in optimizer.splat_states.values()
            if torch.any(state["exp_avg_mu"].abs() > 1e-8)
        )
        aprint(
            f"✓ {n_with_momentum}/{len(optimizer.splat_states)} splats have non-zero momentum"
        )

    with asection("Phase 2: Add new splats (dynamic operation)"):
        # Simulate adding 2 new splats (like seeding operation)
        centers_new = torch.tensor([[12.0, 12.0], [20.0, 8.0]], dtype=torch.float32)
        Ls_new = torch.stack([torch.eye(2) * 1.0, torch.eye(2) * 1.2], dim=0)
        amps_new = torch.tensor([0.4, 0.6], dtype=torch.float32)

        old_n = model.n_splats()
        model.append_(centers_new, Ls_new, amps_new)
        optimizer.add_splats(2, lr_new=0.1)  # Higher LR for new splats

        aprint(f"✓ Added 2 new splats: {old_n} → {model.n_splats()}")

        # Check momentum preservation
        n_preserved = sum(
            1
            for i, state in optimizer.splat_states.items()
            if i < old_n and torch.any(state["exp_avg_mu"].abs() > 1e-8)
        )
        aprint(f"✓ {n_preserved}/{old_n} original splats preserved momentum")

        # Check new splats start with zero momentum (correct)
        n_new_zero = sum(
            1
            for i, state in optimizer.splat_states.items()
            if i >= old_n and torch.all(state["exp_avg_mu"].abs() < 1e-8)
        )
        aprint(f"✓ {n_new_zero}/2 new splats start with zero momentum (correct)")

        # Show different learning rates
        lrs = optimizer.get_effective_learning_rates()
        aprint(
            f"✓ Learning rates: old splats={lrs[:old_n].mean():.3f}, new splats={lrs[old_n:].mean():.3f}"
        )

    with asection("Phase 3: Continue optimization (5 steps)"):
        # Continue optimizing - old splats should resume smoothly
        for step in range(5):
            optimizer.zero_grad()
            pred = model()
            loss = torch.nn.functional.mse_loss(pred, target)
            loss.backward()
            optimizer.step()

            aprint(
                f"  Step {step + 1}: loss={loss.item():.6f}, splats={model.n_splats()}"
            )

        aprint("✅ Optimization continued smoothly with preserved momentum!")

    with asection("Comparison: What Standard Optimizer Would Do"):
        aprint("❌ Standard PyTorch Adam would:")
        aprint("  • Lose ALL momentum for ALL splats (even unchanged ones)")
        aprint("  • Reset ALL learning rates to initial value")
        aprint("  • Cause global disruption and oscillations")
        aprint("  • Need to rebuild momentum from scratch")
        aprint("")
        aprint("✅ Per-Splat Adam:")
        aprint("  • Preserves momentum for unchanged splats")
        aprint("  • Maintains individual learning rates")
        aprint("  • No global disruption")
        aprint("  • Seamless integration of new splats")


def demo_individual_learning_rates():
    """Demonstrate individual learning rate management."""

    aprint("\n🎯 Individual Learning Rate Management Demo")

    # Create model
    shape = (16, 16)
    centers0 = np.array([[4, 4], [8, 8], [12, 12]], dtype=np.float32)
    L0 = np.stack([np.eye(2)] * 3).astype(np.float32)
    amps0 = np.array([0.5, 0.7, 0.3], dtype=np.float32)

    model = GaussianSplatModel(
        shape=shape,
        centers0=centers0,
        L0=L0,
        amps0=amps0,
        sigma_min_diag=[0.5, 0.5],
    )

    optimizer = PerSplatAdam(model, lr=0.02)

    with asection("Individual LR Control"):
        # Set different learning rates for different splats
        optimizer.set_learning_rate(0, 0.01)  # Slow learner
        optimizer.set_learning_rate(1, 0.05)  # Fast learner
        optimizer.set_learning_rate(2, 0.001)  # Very slow learner

        lrs = optimizer.get_effective_learning_rates()
        aprint(f"✓ Set individual LRs: {lrs}")

        # Simulate optimization steps
        target = torch.zeros(shape)
        for step in range(3):
            optimizer.zero_grad()
            pred = model()
            loss = torch.nn.functional.mse_loss(pred, target)
            loss.backward()
            optimizer.step()

            aprint(f"  Step {step + 1}: loss={loss.item():.6f}")

        aprint("✅ Each splat optimizes at its own pace!")


if __name__ == "__main__":
    demo_momentum_preservation()
    demo_individual_learning_rates()
    aprint("\n🎉 Per-splat optimizer demos complete!")
    aprint("Ready to integrate into dynamic Gaussian splatting!")
