"""
Integration helpers for per-splat optimizers with dynamic operations.
"""

from typing import Optional, Union

import torch

from .per_splat_adam import PerSplatAdam
from .per_splat_scheduler import PerSplatExponentialLR, PerSplatReduceLROnPlateau


class ModelOptimizerCoordinator:
    """
    Coordinates model topology changes with per-splat optimizer state management.

    Ensures that when the model adds/removes splats, the optimizer and scheduler
    state are kept in sync without losing momentum for unchanged splats.
    """

    def __init__(
        self,
        model,
        optimizer: PerSplatAdam,
        scheduler: Optional[
            Union[PerSplatReduceLROnPlateau, PerSplatExponentialLR]
        ] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Track operations for debugging
        self.operation_count = 0

    def prune_splats(self, keep_mask: torch.Tensor):
        """
        Prune splats from model and sync optimizer state.

        Args:
            keep_mask: Boolean tensor indicating which splats to keep
        """
        n_before = self.model.n_splats()

        # Update model
        self.model.prune_(keep_mask)

        # Update optimizer state
        self.optimizer.remove_splats(keep_mask)

        # Update scheduler state if present
        if self.scheduler is not None and hasattr(self.scheduler, "remove_splats"):
            self.scheduler.remove_splats(keep_mask)

        n_after = self.model.n_splats()
        n_removed = n_before - n_after
        self.operation_count += 1

        return n_removed

    def add_splats(
        self,
        centers_new: torch.Tensor,
        Ls_new: torch.Tensor,
        amps_new: torch.Tensor,
        lr_new: Optional[float] = None,
    ):
        """
        Add new splats to model and sync optimizer state.

        Args:
            centers_new: New splat centers, shape (n_new, d)
            Ls_new: New splat Cholesky factors, shape (n_new, d, d)
            amps_new: New splat amplitudes, shape (n_new,)
            lr_new: Learning rate for new splats (default: optimizer base_lr)
        """
        n_before = self.model.n_splats()
        n_new = centers_new.shape[0]

        # Update model
        self.model.append_(centers_new, Ls_new, amps_new)

        # Update optimizer state
        self.optimizer.add_splats(n_new, lr_new=lr_new)

        # Update scheduler state if present
        if self.scheduler is not None and hasattr(self.scheduler, "add_splats"):
            self.scheduler.add_splats(n_new)

        n_after = self.model.n_splats()
        assert n_after == n_before + n_new, (
            f"Expected {n_before + n_new} splats, got {n_after}"
        )

        self.operation_count += 1
        return n_new

    def replace_all_splats(
        self,
        centers: torch.Tensor,
        Ls: torch.Tensor,
        amps: torch.Tensor,
        lr_reset: Optional[float] = None,
    ):
        """
        Replace all splats (complete model reset).

        This is used for operations like splitting where the entire
        splat population changes structure.
        """
        n_new = centers.shape[0]

        # Update model
        self.model.replace_with(centers, Ls, amps)

        # Reset optimizer state completely
        self.optimizer.splat_states = {}
        self.optimizer.add_splats(n_new, lr_new=lr_reset)

        # Reset scheduler state if present
        if self.scheduler is not None:
            if hasattr(self.scheduler, "splat_scheduler_states"):
                self.scheduler.splat_scheduler_states = {}
            if hasattr(self.scheduler, "splat_ages"):
                self.scheduler.splat_ages = {}
            if hasattr(self.scheduler, "add_splats"):
                self.scheduler.add_splats(n_new)

        self.operation_count += 1
        return n_new

    def get_status(self) -> dict:
        """Get coordinator status for monitoring."""
        return {
            "model_splats": self.model.n_splats(),
            "optimizer_states": len(self.optimizer.splat_states),
            "operation_count": self.operation_count,
            "learning_rates": {
                "mean": float(
                    torch.mean(self.optimizer.get_effective_learning_rates())
                ),
                "min": float(torch.min(self.optimizer.get_effective_learning_rates())),
                "max": float(torch.max(self.optimizer.get_effective_learning_rates())),
            },
        }


def create_per_splat_optimizer_setup(
    model,
    lr: float = 1e-3,
    scheduler_type: str = "plateau",
    # Optimizer-specific arguments
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    amsgrad: bool = False,
    # Scheduler-specific arguments
    patience: int = 10,
    factor: float = 0.5,
    threshold: float = 1e-4,
    cooldown: int = 0,
    min_lr: float = 1e-8,
    gamma: float = 0.95,
    age_based_decay: bool = True,
    **extra_kwargs,
):
    """
    Factory function to create coordinated per-splat optimizer setup.

    Args:
        model: GaussianSplatModel
        lr: Base learning rate
        scheduler_type: 'plateau', 'exponential', or None

        # Optimizer args
        betas: Adam beta parameters
        eps: Adam epsilon
        weight_decay: L2 penalty
        amsgrad: Whether to use AMSGrad

        # Scheduler args
        patience: Plateau scheduler patience
        factor: LR reduction factor
        threshold: Improvement threshold
        cooldown: Cooldown period
        min_lr: Minimum learning rate
        gamma: Exponential decay rate
        age_based_decay: Whether to use age-based decay

    Returns:
        tuple: (optimizer, scheduler, coordinator)
    """
    # Create optimizer with only optimizer-specific arguments
    optimizer_kwargs = {
        "betas": betas,
        "eps": eps,
        "weight_decay": weight_decay,
        "amsgrad": amsgrad,
    }
    optimizer = PerSplatAdam(model, lr=lr, **optimizer_kwargs)

    # Create scheduler with scheduler-specific arguments
    if scheduler_type == "plateau":
        scheduler_kwargs = {
            "patience": patience,
            "factor": factor,
            "threshold": threshold,
            "cooldown": cooldown,
            "min_lr": min_lr,
        }
        scheduler = PerSplatReduceLROnPlateau(optimizer, **scheduler_kwargs)
    elif scheduler_type == "exponential":
        scheduler_kwargs = {"gamma": gamma, "age_based_decay": age_based_decay}
        scheduler = PerSplatExponentialLR(optimizer, **scheduler_kwargs)
    elif scheduler_type is None:
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

    # Create coordinator
    coordinator = ModelOptimizerCoordinator(model, optimizer, scheduler)

    return optimizer, scheduler, coordinator
