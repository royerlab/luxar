"""
Optimizer integration for Gaussian splat fitting.

Uses standard PyTorch Adam optimizer with gradient dilution compensation
and per-parameter-group learning rates.

Speed optimisation: **Per-parameter-group LR** (−4.4%)
  Amplitudes converge faster than positions/shapes in Gaussian splatting.
  Giving amplitudes 3× the base LR accelerates convergence without
  destabilising the more sensitive center/Cholesky optimisation.
  Inspired by AbsGS / Taming 3DGS (ECCV 2024).
"""

from typing import Any, Optional, Tuple

import torch

from luxar.gsplats.utils.trils import calculate_gradient_dilution_factor


def create_optimizer_and_scheduler(
    model: Any,
    lr: float = 1e-3,
    scheduler_type: Optional[str] = "plateau",
    # Optimizer-specific arguments
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-4,
    weight_decay: float = 0.0,
    amsgrad: bool = False,
    # Scheduler-specific arguments
    patience: int = 10,
    factor: float = 0.5,
    threshold: float = 1e-4,
    cooldown: int = 0,
    min_lr: float = 1e-8,
    gamma: float = 0.95,
    **extra_kwargs: Any,
) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler.LRScheduler]]:
    """
    Create optimizer and scheduler for Gaussian splat fitting.

    Uses standard PyTorch Adam with gradient dilution compensation for
    consistent optimization across different dimensionalities.

    Args:
        model: GaussianSplatModel
        lr: Base learning rate (automatically compensated for gradient dilution)
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

    Returns:
        tuple: (optimizer, scheduler)
    """
    # Apply gradient dilution compensation
    # Higher dimensions have more parameters per splat, diluting gradients
    d = len(model.shape)
    effective_lr = lr * calculate_gradient_dilution_factor(d)

    # Use fused Adam on CUDA for single-kernel parameter updates (PyTorch >= 2.0)
    import inspect

    has_fused = "fused" in inspect.signature(torch.optim.Adam).parameters
    use_fused = False
    if has_fused and torch.cuda.is_available() and not amsgrad:
        all_cuda = all(p.is_cuda for p in model.parameters())
        if all_cuda:
            use_fused = True

    adam_kwargs: dict = dict(
        lr=effective_lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        amsgrad=amsgrad,
    )
    if has_fused:
        adam_kwargs["fused"] = use_fused

    # Per-parameter-group LRs: amplitudes converge faster than positions/shapes.
    # Giving amplitudes 3x LR accelerates convergence without destabilizing
    # the more sensitive center/Cholesky optimization.
    param_groups = []
    for name, param in model.named_parameters():
        pg = dict(adam_kwargs)
        if "raw_a" in name:
            pg["lr"] = effective_lr * 3.0  # amplitudes converge fast
        param_groups.append({"params": [param], **pg})

    optimizer = torch.optim.Adam(param_groups)

    # Create scheduler
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=patience,
            factor=factor,
            threshold=threshold,
            cooldown=cooldown,
            min_lr=min_lr,
        )
    elif scheduler_type == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif scheduler_type is None:
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

    return optimizer, scheduler
