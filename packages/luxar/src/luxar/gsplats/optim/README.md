# Optimizer Utilities for Gaussian Splatting

Standard PyTorch Adam optimizer with gradient dilution compensation for consistent optimization across different dimensionalities.

## Overview

The optim package provides a simple factory function that creates a standard PyTorch Adam optimizer with automatic gradient dilution compensation. This enables efficient optimization of Gaussian splats using the fixed-pool relocation architecture.

**Key Features:**
- **Standard PyTorch Adam**: Fast, vectorized optimization (50x+ faster than per-splat alternatives)
- **Gradient dilution compensation**: Automatic LR scaling for different dimensions
- **Per-parameter-group LR**: Amplitudes get 3x the base learning rate for faster convergence (inspired by AbsGS / Taming 3DGS, ECCV 2024)
- **Fused Adam on CUDA**: Automatic use of fused single-kernel Adam when all parameters are on CUDA (PyTorch 2.0+)
- **Flexible scheduling**: Supports plateau and exponential LR schedulers
- **Simple API**: Single factory function for complete setup

## Quick Start

```python
from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel
from luxar.gsplats.optim import create_optimizer_and_scheduler

# Create model
model = GaussianSplatModel(shape=(256, 256), ...)

# Create optimizer and scheduler
optimizer, scheduler = create_optimizer_and_scheduler(
    model,
    lr=0.01,                      # Base learning rate (auto-compensated)
    scheduler_type="plateau",     # "plateau", "exponential", or None
    patience=10,                  # Plateau scheduler patience
    factor=0.5,                   # LR reduction factor
)

# Standard optimization loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    prediction = model()
    loss = compute_loss(prediction, target)
    loss.backward()
    optimizer.step()

    # Update scheduler (for plateau, pass the loss)
    if scheduler is not None:
        scheduler.step(loss)
```

## API Reference

### create_optimizer_and_scheduler

```python
def create_optimizer_and_scheduler(
    model,
    lr: float = 1e-3,
    scheduler_type: Optional[str] = "plateau",
    # Optimizer arguments
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    amsgrad: bool = False,
    # Scheduler arguments
    patience: int = 10,
    factor: float = 0.5,
    threshold: float = 1e-3,
    cooldown: int = 0,
    min_lr: float = 1e-8,
    gamma: float = 0.95,
    **extra_kwargs: Any,
) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler.LRScheduler]]:
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | GaussianSplatModel | - | The model to optimize |
| `lr` | float | 1e-3 | Base learning rate (automatically compensated for gradient dilution) |
| `scheduler_type` | str or None | "plateau" | Scheduler type: "plateau", "exponential", or None |
| `betas` | tuple | (0.9, 0.999) | Adam beta parameters for momentum |
| `eps` | float | 1e-8 | Numerical stability epsilon |
| `weight_decay` | float | 0.0 | L2 regularization coefficient |
| `amsgrad` | bool | False | Use AMSGrad variant |
| `patience` | int | 10 | Plateau scheduler: iterations without improvement before LR reduction |
| `factor` | float | 0.5 | LR reduction factor (new_lr = lr × factor) |
| `threshold` | float | 1e-3 | Plateau scheduler: minimum improvement threshold |
| `cooldown` | int | 0 | Iterations to wait after LR reduction |
| `min_lr` | float | 1e-8 | Minimum learning rate floor |
| `gamma` | float | 0.95 | Exponential scheduler: decay rate per step |

**Returns:**
- `optimizer`: Standard `torch.optim.Adam` optimizer
- `scheduler`: LR scheduler (ReduceLROnPlateau, ExponentialLR, or None)

## Gradient Dilution Compensation

### The Problem

Higher-dimensional Gaussian splats have more parameters per splat:
- **2D**: 5 parameters (position + covariance + amplitude)
- **3D**: 9 parameters
- **4D**: 14 parameters

When loss gradients backpropagate, they get **diluted** across more parameters in higher dimensions, requiring higher learning rates to achieve equivalent optimization effectiveness.

### The Solution

The optimizer automatically applies gradient dilution compensation:

| Dimension | Factor | Effective LR (base=0.01) |
|-----------|--------|--------------------------|
| 2D | 1.0× | 0.01 |
| 3D | 1.8× | 0.018 |
| 4D | 8.5× | 0.085 |

This compensation is applied transparently - you specify a base learning rate and the optimizer adjusts it based on the model's dimensionality.

## Per-Parameter-Group Learning Rates

Amplitudes converge faster than positions and Cholesky (shape) parameters in Gaussian splatting. To exploit this, the optimizer assigns **3x the base learning rate** to amplitude parameters (`raw_a`), while positions and Cholesky factors use the standard (dilution-compensated) LR. This accelerates convergence without destabilizing the more sensitive center/Cholesky optimization.

This technique is inspired by AbsGS / Taming 3DGS (ECCV 2024).

## Fused Adam on CUDA

When all model parameters reside on CUDA and PyTorch 2.0+ is available, the optimizer automatically enables **fused Adam**, which performs the entire Adam update in a single CUDA kernel. This reduces kernel launch overhead and improves training throughput. Fused Adam is disabled when `amsgrad=True` (unsupported by the fused backend).

## Usage Examples

### Basic Optimization

```python
from luxar.gsplats.optim import create_optimizer_and_scheduler

# Simple setup with plateau scheduler
optimizer, scheduler = create_optimizer_and_scheduler(
    model, lr=0.01, scheduler_type="plateau"
)

for iteration in range(1000):
    optimizer.zero_grad()
    prediction = model()
    loss = mse_loss(prediction, target)
    loss.backward()
    optimizer.step()
    scheduler.step(loss)
```

### Exponential Decay

```python
# Exponential LR decay
optimizer, scheduler = create_optimizer_and_scheduler(
    model,
    lr=0.01,
    scheduler_type="exponential",
    gamma=0.95  # Decay by 5% each step
)

for iteration in range(1000):
    optimizer.zero_grad()
    prediction = model()
    loss = mse_loss(prediction, target)
    loss.backward()
    optimizer.step()
    scheduler.step()  # No loss needed for exponential
```

### No Scheduler

```python
# Optimizer only, no LR scheduling
optimizer, _ = create_optimizer_and_scheduler(
    model, lr=0.01, scheduler_type=None
)
```

### Custom Adam Parameters

```python
# Custom Adam configuration
optimizer, scheduler = create_optimizer_and_scheduler(
    model,
    lr=0.005,
    scheduler_type="plateau",
    betas=(0.85, 0.995),  # More aggressive momentum
    eps=1e-6,
    weight_decay=0.01,    # L2 regularization
    amsgrad=True,         # AMSGrad for stability
    patience=20,
    factor=0.3,
)
```

## Integration with Dynamic Operations

The optimizer works seamlessly with the fixed-pool splat relocation system. When dynamic operations relocate splats:

1. **No tensor shape changes**: Fixed splat pool means optimizer state remains valid
2. **Natural momentum adaptation**: Standard Adam naturally adapts to parameter changes
3. **No explicit state management**: Unlike per-splat optimizers, no manual state updates needed

```python
from luxar.gsplats.fitting.dynamic_ops import apply_dynamic_operations, DynamicOpsConfig

optimizer, scheduler = create_optimizer_and_scheduler(model, lr=0.01)
cfg = DynamicOpsConfig()

for iteration in range(1000):
    optimizer.zero_grad()
    prediction = model()
    loss = compute_loss(prediction, target)
    loss.backward()
    optimizer.step()
    scheduler.step(loss)

    # Dynamic operations work naturally with standard optimizer
    if iteration % 50 == 0:
        apply_dynamic_operations(
            model, target, prediction, cfg,
            max_abs_error_threshold=0.01
        )
```

## Testing

```bash
# Run optimizer tests
hatch run pytest packages/luxar/src/luxar/gsplats/optim/tests/ -v
```

## Related Documentation

- [Dynamic Operations README.md](../fitting/dynamic_ops/README.md) - Fixed-pool relocation
- [Fitting README.md](../fitting/README.md) - Complete fitting pipeline
- [utils/README.md](../utils/README.md) - Gradient dilution calculation
