# Optimizer Integration Specifications

**Version**: 2.0.0
**Last Updated**: 2025-01

## Overview

This module provides a factory function for creating standard PyTorch Adam optimizers with automatic gradient dilution compensation. The design enables efficient optimization using a fixed-pool architecture where splat relocation replaces add/remove operations.

**Key Design Decision**: Use standard PyTorch Adam instead of per-splat optimizers for 50x+ performance improvement.

**Related Specifications**:
- **Gradient Dilution Calculation**: [utils/SPECIFICATIONS.md](../utils/SPECIFICATIONS.md)
- **Dynamic Operations**: [fitting/dynamic_ops/SPECIFICATIONS.md](../fitting/dynamic_ops/SPECIFICATIONS.md)
- **Fitting Pipeline**: [fitting/SPECIFICATIONS.md](../fitting/SPECIFICATIONS.md)

## Architecture

```
create_optimizer_and_scheduler()
        │
        ├── Calculate gradient dilution factor (based on dimension)
        ├── Create torch.optim.Adam with adjusted LR
        └── Create scheduler (optional)
            ├── ReduceLROnPlateau
            ├── ExponentialLR
            └── None
```

## Function Specification

### create_optimizer_and_scheduler

```python
def create_optimizer_and_scheduler(
    model,
    lr: float = 1e-3,
    scheduler_type: Optional[str] = "plateau",
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    amsgrad: bool = False,
    patience: int = 10,
    factor: float = 0.5,
    threshold: float = 1e-4,
    cooldown: int = 0,
    min_lr: float = 1e-8,
    gamma: float = 0.95,
    **extra_kwargs,
) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler.LRScheduler]]:
```

**Algorithm:**

1. Extract dimension `d` from `model.shape`
2. Calculate gradient dilution factor: `factor = calculate_gradient_dilution_factor(d)`
3. Compute effective learning rate: `effective_lr = lr * factor`
4. Create `torch.optim.Adam` with effective LR and other parameters
5. Create scheduler based on `scheduler_type`:
   - `"plateau"` → `ReduceLROnPlateau(optimizer, ...)`
   - `"exponential"` → `ExponentialLR(optimizer, gamma=gamma)`
   - `None` → No scheduler
6. Return `(optimizer, scheduler)`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | GaussianSplatModel | required | Model to optimize |
| `lr` | float | 1e-3 | Base learning rate (auto-compensated) |
| `scheduler_type` | str or None | "plateau" | "plateau", "exponential", or None |
| `betas` | tuple | (0.9, 0.999) | Adam momentum parameters |
| `eps` | float | 1e-8 | Numerical stability |
| `weight_decay` | float | 0.0 | L2 regularization |
| `amsgrad` | bool | False | Use AMSGrad variant |
| `patience` | int | 10 | Plateau: iterations before LR reduction |
| `factor` | float | 0.5 | LR reduction factor |
| `threshold` | float | 1e-4 | Improvement threshold |
| `cooldown` | int | 0 | Post-reduction wait period |
| `min_lr` | float | 1e-8 | Minimum learning rate |
| `gamma` | float | 0.95 | Exponential decay rate |

**Returns:**
- `optimizer`: `torch.optim.Adam` instance
- `scheduler`: LR scheduler or None

## Gradient Dilution Compensation

### Problem Statement

Higher-dimensional Gaussian splats have more parameters:
- **2D**: d + d(d+1)/2 = 2 + 3 = 5 parameters (μ + L)
- **3D**: 3 + 6 = 9 parameters
- **4D**: 4 + 10 = 14 parameters
- **dD**: d + d(d+1)/2 parameters

Loss gradients are distributed across all parameters. More parameters → smaller individual gradients ("dilution").

### Compensation Formula

Computed by `calculate_gradient_dilution_factor(d)` in `utils/trils.py`:

**For d ≤ 3:**
```
factor = params_d / params_2d
```

**For d > 3:**
```
dimensional_complexity = d^0.8
parameter_complexity = params_d / params_2d
factor = dimensional_complexity × parameter_complexity
```

**Values:**

| d | params | factor |
|---|--------|--------|
| 2 | 5 | 1.0 |
| 3 | 9 | 1.8 |
| 4 | 14 | ~8.5 |
| 5 | 20 | ~17 |

## Scheduler Specifications

### ReduceLROnPlateau

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    patience=patience,
    factor=factor,
    threshold=threshold,
    cooldown=cooldown,
    min_lr=min_lr,
)
```

**Usage:**
```python
scheduler.step(loss)  # Pass current loss value
```

**Behavior:**
- Monitors metric and reduces LR when improvement stops
- After `patience` iterations without improvement > `threshold`, multiply LR by `factor`
- Wait `cooldown` iterations before next reduction
- Never reduce below `min_lr`

### ExponentialLR

```python
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
```

**Usage:**
```python
scheduler.step()  # No metric needed
```

**Behavior:**
- Each step: `lr_new = lr * gamma`
- Predictable exponential decay

## Integration with Fixed-Pool Relocation

### Why Standard Adam Works

The fixed-pool architecture keeps tensor shapes constant:

1. **No topology changes**: Splat pool size is fixed
2. **Relocation = parameter update**: Just modifies values, not tensor shapes
3. **Momentum adaptation**: Stale momentum at relocated splat quickly overwritten by new gradients

When a splat is relocated:
1. Its parameters are modified in-place
2. Adam's momentum buffers remain at same indices
3. New gradient at new location overwrites stale momentum
4. Full adaptation within 1-3 iterations

This is much simpler than maintaining per-splat state that must be explicitly synchronized during topology changes.

### Performance Comparison

| Approach | Time per Iteration | Memory | Complexity |
|----------|-------------------|--------|------------|
| Standard Adam | 1x (baseline) | Low | Simple |
| Per-splat Adam | 50x+ slower | Higher | Complex state management |

## File Structure

```
optim/
├── __init__.py           # Exports create_optimizer_and_scheduler
├── integration.py        # Factory function
├── README.md             # Usage guide
├── SPECIFICATIONS.md     # This file
└── tests/
    └── test_integration.py
```

## Dependencies

- `torch.optim.Adam`
- `torch.optim.lr_scheduler.ReduceLROnPlateau`
- `torch.optim.lr_scheduler.ExponentialLR`
- `luxar.gsplats.utils.trils.calculate_gradient_dilution_factor`

## Testing

```bash
hatch run pytest packages/luxar/src/luxar/gsplats/optim/tests/test_integration.py -v
```

**Test Coverage:**
- Optimizer creation
- Scheduler creation (all types)
- Gradient dilution factor application
- Training loop integration
- Invalid scheduler handling
- 2D and 3D models

## Version History

- **v2.0.0** (January 2025): Simplified to standard PyTorch Adam
  - Removed per-splat optimizer (50x+ performance gain)
  - Fixed-pool relocation architecture
  - Standard Adam with gradient dilution compensation

- **v1.0.0** (November 2024): Initial per-splat optimizer (deprecated and removed)
