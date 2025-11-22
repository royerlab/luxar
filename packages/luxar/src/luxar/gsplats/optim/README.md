# Per-Splat Optimization for Gaussian Splatting

Specialized per-splat optimizers and learning rate schedulers that maintain individual optimization state for each Gaussian splat, enabling seamless dynamic operations and momentum preservation.

## Overview

The optim package provides optimization tools specifically designed for Gaussian splatting with dynamic topology changes. Unlike standard PyTorch optimizers that maintain state per-parameter tensor, per-splat optimizers maintain state per individual splat, enabling:

- **Momentum preservation** during splat addition/removal
- **Individual learning rates** per splat for adaptive optimization
- **Seamless dynamic operations** without disrupting optimization
- **Parameter-type-specific rates** to prevent splat proliferation
- **Better convergence** for mixed-age splat populations

This architecture is critical for Gaussian splatting optimization where the number of splats changes during training through seeding, pruning, and merging operations.

## Key Features

### Per-Splat State Management
Each Gaussian splat has its own optimizer state (momentum, learning rate, step count), allowing:
- New splats to start fresh while existing splats preserve their momentum
- Individual learning rate adaptation for different splats
- Efficient topology changes without global state disruption

### Gradient Dilution Compensation
Higher dimensions have more parameters per splat (2D: 5, 3D: 10, 4D: 15), which dilutes gradients. The optimizer automatically compensates:
- **2D**: lr × 1.0 (baseline)
- **3D**: lr × 2.0
- **4D**: lr × 7.1 (enhanced with dimensional complexity factor)

### Parameter-Type-Specific Learning Rates
Different parameter types optimize at different rates (hard-coded multipliers):
- **Position (μ)**: ×0.1 - Slow movement prevents splat migration and proliferation
- **Variance (L_diag, L_off)**: ×1.0 - Normal covariance adaptation
- **Amplitude (a)**: ×2.0 - Fast intensity matching improves convergence
- **Sharpness (s')**: ×0.5 - Conservative shape adaptation (no gradient dilution)

### Dynamic Operation Support
Atomic operations maintain consistency across model, optimizer, and scheduler:
- Adding splats (seeding, splitting)
- Removing splats (pruning)
- Replacing all splats (complete reset)

## Quick Start

### Basic Usage with Optimizer

```python
from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel
from luxar.gsplats.optim import PerSplatAdam

# Create model
model = GaussianSplatModel(shape=(256, 256), ...)

# Create per-splat optimizer
optimizer = PerSplatAdam(
    model,
    lr=0.01,                      # Base learning rate (auto-compensated for dimensions)
    betas=(0.9, 0.999),           # Adam momentum parameters
    eps=1e-8,                     # Numerical stability
    weight_decay=0.0,             # L2 regularization
    amsgrad=False                 # Use AMSGrad variant
)

# Standard optimization loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Forward pass
    prediction = model()
    loss = compute_loss(prediction, target)
    
    # Backward pass
    loss.backward()
    
    # Optimizer step (updates all splats)
    optimizer.step()
```

### Complete Setup with Scheduler and Coordinator

```python
from luxar.gsplats.optim import create_per_splat_optimizer_setup

# One-line setup with factory function
optimizer, scheduler, coordinator = create_per_splat_optimizer_setup(
    model,
    lr=0.01,
    scheduler_type="plateau",     # "plateau", "exponential", or None
    patience=10,                  # Plateau scheduler patience
    factor=0.5,                   # LR reduction factor
    min_lr=1e-8,                  # Minimum learning rate floor
)

# Optimization loop with scheduler
for epoch in range(num_epochs):
    optimizer.zero_grad()
    prediction = model()
    loss = compute_loss(prediction, target)
    loss.backward()
    optimizer.step()
    
    # Update learning rates based on loss
    scheduler.step(loss)
```

## Core Components

### 1. PerSplatAdam Optimizer

The main per-splat Adam optimizer with gradient dilution compensation and parameter-type-specific learning rates.

```python
from luxar.gsplats.optim import PerSplatAdam

optimizer = PerSplatAdam(
    model,
    lr=0.01,                      # Base learning rate
    betas=(0.9, 0.999),           # First and second moment decay rates
    eps=1e-8,                     # Numerical stability epsilon
    weight_decay=0.0,             # L2 penalty coefficient
    amsgrad=False                 # AMSGrad variant for stability
)

# Standard PyTorch-like interface
optimizer.zero_grad()
# ... compute loss and backward ...
optimizer.step()

# Per-splat learning rate control
optimizer.set_learning_rate(splat_idx=5, lr=0.02)  # Boost specific splat
current_lr = optimizer.get_learning_rate(splat_idx=5)
all_lrs = optimizer.get_effective_learning_rates()  # Tensor (n_splats,)

# State serialization
state = optimizer.state_dict()
optimizer.load_state_dict(state)
```

**Key Methods**:
- `step()`: Perform Adam update for all splats
- `zero_grad()`: Clear gradients (standard PyTorch)
- `add_splats(n_new, lr_new=None)`: Add optimizer state for new splats
- `remove_splats(keep_mask)`: Remove state for pruned splats
- `set_learning_rate(splat_idx, lr)`: Set individual splat learning rate
- `get_learning_rate(splat_idx)`: Get individual splat learning rate
- `get_effective_learning_rates()`: Get all learning rates as tensor

### 2. PerSplatReduceLROnPlateau Scheduler

Reduces learning rate for individual splats when their contribution plateaus.

```python
from luxar.gsplats.optim import PerSplatReduceLROnPlateau

scheduler = PerSplatReduceLROnPlateau(
    optimizer,
    mode="min",                   # "min" for loss, "max" for metrics
    factor=0.5,                   # LR reduction factor (new_lr = lr * factor)
    patience=10,                  # Steps without improvement before reduction
    threshold=1e-4,               # Minimum change to qualify as improvement
    cooldown=0,                   # Steps to wait after LR reduction
    min_lr=1e-8,                  # Minimum learning rate floor
    global_patience=20            # Global fallback patience
)

# Update with per-splat or global metrics
for epoch in range(num_epochs):
    # ... optimization ...
    
    # Option 1: Global loss (affects all splats equally)
    scheduler.step(loss.item())
    
    # Option 2: Per-splat losses (individual adaptation)
    per_splat_losses = compute_per_splat_losses(model, target)
    scheduler.step(per_splat_losses)  # Tensor (n_splats,) or dict
    
# Monitor LR reductions
reduction_counts = scheduler.get_lr_reduction_counts()  # Tensor (n_splats,)
```

**Supports Three Metric Formats**:
1. **Scalar** (`float` or `0D tensor`): Global metric applied to all splats
2. **1D Tensor** (`shape [n_splats]`): Per-splat metrics for individual adaptation
3. **Dict** (`{splat_idx: metric}`): Explicit per-splat metrics

### 3. PerSplatExponentialLR Scheduler

Exponential learning rate decay with optional age-based adaptation.

```python
from luxar.gsplats.optim import PerSplatExponentialLR

scheduler = PerSplatExponentialLR(
    optimizer,
    gamma=0.95,                   # Multiplicative decay factor (lr *= gamma)
    age_based_decay=True          # New splats decay slower
)

# Simple exponential decay
for epoch in range(num_epochs):
    # ... optimization ...
    scheduler.step()  # No metrics needed
```

**Age-Based Decay**: New splats get slower decay (still learning), old splats get faster decay (fine-tuning):
- New splat (age=0): `gamma_adjusted = 1.00` (no decay)
- Young splat (age=10): `gamma_adjusted = 0.975` (slow decay)
- Old splat (age=100): `gamma_adjusted ≈ 0.95` (normal decay)

### 4. ModelOptimizerCoordinator

Coordinates dynamic operations across model, optimizer, and scheduler to maintain state consistency.

```python
from luxar.gsplats.optim import ModelOptimizerCoordinator

coordinator = ModelOptimizerCoordinator(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler  # Optional
)

# Prune splats (atomic update across all components)
keep_mask = compute_importance_mask(model)
n_removed = coordinator.prune_splats(keep_mask)

# Add new splats
centers_new = torch.randn(10, 3)  # 10 new 3D splats
Ls_new = torch.eye(3).expand(10, 3, 3)
amps_new = torch.ones(10) * 0.5
n_added = coordinator.add_splats(centers_new, Ls_new, amps_new, lr_new=0.02)

# Replace all splats (complete reset)
n_new = coordinator.replace_all_splats(centers, Ls, amps, lr_reset=0.01)

# Monitor operations
status = coordinator.get_status()
# Returns: {'model_splats': 212, 'optimizer_states': 212, 
#           'operation_count': 5, 'learning_rates': {'mean': 0.015, ...}}
```

### 5. Factory Function

Convenient one-line setup for complete per-splat optimization configuration.

```python
from luxar.gsplats.optim import create_per_splat_optimizer_setup

# Plateau scheduler setup
optimizer, scheduler, coordinator = create_per_splat_optimizer_setup(
    model,
    lr=0.01,
    scheduler_type="plateau",
    patience=10,
    factor=0.5,
    min_lr=1e-8
)

# Exponential scheduler setup
optimizer, scheduler, coordinator = create_per_splat_optimizer_setup(
    model,
    lr=0.01,
    scheduler_type="exponential",
    gamma=0.95,
    age_based_decay=True
)

# No scheduler
optimizer, _, coordinator = create_per_splat_optimizer_setup(
    model,
    lr=0.01,
    scheduler_type=None
)
```

## Usage Examples

### Basic Optimization Loop

```python
from luxar.gsplats.optim import PerSplatAdam

# Create optimizer
optimizer = PerSplatAdam(model, lr=0.01)

# Training loop
for iteration in range(1000):
    optimizer.zero_grad()
    
    # Forward pass
    prediction = model()
    loss = mse_loss(prediction, target)
    
    # Backward pass
    loss.backward()
    
    # Update all splats
    optimizer.step()
    
    if iteration % 100 == 0:
        print(f"Iteration {iteration}, Loss: {loss.item():.6f}")
```

### With Learning Rate Scheduling

```python
from luxar.gsplats.optim import PerSplatAdam, PerSplatReduceLROnPlateau

# Create optimizer and scheduler
optimizer = PerSplatAdam(model, lr=0.01)
scheduler = PerSplatReduceLROnPlateau(
    optimizer,
    patience=10,
    factor=0.5,
    min_lr=1e-8
)

# Training loop with adaptive learning rates
for iteration in range(1000):
    optimizer.zero_grad()
    prediction = model()
    loss = mse_loss(prediction, target)
    loss.backward()
    optimizer.step()
    
    # Update learning rates based on loss
    scheduler.step(loss.item())
    
    if iteration % 50 == 0:
        lrs = optimizer.get_effective_learning_rates()
        print(f"Iter {iteration}, Loss: {loss.item():.6f}, "
              f"LR range: [{lrs.min():.2e}, {lrs.max():.2e}]")
```

### Dynamic Operations (Seeding and Pruning)

```python
from luxar.gsplats.optim import create_per_splat_optimizer_setup

# Setup with coordinator
optimizer, scheduler, coordinator = create_per_splat_optimizer_setup(
    model, lr=0.01, scheduler_type="plateau"
)

# Training with dynamic operations
for iteration in range(1000):
    optimizer.zero_grad()
    prediction = model()
    loss = mse_loss(prediction, target)
    loss.backward()
    optimizer.step()
    scheduler.step(loss.item())
    
    # Periodic dynamic operations
    if iteration % 50 == 0:
        # Prune weak splats
        importance = compute_importance(model)
        keep_mask = importance > 0.001
        n_removed = coordinator.prune_splats(keep_mask)
        
        # Seed new splats where needed
        if should_seed(prediction, target):
            centers_new, Ls_new, amps_new = generate_new_splats(...)
            n_added = coordinator.add_splats(
                centers_new, Ls_new, amps_new, 
                lr_new=0.02  # Higher LR for new splats
            )
            print(f"Added {n_added} splats, removed {n_removed} splats")
```

### State Serialization (Checkpointing)

```python
from luxar.gsplats.optim import PerSplatAdam, PerSplatReduceLROnPlateau

# Training with checkpointing
optimizer = PerSplatAdam(model, lr=0.01)
scheduler = PerSplatReduceLROnPlateau(optimizer, patience=10)

# Save checkpoint
checkpoint = {
    'iteration': iteration,
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'scheduler_state': scheduler.state_dict(),
    'loss': loss.item()
}
torch.save(checkpoint, 'checkpoint.pt')

# Load checkpoint
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state'])
optimizer.load_state_dict(checkpoint['optimizer_state'])
scheduler.load_state_dict(checkpoint['scheduler_state'])
start_iteration = checkpoint['iteration'] + 1
```

## Gradient Dilution Compensation

### The Problem

Higher-dimensional Gaussian splats have more parameters per splat:
- **2D**: 2 (position) + 2 (L_diag) + 1 (L_off) + 1 (amplitude) + 1 (sharpness) = 7 scalars
- **3D**: 3 (position) + 3 (L_diag) + 3 (L_off) + 1 (amplitude) + 1 (sharpness) = 11 scalars
- **4D**: 4 (position) + 4 (L_diag) + 6 (L_off) + 1 (amplitude) + 1 (sharpness) = 16 scalars

When loss gradients backpropagate, they get **diluted** across more parameters in higher dimensions, requiring higher learning rates to achieve equivalent optimization effectiveness.

### The Solution

`PerSplatAdam` automatically compensates using gradient dilution factors:

**For d ≤ 3** (Conservative scaling):
```
gradient_dilution_factor = params_current / params_2d
```

**For d > 3** (Enhanced scaling with dimensional complexity):
```
dimensional_complexity = d^0.8
parameter_complexity = params_current / params_2d
gradient_dilution_factor = dimensional_complexity × parameter_complexity
```

**Examples**:
- **2D** (5 params): factor = 5/5 = **1.0×** (baseline)
- **3D** (10 params): factor = 10/5 = **2.0×**
- **4D** (15 params): factor = 4^0.8 × 15/5 ≈ **7.1×**

**Application**:
```python
# User specifies base_lr = 0.01
optimizer = PerSplatAdam(model_3d, lr=0.01)

# Internally calculated:
# effective_lr = 0.01 × 2.0 = 0.02 (for 3D)
# Applied to position: 0.02 × 0.1 = 0.002
# Applied to variance: 0.02 × 1.0 = 0.02
# Applied to amplitude: 0.02 × 2.0 = 0.04
# Applied to sharpness: 0.01 × 0.5 = 0.005 (uses base_lr, no dilution)
```

**Sharpness Exception**: Sharpness parameters always use `base_lr` without gradient dilution compensation since sharpness is a single scalar regardless of dimension.

## Per-Splat Learning Rates

### Individual Learning Rate Control

Each splat can have its own learning rate, enabling sophisticated adaptation strategies:

```python
# Set learning rate for specific splat
optimizer.set_learning_rate(splat_idx=5, lr=0.02)

# Get learning rate for specific splat
lr = optimizer.get_learning_rate(splat_idx=5)

# Get all learning rates as tensor
all_lrs = optimizer.get_effective_learning_rates()  # Shape: (n_splats,)
```

### Scheduler Integration

Schedulers modify individual splat learning rates automatically:

```python
# Plateau scheduler reduces LR for individual splats
scheduler = PerSplatReduceLROnPlateau(optimizer, patience=10, factor=0.5)

# Per-splat metrics enable individual adaptation
per_splat_losses = torch.tensor([0.5, 0.3, 0.8, 0.2])  # 4 splats
scheduler.step(per_splat_losses)
# Splat 2 (high loss) gets LR reduced, others unchanged

# Global metric affects all splats equally
scheduler.step(global_loss.item())
```

### New Splat Learning Rates

New splats can start with higher learning rates for faster adaptation:

```python
# Add new splats with boosted learning rate
coordinator.add_splats(
    centers_new,
    Ls_new,
    amps_new,
    lr_new=0.02  # 2× base rate for faster initial learning
)
```

## Dynamic Operations

### Adding Splats

Add new Gaussian splats during optimization (seeding, splitting):

```python
# Generate new splat parameters
centers_new = torch.randn(10, 3)  # 10 new 3D splats
Ls_new = torch.eye(3).expand(10, 3, 3)  # Isotropic covariances
amps_new = torch.ones(10) * 0.5  # Moderate amplitudes

# Add with coordinator (updates model, optimizer, scheduler atomically)
n_added = coordinator.add_splats(
    centers_new,
    Ls_new,
    amps_new,
    lr_new=0.02  # Optional: higher LR for new splats
)

# Or directly with optimizer (manual coordination)
optimizer.add_splats(n_new=10, lr_new=0.02)
if scheduler is not None:
    scheduler.add_splats(n_new=10)
```

**Behavior**: New splats get fresh optimizer state (zero momentum) and can have individual learning rates.

### Removing Splats

Remove ineffective splats during optimization (pruning):

```python
# Compute importance metric
importance = compute_importance(model)  # Shape: (n_splats,)

# Create keep mask
keep_mask = importance > 0.001  # Boolean tensor

# Remove with coordinator (atomic update)
n_removed = coordinator.prune_splats(keep_mask)

# Or manually with optimizer
optimizer.remove_splats(keep_mask)
if scheduler is not None:
    scheduler.remove_splats(keep_mask)
```

**Behavior**: Kept splats preserve all momentum and state, indices are automatically reindexed to remain contiguous.

### Replacing All Splats

Complete model reset (e.g., reseeding, dimension changes):

```python
# Generate completely new splat configuration
centers = torch.randn(50, 3)
Ls = torch.eye(3).expand(50, 3, 3)
amps = torch.ones(50) * 0.5

# Replace with coordinator
n_new = coordinator.replace_all_splats(
    centers,
    Ls,
    amps,
    lr_reset=0.01  # Reset all learning rates
)
```

**Behavior**: All optimizer and scheduler state is cleared and re-initialized fresh.

### Momentum Preservation

The key advantage of per-splat optimization is momentum preservation:

```python
# Before operation: 100 splats with optimization history
# Splat 50 has accumulated momentum: exp_avg=[0.1, 0.2, 0.3]

# Remove 20 splats
keep_mask = torch.ones(100, dtype=torch.bool)
keep_mask[10:30] = False  # Remove splats 10-29
coordinator.prune_splats(keep_mask)

# After operation: 80 splats
# Former splat 50 is now splat 30 (reindexed)
# Its momentum is PRESERVED: exp_avg=[0.1, 0.2, 0.3]
# Optimization continues smoothly without disruption
```

Standard PyTorch optimizers would lose ALL momentum during such operations.

## API Reference

For complete API documentation with parameter descriptions, return types, and algorithm details, see:
- **[SPECIFICATIONS.md](./SPECIFICATIONS.md)** - Complete technical specifications
- **[per_splat_adam.py](./per_splat_adam.py)** - Optimizer implementation
- **[per_splat_scheduler.py](./per_splat_scheduler.py)** - Scheduler implementations
- **[integration.py](./integration.py)** - Coordinator and factory function

### Quick Reference

**PerSplatAdam**:
```python
__init__(model, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False)
step() -> bool
zero_grad()
add_splats(n_new_splats, lr_new=None)
remove_splats(keep_mask)
set_learning_rate(splat_idx, lr)
get_learning_rate(splat_idx) -> float
get_effective_learning_rates() -> torch.Tensor
state_dict() -> Dict
load_state_dict(state_dict)
```

**PerSplatReduceLROnPlateau**:
```python
__init__(optimizer, mode="min", factor=0.5, patience=10, threshold=1e-4, 
         cooldown=0, min_lr=1e-8, global_patience=20)
step(metrics)
add_splats(n_new_splats)
remove_splats(keep_mask)
get_lr_reduction_counts() -> torch.Tensor
state_dict() -> Dict
load_state_dict(state_dict)
```

**PerSplatExponentialLR**:
```python
__init__(optimizer, gamma=0.95, age_based_decay=True)
step()
add_splats(n_new_splats)
remove_splats(keep_mask)
```

**ModelOptimizerCoordinator**:
```python
__init__(model, optimizer, scheduler=None)
prune_splats(keep_mask) -> int
add_splats(centers_new, Ls_new, amps_new, lr_new=None) -> int
replace_all_splats(centers, Ls, amps, lr_reset=None) -> int
get_status() -> dict
```

**Factory Function**:
```python
create_per_splat_optimizer_setup(
    model, lr=1e-3, scheduler_type="plateau",
    betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False,
    patience=10, factor=0.5, threshold=1e-4, cooldown=0, min_lr=1e-8,
    gamma=0.95, age_based_decay=True, **extra_kwargs
) -> Tuple[PerSplatAdam, Optional[scheduler], ModelOptimizerCoordinator]
```

## Testing

The optim package has comprehensive test coverage:

```bash
# Run all optim tests
hatch run pytest packages/luxar/src/luxar/gsplats/optim/tests/ -v

# Run specific test files
hatch run pytest packages/luxar/src/luxar/gsplats/optim/tests/test_per_splat_adam.py -v
hatch run pytest packages/luxar/src/luxar/gsplats/optim/tests/test_per_splat_optimizer.py -v
```

**Test Coverage**:
- PerSplatAdam correctness (gradient updates, bias correction)
- Gradient dilution compensation accuracy
- Parameter-type-specific learning rates
- Dynamic operations (add/remove/replace splats)
- Scheduler behavior (plateau detection, exponential decay)
- Coordinator atomic operations
- State serialization/deserialization
- Edge cases and error handling
- Device compatibility (CPU, CUDA, MPS)

## Performance Characteristics

### Computational Complexity

**Per Optimization Step**:
- State initialization: `O(N_new × n_params)` (only when topology changes, lazy)
- Gradient extraction: `O(N_splats × n_params)`
- Momentum updates: `O(N_splats × n_params)`
- Parameter updates: `O(N_splats × n_params)`

**Total**: `O(N_splats × n_params)` per step

**Where**:
- `N_splats` = number of splats
- `n_params` = scalars per splat (2D: 7, 3D: 11, 4D: 16)

### Memory Usage

**Per Splat**:
- Standard Adam: 2 × n_params tensors (exp_avg, exp_avg_sq)
- AMSGrad: 3 × n_params tensors (+ max_exp_avg_sq)
- Metadata: lr (float), base_lr (float), step (int)

**Example** (3D with 10K splats):
- Parameters: 11 scalars per splat
- Momentum buffers: 22 scalars per splat (exp_avg + exp_avg_sq)
- Total: 10K × 22 × 4 bytes ≈ 880 KB (negligible)

### Optimization Tips

1. **Enable gradient dilution**: Already automatic, no action needed
2. **Use AMSGrad for stability**: Set `amsgrad=True` if convergence issues
3. **Tune scheduler patience**: Higher patience for smoother convergence
4. **Monitor LR reductions**: Use `get_lr_reduction_counts()` for debugging
5. **Batch dynamic operations**: Group add/remove when possible
6. **Checkpoint regularly**: Save optimizer state for resumable training

## References

**Related Documentation**:
- [Main gsplats SPECIFICATIONS.md](../SPECIFICATIONS.md) - Core Gaussian splatting concepts
- [utils/SPECIFICATIONS.md](../utils/SPECIFICATIONS.md) - Gradient dilution calculation details
- [fitting/SPECIFICATIONS.md](../fitting/SPECIFICATIONS.md) - Pipeline integration
- [GLOSSARY.md](../GLOSSARY.md) - Terminology and naming conventions

**Key Papers**:
- Kingma & Ba (2014): "Adam: A Method for Stochastic Optimization"
- Reddi et al. (2018): "On the Convergence of Adam and Beyond" (AMSGrad)

## Version History

- **v1.0.0** (January 2025): Initial implementation
  - Per-splat Adam optimizer with gradient dilution compensation
  - Parameter-type-specific learning rates (×0.1, ×1.0, ×2.0, ×0.5)
  - Per-splat plateau and exponential schedulers
  - Model-optimizer coordinator for atomic dynamic operations
  - Comprehensive testing and documentation
  - State serialization support
