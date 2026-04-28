# Metal-Accelerated Gaussian Splatting

High-performance Metal compute backend for Gaussian splatting on Apple Silicon (M-series chips).

## Requirements

- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **Xcode** (full installation, not just Command Line Tools)
- **PyTorch** with MPS support (`torch.backends.mps.is_available()` returns `True`)

## Installation

### 1. Install Xcode (if not already installed)

The Metal compiler requires the full Xcode.app, not just Command Line Tools:

```bash
# Install from Mac App Store or download from developer.apple.com
# After installation, accept the license:
sudo xcodebuild -license accept

# Point xcode-select to Xcode.app (required for Metal compiler):
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer

# Verify metal compiler is available:
xcrun --find metal
```

### 2. Build the Metal Extension (Automatic)

**The Metal extension is automatically compiled on first import.** No manual build step is required.

If you need to manually rebuild (e.g., after modifying the Metal shaders):

```bash
cd packages/luxar/src/luxar/gsplats/models/gsplats/metal

# Build in-place (for development):
python setup.py build_ext --inplace
```

The build process will:
1. Compile `kernels.metal` → `kernels.air` (intermediate)
2. Link `kernels.air` → `default.metallib` (Metal library)
3. Compile `bindings.mm` (C++/Objective-C++) → Python extension

### 3. Verify Installation

```python
from luxar.gsplats.models.gsplats.metal import is_metal_available

if is_metal_available():
    print("✓ Metal backend successfully installed!")
else:
    print("✗ Metal backend not available")
    # Fall back to CPU/MPS PyTorch implementation
```

## Architecture

### Files

```
metal/
├── __init__.py                 # Package interface
├── setup.py                    # Build script (native code compilation)
├── gsplat_model_metal.py       # Python interface (GaussianSplatModelMetal)
├── README.md                   # This file
├── SPECIFICATIONS.md           # Technical specification
├── src/
│   ├── kernels.metal           # Metal compute shaders
│   └── bindings.mm             # C++ dispatcher (PyTorch ↔ Metal)
└── tests/
    ├── __init__.py                    # Package marker
    ├── test_metal_backend.py          # Core functionality tests
    ├── test_metal_conic.py            # Conic computation tests
    ├── test_metal_numerical.py        # Gradient correctness tests
    ├── test_metal_performance.py      # Performance benchmarks
    ├── test_coordinate_transforms.py  # Coordinate transform tests
    └── test_optimizer_compatibility.py # Optimizer integration tests

# Build artifacts (generated, gitignored):
# ├── build/                    # Temporary build files
# ├── *.so                      # Compiled Python extension
# └── src/default.metallib      # Compiled Metal library
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                      Python Layer                           │
│  GaussianSplatModelMetal → MetalSplatFunction               │
│  (torch.autograd.Function)                                  │
└─────────────────────────────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  PyTorch: L → Conic                         │
│  cholesky_inverse(L) → Σ⁻¹  [O(N), CPU/MPS]                │
└─────────────────────────────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              C++ Dispatcher (bindings.mm)                   │
│  dispatch_forward_3d() → Metal kernels                      │
└─────────────────────────────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                Metal Compute Kernels                        │
│  1. preprocess_3d: Count splats per tile                   │
│  2. bin_3d: Populate tile lists                            │
│  3. rasterize_fwd_3d: Pixel-parallel rendering [O(P)]      │
│  4. rasterize_bwd_3d: Gradients with SIMD reduction        │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Hybrid Architecture**:
   - PyTorch: Matrix ops (L→Conic), gradient chain rule
   - Metal: Pixel-parallel rendering (forward/backward)
   - Rationale: Keep gradient graph in PyTorch for consistency

2. **Pixel-Centric Rendering**:
   - Each GPU thread processes one pixel
   - Gathers contributions from nearby splats
   - Avoids atomic scatter bottleneck of splat-centric approach

3. **Tiled Binning (3D only)**:
   - Spatial acceleration: O(P × splats/tile) instead of O(P × N)
   - Tile size = 4³ = 64 threads/tile

4. **SIMD Reduction**:
   - Reduces atomic contention by 32x in backward pass
   - Lane 0 writes aggregated gradients
   - Critical for performance (without it: slower than CPU!)

## Usage

### Basic Example

```python
import torch
from luxar.gsplats.models.gsplats.metal import GaussianSplatModelMetal, is_metal_available

if is_metal_available():
    model = GaussianSplatModelMetal(
        shape=(64, 64, 64),
        centers0=centers,
        L0=L,
        amps0=amps,
        sigma_min_diag=(0.5, 0.5, 0.5),
        truncate=3.0,
        intensity_floor=1e-5,  # Early culling threshold
        device='mps'
    )

    output = model()  # Forward pass uses Metal
    loss = compute_loss(output, target)
    loss.backward()  # Backward pass uses Metal
```

### Integration with Fitting Pipeline

The model automatically uses Metal when available and appropriate:

```python
from luxar.gsplats import fit_gaussian_splats

# Metal will be used automatically if:
# 1. Metal extension is installed
# 2. Device is MPS
# 3. Dimensionality ≤ 3
result = fit_gaussian_splats(
    volume,
    device='mps',  # Use MPS device for Metal acceleration
)
```

## Performance Expectations

> Speedups below are typical observed ranges on test volumes; actual results vary by problem size, dimensionality, and driver/runtime. Treat them as indicative, not guarantees.

Target speedups (compared to CPU PyTorch):
- **M4 Max**: 10-50x faster
- **M3**: 8-30x faster
- **M2**: 5-20x faster
- **M1**: 3-15x faster

Actual speedup depends on:
- Volume size (larger = better GPU utilization)
- Number of splats (N)
- Splat density (overlaps per pixel)

## Troubleshooting

### "Metal backend not available"

The extension auto-compiles on first import. If it fails, check:

```bash
# 1. Xcode installed and active?
xcrun --find metal
# If this fails, run: sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer

# 2. PyTorch MPS available?
python -c "import torch; print(torch.backends.mps.is_available())"

# 3. Extension importable (after auto-build)?
python -c "from luxar.gsplats.models.gsplats.metal import is_metal_available; print(is_metal_available())"
```

### Build Errors

**"xcrun: error: unable to find utility 'metal'"**
→ Switch xcode-select to Xcode.app (see Installation step 1)

**"Failed to load Metal library"**
→ Run `python setup.py build_ext --inplace` again
→ Check that `src/default.metallib` exists

**Compilation errors in bindings.mm**
→ Ensure PyTorch headers are findable
→ Try: `pip install torch --upgrade`

### Runtime Errors

**"Tensor must be on MPS device"**
→ Move model to MPS: `model.to('mps')`

**"Buffer overflow"**
→ Report as bug (storage_offset handling issue)

**NaN gradients**
→ Check `intensity_floor` parameter (too high = zero gradients)
→ Verify PyTorch gradcheck passes

## Limitations

- **3D only**: The Metal backend only supports 3D volumes. For 2D or nD, use `GaussianSplatModel`.
- **MPS device required**: CPU tensors not supported
- **No mixed precision**: float32 only

## Development

### Running Tests

```bash
# Unit tests
cd packages/luxar
hatch run pytest src/luxar/gsplats/models/gsplats/metal/tests/

# Gradient/numerical tests
hatch run pytest src/luxar/gsplats/models/gsplats/metal/tests/test_metal_numerical.py -v

# Benchmark
python packages/luxar/examples/benchmark_m4_max.py
```

### Profiling

```python
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.MPS]
) as prof:
    output = model()
    loss.backward()

print(prof.key_averages().table(sort_by="self_cpu_time_total"))
```

## References

- [METAL_SPLATTING_IMPLEMENTATION_SPEC.md](../../../../../../../docs/guides/developer/METAL_SPLATTING_IMPLEMENTATION_SPEC.md) - Complete implementation specification
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) - Original paper

## License

Same as Luxar project.
