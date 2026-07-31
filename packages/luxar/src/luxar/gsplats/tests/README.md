# GSplats Test Suite

This directory contains comprehensive tests for the GSplats package, following the "in context" testing pattern where tests are located close to the code they test.

## Test Structure

The tests are organized following the "in-subpackage" pattern, where each subpackage has its own tests/ directory for unit tests, and integration tests remain in the main gsplats/tests/:

```
gsplats/
├── tests/                          # Integration tests
│   ├── __init__.py
│   ├── test_batch.py               # Batch fitting orchestration
│   ├── test_cholesky_dim_ops.py    # Cholesky dimension operations
│   ├── test_culling.py             # Splat culling algorithms
│   ├── test_fit_gsplats.py         # Full fitting pipeline (requires torch/scipy)
│   ├── test_gpu_profile.py         # GPU profiling and benchmarks
│   ├── test_gsplat_data.py         # GSplatData class tests
│   ├── test_gsplats_integration.py # Integration tests
│   ├── test_metrics.py             # Quality metrics (PSNR, SSIM, MSE)
│   ├── test_progressive_fitting.py # Progressive fitting pipeline
│   ├── test_slurm_gen.py           # Generated Slurm control-flow behavior
│   ├── test_spatial_volume_filter.py # Spatial volume filtering
│   ├── test_tiled_fitting.py       # Tiled fitting for large volumes
│   └── README.md                   # This file
├── fitting/
│   └── tests/                      # Fitting pipeline unit tests
│       ├── __init__.py
│       ├── test_fitting_config.py      # Configuration validation
│       ├── test_fitting_preprocessing.py # Data preprocessing
│       ├── test_fitting_validation.py   # Input validation
│       ├── test_initialization.py       # Model initialization
│       ├── test_losses.py               # Loss functions
│       ├── test_optimization.py         # Optimization loop
│       ├── test_results.py              # Result finalization
│       └── test_visualization.py        # Visualization helpers
├── optim/
│   └── tests/                      # Optimizer tests
│       ├── __init__.py
│       └── test_integration.py          # Standard optimizer integration
├── utils/
│   └── tests/                      # Utils-specific tests
│       ├── __init__.py
│       └── test_trils.py           # Triangular matrix operations
├── models/
│   ├── utils/tests/                # Model utility tests
│   │   ├── __init__.py
│   │   ├── test_inverse_softplus.py # Inverse softplus function
│   │   └── test_lt_solver.py       # Lower triangular solver
│   └── gsplats/tests/              # GSplat model tests
│       ├── __init__.py
│       ├── test_gsplat_model.py    # GaussianSplatModel class
│       └── test_rendering.py       # Rendering functions
└── multiscale/
    └── tests/                      # Multiscale decomposition tests
        ├── __init__.py
        ├── test_decomposition_basic.py
        └── test_energy_distribution.py
```

## Dependencies and Test Execution

### Core Tests (Always Available)
These tests only require numpy and can be run in any environment:

```bash
# Test triangular matrix utilities
hatch run pytest packages/luxar/src/luxar/gsplats/utils/tests/test_trils.py
```

### Optional Dependency Tests

Some tests require additional dependencies:

- **PyTorch tests**: `test_lt_solver.py`, `test_inverse_softplus.py` (torch-dependent tests)
- **SciPy tests**: `test_candidates.py` (scipy-dependent tests)

These tests are designed to skip gracefully when dependencies are missing.

## Running Tests

### Run All Available Tests
```bash
# Run all tests that can execute with current dependencies
hatch run pytest packages/luxar/src/luxar/gsplats/ -v
```

### Run Core Tests Only
```bash
# Run only the tests that don't require external dependencies
hatch run pytest packages/luxar/src/luxar/gsplats/utils/tests/test_trils.py -v
```

### Run with Coverage
```bash
# Generate coverage report
hatch run pytest --cov=luxar.gsplats packages/luxar/src/luxar/gsplats/ --cov-report=html
```

## Test Categories

### 1. Utility Function Tests (`utils/tests/`)

**`test_trils.py`** - Comprehensive tests for triangular matrix operations:
- Matrix packing/unpacking operations
- Shape validation and consistency
- Data type preservation
- Round-trip correctness
- Edge cases and error handling

**Coverage**: Covers all functions in `trils.py`

### 2. Fitting Pipeline Tests (`fitting/tests/`)

**Unit Tests for Modular Fitting Pipeline:**
- `test_fitting_config.py` - Configuration dataclass validation
- `test_fitting_preprocessing.py` - Data preprocessing and normalization
- `test_fitting_validation.py` - Input validation at API boundaries
- `test_initialization.py` - Model and optimizer initialization
- `test_losses.py` - Loss functions and regularization
- `test_optimization.py` - Optimization loop and convergence
- `test_results.py` - Result finalization and statistics
- `test_visualization.py` - Visualization helpers (napari mocked)

### 3. Integration Tests (`tests/`)

**`test_fit_gsplats.py`** (requires torch + scipy):
- Gaussian splat fitting optimization pipeline
- Input validation and parameter constraints
- Loss function testing (MSE and Poisson)
- Convergence and optimization properties
- Device support (CPU/CUDA)
- Edge cases (uniform images, empty candidates)

**`test_gsplats_integration.py`** (requires torch):
- End-to-end integration tests for the gsplats pipeline

**`test_gsplat_data.py`**:
- GSplatData class tests (save/load, concatenation, merging)

**`test_batch.py`**:
- Batch manifest creation and serialization
- Task ID encoding/decoding
- Slurm script generation

**`test_slurm_gen.py`**:
- Generated sequential task-packing exit behavior
- Failure retention across later successes and padded-tail breaks
- Literal Slurm log/output paths and preset arguments with spaces, quotes, and shell metacharacters
- Rejection of output-directory line terminators that could split an sbatch directive

**`test_cholesky_dim_ops.py`** (requires torch):
- Cholesky factor dimension operations and transformations

**`test_culling.py`** (requires torch):
- Splat culling algorithms (cumulative, redundancy, error-budget)

**`test_gpu_profile.py`**:
- GPU profiling and benchmark data handling

**`test_metrics.py`**:
- Quality metrics computation (PSNR, SSIM, MSE)
- Comparison between original and reconstructed volumes

**`test_progressive_fitting.py`** (requires torch):
- Progressive fitting pipeline with iterative residual refinement

**`test_spatial_volume_filter.py`** (requires torch):
- Spatial volume filtering for splat datasets

**`test_tiled_fitting.py`** (requires torch):
- Tiled fitting for large volumes with overlap and stitching

### 4. Advanced Function Tests (Optional Dependencies)

**`test_inverse_softplus.py`** (numpy + optional torch):
- Numerical stability of inverse softplus
- Cross-validation with PyTorch when available
- Edge case handling (small/large values)
- Data type preservation

**`test_lt_solver.py`** (requires torch):
- Cross-version PyTorch compatibility
- Lower triangular system solving
- Batch processing capabilities
- Gradient flow verification
- Device compatibility (CPU/CUDA)

**`test_gsplat_model.py`** (requires torch):
- GaussianSplatModel class initialization and validation
- Parameter transformation and constraint enforcement
- Forward pass rendering in 2D/3D
- Gradient computation and optimization
- Batched vs sequential rendering consistency
- Edge cases and numerical stability

**`test_rendering.py`** (requires torch):
- PyTorch and NumPy rendering functions
- Batched rendering optimization
- Multi-dimensional Gaussian rendering (2D/3D)
- Amplitude-aware culling features
- Performance with large numbers of splats
- Numerical stability with extreme parameter values

## Test Quality Metrics

### Current Test Coverage
- **Core utilities**: 100% function coverage (trils, inverse_softplus)
- **Algorithmic patterns**: 90%+ pattern coverage (validation, arrays, distances)
- **Input validation**: Comprehensive validation pattern tests
- **Error handling**: Edge cases and error conditions covered
- **GaussianSplatModel**: Complete class testing (35+ test methods)
- **Optimization pipeline**: Full fit_gaussian_splats testing (25+ test methods)
- **Rendering functions**: Comprehensive rendering tests (25+ test methods)

### Test Execution

Run the full test suite:
```bash
hatch run pytest packages/luxar/src/luxar/gsplats/ -v
```

The test count and execution time varies depending on available dependencies (torch, scipy, CUDA).

## Adding New Tests

When adding new functionality to gsplats:

1. **Create tests in context**: Place tests in the appropriate `tests/` folder near the code
2. **Follow naming conventions**: Use `test_<module_name>.py` for test files
3. **Handle dependencies gracefully**: Use `pytest.mark.skipif` for optional dependencies
4. **Include comprehensive coverage**: Test normal cases, edge cases, and error conditions
5. **Update this README**: Document new test files and their purpose

## Key Improvements Made

### Code Quality Fixes Applied:
1. **Input validation**: Added comprehensive validation to seed generation functions
2. **Division by zero protection**: Fixed uniform image handling in `fit_gsplats`
3. **Performance optimizations**: Reduced redundant computations in candidate detection
4. **Code consistency**: Removed unused parameters and standardized error handling

### Test Design Principles:
1. **Comprehensive coverage**: Each function tested with multiple scenarios
2. **Edge case handling**: Empty arrays, boundary conditions, invalid inputs
3. **Numerical stability**: Tests for floating-point edge cases
4. **Cross-validation**: Comparison with reference implementations when possible
5. **Graceful degradation**: Tests skip when dependencies unavailable

## Dependencies

### Required
- `numpy>=1.24`
- `pytest>=7.4.0`
- `torch` (for model, fitting, and optimization tests)
- `scipy` (for seed generation and spatial operations)

### Optional
- CUDA GPU (for GPU-specific tests, skipped gracefully when unavailable)
